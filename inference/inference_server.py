"""
inference_server.py  –  Runs inside the container.

Loads GeneratorSketch once, then listens on a TCP port for inference
requests from inference_gui.py running on the host.

Protocol (all messages prefixed with a 4-byte big-endian length):
  Client → Server : pickle of dict  { 'img': np.ndarray, 'edge': np.ndarray,
                                       'mask': np.ndarray, 'invert_edge': bool }
  Server → Client : pickle of dict  { 'composite': np.ndarray,   # (H,W) uint8
                                       'highlight': np.ndarray,   # (H,W,3) uint8
                                       'pred_raw':  np.ndarray }  # (H,W) float32
             OR    : pickle of dict  { 'error': str }

Run inside the container (from the repository root; scripts/inference_gui.sh does this for you):
    python3 inference/inference_server.py --model <outputs>/tsr/.../checkpoint-000xxx.pth

The port (default 9999) must be forwarded when starting the container:
    docker run ... -p 9999:9999 ...      (docker/run.sh does this)
"""
# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).

import os
import sys
import socket
import struct
import pickle
import argparse
import logging
import threading

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('InferenceServer')

# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------
HEADER = struct.Struct('>I')   # 4-byte big-endian uint32 = payload length

def recv_all(sock, n):
    """Receive exactly n bytes, blocking until done."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('Connection closed by client')
        buf.extend(chunk)
    return bytes(buf)

def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(HEADER.pack(len(data)) + data)

def recv_msg(sock):
    raw_len = recv_all(sock, HEADER.size)
    (length,) = HEADER.unpack(raw_len)
    data = recv_all(sock, length)
    return pickle.loads(data)

# ---------------------------------------------------------------------------
# Model loading  (shared with generate_edges.py and evaluation/eval_core.py)
# ---------------------------------------------------------------------------
def load_generator(checkpoint_path, device, resolution=512, network='networks.new_mat.GeneratorSketch',
                   extra_kwargs=None):
    """extra_kwargs: additional constructor args (e.g. dec_norm from training_options.json).
    Every kwarg is dropped if the network class does not accept it."""
    import inspect
    import torch

    here = os.path.dirname(os.path.abspath(__file__))
    for p in [here, os.path.dirname(here)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import dnnlib

    log.info(f'Loading checkpoint: {checkpoint_path}')
    try:   # checkpoints also hold numpy RNG states -> torch>=2.6 needs weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:   # torch < 1.13 has no weights_only argument
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'G_ema_state' in checkpoint:
        state_dict = checkpoint['G_ema_state']
        log.info('  Using G_ema_state')
    elif 'G_state' in checkpoint:
        state_dict = checkpoint['G_state']
        log.info('  Using G_state')
    elif isinstance(checkpoint, dict) and any(
            k.startswith(('enc0', 'enc1', 'conv_first')) for k in checkpoint):
        state_dict = checkpoint
        log.info('  Using raw state dict')
    else:
        raise ValueError(
            f'Cannot find model state dict.\nKeys: {list(checkpoint.keys())[:10]}')

    # Detect in_channels and dim
    for key in ('conv_first.conv.weight', 'enc0.conv.weight'):
        if key in state_dict:
            in_channels = state_dict[key].shape[1]
            dim = state_dict[key].shape[0]
            log.info(f'  in_channels={in_channels}, dim={dim} (from {key})')
            break
    else:
        in_channels = 3
        dim = 32
        log.warning('  Could not detect in_channels or dim, defaulting to in_channels=3, dim=32')

    # Detect down_time
    enc_keys = [k for k in state_dict if k.startswith('enc_conv.')]
    indices = set()
    for k in enc_keys:
        parts = k.split('.')
        try:
            indices.add(int(parts[1]))
        except (IndexError, ValueError):
            pass
    down_time = len(indices) if indices else 1
    log.info(f'  down_time={down_time}')

    G_kwargs = dnnlib.EasyDict(
        img_resolution=resolution,
        in_channels=in_channels,
        dim=dim,
        down_time_override=down_time,
        # new_mat.GeneratorSketch: older checkpoints were trained without the mask embedding
        mask_to_decoder=any(k.startswith('mask_embed.') for k in state_dict),
    )
    # Decoder norm: affine norm weights in the checkpoint mean it was trained with a norm.
    # 'in' was the old default (training_options.json only records dec_norm when it was set).
    has_dec_norm = any('.norm_dw.' in k for k in state_dict)
    G_kwargs.dec_norm = 'in' if has_dec_norm else 'none'
    G_kwargs.update(extra_kwargs or {})
    if has_dec_norm and G_kwargs.get('dec_norm') == 'none':
        G_kwargs.dec_norm = 'in'     # weights exist -> a norm was used, whatever the options say

    # Keep only kwargs the class accepts (network variants have different signatures)
    cls = dnnlib.util.get_obj_by_name(network)
    params = inspect.signature(cls.__init__).parameters
    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        dropped = [k for k in G_kwargs if k not in params]
        for k in dropped:
            G_kwargs.pop(k)
        if dropped:
            log.info(f'  {network} does not take {dropped}; ignored')
    log.info(f'  Network: {network}  kwargs: {dict(G_kwargs)}')
    G = cls(**G_kwargs).to(device).eval().requires_grad_(False)

    missing, unexpected = G.load_state_dict(state_dict, strict=False)
    n_params = sum(p.numel() for p in G.parameters())
    log.info(f'  Loaded  {n_params:,} params  '
             f'(missing={len(missing)}  unexpected={len(unexpected)})')
    if missing or unexpected:
        log.warning(f'  WEIGHTS DO NOT MATCH {network}: missing {list(missing)[:4]}  '
                    f'unexpected {list(unexpected)[:4]} -> wrong network class?')
    G.load_report = (list(missing), list(unexpected))
    return G, in_channels

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
INVERT_EDGE_DEFAULT = True   # can be overridden per-request

def run_inference(G, device, img_np, edge_np, mask_np, invert_edge):
    """
    img_np   : (H, W) or (H, W, 3)  uint8
    edge_np  : (H, W)                uint8
    mask_np  : (H, W)                uint8  white=hole
    Returns dict with 'composite', 'highlight', 'pred_raw'.
    """
    import torch

    H, W = mask_np.shape[:2]

    # ── image ──
    if img_np.ndim == 3:
        gray = (img_np[:,:,0].astype(np.float32) * 0.299 +
                img_np[:,:,1].astype(np.float32) * 0.587 +
                img_np[:,:,2].astype(np.float32) * 0.114)
    else:
        gray = img_np.astype(np.float32)
    img_t = torch.from_numpy((gray / 127.5) - 1.0).float().to(device)
    img_t = img_t.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)

    # ── edge ──
    e = edge_np.astype(np.float32)
    if invert_edge:
        e = 255.0 - e
    edge_t = torch.from_numpy(e / 255.0).float().to(device)
    edge_t = edge_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # ── mask ──
    m = mask_np.astype(np.float32) / 255.0
    mask_t = torch.from_numpy(m).float().to(device)
    mask_t = (mask_t > 0.5).float()
    mask_t = mask_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    with torch.no_grad():
        logits      = G(img_t, edge_t, mask_t)
        pred        = torch.sigmoid(logits)
        pred_binary = (pred > 0.5).float()
        composite   = edge_t * (1 - mask_t) + pred_binary * mask_t

    pred_np  = pred[0, 0].cpu().numpy()       # (H,W) float32
    comp_np  = composite[0, 0].cpu().numpy()  # (H,W) float32 [0,1]
    mask_out = mask_t[0, 0].cpu().numpy()     # (H,W) {0,1}

    # Black-on-White composite
    comp_bow = ((1.0 - comp_np) * 255).clip(0, 255).astype(np.uint8)

    # Highlight
    edge_rgb = np.stack([comp_bow, comp_bow, comp_bow], axis=-1).astype(np.float32)
    overlay  = np.array([100, 150, 255], dtype=np.float32)
    alpha    = 0.35
    highlight = edge_rgb.copy()
    for c in range(3):
        highlight[:, :, c] = (
            edge_rgb[:, :, c] * (1 - mask_out * alpha)
            + mask_out * alpha * overlay[c]
        )
    highlight = highlight.clip(0, 255).astype(np.uint8)

    return {
        'composite': comp_bow,
        'highlight': highlight,
        'pred_raw':  pred_np,
    }

# ---------------------------------------------------------------------------
# Client handler
# ---------------------------------------------------------------------------
def handle_client(conn, addr, G, device):
    log.info(f'Client connected: {addr}')
    try:
        while True:
            try:
                req = recv_msg(conn)
            except ConnectionError:
                break

            try:
                img_np      = req['img']
                edge_np     = req['edge']
                mask_np     = req['mask']
                invert_edge = req.get('invert_edge', INVERT_EDGE_DEFAULT)

                result = run_inference(G, device, img_np, edge_np, mask_np, invert_edge)
                send_msg(conn, result)
            except Exception as exc:
                import traceback
                err = traceback.format_exc()
                log.error(f'Inference error:\n{err}')
                send_msg(conn, {'error': str(exc)})
    finally:
        conn.close()
        log.info(f'Client disconnected: {addr}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Inference TCP server (runs in the container)')
    parser.add_argument('--model',  required=True,
                        help='Path to .pth checkpoint inside the container')
    parser.add_argument('--host',   default='0.0.0.0',
                        help='Bind host (default: 0.0.0.0)')
    parser.add_argument('--port',   type=int, default=9999,
                        help='TCP port to listen on (default: 9999)')
    parser.add_argument('--cpu',    action='store_true',
                        help='Force CPU even if CUDA is available')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Resolution of the model (default: 512)')
    parser.add_argument('--network', type=str, default=None,
                        help="Network class name (default: from the run's training_options.json, "
                             "else networks.new_mat.GeneratorSketch)")
    args = parser.parse_args()

    import torch
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    log.info(f'Device: {device}')

    # Architecture options (swin_after, bottleneck, dec_norm, ...) come from training_options.json
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from evaluation.eval_core import resolve_network
    cls, extra_kwargs, source = resolve_network(args.model, args.network, 'networks.new_mat.GeneratorSketch')
    log.info(f'Network from {source}: {cls} {extra_kwargs}')

    G, in_channels = load_generator(args.model, device, args.resolution, cls, extra_kwargs)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(4)
    log.info(f'Listening on {args.host}:{args.port}  – waiting for GUI connections…')

    try:
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=handle_client,
                                  args=(conn, addr, G, device),
                                  daemon=True)
            t.start()
    except KeyboardInterrupt:
        log.info('Server stopped.')
    finally:
        srv.close()

if __name__ == '__main__':
    main()
