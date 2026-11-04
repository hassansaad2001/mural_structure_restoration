# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
"""End to end on CPU: checkpoint loading, generate_edges.py, train_tsr.py --dry-run,
the inference server (TCP round trip) and the evaluation server in --headless mode."""
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
from PIL import Image

from conftest import ROOT, RES

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONWARNINGS='ignore')


def _run(args, timeout=600):
    r = subprocess.run([sys.executable] + args, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, f'{" ".join(args)} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}'
    return r.stdout


@pytest.mark.parametrize('name', ['ours', 'edgeconnect', 'zits_tsr'])
def test_checkpoint_loads_with_all_weights(checkpoints, name):
    import torch
    from evaluation.eval_core import resolve_network
    from inference.inference_server import load_generator
    cls, extra, src = resolve_network(str(checkpoints[name]), None, 'networks.new_mat.GeneratorSketch')
    assert src == 'training_options.json'
    G, _ = load_generator(str(checkpoints[name]), torch.device('cpu'), RES, cls, extra)
    assert G.load_report == ([], []), f'missing/unexpected weights: {G.load_report}'


@pytest.mark.parametrize('name', ['ours', 'zits_tsr'])
def test_generate_edges(data_root, checkpoints, tmp_path, name):
    out = tmp_path / 'out'
    _run(['generate_edges.py', '--network', str(checkpoints[name]), '--cpu', '--save-raw',
          '--dpath', str(data_root / 'val' / 'images'), '--epath', str(data_root / 'val' / 'edges'),
          '--mpath', str(data_root / 'masks'), '--outdir', str(out)])
    for sub in ('results', 'highlights', 'raw'):
        assert len(os.listdir(out / sub)) == 3


@pytest.mark.parametrize('preset', ['ours', 'edgeconnect', 'zits_tsr'])
def test_train_dry_run(data_root, tmp_path, preset):
    stdout = _run(['train_tsr.py', '--dry-run', '--preset', preset, '--gpus', '1', '--outdir', str(tmp_path),
                   '--data-edge', str(data_root / 'train' / 'edges'), '--data-image', str(data_root / 'train' / 'images'),
                   '--data-val-edge', str(data_root / 'val' / 'edges'), '--data-val-image', str(data_root / 'val' / 'images'),
                   '--mask-path', str(data_root / 'masks'), '--batch-gpu', '1'])
    opts = json.loads(stdout[stdout.index('{'): stdout.rindex('}') + 1])
    expected = {'ours': 'networks.new_mat.GeneratorSketch',
                'edgeconnect': 'networks.baselines.edgeconnect.EdgeConnectEdgeGenerator',
                'zits_tsr': 'networks.baselines.zits_tsr.ZITSTSREdge'}[preset]
    assert opts['G_kwargs']['class_name'] == expected


def _free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_inference_server_round_trip(data_root, checkpoints):
    from inference.inference_server import send_msg, recv_msg
    port = _free_port()
    proc = subprocess.Popen([sys.executable, 'inference/inference_server.py', '--model', str(checkpoints['ours']),
                             '--port', str(port), '--host', '127.0.0.1', '--cpu'],
                            cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(120):
            try:
                sock = socket.create_connection(('127.0.0.1', port), timeout=1)
                break
            except OSError:
                assert proc.poll() is None, proc.stdout.read()
                time.sleep(1)
        else:
            pytest.fail('inference server did not start')
        sock.settimeout(300)        # the 1 s timeout was only for connecting; CPU inference takes longer
        img = np.array(Image.open(next((data_root / 'val' / 'images').iterdir())).convert('L'))
        edge = np.array(Image.open(next((data_root / 'val' / 'edges').iterdir())).convert('L'))
        mask = np.array(Image.open(next((data_root / 'masks').iterdir())).convert('L'))
        send_msg(sock, {'img': img, 'edge': edge, 'mask': mask, 'invert_edge': True})
        r = recv_msg(sock)
        sock.close()
        assert 'error' not in r, r.get('error')
        assert r['composite'].shape == (RES, RES) and r['highlight'].shape == (RES, RES, 3)
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def test_eval_server_headless(data_root, checkpoints, tmp_path):
    from evaluation.eval_core import EvalSettings
    st = EvalSettings(image_dir=str(data_root / 'val' / 'images'), edge_dir=str(data_root / 'val' / 'edges'),
                      mask_dir=str(data_root / 'masks'), results_dir=str(tmp_path / 'results'),
                      checkpoints=[str(checkpoints['ours']), str(checkpoints['edgeconnect'])],
                      mask_pairing='pool', masks_per_image=2, batch_size=2, force_cpu=True,
                      compute_fid=False, compute_lpips=False, min_edge_pixels=5, min_edge_density=0.0,
                      small_lo=0.0, small_hi=25.0, medium_lo=25.0, medium_hi=40.0, large_lo=40.0, large_hi=100.0)
    st.save_json(tmp_path / 'settings.json')
    _run(['evaluation/eval_server.py', '--headless', '--settings', str(tmp_path / 'settings.json')])
    files = [f for _, _, fs in os.walk(tmp_path / 'results') for f in fs]
    assert any(f.endswith('.xlsx') for f in files), files
    assert any(f.endswith('.csv') for f in files), files
