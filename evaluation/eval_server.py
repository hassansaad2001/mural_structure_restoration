"""
eval_server.py  –  Runs inside the container (torch + GPU).

Does all model work for eval_checkpoints_gui.py, which runs on the host:
sample filtering, grid previews, full checkpoint evaluation and the export
(Excel / CSV / grids are written straight into the mounted results folder).

Protocol: see eval_core.py ("Client <-> server protocol"). One job at a time
(the GPU is shared); a second client gets a 'server busy' error.

Run inside the container (from the repository root; scripts/eval_gui.sh does this for you):
    python3 evaluation/eval_server.py      # listens on 0.0.0.0:9998

The port must be published when starting the container:
    docker run ... -p 9998:9998 ...      (docker/run.sh does this)

Headless (no GUI at all), with a settings JSON written by the GUI
(File > Save Settings…; paths are translated with its path_map):
    python3 evaluation/eval_server.py --headless --settings <path>/eval_settings.json
"""
# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).

import io
import os
import sys
import time
import socket
import select
import argparse
import logging
from collections import OrderedDict
import threading
import traceback

from PIL import Image

import eval_core as ec

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('EvalServer')

_job_lock = threading.Lock()


class SampleSession:
    """Keeps an evaluator (LPIPS loaded) and up to MODEL_CACHE_SIZE generators on the GPU,
    so clicking through samples in the viewer does not reload every checkpoint."""

    def __init__(self):
        self.ev = None
        self.key = None
        self.models = OrderedDict()

    def get(self, st):
        key = (st.force_cpu, st.compute_lpips)
        if self.ev is None or key != self.key:
            self.clear()
            self.ev = ec.CheckpointEvaluator(st, None, None, model_cache=self.models)
            self.ev._setup(need_fid=False)
            self.key = key
        self.ev.s = st
        return self.ev

    def clear(self):
        self.models.clear()
        if self.ev is not None:
            self.ev._free()
        self.ev = self.key = None


_session = SampleSession()


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
def _environment():
    info = {'python': sys.version.split()[0]}
    try:
        import torch
        info['torch'] = torch.__version__
        info['cuda'] = torch.cuda.device_count() > 0 and torch.cuda.is_available()
        if info['cuda']:
            info['gpu'] = torch.cuda.get_device_name(0)
    except ImportError:
        info['torch'] = 'MISSING'
    for mod in ('lpips', 'openpyxl', 'requests', 'scipy', 'timm'):
        try:
            __import__(mod)
            info[mod] = 'ok'
        except ImportError:
            info[mod] = 'MISSING'
    return info


def _filter(st, logf, progress, cancel, use_cache=True):
    fr = ec.SampleFilter(st, log=logf).run(progress=progress, cancel=cancel, use_cache=use_cache)
    os.makedirs(st.results_dir, exist_ok=True)
    ec.write_rejected_csv(fr, os.path.join(st.results_dir, 'rejected_samples.csv'))
    return fr


def _png(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def run_job(req, logf, progress, cancel):
    cmd = req['cmd']
    if cmd == 'ping':
        return {'env': _environment()}

    st = ec.EvalSettings.from_dict(req['settings'])
    errs = st.validate(need_checkpoints=(cmd != 'filter'))
    if errs:
        raise ValueError('Invalid settings on the server side '
                         '(check the path map / container mount):\n  ' + '\n  '.join(errs))

    if cmd == 'sample':
        # 'sample' is an accepted-sample dict from a previous filter result (container paths).
        ev = _session.get(st)
        ev.log, ev.progress, ev.cancel = logf, progress, cancel
        res = ev.sample(req['sample'])
        for k in ('input', 'masked', 'gt'):
            res[k] = _png(Image.fromarray(res[k]))
        for p in res['preds']:
            for k in ('comp', 'raw'):
                if k in p:
                    p[k] = _png(Image.fromarray(p[k]))
        return res

    _session.clear()               # free GPU memory before full jobs
    fr = _filter(st, logf, progress, cancel, use_cache=req.get('use_cache', True))
    out = {'filter': fr.to_json(), 'from_cache': fr.from_cache}
    if cmd == 'filter':
        return out

    gi = ec.MaskGrouper(st).assign(fr.accepted)
    for g in ec.GROUPS + (ec.OUT_OF_RANGE, ec.ALL):
        logf(f'[Groups] {g}: {gi.count(g)}')
    for w in gi.warnings:
        logf('[Groups] WARNING: ' + w)
    ev = ec.CheckpointEvaluator(st, fr, gi, log=logf, progress=progress, cancel=cancel)

    if cmd == 'preview':
        out['grids'] = {g: _png(img) for g, img in ev.preview().items()}
        return out

    if cmd == 'evaluate':
        res = ev.run()
        out['run_dir'] = ec.export_results(st, fr, gi, res, log=logf)
        out.update(ckpt_names=res.ckpt_names, ckpt_paths=res.ckpt_paths, summary=res.summary,
                   notes=res.notes, device=res.device, ckpt_networks=res.ckpt_networks)
        return out

    raise ValueError(f'Unknown command: {cmd!r}')


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------
def handle_client(conn, addr):
    send_lock = threading.Lock()
    cancel = threading.Event()

    def send(obj):
        try:
            with send_lock:
                ec.send_msg(conn, obj)
        except OSError:
            cancel.set()             # client went away -> stop the job

    try:
        req = ec.recv_msg(conn)
    except (ConnectionError, OSError):
        conn.close()
        return
    cmd = req.get('cmd')
    log.info(f'{addr}: {cmd}')

    if cmd != 'ping' and not _job_lock.acquire(blocking=False):
        send({'type': 'error', 'msg': 'Server busy: another evaluation is running.', 'tb': ''})
        conn.close()
        return

    last = [0.0]

    def logf(msg):
        log.info(msg)
        send({'type': 'log', 'msg': msg})

    def progress(frac, text):
        now = time.time()
        if frac >= 1.0 or now - last[0] > 0.1:
            last[0] = now
            send({'type': 'progress', 'frac': frac, 'text': text})

    outcome = {}

    def job():
        try:
            outcome['msg'] = dict(run_job(req, logf, progress, cancel), type='result')
        except ec.Cancelled:
            outcome['msg'] = {'type': 'cancelled'}
        except Exception as e:
            tb = traceback.format_exc()
            log.error(tb)
            outcome['msg'] = {'type': 'error', 'msg': f'{type(e).__name__}: {e}', 'tb': tb}

    t = threading.Thread(target=job, daemon=True)
    t.start()
    try:
        # Listen for 'cancel' (or a disconnect) while the job runs.
        while t.is_alive():
            r, _, _ = select.select([conn], [], [], 0.5)
            if r:
                try:
                    msg = ec.recv_msg(conn)
                    if msg.get('cmd') == 'cancel':
                        log.info(f'{addr}: cancel requested')
                        cancel.set()
                except (ConnectionError, OSError):
                    cancel.set()
                    t.join()
                    break
        t.join()
        send(outcome.get('msg', {'type': 'error', 'msg': 'job ended without result', 'tb': ''}))
    finally:
        if cmd != 'ping':
            _job_lock.release()
        conn.close()
        log.info(f'{addr}: done ({outcome.get("msg", {}).get("type")})')


# ---------------------------------------------------------------------------
# Headless
# ---------------------------------------------------------------------------
def run_headless(settings_path):
    st = ec.settings_for_server(ec.EvalSettings.load_json(settings_path))

    def progress(frac, text):
        print(f'\r  {frac * 100:5.1f}%  {text:<60}', end='', flush=True)
        if frac >= 1.0:
            print()

    try:
        res = run_job({'cmd': 'evaluate', 'settings': st.to_dict()}, print, progress, None)
    except ValueError as e:
        print(e)
        return 2
    print(f'Done: {res["run_dir"]}')
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Checkpoint evaluation server (runs in the container)')
    ap.add_argument('--host', default='0.0.0.0', help='Bind host (default: 0.0.0.0)')
    ap.add_argument('--port', type=int, default=ec.DEFAULT_EVAL_PORT,
                    help=f'TCP port (default: {ec.DEFAULT_EVAL_PORT})')
    ap.add_argument('--headless', action='store_true', help='Run one evaluation and exit')
    ap.add_argument('--settings', help='Settings JSON (for --headless)')
    args = ap.parse_args()

    if args.headless:
        if not args.settings:
            ap.error('--headless requires --settings')
        sys.exit(run_headless(args.settings))

    env = _environment()
    log.info('Environment: ' + ', '.join(f'{k}={v}' for k, v in env.items()))

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(4)
    log.info(f'Listening on {args.host}:{args.port}  – waiting for eval GUI connections…')
    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        log.info('Server stopped.')
    finally:
        srv.close()


if __name__ == '__main__':
    main()
