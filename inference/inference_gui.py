# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
"""
inference_gui.py  –  Runs on the host (no GPU / no torch needed here).

GUI that mirrors crop_mask_tool.py but sends crops + mask to
inference_server.py running inside the container via a TCP socket,
then displays the returned results.

Usage:
    python3 inference_gui.py [--server localhost:9999]
                             [--input-dir ./images]
                             [--edge-dir  ./edges]

Then:
  1. Click "🔧 Server…" to confirm the server address.
  2. Open input / edge folders via File menu or toolbar.
  3. Click on the source image to position the 512×512 crop.
  4. Draw a mask (LMB = paint hole, RMB = erase).
  5. Press ⚡ Run  (or Space / Enter) to send to the server and see results.
"""

import os
import sys
import socket
import struct
import pickle
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CROP_SIZE  = 512
DISPLAY_SIZE = 256
MAX_PREV_W = 800
MAX_PREV_H = 600
BRUSH_MIN  = 2
BRUSH_MAX  = 60
BRUSH_DEF  = 16
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 9999

# ---------------------------------------------------------------------------
# Protocol  (must match inference_server.py exactly)
# ---------------------------------------------------------------------------
_HEADER = struct.Struct('>I')

def _recv_all(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('Connection closed by server')
        buf.extend(chunk)
    return bytes(buf)

def _send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(_HEADER.pack(len(data)) + data)

def _recv_msg(sock):
    raw = _recv_all(sock, _HEADER.size)
    (length,) = _HEADER.unpack(raw)
    return pickle.loads(_recv_all(sock, length))

# ---------------------------------------------------------------------------
# Server connection (persistent, re-connects on demand)
# ---------------------------------------------------------------------------
class ServerConnection:
    """Thin wrapper around a persistent TCP socket to the inference server."""

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self._sock = None
        self._lock = threading.Lock()

    # -- address helpers --
    def set_address(self, host, port):
        with self._lock:
            if self._sock:
                try: self._sock.close()
                except Exception: pass
                self._sock = None
            self.host = host
            self.port = port

    @property
    def address(self):
        return f'{self.host}:{self.port}'

    # -- connect / disconnect --
    def connect(self):
        """Connect (or reconnect) to the server. Raises on failure."""
        with self._lock:
            if self._sock:
                try: self._sock.close()
                except Exception: pass
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect((self.host, self.port))
            s.settimeout(120)   # 2-min timeout for inference
            self._sock = s

    def disconnect(self):
        with self._lock:
            if self._sock:
                try: self._sock.close()
                except Exception: pass
                self._sock = None

    # -- inference call --
    def infer(self, img_np, edge_np, mask_np, invert_edge=True):
        """Send arrays → get result dict back. Thread-safe."""
        with self._lock:
            if self._sock is None:
                raise ConnectionError('Not connected to server')
            _send_msg(self._sock, {
                'img':         img_np,
                'edge':        edge_np,
                'mask':        mask_np,
                'invert_edge': invert_edge,
            })
            return _recv_msg(self._sock)


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------
class InferenceGUI:

    def __init__(self, root, server_addr=f'{DEFAULT_HOST}:{DEFAULT_PORT}',
                 input_dir=None, edge_dir=None, crop_size=CROP_SIZE):
        self.root      = root
        self.root.title('Mural Structure — Inference GUI')
        self.crop_size = crop_size

        # Parse initial server address
        host, port = self._parse_addr(server_addr)
        self.conn  = ServerConnection(host, port)
        self.invert_edge = True

        # Image state
        self.input_dir    = input_dir
        self.edge_dir     = edge_dir
        self.image_files  = []
        self.current_idx  = 0
        self.pil_full     = None
        self.pil_edge_full= None
        self.display_scale= 1.0
        self._preview_offset_x = 0
        self._preview_offset_y = 0
        self.crop_x = 0
        self.crop_y = 0
        self.pil_crop      = None
        self.pil_edge_crop = None
        self.mask_array    = None

        # Brush
        self.brush_size = BRUSH_DEF
        self.drawing    = False
        self.erasing    = False

        # Build UI
        self._apply_dark_theme()
        self._build_menu()
        self._build_toolbar()
        self._build_main_layout()
        self._build_statusbar()

        # Key bindings
        self.root.bind('<Left>',   lambda e: self._prev_image())
        self.root.bind('<Right>',  lambda e: self._next_image())
        self.root.bind('<Return>', lambda e: self._run_inference())
        self.root.bind('<space>',  lambda e: self._run_inference())
        self.root.bind('<Escape>', lambda e: self._clear_mask())

        # Auto-load
        if self.input_dir:
            self._scan_input_dir()
            if self.image_files:
                self._load_image(0)

    # -----------------------------------------------------------------------
    # Dark theme
    # -----------------------------------------------------------------------
    def _apply_dark_theme(self):
        self.root.configure(bg='#1e1e2e')
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('.',
                    background='#1e1e2e', foreground='#cdd6f4',
                    fieldbackground='#313244', troughcolor='#45475a')
        s.configure('TLabel',      background='#1e1e2e', foreground='#cdd6f4')
        s.configure('TButton',     background='#313244', foreground='#cdd6f4')
        s.configure('TSeparator',  background='#45475a')
        s.configure('TLabelframe',
                    background='#1e1e2e', foreground='#89b4fa',
                    bordercolor='#45475a')
        s.configure('TLabelframe.Label',
                    background='#1e1e2e', foreground='#89b4fa',
                    font=('Consolas', 9, 'bold'))

    # -----------------------------------------------------------------------
    # Menu
    # -----------------------------------------------------------------------
    def _build_menu(self):
        mk = dict(bg='#1e1e2e', fg='#cdd6f4',
                  activebackground='#45475a', activeforeground='#cdd6f4')
        menubar  = tk.Menu(self.root, **mk)
        file_m   = tk.Menu(menubar, tearoff=0, **mk)
        file_m.add_command(label='Open Input Folder…',  command=self._pick_input_dir)
        file_m.add_command(label='Open Edges Folder…',  command=self._pick_edge_dir)
        file_m.add_separator()
        file_m.add_command(label='Exit', command=self.root.quit)
        menubar.add_cascade(label='File', menu=file_m)

        opt_m = tk.Menu(menubar, tearoff=0, **mk)
        opt_m.add_command(label='Set Server Address…',  command=self._pick_server)
        opt_m.add_command(label='Toggle Invert Edge',   command=self._toggle_invert_edge)
        menubar.add_cascade(label='Options', menu=opt_m)
        self.root.config(menu=menubar)

    # -----------------------------------------------------------------------
    # Toolbar
    # -----------------------------------------------------------------------
    def _build_toolbar(self):
        tb = tk.Frame(self.root, bg='#181825', bd=0, pady=4)
        tb.pack(side=tk.TOP, fill=tk.X)

        btn = dict(bg='#313244', fg='#cdd6f4', relief=tk.FLAT,
                   font=('Segoe UI', 9), cursor='hand2',
                   activebackground='#45475a', activeforeground='#cdd6f4',
                   bd=0, padx=8, pady=4)
        acc = dict(bg='#89b4fa', fg='#1e1e2e', relief=tk.FLAT,
                   font=('Segoe UI', 9, 'bold'), cursor='hand2',
                   activebackground='#b4befe', activeforeground='#1e1e2e',
                   bd=0, padx=10, pady=4)
        go  = dict(bg='#a6e3a1', fg='#1e1e2e', relief=tk.FLAT,
                   font=('Segoe UI', 10, 'bold'), cursor='hand2',
                   activebackground='#94e2d5', activeforeground='#1e1e2e',
                   bd=0, padx=14, pady=4)

        # Navigation
        tk.Button(tb, text='◀', command=self._prev_image, **btn).pack(side=tk.LEFT, padx=(8,2))
        tk.Button(tb, text='▶', command=self._next_image, **btn).pack(side=tk.LEFT, padx=2)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Server button
        tk.Button(tb, text='🐳 Server…', command=self._pick_server, **acc).pack(side=tk.LEFT, padx=2)
        self.server_label = tk.Label(tb, text=f'→ {self.conn.address}',
                                     bg='#181825', fg='#6c7086',
                                     font=('Consolas', 9))
        self.server_label.pack(side=tk.LEFT, padx=4)

        # Connect button
        self.conn_btn = tk.Button(tb, text='Connect', command=self._connect_server, **btn)
        self.conn_btn.pack(side=tk.LEFT, padx=2)
        self.conn_status = tk.Label(tb, text='●  disconnected',
                                    bg='#181825', fg='#f38ba8',
                                    font=('Consolas', 9))
        self.conn_status.pack(side=tk.LEFT, padx=4)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Brush
        tk.Label(tb, text='Brush:', bg='#181825', fg='#cdd6f4',
                 font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0,2))
        self.brush_var = tk.IntVar(value=self.brush_size)
        tk.Scale(tb, from_=BRUSH_MIN, to=BRUSH_MAX,
                 orient=tk.HORIZONTAL, variable=self.brush_var,
                 length=120, showvalue=True,
                 bg='#181825', fg='#cdd6f4', troughcolor='#45475a',
                 highlightthickness=0,
                 command=lambda v: setattr(self, 'brush_size', int(v))
                 ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Mask actions
        tk.Button(tb, text='Clear Mask', command=self._clear_mask, **btn).pack(side=tk.LEFT, padx=2)
        tk.Button(tb, text='Fill Mask',  command=self._fill_mask,  **btn).pack(side=tk.LEFT, padx=2)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Run
        self.run_btn = tk.Button(tb, text='⚡ Run Inference',
                                  command=self._run_inference, **go)
        self.run_btn.pack(side=tk.LEFT, padx=4)

        # Counter
        self.counter_label = tk.Label(tb, text='0 / 0',
                                       bg='#181825', fg='#6c7086',
                                       font=('Segoe UI', 9))
        self.counter_label.pack(side=tk.RIGHT, padx=12)

    # -----------------------------------------------------------------------
    # Main 3-column layout
    # -----------------------------------------------------------------------
    def _build_main_layout(self):
        outer = tk.Frame(self.root, bg='#1e1e2e')
        outer.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        pane = tk.PanedWindow(outer, orient=tk.HORIZONTAL, sashwidth=6,
                               bg='#45475a', bd=0)
        pane.pack(fill=tk.BOTH, expand=True)

        def lf(parent, label, label_fg='#89b4fa'):
            return tk.LabelFrame(parent, text=label,
                                  bg='#1e1e2e', fg=label_fg,
                                  font=('Consolas', 9, 'bold'),
                                  bd=1, relief=tk.FLAT, highlightthickness=1,
                                  highlightbackground='#45475a')

        # ── Left: full-image preview ──────────────────────────────────────
        left = lf(pane, 'Source Image  (click to position crop)')
        pane.add(left, minsize=320)
        self.preview_canvas = tk.Canvas(left, bg='#11111b', cursor='crosshair',
                                         highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind('<Button-1>',   self._on_preview_click)
        self.preview_canvas.bind('<B1-Motion>',  self._on_preview_click)
        self.preview_canvas.bind('<Configure>',  self._on_preview_resize)

        # ── Middle: crop + mask ───────────────────────────────────────────
        mid = tk.Frame(pane, bg='#1e1e2e')
        pane.add(mid, minsize=DISPLAY_SIZE + 20)

        crop_lf = lf(mid, 'Crop Preview')
        crop_lf.pack(side=tk.TOP, padx=4, pady=(4, 2))
        self.crop_canvas = tk.Canvas(crop_lf,
                                      width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                                      bg='#11111b', highlightthickness=0)
        self.crop_canvas.pack()

        mask_lf = lf(mid, 'Mask  (LMB = hole  RMB = erase)', label_fg='#f38ba8')
        mask_lf.pack(side=tk.TOP, padx=4, pady=(2, 4))
        self.mask_canvas = tk.Canvas(mask_lf,
                                      width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                                      bg='black', cursor='circle',
                                      highlightthickness=0)
        self.mask_canvas.pack()
        self.mask_canvas.bind('<Button-1>',        self._mask_start_draw)
        self.mask_canvas.bind('<B1-Motion>',       self._mask_draw)
        self.mask_canvas.bind('<ButtonRelease-1>', self._mask_stop_draw)
        self.mask_canvas.bind('<Button-3>',        self._mask_start_erase)
        self.mask_canvas.bind('<B3-Motion>',       self._mask_erase)
        self.mask_canvas.bind('<ButtonRelease-3>', self._mask_stop_erase)

        self.info_label = tk.Label(mid, text='',
                                    bg='#1e1e2e', fg='#6c7086',
                                    font=('Consolas', 8), anchor='w', justify='left')
        self.info_label.pack(fill=tk.X, padx=6, pady=2)

        # ── Right: results ────────────────────────────────────────────────
        right = tk.Frame(pane, bg='#1e1e2e')
        pane.add(right, minsize=DISPLAY_SIZE + 20)

        # Progress bar (shown during inference)
        pf = tk.Frame(right, bg='#1e1e2e')
        pf.pack(fill=tk.X, padx=4, pady=2)
        self.progress_label = tk.Label(pf, text='', bg='#1e1e2e', fg='#a6e3a1',
                                        font=('Consolas', 9))
        self.progress_label.pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(pf, mode='indeterminate', length=120)

        # Result panels
        comp_lf = lf(right, 'Composite  (Black-on-White)', label_fg='#a6e3a1')
        comp_lf.pack(side=tk.TOP, padx=4, pady=(4, 1))
        self.result_comp_canvas = tk.Canvas(comp_lf,
                                             width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                                             bg='#11111b', highlightthickness=0)
        self.result_comp_canvas.pack()

        hl_lf = lf(right, 'Highlight  (mask tinted blue)', label_fg='#89dceb')
        hl_lf.pack(side=tk.TOP, padx=4, pady=1)
        self.result_hl_canvas = tk.Canvas(hl_lf,
                                           width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                                           bg='#11111b', highlightthickness=0)
        self.result_hl_canvas.pack()

        raw_lf = lf(right, 'Raw sigmoid output', label_fg='#fab387')
        raw_lf.pack(side=tk.TOP, padx=4, pady=1)
        self.result_raw_canvas = tk.Canvas(raw_lf,
                                            width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                                            bg='#11111b', highlightthickness=0)
        self.result_raw_canvas.pack()

    def _build_statusbar(self):
        self.statusbar = tk.Label(
            self.root,
            text='Ready. Connect to the inference server, then open an input folder.',
            bg='#181825', fg='#6c7086',
            font=('Consolas', 8), anchor='w', bd=0, pady=3)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X, padx=6)

    # -----------------------------------------------------------------------
    # Server helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _parse_addr(addr_str):
        """Parse 'host:port' → (host, int(port))."""
        if ':' in addr_str:
            parts = addr_str.rsplit(':', 1)
            return parts[0], int(parts[1])
        return addr_str, DEFAULT_PORT

    def _pick_server(self):
        dlg = tk.Toplevel(self.root)
        dlg.title('Inference Server Address')
        dlg.configure(bg='#1e1e2e')
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text='Server address  (host:port)',
                 bg='#1e1e2e', fg='#cdd6f4',
                 font=('Segoe UI', 9)).grid(row=0, column=0, padx=12, pady=(12, 4), sticky='w')

        var = tk.StringVar(value=self.conn.address)
        entry = tk.Entry(dlg, textvariable=var, width=28,
                         bg='#313244', fg='#cdd6f4',
                         insertbackground='white',
                         font=('Consolas', 10), bd=0, highlightthickness=1,
                         highlightbackground='#45475a')
        entry.grid(row=1, column=0, padx=12, pady=4)
        entry.focus()

        def _apply():
            try:
                host, port = self._parse_addr(var.get().strip())
                self.conn.set_address(host, port)
                self.server_label.config(text=f'→ {self.conn.address}')
                self.conn_status.config(text='●  disconnected', fg='#f38ba8')
                self._status(f'Server set to {self.conn.address}')
            except Exception as e:
                messagebox.showerror('Bad address', str(e), parent=dlg)
                return
            dlg.destroy()

        btn_kw = dict(bg='#89b4fa', fg='#1e1e2e', relief=tk.FLAT,
                      font=('Segoe UI', 9, 'bold'), cursor='hand2', bd=0,
                      padx=10, pady=4)
        tk.Button(dlg, text='Apply & Disconnect', command=_apply,
                  **btn_kw).grid(row=2, column=0, pady=(8, 12))
        dlg.bind('<Return>', lambda e: _apply())

    def _connect_server(self):
        self.conn_btn.config(state=tk.DISABLED)
        self._status(f'Connecting to {self.conn.address}…')

        def _worker():
            try:
                self.conn.connect()
                self.root.after(0, self._connected_ok)
            except Exception as exc:
                self.root.after(0, lambda: self._connect_error(str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _connected_ok(self):
        self.conn_btn.config(state=tk.NORMAL)
        self.conn_status.config(text='●  connected', fg='#a6e3a1')
        self._status(f'Connected to server at {self.conn.address}')

    def _connect_error(self, err):
        self.conn_btn.config(state=tk.NORMAL)
        self.conn_status.config(text='●  disconnected', fg='#f38ba8')
        messagebox.showerror('Connection Failed',
                              f'Cannot reach {self.conn.address}\n\n{err}')
        self._status(f'Connection failed: {err}')

    def _toggle_invert_edge(self):
        self.invert_edge = not self.invert_edge
        state = 'ON' if self.invert_edge else 'OFF'
        self._status(f'Invert-edge: {state}  (ON = Black-on-White → 1=edge)')

    # -----------------------------------------------------------------------
    # Directory pickers
    # -----------------------------------------------------------------------
    def _pick_input_dir(self):
        d = filedialog.askdirectory(title='Select Input Image Folder')
        if d:
            self.input_dir = d
            self._scan_input_dir()
            if self.image_files:
                self._load_image(0)

    def _pick_edge_dir(self):
        d = filedialog.askdirectory(title='Select Edges Folder')
        if d:
            self.edge_dir = d
            self._status(f'Edges folder: {d}')
            if self.pil_full is not None:
                self._load_image(self.current_idx)

    # -----------------------------------------------------------------------
    # Image management
    # -----------------------------------------------------------------------
    def _scan_input_dir(self):
        self.image_files = sorted([
            f for f in os.listdir(self.input_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])
        self.current_idx = 0
        self._status(f'Found {len(self.image_files)} images in {self.input_dir}')
        self._update_counter()

    def _load_image(self, idx):
        if not self.image_files:
            return
        idx = max(0, min(idx, len(self.image_files) - 1))
        self.current_idx = idx
        fname = self.image_files[idx]

        try:
            self.pil_full = Image.open(
                os.path.join(self.input_dir, fname)).convert('RGB')
        except Exception as e:
            messagebox.showerror('Error', f'Cannot open {fname}:\n{e}')
            return

        self.pil_edge_full = None
        if self.edge_dir:
            stem = os.path.splitext(fname)[0]
            for f in os.listdir(self.edge_dir):
                if (os.path.splitext(f)[0] == stem and
                        os.path.splitext(f)[1].lower() in IMAGE_EXTS):
                    try:
                        self.pil_edge_full = Image.open(
                            os.path.join(self.edge_dir, f))
                    except Exception:
                        pass
                    break

        self._update_counter()
        self._display_preview()
        self._reset_panels()

        edge_ok = 'edge ✓' if self.pil_edge_full else 'NO edge'
        self._status(f'Loaded: {fname}  {self.pil_full.width}×{self.pil_full.height}  ({edge_ok})')

        cx = max(0, (self.pil_full.width  - self.crop_size) // 2)
        cy = max(0, (self.pil_full.height - self.crop_size) // 2)
        self._set_crop(cx, cy)

    def _prev_image(self):
        if self.current_idx > 0:
            self._load_image(self.current_idx - 1)

    def _next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self._load_image(self.current_idx + 1)

    # -----------------------------------------------------------------------
    # Preview canvas
    # -----------------------------------------------------------------------
    def _display_preview(self):
        if self.pil_full is None:
            return
        cw = self.preview_canvas.winfo_width()
        ch = self.preview_canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = MAX_PREV_W, MAX_PREV_H

        sx = cw / self.pil_full.width
        sy = ch / self.pil_full.height
        self.display_scale = min(sx, sy, 1.0)
        dw = int(self.pil_full.width  * self.display_scale)
        dh = int(self.pil_full.height * self.display_scale)

        resized = self.pil_full.resize((dw, dh), Image.LANCZOS)
        self._preview_tk = ImageTk.PhotoImage(resized)
        self.preview_canvas.delete('all')
        self.preview_canvas.create_image(cw // 2, ch // 2,
                                          image=self._preview_tk, anchor='center')
        self._preview_offset_x = (cw - dw) // 2
        self._preview_offset_y = (ch - dh) // 2

    def _on_preview_resize(self, event):
        self._display_preview()
        if self.pil_crop is not None:
            self._draw_crop_rect()

    def _on_preview_click(self, event):
        if self.pil_full is None:
            return
        fx = (event.x - self._preview_offset_x) / self.display_scale
        fy = (event.y - self._preview_offset_y) / self.display_scale
        cx = max(0, min(int(fx - self.crop_size/2), self.pil_full.width  - self.crop_size))
        cy = max(0, min(int(fy - self.crop_size/2), self.pil_full.height - self.crop_size))
        self._set_crop(cx, cy)

    def _set_crop(self, cx, cy):
        self.crop_x = cx
        self.crop_y = cy
        self.pil_crop = self.pil_full.crop(
            (cx, cy, cx + self.crop_size, cy + self.crop_size))
        if self.pil_edge_full is not None:
            self.pil_edge_crop = self.pil_edge_full.crop(
                (cx, cy, cx + self.crop_size, cy + self.crop_size))
        else:
            self.pil_edge_crop = None
        self.mask_array = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)
        self._draw_crop_rect()
        self._display_crop()
        self._display_mask()
        self._update_info()

    def _draw_crop_rect(self):
        self.preview_canvas.delete('crop_rect')
        if self.pil_full is None:
            return
        x1 = self._preview_offset_x + self.crop_x * self.display_scale
        y1 = self._preview_offset_y + self.crop_y * self.display_scale
        x2 = x1 + self.crop_size * self.display_scale
        y2 = y1 + self.crop_size * self.display_scale
        self.preview_canvas.create_rectangle(
            x1, y1, x2, y2, outline='#a6e3a1', width=2, dash=(4,4), tags='crop_rect')

    def _reset_panels(self):
        self.pil_crop = self.pil_edge_crop = self.mask_array = None
        for c in (self.crop_canvas, self.mask_canvas,
                   self.result_comp_canvas, self.result_hl_canvas,
                   self.result_raw_canvas):
            c.delete('all')

    # -----------------------------------------------------------------------
    # Crop display
    # -----------------------------------------------------------------------
    def _display_crop(self):
        if self.pil_crop is None:
            return
        self._crop_tk = ImageTk.PhotoImage(self.pil_crop.resize((DISPLAY_SIZE, DISPLAY_SIZE)))
        self.crop_canvas.delete('all')
        self.crop_canvas.create_image(0, 0, image=self._crop_tk, anchor='nw')

    # -----------------------------------------------------------------------
    # Mask drawing
    # -----------------------------------------------------------------------
    def _display_mask(self):
        if self.mask_array is None or self.pil_crop is None:
            return
        arr = np.array(self.pil_crop).copy()
        mb  = self.mask_array > 0
        arr[mb, 0] = np.clip(arr[mb, 0].astype(int) + 80, 0, 255).astype(np.uint8)
        arr[mb, 1] = (arr[mb, 1] * 0.3).astype(np.uint8)
        arr[mb, 2] = (arr[mb, 2] * 0.3).astype(np.uint8)
        self._mask_tk = ImageTk.PhotoImage(Image.fromarray(arr).resize((DISPLAY_SIZE, DISPLAY_SIZE)))
        self.mask_canvas.delete('all')
        self.mask_canvas.create_image(0, 0, image=self._mask_tk, anchor='nw')

    def _paint_mask(self, x, y, value):
        if self.mask_array is None:
            return
        scale = self.crop_size / DISPLAY_SIZE
        x = int(x * scale)
        y = int(y * scale)
        r  = self.brush_size // 2
        yc, xc = np.ogrid[-r:r+1, -r:r+1]
        circ = xc**2 + yc**2 <= r**2
        y1, y2 = max(0, y-r), min(self.crop_size, y+r+1)
        x1, x2 = max(0, x-r), min(self.crop_size, x+r+1)
        cy1 = max(0, r-y);  cy2 = cy1 + (y2-y1)
        cx1 = max(0, r-x);  cx2 = cx1 + (x2-x1)
        self.mask_array[y1:y2, x1:x2][circ[cy1:cy2, cx1:cx2]] = value

    def _mask_start_draw(self, e):
        self.drawing = True;  self._paint_mask(e.x, e.y, 255); self._display_mask()
    def _mask_draw(self, e):
        if self.drawing:      self._paint_mask(e.x, e.y, 255); self._display_mask()
    def _mask_stop_draw(self, e):
        self.drawing = False; self._update_info()
    def _mask_start_erase(self, e):
        self.erasing = True;  self._paint_mask(e.x, e.y, 0);   self._display_mask()
    def _mask_erase(self, e):
        if self.erasing:      self._paint_mask(e.x, e.y, 0);   self._display_mask()
    def _mask_stop_erase(self, e):
        self.erasing = False; self._update_info()
    def _clear_mask(self):
        if self.mask_array is not None:
            self.mask_array[:] = 0; self._display_mask(); self._update_info()
    def _fill_mask(self):
        if self.mask_array is not None:
            self.mask_array[:] = 255; self._display_mask(); self._update_info()

    # -----------------------------------------------------------------------
    # Inference (sends data to the inference server)
    # -----------------------------------------------------------------------
    def _run_inference(self):
        if self.pil_crop is None or self.mask_array is None:
            messagebox.showwarning('No Crop', 'Click on the source image to select a crop first.')
            return
        if np.all(self.mask_array == 0):
            messagebox.showwarning('Empty Mask', 'Paint a hole region on the mask canvas first.')
            return
        if self.conn._sock is None:
            messagebox.showwarning('Not Connected',
                                    f'Not connected to server.\n'
                                    f'Click "Connect" (server: {self.conn.address}).')
            return

        self.run_btn.config(state=tk.DISABLED, text='Running…')
        self.progress_label.config(text='Sending to server…')
        self.progress_bar.pack(side=tk.LEFT, padx=4)
        self.progress_bar.start(10)
        self._status('Running inference on the server…')

        # Prepare arrays
        img_np  = np.array(self.pil_crop.convert('L'), dtype=np.uint8)
        if self.pil_edge_crop is not None:
            edge_np = np.array(self.pil_edge_crop.convert('L'), dtype=np.uint8)
        else:
            edge_np = np.full((self.crop_size, self.crop_size), 255, dtype=np.uint8)
        mask_np = self.mask_array.copy()
        invert  = self.invert_edge

        def _worker():
            try:
                result = self.conn.infer(img_np, edge_np, mask_np, invert)
                if 'error' in result:
                    self.root.after(0, lambda: self._inference_error(result['error']))
                else:
                    self.root.after(0, lambda: self._show_results(result))
            except Exception as exc:
                err = str(exc)
                # Try to reconnect once on broken pipe
                try:
                    self.conn.connect()
                    result = self.conn.infer(img_np, edge_np, mask_np, invert)
                    if 'error' in result:
                        self.root.after(0, lambda: self._inference_error(result['error']))
                    else:
                        self.root.after(0, lambda: self._show_results(result))
                except Exception as exc2:
                    err2 = f'{err}\n\nReconnect also failed: {exc2}'
                    self.root.after(0, lambda: self._inference_error(err2))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_results(self, res):
        self._inference_done()

        comp = Image.fromarray(res['composite'], 'L')
        self._comp_tk = ImageTk.PhotoImage(comp.resize((DISPLAY_SIZE, DISPLAY_SIZE)))
        self.result_comp_canvas.delete('all')
        self.result_comp_canvas.create_image(0, 0, image=self._comp_tk, anchor='nw')

        hl = Image.fromarray(res['highlight'], 'RGB')
        self._hl_tk = ImageTk.PhotoImage(hl.resize((DISPLAY_SIZE, DISPLAY_SIZE)))
        self.result_hl_canvas.delete('all')
        self.result_hl_canvas.create_image(0, 0, image=self._hl_tk, anchor='nw')

        raw = (res['pred_raw'] * 255).clip(0, 255).astype(np.uint8)
        self._raw_tk = ImageTk.PhotoImage(Image.fromarray(raw, 'L').resize((DISPLAY_SIZE, DISPLAY_SIZE)))
        self.result_raw_canvas.delete('all')
        self.result_raw_canvas.create_image(0, 0, image=self._raw_tk, anchor='nw')

        hole_pct = np.mean(self.mask_array > 0) * 100
        self._status(f'Inference done ✓   hole={hole_pct:.1f}%  invert_edge={self.invert_edge}')

    def _inference_error(self, err):
        self._inference_done()
        self._status('Inference failed — see error dialog')
        messagebox.showerror('Inference Error', err)

    def _inference_done(self):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_label.config(text='')
        self.run_btn.config(state=tk.NORMAL, text='⚡ Run Inference')

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _update_counter(self):
        total = len(self.image_files)
        cur   = self.current_idx + 1 if total > 0 else 0
        self.counter_label.config(text=f'{cur} / {total}')

    def _update_info(self):
        if self.mask_array is None:
            self.info_label.config(text=''); return
        hole_pct = np.mean(self.mask_array > 0) * 100
        fname    = self.image_files[self.current_idx] if self.image_files else ''
        edge_ok  = '✓' if self.pil_edge_full is not None else '✗ missing'
        srv_ok   = '✓ connected' if self.conn._sock else '✗ not connected'
        self.info_label.config(text=(
            f'File: {fname}\n'
            f'Edge: {edge_ok}   Server: {srv_ok}\n'
            f'Crop: ({self.crop_x}, {self.crop_y})\n'
            f'Hole: {hole_pct:.1f}%   InvertEdge: {self.invert_edge}'
        ))

    def _status(self, msg):
        self.statusbar.config(text=msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Inference GUI (server runs in the container)')
    parser.add_argument('--server',    default=f'{DEFAULT_HOST}:{DEFAULT_PORT}',
                        help=f'Inference server address (default: {DEFAULT_HOST}:{DEFAULT_PORT})')
    parser.add_argument('--input-dir', default=None, help='Input image folder')
    parser.add_argument('--edge-dir',  default=None, help='Edge map folder')
    parser.add_argument('--size',      type=int, default=CROP_SIZE,
                        help=f'Crop size (default: {CROP_SIZE})')
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry('1500x840')
    root.minsize(1000, 600)
    InferenceGUI(root,
                 server_addr=args.server,
                 input_dir=args.input_dir,
                 edge_dir=args.edge_dir,
                 crop_size=args.size)
    root.mainloop()


if __name__ == '__main__':
    main()
