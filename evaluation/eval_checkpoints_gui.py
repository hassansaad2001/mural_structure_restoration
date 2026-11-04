"""
eval_checkpoints_gui.py  –  Runs on the host (numpy + Pillow + tkinter; no torch).

Compare several GeneratorSketch checkpoints on one test set, with valid-sample
filtering and Small / Medium / Large mask groups. All model work (filter,
inference, metrics, Excel export) is done by eval_server.py inside the container;
this GUI sends the settings over TCP and shows logs, progress and results.

Usage (from the repository root):
    bash scripts/eval_gui.sh                 # starts the server in the container + this GUI
  or by hand:
    # container: python3 evaluation/eval_server.py
    # host:      python3 evaluation/eval_checkpoints_gui.py [--settings my.json] [--server HOST:PORT]

Paths are entered as host paths and translated with the "Path map"
(e.g. /home/<user>/MAT=/master, matching `docker run -v /home/<user>/MAT:/master`, which
docker/run.sh uses). Leave it empty when GUI and server see the same paths.
"""
# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).

import io
import os
import sys
import glob
import queue
import socket
import subprocess
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import eval_core as ec
from eval_core import EvalSettings, GROUPS, ALL, OUT_OF_RANGE, SUMMARY_GROUPS, METRICS

# Palette shared with inference_gui.py
BG, BG2, PANEL, FG, MUTED = '#1e1e2e', '#181825', '#313244', '#cdd6f4', '#6c7086'
ACCENT, GOOD, WARN, BAD = '#89b4fa', '#a6e3a1', '#f9e2af', '#f38ba8'
BEST_BG = '#2e4a33'

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
# docker/run.sh mounts the parent of the repository as /master (docker run -v <parent>:/master)
DEFAULT_PATH_MAP = f'{os.path.dirname(REPO_ROOT)}=/master'


class EvalGUI:

    def __init__(self, root, settings=None):
        self.root = root
        self.root.title('MAT Sketch — Checkpoint Evaluation')
        self.q = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self._sock = None              # socket of the running server job (for cancel)

        self.filter_result = None
        self.group_info = None
        self.results = None
        self.run_dir = None
        self._preview_refs = []
        self._regroup_pending = False

        self._apply_theme()
        self._make_vars(settings or EvalSettings())
        self._build_menu()
        self._build_body()
        self._build_bottom()
        self.root.after(100, self._poll_queue)

    # -----------------------------------------------------------------------
    # Theme / variables
    # -----------------------------------------------------------------------
    def _apply_theme(self):
        self.root.configure(bg=BG)
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('.', background=BG, foreground=FG, fieldbackground=PANEL,
                    troughcolor='#45475a', bordercolor='#45475a')
        s.configure('TNotebook.Tab', background=PANEL, foreground=FG, padding=(12, 4))
        s.map('TNotebook.Tab', background=[('selected', '#45475a')])
        s.configure('TLabelframe', background=BG, bordercolor='#45475a')
        s.configure('TLabelframe.Label', background=BG, foreground=ACCENT,
                    font=('Consolas', 9, 'bold'))
        s.configure('TCheckbutton', background=BG, foreground=FG)
        s.configure('TRadiobutton', background=BG, foreground=FG)
        s.map('TCheckbutton', background=[('active', BG)])
        s.map('TRadiobutton', background=[('active', BG)])
        s.configure('TButton', background=PANEL, foreground=FG)
        s.configure('Go.TButton', background=GOOD, foreground=BG, font=('Segoe UI', 10, 'bold'))
        s.configure('TEntry', fieldbackground=PANEL, foreground=FG, insertcolor=FG)
        s.configure('TSpinbox', fieldbackground=PANEL, foreground=FG, arrowcolor=FG)
        s.configure('TCombobox', fieldbackground=PANEL, foreground=FG)
        s.configure('Treeview', background=PANEL, fieldbackground=PANEL, foreground=FG,
                    font=('Consolas', 9), rowheight=20)
        s.configure('Treeview.Heading', background='#45475a', foreground=ACCENT)
        s.map('Treeview', background=[('selected', '#585b70')])

    def _make_vars(self, st):
        """One tk variable per EvalSettings field (except the checkpoint list)."""
        self.vars = {}
        for f in ec.dataclasses.fields(EvalSettings):
            if f.name == 'checkpoints':
                continue
            v = getattr(st, f.name)
            if isinstance(v, bool):
                var = tk.BooleanVar(value=v)
            elif isinstance(v, int):
                var = tk.IntVar(value=v)
            elif isinstance(v, float):
                var = tk.DoubleVar(value=v)
            else:
                var = tk.StringVar(value=v)
            self.vars[f.name] = var
        if not self.vars['path_map'].get():
            self.vars['path_map'].set(DEFAULT_PATH_MAP)
        self._init_ckpt_specs = st.ckpt_specs()
        regroup_fields = ('grouping_mode', 'small_lo', 'small_hi', 'medium_lo', 'medium_hi',
                          'large_lo', 'large_hi', 'fid_min_group_size')
        for k in regroup_fields:
            self.vars[k].trace_add('write', lambda *_: self._schedule_regroup())

    def settings(self):
        d = {}
        for k, var in self.vars.items():
            try:
                d[k] = var.get()
            except tk.TclError:
                raise ValueError(f'Invalid value for "{k}"')
        rows = self._ckpt_rows()
        d['checkpoints'] = [p for p, _ in rows]
        d['checkpoint_networks'] = [n for _, n in rows]
        return EvalSettings.from_dict(d)

    def load_settings(self, st):
        for k, var in self.vars.items():
            var.set(getattr(st, k))
        self.ckpt_tree.delete(*self.ckpt_tree.get_children())
        for p, n in st.ckpt_specs():
            self._ckpt_insert(p, n)

    # -----------------------------------------------------------------------
    # Menu
    # -----------------------------------------------------------------------
    def _build_menu(self):
        mk = dict(bg=BG, fg=FG, activebackground='#45475a', activeforeground=FG)
        mb = tk.Menu(self.root, **mk)
        fm = tk.Menu(mb, tearoff=0, **mk)
        fm.add_command(label='Load Settings…', command=self._load_settings_file)
        fm.add_command(label='Save Settings…', command=self._save_settings_file)
        fm.add_separator()
        fm.add_command(label='Save Sample List…', command=self._save_sample_list)
        fm.add_command(label='Export Fixed Test Set…', command=self._export_test_set)
        fm.add_command(label='Load Sample List…', command=self._load_sample_list)
        fm.add_command(label='Clear Sample List', command=self._clear_sample_list)
        fm.add_separator()
        fm.add_command(label='Open Results Folder', command=self._open_results)
        fm.add_separator()
        fm.add_command(label='Exit', command=self.root.quit)
        mb.add_cascade(label='File', menu=fm)
        self.root.config(menu=mb)

    # -----------------------------------------------------------------------
    # Layout helpers
    # -----------------------------------------------------------------------
    def _row_path(self, parent, r, label, key, kind='dir'):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky='w', padx=6, pady=3)
        ttk.Entry(parent, textvariable=self.vars[key], width=70).grid(
            row=r, column=1, sticky='we', padx=4, pady=3)

        def pick():
            p = filedialog.askdirectory(title=label) if kind == 'dir' else filedialog.askopenfilename()
            if p:
                self.vars[key].set(p)
        ttk.Button(parent, text='…', width=3, command=pick).grid(row=r, column=2, padx=4)

    @staticmethod
    def _spin(parent, var, lo, hi, inc, width=8, fmt=None):
        kw = dict(from_=lo, to=hi, increment=inc, textvariable=var, width=width)
        if fmt:
            kw['format'] = fmt
        return ttk.Spinbox(parent, **kw)

    def _labeled(self, parent, r, c, text, widget):
        ttk.Label(parent, text=text).grid(row=r, column=c, sticky='w', padx=(8, 2), pady=3)
        widget.grid(row=r, column=c + 1, sticky='w', padx=(0, 8), pady=3)
        return widget

    # -----------------------------------------------------------------------
    # Body: notebook with 3 tabs
    # -----------------------------------------------------------------------
    def _build_body(self):
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 2))
        self._build_setup_tab()
        self._build_filter_tab()
        self._build_results_tab()
        self._build_sample_tab()

    def _build_setup_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text='1 · Data & Checkpoints')
        tab.columnconfigure(0, weight=1)

        srv = ttk.LabelFrame(tab, text='🐳 Eval server (eval_server.py, in the container)')
        srv.grid(row=0, column=0, sticky='we', padx=6, pady=6)
        srv.columnconfigure(3, weight=1)
        ttk.Label(srv, text='Address').grid(row=0, column=0, sticky='w', padx=6, pady=3)
        ttk.Entry(srv, textvariable=self.vars['server_addr'], width=18).grid(row=0, column=1, sticky='w')
        ttk.Label(srv, text='Path map (host=container)').grid(row=0, column=2, sticky='e', padx=(16, 4))
        ttk.Entry(srv, textvariable=self.vars['path_map'], width=40).grid(row=0, column=3, sticky='we')
        ttk.Button(srv, text='Test connection', command=self._ping).grid(row=0, column=4, padx=6)
        self.srv_status = tk.Label(srv, text='●  not checked', bg=BG, fg=MUTED, font=('Consolas', 9))
        self.srv_status.grid(row=1, column=0, columnspan=5, sticky='w', padx=6, pady=(0, 4))

        paths = ttk.LabelFrame(tab, text='Folders (host paths)')
        paths.grid(row=1, column=0, sticky='we', padx=6, pady=6)
        paths.columnconfigure(1, weight=1)
        self._row_path(paths, 0, 'Images', 'image_dir')
        self._row_path(paths, 1, 'Edges (GT)', 'edge_dir')
        self._row_path(paths, 2, 'Masks', 'mask_dir')
        self._row_path(paths, 3, 'Results', 'results_dir')

        ck = ttk.LabelFrame(tab, text='Checkpoints (evaluated in this order)')
        ck.grid(row=2, column=0, sticky='nsew', padx=6, pady=6)
        tab.rowconfigure(2, weight=1)
        ck.columnconfigure(0, weight=1)
        ck.rowconfigure(0, weight=1)
        # values: (path, network shown, override); the override column is hidden.
        self.ckpt_tree = ttk.Treeview(ck, columns=('path', 'network', 'override'),
                                      displaycolumns=('path', 'network'),
                                      show='headings', selectmode='extended', height=8)
        self.ckpt_tree.heading('path', text='Checkpoint')
        self.ckpt_tree.heading('network', text='Network class   (double-click to set)')
        self.ckpt_tree.column('path', width=560)
        self.ckpt_tree.column('network', width=420)
        self.ckpt_tree.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
        self.ckpt_tree.bind('<Double-1>', lambda e: self._set_ckpt_network())
        for p, n in self._init_ckpt_specs:
            self._ckpt_insert(p, n)
        self.vars['network'].trace_add('write', lambda *_: self._ckpt_refresh())
        bf = ttk.Frame(ck)
        bf.grid(row=0, column=1, sticky='n', padx=4, pady=4)
        for text, cmd in (('Add files…', self._add_ckpt_files),
                          ('Add folder…', self._add_ckpt_folder),
                          ('Set network…', self._set_ckpt_network),
                          ('Move up', lambda: self._move_ckpt(-1)),
                          ('Move down', lambda: self._move_ckpt(1)),
                          ('Remove', self._remove_ckpt),
                          ('Clear', lambda: self.ckpt_tree.delete(*self.ckpt_tree.get_children()))):
            ttk.Button(bf, text=text, command=cmd, width=13).pack(fill=tk.X, pady=2)

        opt = ttk.LabelFrame(tab, text='Model & data conventions')
        opt.grid(row=3, column=0, sticky='we', padx=6, pady=6)
        v = self.vars
        ttk.Label(opt, text='Default network class').grid(row=0, column=0, sticky='w', padx=8)
        ttk.Entry(opt, textvariable=v['network'], width=40).grid(row=0, column=1, columnspan=3,
                                                                  sticky='w', pady=3)
        self._labeled(opt, 0, 4, 'Resolution', self._spin(opt, v['resolution'], 64, 4096, 64))
        self._labeled(opt, 0, 6, 'Batch size', self._spin(opt, v['batch_size'], 1, 256, 1))
        ttk.Checkbutton(opt, text='Force CPU', variable=v['force_cpu']).grid(row=0, column=8, padx=8)

        ttk.Checkbutton(opt, text='Edge files are black-on-white',
                        variable=v['edge_black_on_white']).grid(row=1, column=0, columnspan=2,
                                                                 sticky='w', padx=8)
        ttk.Checkbutton(opt, text='Mask files: white = hole',
                        variable=v['mask_white_is_hole']).grid(row=1, column=2, columnspan=2,
                                                                sticky='w', padx=8)
        self._labeled(opt, 1, 4, 'Fit to resolution',
                      ttk.Combobox(opt, textvariable=v['fit_mode'], state='readonly', width=11,
                                   values=['center_crop', 'resize']))

        self._labeled(opt, 2, 0, 'Mask pairing',
                      ttk.Combobox(opt, textvariable=v['mask_pairing'], state='readonly', width=9,
                                   values=['auto', 'by_name', 'pool']))
        self._labeled(opt, 2, 2, 'Masks / image (pool)', self._spin(opt, v['masks_per_image'], 1, 100, 1))
        self._labeled(opt, 2, 4, 'Pool seed', self._spin(opt, v['seed'], 0, 10 ** 9, 1))
        ttk.Label(opt, foreground=MUTED, justify='left', text=(
            'auto = by_name if any mask stem equals an image stem or starts with "<stem>_", "<stem>-", "<stem>."; '
            'otherwise a shared pool with fixed per-image seeded draws.')).grid(
            row=3, column=0, columnspan=9, sticky='w', padx=8, pady=(0, 4))

        met = ttk.LabelFrame(tab, text='Metrics')
        met.grid(row=4, column=0, sticky='we', padx=6, pady=6)
        self._labeled(met, 0, 0, 'PSNR/SSIM/L1 region',
                      ttk.Combobox(met, textvariable=v['metric_region'], state='readonly', width=6,
                                   values=['hole', 'full']))
        ttk.Checkbutton(met, text='Binarize prediction (>0.5)', variable=v['binarize_pred']).grid(
            row=0, column=2, padx=8)
        ttk.Checkbutton(met, text='LPIPS', variable=v['compute_lpips']).grid(row=0, column=3, padx=8)
        ttk.Checkbutton(met, text='FID', variable=v['compute_fid']).grid(row=0, column=4, padx=8)
        self._labeled(met, 0, 5, 'FID warn below n =', self._spin(met, v['fid_min_group_size'], 2, 100000, 10))

    def _build_filter_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text='2 · Filter & Groups')
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        v = self.vars

        # ── A) Filter ─────────────────────────────────────────────────────
        fl = ttk.LabelFrame(tab, text='A) Reject samples whose hole has no useful edges')
        fl.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        self._labeled(fl, 0, 0, 'Edge threshold (0..1)', self._spin(fl, v['edge_threshold'], 0.0, 1.0, 0.05, fmt='%.2f'))
        self._labeled(fl, 1, 0, 'Min edge pixels in hole', self._spin(fl, v['min_edge_pixels'], 0, 10 ** 7, 5))
        self._labeled(fl, 2, 0, 'Min edge density in hole',
                      self._spin(fl, v['min_edge_density'], 0.0, 1.0, 0.0005, fmt='%.4f'))
        ttk.Checkbutton(fl, variable=v['require_connected'],
                        text='Require hole edges to cross the hole boundary\n'
                             '(connected to valid-region edges; stricter)').grid(
            row=3, column=0, columnspan=2, sticky='w', padx=8, pady=4)
        ttk.Button(fl, text='Run filter', command=self._run_filter).grid(
            row=4, column=0, sticky='w', padx=8, pady=6)
        self.filter_stats = tk.Label(fl, text='Not run yet.', bg=BG, fg=MUTED, justify='left',
                                     font=('Consolas', 9), anchor='w')
        self.filter_stats.grid(row=5, column=0, columnspan=2, sticky='w', padx=8, pady=4)

        # ── B) Groups ─────────────────────────────────────────────────────
        gr = ttk.LabelFrame(tab, text='B) Mask-ratio groups (hole area / image area)')
        gr.grid(row=0, column=1, sticky='nsew', padx=6, pady=6)
        ttk.Radiobutton(gr, text='Fixed ranges (%)', value='fixed',
                        variable=v['grouping_mode']).grid(row=0, column=0, columnspan=2, sticky='w', padx=8)
        ttk.Radiobutton(gr, text='Auto (tertiles, equal counts)', value='tertiles',
                        variable=v['grouping_mode']).grid(row=0, column=2, columnspan=3, sticky='w', padx=8)
        for r, g in enumerate(GROUPS, start=1):
            key = g.lower()
            ttk.Label(gr, text=g).grid(row=r, column=0, sticky='w', padx=8, pady=2)
            self._spin(gr, v[f'{key}_lo'], 0, 100, 1, width=6, fmt='%.1f').grid(row=r, column=1, pady=2)
            ttk.Label(gr, text='–').grid(row=r, column=2)
            self._spin(gr, v[f'{key}_hi'], 0, 100, 1, width=6, fmt='%.1f').grid(row=r, column=3, pady=2)
            ttk.Label(gr, text='%').grid(row=r, column=4, sticky='w')
        self.group_stats = tk.Label(gr, text='Run the filter to see group sizes.', bg=BG, fg=MUTED,
                                    justify='left', font=('Consolas', 9), anchor='w')
        self.group_stats.grid(row=4, column=0, columnspan=6, sticky='w', padx=8, pady=6)

        # ── D) Grids ──────────────────────────────────────────────────────
        gd = ttk.LabelFrame(tab, text='Comparison grids')
        gd.grid(row=1, column=0, columnspan=2, sticky='we', padx=6, pady=6)
        self._labeled(gd, 0, 0, 'Grid samples per group', self._spin(gd, v['grid_per_group'], 0, 200, 1))
        self._labeled(gd, 0, 2, 'Grid seed', self._spin(gd, v['grid_seed'], 0, 10 ** 9, 1))
        self._labeled(gd, 0, 4, 'Tile size (px)', self._spin(gd, v['grid_tile'], 64, 1024, 64))
        ttk.Button(gd, text='Preview grid', command=self._run_preview).grid(row=0, column=6, padx=8)

        # ── Fixed sample list (manifest) ──────────────────────────────────
        ml = ttk.LabelFrame(tab, text='Fixed sample list  (same images / edges / masks for every model)')
        ml.grid(row=2, column=0, columnspan=2, sticky='we', padx=6, pady=6)
        for c, (text, cmd) in enumerate((('Save sample list…', self._save_sample_list),
                                         ('Export fixed test set…', self._export_test_set),
                                         ('Load sample list…', self._load_sample_list),
                                         ('Clear (use folders)', self._clear_sample_list))):
            ttk.Button(ml, text=text, command=cmd).grid(row=0, column=c, padx=6, pady=4, sticky='w')
        self.manifest_status = tk.Label(ml, text='', bg=BG, fg=MUTED, font=('Consolas', 9),
                                        anchor='w', justify='left')
        self.manifest_status.grid(row=1, column=0, columnspan=4, sticky='w', padx=6, pady=(0, 4))
        ttk.Label(ml, foreground=MUTED, justify='left', text=(
            'Save: list of the accepted samples (JSON + CSV with relative and absolute paths, groups, '
            'and the filter/group settings).\n'
            'Export: also writes the processed 512×512 images/, edges/, masks/ (white = hole) for '
            'running other models on exactly the same inputs.\n'
            'Every evaluation also saves its list as samples.json in its results folder.')).grid(
            row=2, column=0, columnspan=4, sticky='w', padx=6, pady=(0, 4))
        self.vars['manifest_path'].trace_add('write', lambda *_: self._update_manifest_status())
        self._update_manifest_status()

    def _build_results_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text='3 · Results')
        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(top, text='Group:').pack(side=tk.LEFT)
        self.res_group = tk.StringVar(value=ALL)
        cb = ttk.Combobox(top, textvariable=self.res_group, state='readonly', width=8,
                          values=list(SUMMARY_GROUPS))
        cb.pack(side=tk.LEFT, padx=4)
        cb.bind('<<ComboboxSelected>>', lambda e: self._render_table())
        self.res_side = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text='All groups side by side', variable=self.res_side,
                        command=self._render_table).pack(side=tk.LEFT, padx=10)
        ttk.Button(top, text='Open results folder', command=self._open_results).pack(side=tk.RIGHT)
        self.res_info = tk.Label(top, text='', bg=BG, fg=MUTED, font=('Consolas', 9))
        self.res_info.pack(side=tk.RIGHT, padx=10)

        wrap = ttk.Frame(tab)
        wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.table_canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        ys = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=self.table_canvas.yview)
        xs = ttk.Scrollbar(wrap, orient=tk.HORIZONTAL, command=self.table_canvas.xview)
        self.table_canvas.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side=tk.RIGHT, fill=tk.Y)
        xs.pack(side=tk.BOTTOM, fill=tk.X)
        self.table_canvas.pack(fill=tk.BOTH, expand=True)
        self.table = tk.Frame(self.table_canvas, bg=BG)
        self.table_canvas.create_window((0, 0), window=self.table, anchor='nw')
        self.table.bind('<Configure>', lambda e: self.table_canvas.configure(
            scrollregion=self.table_canvas.bbox('all')))
        tk.Label(self.table, text='No results yet.', bg=BG, fg=MUTED).grid(row=0, column=0)

    def _build_sample_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text='4 · Sample viewer')
        pane = tk.PanedWindow(tab, orient=tk.HORIZONTAL, sashwidth=6, bg='#45475a', bd=0)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ── Left: accepted samples ────────────────────────────────────────
        left = ttk.Frame(pane)
        pane.add(left, minsize=360, width=400)
        top = ttk.Frame(left)
        top.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(top, text='Group').pack(side=tk.LEFT)
        self.smp_group = tk.StringVar(value=ALL)
        cb = ttk.Combobox(top, textvariable=self.smp_group, state='readonly', width=11,
                          values=[ALL] + list(GROUPS) + [OUT_OF_RANGE])
        cb.pack(side=tk.LEFT, padx=4)
        cb.bind('<<ComboboxSelected>>', lambda e: self._populate_samples())
        ttk.Label(top, text='Search').pack(side=tk.LEFT, padx=(8, 0))
        self.smp_search = tk.StringVar()
        ttk.Entry(top, textvariable=self.smp_search, width=14).pack(side=tk.LEFT, padx=4)
        self.smp_search.trace_add('write', lambda *_: self._populate_samples())

        cols = ('id', 'group', 'ratio', 'edges')
        self.smp_tree = ttk.Treeview(left, columns=cols, show='headings', selectmode='browse')
        for c, text, w in (('id', 'Sample', 170), ('group', 'Group', 70),
                           ('ratio', 'Mask %', 60), ('edges', 'Edge px', 65)):
            self.smp_tree.heading(c, text=text, command=lambda c=c: self._sort_samples(c))
            self.smp_tree.column(c, width=w, anchor='w' if c == 'id' else 'e')
        ys = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.smp_tree.yview)
        self.smp_tree.configure(yscrollcommand=ys.set)
        ys.pack(side=tk.RIGHT, fill=tk.Y)
        self.smp_tree.pack(fill=tk.BOTH, expand=True)
        self.smp_tree.bind('<<TreeviewSelect>>', lambda e: self._on_sample_select())
        self.smp_tree.bind('<Return>', lambda e: self._run_sample())
        self.smp_tree.bind('<Double-1>', lambda e: self._run_sample())
        self._smp_sort = ('id', False)

        bot = ttk.Frame(left)
        bot.pack(fill=tk.X, pady=(4, 0))
        self.smp_auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(bot, text='Run on select', variable=self.smp_auto).pack(side=tk.LEFT)
        ttk.Button(bot, text='▶ Run all checkpoints', command=self._run_sample).pack(side=tk.RIGHT)
        self.smp_count = tk.Label(left, text='Run the filter to list samples.', bg=BG, fg=MUTED,
                                  font=('Consolas', 8), anchor='w')
        self.smp_count.pack(fill=tk.X)

        # ── Right: results for the selected sample ────────────────────────
        right = ttk.Frame(pane)
        pane.add(right, minsize=500)
        opts = ttk.Frame(right)
        opts.pack(fill=tk.X, pady=(0, 4))
        self.smp_view = tk.StringVar(value='comp')
        ttk.Radiobutton(opts, text='Composite', value='comp', variable=self.smp_view,
                        command=self._render_sample).pack(side=tk.LEFT)
        ttk.Radiobutton(opts, text='Raw sigmoid', value='raw', variable=self.smp_view,
                        command=self._render_sample).pack(side=tk.LEFT, padx=8)
        ttk.Label(opts, text='Tile').pack(side=tk.LEFT, padx=(12, 2))
        self.smp_tile = tk.IntVar(value=256)
        tcb = ttk.Combobox(opts, textvariable=self.smp_tile, state='readonly', width=5,
                           values=[192, 256, 384, 512])
        tcb.pack(side=tk.LEFT)
        tcb.bind('<<ComboboxSelected>>', lambda e: self._render_sample())
        ttk.Label(opts, text='Columns').pack(side=tk.LEFT, padx=(12, 2))
        self.smp_cols = tk.IntVar(value=4)
        ccb = ttk.Combobox(opts, textvariable=self.smp_cols, state='readonly', width=3,
                           values=[2, 3, 4, 5, 6])
        ccb.pack(side=tk.LEFT)
        ccb.bind('<<ComboboxSelected>>', lambda e: self._render_sample())
        self.smp_info = tk.Label(right, text='Select a sample on the left.', bg=BG, fg=MUTED,
                                 font=('Consolas', 9), anchor='w', justify='left')
        self.smp_info.pack(fill=tk.X)

        wrap = ttk.Frame(right)
        wrap.pack(fill=tk.BOTH, expand=True)
        self.smp_canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        sy = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=self.smp_canvas.yview)
        sx = ttk.Scrollbar(wrap, orient=tk.HORIZONTAL, command=self.smp_canvas.xview)
        self.smp_canvas.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side=tk.RIGHT, fill=tk.Y)
        sx.pack(side=tk.BOTTOM, fill=tk.X)
        self.smp_canvas.pack(fill=tk.BOTH, expand=True)
        self.smp_frame = tk.Frame(self.smp_canvas, bg=BG)
        self.smp_canvas.create_window((0, 0), window=self.smp_frame, anchor='nw')
        self.smp_frame.bind('<Configure>', lambda e: self.smp_canvas.configure(
            scrollregion=self.smp_canvas.bbox('all')))

        self._smp_data = None          # last server result (PIL images decoded)
        self._smp_refs = []
        self._smp_pending = None       # sample selected while a job was running

    def _build_bottom(self):
        bar = tk.Frame(self.root, bg=BG2)
        bar.pack(fill=tk.X, side=tk.TOP, padx=6, pady=2)
        self.run_btn = ttk.Button(bar, text='⚡ Run Evaluation', style='Go.TButton',
                                  command=self._run_eval)
        self.run_btn.pack(side=tk.LEFT, padx=4, pady=4)
        self.cancel_btn = ttk.Button(bar, text='Cancel', command=self._cancel_job, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=4)
        self.pbar = ttk.Progressbar(bar, mode='determinate', length=300, maximum=1.0)
        self.pbar.pack(side=tk.LEFT, padx=8)
        self.plabel = tk.Label(bar, text='', bg=BG2, fg=GOOD, font=('Consolas', 9))
        self.plabel.pack(side=tk.LEFT)

        self.logbox = tk.Text(self.root, height=9, bg='#11111b', fg=FG, font=('Consolas', 8),
                              bd=0, highlightthickness=0, wrap='word')
        self.logbox.pack(fill=tk.X, side=tk.TOP, padx=6, pady=(0, 6))

    # -----------------------------------------------------------------------
    # Checkpoint list actions
    # -----------------------------------------------------------------------
    def _network_label(self, path, override):
        try:
            default = self.vars['network'].get()
        except tk.TclError:
            default = ''
        cls, _, src = ec.resolve_network(path, override, default)
        return cls if src == 'manual' else f'auto → {cls}  ({src})'

    def _ckpt_insert(self, path, override=''):
        self.ckpt_tree.insert('', tk.END, values=(path, self._network_label(path, override), override))

    def _ckpt_rows(self):
        return [(self.ckpt_tree.set(i, 'path'), self.ckpt_tree.set(i, 'override'))
                for i in self.ckpt_tree.get_children()]

    def _ckpt_refresh(self):
        for i in self.ckpt_tree.get_children():
            self.ckpt_tree.set(i, 'network', self._network_label(self.ckpt_tree.set(i, 'path'),
                                                                 self.ckpt_tree.set(i, 'override')))

    def _add_ckpt_files(self):
        for p in filedialog.askopenfilenames(title='Select checkpoints',
                                             filetypes=[('PyTorch checkpoint', '*.pth *.pt'), ('All', '*')]):
            self._ckpt_insert(p)

    def _add_ckpt_folder(self):
        d = filedialog.askdirectory(title='Folder with checkpoint-*.pth')
        if not d:
            return
        found = sorted(glob.glob(os.path.join(d, 'checkpoint-[0-9]*.pth')))
        if not found:
            found = sorted(glob.glob(os.path.join(d, '*.pth')))
        for p in found:
            self._ckpt_insert(p)
        self._log(f'Added {len(found)} checkpoints from {d}')

    def _move_ckpt(self, delta):
        tree = self.ckpt_tree
        sel = list(tree.selection())
        if delta > 0:
            sel.reverse()
        for item in sel:
            idx = tree.index(item)
            tree.move(item, '', max(0, idx + delta))

    def _remove_ckpt(self):
        self.ckpt_tree.delete(*self.ckpt_tree.selection())

    @staticmethod
    def _known_networks():
        """Generator classes found in networks/*.py (importable module names only)."""
        import re
        out = []
        for f in sorted(glob.glob(os.path.join(REPO_ROOT, 'networks', '*.py'))):
            mod = os.path.splitext(os.path.basename(f))[0]
            if not mod.isidentifier():
                continue
            try:
                with open(f, encoding='utf-8', errors='ignore') as fh:
                    src = fh.read()
            except OSError:
                continue
            out += [f'networks.{mod}.{c}' for c in re.findall(r'^class (\w*Generator\w*)\(', src, re.M)]
        return out

    def _set_ckpt_network(self):
        sel = self.ckpt_tree.selection()
        if not sel:
            messagebox.showinfo('Set network', 'Select one or more checkpoints first.')
            return
        dlg = tk.Toplevel(self.root)
        dlg.title('Network class')
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.grab_set()
        ttk.Label(dlg, text=f'Network class for {len(sel)} checkpoint(s).\n'
                            'Leave empty for auto: training_options.json next to the checkpoint,\n'
                            'otherwise the default "Network class" below.').grid(
            row=0, column=0, padx=12, pady=(12, 6), sticky='w')
        var = tk.StringVar(value=self.ckpt_tree.set(sel[0], 'override'))
        cb = ttk.Combobox(dlg, textvariable=var, width=60, values=[''] + self._known_networks())
        cb.grid(row=1, column=0, padx=12, pady=4)
        cb.focus()

        def apply():
            v = var.get().strip()
            for i in sel:
                self.ckpt_tree.set(i, 'override', v)
                self.ckpt_tree.set(i, 'network', self._network_label(self.ckpt_tree.set(i, 'path'), v))
            dlg.destroy()
        ttk.Button(dlg, text='Apply', command=apply).grid(row=2, column=0, pady=(6, 12))
        dlg.bind('<Return>', lambda e: apply())

    # -----------------------------------------------------------------------
    # Settings files
    # -----------------------------------------------------------------------
    def _save_settings_file(self):
        try:
            st = self.settings()
        except ValueError as e:
            messagebox.showerror('Invalid settings', str(e))
            return
        p = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON', '*.json')])
        if p:
            st.save_json(p)
            self._log(f'Settings saved to {p}')

    def _load_settings_file(self):
        p = filedialog.askopenfilename(filetypes=[('JSON', '*.json')])
        if p:
            self.load_settings(EvalSettings.load_json(p))
            self._log(f'Settings loaded from {p}')

    def _open_results(self):
        d = self.run_dir or self.vars['results_dir'].get()
        if not d or not os.path.isdir(d):
            messagebox.showinfo('Results', 'No results folder yet.')
            return
        try:   # WSL: open in Windows Explorer (fails on native Linux -> xdg-open)
            win = subprocess.run(['wslpath', '-w', d], capture_output=True, text=True).stdout.strip()
            subprocess.Popen(['explorer.exe', win])
            return
        except OSError:
            pass
        try:
            subprocess.Popen(['xdg-open', d])
        except OSError:
            messagebox.showinfo('Results', d)

    # -----------------------------------------------------------------------
    # Background jobs
    # -----------------------------------------------------------------------
    def _log(self, msg):
        self.logbox.insert(tk.END, msg + '\n')
        self.logbox.see(tk.END)

    def _start_job(self, target, *args):
        if self.busy:
            messagebox.showinfo('Busy', 'A job is already running.')
            return False
        self.busy = True
        self.cancel.clear()
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.pbar['value'] = 0

        def wrapper():
            try:
                target(*args)
            except ec.Cancelled:
                self.q.put(('log', 'Cancelled.'))
            except (ConnectionError, RuntimeError) as e:     # server-side errors are already logged
                self.q.put(('error', str(e)))
            except Exception as e:
                self.q.put(('log', traceback.format_exc()))
                self.q.put(('error', f'{type(e).__name__}: {e}'))
            finally:
                self.q.put(('finished',))
        threading.Thread(target=wrapper, daemon=True).start()
        return True

    def _qlog(self, msg):
        self.q.put(('log', msg))

    def _qprog(self, frac, text):
        self.q.put(('progress', frac, text))

    def _poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == 'log':
                    self._log(msg[1])
                elif kind == 'progress':
                    self.pbar['value'] = msg[1]
                    self.plabel.config(text=msg[2])
                elif kind == 'filter_done':
                    self._on_filter_done(msg[1])
                elif kind == 'eval_done':
                    self._on_eval_done(msg[1], msg[2])
                elif kind == 'preview_done':
                    self._show_preview(msg[1])
                elif kind == 'ping_done':
                    self._on_ping(msg[1])
                elif kind == 'sample_done':
                    self._on_sample_done(msg[1])
                elif kind == 'export_done':
                    self._on_export_done(msg[1])
                elif kind == 'error':
                    text = msg[1]
                    if len(text) > 1500:
                        text = text[:1500] + '\n…\n(full message in the log below)'
                    messagebox.showerror('Error', text)
                elif kind == 'finished':
                    self.busy = False
                    self.run_btn.config(state=tk.NORMAL)
                    self.cancel_btn.config(state=tk.DISABLED)
                    if self._smp_pending is not None:      # a sample was clicked meanwhile
                        self._smp_pending = None
                        if not self.cancel.is_set():
                            self.root.after_idle(self._run_sample)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _get_settings_or_warn(self, need_checkpoints):
        try:
            st = self.settings()
        except ValueError as e:
            messagebox.showerror('Invalid settings', str(e))
            return None
        errs = st.validate(need_checkpoints=need_checkpoints)
        if errs:
            messagebox.showerror('Invalid settings', '\n'.join(errs))
            return None
        return st

    # -- server communication (worker thread) --
    def _server_job(self, cmd, st, **extra):
        """Run one command on eval_server.py, relaying logs/progress; returns the result dict."""
        host, port = ec.parse_addr(st.server_addr)
        srv_settings = ec.settings_for_server(st)
        try:
            sock = socket.create_connection((host, port), timeout=10)
        except OSError as e:
            raise ConnectionError(
                f'Cannot reach the eval server at {host}:{port}\n({e})\n\n'
                f'Start it with:  bash scripts/eval_gui.sh   (starts server + GUI)\n'
                f'or in the container:  python3 evaluation/eval_server.py\n'
                f'(docker/run.sh publishes port 9998)') from e
        sock.settimeout(None)
        try:
            ec.send_msg(sock, dict(extra, cmd=cmd, settings=srv_settings.to_dict()))
            self._sock = sock                  # from now on Cancel may send on it
            if self.cancel.is_set():
                self._send_cancel()
            while True:
                msg = ec.recv_msg(sock)
                kind = msg['type']
                if kind == 'log':
                    self._qlog(msg['msg'])
                elif kind == 'progress':
                    self._qprog(msg['frac'], msg['text'])
                elif kind == 'result':
                    return msg
                elif kind == 'cancelled':
                    raise ec.Cancelled()
                elif kind == 'error':
                    if msg.get('tb'):
                        self._qlog(msg['tb'])
                    raise RuntimeError(msg['msg'])
        except ConnectionError as e:
            if self.cancel.is_set():
                raise ec.Cancelled()
            raise ConnectionError(f'Lost connection to the eval server: {e}') from e
        finally:
            self._sock = None
            sock.close()

    def _send_cancel(self):
        sock = self._sock
        if sock is not None:
            try:
                ec.send_msg(sock, {'cmd': 'cancel'})
            except OSError:
                pass

    def _cancel_job(self):
        self.cancel.set()
        self._send_cancel()
        self._log('Cancel requested…')

    def _post_filter(self, msg):
        fr = ec.FilterResult.from_json(msg['filter'], from_cache=msg['from_cache'])
        self.q.put(('filter_done', fr))
        return fr

    # -- connection test --
    def _ping(self):
        try:
            st = self.settings()
            ec.parse_path_map(st.path_map)
        except ValueError as e:
            messagebox.showerror('Invalid settings', str(e))
            return
        self._start_job(lambda: self.q.put(('ping_done', self._server_job('ping', st)['env'])))

    def _on_ping(self, env):
        missing = [k for k, v in env.items() if v == 'MISSING']
        dev = env.get('gpu', 'CPU only') if env.get('cuda') else 'CPU only'
        txt = f'●  connected — torch {env.get("torch")}, {dev}'
        if missing:
            txt += f'   missing in container: {", ".join(missing)}'
        self.srv_status.config(text=txt, fg=WARN if missing or not env.get('cuda') else GOOD)
        self._log('[Server] ' + ', '.join(f'{k}={v}' for k, v in env.items()))

    # -- filter --
    def _run_filter(self):
        st = self._get_settings_or_warn(need_checkpoints=False)
        if st:
            self._start_job(lambda: self._post_filter(self._server_job('filter', st)))

    def _on_filter_done(self, fr):
        self.filter_result = fr
        rc = fr.reason_counts
        lines = [f'Total samples : {fr.total}   (mask pairing: {fr.pairing})',
                 f'Accepted      : {len(fr.accepted)}',
                 f'Rejected      : {len(fr.rejected)}']
        lines += [f'   {r:<20}: {rc.get(r, 0)}' for r in ec.REJECT_REASONS]
        lines.append(f'{"(from cache " + fr.key + ")" if fr.from_cache else "cache key " + fr.key}')
        lines.append('rejected_samples.csv written to the results folder')
        self.filter_stats.config(text='\n'.join(lines), fg=FG)
        self._regroup()

    # -- grouping (cheap, recomputed live) --
    def _schedule_regroup(self):
        if not self._regroup_pending:
            self._regroup_pending = True
            self.root.after(200, self._regroup)

    def _regroup(self):
        self._regroup_pending = False
        if self.filter_result is None:
            return
        try:
            st = self.settings()
            gi = ec.MaskGrouper(st).assign(self.filter_result.accepted)
        except (ValueError, tk.TclError) as e:
            self.group_stats.config(text=f'Invalid groups: {e}', fg=BAD)
            self.group_info = None
            return
        self.group_info = gi
        acc = self.filter_result.accepted
        lines = []
        for g in GROUPS + (OUT_OF_RANGE, ALL):
            stt = gi.ratio_stats(g, acc)
            b = gi.bounds.get(g)
            rng = f'{b[0] * 100:5.1f}–{b[1] * 100:5.1f}%' if b and b[0] == b[0] else ' ' * 12
            mean = f'mean {stt["mean"] * 100:5.1f}%' if stt['mean'] == stt['mean'] else ''
            flag = '  ⚠ <' + str(st.fid_min_group_size) if (g in GROUPS and gi.count(g) < st.fid_min_group_size) else ''
            lines.append(f'{g:<13}{rng}  n={gi.count(g):<6}{mean}{flag}')
        if gi.mode == 'tertiles':
            lines.append('Auto boundaries shown are the min/max ratio inside each tertile.')
        lines.append('"All" = every accepted sample (includes Out of range).')
        self.group_stats.config(text='\n'.join(lines), fg=WARN if gi.warnings else FG)
        self._populate_samples()

    # -- evaluation --
    def _run_eval(self):
        st = self._get_settings_or_warn(need_checkpoints=True)
        if not st:
            return
        self._start_job(self._eval_worker, st)

    def _eval_worker(self, st):
        msg = self._server_job('evaluate', st)
        self._post_filter(msg)
        res = ec.EvalResults(ckpt_names=msg['ckpt_names'], ckpt_paths=msg['ckpt_paths'],
                             per_image={}, summary=msg['summary'], grid_images={},
                             notes=msg['notes'], device=msg['device'],
                             ckpt_networks=msg.get('ckpt_networks', []))
        run_dir = ec.map_path(msg['run_dir'], ec.parse_path_map(st.path_map), reverse=True)
        self.q.put(('eval_done', res, run_dir))

    def _on_eval_done(self, res, run_dir):
        self.results = res
        self.run_dir = run_dir
        self.res_info.config(text=f'saved: {run_dir}')
        self._render_table()
        self.nb.select(2)

    # -- preview --
    def _run_preview(self):
        st = self._get_settings_or_warn(need_checkpoints=True)
        if st:
            self._start_job(self._preview_worker, st)

    def _preview_worker(self, st):
        msg = self._server_job('preview', st)
        self._post_filter(msg)
        grids = {g: Image.open(io.BytesIO(png)) for g, png in msg['grids'].items()}
        self.q.put(('preview_done', grids))

    def _show_preview(self, grids):
        win = tk.Toplevel(self.root)
        win.title('Grid preview')
        win.configure(bg=BG)
        win.geometry('1400x850')
        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True)
        self._preview_refs = []
        for g, img in grids.items():
            fr = ttk.Frame(nb)
            nb.add(fr, text=f'{g} ({len(self.group_info.members[g]) if self.group_info else "?"})')
            cv = tk.Canvas(fr, bg='white', highlightthickness=0)
            ys = ttk.Scrollbar(fr, orient=tk.VERTICAL, command=cv.yview)
            xs = ttk.Scrollbar(fr, orient=tk.HORIZONTAL, command=cv.xview)
            cv.configure(yscrollcommand=ys.set, xscrollcommand=xs.set,
                         scrollregion=(0, 0, img.width, img.height))
            ys.pack(side=tk.RIGHT, fill=tk.Y)
            xs.pack(side=tk.BOTTOM, fill=tk.X)
            cv.pack(fill=tk.BOTH, expand=True)
            tkimg = ImageTk.PhotoImage(img)
            self._preview_refs.append(tkimg)
            cv.create_image(0, 0, image=tkimg, anchor='nw')

    # -----------------------------------------------------------------------
    # Fixed sample lists
    # -----------------------------------------------------------------------
    def _to_local(self):
        """Translate container paths (from the server's filter result) back to host paths."""
        pairs = ec.parse_path_map(self.vars['path_map'].get())
        return lambda p: ec.map_path(p, pairs, reverse=True) if p else p

    def _update_manifest_status(self):
        p = self.vars['manifest_path'].get()
        if not p:
            self.manifest_status.config(
                text='Not used: samples are discovered from the folders and filtered by the thresholds.',
                fg=MUTED)
            return
        try:
            n = len(ec.read_manifest(p)['samples'])
            txt, fg = f'USING SAMPLE LIST: {p}\n  {n} samples — thresholds are NOT re-applied ' \
                      f'(only missing files / empty holes are dropped).', GOOD
        except Exception as e:
            txt, fg = f'Sample list {p}: cannot read ({e})', BAD
        self.manifest_status.config(text=txt, fg=fg)

    def _need_filter_result(self):
        if self.filter_result is None or not self.filter_result.accepted:
            messagebox.showinfo('Sample list', 'Run the filter first (tab 2 → "Run filter").')
            return None
        try:
            st = self.settings()
        except ValueError as e:
            messagebox.showerror('Invalid settings', str(e))
            return None
        if self.filter_result.settings != ec.settings_for_server(st).filter_settings():
            if not messagebox.askyesno('Sample list', 'Settings changed since the last filter run.\n'
                                                      'Save the samples of the LAST filter run anyway?'):
                return None
        return st

    def _save_sample_list(self):
        st = self._need_filter_result()
        if st is None:
            return
        p = filedialog.asksaveasfilename(title='Save sample list', defaultextension='.json',
                                         initialfile='samples.json',
                                         initialdir=st.results_dir or None,
                                         filetypes=[('Sample list', '*.json')])
        if not p:
            return
        j, c = ec.save_sample_list(p, st, self.filter_result, self.group_info, to_local=self._to_local())
        self._log(f'Sample list saved: {j}  (+ {os.path.basename(c)}), '
                  f'{len(self.filter_result.accepted)} samples')

    def _export_test_set(self):
        st = self._need_filter_result()
        if st is None:
            return
        d = filedialog.askdirectory(title='Empty folder for the fixed test set')
        if not d:
            return
        if os.listdir(d) and not messagebox.askyesno('Export', f'{d} is not empty. Write into it anyway?'):
            return
        fr, gi, to_local = self.filter_result, self.group_info, self._to_local()

        def worker():
            j = ec.export_fixed_test_set(d, st, fr, gi, to_local=to_local,
                                         progress=self._qprog, cancel=self.cancel)
            self._qlog(f'Fixed test set exported: {d}  ({len(fr.accepted)} samples)')
            self.q.put(('export_done', j))
        self._start_job(worker)

    def _on_export_done(self, json_path):
        if messagebox.askyesno('Export done', f'Written:\n{os.path.dirname(json_path)}\n\n'
                                              'Use this exported set for evaluation now?'):
            # after the job's 'finished' message has been handled (busy cleared)
            self.root.after(300, lambda: self._load_sample_list(json_path))

    def _load_sample_list(self, path=None):
        path = path or filedialog.askopenfilename(title='Load sample list',
                                                  filetypes=[('Sample list', '*.json *.csv'),
                                                             ('All', '*')])
        if not path:
            return
        try:
            m = ec.read_manifest(path)
        except Exception as e:
            messagebox.showerror('Load sample list', str(e))
            return
        restored = []
        for k, v in (m.get('settings') or {}).items():          # reproduce the same crops & groups
            if k in self.vars:
                self.vars[k].set(v)
                restored.append(k)
        pairs = ec.parse_path_map(self.vars['path_map'].get())
        warn = []
        for k in ('image', 'edge', 'mask'):
            d = m.get(f'{k}_dir') or ''
            if d and not os.path.isdir(d):                       # list written on the server side
                d = ec.map_path(d, pairs, reverse=True)
            if d and os.path.isdir(d):
                self.vars[f'{k}_dir'].set(d)
            elif d:
                warn.append(f'{k} folder not found: {m.get(k + "_dir")} (kept the current one)')
        self.vars['manifest_path'].set(os.path.abspath(path))
        self.filter_result = None
        self._log(f'Sample list loaded: {path}  ({len(m["samples"])} samples; '
                  f'restored {len(restored)} settings)')
        for w in warn:
            self._log('  WARNING: ' + w)
        if not self.busy:
            self._run_filter()

    def _clear_sample_list(self):
        self.vars['manifest_path'].set('')
        self.filter_result = None
        self._log('Sample list cleared: samples come from the folders again (run the filter).')

    # -----------------------------------------------------------------------
    # Sample viewer
    # -----------------------------------------------------------------------
    def _populate_samples(self):
        tree = self.smp_tree
        tree.delete(*tree.get_children())
        fr = self.filter_result
        if fr is None:
            return
        labels = self.group_info.labels if self.group_info else ['?'] * len(fr.accepted)
        g_sel, q = self.smp_group.get(), self.smp_search.get().strip().lower()
        rows = []
        for i, a in enumerate(fr.accepted):
            if g_sel != ALL and labels[i] != g_sel:
                continue
            if q and q not in a['id'].lower():
                continue
            rows.append((i, a['id'], labels[i], a['mask_ratio'] * 100, a['edge_pixels_in_hole']))
        col, rev = self._smp_sort
        key = {'id': 1, 'group': 2, 'ratio': 3, 'edges': 4}[col]
        rows.sort(key=lambda r: r[key], reverse=rev)
        for i, sid, g, r, e in rows:
            tree.insert('', tk.END, iid=str(i), values=(sid, g, f'{r:.1f}', e))
        self.smp_count.config(text=f'{len(rows)} of {len(fr.accepted)} accepted samples shown')

    def _sort_samples(self, col):
        prev, rev = self._smp_sort
        self._smp_sort = (col, not rev if prev == col else False)
        self._populate_samples()

    def _selected_sample_index(self):
        sel = self.smp_tree.selection()
        return int(sel[0]) if sel else None

    def _on_sample_select(self):
        if self.smp_auto.get():
            self._run_sample()

    def _run_sample(self):
        idx = self._selected_sample_index()
        if idx is None or self.filter_result is None:
            return
        if self.busy:
            self._smp_pending = idx            # run when the current job finishes
            return
        st = self._get_settings_or_warn(need_checkpoints=True)
        if not st:
            return
        smp = self.filter_result.accepted[idx]
        label = self.group_info.labels[idx] if self.group_info else '?'
        self.smp_info.config(text=f'Running {smp["id"]} on {len(st.checkpoints)} checkpoint(s)…', fg=MUTED)
        self._start_job(self._sample_worker, st, smp, label)

    def _sample_worker(self, st, smp, label):
        msg = self._server_job('sample', st, sample=smp)
        dec = lambda b: Image.open(io.BytesIO(b)).convert('RGB')
        data = {'sample': smp, 'group': label,
                'input': dec(msg['input']), 'masked': dec(msg['masked']), 'gt': dec(msg['gt']),
                'preds': []}
        for p in msg['preds']:
            p = dict(p)
            for k in ('comp', 'raw'):
                if k in p:
                    p[k] = dec(p[k])
            data['preds'].append(p)
        self.q.put(('sample_done', data))

    def _on_sample_done(self, data):
        self._smp_data = data
        self._render_sample()

    def _tile_photo(self, pil, tile):
        import numpy as np
        arr = ec._to_tile(np.asarray(pil), tile)
        ph = ImageTk.PhotoImage(Image.fromarray(arr))
        self._smp_refs.append(ph)
        return ph

    def _render_sample(self):
        for w in self.smp_frame.winfo_children():
            w.destroy()
        self._smp_refs = []
        data = self._smp_data
        if data is None:
            return
        smp, tile, ncol = data['sample'], self.smp_tile.get(), self.smp_cols.get()
        view = self.smp_view.get()
        self.smp_info.config(fg=FG, text=(
            f'{smp["id"]}   group: {data["group"]}   mask: {smp["mask_ratio"] * 100:.1f}%   '
            f'edge px in hole: {smp["edge_pixels_in_hole"]}   density: {smp["edge_density"]:.4f}\n'
            f'image: {smp["image_name"]}   mask: {smp["mask_name"]}'))

        shown = [k for k, _, _ in METRICS if k != 'fid']
        ok = [p for p in data['preds'] if 'metrics' in p]
        best = {k: ec.best_index([p['metrics'][k] for p in ok], k) for k in shown}

        cells = [('Input (hole red)', None, data['input'], None),
                 ('Masked edges', None, data['masked'], None),
                 ('Ground truth', None, data['gt'], None)]
        for p in data['preds']:
            cells.append((p['name'], p, p.get(view), ok.index(p) if p in ok else None))

        for n, (title, p, img, ok_idx) in enumerate(cells):
            box = tk.Frame(self.smp_frame, bg=PANEL, padx=4, pady=4)
            box.grid(row=n // ncol, column=n % ncol, sticky='n', padx=4, pady=4)
            tk.Label(box, text=title, bg=PANEL, fg=ACCENT, font=('Consolas', 9, 'bold'),
                     wraplength=tile, justify='left').pack(anchor='w')
            if p is not None:
                net = p['network'].replace('networks.', '')
                stem = os.path.splitext(os.path.basename(p['path']))[0]
                tk.Label(box, text=f'{stem} · {net}  ({p["source"]})', bg=PANEL, fg=MUTED,
                         font=('Consolas', 7), wraplength=tile, justify='left').pack(anchor='w')
            if img is not None:
                tk.Label(box, image=self._tile_photo(img, tile), bg=PANEL, bd=0).pack()
            if p is not None and 'error' in p:
                tk.Label(box, text=p['error'], bg=PANEL, fg=BAD, font=('Consolas', 8),
                         wraplength=tile, justify='left').pack(anchor='w')
            if p is not None and 'warning' in p:
                tk.Label(box, text=p['warning'], bg=PANEL, fg=WARN, font=('Consolas', 8),
                         wraplength=tile, justify='left').pack(anchor='w')
            if p is not None and 'metrics' in p:
                mf = tk.Frame(box, bg=PANEL)
                mf.pack(fill=tk.X, pady=(2, 0))
                for j, k in enumerate(shown):
                    v = p['metrics'][k]
                    txt = '—' if v != v else (f'{v:.2f}' if k in ('psnr', 'hd95') else f'{v:.4f}')
                    is_best = len(ok) > 1 and best[k] == ok_idx
                    tk.Label(mf, text=f'{ec.METRIC_LABEL[k]}{"↑" if ec.HIGHER_BETTER[k] else "↓"} {txt}',
                             bg=BEST_BG if is_best else PANEL, fg=GOOD if is_best else FG,
                             font=('Consolas', 8, 'bold' if is_best else 'normal')).grid(
                        row=j // 2, column=j % 2, sticky='w', padx=(0, 6))

    # -----------------------------------------------------------------------
    # Results table (per-cell best highlighting)
    # -----------------------------------------------------------------------
    def _render_table(self):
        for w in self.table.winfo_children():
            w.destroy()
        res = self.results
        if res is None:
            tk.Label(self.table, text='No results yet.', bg=BG, fg=MUTED).grid(row=0, column=0)
            return
        groups = list(SUMMARY_GROUPS) if self.res_side.get() else [self.res_group.get()]
        metric_keys = [k for k, _, _ in METRICS]
        cols = [(g, k) for k in metric_keys for g in groups] if len(groups) > 1 else \
               [(groups[0], k) for k in metric_keys]

        cell = dict(font=('Consolas', 9), padx=8, pady=3, bd=0, anchor='e')
        tk.Label(self.table, text='checkpoint', bg=PANEL, fg=ACCENT, **dict(cell, anchor='w')).grid(
            row=0, column=0, sticky='nsew', padx=1, pady=1)
        n_txt = '  '.join(f'{g}:{res.summary[g][res.ckpt_names[0]]["n"]}' for g in groups)
        tk.Label(self.table, text=f'n  ({n_txt})', bg=BG, fg=MUTED,
                 font=('Consolas', 8)).grid(row=len(res.ckpt_names) + 1, column=0, columnspan=6,
                                            sticky='w', pady=4)
        for j, (g, k) in enumerate(cols, start=1):
            arrow = '↑' if ec.HIGHER_BETTER[k] else '↓'
            head = f'{ec.METRIC_LABEL[k]} {arrow}' + (f'\n{g}' if len(groups) > 1 else '')
            tk.Label(self.table, text=head, bg=PANEL, fg=ACCENT, **cell).grid(
                row=0, column=j, sticky='nsew', padx=1, pady=1)

        for j, (g, k) in enumerate(cols, start=1):
            vals = [res.summary[g][n][k] for n in res.ckpt_names]
            best = ec.best_index(vals, k)
            small = 0 < res.summary[g][res.ckpt_names[0]]['n'] < self.vars['fid_min_group_size'].get()
            for i, v in enumerate(vals):
                txt = '—' if v is None or v != v else (f'{v:.2f}' if k in ('psnr', 'fid', 'hd95') else f'{v:.4f}')
                if k == 'fid' and small and txt != '—':
                    txt += ' ⚠'
                is_best = (i == best)
                tk.Label(self.table, text=txt, bg=BEST_BG if is_best else BG,
                         fg=GOOD if is_best else FG,
                         **dict(cell, font=('Consolas', 9, 'bold' if is_best else 'normal'))).grid(
                    row=i + 1, column=j, sticky='nsew', padx=1, pady=1)
        nets = res.ckpt_networks or [''] * len(res.ckpt_names)
        for i, (n, path, net) in enumerate(zip(res.ckpt_names, res.ckpt_paths, nets)):
            stem = os.path.splitext(os.path.basename(path))[0]
            txt = f'{n}\n  {stem}' + (f' · {net.replace("networks.", "")}' if net else '')
            tk.Label(self.table, text=txt, bg=BG, fg=FG, justify='left',
                     **dict(cell, anchor='w')).grid(row=i + 1, column=0, sticky='nsew', padx=1, pady=1)
        if res.notes:
            tk.Label(self.table, text='\n'.join(res.notes), bg=BG, fg=WARN, justify='left',
                     font=('Consolas', 8)).grid(row=len(res.ckpt_names) + 2, column=0,
                                                columnspan=12, sticky='w')


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Checkpoint evaluation GUI (server runs in the container)')
    ap.add_argument('--settings', help='Settings JSON (saved from the GUI)')
    ap.add_argument('--server', help='Eval server address host:port (overrides the settings file)')
    ap.add_argument('--path-map', help="'HOST_PREFIX=CONTAINER_PREFIX[;...]' (overrides the settings file; "
                                       "'' = GUI and server see the same paths)")
    args = ap.parse_args()

    st = EvalSettings.load_json(args.settings) if args.settings else EvalSettings()
    if args.server:
        st.server_addr = args.server
    if args.path_map is not None:
        st.path_map = args.path_map

    root = tk.Tk()
    root.geometry('1350x900')
    root.minsize(1000, 700)
    EvalGUI(root, st)
    root.mainloop()


if __name__ == '__main__':
    main()
