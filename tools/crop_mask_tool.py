# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
"""Interactive GUI tool for selecting 256x256 crops and drawing masks, 
now robustly saving the corresponding cropped edge maps.
"""
import os
import sys
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CROP_SIZE = 256
MAX_PREVIEW_W = 800   # Max width for the full-image preview canvas
MAX_PREVIEW_H = 600   # Max height for the full-image preview canvas
BRUSH_MIN = 2
BRUSH_MAX = 60
BRUSH_DEFAULT = 16
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class CropMaskTool:
    def __init__(self, root, input_dir=None, edge_dir=None, crop_dir=None, mask_dir=None, edge_out_dir=None, crop_size=CROP_SIZE):
        self.root = root
        self.root.title("Crop, Edge & Mask Tool")
        self.crop_size = crop_size

        # State
        self.input_dir = input_dir
        self.edge_dir = edge_dir
        self.crop_dir = crop_dir
        self.mask_dir = mask_dir
        self.edge_out_dir = edge_out_dir
        
        self.image_files = []
        self.current_idx = 0

        # Current image data
        self.pil_full = None        # Full-resolution PIL image
        self.pil_edge_full = None   # Full-resolution corresponding edge map
        self.display_scale = 1.0    # Scale factor for preview
        self.crop_x = 0             # Crop top-left in full-res coords
        self.crop_y = 0
        self.pil_crop = None        # 256x256 crop (PIL)
        self.pil_edge_crop = None   # 256x256 edge crop (PIL)
        self.mask_array = None      # numpy uint8 (H, W), 0=valid, 255=hole

        # Brush
        self.brush_size = BRUSH_DEFAULT
        self.drawing = False
        self.erasing = False

        # ---- Build UI ----
        self._build_menu()
        self._build_toolbar()
        self._build_canvases()
        self._build_statusbar()

        # Key bindings
        self.root.bind('<Left>', lambda e: self._prev_image())
        self.root.bind('<Right>', lambda e: self._next_image())
        self.root.bind('<Return>', lambda e: self._save_current())
        self.root.bind('<space>', lambda e: self._save_current())
        self.root.bind('<Escape>', lambda e: self._clear_mask())

        # Auto-load if dirs given
        if self.input_dir:
            self._scan_input_dir()
            if self.image_files:
                self._load_image(0)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Input Folder…", command=self._pick_input_dir)
        file_menu.add_command(label="Open Edges Folder…", command=self._pick_edge_dir)
        file_menu.add_separator()
        file_menu.add_command(label="Set Crop Output Folder…", command=self._pick_crop_dir)
        file_menu.add_command(label="Set Edge Output Folder…", command=self._pick_edge_out_dir)
        file_menu.add_command(label="Set Mask Output Folder…", command=self._pick_mask_dir)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def _build_toolbar(self):
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)

        # Navigation
        tk.Button(toolbar, text="◀ Prev", command=self._prev_image, width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Next ▶", command=self._next_image, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Output configuration button added directly to toolbar
        tk.Button(toolbar, text="📁 Set Edge Output", command=self._pick_edge_out_dir, bg='#f0f0f0').pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Brush size
        tk.Label(toolbar, text="Brush:").pack(side=tk.LEFT, padx=(4, 0))
        self.brush_var = tk.IntVar(value=self.brush_size)
        self.brush_scale = tk.Scale(
            toolbar, from_=BRUSH_MIN, to=BRUSH_MAX,
            orient=tk.HORIZONTAL, variable=self.brush_var,
            length=140, showvalue=True,
            command=lambda v: setattr(self, 'brush_size', int(v))
        )
        self.brush_scale.pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Actions
        tk.Button(toolbar, text="Clear Mask", command=self._clear_mask, width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Fill Mask", command=self._fill_mask, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.save_btn = tk.Button(toolbar, text="💾 Save & Next", command=self._save_current,
                                   width=14, bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'))
        self.save_btn.pack(side=tk.LEFT, padx=4)

        # Image counter label
        self.counter_label = tk.Label(toolbar, text="0 / 0", font=('Arial', 10))
        self.counter_label.pack(side=tk.RIGHT, padx=8)

    def _build_canvases(self):
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=6)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # --- Left: full image preview ---
        left_frame = tk.LabelFrame(pane, text="Source Image (click to select crop)")
        pane.add(left_frame, minsize=300)

        self.preview_canvas = tk.Canvas(left_frame, bg='#2b2b2b', cursor='crosshair')
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind('<Button-1>', self._on_preview_click)
        self.preview_canvas.bind('<B1-Motion>', self._on_preview_click)
        self.preview_canvas.bind('<Configure>', self._on_preview_resize)

        # --- Right: crop + mask ---
        right_frame = tk.Frame(pane)
        pane.add(right_frame, minsize=self.crop_size + 20)

        # Crop display
        crop_label_frame = tk.LabelFrame(right_frame, text="Crop Preview")
        crop_label_frame.pack(side=tk.TOP, padx=4, pady=(4, 2))
        self.crop_canvas = tk.Canvas(crop_label_frame, width=self.crop_size, height=self.crop_size, bg='#1e1e1e')
        self.crop_canvas.pack()

        # Mask drawing
        mask_label_frame = tk.LabelFrame(right_frame, text="Mask (LMB=paint hole, RMB=erase)")
        mask_label_frame.pack(side=tk.TOP, padx=4, pady=(2, 4))
        self.mask_canvas = tk.Canvas(mask_label_frame, width=self.crop_size, height=self.crop_size, bg='black',
                                     cursor='circle')
        self.mask_canvas.pack()

        # Mask canvas events
        self.mask_canvas.bind('<Button-1>', self._mask_start_draw)
        self.mask_canvas.bind('<B1-Motion>', self._mask_draw)
        self.mask_canvas.bind('<ButtonRelease-1>', self._mask_stop_draw)
        self.mask_canvas.bind('<Button-3>', self._mask_start_erase)
        self.mask_canvas.bind('<B3-Motion>', self._mask_erase)
        self.mask_canvas.bind('<ButtonRelease-3>', self._mask_stop_erase)

        # Info label
        self.info_label = tk.Label(right_frame, text="", font=('Consolas', 9), anchor='w', justify='left')
        self.info_label.pack(fill=tk.X, padx=6, pady=4)

    def _build_statusbar(self):
        self.statusbar = tk.Label(self.root, text="Ready. Open an input folder to begin.",
                                   bd=1, relief=tk.SUNKEN, anchor='w', font=('Arial', 9))
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    # -----------------------------------------------------------------------
    # Directory selection
    # -----------------------------------------------------------------------
    def _pick_input_dir(self):
        d = filedialog.askdirectory(title="Select Input Image Folder")
        if d:
            self.input_dir = d
            self._scan_input_dir()
            if self.image_files:
                self._load_image(0)

    def _pick_edge_dir(self):
        d = filedialog.askdirectory(title="Select Corresponding Edges Folder")
        if d:
            self.edge_dir = d
            self._status(f"Edges folder set: {self.edge_dir}")
            if self.pil_full is not None:
                self._load_image(self.current_idx)

    def _pick_crop_dir(self):
        d = filedialog.askdirectory(title="Select Crop Output Folder")
        if d:
            self.crop_dir = d
            os.makedirs(self.crop_dir, exist_ok=True)
            self._status(f"Crop output: {self.crop_dir}")

    def _pick_edge_out_dir(self):
        d = filedialog.askdirectory(title="Select Edge Output Folder")
        if d:
            self.edge_out_dir = d
            os.makedirs(self.edge_out_dir, exist_ok=True)
            self._status(f"Edge output folder set: {self.edge_out_dir}")
            self._update_info()

    def _pick_mask_dir(self):
        d = filedialog.askdirectory(title="Select Mask Output Folder")
        if d:
            self.mask_dir = d
            os.makedirs(self.mask_dir, exist_ok=True)
            self._status(f"Mask output: {self.mask_dir}")

    def _scan_input_dir(self):
        self.image_files = sorted([
            f for f in os.listdir(self.input_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])
        self.current_idx = 0
        self._status(f"Found {len(self.image_files)} images in {self.input_dir}")
        self._update_counter()

    # -----------------------------------------------------------------------
    # Image loading
    # -----------------------------------------------------------------------
    def _load_image(self, idx):
        if not self.image_files:
            return

        idx = max(0, min(idx, len(self.image_files) - 1))
        self.current_idx = idx
        fname = self.image_files[idx]
        fpath = os.path.join(self.input_dir, fname)

        try:
            self.pil_full = Image.open(fpath).convert('RGB')
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open {fname}:\n{e}")
            return

        # Extension-agnostic matching for the edge map
        self.pil_edge_full = None
        if self.edge_dir:
            stem = os.path.splitext(fname)[0]
            edge_file_found = None
            
            for f in os.listdir(self.edge_dir):
                if os.path.splitext(f)[0] == stem and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    edge_file_found = f
                    break
            
            if edge_file_found:
                try:
                    self.pil_edge_full = Image.open(os.path.join(self.edge_dir, edge_file_found))
                except Exception as e:
                    self._status(f"Warning: Failed to read edge map: {edge_file_found}")
            else:
                self._status(f"Warning: No edge file named '{stem}.*' found in {self.edge_dir}")

        self._update_counter()
        self._display_preview()
        self._reset_crop_and_mask()
        
        edge_status = " (Edge Bound)" if self.pil_edge_full else " (NO Edge Map Found)"
        self._status(f"Loaded: {fname}  ({self.pil_full.width}×{self.pil_full.height}){edge_status}")

        # Auto-place crop at center
        cx = max(0, (self.pil_full.width - self.crop_size) // 2)
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
            cw, ch = MAX_PREVIEW_W, MAX_PREVIEW_H

        sx = cw / self.pil_full.width
        sy = ch / self.pil_full.height
        self.display_scale = min(sx, sy, 1.0)
        dw = int(self.pil_full.width * self.display_scale)
        dh = int(self.pil_full.height * self.display_scale)

        resized = self.pil_full.resize((dw, dh), Image.LANCZOS)
        self._preview_tk = ImageTk.PhotoImage(resized)
        self.preview_canvas.delete('all')
        self.preview_canvas.create_image(cw // 2, ch // 2, image=self._preview_tk, anchor='center')

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

        cx = int(fx - self.crop_size / 2)
        cy = int(fy - self.crop_size / 2)

        cx = max(0, min(cx, self.pil_full.width - self.crop_size))
        cy = max(0, min(cy, self.pil_full.height - self.crop_size))
        self._set_crop(cx, cy)

    def _set_crop(self, cx, cy):
        self.crop_x = cx
        self.crop_y = cy
        
        self.pil_crop = self.pil_full.crop((cx, cy, cx + self.crop_size, cy + self.crop_size))
        
        if self.pil_edge_full is not None:
            self.pil_edge_crop = self.pil_edge_full.crop((cx, cy, cx + self.crop_size, cy + self.crop_size))
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
            x1, y1, x2, y2,
            outline='#00FF00', width=2, dash=(4, 4), tags='crop_rect'
        )

    def _reset_crop_and_mask(self):
        self.pil_crop = None
        self.pil_edge_crop = None
        self.mask_array = None
        self.crop_canvas.delete('all')
        self.mask_canvas.delete('all')

    # -----------------------------------------------------------------------
    # Crop display
    # -----------------------------------------------------------------------
    def _display_crop(self):
        if self.pil_crop is None:
            return
        self._crop_tk = ImageTk.PhotoImage(self.pil_crop)
        self.crop_canvas.delete('all')
        self.crop_canvas.create_image(0, 0, image=self._crop_tk, anchor='nw')

    # -----------------------------------------------------------------------
    # Mask drawing
    # -----------------------------------------------------------------------
    def _display_mask(self):
        if self.mask_array is None or self.pil_crop is None:
            return
        crop_arr = np.array(self.pil_crop).copy()
        mask_bool = self.mask_array > 0
        overlay = crop_arr.copy()
        overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0].astype(int) + 100, 0, 255).astype(np.uint8)
        overlay[mask_bool, 1] = (overlay[mask_bool, 1] * 0.4).astype(np.uint8)
        overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.4).astype(np.uint8)
        self._mask_tk = ImageTk.PhotoImage(Image.fromarray(overlay))
        self.mask_canvas.delete('all')
        self.mask_canvas.create_image(0, 0, image=self._mask_tk, anchor='nw')

    def _paint_mask(self, x, y, value):
        if self.mask_array is None:
            return
        r = self.brush_size // 2
        y_coords, x_coords = np.ogrid[-r:r+1, -r:r+1]
        circle = x_coords**2 + y_coords**2 <= r**2
        y1, y2 = max(0, y - r), min(self.crop_size, y + r + 1)
        x1, x2 = max(0, x - r), min(self.crop_size, x + r + 1)
        cy1 = max(0, r - y)
        cy2 = cy1 + (y2 - y1)
        cx1 = max(0, r - x)
        cx2 = cx1 + (x2 - x1)
        region = circle[cy1:cy2, cx1:cx2]
        if value > 0:
            self.mask_array[y1:y2, x1:x2][region] = 255
        else:
            self.mask_array[y1:y2, x1:x2][region] = 0

    def _mask_start_draw(self, event):
        self.drawing = True
        self._paint_mask(event.x, event.y, 255)
        self._display_mask()

    def _mask_draw(self, event):
        if self.drawing:
            self._paint_mask(event.x, event.y, 255)
            self._display_mask()

    def _mask_stop_draw(self, event):
        self.drawing = False
        self._update_info()

    def _mask_start_erase(self, event):
        self.erasing = True
        self._paint_mask(event.x, event.y, 0)
        self._display_mask()

    def _mask_erase(self, event):
        if self.erasing:
            self._paint_mask(event.x, event.y, 0)
            self._display_mask()

    def _mask_stop_erase(self, event):
        self.erasing = False
        self._update_info()

    def _clear_mask(self):
        if self.mask_array is not None:
            self.mask_array[:] = 0
            self._display_mask()
            self._update_info()

    def _fill_mask(self):
        if self.mask_array is not None:
            self.mask_array[:] = 255
            self._display_mask()
            self._update_info()

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    def _save_current(self):
        if self.pil_crop is None or self.mask_array is None:
            messagebox.showwarning("Nothing to save", "Select a crop first.")
            return

        # Ensure output dirs exist
        if not self.crop_dir:
            self._pick_crop_dir()
            if not self.crop_dir: return
        if not self.mask_dir:
            self._pick_mask_dir()
            if not self.mask_dir: return
        if self.pil_edge_crop is not None and not self.edge_out_dir:
            self._pick_edge_out_dir()
            if not self.edge_out_dir: return

        # Filename configurations
        src_name = self.image_files[self.current_idx]
        stem = os.path.splitext(src_name)[0]
        crop_fname = f"{stem}.png"
        mask_fname = f"{stem}.png"

        # Save source crop
        crop_gray = self.pil_crop.convert('L')
        crop_path = os.path.join(self.crop_dir, crop_fname)
        crop_gray.save(crop_path)

        # Save edge crop to your explicitly defined edge output folder
        if self.pil_edge_crop is not None and self.edge_out_dir:
            edge_gray = self.pil_edge_crop.convert('L')
            edge_path = os.path.join(self.edge_out_dir, crop_fname)
            edge_gray.save(edge_path)
            edge_msg = " + edge"
        else:
            edge_msg = " (NO EDGE SAVED)"

        # Save mask
        mask_img = Image.fromarray(self.mask_array, mode='L')
        mask_path = os.path.join(self.mask_dir, mask_fname)
        mask_img.save(mask_path)

        hole_pct = np.mean(self.mask_array > 0) * 100
        self._status(f"Saved: {crop_fname}{edge_msg} (hole: {hole_pct:.1f}%)")

        # Auto-advance
        if self.current_idx < len(self.image_files) - 1:
            self._load_image(self.current_idx + 1)
        else:
            messagebox.showinfo("Done", "All images have been processed!")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _update_counter(self):
        total = len(self.image_files)
        cur = self.current_idx + 1 if total > 0 else 0
        self.counter_label.config(text=f"{cur} / {total}")

    def _update_info(self):
        if self.mask_array is None:
            self.info_label.config(text="")
            return
        hole_pct = np.mean(self.mask_array > 0) * 100
        fname = self.image_files[self.current_idx] if self.image_files else ""
        edge_loaded = "Yes" if self.pil_edge_full is not None else "No"
        edge_out_set = self.edge_out_dir if self.edge_out_dir else "Not Set"
        self.info_label.config(text=(
            f"File: {fname}\n"
            f"Edge Map Loaded: {edge_loaded}\n"
            f"Edge Save Dest: {edge_out_set}\n"
            f"Crop: ({self.crop_x}, {self.crop_y}) → "
            f"({self.crop_x + self.crop_size}, {self.crop_y + self.crop_size})\n"
            f"Mask hole: {hole_pct:.1f}%\n"
            f"Convention: white=Hole, black=Valid"
        ))

    def _status(self, msg):
        self.statusbar.config(text=msg)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Crop, Edge & Mask Tool for TSR datasets")
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Input folder containing source images')
    parser.add_argument('--edge-dir', type=str, default=None,
                        help='Input folder containing matching full-size edge maps')
    parser.add_argument('--crop-dir', type=str, default=None,
                        help='Output folder for saved crops')
    parser.add_argument('--edge-out-dir', type=str, default=None,
                        help='Output folder for saved edge crops')
    parser.add_argument('--mask-dir', type=str, default=None,
                        help='Output folder for saved masks')
    parser.add_argument('--size', type=int, default=CROP_SIZE,
                        help=f'Crop size (default: {CROP_SIZE})')
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("1200x700")
    root.minsize(800, 500)
    app = CropMaskTool(
        root,
        input_dir=args.input_dir,
        edge_dir=args.edge_dir,
        crop_dir=args.crop_dir,
        mask_dir=args.mask_dir,
        edge_out_dir=args.edge_out_dir,
        crop_size=args.size,
    )
    root.mainloop()

if __name__ == '__main__':
    main()