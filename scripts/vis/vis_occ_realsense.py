#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize RealSense open-world occupancy npz (PRED only, NO GT required) with Mayavi.

This script is RealSense-oriented and works with the sparse npz dumped by:
  scripts/src/eval_occ_realsense.py  (keys: vox_coords, vox_labels, origin, voxel_size)

Label convention:
- 0: empty/free (not rendered)
- 1..num_sem: semantic classes

Features:
- Visualize either a single npz or an incremental sequence folder (--npz_seq_dir)
- Deterministic HSV palette (same style as scripts/src/eval_occ_realsense.py)
- Query/highlight: only show one class in color, other classes gray (--query_label or --query_name)
- Save a per-scene legend PNG containing only labels present in this scene (--save_legend)

Examples:

(1) Visualize one frame npz:
python scripts/vis/vis_occ_realsense.py \
  --npz /data/FreeOcc/outputs/realsense_visualization/occ_vis_realsense/realsense0_rgbd/npz_seq/occ_frame_000120_rgbd_raw.npz \
  --names_txt ./src/scannet_utils/realsense0_top50.txt \
  --save_legend

(2) Visualize a sequence (incremental):
python scripts/vis/vis_occ_realsense.py \
  --npz_seq_dir /data/FreeOcc/outputs/realsense_visualization/occ_vis_realsense/realsense_2_cup_rgbd/npz_seq \
  --npz_seq_glob "occ_*.npz" \
  --names_txt ./src/scannet_utils/realsense_2cup.txt

(3) Query/highlight by label id:
python scripts/vis/vis_occ_realsense.py \
  --npz_seq_dir /data/FreeOcc/outputs/realsense_visualization/occ_vis_realsense/realsense0_rgbd/npz_seq \
  --names_txt ./src/scannet_utils/realsense0_top50.txt \
  --query_label 10

(4) Query/highlight by class name:
python scripts/vis/vis_occ_realsense.py \
  --npz_seq_dir .../npz_seq \
  --names_txt ./src/scannet_utils/realsense0_top50.txt \
  --query_name chair
"""

import argparse
import os
from typing import Optional, List, Sequence, Tuple

import numpy as np


# -------------------------
# IO
# -------------------------
def _load_npz_sparse(path: str):
    data = np.load(path)
    need = ["vox_coords", "vox_labels", "origin", "voxel_size"]
    for k in need:
        if k not in data.files:
            raise KeyError(f"Missing key '{k}' in npz: {path}. Has keys={data.files}")

    vox_coords = data["vox_coords"].astype(np.int32)  # [M,3]
    vox_labels = data["vox_labels"].astype(np.int32).reshape(-1)  # [M]
    origin = data["origin"].astype(np.float32).reshape(3)
    voxel_size = float(np.array(data["voxel_size"]).reshape(()))
    if vox_coords.ndim != 2 or vox_coords.shape[1] != 3:
        raise ValueError(f"vox_coords must be [M,3], got {vox_coords.shape}")
    if vox_coords.shape[0] != vox_labels.shape[0]:
        raise ValueError(f"vox_coords and vox_labels size mismatch: {vox_coords.shape[0]} vs {vox_labels.shape[0]}")
    return vox_coords, vox_labels, origin, voxel_size


def _read_names_txt(path: str) -> List[str]:
    names: List[str] = []
    if not path:
        return names
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            names.append(s)
    return names


# -------------------------
# Palette / LUT
# -------------------------
def _make_palette_hsv_rgba(n: int, seed: int = 0) -> np.ndarray:
    """Deterministic palette -> RGBA uint8, with stronger separation for many classes.

    Strategy:
    - Golden-ratio hue stepping (good dispersion)
    - Cycle several (S,V) pairs to increase differences in saturation/value
    """
    if n <= 0:
        return np.zeros((0, 4), dtype=np.uint8)

    n = int(n)
    phi = 0.618033988749895  # golden ratio conjugate
    h0 = (float(seed) * 0.3123) % 1.0

    sv_cycle = [
        (0.85, 0.95),
        (0.65, 0.95),
        (0.85, 0.80),
        (0.60, 0.85),
    ]

    hsv = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        hsv[i, 0] = (h0 + phi * i) % 1.0
        s, v = sv_cycle[i % len(sv_cycle)]
        hsv[i, 1] = s
        hsv[i, 2] = v

    # HSV -> RGB (vectorized)
    h = hsv[:, 0] * 6.0
    ii = np.floor(h).astype(np.int32)
    f = h - ii

    s = hsv[:, 1]
    v = hsv[:, 2]
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = np.zeros(n, dtype=np.float32)
    g = np.zeros(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)

    im = ii % 6
    m = (im == 0); r[m], g[m], b[m] = v[m], t[m], p[m]
    m = (im == 1); r[m], g[m], b[m] = q[m], v[m], p[m]
    m = (im == 2); r[m], g[m], b[m] = p[m], v[m], t[m]
    m = (im == 3); r[m], g[m], b[m] = p[m], q[m], v[m]
    m = (im == 4); r[m], g[m], b[m] = t[m], p[m], v[m]
    m = (im == 5); r[m], g[m], b[m] = v[m], p[m], q[m]

    rgb = (np.stack([r, g, b], axis=1) * 255.0).clip(0, 255).astype(np.uint8)
    rgba = np.concatenate([rgb, 255 * np.ones((n, 1), dtype=np.uint8)], axis=1)
    return rgba


def _get_lut_rgba(num_sem: int) -> np.ndarray:
    """LUT for labels 1..num_sem (size num_sem x 4)."""
    return _make_palette_hsv_rgba(int(num_sem), seed=0)


def _get_query_lut_rgba(
    base_lut_rgba: np.ndarray,
    query_label: int,
    gray_rgba: Tuple[int, int, int, int] = (200, 200, 200, 255),
) -> np.ndarray:
    """Highlight only query_label; others gray. base_lut_rgba is [num_sem,4]."""
    num_sem = int(base_lut_rgba.shape[0])
    lut = np.tile(np.array(gray_rgba, dtype=np.uint8).reshape(1, 4), (num_sem, 1))
    q = int(query_label)
    if 1 <= q <= num_sem:
        lut[q - 1] = base_lut_rgba[q - 1]
    return lut


# -------------------------
# Legend helpers
# -------------------------
def _present_labels_from_sparse(vox_labels: np.ndarray, vmin: int, vmax: int) -> List[int]:
    x = vox_labels.reshape(-1).astype(np.int32)
    x = x[(x != 0) & (x >= int(vmin)) & (x <= int(vmax))]
    if x.size == 0:
        return []
    return sorted(np.unique(x).tolist())


def _save_legend_png(
    out_path: str,
    present_labels: Sequence[int],
    lut_rgba: np.ndarray,
    names: Optional[List[str]] = None,
    title: Optional[str] = None,
    ncols: int = 6,
):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        raise ImportError("Saving legend requires pillow (PIL). Please install pillow in mayavi env.") from e

    labels = list(present_labels)
    if len(labels) == 0:
        return

    # layout
    pad = 14
    header_h = 40 if title else 10
    row_h = 34
    swatch = 20
    gap = 10

    n = len(labels)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / float(ncols)))

    col_w = 260
    W = pad * 2 + col_w * ncols
    H = pad * 2 + header_h + row_h * nrows

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_bold = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_bold = font

    y0 = pad
    if title:
        draw.text((pad, y0), title, fill=(0, 0, 0), font=font_bold)
        y0 += header_h
    else:
        y0 += header_h

    num_sem = int(lut_rgba.shape[0])

    for idx, lb in enumerate(labels):
        r = idx // ncols
        c = idx % ncols
        x = pad + c * col_w
        y = y0 + r * row_h

        if 1 <= int(lb) <= num_sem:
            color = lut_rgba[int(lb) - 1, :3].tolist()  # RGB
        else:
            color = [180, 180, 180]

        draw.rectangle([x, y + 6, x + swatch, y + 6 + swatch], fill=tuple(color), outline=(0, 0, 0))

        if names is not None and 1 <= int(lb) <= len(names):
            text = names[int(lb) - 1]
        else:
            text = f"label {int(lb)}"
        draw.text((x + swatch + gap, y + 4), text, fill=(0, 0, 0), font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


# -------------------------
# Visualization (Mayavi)
# -------------------------
def _sparse_voxel_centers(
    vox_coords: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """coords [M,3] -> centers xyz [M,3]."""
    return origin.reshape(1, 3) + (vox_coords.astype(np.float32) + 0.5) * float(voxel_size)


def draw_sparse_voxels_mayavi(
    vox_coords: np.ndarray,
    vox_labels: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
    num_sem: int,
    title: Optional[str] = None,
    query_label: int = 0,
    query_gray: Tuple[int, int, int, int] = (200, 200, 200, 255),
    geometry_only: bool = False,
    opacity: float = 1.0,
    max_points: int = 350000,
):
    from mayavi import mlab

    centers = _sparse_voxel_centers(vox_coords, origin, voxel_size)
    labels = vox_labels.reshape(-1).astype(np.int32)

    keep = labels > 0
    centers = centers[keep]
    labels = labels[keep]

    if centers.shape[0] == 0:
        print(f"[Mayavi-RealSense] {title or 'scene'}: no occupied voxels.")
        return

    # cap render points for speed
    if centers.shape[0] > int(max_points):
        idx = np.random.choice(centers.shape[0], size=int(max_points), replace=False)
        centers = centers[idx]
        labels = labels[idx]

    # clamp labels for coloring
    vmin = 1
    vmax = int(num_sem)
    labels_vis = np.clip(labels, vmin, vmax).astype(np.int32)

    if geometry_only:
        lut = np.tile(np.array([254, 194, 136, 255], dtype=np.uint8), (int(num_sem), 1))
    else:
        full_lut = _get_lut_rgba(int(num_sem))
        if int(query_label) > 0:
            lut = _get_query_lut_rgba(full_lut, int(query_label), gray_rgba=query_gray)
        else:
            lut = full_lut

    print(
        f"[Mayavi-RealSense] {title or 'scene'}: keep voxels={centers.shape[0]} "
        f"| labels(min,max)=({int(labels.min())},{int(labels.max())}) "
        f"| colored_by=[1..{vmax}]"
        + (f" | query={int(query_label)}" if int(query_label) > 0 else "")
    )

    fig = mlab.figure(size=(1920, 1080), bgcolor=(1, 1, 1))
    if title:
        try:
            fig.name = title
        except Exception:
            pass

    p = mlab.points3d(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        labels_vis,
        scale_factor=float(voxel_size),
        mode="cube",
        opacity=float(opacity),
        vmin=vmin,
        vmax=vmax,
    )
    p.module_manager.scalar_lut_manager.lut.table = lut

    if geometry_only:
        p.actor.property.interpolation = "flat"
        p.actor.property.edge_visibility = True
        p.actor.property.line_width = 1.0

    p.glyph.scale_mode = "scale_by_vector"

    try:
        fig.scene.renderer.reset_camera()
        fig.scene.renderer.reset_camera_clipping_range()
        fig.scene.render()
    except Exception:
        pass

    print(f"[Mayavi-RealSense] Showing: {title or 'scene'} (close window to continue)")
    mlab.show()


def _query_from_name(query_name: str, names: List[str]) -> int:
    q = str(query_name or "").strip()
    if not q:
        return 0
    name2id = {n: (i + 1) for i, n in enumerate(names)}
    return int(name2id.get(q, 0))


def main():
    parser = argparse.ArgumentParser("Visualize RealSense sparse occupancy npz (pred only) with Mayavi")
    parser.add_argument("--npz", type=str, default="", help="path to a single sparse occ npz")
    parser.add_argument(
        "--npz_seq_dir",
        type=str,
        default="",
        help="If set, visualize all npz in this folder sequentially.",
    )
    parser.add_argument("--npz_seq_glob", type=str, default="occ_*.npz", help="glob pattern inside --npz_seq_dir")

    parser.add_argument("--names_txt", type=str, default="", help="name list, line i -> label i+1")
    parser.add_argument("--num_sem", type=int, default=0, help="override semantic class count (0 -> infer from names_txt)")
    parser.add_argument("--save_legend", action="store_true")
    parser.add_argument("--legend_out", type=str, default="", help="output legend png path")
    parser.add_argument("--legend_cols", type=int, default=6)

    parser.add_argument("--query_label", type=int, default=0, help="If >0, highlight only this label; others gray")
    parser.add_argument("--query_name", type=str, default="", help="Highlight by class name (needs --names_txt)")
    parser.add_argument("--query_gray", type=str, default="200,200,200,255")

    parser.add_argument("--geometry_only", action="store_true")
    parser.add_argument("--opacity", type=float, default=1.0)
    parser.add_argument("--max_points", type=int, default=350000)

    args = parser.parse_args()

    names = _read_names_txt(args.names_txt.strip()) if args.names_txt.strip() else []
    if int(args.num_sem) > 0:
        num_sem = int(args.num_sem)
    else:
        num_sem = int(len(names)) if len(names) > 0 else 50  # fallback for your setting

    base_lut = _get_lut_rgba(num_sem)

    qgray = tuple(int(x) for x in str(args.query_gray).split(",")) if args.query_gray else (200, 200, 200, 255)

    query_label = int(args.query_label)
    if query_label <= 0 and args.query_name.strip():
        query_label = _query_from_name(args.query_name.strip(), names)

    # -------- sequence first --------
    if args.npz_seq_dir.strip():
        import glob

        seq_dir = os.path.abspath(args.npz_seq_dir.strip())
        seq_paths = sorted(glob.glob(os.path.join(seq_dir, str(args.npz_seq_glob))))
        if len(seq_paths) == 0:
            print(f"[NPZ-SEQ][WARN] no npz matched: {os.path.join(seq_dir, str(args.npz_seq_glob))}")
        else:
            print(f"[NPZ-SEQ] visualize {len(seq_paths)} npz files from: {seq_dir}")
            for k, pth in enumerate(seq_paths):
                vox_coords, vox_labels, origin, vs = _load_npz_sparse(pth)
                stem = os.path.splitext(os.path.basename(pth))[0]
                title = f"seq[{k+1}/{len(seq_paths)}] {stem}" + (f" [query {query_label}]" if query_label > 0 else "")
                draw_sparse_voxels_mayavi(
                    vox_coords=vox_coords,
                    vox_labels=vox_labels,
                    voxel_size=vs,
                    origin=origin,
                    num_sem=num_sem,
                    title=title,
                    query_label=query_label,
                    query_gray=qgray,
                    geometry_only=bool(args.geometry_only),
                    opacity=float(args.opacity),
                    max_points=int(args.max_points),
                )

                if args.save_legend:
                    present = _present_labels_from_sparse(vox_labels, vmin=1, vmax=num_sem)
                    lut = _get_query_lut_rgba(base_lut, query_label, gray_rgba=qgray) if query_label > 0 else base_lut
                    out_png = args.legend_out.strip()
                    if not out_png:
                        out_png = os.path.join(seq_dir, f"legend_{stem}.png")
                    _save_legend_png(
                        out_path=out_png,
                        present_labels=present,
                        lut_rgba=lut,
                        names=(names if len(names) > 0 else None),
                        title=f"RealSense legend: {os.path.basename(seq_dir)}",
                        ncols=int(args.legend_cols),
                    )
                    print(f"[Legend] saved: {out_png} | labels: {present}")

    # -------- single --------
    if args.npz.strip():
        npz_path = os.path.abspath(args.npz.strip())
        vox_coords, vox_labels, origin, vs = _load_npz_sparse(npz_path)

        if args.save_legend:
            npz_dir = os.path.dirname(npz_path)
            stem = os.path.splitext(os.path.basename(npz_path))[0]
            out_png = args.legend_out.strip() or os.path.join(npz_dir, f"legend_{stem}.png")
            present = _present_labels_from_sparse(vox_labels, vmin=1, vmax=num_sem)
            lut = _get_query_lut_rgba(base_lut, query_label, gray_rgba=qgray) if query_label > 0 else base_lut
            _save_legend_png(
                out_path=out_png,
                present_labels=present,
                lut_rgba=lut,
                names=(names if len(names) > 0 else None),
                title=f"RealSense legend: {os.path.basename(npz_dir)}",
                ncols=int(args.legend_cols),
            )
            print(f"[Legend] saved: {out_png} | labels: {present}")

        stem = os.path.splitext(os.path.basename(npz_path))[0]
        title = f"{stem}" + (f" [query {query_label}]" if query_label > 0 else "")
        draw_sparse_voxels_mayavi(
            vox_coords=vox_coords,
            vox_labels=vox_labels,
            voxel_size=vs,
            origin=origin,
            num_sem=num_sem,
            title=title,
            query_label=query_label,
            query_gray=qgray,
            geometry_only=bool(args.geometry_only),
            opacity=float(args.opacity),
            max_points=int(args.max_points),
        )

    if (not args.npz.strip()) and (not args.npz_seq_dir.strip()):
        raise ValueError("Please provide --npz or --npz_seq_dir")


if __name__ == "__main__":
    main()
