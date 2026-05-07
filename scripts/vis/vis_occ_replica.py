#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualize Replica occupancy npz (gt/pred) with Mayavi (101 semantic classes) and save a per-scene legend.

This script is Replica-only:
- label 0: empty/free (not rendered)
- label 255: ignore/unknown (not rendered)
- label 1..101: semantic classes (rendered with deterministic HSV palette)

Legend:
- Read class names from a txt file (101 lines, line i -> label i+1)
- Detect which labels appear in the current scene (from gt/pred)
- Save a legend PNG containing only the labels present in this scene

Bash all:
python scripts/vis/vis_occ_replica.py \
  --npz /data/FreeOcc/outputs/ours_replica_rgbd/occ_vis/room2_rgbd/occ.npz \
  --which both \
  --names_txt src/scannet_utils/replica_name.txt \
  --save_legend

Bash 8:
python scripts/vis/vis_occ_replica.py \
  --npz outputs/ours_replica_rgbd/occ_vis/office0_rgbd/occ.npz \
  --which both \
  --overlap 8 \
  --save_legend \
  --names_txt src/scannet_utils/replica_name.txt

Bash sequence:
python scripts/vis/vis_occ_replica.py \
  --npz outputs/ours_visualization/occ_vis/office0_rgbd/occ.npz \
  --npz_seq_dir outputs/ours_visualization/occ_vis/office0_rgbd/npz_seq \
  --which both \
  --names_txt src/scannet_utils/replica_name.txt \
  --save_legend

Bash query sequence:
python scripts/vis/vis_occ_replica.py \
  --npz outputs/ours_visualization/occ_vis/room2_rgbd/occ.npz \
  --npz_seq_dir outputs/ours_visualization/occ_vis/room2_rgbd/npz_seq \
  --which both \
  --query_label 10

Bash geometry only:
python scripts/vis/vis_occ_replica.py \
  --npz outputs/ours_replica_rgbd/occ_vis/office0_rgbd/occ.npz \
  --which both \
  --geometry_only
"""

import argparse
import os
from typing import Optional, List, Sequence

import numpy as np


REPLICA_NUM_SEM = 101
REPLICA_VMAX = 101

# 8-class overlap set (Replica label ids)
OVERLAP8_LABELS = {
    31,  # ceiling
    40,  # floor
    93,  # wall
    97,  # window
    20,  # chair
    7,   # bed
    76,  # sofa
    80,  # table
}
OVERLAP8_NAMES = {
    31: "ceiling",
    40: "floor",
    93: "wall",
    97: "window",
    20: "chair",
    7: "bed",
    76: "sofa",
    80: "table",
}

# ScanNet-style colors to use for the 8 overlap classes in overlap-8 mode only.
# Order corresponds to labels: ceiling, floor, wall, window, chair, bed, sofa, table.
OVERLAP8_SCANNET_RGBA = {
    31: (214, 38, 40, 255),   # ceiling
    40: (43, 160, 4, 255),    # floor
    93: (158, 216, 229, 255),
    97: (114, 158, 206, 255),
    20: (204, 204, 91, 255),
    7: (255, 186, 119, 255),
    76: (147, 102, 188, 255),
    80: (30, 119, 181, 255),
}


def _load_npz(path: str):
    data = np.load(path)
    gt = data["gt"].astype(np.int32)
    pred = data["pred"].astype(np.int32)
    valid_mask = data["valid_mask"].astype(bool)
    voxel_origin = data["voxel_origin"].astype(np.float32).reshape(3)
    voxel_size = float(data["voxel_size"].reshape(()))

    # ---- compatibility: some dumps may store gt/pred as flattened vectors ----
    # We restore shapes using valid_mask (preferred) or gt.
    if gt.ndim == 1 and valid_mask.ndim == 3 and gt.size == valid_mask.size:
        gt = gt.reshape(valid_mask.shape)
    if pred.ndim == 1 and valid_mask.ndim == 3 and pred.size == valid_mask.size:
        pred = pred.reshape(valid_mask.shape)

    # Fallback: if valid_mask isn't 3D but gt is, use gt's shape.
    if pred.ndim == 1 and gt.ndim == 3 and pred.size == gt.size:
        pred = pred.reshape(gt.shape)
    if gt.ndim == 1 and pred.ndim == 3 and gt.size == pred.size:
        gt = gt.reshape(pred.shape)

    return gt, pred, valid_mask, voxel_origin, voxel_size


def _read_names_txt_101(path: str) -> List[str]:
    names: List[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            names.append(s)
    if len(names) < REPLICA_NUM_SEM:
        raise ValueError(f"names_txt must have at least {REPLICA_NUM_SEM} lines, got {len(names)}: {path}")
    return names[:REPLICA_NUM_SEM]


def _make_palette_hsv_rgba(n: int) -> np.ndarray:
    """Deterministic HSV palette -> RGBA uint8, shape [n,4]."""
    if n <= 0:
        return np.zeros((0, 4), dtype=np.uint8)

    h = (np.arange(n, dtype=np.float32) / float(n)) * 6.0
    i = np.floor(h).astype(np.int32)
    f = h - i

    s = 0.75
    v = 0.95
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = np.zeros(n, dtype=np.float32)
    g = np.zeros(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)

    i_mod = i % 6
    m = (i_mod == 0); r[m], g[m], b[m] = v, t[m], p
    m = (i_mod == 1); r[m], g[m], b[m] = q[m], v, p
    m = (i_mod == 2); r[m], g[m], b[m] = p, v, t[m]
    m = (i_mod == 3); r[m], g[m], b[m] = p, q[m], v
    m = (i_mod == 4); r[m], g[m], b[m] = t[m], p, v
    m = (i_mod == 5); r[m], g[m], b[m] = v, p, q[m]

    rgba = (np.stack([r, g, b], axis=1) * 255.0).clip(0, 255).astype(np.uint8)
    rgba = np.concatenate([rgba, 255 * np.ones((n, 1), dtype=np.uint8)], axis=1)
    return rgba


def _get_replica_lut_rgba() -> np.ndarray:
    """LUT table for labels 1..101 (size 101x4)."""
    return _make_palette_hsv_rgba(REPLICA_NUM_SEM)


def _get_overlap8_lut_rgba(
    full_lut_rgba: np.ndarray,
    gray_rgba: tuple[int, int, int, int] = (180, 180, 180, 255),
) -> np.ndarray:
    """Build a 101x4 LUT for overlap-8 mode.

    - Overlap8 labels use ScanNet-style fixed colors (requested)
    - All other labels become gray

    full_lut_rgba is unused now but kept for backward compatibility.
    """
    lut = np.tile(np.array(gray_rgba, dtype=np.uint8).reshape(1, 4), (REPLICA_NUM_SEM, 1))
    for lb in OVERLAP8_LABELS:
        if 1 <= lb <= REPLICA_NUM_SEM:
            rgba = OVERLAP8_SCANNET_RGBA.get(int(lb))
            if rgba is not None:
                lut[lb - 1] = np.array(rgba, dtype=np.uint8)
            else:
                # fallback: keep previous behavior (use full lut color)
                lut[lb - 1] = full_lut_rgba[lb - 1]
    return lut


def _scene_present_labels(
    gt: np.ndarray,
    pred: np.ndarray,
    include: str,
) -> List[int]:
    """Return sorted label IDs present in this scene (within 1..101)."""
    labs: List[int] = []
    if include in ("gt", "both"):
        labs.append(gt.reshape(-1))
    if include in ("pred", "both"):
        labs.append(pred.reshape(-1))
    if len(labs) == 0:
        return []
    x = np.concatenate(labs, axis=0).astype(np.int32)
    x = x[(x != 0) & (x != 255) & (x >= 1) & (x <= REPLICA_VMAX)]
    if x.size == 0:
        return []
    return sorted(np.unique(x).tolist())


def _scene_present_labels_overlap8(
    gt: np.ndarray,
    pred: np.ndarray,
    include: str,
) -> List[int]:
    """Return sorted label IDs present in this scene. In overlap8 mode, only list the 8 classes for legend."""
    present = _scene_present_labels(gt=gt, pred=pred, include=include)
    return [lb for lb in present if lb in OVERLAP8_LABELS]


def _save_legend_png(
    out_path: str,
    present_labels: Sequence[int],
    lut_rgba: np.ndarray,
    names_txt: Optional[str] = None,
    title: Optional[str] = None,
    ncols: int = 6,
):
    """Save a legend image that contains only present labels.

    lut_rgba: shape [101,4], index (label-1)
    names_txt: 101 lines, line i -> label i+1
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        raise ImportError("Saving legend requires pillow (PIL). Please install pillow in mayavi env.") from e

    labels = list(present_labels)
    if len(labels) == 0:
        return

    names = None
    if names_txt:
        names = _read_names_txt_101(names_txt)

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
    font = None
    font_bold = None
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

    for idx, lb in enumerate(labels):
        r = idx // ncols
        c = idx % ncols
        x = pad + c * col_w
        y = y0 + r * row_h

        color = lut_rgba[int(lb) - 1, :3].tolist()  # RGB
        draw.rectangle([x, y + 6, x + swatch, y + 6 + swatch], fill=tuple(color), outline=(0, 0, 0))

        # Text: prefer pure class name; fallback to label id if names not provided.
        if names is not None and 1 <= lb <= len(names):
            text = names[lb - 1]
        else:
            text = OVERLAP8_NAMES.get(int(lb), f"label {lb}")

        draw.text((x + swatch + gap, y + 4), text, fill=(0, 0, 0), font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def _get_query_lut_rgba(
    base_lut_rgba: np.ndarray,
    query_label: int,
    gray_rgba: tuple[int, int, int, int] = (200, 200, 200, 255),
) -> np.ndarray:
    """Build a LUT that highlights only query_label and makes others gray."""
    lut = np.tile(np.array(gray_rgba, dtype=np.uint8).reshape(1, 4), (REPLICA_NUM_SEM, 1))
    q = int(query_label)
    if 1 <= q <= REPLICA_NUM_SEM:
        lut[q - 1] = base_lut_rgba[q - 1]
    return lut


def draw_voxel_mayavi_from_np(
    voxels: np.ndarray,
    valid_mask: np.ndarray,
    voxel_size: float,
    voxel_origin: np.ndarray,
    title: Optional[str] = None,
    overlap8_mode: bool = False,
    query_label: int = 0,
    query_gray: tuple[int, int, int, int] = (200, 200, 200, 255),
    geometry_only: bool = False,
):
    from mayavi import mlab

    # voxel centers (match (w,h,z) tensor convention)
    w, h, z = voxels.shape
    g_xx = np.arange(0, w)
    g_yy = np.arange(0, h)
    g_zz = np.arange(0, z)
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz, indexing="ij")

    coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
    coords = coords * float(voxel_size) + float(voxel_size) / 2.0
    coords = coords + voxel_origin.reshape(1, 3)

    labels = voxels.reshape(-1).astype(np.int32)
    mask = valid_mask.reshape(-1).astype(bool)

    ignore = (labels == 255)

    # Render anything >0 (occupied), ignore 255.
    keep = mask & (~ignore) & (labels > 0)
    coords = coords[keep]
    labels = labels[keep]

    # Clamp labels into [1,101] so they can be colored.
    labels_vis = np.clip(labels, 1, REPLICA_VMAX).astype(np.int32)

    if geometry_only:
        lut = np.tile(np.array([254, 194, 136, 255], dtype=np.uint8), (REPLICA_NUM_SEM, 1))
    else:
        # Choose LUT: full 101-class or overlap-8 (others gray)
        full_lut = _get_replica_lut_rgba()
        base_lut = _get_overlap8_lut_rgba(full_lut) if overlap8_mode else full_lut

        if int(query_label) > 0:
            lut = _get_query_lut_rgba(base_lut_rgba=base_lut, query_label=int(query_label), gray_rgba=query_gray)
        else:
            lut = base_lut

    print(
        f"[Mayavi-Replica] {title or 'scene'}: keep voxels = {coords.shape[0]} / {voxels.size} "
        f"| labels(min,max)=({int(labels.min()) if labels.size else -1},{int(labels.max()) if labels.size else -1}) "
        f"| colored_by=[1..{REPLICA_VMAX}]"
        + (" | overlap8_mode=ON" if overlap8_mode else "")
    )

    fig = mlab.figure(size=(1920, 1080), bgcolor=(1, 1, 1))
    if title:
        try:
            fig.name = title
        except Exception:
            pass

    if coords.shape[0] > 0:
        p = mlab.points3d(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            labels_vis,
            scale_factor=float(voxel_size),
            mode="cube",
            opacity=0.3,
            vmin=1,
            vmax=REPLICA_VMAX,
        )

        p.module_manager.scalar_lut_manager.lut.table = lut

        if geometry_only:
            p.actor.property.interpolation = 'flat'
            p.actor.property.edge_visibility = True
            p.actor.property.line_width = 1.0

        p.glyph.scale_mode = "scale_by_vector"

        try:
            fig.scene.renderer.reset_camera()
            fig.scene.renderer.reset_camera_clipping_range()
            fig.scene.render()
        except Exception:
            pass

    print(f"[Mayavi-Replica] Showing: {title or 'scene'} (close window to continue)")
    mlab.show()


def main():
    parser = argparse.ArgumentParser("Visualize Replica occupancy npz (gt/pred) with Mayavi")
    parser.add_argument("--npz", type=str, required=True, help="path to occ.npz")
    parser.add_argument("--which", type=str, default="both", choices=["gt", "pred", "both"], help="render gt/pred")

    # legend
    parser.add_argument("--save_legend", action="store_true", help="save per-scene legend PNG")
    parser.add_argument(
        "--legend_include",
        type=str,
        default="both",
        choices=["gt", "pred", "both"],
        help="detect present labels from gt/pred/both when saving legend",
    )
    parser.add_argument(
        "--names_txt",
        type=str,
        default="",
        help="path to replica name list (101 lines). line i -> label i+1",
    )
    parser.add_argument(
        "--legend_out",
        type=str,
        default="",
        help="output legend png path (default: <npz_dir>/legend_<npz_stem>.png)",
    )
    parser.add_argument("--legend_cols", type=int, default=6, help="legend columns")

    # overlap8 mode
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        choices=[0, 8],
        help="0: show all classes with their colors; 8: only the overlap-8 classes keep color, others are gray",
    )

    parser.add_argument(
        "--npz_seq_dir",
        type=str,
        default="",
        help="If set, visualize all npz in this folder sequentially (e.g. .../occ_vis/<scene>_<mode>/npz_seq), then visualize --npz at the end.",
    )
    parser.add_argument(
        "--npz_seq_glob",
        type=str,
        default="occ_*.npz",
        help="glob pattern inside --npz_seq_dir",
    )
    parser.add_argument(
        "--query_label",
        type=int,
        default=0,
        help="If >0, only highlight this label in GT visualization (others become light gray).",
    )

    parser.add_argument(
    "--geometry_only", 
    action="store_true", 
    help="If set, visualize as pure gray geometry without semantic colors."
    )

    args = parser.parse_args()

    # -----------------------
    # Optional: visualize npz sequence first (incremental occ)
    # -----------------------
    if args.npz_seq_dir.strip():
        import glob

        seq_dir = os.path.abspath(args.npz_seq_dir.strip())
        seq_paths = sorted(glob.glob(os.path.join(seq_dir, str(args.npz_seq_glob))))

        if len(seq_paths) == 0:
            print(f"[NPZ-SEQ][WARN] no npz matched: {os.path.join(seq_dir, str(args.npz_seq_glob))}")
        else:
            print(f"[NPZ-SEQ] visualize {len(seq_paths)} npz files from: {seq_dir}")

            for k, pth in enumerate(seq_paths):
                gt_k, pred_k, valid_k, origin_k, vs_k = _load_npz(pth)

                # window title show progress
                stem = os.path.splitext(os.path.basename(pth))[0]
                title_prefix = f"seq[{k+1}/{len(seq_paths)}] {stem}"

                if args.which in ("pred", "both"):
                    draw_voxel_mayavi_from_np(
                        voxels=pred_k,
                        valid_mask=valid_k,
                        voxel_size=vs_k,
                        voxel_origin=origin_k,
                        title=f"{title_prefix} pred" + (f" [query {int(args.query_label)}]" if int(args.query_label) > 0 else ""),
                        overlap8_mode=(int(args.overlap) == 8),
                        query_label=(int(args.query_label) if int(args.query_label) > 0 else 0),
                        geometry_only=args.geometry_only,
                    )

                if args.which in ("gt", "both"):
                    draw_voxel_mayavi_from_np(
                        voxels=gt_k,
                        valid_mask=valid_k,
                        voxel_size=vs_k,
                        voxel_origin=origin_k,
                        title=f"{title_prefix} gt",
                        overlap8_mode=(int(args.overlap) == 8),
                        query_label=(int(args.query_label) if int(args.query_label) > 0 else 0),
                        geometry_only=args.geometry_only,
                    )

    gt, pred, valid_mask, voxel_origin, voxel_size = _load_npz(args.npz)
    overlap8_mode = (int(args.overlap) == 8)

    # Save legend (only labels present in this scene)
    if args.save_legend:
        npz_dir = os.path.dirname(os.path.abspath(args.npz))
        stem = os.path.splitext(os.path.basename(args.npz))[0]
        out_path = args.legend_out.strip() or os.path.join(npz_dir, f"legend_{stem}.png")

        full_lut = _get_replica_lut_rgba()
        lut = _get_overlap8_lut_rgba(full_lut) if overlap8_mode else full_lut

        if overlap8_mode:
            present = _scene_present_labels_overlap8(gt=gt, pred=pred, include=args.legend_include)
        else:
            present = _scene_present_labels(gt=gt, pred=pred, include=args.legend_include)

        title = f"Replica legend: {os.path.basename(npz_dir)} ({args.legend_include})" + (" [overlap8]" if overlap8_mode else "")
        _save_legend_png(
            out_path=out_path,
            present_labels=present,
            lut_rgba=lut,
            names_txt=(args.names_txt.strip() or None),
            title=title,
            ncols=int(args.legend_cols),
        )
        print(f"[Legend] saved: {out_path} | labels: {present}")

    if args.which in ("pred", "both"):
        # pred: apply query_label highlight
        draw_voxel_mayavi_from_np(
            voxels=pred,
            valid_mask=valid_mask,
            voxel_size=voxel_size,
            voxel_origin=voxel_origin,
            title="pred" + (f" [query {int(args.query_label)}]" if int(args.query_label) > 0 else ""),
            overlap8_mode=overlap8_mode,
            query_label=(int(args.query_label) if int(args.query_label) > 0 else 0),
            geometry_only=args.geometry_only,
        )

    if args.which in ("gt", "both"):
        # gt: apply query_label highlight
        draw_voxel_mayavi_from_np(
            voxels=gt,
            valid_mask=valid_mask,
            voxel_size=voxel_size,
            voxel_origin=voxel_origin,
            title="gt" + (f" [query {int(args.query_label)}]" if int(args.query_label) > 0 else ""),
            overlap8_mode=overlap8_mode,
            query_label=(int(args.query_label) if int(args.query_label) > 0 else 0),
            geometry_only=args.geometry_only,
        )


if __name__ == "__main__":
    main()
