#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualize occupancy npz (gt/pred) with Mayavi.

In mayavi conda environment.

Bash:
python scripts/vis/vis_occ_scannet.py \
  --npz outputs/embodied_scannet_all/occ_vis/scene0416_03_rgbd/occ.npz \
  --which both \
  --save_legend

Bash sequence (incremental occ):
python scripts/vis/vis_occ_scannet.py \
  --npz outputs/scannet_visualization/occ_vis/scene0000_00_rgbd/occ.npz \
  --npz_seq_dir outputs/scannet_visualization/occ_vis/scene0000_00_rgbd/npz_seq \
  --which pred

Bash geometry only:
python scripts/vis/vis_occ_scannet.py \
  --npz outputs/monogs_rgbd/occ_vis/scene0089_00_mono/occ.npz \
  --which both \
  --save_legend \
  --geometry_only
"""

import os
import argparse
from typing import Optional

import numpy as np


SCANNET_11_CLASS_NAMES = [
    "ceiling",
    "floor",
    "wall",
    "window",
    "chair",
    "bed",
    "sofa",
    "table",
    "tvs",
    "furniture",
    "objects",
]

SCANNET_11_RGBA = np.array(
    [
        [214, 38, 40, 255],
        [43, 160, 4, 255],
        [158, 216, 229, 255],
        [114, 158, 206, 255],
        [204, 204, 91, 255],
        [255, 186, 119, 255],
        [147, 102, 188, 255],
        [30, 119, 181, 255],
        [160, 188, 33, 255],
        [255, 127, 12, 255],
        [196, 175, 214, 255],
    ],
    dtype=np.uint8,
)


def _load_npz(path: str):
    data = np.load(path)
    gt = data["gt"].astype(np.int32)
    pred = data["pred"].astype(np.int32)
    valid_mask = data["valid_mask"].astype(bool)
    voxel_origin = data["voxel_origin"].astype(np.float32).reshape(3)
    voxel_size = float(data["voxel_size"].reshape(()))

    # ---- compatibility: some dumps may store gt/pred as flattened vectors ----
    if gt.ndim == 1 and valid_mask.ndim == 3 and gt.size == valid_mask.size:
        gt = gt.reshape(valid_mask.shape)
    if pred.ndim == 1 and valid_mask.ndim == 3 and pred.size == valid_mask.size:
        pred = pred.reshape(valid_mask.shape)

    if pred.ndim == 1 and gt.ndim == 3 and pred.size == gt.size:
        pred = pred.reshape(gt.shape)
    if gt.ndim == 1 and pred.ndim == 3 and gt.size == pred.size:
        gt = gt.reshape(pred.shape)

    return gt, pred, valid_mask, voxel_origin, voxel_size


def _present_semantic_labels_1to11(gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Return sorted unique labels in [1..11] that appear in gt/pred inside valid_mask."""
    m = valid_mask.astype(bool)
    labels = []
    for a in (gt, pred):
        if a is None:
            continue
        aa = np.asarray(a)
        if aa.shape != m.shape:
            continue
        v = aa[m].reshape(-1)
        v = v[(v >= 1) & (v <= 11)]
        if v.size:
            labels.append(v)
    if not labels:
        return np.zeros((0,), dtype=np.int32)
    u = np.unique(np.concatenate(labels, axis=0).astype(np.int32))
    u = u[(u >= 1) & (u <= 11)]
    return np.sort(u)


def _save_scannet_legend_png(save_path: str, title: str, present_labels: np.ndarray):
    """Save a simple legend image (names only) for ScanNet 11 classes.

    present_labels: array of ints in [1..11]
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        raise ImportError("Legend generation requires pillow. Please install pillow or disable --save_legend") from e

    labels = [int(x) for x in present_labels.tolist()]
    # fallback: if empty, still make an image with title
    n = len(labels)

    # layout
    padding = 18
    swatch = 22
    row_h = 34
    cols = 8

    # font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        font_title = ImageFont.load_default()

    title_h = 44
    rows = max(1, int(np.ceil(max(n, 1) / float(cols))))

    # estimate text width
    def _text_w(s: str, fnt):
        try:
            bbox = fnt.getbbox(s)
            return bbox[2] - bbox[0]
        except Exception:
            return 10 * len(s)

    max_name_w = 0
    for lb in labels:
        name = SCANNET_11_CLASS_NAMES[lb - 1]
        max_name_w = max(max_name_w, _text_w(name, font))
    max_name_w = max(max_name_w, 80)

    cell_w = swatch + 12 + max_name_w + 24
    W = padding * 2 + cols * cell_w
    H = padding * 2 + title_h + rows * row_h

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((padding, padding), title, fill=(0, 0, 0), font=font_title)

    start_y = padding + title_h
    for i, lb in enumerate(labels):
        r = i // cols
        c = i % cols
        x0 = padding + c * cell_w
        y0 = start_y + r * row_h

        rgba = SCANNET_11_RGBA[lb - 1].tolist()
        rgb = tuple(int(v) for v in rgba[:3])
        draw.rectangle([x0, y0 + 6, x0 + swatch, y0 + 6 + swatch], fill=rgb, outline=(0, 0, 0))

        name = SCANNET_11_CLASS_NAMES[lb - 1]
        draw.text((x0 + swatch + 12, y0 + 6), name, fill=(0, 0, 0), font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


def draw_voxel_mayavi_from_np(
    voxels: np.ndarray,
    valid_mask: np.ndarray,
    voxel_size: float,
    voxel_origin: np.ndarray,
    title: Optional[str] = None,
    geometry_only: bool = False,
):
    from mayavi import mlab

    # voxel centers
    w, h, z = voxels.shape
    g_xx = np.arange(0, w)
    g_yy = np.arange(0, h)
    g_zz = np.arange(0, z)
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
    coords = coords * float(voxel_size) + float(voxel_size) / 2.0
    coords = coords + voxel_origin.reshape(1, 3)

    labels = voxels.reshape(-1).astype(np.int32)
    mask = valid_mask.reshape(-1).astype(bool)

    keep = mask & (labels > 0) & (labels < 12)
    coords = coords[keep]
    labels = labels[keep]

    fig = mlab.figure(size=(1920, 1080), bgcolor=(1, 1, 1))
    if title:
        try:
            fig.name = title
        except Exception:
            pass

    if coords.shape[0] > 0:
        if geometry_only:
            p = mlab.points3d(
                coords[:, 0], coords[:, 1], coords[:, 2],
                color=(0.98, 0.36, 0.36),
                scale_factor=float(voxel_size),
                mode="cube",
                opacity=1.0,
            )
        else:
            p = mlab.points3d(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                labels,
                scale_factor=float(voxel_size),
                mode="cube",
                opacity=1.0,
                vmin=1,
                vmax=11,
            )
            p.module_manager.scalar_lut_manager.lut.table = SCANNET_11_RGBA
            p.glyph.scale_mode = "scale_by_vector"

        if geometry_only:
            p.actor.property.interpolation = 'flat'
            p.actor.property.edge_visibility = True
            p.actor.property.line_width = 1.0
            p.actor.property.edge_color = (0.0, 0.0, 0.0)

        try:
            fig.scene.renderer.reset_camera()
            fig.scene.renderer.reset_camera_clipping_range()
            fig.scene.render()
        except Exception:
            pass

    print(f"[Mayavi-ScanNet] Showing: {title or 'scene'} (close window to continue)")
    mlab.show()


def main():
    parser = argparse.ArgumentParser("Visualize occupancy npz (gt/pred) with Mayavi")
    parser.add_argument("--npz", type=str, required=True, help="path to occ.npz (final)")
    parser.add_argument("--which", type=str, default="both", choices=["gt", "pred", "both"], help="render gt/pred")

    # sequence mode: visualize all incremental occ npz first
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

    # legend png
    parser.add_argument(
        "--save_legend",
        action="store_true",
        help="If set, generate a legend PNG (only classes present in this scene) next to the final npz.",
    )
    parser.add_argument(
        "--legend_path",
        type=str,
        default="",
        help="Optional path to save legend PNG. Default: <dir_of_npz>/legend_scannet_<which>.png",
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
                stem = os.path.splitext(os.path.basename(pth))[0]
                title_prefix = f"seq[{k+1}/{len(seq_paths)}] {stem}"

                if args.which in ("pred", "both"):
                    draw_voxel_mayavi_from_np(
                        voxels=pred_k,
                        valid_mask=valid_k,
                        voxel_size=vs_k,
                        voxel_origin=origin_k,
                        title=f"{title_prefix} pred",
                        geometry_only=args.geometry_only
                    )

                if args.which in ("gt", "both"):
                    draw_voxel_mayavi_from_np(
                        voxels=gt_k,
                        valid_mask=valid_k,
                        voxel_size=vs_k,
                        voxel_origin=origin_k,
                        title=f"{title_prefix} gt",
                        geometry_only=args.geometry_only
                    )

    # -----------------------
    # Final occ
    # -----------------------
    gt, pred, valid_mask, voxel_origin, voxel_size = _load_npz(args.npz)

    # ---- legend (based on final gt/pred + valid_mask) ----
    if args.save_legend:
        present = _present_semantic_labels_1to11(gt=gt, pred=pred, valid_mask=valid_mask)
        if not args.legend_path.strip():
            out_path = os.path.join(
                os.path.dirname(os.path.abspath(args.npz)),
                f"legend_scannet_{args.which}.png",
            )
        else:
            out_path = os.path.abspath(args.legend_path.strip())

        scene_title = f"ScanNet legend: {os.path.splitext(os.path.basename(args.npz))[0]} ({args.which})"
        _save_scannet_legend_png(save_path=out_path, title=scene_title, present_labels=present)
        print(f"[Legend] saved: {out_path} | present labels: {present.tolist() if present.size else []}")

    if args.which in ("pred", "both"):
        draw_voxel_mayavi_from_np(
            voxels=pred,
            valid_mask=valid_mask,
            voxel_size=voxel_size,
            voxel_origin=voxel_origin,
            title="pred",
            geometry_only=args.geometry_only
        )

    if args.which in ("gt", "both"):
        draw_voxel_mayavi_from_np(
            voxels=gt,
            valid_mask=valid_mask,
            voxel_size=voxel_size,
            voxel_origin=voxel_origin,
            title="gt",
            geometry_only=args.geometry_only
        )


if __name__ == "__main__":
    main()
