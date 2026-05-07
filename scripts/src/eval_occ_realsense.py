#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RealSense open-world occupancy incremental visualization (NO GT occ required).

Example:
python scripts/src/eval_occ_realsense.py \
  --exp_path /data/FreeOcc/outputs/realsense_visualization \
  --scene realsense_2cup \
  --mode rgbd \
  --names_txt ./src/scannet_utils/realsense_2cup.txt \
  --voxel_size 0.01 \
  --vis_backend plotly \
  --dump_npz_seq \
  --save_vis_seq \
  --dump_colored_points_ply \
  --class_query chair

It reads plys from:
  <exp_path>/<scene>_<mode>/mesh/frame_*_raw.ply   (or --ply_glob)

Outputs to:
  <exp_path>/occ_vis_realsense/<scene>_<mode>/
      npz_seq/occ_frame_000123_rgbd_raw.npz
      vis_seq/pred_frame_000123_rgbd_raw.png
      ply_seq/pred_points_frame_000123_rgbd_raw.ply
"""

import os
import glob
import argparse
from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
from src.gaussian_splatting.scene.gaussian_model import GaussianModel
try:
    import plotly.graph_objects as go
except Exception:
    go = None


# -------------------------
# Palette / names
# -------------------------
def _read_names_txt(path: str) -> List[str]:
    names: List[str] = []
    if not path:
        return names
    with open(path, "r") as f:
        for ln in f:
            s = ln.strip()
            if (not s) or s.startswith("#"):
                continue
            names.append(s)
    return names


def _make_palette_hsv_uint8(n: int, seed: int = 0) -> np.ndarray:
    """Deterministic HSV palette -> RGB uint8 [n,3]."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)

    hsv = np.zeros((n, 3), dtype=np.float32)
    hsv[:, 0] = (np.arange(n, dtype=np.float32) / float(n))  # H
    hsv[:, 1] = 0.75
    hsv[:, 2] = 0.95

    h = hsv[:, 0] * 6.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    p = hsv[:, 2] * (1.0 - hsv[:, 1])
    q = hsv[:, 2] * (1.0 - hsv[:, 1] * f)
    t = hsv[:, 2] * (1.0 - hsv[:, 1] * (1.0 - f))

    r = np.zeros(n, dtype=np.float32)
    g = np.zeros(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)

    i_mod = i % 6
    m = (i_mod == 0); r[m], g[m], b[m] = hsv[:, 2][m], t[m], p[m]
    m = (i_mod == 1); r[m], g[m], b[m] = q[m], hsv[:, 2][m], p[m]
    m = (i_mod == 2); r[m], g[m], b[m] = p[m], hsv[:, 2][m], t[m]
    m = (i_mod == 3); r[m], g[m], b[m] = p[m], q[m], hsv[:, 2][m]
    m = (i_mod == 4); r[m], g[m], b[m] = t[m], p[m], hsv[:, 2][m]
    m = (i_mod == 5); r[m], g[m], b[m] = hsv[:, 2][m], p[m], q[m]

    rgb = np.stack([r, g, b], axis=1)
    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def build_color_table_uint8(num_sem: int) -> np.ndarray:
    """
    label convention:
      0: free/empty
      1..num_sem: semantic
    return [num_sem+1, 3] uint8, with 0->black
    """
    tab = np.zeros((int(num_sem) + 1, 3), dtype=np.uint8)
    tab[1:, :] = _make_palette_hsv_uint8(int(num_sem))
    return tab


def parse_class_query(query: str, names: List[str]) -> Optional[int]:
    """
    Return 1-based label id if query matches.
    query can be:
      - "" / None -> None (no highlight)
      - integer string -> label id
      - class name -> map via names_txt line index+1
    """
    if query is None:
        return None
    q = str(query).strip()
    if not q:
        return None
    if q.isdigit():
        v = int(q)
        return v if v >= 0 else None
    # by name
    name2id = {n: (i + 1) for i, n in enumerate(names)}
    return name2id.get(q, None)


# -------------------------
# Voxelization (no GT occ)
# -------------------------
def voxelize_points_majority_vote(
    pts_xyz: np.ndarray,
    labels_1based: np.ndarray,
    voxel_size: float,
    origin: Optional[np.ndarray] = None,
    max_voxels: int = 256 * 256 * 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a sparse voxel map from points.
    - Each point assigned to voxel index = floor((p - origin) / voxel_size)
    - For each voxel, label = majority vote of point labels (0 allowed)
    Returns:
      vox_coords: [M,3] int32 voxel indices (ix,iy,iz)
      vox_labels: [M] int16 voxel label per voxel
      origin: [3] float32 used origin
    """
    assert pts_xyz.ndim == 2 and pts_xyz.shape[1] == 3
    assert labels_1based.ndim == 1 and labels_1based.shape[0] == pts_xyz.shape[0]
    voxel_size = float(voxel_size)

    if origin is None:
        origin = pts_xyz.min(axis=0)
    origin = np.array(origin, dtype=np.float32).reshape(3)

    ijk = np.floor((pts_xyz - origin[None, :]) / voxel_size).astype(np.int32)

    # hash voxels
    # use structured array for uniqueness
    key = ijk.view([("x", np.int32), ("y", np.int32), ("z", np.int32)]).reshape(-1)
    uniq, inv = np.unique(key, return_inverse=True)
    M = uniq.shape[0]
    if M > max_voxels:
        # subsample voxels (keep first max_voxels) to avoid huge memory
        keep_vox = np.arange(M, dtype=np.int32)[:max_voxels]
        mask_keep = np.isin(inv, keep_vox)
        ijk = ijk[mask_keep]
        labels_1based = labels_1based[mask_keep]
        key = ijk.view([("x", np.int32), ("y", np.int32), ("z", np.int32)]).reshape(-1)
        uniq, inv = np.unique(key, return_inverse=True)
        M = uniq.shape[0]

    # majority vote per voxel (avoid huge dense counts by sorting)
    order = np.argsort(inv, kind="mergesort")
    inv_s = inv[order]
    lab_s = labels_1based[order].astype(np.int32)

    # group boundaries
    edges = np.flatnonzero(np.diff(inv_s)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, inv_s.shape[0]]

    vox_labels = np.zeros((M,), dtype=np.int16)
    for gi, (s, e) in enumerate(zip(starts, ends)):
        labs = lab_s[s:e]
        if labs.size == 0:
            vox_labels[gi] = 0
            continue
        # bincount majority
        bc = np.bincount(labs, minlength=int(labs.max()) + 1)
        vox_labels[gi] = int(np.argmax(bc))

    vox_coords = np.zeros((M, 3), dtype=np.int32)
    vox_coords[:, 0] = uniq["x"]
    vox_coords[:, 1] = uniq["y"]
    vox_coords[:, 2] = uniq["z"]
    return vox_coords, vox_labels, origin


def voxel_centers_from_coords(vox_coords: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel index -> center xyz."""
    return origin[None, :] + (vox_coords.astype(np.float32) + 0.5) * float(voxel_size)


def dump_npz_sparse_occ(path: str, vox_coords: np.ndarray, vox_labels: np.ndarray, origin: np.ndarray, voxel_size: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        vox_coords=vox_coords.astype(np.int32),
        vox_labels=vox_labels.astype(np.int16),
        origin=np.array(origin, dtype=np.float32).reshape(3),
        voxel_size=np.float32(voxel_size),
    )


def save_xyzrgb_as_ply(path: str, xyz: np.ndarray, rgb_uint8: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb_uint8 = np.asarray(rgb_uint8, dtype=np.uint8).reshape(-1, 3)
    assert xyz.shape[0] == rgb_uint8.shape[0]

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb_uint8):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# -------------------------
# Plotly sparse voxel visualization
# -------------------------
def draw_sparse_voxels_plotly(
    centers: np.ndarray,
    labels: np.ndarray,
    color_table_uint8: np.ndarray,
    voxel_size: float,
    save_path: Optional[str],
    gray_others: bool = False,
    highlight_label: Optional[int] = None,
    gray_rgb=(160, 160, 160),
    max_voxels_vis: int = 200000,
    camera_d: float = 2.5,
):
    if go is None:
        raise ImportError("plotly is required for plotly backend.")

    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    labels = np.asarray(labels).reshape(-1).astype(np.int32)
    assert centers.shape[0] == labels.shape[0]

    # filter non-empty
    keep = labels != 0
    centers = centers[keep]
    labels = labels[keep]
    if centers.shape[0] == 0:
        return

    # cap for rendering speed
    if centers.shape[0] > int(max_voxels_vis):
        idx = np.random.choice(centers.shape[0], size=int(max_voxels_vis), replace=False)
        centers = centers[idx]
        labels = labels[idx]

    # colors
    lab_clip = np.clip(labels, 0, color_table_uint8.shape[0] - 1)
    colors = color_table_uint8[lab_clip].copy()  # [N,3] uint8

    if gray_others and (highlight_label is not None):
        other = labels != int(highlight_label)
        colors[other] = np.array(gray_rgb, dtype=np.uint8)

    colors_rgba = np.concatenate([colors, 255 * np.ones((colors.shape[0], 1), dtype=np.uint8)], axis=1)

    # cubes as ONE Mesh3d
    half = float(voxel_size) * 0.5
    offsets = np.array(
        [
            [-half, -half, -half],
            [-half, +half, -half],
            [+half, +half, -half],
            [+half, -half, -half],
            [-half, -half, +half],
            [-half, +half, +half],
            [+half, +half, +half],
            [+half, -half, +half],
        ],
        dtype=np.float32,
    )
    TRI = np.array(
        [
            (0, 2, 1), (0, 3, 2),  # bottom
            (4, 5, 6), (4, 6, 7),  # top
            (0, 7, 3), (0, 4, 7),  # front
            (1, 2, 6), (1, 6, 5),  # back
            (0, 1, 5), (0, 5, 4),  # left
            (3, 7, 6), (3, 6, 2),  # right
        ],
        dtype=np.int32,
    )

    N = centers.shape[0]
    verts = centers[:, None, :] + offsets[None, :, :]
    verts = verts.reshape(-1, 3)

    base = (np.arange(N, dtype=np.int32) * 8)[:, None]
    I = (base + TRI[None, :, 0]).reshape(-1)
    J = (base + TRI[None, :, 1]).reshape(-1)
    K = (base + TRI[None, :, 2]).reshape(-1)

    vcolor = np.repeat(colors_rgba, repeats=8, axis=0)

    # camera framing
    mins = centers.min(axis=0)
    maxs = centers.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.linalg.norm(maxs - mins) + 1e-6)

    eye = dict(x=float(center[0]), y=float(center[1]), z=float(center[2] - camera_d * span))

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=I, j=J, k=K,
            vertexcolor=vcolor,
            flatshading=True,
            lighting=dict(ambient=0.92, diffuse=0.8, specular=0.02, roughness=0.98, fresnel=0.01),
            lightposition=dict(x=float(center[0] + span), y=float(center[1] - span), z=float(center[2] + span)),
            opacity=1.0,
            showscale=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=eye,
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=-1.0, z=0.0),
                projection=dict(type="perspective"),
            ),
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        title="RealSense Open-World OCC (sparse voxels)",
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path, width=1920, height=1080, scale=1)


# -------------------------
# main
# -------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("RealSense open-world occ incremental visualization (no GT)")
    ap.add_argument("--exp_path", type=str, required=True)
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--mode", type=str, default="rgbd")
    ap.add_argument("--mesh_subdir", type=str, default="mesh")
    ap.add_argument("--ply_glob", type=str, default="frame_*_raw.ply", help="glob under mesh dir")

    ap.add_argument("--names_txt", type=str, default="", help="e.g. ./src/scannet_utils/realsense0_top50.txt")
    ap.add_argument("--num_sem", type=int, default=0, help="override semantic class count (0 -> infer from names_txt)")
    ap.add_argument("--voxel_size", type=float, default=0.08)
    ap.add_argument("--max_voxels_build", type=int, default=256 * 256 * 256)
    ap.add_argument("--max_voxels_vis", type=int, default=200000)

    ap.add_argument("--dump_npz_seq", action="store_true")
    ap.add_argument("--save_vis_seq", action="store_true")
    ap.add_argument("--dump_colored_points_ply", action="store_true", help="dump colored voxel-centers as pointcloud ply")

    ap.add_argument("--vis_backend", type=str, default="plotly", choices=["plotly"])
    ap.add_argument("--vis_camera_d", type=float, default=2.5)

    ap.add_argument("--class_query", type=str, default="", help="highlight class name or label id; others gray")
    ap.add_argument("--gray_others", action="store_true", help="when class_query is set, gray all other classes")
    ap.add_argument("--gray_rgb", type=str, default="160,160,160")

    args = ap.parse_args()

    scene_run_dir = os.path.join(args.exp_path, f"{args.scene}_{args.mode}")
    mesh_dir = os.path.join(scene_run_dir, str(args.mesh_subdir))
    ply_paths = sorted(glob.glob(os.path.join(mesh_dir, args.ply_glob)))
    if len(ply_paths) == 0:
        raise FileNotFoundError(f"No ply matched: {os.path.join(mesh_dir, args.ply_glob)}")

    names = _read_names_txt(args.names_txt) if args.names_txt else []
    if args.num_sem and args.num_sem > 0:
        num_sem = int(args.num_sem)
    else:
        num_sem = int(len(names)) if len(names) > 0 else 50  # fallback to 50 for your realsense case

    color_table = build_color_table_uint8(num_sem=num_sem)

    highlight_label = parse_class_query(args.class_query, names)
    gray_rgb = tuple(int(x) for x in args.gray_rgb.split(",")) if args.gray_rgb else (160, 160, 160)

    out_root = os.path.join(args.exp_path, "occ_vis_realsense", f"{args.scene}_{args.mode}")
    npz_dir = os.path.join(out_root, "npz_seq")
    vis_dir = os.path.join(out_root, "vis_seq")
    pts_ply_dir = os.path.join(out_root, "ply_seq")
    os.makedirs(out_root, exist_ok=True)

    print(f"[INFO] {len(ply_paths)} plys found.")
    print(f"[INFO] num_sem={num_sem} names_txt={args.names_txt}")
    if highlight_label is not None:
        print(f"[INFO] highlight_label={highlight_label} gray_others={args.gray_others}")

    for i, ply_path in enumerate(ply_paths):
        stem = os.path.splitext(os.path.basename(ply_path))[0]  # frame_000123_rgbd_raw
        print(f"[{i+1:04d}/{len(ply_paths):04d}] {stem}")

        g = GaussianModel(sh_degree=0)
        g.load_ply(ply_path)

        pts = g.get_xyz.detach().cpu().numpy().astype(np.float32)

        # semantic labels from ov_feat if present; else fallback to 1 (single class)
        if getattr(g, "ov_feat", None) is not None and torch.is_tensor(g.ov_feat):
            # g.ov_feat expected [N,C] or [C,H,W]? In your pipeline gs stores per-point feat [N,C]
            ov = g.ov_feat
            if ov.dim() == 2 and ov.shape[0] == pts.shape[0]:
                lab = torch.argmax(ov, dim=1).detach().cpu().numpy().astype(np.int32)
                # convert 0-based to 1-based
                labels_1based = (lab + 1).astype(np.int32)
            else:
                labels_1based = np.ones((pts.shape[0],), dtype=np.int32)
        else:
            labels_1based = np.ones((pts.shape[0],), dtype=np.int32)

        # clamp to palette range
        labels_1based = np.clip(labels_1based, 0, num_sem).astype(np.int32)

        # sparse voxelization + majority voting
        vox_coords, vox_labels, origin = voxelize_points_majority_vote(
            pts_xyz=pts,
            labels_1based=labels_1based,
            voxel_size=float(args.voxel_size),
            origin=None,
            max_voxels=int(args.max_voxels_build),
        )

        centers = voxel_centers_from_coords(vox_coords, origin, float(args.voxel_size))

        # dump npz
        if args.dump_npz_seq:
            npz_path = os.path.join(npz_dir, f"occ_{stem}.npz")
            dump_npz_sparse_occ(npz_path, vox_coords, vox_labels, origin, float(args.voxel_size))

        # dump colored points ply (voxel centers)
        if args.dump_colored_points_ply:
            lab_clip = np.clip(vox_labels.astype(np.int32), 0, color_table.shape[0] - 1)
            colors = color_table[lab_clip]
            if args.gray_others and (highlight_label is not None):
                other = vox_labels.astype(np.int32) != int(highlight_label)
                colors = colors.copy()
                colors[other] = np.array(gray_rgb, dtype=np.uint8)
            ply_out = os.path.join(pts_ply_dir, f"pred_points_{stem}.ply")
            save_xyzrgb_as_ply(ply_out, centers, colors)

        # visualization
        if args.save_vis_seq:
            png_out = os.path.join(vis_dir, f"pred_{stem}.png")
            draw_sparse_voxels_plotly(
                centers=centers,
                labels=vox_labels,
                color_table_uint8=color_table,
                voxel_size=float(args.voxel_size),
                save_path=png_out,
                gray_others=bool(args.gray_others),
                highlight_label=highlight_label,
                gray_rgb=gray_rgb,
                max_voxels_vis=int(args.max_voxels_vis),
                camera_d=float(args.vis_camera_d),
            )

    print(f"[DONE] saved to: {out_root}")


if __name__ == "__main__":
    main()
