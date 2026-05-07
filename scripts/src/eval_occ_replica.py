#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate Replica occupancy from Gaussian PLY files and optionally export
visualization assets.

Example:
python scripts/src/eval_occ_replica.py \
  --exp_path /data/FreeOcc/outputs/ours_visualization \
  --mode rgbd \
  --scenes room2 \
  --scene_occ_root /data/datasets/slam/Replica_OCC/Replica_OCC \
  --save_vis \
  --dump_npz \
  --dump_npz_all_ply \
  --vis_backend plotly
"""

import os
import argparse
from typing import List, Optional
import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_matrix
# NOTE: plotly is only needed when --save_vis and --vis_backend=plotly
try:
    import plotly.graph_objects as go
except Exception:
    go = None
from src.gaussian_splatting.scene.gaussian_model import GaussianModel
from src.scannet_utils.eval_utils import SSCMetricsTorch
from src.scannet_utils.dataloader import load_full_scene_occ
from src.gaussian_mapping import gaussians_to_occ, extract_gt_occupied_points


def save_xyz_as_ply(path, xyz: np.ndarray, color=None):
    """
    xyz: (N,3) float
    color: None or (N,3) uint8. If None, write white points.
    """
    xyz = xyz.astype(np.float32)
    N = xyz.shape[0]
    if color is None:
        color = np.full((N, 3), 255, dtype=np.uint8)
    else:
        color = color.astype(np.uint8)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, color):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def dump_debug_occupied_points(
    out_dir: str,
    scene: str,
    mode: str,
    scene_data: dict,
    gt_occ: torch.Tensor,
    pred_occ: torch.Tensor,
) -> None:
    """Dump GT and predicted occupied voxels as red/green point clouds."""
    occ_pts = scene_data.get("occ_points", None)
    if occ_pts is None:
        print("[DebugPoints][Warn] scene_data has no occ_points; skip debug point export.")
        return

    pts_flat = np.asarray(occ_pts).reshape(-1, 3)
    gt_flat = gt_occ.reshape(-1).detach().cpu().numpy()
    pred_flat = pred_occ.reshape(-1).detach().cpu().numpy()

    gt_occ_pts = pts_flat[gt_flat > 0]
    pred_occ_pts = pts_flat[pred_flat > 0]

    os.makedirs(out_dir, exist_ok=True)
    save_xyz_as_ply(
        os.path.join(out_dir, f"{scene}_{mode}_gt_occ.ply"),
        gt_occ_pts,
        color=np.array([255, 0, 0], dtype=np.uint8)[None, :].repeat(gt_occ_pts.shape[0], axis=0),
    )
    save_xyz_as_ply(
        os.path.join(out_dir, f"{scene}_{mode}_pred_occ.ply"),
        pred_occ_pts,
        color=np.array([0, 255, 0], dtype=np.uint8)[None, :].repeat(pred_occ_pts.shape[0], axis=0),
    )
    print(f"[DebugPoints] saved PLYs to {out_dir}")


def parse_scenes(s: str) -> List[str]:
    s = s.replace(",", " ").strip()
    return [x for x in s.split() if x]


def read_scenes_txt(path: str) -> List[str]:
    scenes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            scenes.append(line.split()[0])
    return scenes


def _make_palette_hsv_rgb_float(n: int) -> np.ndarray:
    """Deterministic HSV palette -> RGB float in [0,1], shape [n,3]."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)

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

    return np.stack([r, g, b], axis=1).astype(np.float32)


def _get_replica_color_table_rgb_float(num_sem: int = 101) -> np.ndarray:
    """Return color_table indexed by label id: [num_sem+1,3] float. label=0 is empty(black)."""
    tab = np.zeros((int(num_sem) + 1, 3), dtype=np.float32)
    tab[1:, :] = _make_palette_hsv_rgb_float(int(num_sem))
    return tab


def _normalize_replica_occ_labels(x: np.ndarray) -> np.ndarray:
    """Normalize Replica_OCC labels for visualization.

    Replica_OCC GT often uses 255 as unknown/ignore.
    In our visualization conventions, 0 is empty/unknown and labels 1..K are semantic.
    """
    x = np.asarray(x)
    if x.dtype == np.bool_:
        return x
    return np.where(x == 255, 0, x)


def _dump_occ_npz(
    save_path: str,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
    valid_mask: np.ndarray,
    voxel_origin: np.ndarray,
    voxel_size: float,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    gt_np = _normalize_replica_occ_labels(gt_np)
    pred_np = _normalize_replica_occ_labels(pred_np)

    np.savez_compressed(
        save_path,
        gt=gt_np.astype(np.int16),
        pred=pred_np.astype(np.int16),
        valid_mask=valid_mask.astype(np.bool_),
        voxel_origin=np.array(voxel_origin, dtype=np.float32).reshape(3),
        voxel_size=np.float32(voxel_size),
    )


def dump_npz_for_all_plys_in_scene(
    scene: str,
    mode: str,
    exp_path: str,
    scene_data: dict,
    device: torch.device,
    vis_scene_dir: str,
    voxel_size_fallback: float,
    ply_glob: str = "frame_*_*.ply",
):
    """
    For a given scene, iterate over all incremental ply files under:
      <exp_path>/<scene>_<mode>/mesh/<ply_glob>
    and dump corresponding occ npz into:
      <vis_scene_dir>/npz_seq/occ_<ply_stem>.npz

    This does NOT change evaluation; it's only for incremental visualization.
    """
    import glob

    mesh_dir = os.path.join(exp_path, f"{scene}_{mode}", "mesh")
    ply_paths = sorted(glob.glob(os.path.join(mesh_dir, ply_glob)))

    if len(ply_paths) == 0:
        print(f"[DUMP_SEQ] No ply matched: {os.path.join(mesh_dir, ply_glob)}")
        return

    # output folder for the sequence
    out_dir = os.path.join(vis_scene_dir, "npz_seq")
    os.makedirs(out_dir, exist_ok=True)

    voxel_origin = np.array(scene_data.get("origin", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
    voxel_size = float(scene_data.get("voxel_size", voxel_size_fallback))
    valid_mask = np.ones_like(scene_data["occ_labels"], dtype=np.bool_)

    print(f"[DUMP_SEQ] dumping {len(ply_paths)} plys -> {out_dir}")
    for i, ply_path in enumerate(ply_paths):
        stem = os.path.splitext(os.path.basename(ply_path))[0]  # e.g. frame_000123_rgbd
        out_npz = os.path.join(out_dir, f"occ_{stem}.npz")

        if os.path.exists(out_npz):
            # skip existing to save time (optional)
            continue

        try:
            g = GaussianModel(sh_degree=0)
            g.load_ply(ply_path)

            pred_occ = gaussians_to_occ(
                g.get_xyz,
                g.get_features.squeeze(1),
                g.get_scaling,
                quaternion_to_matrix(g.get_rotation),
                g.get_opacity,
                g.ov_feat,
                scene_data,
            )

            # dump npz
            _dump_occ_npz(
                save_path=out_npz,
                gt_np=scene_data["occ_labels"],
                pred_np=pred_occ.squeeze(0).detach().cpu().numpy() if pred_occ.dim() == 4 else pred_occ.detach().cpu().numpy(),
                valid_mask=valid_mask,
                voxel_origin=voxel_origin,
                voxel_size=voxel_size,
            )

            if (i % 10) == 0:
                print(f"[DUMP_SEQ] [{i+1}/{len(ply_paths)}] saved: {out_npz}")
        except Exception as e:
            print(f"[DUMP_SEQ][WARN] failed: {ply_path} -> {type(e).__name__}: {e}")


def get_gt_occ(scene_name: str, scene_occ_root: str, voxel_size: float):
    scene_data = load_full_scene_occ(
        scene_name=scene_name,
        scene_occ_root=scene_occ_root,
        voxel_size=voxel_size,
        to_torch=False,
    )

    # Keep label 255 for metric evaluation. SSCMetricsTorch ignores 255; converting
    # it to 0 would incorrectly treat unknown voxels as free space and lower IoU.
    print("occ_labels shape:", scene_data["occ_labels"].shape)
    if "scene_dim" in scene_data:
        print("scene_dim:", scene_data["scene_dim"])
    if "origin" in scene_data:
        print("origin:", scene_data["origin"])
    if "voxel_size" in scene_data:
        print("voxel_size:", scene_data["voxel_size"])
    gt_occ_pts = extract_gt_occupied_points(scene_data, min_label=0)
    return scene_data, gt_occ_pts


def ensure_pred_shape(pred_occ: torch.Tensor, gt_occ: torch.Tensor) -> torch.Tensor:
    """Normalize pred_occ shape to [1, ...] and match gt_occ for evaluator.add_batch."""
    if pred_occ.dim() == gt_occ.dim() - 1:
        pred_occ = pred_occ.unsqueeze(0)

    if pred_occ.shape != gt_occ.shape:
        # Fallback: try to view it into the GT shape.
        pred_occ = pred_occ.view(gt_occ.shape)
    return pred_occ


def _read_names_txt(path: str) -> List[str]:
    names: List[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            names.append(s)
    return names


def _names_to_labels_1based(names_txt_path: str, class_names: List[str]) -> List[int]:
    """Map class names to 1-based label IDs.

    Convention: each line in names_txt is a semantic class name; label=line_number+1,
    and 0 is free. Unmatched names are ignored.
    """
    all_names = _read_names_txt(names_txt_path)
    name2label = {n: (i + 1) for i, n in enumerate(all_names)}

    out: List[int] = []
    seen = set()
    for n in class_names:
        lb = name2label.get(n, None)
        if lb is None:
            continue
        if lb in seen:
            continue
        seen.add(lb)
        out.append(int(lb))
    return out


def get_grid_coords(dims, resolution):
    """Return voxel center coords for a [W,H,Z] grid.

    dims: (w,h,z)
    resolution: (vx,vy,vz)
    return: [w*h*z, 3] centers in local frame (origin at [0,0,0])
    """
    w, h, z = [int(x) for x in dims]
    vx, vy, vz = [float(r) for r in resolution]

    xs = (np.arange(w, dtype=np.float32) + 0.5) * vx
    ys = (np.arange(h, dtype=np.float32) + 0.5) * vy
    zs = (np.arange(z, dtype=np.float32) + 0.5) * vz
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X, Y, Z], axis=-1).reshape(-1, 3)


# -------------------------
# Plotly voxel visualization (KEEP ALL CUBES)
# -------------------------
def draw_voxel_plotly_image(
    voxels: np.ndarray,
    fov_mask: np.ndarray,
    voxel_size: float = 0.05,
    vox_origin: Optional[np.ndarray] = None,
    intrinsic=None,
    cam_pose=None,
    d: float = 1.25,
    save_path: Optional[str] = None,
    scene_origin: Optional[np.ndarray] = None,
    scene_size: Optional[np.ndarray] = None,
    max_points: Optional[int] = None,   # None -> no subsample
    num_sem: int = 101
):
    """Render all occupied voxels as a single Plotly Mesh3d.

    Notes:
      - each cube is merged into one Mesh3d with per-vertex colors
      - perspective projection is explicit
      - high ambient lighting avoids overly dark renders
    """
    assert voxels.ndim == 3, f"voxels must be [W,H,Z], got {voxels.shape}"
    w, h, z = voxels.shape

    if vox_origin is None:
        vox_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        vox_origin = np.array(vox_origin, dtype=np.float32).reshape(3)

    color_table = _get_replica_color_table_rgb_float(num_sem)

    # ---- voxel centers in world ----
    grid_coords = get_grid_coords((w, h, z), (voxel_size, voxel_size, voxel_size)) + vox_origin.reshape(1, 3)
    grid_values = voxels.reshape(-1).astype(np.int32)

    # ---- mask & non-empty ----
    fov_flat = fov_mask.reshape(-1).astype(bool)
    keep = fov_flat & (grid_values != 0)
    coords = grid_coords[keep].astype(np.float32)   # [N,3]
    labels = grid_values[keep].astype(np.int32)     # [N]

    fig = go.Figure()

    if coords.shape[0] == 0:
        fig.update_layout(
            title="Voxel Visualization (empty)",
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=False,
        )
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path, width=1920, height=1080, scale=1)
        return fig

    # Optional subsample. By default, keep all cubes.
    n = coords.shape[0]
    if (max_points is not None) and (max_points > 0) and (n > max_points):
        idx = np.random.choice(n, size=max_points, replace=False)
        coords = coords[idx]
        labels = labels[idx]

    # per-voxel colors
    colors = color_table[np.clip(labels, 0, color_table.shape[0] - 1)]  # [N,3] float
    colors_rgba255 = np.concatenate(
        [np.clip(colors * 255.0, 0, 255).astype(np.uint8),
         255 * np.ones((colors.shape[0], 1), dtype=np.uint8)],
        axis=1
    )  # [N,4]

    # ---------- scene ranges ----------
    if scene_origin is None:
        scene_origin = vox_origin.copy()
    else:
        scene_origin = np.array(scene_origin, dtype=np.float32).reshape(3)

    if scene_size is None:
        scene_size = np.array([w * voxel_size, h * voxel_size, z * voxel_size], dtype=np.float32)
    else:
        scene_size = np.array(scene_size, dtype=np.float32).reshape(3)

    ox, oy, oz = [float(v) for v in scene_origin.tolist()]
    sx, sy, sz = [float(v) for v in scene_size.tolist()]
    x_rng = [ox, ox + sx]
    y_rng = [oy, oy + sy]
    z_rng = [oz, oz + sz]

    # ---------- camera: look toward the scene center along -Z ----------
    center = scene_origin + 0.5 * scene_size
    cx, cy, cz = [float(v) for v in center.tolist()]
    max_dim = float(max(sx, sy, sz))

    # Larger d moves the camera farther away and weakens perspective.
    eye = dict(x=cx, y=cy, z=cz - float(d) * max_dim)

    camera_layout = dict(
        eye=eye,
        center=dict(x=0.0, y=0.0, z=0.0),
        up=dict(x=0.0, y=-1.0, z=0.0),
        projection=dict(type="perspective"),
    )

    # ---------- lighting ----------
    lighting = dict(
        ambient=0.92,
        diffuse=0.80,
        specular=0.02,
        roughness=0.98,
        fresnel=0.01,
    )
    lightpos = dict(
        x=cx + 0.8 * max_dim,
        y=cy - 0.8 * max_dim,
        z=cz + 1.2 * max_dim,
    )

    # ---------- build ONE Mesh3d for ALL cubes ----------
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
        dtype=np.float32
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
        dtype=np.int32
    )

    N = coords.shape[0]
    verts = coords[:, None, :] + offsets[None, :, :]   # [N,8,3]
    verts = verts.reshape(-1, 3)                       # [N*8,3]

    base = (np.arange(N, dtype=np.int32) * 8)[:, None]  # [N,1]
    I = (base + TRI[None, :, 0]).reshape(-1)
    J = (base + TRI[None, :, 1]).reshape(-1)
    K = (base + TRI[None, :, 2]).reshape(-1)

    vcolor = np.repeat(colors_rgba255, repeats=8, axis=0)  # [N*8,4]

    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=I, j=J, k=K,
            vertexcolor=vcolor,
            flatshading=True,
            lighting=lighting,
            lightposition=lightpos,
            opacity=1.0,
            showscale=False,
            hoverinfo="skip",
        )
    )

    show_axis = False
    fig.update_layout(
        scene=dict(
            domain=dict(x=[0, 1], y=[0, 1]),
            xaxis=dict(range=x_rng, autorange=False, visible=show_axis, showgrid=show_axis, zeroline=False, showbackground=False),
            yaxis=dict(range=y_rng, autorange=False, visible=show_axis, showgrid=show_axis, zeroline=False, showbackground=False),
            zaxis=dict(range=z_rng, autorange=False, visible=show_axis, showgrid=show_axis, zeroline=False, showbackground=False),
            aspectmode="manual",
            aspectratio=dict(x=float(sx), y=float(sy), z=float(sz)),
            camera=camera_layout,
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="Voxel Visualization with Camera View" if intrinsic is not None else "Voxel Visualization",
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path, width=1920, height=1080, scale=1)

    return fig


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Evaluate multi-class occupancy metrics from Gaussian PLY")

    parser.add_argument("--exp_path", type=str, required=True, help="experiment root path")
    parser.add_argument("--mode", type=str, default="mono", help="e.g. mono / stereo ...")

    parser.add_argument("--scenes", type=str, default="", help="scene0000_00,scene0001_00 ...")
    parser.add_argument("--scenes_txt", type=str, default="", help="txt file with one scene per line")

    parser.add_argument("--scene_occ_root", type=str, default="/data/datasets/slam/Replica_OCC/Replica_OCC")
    parser.add_argument("--voxel_size", type=float, default=0.08)

    parser.add_argument("--num_classes", type=int, default=102, help="for SSCMetricsTorch(K)")
    parser.add_argument("--device", type=str, default="cuda")

    # Used to find labels by class name when selecting the 8 overlapping classes.
    parser.add_argument(
        "--gt_names_path",
        type=str,
        default="./src/scannet_utils/replica_name.txt",
        help="full semantic name list used by GT label space (txt line i -> label i+1; 0 is free)",
    )

    # vis
    parser.add_argument("--save_vis", action="store_true", help="save GT/Pred voxel visualization")
    parser.add_argument("--vis_dir", type=str, default="", help="default: <exp_path>/occ_vis")
    parser.add_argument("--vis_d", type=float, default=3.0, help="camera distance multiplier")
    parser.add_argument("--vis_scene_size_pad", type=float, default=1.0, help="scene_size + pad")
    parser.add_argument("--vis_max_points", type=int, default=-1, help="<=0 keep all cubes; >0 subsample cubes")

    parser.add_argument(
        "--dump_npz",
        action="store_true",
        help="dump gt/pred/valid_mask/voxel_origin/voxel_size to <vis_dir>/<scene>_<mode>/occ.npz",
    )
    parser.add_argument(
        "--vis_backend",
        type=str,
        default="plotly",
        choices=["plotly", "mayavi"],
        help="visualization backend",
    )

    # replica palette: usually 101 semantic classes (labels 1..101)
    parser.add_argument("--replica_num_sem", type=int, default=101, help="replica semantic classes count for visualization")

    parser.add_argument(
        "--dump_npz_all_ply",
        action="store_true",
        help="dump occ npz for EVERY ply in <scene>_<mode>/mesh (e.g. frame_*.ply), for incremental occ visualization",
    )
    parser.add_argument(
        "--ply_glob",
        type=str,
        default="frame_*_*.ply",
        help="glob pattern under <exp_path>/<scene>_<mode>/mesh/ to select incremental ply files",
    )
    parser.add_argument(
        "--dump_debug_points",
        action="store_true",
        help="dump GT and predicted occupied voxels as red/green point-cloud PLY files",
    )
    parser.add_argument(
        "--debug_points_dir",
        type=str,
        default="",
        help="default: <exp_path>/occ_debug_points",
    )
    parser.add_argument(
        "--print_topk",
        action="store_true",
        help="print mIoU_top10/20/30/40 in summary",
    )

    args = parser.parse_args()

    # default vis_dir
    if not args.vis_dir:
        args.vis_dir = os.path.join(args.exp_path, "occ_vis")

    if args.save_vis and args.vis_backend == "plotly" and go is None:
        raise ImportError(
            "plotly is required for --vis_backend=plotly, but it is not installed in this environment. "
            "Install it or use --vis_backend=mayavi / --dump_npz for offline rendering."
        )

    if args.scenes_txt:
        scenes = read_scenes_txt(args.scenes_txt)
    else:
        scenes = parse_scenes(args.scenes)

    if len(scenes) == 0:
        raise ValueError("No scenes provided. Use --scenes or --scenes_txt")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    evaluator = SSCMetricsTorch(args.num_classes)

    n_total, n_ok, n_missing = 0, 0, 0

    for scene in scenes:
        n_total += 1
        print(f"\n[Eval] Scene: {scene}")

        # 1) GT occ
        scene_data, _ = get_gt_occ(
            scene_name=scene,
            scene_occ_root=args.scene_occ_root,
            voxel_size=args.voxel_size,
        )
        print("GT occ_labels shape:", scene_data["occ_labels"].shape)
        gt_unique_np = np.unique(scene_data["occ_labels"]).astype(np.int64)
        print("GT unique labels:", gt_unique_np)

        vis_scene_dir = os.path.join(args.vis_dir, f"{scene}_{args.mode}")
        os.makedirs(vis_scene_dir, exist_ok=True)
        if args.dump_npz_all_ply:
            dump_npz_for_all_plys_in_scene(
                scene=scene,
                mode=args.mode,
                exp_path=args.exp_path,
                scene_data=scene_data,
                device=device,
                vis_scene_dir=vis_scene_dir,
                voxel_size_fallback=float(args.voxel_size),
                ply_glob=str(args.ply_glob),
            )

        # 2) PLY path
        scene_path = os.path.join(args.exp_path, f"{scene}_{args.mode}", "mesh", f"final_{args.mode}.ply")
        if not os.path.exists(scene_path):
            print(f"[Warn] Missing ply: {scene_path}  -> skip")
            n_missing += 1
            continue

        # 3) Load gaussians
        g = GaussianModel(sh_degree=0)
        g.load_ply(scene_path)

        # 4) pred occ from gaussians
        ov_feat = g.ov_feat

        pred_occ = gaussians_to_occ(
            g.get_xyz,
            g.get_features.squeeze(1),
            g.get_scaling,
            quaternion_to_matrix(g.get_rotation),
            g.get_opacity,
            ov_feat,
            scene_data,
        )

        print("pred_occ before ensure shape:", pred_occ.shape, pred_occ.dtype)
        print("pred unique labels:", torch.unique(pred_occ.cpu()))

        # 5) GT occ tensor
        gt_occ = torch.from_numpy(scene_data["occ_labels"]).to(device=device)
        gt_occ = gt_occ.long().unsqueeze(0)  # [1, ...]

        # Normalize pred device / dtype / shape.
        if torch.is_tensor(pred_occ):
            pred_occ = pred_occ.to(device=device).long()
        else:
            raise TypeError(f"gaussians_to_occ should return torch.Tensor, got {type(pred_occ)}")

        pred_occ = ensure_pred_shape(pred_occ, gt_occ)

        if args.dump_debug_points:
            debug_points_dir = args.debug_points_dir if args.debug_points_dir else os.path.join(args.exp_path, "occ_debug_points")
            dump_debug_occupied_points(debug_points_dir, scene, args.mode, scene_data, gt_occ, pred_occ)

        if args.dump_npz:
            voxel_origin = np.array(scene_data.get("origin", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
            voxel_size = float(scene_data.get("voxel_size", args.voxel_size))
            valid_mask = np.ones_like(scene_data["occ_labels"], dtype=np.bool_)

            npz_path = os.path.join(vis_scene_dir, "occ.npz")
            _dump_occ_npz(
                save_path=npz_path,
                gt_np=scene_data["occ_labels"],
                pred_np=pred_occ.squeeze(0).detach().cpu().numpy(),
                valid_mask=valid_mask,
                voxel_origin=voxel_origin,
                voxel_size=voxel_size,
            )
            print(f"[DUMP] occ npz saved to: {npz_path}")

        if args.save_vis:
            os.makedirs(vis_scene_dir, exist_ok=True)

            # NOTE: draw_voxel_plotly_image expects [W,H,Z]
            gt_np = _normalize_replica_occ_labels(scene_data["occ_labels"]).astype(np.int32)
            pred_np = _normalize_replica_occ_labels(
                pred_occ.squeeze(0).detach().cpu().numpy()
            ).astype(np.int32)

            # valid mask (FOV) from loader if present; otherwise all True
            if "valid_mask" in scene_data:
                valid_mask = scene_data["valid_mask"].astype(bool)
            else:
                valid_mask = np.ones_like(gt_np, dtype=bool)

            # scene size for framing
            w, h, z = gt_np.shape
            voxel_origin = np.array(scene_data.get("origin", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
            scene_size = np.array([w, h, z], dtype=np.float32) * float(args.voxel_size)
            scene_size = scene_size + float(args.vis_scene_size_pad)

            if args.vis_backend == "plotly":
                gt_png = os.path.join(vis_scene_dir, "gt.png")
                pred_png = os.path.join(vis_scene_dir, "pred.png")

                max_points = None if (args.vis_max_points is None or args.vis_max_points <= 0) else int(args.vis_max_points)

                draw_voxel_plotly_image(
                    gt_np,
                    valid_mask,
                    voxel_size=float(args.voxel_size),
                    vox_origin=voxel_origin,
                    d=float(args.vis_d),
                    save_path=gt_png,
                    scene_origin=voxel_origin,
                    scene_size=scene_size,
                    max_points=max_points,
                    num_sem=int(args.replica_num_sem),
                )
                draw_voxel_plotly_image(
                    pred_np,
                    valid_mask,
                    voxel_size=float(args.voxel_size),
                    vox_origin=voxel_origin,
                    d=float(args.vis_d),
                    save_path=pred_png,
                    scene_origin=voxel_origin,
                    scene_size=scene_size,
                    max_points=max_points,
                    num_sem=int(args.replica_num_sem),
                )
                print(f"[VIS] saved plotly images to: {vis_scene_dir}")
            else:
                print("[VIS] vis_backend=mayavi is not implemented in this script. Use --dump_npz and scripts/vis/vis_occ_replica.py.")

        print("num_classes in evaluator:", evaluator.n_classes)

        print("GT unique labels (this scene):", np.unique(scene_data["occ_labels"]))
        print("Pred unique labels (this scene):", torch.unique(pred_occ).cpu())

        evaluator.add_batch(pred_occ, gt_occ)
        n_ok += 1

    metrics = evaluator.get_stats(distributed=False)
    iou_ssc = metrics['iou_ssc']   # Tensor[num_classes]
    print("iou_ssc.shape:", iou_ssc.shape)

    # mIoU over the 8 classes overlapping with ScanNet.
    overlap_8_names = [
        "ceiling",
        "floor",
        "wall",
        "window",
        "chair",
        "bed",
        "sofa",
        "table",
    ]
    overlap_8_labels = _names_to_labels_1based(args.gt_names_path, overlap_8_names)
    overlap_8_labels = [lb for lb in overlap_8_labels if 0 <= lb < int(iou_ssc.numel())]
    if len(overlap_8_labels) > 0:
        overlap_8_iou = iou_ssc[overlap_8_labels]
        miou_overlap8 = overlap_8_iou.mean().item()
    else:
        overlap_8_iou = iou_ssc.new_empty((0,))
        miou_overlap8 = float("nan")

    print("\n==================== Summary ====================")
    print(f"Total scenes: {n_total} | evaluated: {n_ok} | missing ply: {n_missing}")
    print(f"IoU  = {metrics['iou']}")
    print(f"mIoU = {miou_overlap8}")
    print(f"IoU per class: {overlap_8_iou}")

    if args.print_topk:
        # Top-K classes by IoU, excluding free=0.
        iou_ssc_no_free = iou_ssc.clone()
        iou_ssc_no_free[0] = -1
        sorted_vals, _ = iou_ssc_no_free.sort(descending=True)

        for topk in [10, 20, 30, 40]:
            topk_vals = sorted_vals[:topk]
            topk_mean = topk_vals.clamp(min=0).mean().item()
            print(f"mIoU_top{topk} = {topk_mean:.6f}")


if __name__ == "__main__":
    main()
