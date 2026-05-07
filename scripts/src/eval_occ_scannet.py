#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate ScanNet occupancy from Gaussian PLY files and optionally export
visualization assets.

Example:
python scripts/src/eval_occ_scannet.py \
    --exp_path ./outputs/embodied_scannet_all \
    --mode rgbd \
    --scenes scene0416_03 \
    --save_vis \
    --dump_npz \
    --vis_backend plotly

Sequence example:
python scripts/src/eval_occ_scannet.py \
  --exp_path ./outputs/scannet_visualization \
  --mode rgbd \
  --scenes scene0000_00 \
  --scene_occ_root /data/datasets/slam/scannet200/scene_occ \
  --save_vis \
  --dump_npz \
  --dump_npz_all_ply \
  --vis_backend plotly
"""

import os
import argparse
from typing import List, Tuple, Dict, Optional
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


def save_xyz_as_ply(path: str, xyz: np.ndarray, color=None) -> None:
    """Save XYZ points as an ASCII PLY file."""
    xyz = xyz.astype(np.float32)
    n_pts = xyz.shape[0]
    if color is None:
        color = np.full((n_pts, 3), 255, dtype=np.uint8)
    else:
        color = color.astype(np.uint8)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_pts}\n")
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
    scene_data: Dict,
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

# -------------------------
# Optional Mayavi backend (only imported when needed)
# -------------------------


def draw_voxel_mayavi_image(
    voxels: np.ndarray,
    fov_mask: np.ndarray,
    voxel_size: float = 0.05,
    vox_origin: Optional[np.ndarray] = None,
    sem: bool = True,
    save_path: Optional[str] = None,
    offscreen: bool = False,
):
    """Mayavi backend (ported from vis_embodied.draw).

    Notes:
      - keep consistent with vis_embodied.py: only render semantic labels 1..11 (exclude 0 and 12)
      - fov_mask is treated as a boolean valid-mask
    """
    assert voxels.ndim == 3, f"voxels must be [W,H,Z], got {voxels.shape}"

    if vox_origin is None:
        vox_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        vox_origin = np.array(vox_origin, dtype=np.float32).reshape(3)

    # local helper (match vis_embodied.get_grid_coords)
    def _get_grid_coords(dims, resolution):
        g_xx = np.arange(0, dims[0])
        g_yy = np.arange(0, dims[1])
        g_zz = np.arange(0, dims[2])
        xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
        coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T.astype(np.float32)
        resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])
        coords_grid = (coords_grid * resolution) + resolution / 2.0
        return coords_grid

    # Compute voxel centers in world
    grid_coords = _get_grid_coords(list(voxels.shape), [voxel_size] * 3) + vox_origin.reshape(1, 3)
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T  # [N,4] (x,y,z,label)

    # apply valid mask
    fov_flat = fov_mask.reshape(-1).astype(bool)
    fov_grid_coords = grid_coords[fov_flat]

    # Remove empty and unknown (match vis_embodied): label in [1..11]
    fov_voxels = fov_grid_coords[(fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 12)]

    # Lazy import mayavi
    try:
        from mayavi import mlab
    except Exception as e:
        raise ImportError(
            "Mayavi backend requested but 'mayavi' is not available. "
            "Please install mayavi (and vtk/qt) or use --vis_backend plotly."
        ) from e

    mlab.options.offscreen = bool(offscreen)

    figure = mlab.figure(size=(1920, 1080), bgcolor=(1, 1, 1))

    if fov_voxels.shape[0] == 0:
        # still output an empty image for consistency
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            mlab.savefig(save_path, size=(1920, 1080))
        mlab.close(figure)
        return

    if not sem:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0], fov_voxels[:, 1], fov_voxels[:, 2], fov_voxels[:, 3],
            colormap="jet",
            scale_factor=1.0 * float(voxel_size),
            mode="cube",
            opacity=1.0,
        )
    else:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0], fov_voxels[:, 1], fov_voxels[:, 2], fov_voxels[:, 3],
            scale_factor=1.0 * float(voxel_size),
            mode="cube",
            opacity=1.0,
            vmin=1,
            vmax=11,
        )

        # semantic LUT (same palette as vis_embodied)
        lut = np.array(
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
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = lut

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"

    mlab.view(azimuth=180, elevation=0)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mlab.savefig(save_path, size=(1920, 1080))

    mlab.close(figure)


# -------------------------
# scene list helpers
# -------------------------
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


# -------------------------
# geometry helpers
# -------------------------
def get_grid_coords(dims: Tuple[int, int, int], resolution: Tuple[float, float, float]) -> np.ndarray:
    """
    dims: (W,H,Z)
    return: [N,3] centers in local coordinates (origin at 0,0,0)
    """
    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz, indexing="xy")
    coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
    res = np.array(resolution, dtype=np.float32).reshape(1, 3)
    coords = (coords * res) + res / 2.0
    return coords


def _infer_scene_origin_and_size(scene_data: Dict, voxel_size: float, dims_whz: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Infer scene origin/size from scene_data, falling back to origin=0 and dims*voxel_size."""
    origin_keys = ["global_scene_origin", "scene_origin", "vox_origin", "origin"]
    size_keys = ["global_scene_size", "scene_size", "size"]

    origin = None
    for k in origin_keys:
        if k in scene_data:
            origin = scene_data[k]
            break

    size = None
    for k in size_keys:
        if k in scene_data:
            size = scene_data[k]
            break

    if origin is None:
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        origin = np.array(origin, dtype=np.float32).reshape(3)

    if size is None:
        w, h, z = dims_whz
        size = np.array([w * voxel_size, h * voxel_size, z * voxel_size], dtype=np.float32)
    else:
        size = np.array(size, dtype=np.float32).reshape(3)

    return origin, size


def _build_sem_color_table_rgb_float() -> np.ndarray:
    """
    0: empty/free (not rendered)
    1..11: semantic colors
    12: other/unknown (gray)
    return: [13,3] float in [0,1]
    """
    color_table = np.array(
        [
            [0, 0, 0],          # 0 empty
            [214, 38, 40],      # 1
            [43, 160, 4],       # 2
            [158, 216, 229],    # 3
            [114, 158, 206],    # 4
            [204, 204, 91],     # 5
            [255, 186, 119],    # 6
            [147, 102, 188],    # 7
            [30, 119, 181],     # 8
            [160, 188, 33],     # 9
            [255, 127, 12],     # 10
            [196, 175, 214],    # 11
            [128, 128, 128],    # 12 unknown/other
        ],
        dtype=np.float32
    ) / 255.0
    return color_table


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

    color_table = _build_sem_color_table_rgb_float()

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


# -------------------------
# eval helpers
# -------------------------
def get_gt_occ(scene_name: str, scene_occ_root: str, voxel_size: float):
    scene_data = load_full_scene_occ(
        scene_name=scene_name,
        scene_occ_root=scene_occ_root,
        voxel_size=voxel_size,
        to_torch=False,
    )
    _ = extract_gt_occupied_points(scene_data, min_label=0)
    return scene_data


def ensure_pred_shape(pred_occ: torch.Tensor, gt_occ: torch.Tensor) -> torch.Tensor:
    if pred_occ.dim() == gt_occ.dim() - 1:
        pred_occ = pred_occ.unsqueeze(0)
    if pred_occ.shape != gt_occ.shape:
        pred_occ = pred_occ.view(gt_occ.shape)
    return pred_occ


def _pick_valid_mask(scene_data: Dict, fallback_shape: Tuple[int, int, int]) -> np.ndarray:
    """Pick a valid mask from scene_data, falling back to an all-True mask."""
    for k in ["occ_mask_valid", "global_mask", "mask"]:
        if k in scene_data:
            m = scene_data[k]
            m = np.array(m).astype(bool)
            if m.shape == fallback_shape:
                return m
    return np.ones(fallback_shape, dtype=bool)


def _dump_occ_npz(
    save_path: str,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
    valid_mask: np.ndarray,
    voxel_origin: np.ndarray,
    voxel_size: float,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    scene_data: Dict,
    vis_scene_dir: str,
    voxel_size_fallback: float,
    ply_glob: str = "frame_*_*.ply",
):
    """Dump occ npz for EVERY ply in <scene>_<mode>/mesh for incremental visualization.

    Input:
      - PLY folder: <exp_path>/<scene>_<mode>/mesh/<ply_glob>
    Output:
      - NPZ folder: <vis_scene_dir>/npz_seq/occ_<ply_stem>.npz

    This does NOT change evaluation; it's only for incremental visualization playback
    (see scripts/vis/vis_occ_scannet.py).
    """
    import glob

    mesh_dir = os.path.join(exp_path, f"{scene}_{mode}", "mesh")
    ply_paths = sorted(glob.glob(os.path.join(mesh_dir, ply_glob)))

    if len(ply_paths) == 0:
        print(f"[DUMP_SEQ] No ply matched: {os.path.join(mesh_dir, ply_glob)}")
        return

    out_dir = os.path.join(vis_scene_dir, "npz_seq")
    os.makedirs(out_dir, exist_ok=True)

    # origin/voxel_size/valid_mask should match what final occ.npz uses
    gt = np.asarray(scene_data["occ_labels"]).astype(np.int32)
    voxel_origin, _scene_size = _infer_scene_origin_and_size(scene_data, float(voxel_size_fallback), tuple(gt.shape))
    voxel_size = float(scene_data.get("voxel_size", voxel_size_fallback))
    valid_mask = _pick_valid_mask(scene_data, gt.shape)

    print(f"[DUMP_SEQ] dumping {len(ply_paths)} plys -> {out_dir}")

    for i, ply_path in enumerate(ply_paths):
        stem = os.path.splitext(os.path.basename(ply_path))[0]
        out_npz = os.path.join(out_dir, f"occ_{stem}.npz")

        if os.path.exists(out_npz):
            continue

        try:
            g = GaussianModel(sh_degree=0)
            g.load_ply(ply_path)

            ov_feat = getattr(g, "ov_feat", None)
            pred_occ = gaussians_to_occ(
                g.get_xyz,
                g.get_features.squeeze(1),
                g.get_scaling,
                quaternion_to_matrix(g.get_rotation),
                g.get_opacity,
                ov_feat,
                scene_data,
            )

            if not torch.is_tensor(pred_occ):
                raise TypeError(f"gaussians_to_occ should return torch.Tensor, got {type(pred_occ)}")

            pred_np = pred_occ.squeeze(0).detach().cpu().numpy() if pred_occ.dim() == 4 else pred_occ.detach().cpu().numpy()

            _dump_occ_npz(
                save_path=out_npz,
                gt_np=gt,
                pred_np=pred_np,
                valid_mask=valid_mask,
                voxel_origin=voxel_origin,
                voxel_size=voxel_size,
            )

            if (i % 10) == 0:
                print(f"[DUMP_SEQ] [{i+1}/{len(ply_paths)}] saved: {out_npz}")
        except Exception as e:
            print(f"[DUMP_SEQ][WARN] failed: {ply_path} -> {type(e).__name__}: {e}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Evaluate multi-class occupancy metrics from Gaussian PLY + save GT/Pred visualization (plotly/mayavi)")

    parser.add_argument("--exp_path", type=str, required=True, help="experiment root path")
    parser.add_argument("--mode", type=str, default="mono", help="e.g. mono / stereo ...")

    parser.add_argument("--scenes", type=str, default="", help="scene0000_00,scene0001_00 ...")
    parser.add_argument("--scenes_txt", type=str, default="", help="txt file with one scene per line")

    parser.add_argument("--scene_occ_root", type=str, default="/data/datasets/slam/scannet200/scene_occ")
    parser.add_argument("--voxel_size", type=float, default=0.08)

    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")

    # vis
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--vis_dir", type=str, default="", help="default: <exp_path>/occ_vis")
    parser.add_argument("--vis_d", type=float, default=3.0, help="camera distance multiplier (like your d=3)")
    parser.add_argument("--vis_scene_size_pad", type=float, default=1.0, help="scene_size + pad (like scene_size+1)")
    parser.add_argument("--vis_max_points", type=int, default=-1, help="<=0 keep all cubes; >0 subsample cubes")

    # NEW: dump npz for external mayavi visualization (recommended when mayavi/qt conflicts)
    parser.add_argument(
        "--dump_npz",
        action="store_true",
        help="dump gt/pred/valid_mask/voxel_origin/voxel_size to <vis_dir>/<scene>_<mode>/occ.npz",
    )

    # NEW: select visualization backend
    parser.add_argument(
        "--vis_backend",
        type=str,
        default="mayavi",
        choices=["mayavi", "plotly"],
        help="visualization backend (default: mayavi to match vis_embodied)",
    )

    # NEW: dump occ npz for EVERY ply in <scene>_<mode>/mesh (e.g. frame_*.ply), for incremental occ visualization
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

    args = parser.parse_args()

    if args.save_vis and args.vis_backend == "plotly" and go is None:
        raise ImportError("Plotly backend requested but plotly is not available. Please install plotly+kaleido or use --vis_backend mayavi")

    if args.scenes_txt:
        scenes = read_scenes_txt(args.scenes_txt)
    else:
        scenes = parse_scenes(args.scenes)

    if len(scenes) == 0:
        raise ValueError("No scenes provided. Use --scenes or --scenes_txt")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    evaluator = SSCMetricsTorch(args.num_classes)

    if args.save_vis or args.dump_npz or args.dump_npz_all_ply:
        vis_root = args.vis_dir if args.vis_dir else os.path.join(args.exp_path, "occ_vis")
        os.makedirs(vis_root, exist_ok=True)
    else:
        vis_root = None

    max_points = None if (args.vis_max_points is None or args.vis_max_points <= 0) else int(args.vis_max_points)

    n_total, n_ok, n_missing = 0, 0, 0

    for scene in scenes:
        n_total += 1
        print(f"\n[Eval] Scene: {scene}")

        scene_data = get_gt_occ(scene, args.scene_occ_root, args.voxel_size)

        scene_path = os.path.join(args.exp_path, f"{scene}_{args.mode}", "mesh", f"final_{args.mode}.ply")
        if not os.path.exists(scene_path):
            print(f"[Warn] Missing ply: {scene_path}  -> skip")
            n_missing += 1
            continue

        # ----- optional: dump npz for ALL incremental plys (sequence) -----
        if args.dump_npz_all_ply:
            vis_scene_dir = os.path.join(vis_root, f"{scene}_{args.mode}")
            os.makedirs(vis_scene_dir, exist_ok=True)

            dump_npz_for_all_plys_in_scene(
                scene=scene,
                mode=args.mode,
                exp_path=args.exp_path,
                scene_data=scene_data,
                vis_scene_dir=vis_scene_dir,
                voxel_size_fallback=float(args.voxel_size),
                ply_glob=str(args.ply_glob),
            )

        g = GaussianModel(sh_degree=0)
        g.load_ply(scene_path)

        ov_feat = getattr(g, "ov_feat", None)
        pred_occ = gaussians_to_occ(
            g.get_xyz,
            g.get_features.squeeze(1),
            g.get_scaling,
            quaternion_to_matrix(g.get_rotation),
            g.get_opacity,
            ov_feat,
            scene_data,
        )

        gt_occ = torch.from_numpy(scene_data["occ_labels"]).to(device=device).long().unsqueeze(0)

        if torch.is_tensor(pred_occ):
            pred_occ = pred_occ.to(device=device).long()
        else:
            raise TypeError(f"gaussians_to_occ should return torch.Tensor, got {type(pred_occ)}")

        pred_occ = ensure_pred_shape(pred_occ, gt_occ)

        evaluator.add_batch(pred_occ, gt_occ)
        n_ok += 1

        if args.dump_debug_points:
            debug_points_dir = args.debug_points_dir if args.debug_points_dir else os.path.join(args.exp_path, "occ_debug_points")
            dump_debug_occupied_points(debug_points_dir, scene, args.mode, scene_data, gt_occ, pred_occ)

        if args.dump_npz or args.save_vis:
            gt_np = gt_occ[0].detach().cpu().numpy().astype(np.int32)
            pred_np = pred_occ[0].detach().cpu().numpy().astype(np.int32)

            dims = tuple(gt_np.shape)  # (W,H,Z)
            voxel_origin, scene_size = _infer_scene_origin_and_size(scene_data, args.voxel_size, dims)
            scene_size = scene_size + float(args.vis_scene_size_pad)

            valid_mask = _pick_valid_mask(scene_data, gt_np.shape)

            out_dir = os.path.join(vis_root, f"{scene}_{args.mode}")
            os.makedirs(out_dir, exist_ok=True)

            if args.dump_npz:
                npz_path = os.path.join(out_dir, "occ.npz")
                _dump_occ_npz(
                    npz_path,
                    gt_np=gt_np,
                    pred_np=pred_np,
                    valid_mask=valid_mask,
                    voxel_origin=voxel_origin,
                    voxel_size=float(args.voxel_size),
                )
                print(f"[Dump] saved: {npz_path}")

        if args.save_vis:
            save_path = os.path.join(out_dir, "gt.png")
            if args.vis_backend == "plotly":
                draw_voxel_plotly_image(
                    gt_np,
                    valid_mask,
                    voxel_size=args.voxel_size,
                    vox_origin=voxel_origin,
                    scene_size=scene_size,
                    intrinsic=None,
                    d=float(args.vis_d),
                    save_path=save_path,
                    max_points=max_points,
                )
            else:
                draw_voxel_mayavi_image(
                    gt_np,
                    valid_mask,
                    voxel_size=float(args.voxel_size),
                    vox_origin=voxel_origin,
                    sem=True,
                    save_path=save_path,
                    offscreen=False,
                )

            save_path = os.path.join(out_dir, "pred.png")
            if args.vis_backend == "plotly":
                draw_voxel_plotly_image(
                    pred_np,
                    valid_mask,
                    voxel_size=args.voxel_size,
                    vox_origin=voxel_origin,
                    scene_size=scene_size,
                    intrinsic=None,
                    d=float(args.vis_d),
                    save_path=save_path,
                    max_points=max_points,
                )
            else:
                draw_voxel_mayavi_image(
                    pred_np,
                    valid_mask,
                    voxel_size=float(args.voxel_size),
                    vox_origin=voxel_origin,
                    sem=True,
                    save_path=save_path,
                    offscreen=False,
                )

            print(f"[Vis] ({args.vis_backend}) saved to: {out_dir}/gt.png and {out_dir}/pred.png")

    metrics = evaluator.get_stats(distributed=False)
    print("\n==================== Summary ====================")
    print(f"Total scenes: {n_total} | evaluated: {n_ok} | missing ply: {n_missing}")
    print(f"IoU  = {metrics['iou']}")
    print(f"mIoU = {metrics['iou_ssc'][1:].mean()}")
    print(f"IoU per class: {metrics['iou_ssc']}")


if __name__ == "__main__":
    main()
