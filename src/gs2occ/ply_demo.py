#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import open3d as o3d

# marching cubes
try:
    from skimage import measure
except Exception as e:
    raise ImportError("Missing dependency skimage. Install via: pip install scikit-image") from e

# 读 3DGS ply
try:
    from plyfile import PlyData
except Exception as e:
    raise ImportError("Missing dependency plyfile. Install via: pip install plyfile") from e

# # LocalAggregator (你那个 NameError 就是这里没导入到)
# try:
#     # 当以项目根目录运行（benchmark_scene.py 在 leo-slam 下）
#     from gs2occ.localagg_prob.local_aggregate_prob import LocalAggregator
# except ModuleNotFoundError:
#     # 当你 cd 到 gs2occ/ 目录下直接跑 ply_demo.py
#     from localagg_prob.local_aggregate_prob import LocalAggregator

from .localagg_prob.local_aggregate_prob import LocalAggregator


# -----------------------------
# Utils: seed
# -----------------------------
def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# IO: point cloud (.ply/.pcd/...)
# -----------------------------
def load_point_cloud(path: str):
    """
    支持读取 .ply / .pcd / ... 点云，返回:
        points: [N,3] torch.float32 (CPU)
        colors: [N,3] torch.float32 (CPU, 0~1) or None
    """
    path = str(path)
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Failed to read point cloud or empty: {path}")

    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected [N,3], got {pts.shape}")

    cols = None
    if len(pcd.colors) > 0:
        cols_np = np.asarray(pcd.colors, dtype=np.float32)
        if cols_np.shape == pts.shape:
            cols = cols_np
        else:
            print(f"[WARN] colors shape mismatch: {cols_np.shape} vs {pts.shape}, ignore colors.")

    return torch.from_numpy(pts), (torch.from_numpy(cols) if cols is not None else None)


# -----------------------------
# Downsample
# -----------------------------
def downsample_points(
    points: torch.Tensor,
    colors: torch.Tensor | None,
    method: str,
    voxel_size: float,
    max_points: int,
    seed: int,
):
    """
    points/colors on CPU.
    method:
      - none
      - voxel  (Open3D voxel_down_sample)
      - random (uniform random)
    """
    assert points.device.type == "cpu"

    if method == "none":
        if max_points > 0 and points.shape[0] > max_points:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(points.shape[0], generator=g)[:max_points]
            points = points[idx]
            colors = colors[idx] if colors is not None else None
        return points, colors

    if method == "random":
        if max_points <= 0:
            return points, colors
        if points.shape[0] <= max_points:
            return points, colors
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(points.shape[0], generator=g)[:max_points]
        return points[idx], (colors[idx] if colors is not None else None)

    if method == "voxel":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.numpy().astype(np.float64))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors.numpy().astype(np.float64))

        pcd_ds = pcd.voxel_down_sample(voxel_size=float(voxel_size))
        pts = np.asarray(pcd_ds.points, dtype=np.float32)
        cols = np.asarray(pcd_ds.colors, dtype=np.float32) if len(pcd_ds.colors) > 0 else None

        points_ds = torch.from_numpy(pts)
        colors_ds = torch.from_numpy(cols) if cols is not None and cols.shape == pts.shape else None

        # 再做一个 max_points 限制（防止 voxel 后还是太大）
        if max_points > 0 and points_ds.shape[0] > max_points:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(points_ds.shape[0], generator=g)[:max_points]
            points_ds = points_ds[idx]
            colors_ds = colors_ds[idx] if colors_ds is not None else None

        return points_ds, colors_ds

    raise ValueError(f"Unknown downsample method: {method}")


def downsample_gaussians(
    means3D, scales, cov3D, opas, semantics,
    max_gaussians: int, seed: int,
    use_topk_opacity: bool = True,
):
    if max_gaussians <= 0 or means3D.shape[0] <= max_gaussians:
        return means3D, scales, cov3D, opas, semantics

    if use_topk_opacity:
        # ✅ 按 opacity 取最“可信”的高斯，避免 floaters
        score = opas.squeeze(-1) if opas.ndim == 2 else opas
        idx = torch.topk(score, k=max_gaussians, largest=True).indices
        return means3D[idx], scales[idx], cov3D[idx], opas[idx], semantics[idx]

    # fallback：随机（不推荐）
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    idx_cpu = torch.randperm(means3D.shape[0], generator=g, device="cpu")[:max_gaussians]
    idx = idx_cpu.to(device=means3D.device)
    return means3D[idx], scales[idx], cov3D[idx], opas[idx], semantics[idx]




# -----------------------------
# Grid builder
# -----------------------------
def build_grid_from_bbox(pc_min: torch.Tensor, pc_max: torch.Tensor, grid_size: float):
    """
    返回:
      pts_grid: [H*W*D, 3] (角点坐标)
      H,W,D
    """
    device = pc_min.device
    bbox_size = pc_max - pc_min
    dims = torch.ceil(bbox_size / grid_size).to(torch.long) + 1
    H, W, D = dims.tolist()

    xs = torch.arange(H, device=device, dtype=torch.float32)
    ys = torch.arange(W, device=device, dtype=torch.float32)
    zs = torch.arange(D, device=device, dtype=torch.float32)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    pts_grid = torch.stack([X, Y, Z], dim=-1) * grid_size + pc_min[None, None, :]

    pts_grid = pts_grid.reshape(-1, 3)
    return pts_grid, H, W, D


# -----------------------------
# PointCloud -> “fake gaussians”
# -----------------------------
def build_gaussians_from_points(points: torch.Tensor, base_scale: torch.Tensor, rot_mat: torch.Tensor, colors=None):
    N = points.shape[0]
    device = points.device

    means3D = points.clone()
    scales = base_scale[None, :].to(device).expand(N, 3)

    S = torch.diag(base_scale.to(device) ** 2)
    cov_single = rot_mat.to(device) @ S @ rot_mat.to(device).T
    cov3D = cov_single[None, :, :].expand(N, 3, 3).contiguous()

    opas = torch.ones(N, 1, device=device, dtype=torch.float32)

    if colors is not None:
        semantics = colors.to(device=device, dtype=torch.float32)
        if semantics.ndim != 2 or semantics.shape[1] != 3:
            semantics = torch.ones(N, 1, device=device, dtype=torch.float32)
    else:
        semantics = torch.ones(N, 1, device=device, dtype=torch.float32)

    return means3D, scales, cov3D, opas, semantics


# -----------------------------
# 3DGS PLY -> gaussians
# -----------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def _quat_to_rotmat(q: torch.Tensor, order="wxyz"):
    """
    q: [N,4]
    order: wxyz or xyzw
    """
    if order == "wxyz":
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif order == "xyzw":
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        raise ValueError("quat_order must be 'wxyz' or 'xyzw'")

    # normalize
    norm = torch.sqrt(w*w + x*x + y*y + z*z + 1e-12)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # rot matrices
    R = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=torch.float32)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def load_3dgs_ply_as_gaussians(
    ply_path: str,
    device: torch.device,
    max_gaussians: int,
    seed: int,
    scale_is_log: bool = True,
    opacity_is_logit: bool = True,
    quat_order: str = "wxyz",
):
    """
    读取常见 3DGS / Photo-SLAM 的 point_cloud.ply（vertex 里含 x y z, scale_*, rot_*, opacity 等）
    返回:
      means3D:  [N,3]
      scales:   [N,3]  (positive)
      cov3D:    [N,3,3]
      opas:     [N,1]  (0~1)
      semantics:[N,C]  (尽量给 3 通道)
    """
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    names = v.dtype.names

    def need(keys):
        for k in keys:
            if k not in names:
                return False
        return True

    # xyz
    if not need(("x", "y", "z")):
        raise ValueError(f"PLY vertex missing xyz fields. got names={names}")
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # scales
    # 常见字段: scale_0 scale_1 scale_2 或 scaling_0...
    scale_keys = None
    for cand in [("scale_0", "scale_1", "scale_2"), ("scaling_0", "scaling_1", "scaling_2")]:
        if need(cand):
            scale_keys = cand
            break
    if scale_keys is None:
        raise ValueError(f"PLY missing scale fields. tried scale_*/scaling_*. got names={names}")
    scl = np.stack([v[scale_keys[0]], v[scale_keys[1]], v[scale_keys[2]]], axis=1).astype(np.float32)

    # rotation quaternion
    # 常见字段: rot_0 rot_1 rot_2 rot_3 或 rotation_0...
    rot_keys = None
    for cand in [("rot_0","rot_1","rot_2","rot_3"), ("rotation_0","rotation_1","rotation_2","rotation_3")]:
        if need(cand):
            rot_keys = cand
            break
    if rot_keys is None:
        raise ValueError(f"PLY missing rotation quaternion fields. tried rot_*/rotation_*. got names={names}")
    quat = np.stack([v[rot_keys[0]], v[rot_keys[1]], v[rot_keys[2]], v[rot_keys[3]]], axis=1).astype(np.float32)

    # opacity
    opa_key = None
    for cand in ["opacity", "opacities", "alpha"]:
        if cand in names:
            opa_key = cand
            break
    if opa_key is None:
        raise ValueError(f"PLY missing opacity field. tried opacity/opacities/alpha. got names={names}")
    opa = np.asarray(v[opa_key]).astype(np.float32).reshape(-1, 1)

    # semantics / color: 优先 rgb，否则 features_dc_*
    sem = None
    if need(("red","green","blue")):
        sem = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)
        # 若是 0~255
        if sem.max() > 1.5:
            sem = sem / 255.0
    else:
        # Photo-SLAM/3DGS 常见: f_dc_0..2 或 features_dc_0..2
        for cand in [("f_dc_0","f_dc_1","f_dc_2"), ("features_dc_0","features_dc_1","features_dc_2")]:
            if need(cand):
                sem = np.stack([v[cand[0]], v[cand[1]], v[cand[2]]], axis=1).astype(np.float32)
                # 这通常是 SH 的 DC 系数，不一定在 0~1，这里先 sigmoid 到可视化范围
                sem = 1.0 / (1.0 + np.exp(-sem))
                break
    if sem is None:
        sem = np.ones((xyz.shape[0], 1), dtype=np.float32)

    means3D = torch.from_numpy(xyz).to(device=device)
    scales_raw = torch.from_numpy(scl).to(device=device)
    quat = torch.from_numpy(quat).to(device=device)
    opas_raw = torch.from_numpy(opa).to(device=device)
    semantics = torch.from_numpy(sem).to(device=device)

    # decode
    if scale_is_log:
        scales = torch.exp(scales_raw).clamp(min=1e-6)
    else:
        scales = scales_raw.clamp(min=1e-6)

    if opacity_is_logit:
        opas = _sigmoid(opas_raw)
    else:
        opas = opas_raw.clamp(0.0, 1.0)

    R = _quat_to_rotmat(quat, order=quat_order)
    cov3D = R @ torch.diag_embed(scales**2) @ R.transpose(1, 2)

    means3D, scales, cov3D, opas, semantics = downsample_gaussians(
        means3D, scales, cov3D, opas, semantics,
        max_gaussians=max_gaussians, seed=seed
    )
    return means3D, scales, cov3D, opas, semantics


# -----------------------------
# Occupancy export: voxel points
# -----------------------------
def save_occ_as_voxel_ply(density, pc_min, grid_size, out_ply_path, thr=0.2, sem_grid=None):
    if isinstance(density, torch.Tensor):
        density_np = density.detach().cpu().numpy()
    else:
        density_np = np.asarray(density)

    if isinstance(pc_min, torch.Tensor):
        pc_min_np = pc_min.detach().cpu().numpy().astype(np.float64)
    else:
        pc_min_np = np.asarray(pc_min, dtype=np.float64)

    assert density_np.ndim == 3, f"density must be [H,W,D], got {density_np.shape}"
    H, W, D = density_np.shape

    occ_mask = density_np > thr
    if not np.any(occ_mask):
        print("[WARN] no voxel above threshold, nothing to save.")
        return

    idxs = np.argwhere(occ_mask)
    centers = pc_min_np[None, :] + (idxs.astype(np.float64) + 0.5) * float(grid_size)

    if sem_grid is not None:
        sem_np = sem_grid.detach().cpu().numpy() if isinstance(sem_grid, torch.Tensor) else np.asarray(sem_grid)
        assert sem_np.shape[:3] == density_np.shape
        C = sem_np.shape[3]
        sem_values = sem_np[occ_mask]
        if C < 3:
            if C == 1:
                sem_values = np.repeat(sem_values, 3, axis=1)
            elif C == 2:
                sem_values = np.concatenate([sem_values, sem_values[:, :1]], axis=1)
        else:
            sem_values = sem_values[:, :3]
        colors = np.clip(sem_values, 0.0, 1.0).astype(np.float64)
    else:
        occ_values = density_np[occ_mask]
        m = occ_values.max()
        occ_norm = (occ_values / (m + 1e-6)).astype(np.float64)
        colors = np.stack([occ_norm, occ_norm, occ_norm], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    os.makedirs(str(Path(out_ply_path).parent), exist_ok=True)
    o3d.io.write_point_cloud(out_ply_path, pcd, write_ascii=False)
    print(f"[INFO] Saved voxel occupancy point cloud to: {out_ply_path}")


# -----------------------------
# Visualization: marching cubes mesh
# -----------------------------
def visualize_occ_as_mesh(density, pc_min, grid_size, level=0.2):
    vol = density.detach().cpu().numpy() if isinstance(density, torch.Tensor) else np.asarray(density)
    origin = pc_min.detach().cpu().numpy() if isinstance(pc_min, torch.Tensor) else np.asarray(pc_min)

    vol = vol.astype(np.float32)

    if vol.max() < level:
        print(f"[WARN] volume max {vol.max():.4f} < level {level:.4f}. try smaller --mc-level / --thr")
        return

    verts, faces, normals, _ = measure.marching_cubes(
        vol, level=float(level), spacing=(grid_size, grid_size, grid_size)
    )
    verts = verts + origin[None, :]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # 灰度：按 occ 值上色
    v = (verts - origin[None, :]) / grid_size
    vi = np.round(v).astype(np.int32)
    vi[:, 0] = np.clip(vi[:, 0], 0, vol.shape[0] - 1)
    vi[:, 1] = np.clip(vi[:, 1], 0, vol.shape[1] - 1)
    vi[:, 2] = np.clip(vi[:, 2], 0, vol.shape[2] - 1)
    occ_val = vol[vi[:, 0], vi[:, 1], vi[:, 2]]
    occ_val = occ_val / (occ_val.max() + 1e-6)
    colors = np.stack([occ_val] * 3, axis=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([mesh], window_name="OCC (Marching Cubes)")


# -----------------------------
# GS -> OCC “rebuild” mode (关键：解决薄膜)
# -----------------------------
def rebuild_gaussians_for_occ(
    means3D: torch.Tensor,
    semantics: torch.Tensor,
    fixed_scale: float,
    fixed_opacity: float,
):
    """
    把 3DGS 的高斯“重置”为几何占据友好的高斯：
      - scale 统一为 fixed_scale
      - opacity 统一为 fixed_opacity
      - rotation 用单位阵（cov = diag(scale^2)）
    """
    device = means3D.device
    N = means3D.shape[0]

    scales = torch.full((N, 3), float(fixed_scale), device=device, dtype=torch.float32)
    cov3D = torch.diag_embed(scales**2)
    opas = torch.full((N, 1), float(fixed_opacity), device=device, dtype=torch.float32)

    if semantics is None:
        semantics = torch.ones((N, 1), device=device, dtype=torch.float32)

    return means3D, scales, cov3D, opas, semantics

def bbox_by_quantile(xyz: torch.Tensor, q: float = 0.01):
    # xyz: [N,3] on GPU/CPU都行
    lo = torch.quantile(xyz, q, dim=0)
    hi = torch.quantile(xyz, 1.0 - q, dim=0)
    return lo, hi

def mask_in_bbox(xyz: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
    m = (xyz[:,0] >= lo[0]) & (xyz[:,0] <= hi[0]) & \
        (xyz[:,1] >= lo[1]) & (xyz[:,1] <= hi[1]) & \
        (xyz[:,2] >= lo[2]) & (xyz[:,2] <= hi[2])
    return m


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True, help="输入点云(.ply/.pcd/...) 或 3DGS PLY")
    parser.add_argument("--input-type", type=str, choices=["pc", "gs"], required=True, help="pc=点云, gs=3DGS ply")

    # seed
    parser.add_argument("--seed", type=int, default=0, help="随机种子")

    # downsample for point cloud
    parser.add_argument("--ds", type=str, default="voxel", choices=["none", "voxel", "random"], help="点云降采样方式")
    parser.add_argument("--ds-voxel", type=float, default=0.02, help="voxel downsample size (meters)")
    parser.add_argument("--max-points", type=int, default=500000, help="点云最大点数(<=0不限制)")

    # downsample for gaussians
    parser.add_argument("--max-gaussians", type=int, default=30000, help="最多保留多少个高斯(<=0不限制)")

    # GS->OCC mode (关键)
    parser.add_argument("--gs-occ-mode", type=str, default="rebuild", choices=["raw", "rebuild"],
                        help="raw=用3DGS原始scale/opacity(可能薄膜), rebuild=重置为几何占据高斯(推荐)")
    parser.add_argument("--gs-fixed-scale", type=float, default=0.03, help="rebuild模式：统一scale (meters)")
    parser.add_argument("--gs-fixed-opacity", type=float, default=1.0, help="rebuild模式：统一opacity(0~1)")

    # occ grid
    parser.add_argument("--grid-size", type=float, default=0.05, help="体素大小(m)")
    parser.add_argument("--scale-multiplier", type=float, default=3.0, help="LocalAggregator scale_multiplier")
    parser.add_argument("--radii-min", type=int, default=1, help="LocalAggregator radii_min (>=1)")

    # output / visualization
    parser.add_argument("--out", type=str, default="output/occ", help="输出前缀(会生成 *_vox.ply)")
    parser.add_argument("--thr", type=float, default=0.2, help="occ阈值(保存/可视化)")
    parser.add_argument("--vis", action="store_true", help="是否 marching-cubes 可视化")
    parser.add_argument("--mc-level", type=float, default=0.2, help="marching cubes level(建议≈thr)")

    args = parser.parse_args()
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -------- A. Load input -> gaussians --------
    if args.input_type == "pc":
        points, colors = load_point_cloud(args.input)
        print(f"[INFO] Raw points: {points.shape[0]}")

        points, colors = downsample_points(
            points, colors,
            method=args.ds,
            voxel_size=args.ds_voxel,
            max_points=args.max_points,
            seed=args.seed,
        )
        print(f"[INFO] After downsample ({args.ds}): {points.shape[0]}")

        points = points.to(device)
        colors = colors.to(device) if colors is not None else None

        pc_min = points.min(dim=0).values
        pc_max = points.max(dim=0).values
        print(f"[INFO] bbox min: {pc_min.detach().cpu().numpy()}, max: {pc_max.detach().cpu().numpy()}")

        # 伪造“几何高斯”
        BASE_SCALE = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32)
        ROT_MAT = torch.eye(3, dtype=torch.float32)
        means3D, scales, cov3D, opas, semantics = build_gaussians_from_points(points, BASE_SCALE, ROT_MAT, colors=colors)

    else:
        means3D, scales, cov3D, opas, semantics = load_3dgs_ply_as_gaussians(
            args.input,
            device=device,
            max_gaussians=args.max_gaussians,
            seed=args.seed,
            scale_is_log=True,
            opacity_is_logit=True,
            quat_order="wxyz",
        )
        print(f"[INFO] Loaded gaussians: {means3D.shape[0]}")

        lo, hi = bbox_by_quantile(means3D, q=0.01)   # 1%~99% bbox，去离群点
        m = mask_in_bbox(means3D, lo, hi)

        means3D   = means3D[m]
        scales    = scales[m]
        cov3D     = cov3D[m]
        opas      = opas[m]
        semantics = semantics[m]

        print(f"[INFO] After quantile bbox filter: {means3D.shape[0]} gaussians kept")
        pc_min, pc_max = lo, hi

        # 关键：解决“薄膜”
        if args.gs_occ_mode == "rebuild":
            means3D, scales, cov3D, opas, semantics = rebuild_gaussians_for_occ(
                means3D, semantics,
                fixed_scale=args.gs_fixed_scale,
                fixed_opacity=args.gs_fixed_opacity,
            )
            print(f"[INFO] GS occ-mode=rebuild, fixed_scale={args.gs_fixed_scale}, fixed_opacity={args.gs_fixed_opacity}")
        else:
            print("[INFO] GS occ-mode=raw (may look like thin surfaces)")

        pc_min = means3D.min(dim=0).values
        pc_max = means3D.max(dim=0).values
        print(f"[INFO] bbox min: {pc_min.detach().cpu().numpy()}, max: {pc_max.detach().cpu().numpy()}")

    # -------- B. Build query grid --------
    pts_grid, H, W, D = build_grid_from_bbox(pc_min, pc_max, args.grid_size)
    pts_grid = pts_grid.to(device)
    print(f"[INFO] Grid dims HxWxD = {H} x {W} x {D} (total {H*W*D} voxels)")

    # -------- C. LocalAggregator --------
    agg = LocalAggregator(
        scale_multiplier=args.scale_multiplier,
        H=H, W=W, D=D,
        pc_min=pc_min.detach().cpu().tolist(),
        grid_size=args.grid_size,
        radii_min=args.radii_min,
    ).to(device)

    pts_b       = pts_grid.unsqueeze(0)
    means3D_b   = means3D.unsqueeze(0)
    scales_b    = scales.unsqueeze(0)
    cov3D_b     = cov3D.unsqueeze(0)
    opas_b      = opas.unsqueeze(0)
    semantics_b = semantics.unsqueeze(0)

    origin_use = pc_min.to(device)

    with torch.no_grad():
        logits, bin_logits, density = agg(
            pts=pts_b,
            means3D=means3D_b,
            opas=opas_b,
            semantics=semantics_b,
            scales=scales_b,
            cov3D=cov3D_b,
            metas=None,
            origin_use=origin_use,
        )

    # density: [H*W*D] (见你原本的 reshape 流程 :contentReference[oaicite:1]{index=1})
    density_np = density.reshape(H, W, D).detach().cpu().numpy()
    occ_np = 1.0 - np.exp(-density_np)  # 你后来加的这步是对的（occupancy from density）

    print(f"[INFO] density min={density_np.min():.6f}, max={density_np.max():.6f}")
    print(f"[INFO] occ     min={occ_np.min():.6f}, max={occ_np.max():.6f}")

    # -------- D. Visualization (mesh) --------
    if args.vis:
        visualize_occ_as_mesh(
            density=occ_np,
            pc_min=pc_min.detach().cpu(),
            grid_size=args.grid_size,
            level=args.mc_level,
        )

    # -------- E. Save voxel occupancy points --------
    out_ply = args.out + "_vox.ply"
    save_occ_as_voxel_ply(
        density=occ_np,
        pc_min=pc_min.detach().cpu(),
        grid_size=args.grid_size,
        out_ply_path=out_ply,
        thr=args.thr,
        sem_grid=None,  # 先不强行上语义，避免“全黑/不可解释”问题
    )
    print("[DONE]")


if __name__ == "__main__":
    main()
