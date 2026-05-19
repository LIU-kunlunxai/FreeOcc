#!/usr/bin/env python3
"""
从 FreeOcc 生成的 PLY 构建语义占用网格，支持 XY 平面俯视图。

用法:
    python scripts/occ_from_ply.py \
        --input /path/to/final_rgbd.ply \
        --output /path/to/output_dir \
        --grid-size 0.1 \
        --thr 0.2

输出:
    - occ_voxel.ply         3D 占用体素点云 (RGB 着色)
    - occ_sem_topdown.png   XY 平面俯视图 (语义着色)
    - occ_density_zmax.png  XY 平面俯视图 (按密度着色)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
from plyfile import PlyData

# 添加项目路径
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "thirdparty", "Trident"))

from src.gs2occ.localagg_prob.local_aggregate_prob import LocalAggregator


def _sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def load_ply(ply_path: str, device: torch.device, max_gaussians: int = 0):
    """读取 FreeOcc 的 PLY，提取 3D 高斯参数 + 语义特征."""
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    names = v.dtype.names
    n = len(v)

    print(f"[INFO] PLY: {n} vertices, fields: {names}")

    # xyz
    xyz = torch.from_numpy(
        np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    ).to(device)

    # opacity (logit -> probability)
    opa_raw = torch.from_numpy(v["opacity"].astype(np.float32)).to(device)
    if opa_raw.ndim == 1:
        opa_raw = opa_raw.unsqueeze(-1)
    opas = _sigmoid(opa_raw)

    # scales (log scale -> positive)
    scl = torch.from_numpy(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    ).to(device)
    scales = torch.exp(scl).clamp(min=1e-6)

    # rotation quaternion (wxyz) -> rot matrix
    quat = torch.from_numpy(
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    ).to(device)
    w, x, y, zz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    norm = torch.sqrt(w*w + x*x + y*y + zz*zz + 1e-12)
    w, x, y, zz = w/norm, x/norm, y/norm, zz/norm
    R = torch.zeros((quat.shape[0], 3, 3), device=device)
    R[:, 0, 0] = 1 - 2*(y*y + zz*zz); R[:, 0, 1] = 2*(x*y - zz*w); R[:, 0, 2] = 2*(x*zz + y*w)
    R[:, 1, 0] = 2*(x*y + zz*w); R[:, 1, 1] = 1 - 2*(x*x + zz*zz); R[:, 1, 2] = 2*(y*zz - x*w)
    R[:, 2, 0] = 2*(x*zz - y*w); R[:, 2, 1] = 2*(y*zz + x*w); R[:, 2, 2] = 1 - 2*(x*x + y*y)

    # covariance
    cov3D = R @ torch.diag_embed(scales**2) @ R.transpose(1, 2)

    # RGB (f_dc)
    colors = torch.from_numpy(
        np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    ).to(device)
    colors = _sigmoid(colors)

    # Open-vocabulary features
    ov_feat_names = sorted([k for k in names if k.startswith("ov_feat_")],
                           key=lambda x: int(x.split("_")[-1]))
    if ov_feat_names:
        ov_data = np.stack([v[k] for k in ov_feat_names], axis=1).astype(np.float32)
        semantics = torch.from_numpy(ov_data).to(device)
        print(f"[INFO] OV features: {len(ov_feat_names)} dims")
    else:
        semantics = colors
        print("[INFO] No OV features, using RGB as semantics")

    # 用 top-K opacity 筛选
    if max_gaussians > 0 and n > max_gaussians:
        score = opas.squeeze(-1)
        idx = torch.topk(score, k=max_gaussians, largest=True).indices
        xyz, scales, cov3D, opas, colors, semantics = \
            xyz[idx], scales[idx], cov3D[idx], opas[idx], colors[idx], semantics[idx]
        print(f"[INFO] Downsampled to {max_gaussians} gaussians")

    return xyz, scales, cov3D, opas, colors, semantics


def build_grid(lo: torch.Tensor, hi: torch.Tensor, grid_size: float, device):
    """在 bbox 内生成体素网格查询点."""
    bsz = hi - lo
    H, W, D = [int((bsz[i] / grid_size).ceil().item()) + 1 for i in range(3)]

    xs = torch.arange(H, device=device, dtype=torch.float32)
    ys = torch.arange(W, device=device, dtype=torch.float32)
    zs = torch.arange(D, device=device, dtype=torch.float32)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    pts = torch.stack([X, Y, Z], dim=-1) * grid_size + lo[None, None, None, :]
    return pts.reshape(-1, 3), H, W, D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="FreeOcc PLY 路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--grid-size", type=float, default=0.1, help="体素大小 (m)")
    parser.add_argument("--thr", type=float, default=0.2, help="占用阈值")
    parser.add_argument("--max-gaussians", type=int, default=200000, help="最多高斯数")
    parser.add_argument("--scale-multiplier", type=float, default=3.0)
    parser.add_argument("--radii-min", type=int, default=1)
    parser.add_argument("--bbox-margin", type=float, default=0.5, help="bbox 扩展边距 (m)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── 1. 加载 ──
    xyz, scales, cov3D, opas, colors, semantics = load_ply(
        args.input, device, args.max_gaussians
    )
    N = xyz.shape[0]

    # 过滤离群点
    lo_q = torch.quantile(xyz, 0.01, dim=0)
    hi_q = torch.quantile(xyz, 0.99, dim=0)
    margin = torch.tensor([args.bbox_margin] * 3, device=device)
    lo, hi = lo_q - margin, hi_q + margin
    print(f"[INFO] BBox: [{lo.cpu().numpy()} → {hi.cpu().numpy()}]")

    m = (xyz[:,0] >= lo[0]) & (xyz[:,0] <= hi[0]) & \
        (xyz[:,1] >= lo[1]) & (xyz[:,1] <= hi[1]) & \
        (xyz[:,2] >= lo[2]) & (xyz[:,2] <= hi[2])
    xyz, scales, cov3D, opas, colors, semantics = \
        xyz[m], scales[m], cov3D[m], opas[m], colors[m], semantics[m]
    print(f"[INFO] After bbox filter: {xyz.shape[0]} gaussians")

    # ── 2. 构建强度查询网格 ──
    pts, H, W, D = build_grid(lo, hi, args.grid_size, device)
    print(f"[INFO] Grid: {H}×{W}×{D} = {H*W*D} voxels")

    # ── 3. LocalAggregator 投影 ──
    agg = LocalAggregator(
        scale_multiplier=args.scale_multiplier,
        H=H, W=W, D=D,
        pc_min=lo.cpu().tolist(),
        grid_size=args.grid_size,
        radii_min=args.radii_min,
    ).to(device)

    with torch.no_grad():
        logits, bin_logits, density = agg(
            pts=pts.unsqueeze(0),
            means3D=xyz.unsqueeze(0),
            opas=opas.unsqueeze(0),
            semantics=semantics.unsqueeze(0),
            scales=scales.unsqueeze(0),
            cov3D=cov3D.unsqueeze(0),
            metas=None,
            origin_use=lo,
        )

    density_3d = density.reshape(H, W, D)
    occ_3d = 1.0 - torch.exp(-density_3d)
    occ_np = occ_3d.cpu().numpy()
    print(f"[INFO] density: [{density_3d.min():.4f}, {density_3d.max():.4f}]")
    print(f"[INFO] occ:     [{occ_np.min():.4f}, {occ_np.max():.4f}]")

    # ── 4. 语义投影 ──
    logits_sem_3d = logits.reshape(H, W, D, -1)  # 语义 logits
    sem_feat_dim = logits_sem_3d.shape[-1]

    # 用 occ 做 mask
    occ_mask = occ_np > args.thr
    n_occ = int(occ_mask.sum())
    print(f"[INFO] Occupied voxels: {n_occ} / {H*W*D}")

    if n_occ == 0:
        print("[WARN] 无占用体素，降低 --thr")
        return

    # ── 5. 保存 3D 占用体素点云 ──
    occ_pts_np = lo.cpu().numpy() + \
        (np.argwhere(occ_mask).astype(np.float32) + 0.5) * args.grid_size

    # 用 RGB 着色（也可用语义 PCA）
    vals = occ_np[occ_mask]
    vals_norm = np.clip(vals / (vals.max() + 1e-6), 0, 1)
    colors_np = np.repeat(vals_norm[:, None], 3, axis=1)  # BGR→RGB, [0,1]

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(occ_pts_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np.astype(np.float64))
    voxel_path = os.path.join(args.output, "occ_voxel.ply")
    o3d.io.write_point_cloud(voxel_path, pcd)
    print(f"[INFO] Saved: {voxel_path}")

    # ── 6. XY 平面俯视图 ──
    # 沿 Z 轴取最大占用
    occ_xy = occ_np.max(axis=0)  # [W, D] — 注意: H=X(深度), W=Y(左右), D=Z(高度)
    # occ_np shape is [H, W, D] where H=X, W=Y, D=Z
    # XY plane = dims 0 and 1, so take max over dim 2
    occ_xy = occ_np.max(axis=2)  # [H, W] = [X, Y]

    # 翻转让图像上方对应前方
    occ_xy_img = np.flipud(occ_xy.T)  # [W, H] → [H_img, W_img], Y轴翻转

    occ_xy_uint8 = np.clip(occ_xy_img / (occ_xy_img.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    occ_color = cv2.applyColorMap(occ_xy_uint8, cv2.COLORMAP_JET)
    topdown_path = os.path.join(args.output, "occ_topdown_zmax.png")
    cv2.imwrite(topdown_path, occ_color)
    print(f"[INFO] Saved: {topdown_path}")

    # ── 7. 语义俯视图 (argmax 类别标签着色) ──
    if sem_feat_dim >= 2:
        sem_np = logits_sem_3d.cpu().numpy()  # [H, W, D, C]

        # 每个体素取 argmax 类别
        labels_3d = np.argmax(sem_np, axis=-1)  # [H, W, D]
        n_classes = sem_feat_dim

        # 调色板 (10+1 类, 用 HSV 色环均匀分布)
        palette = []
        for i in range(n_classes):
            hue = i / max(n_classes, 1) * 180
            c = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
            palette.append(c.tolist())
        palette = np.array(palette, dtype=np.uint8)  # [C, 3]

        # 占用体素的类别
        occ_idx = np.argwhere(occ_mask)  # [N_occ, 3]
        labels_occ = labels_3d[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]]

        # 3D 语义 RGB
        sem_3d_rgb = np.zeros((H, W, D, 3), dtype=np.uint8)
        sem_3d_rgb[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] = palette[labels_occ]

        # 保存带类别的体素 PLY
        sem_pcd_path = os.path.join(args.output, "occ_voxel_sem_label.ply")
        sem_colors_3d = sem_3d_rgb[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] / 255.0
        pcd_sem = o3d.geometry.PointCloud()
        pcd_sem.points = o3d.utility.Vector3dVector(occ_pts_np)
        pcd_sem.colors = o3d.utility.Vector3dVector(sem_colors_3d.astype(np.float64))
        o3d.io.write_point_cloud(sem_pcd_path, pcd_sem)
        print(f"[INFO] Saved: {sem_pcd_path}")

        # XY 俯视图: 沿 Z 取 occ 最大的那个体素的类别
        z_idx = np.argmax(occ_np, axis=2)  # [H, W]
        h_idx, w_idx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        labels_xy = labels_3d[h_idx, w_idx, z_idx]  # [H, W]
        sem_xy = palette[labels_xy]  # [H, W, 3]

        sem_xy_img = np.flipud(sem_xy.transpose(1, 0, 2))  # 转置+上下翻转
        sem_path = os.path.join(args.output, "occ_topdown_sem_label.png")
        cv2.imwrite(sem_path, sem_xy_img)
        print(f"[INFO] Saved: {sem_path}")

        # 类别图例
        legend_h = n_classes * 20 + 10
        legend = np.ones((legend_h, 200, 3), dtype=np.uint8) * 255
        for i in range(n_classes):
            y0 = 10 + i * 20
            cv2.rectangle(legend, (10, y0), (30, y0 + 15), palette[i].tolist(), -1)
            cv2.putText(legend, f"cls{i}", (40, y0 + 13),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        legend_path = os.path.join(args.output, "occ_sem_legend.png")
        cv2.imwrite(legend_path, legend)
        print(f"[INFO] Saved legend: {legend_path}")
        print(f"[INFO]   cls0=ceiling, cls1=floor, cls2=wall, cls3=window, cls4=chair")
        print(f"[INFO]   cls5=bed, cls6=sofa, cls7=table, cls8=tv, cls9=furniture, cls10=objects")

    print("[DONE]")


if __name__ == "__main__":
    main()
