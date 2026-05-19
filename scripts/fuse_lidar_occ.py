#!/usr/bin/env python3
"""将 FreeOcc 语义占位映射到 LiDAR 地图上，生成几何准确 + 语义着色的融合点云。

用法:
    python scripts/fuse_lidar_occ.py \
        --lidar /path/to/map.pcd \                    # LiDAR 全局地图
        --occ /path/to/occ_voxel_sem_label.ply \       # FreeOcc 语义占位体素
        --output /path/to/fused_semantic.ply
"""

import argparse, os, sys
import numpy as np
import cv2
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_pcd(path: str) -> np.ndarray:
    """加载 PCD/PLY 点云，返回 (N,3) float32."""
    if path.endswith(".ply"):
        from plyfile import PlyData
        ply = PlyData.read(path)
        v = ply["vertex"].data
        return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    elif path.endswith(".pcd"):
        # 二进制 PCD
        with open(path, "rb") as f:
            header, n, fields = [], 0, 3
            while True:
                line = f.readline().decode().strip()
                header.append(line)
                if line.startswith("POINTS"): n = int(line.split()[1])
                if line.startswith("FIELDS"): fields = len(line.split()[1:])
                if line.startswith("DATA"): break
            data = np.frombuffer(f.read(), dtype=np.float32).reshape(n, fields)
            return data[:, :3].astype(np.float32)
    else:
        raise ValueError(f"unsupported format: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidar", required=True, help="LiDAR 全局地图 (PCD/PLY)")
    parser.add_argument("--occ", required=True, help="FreeOcc 语义体素 PLY")
    parser.add_argument("--output", required=True, help="输出融合 PLY")
    parser.add_argument("--radius", type=float, default=0.3, help="匹配半径 (m)")
    parser.add_argument("--max-points", type=int, default=500000)
    args = parser.parse_args()

    # 1. 加载 LiDAR 地图
    print(f"[1/4] Loading LiDAR map: {args.lidar}")
    lidar_pts = load_pcd(args.lidar)
    print(f"  {len(lidar_pts)} points")

    # 2. 加载 FreeOcc 语义体素
    print(f"[2/4] Loading FreeOcc occupancy: {args.occ}")
    occ_voxels = load_pcd(args.occ)  # (M, 3)

    from plyfile import PlyData
    ply = PlyData.read(args.occ)
    occ_colors = np.stack([ply["vertex"]["red"], ply["vertex"]["green"],
                           ply["vertex"]["blue"]], axis=1)  # (M, 3) uint8
    print(f"  {len(occ_voxels)} voxels")

    # 3. 用 KD-Tree 匹配：每个 LiDAR 点找最近的占位体素
    print(f"[3/4] KD-Tree matching (radius={args.radius}m) ...")
    tree = cKDTree(occ_voxels)
    dists, idxs = tree.query(lidar_pts, k=1)

    matched = dists < args.radius
    n_matched = matched.sum()
    print(f"  {n_matched}/{len(lidar_pts)} matched ({n_matched/len(lidar_pts)*100:.1f}%)")

    # 4. 融合：LiDAR 位置 + FreeOcc 颜色
    print(f"[4/4] Saving fused map ...")
    colors = np.zeros((len(lidar_pts), 3), dtype=np.uint8)
    colors[matched] = occ_colors[idxs[matched]]

    # 子采样
    if len(lidar_pts) > args.max_points:
        keep = np.zeros(len(lidar_pts), dtype=bool)
        matched_idx = np.where(matched)[0]
        unmatched_idx = np.where(~matched)[0]
        n_m = min(args.max_points // 2, len(matched_idx))
        n_u = min(args.max_points - n_m, len(unmatched_idx))
        keep[np.random.choice(matched_idx, n_m, replace=False)] = True
        keep[np.random.choice(unmatched_idx, n_u, replace=False)] = True
        lidar_pts = lidar_pts[keep]
        colors = colors[keep]

    # 写 PLY
    with open(args.output, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(lidar_pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(lidar_pts)):
            f.write(f"{lidar_pts[i,0]:.6f} {lidar_pts[i,1]:.6f} {lidar_pts[i,2]:.6f} "
                    f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")

    print(f"  Saved: {args.output} ({len(lidar_pts)} points)")
    print("[DONE]")


if __name__ == "__main__":
    main()
