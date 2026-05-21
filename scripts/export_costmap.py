#!/usr/bin/env python3
"""从 LiDAR + 语义 PointCloud 导出 2D costmap (PGM + YAML)，ROS2 Nav2 可直接用"""

import argparse, os, sys
import numpy as np
import cv2

def load_xyz(path, max_pts=5000000):
    """加载 PLY/PCD 的 XYZ."""
    if path.endswith(".ply"):
        from plyfile import PlyData
        ply = PlyData.read(path)
        v = ply["vertex"].data
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    elif path.endswith(".pcd"):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
    else:
        raise ValueError(f"unsupported format: {path}")
    if len(pts) > max_pts:
        pts = pts[np.random.choice(len(pts), max_pts, replace=False)]
    return pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="fused.ply 或 LiDAR PCD")
    parser.add_argument("--output", required=True, help="输出前缀 (生成 .pgm + .yaml)")
    parser.add_argument("--resolution", type=float, default=0.05, help="栅格分辨率 (m)")
    parser.add_argument("--z-floor", type=float, default=0.0, help="地面高度 (m)")
    parser.add_argument("--z-thresh", type=float, default=0.3, help="障碍物高度阈值 (> 此值为障碍)")
    parser.add_argument("--z-ceiling", type=float, default=2.5, help="天花板高度")
    parser.add_argument("--margin", type=int, default=20, help="边界填充 (像素)")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    pts = load_xyz(args.input)
    print(f"  {len(pts)} points")

    # 过滤高度范围
    pts = pts[(pts[:, 2] > args.z_floor + 0.05) & (pts[:, 2] < args.z_ceiling)]
    print(f"  After height filter ({args.z_floor+0.05}m ~ {args.z_ceiling}m): {len(pts)}")

    # XY 范围
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    print(f"  XY: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]")

    # 栅格大小
    res = args.resolution
    w = int((x_max - x_min) / res) + 1 + 2 * args.margin
    h = int((y_max - y_min) / res) + 1 + 2 * args.margin
    print(f"  Grid: {w}×{h} ({w*res:.1f}m × {h*res:.1f}m)")

    # 投影到栅格
    grid = np.zeros((h, w), dtype=np.uint8)
    x_idx = ((pts[:, 0] - x_min) / res + args.margin).astype(int)
    y_idx = ((pts[:, 1] - y_min) / res + args.margin).astype(int)
    valid = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
    x_idx, y_idx = x_idx[valid], y_idx[valid]

    # 标记障碍物 (离地 > z_thresh)
    z = pts[valid, 2] - args.z_floor
    obstacle = z > args.z_thresh
    np.add.at(grid, (y_idx[obstacle], x_idx[obstacle]), 1)
    grid = np.clip(grid, 0, 1)

    # ROS costmap 规范: 0=自由, 100=占据, -1=未知
    costmap = np.full((h, w), 255, dtype=np.uint8)  # 未知=255
    costmap[grid == 0] = 254  # 自由空间
    costmap[grid > 0] = 0    # 障碍物

    # 保存 PGM
    pgm_path = args.output + ".pgm"
    cv2.imwrite(pgm_path, costmap)
    print(f"  Saved: {pgm_path}")

    # 保存 YAML
    yaml_path = args.output + ".yaml"
    origin_x = x_min - args.margin * res
    origin_y = y_min - args.margin * res
    with open(yaml_path, "w") as f:
        f.write(f"image: {os.path.basename(pgm_path)}\n")
        f.write(f"resolution: {res}\n")
        f.write(f"origin: [{origin_x:.4f}, {origin_y:.4f}, 0.0]\n")
        f.write("negate: 0\noccupied_thresh: 0.45\nfree_thresh: 0.55\n")
    print(f"  Saved: {yaml_path}")
    print(f"  Stats: occupied={obstacle.sum()}, free={len(pts)-obstacle.sum()}")
    print("[DONE]")


if __name__ == "__main__":
    main()
