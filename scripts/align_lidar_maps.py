#!/usr/bin/env python3
"""对齐两个 LiDAR 会话的地图 (ICP 或手动选点)"""

import argparse, os, sys, struct
import numpy as np
import open3d as o3d


def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Failed to load {path}")
    return pcd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=False, help="源 PCD (如 0519)")
    parser.add_argument("--target", required=False, help="目标 PCD (如 0518 基准)")
    parser.add_argument("--output", required=False, help="输出变换矩阵 .txt")
    parser.add_argument("--voxel", type=float, default=0.1, help="降采样体素 (m)")
    parser.add_argument("--max-dist", type=float, default=0.5, help="ICP 最大对应距离")
    args = parser.parse_args()

    # args.source = "/home/liu/workspace/kunlunxai_record/record_dataset/bag_20260519_152901_dataset/other_data/map.pcd"
    # args.target = "/home/liu/workspace/kunlunxai_record/record_dataset/map/processed_scans_20260518_181232.pcd"
    # args.output = "/home/liu/workspace/kunlunxai_record/record_dataset/map/T_0519_to_0518.txt"


    print(f"Loading source: {args.source}")
    src = load_pcd(args.source)
    print(f"  {len(src.points)} points")

    print(f"Loading target: {args.target}")
    tgt = load_pcd(args.target)
    print(f"  {len(tgt.points)} points")

    # 降采样
    src_ds = src.voxel_down_sample(voxel_size=args.voxel)
    tgt_ds = tgt.voxel_down_sample(voxel_size=args.voxel)
    print(f"After voxel {args.voxel}m: src={len(src_ds.points)}, tgt={len(tgt_ds.points)}")

    # 1. FPFH 粗对齐
    print("Step 1: FPFH feature matching...")
    radius_feature = max(args.voxel * 3, 0.3)
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_ds, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_ds, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_ds, tgt_ds, src_fpfh, tgt_fpfh, mutual_filter=True,
        max_correspondence_distance=args.max_dist * 2,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3, checkers=[], criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    print(f"  RANSAC Fitness: {result_ransac.fitness:.4f}, RMSE: {result_ransac.inlier_rmse:.4f}")

    # 2. ICP 精调 (point-to-plane)
    print("Step 2: ICP refinement...")
    src_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))
    tgt_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        src_ds, tgt_ds, args.max_dist, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(f"  Fitness: {result.fitness:.4f}")
    print(f"  RMSE:    {result.inlier_rmse:.4f}")
    print(f"  Transform:\n{result.transformation}")

    # 保存
    np.savetxt(args.output, result.transformation, fmt="%.12f")
    print(f"\nSaved: {args.output}")
    print(f"Usage: T_w_target = transform @ T_w_source")


if __name__ == "__main__":
    main()
