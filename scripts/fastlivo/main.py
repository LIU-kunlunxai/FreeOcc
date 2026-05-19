#!/usr/bin/env python3
"""
Semantic map RGB pipeline: LIO poses + Depth-ICP calibration + Colorize.

Usage:
    python -m pipeline.main                     # use default config
    python -m pipeline.main --config my.yaml    # use YAML config
    python -m pipeline.main --skip-calib        # skip calibration, use existing extrinsic
    python -m pipeline.main --h265              # use H265 color topic
"""

import argparse
import os
import sys
import time
import numpy as np

from .config import PipelineConfig
from .poses import load_poses_tum, load_global_map
from .io_bag import read_color_images, read_h265_images
from .io_pcd import voxel_downsample, save_pcd_binary
from .calibrate import calibrate_extrinsic
from .colorize import colorize_map


def log(msg):
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


def load_extrinsic(path: str):
    """Load R_cl, t_cl from 4x4 matrix file."""
    T = np.loadtxt(path, delimiter=',')
    return T[:3, :3], T[:3, 3]


def main():
    parser = argparse.ArgumentParser(description="LiDAR-Camera RGB colorization pipeline")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--skip-calib", action="store_true", help="Skip calibration, load existing extrinsic")
    parser.add_argument("--extrinsic", type=str, help="Path to extrinsic file (for --skip-calib)")
    parser.add_argument("--h265", action="store_true", help="Use H265 compressed color topic")
    parser.add_argument("--output", type=str, help="Output PCD path")
    parser.add_argument("--voxel", type=float, help="Colorization voxel size (default: 0.01m)")
    args = parser.parse_args()

    t_total = time.time()

    # 1. Config
    if args.config:
        cfg = PipelineConfig.from_yaml(args.config)
    else:
        cfg = PipelineConfig()

    if args.voxel:
        cfg.color_voxel_size = args.voxel
    os.makedirs(cfg.output_dir, exist_ok=True)

    log("=" * 60)
    log("Semantic Map RGB Pipeline")
    log(f"  Bag: {cfg.bag_path}")
    log(f"  Map: {cfg.map_pcd_dir}")
    log(f"  Output: {cfg.output_dir}")
    log("=" * 60)

    # 2. Load poses
    log("\n[1/5] Loading poses...")
    poses = load_poses_tum(cfg.pose_file)
    log(f"  {len(poses)} poses loaded")

    # 3. Build global map (PCDs already in world frame from LIO)
    log(f"\n[2/5] Building global map ({cfg.color_voxel_size}m voxel)...")
    map_pts = load_global_map(cfg.map_pcd_dir, voxel_size=cfg.color_voxel_size, print_fn=log)

    # 4. Calibrate extrinsic
    if args.skip_calib:
        ext_path = args.extrinsic or os.path.join(cfg.output_dir, "extrinsic_optimized.txt")
        log(f"\n[3/5] Loading extrinsic from {ext_path}")
        R_cl, t_cl = load_extrinsic(ext_path)
        R_lc = R_cl.T
        t_lc = -R_cl.T @ t_cl
    else:
        log(f"\n[3/5] Calibrating extrinsic (Depth-ICP)...")
        calib_map = voxel_downsample(map_pts, cfg.calib_voxel_size)
        log(f"  Calibration map: {len(calib_map)} pts ({cfg.calib_voxel_size}m voxel)")
        R_lc, t_lc = calibrate_extrinsic(cfg, calib_map, poses, print_fn=log)

    log(f"  R_lc:\n{np.array2string(R_lc.T, precision=6)}")
    log(f"  t_cl: {-R_lc.T @ t_lc}")

    # 5. Read color images
    log("\n[4/5] Reading color images...")
    if args.h265:
        images = read_h265_images(cfg.bag_path, cfg.h265_topic)
        log(f"  {len(images)} H265 frames decoded")
    else:
        images = read_color_images(cfg.bag_path, cfg.color_topic)
        log(f"  {len(images)} raw frames")

    # 6. Colorize
    log("\n[5/5] Colorizing map...")
    colors, valid_mask = colorize_map(cfg, map_pts, images, poses, R_lc, t_lc, print_fn=log)

    # 7. Save
    output_path = args.output or os.path.join(cfg.output_dir, "colored_map.pcd")
    save_pcd_binary(output_path, map_pts[valid_mask], colors)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    log(f"\n  Saved: {output_path} ({size_mb:.1f} MB)")

    elapsed = time.time() - t_total
    log(f"\n{'=' * 60}")
    log(f"Done in {elapsed:.0f}s. Open in CloudCompare to verify.")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
