"""Depth-ICP extrinsic calibration: align LiDAR map points to depth camera point clouds."""

import numpy as np
import cv2
import os
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from multiprocessing import Pool

from .config import PipelineConfig
from .io_bag import read_depth_images, depth_to_pointcloud
from .poses import get_pose_times, find_nearest_pose

# Module-level globals for multiprocessing (set before pool.map)
_g_frames = None


def _eval_frame(args):
    """Worker: compute ICP cost for one frame."""
    fi, x = args
    lidar_local, cam_pts = _g_frames[fi]
    R_cl = cv2.Rodrigues(x[:3].reshape(3, 1))[0]
    t_cl = x[3:6]

    lidar_in_cam = (R_cl @ lidar_local.T).T + t_cl
    mask = (lidar_in_cam[:, 2] > 0.3) & (lidar_in_cam[:, 2] < 8.0)
    lidar_in_cam = lidar_in_cam[mask]
    if len(lidar_in_cam) < 50:
        return None

    tree = cKDTree(cam_pts)
    dists, _ = tree.query(lidar_in_cam, k=1)
    inliers = dists < 0.2
    n_inl = inliers.sum()
    if n_inl < 30:
        return None
    return np.mean(dists[inliers] ** 2), n_inl


def _cost_parallel(x, pool):
    """Aggregate ICP cost across all frames."""
    args = [(i, x) for i in range(len(_g_frames))]
    results = pool.map(_eval_frame, args)
    total_cost = 0.0
    total_inl = 0
    for r in results:
        if r is None:
            continue
        c, n = r
        total_cost += c * n
        total_inl += n
    if total_inl == 0:
        return 1e10
    return total_cost / total_inl


def calibrate_extrinsic(cfg: PipelineConfig, map_pts: np.ndarray,
                        poses: list, print_fn=print) -> tuple:
    """
    Run depth-ICP extrinsic optimization.

    Returns (R_lc_opt, t_lc_opt) in camera→lidar convention.
    Also saves result to output_dir/extrinsic_optimized.txt.
    """
    global _g_frames

    print_fn("[Calib] Reading depth images...")
    depth_frames = read_depth_images(cfg.bag_path, cfg.depth_topic)
    print_fn(f"  {len(depth_frames)} depth frames")

    print_fn("[Calib] Precomputing frame pairs...")
    pose_times = get_pose_times(poses)
    _g_frames = []

    for ts, depth in depth_frames:
        idx = find_nearest_pose(ts, pose_times, cfg.max_time_offset)
        if idx is None:
            continue
        _, T_wl = poses[idx]
        T_lw = np.linalg.inv(T_wl)

        # Map points → lidar local frame, range filter
        pts_local = (T_lw[:3, :3] @ map_pts.T).T + T_lw[:3, 3]
        dists_sq = np.sum(pts_local ** 2, axis=1)
        mask = (dists_sq > 0.25) & (dists_sq < 64.0)
        pts_local = pts_local[mask].astype(np.float32)

        # Depth → camera point cloud
        cam_pts = depth_to_pointcloud(
            depth, cfg.fx, cfg.fy, cfg.cx, cfg.cy, cfg.calib_max_depth)
        if len(cam_pts) > cfg.calib_cam_pts_limit:
            cam_pts = cam_pts[np.random.choice(len(cam_pts), cfg.calib_cam_pts_limit, replace=False)]

        if len(pts_local) > 200 and len(cam_pts) > 200:
            _g_frames.append((pts_local, cam_pts))

    print_fn(f"  {len(_g_frames)} frame pairs ready")

    # Initial guess: R_cl = R_lc^T
    R_cl_init = cfg.R_cl_init
    t_cl_init = cfg.t_cl_init
    rvec_init = cv2.Rodrigues(R_cl_init)[0].flatten()
    t_init = t_cl_init.copy()
    x0 = np.concatenate([rvec_init, t_init])

    print_fn(f"[Calib] Optimizing ({cfg.n_workers} workers, max {cfg.calib_max_iter} iter)...")
    with Pool(cfg.n_workers) as pool:
        t0 = time.time()
        c0 = _cost_parallel(x0, pool)
        print_fn(f"  Initial cost: {c0:.6f} ({time.time() - t0:.2f}s)")

        iter_count = [0]

        def callback(xk):
            iter_count[0] += 1
            if iter_count[0] % 5 == 0:
                c = _cost_parallel(xk, pool)
                dr = np.linalg.norm(xk[:3] - rvec_init) * 180 / np.pi
                dt_val = np.linalg.norm(xk[3:6] - t_init)
                print_fn(f"  iter {iter_count[0]}: cost={c:.6f}, dR={dr:.3f}deg, dT={dt_val:.4f}m")

        result = minimize(
            lambda x: _cost_parallel(x, pool), x0,
            method='Powell', callback=callback,
            options={'maxiter': cfg.calib_max_iter, 'xtol': 1e-6, 'ftol': 1e-7, 'disp': False})

    elapsed = time.time() - t0
    rvec_opt = result.x[:3]
    t_cl_opt = result.x[3:6]
    R_cl_opt = cv2.Rodrigues(rvec_opt.reshape(3, 1))[0]

    dr = np.linalg.norm(rvec_opt - rvec_init) * 180 / np.pi
    dt_val = np.linalg.norm(t_cl_opt - t_init)
    print_fn(f"  Done in {elapsed:.0f}s, {result.nfev} evals")
    print_fn(f"  Cost: {c0:.6f} -> {result.fun:.6f}")
    print_fn(f"  R change: {dr:.3f} deg, T change: {dt_val:.4f} m")

    # Convert back to camera→lidar convention for storage
    R_lc_opt = R_cl_opt.T
    t_lc_opt = -R_cl_opt.T @ t_cl_opt

    # Save
    ext_file = os.path.join(cfg.output_dir, "extrinsic_optimized.txt")
    os.makedirs(cfg.output_dir, exist_ok=True)
    T44 = np.eye(4)
    T44[:3, :3] = R_cl_opt
    T44[:3, 3] = t_cl_opt
    with open(ext_file, 'w') as f:
        for i in range(4):
            f.write(','.join(f'{T44[i, j]:.8f}' for j in range(4)) + '\n')
    print_fn(f"  Saved: {ext_file}")

    _g_frames = None
    return R_lc_opt, t_lc_opt
