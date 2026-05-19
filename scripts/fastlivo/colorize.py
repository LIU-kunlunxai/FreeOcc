"""Colorize LiDAR map using camera images and calibrated extrinsic."""

import numpy as np
import cv2
from multiprocessing import Pool, shared_memory

from .config import PipelineConfig
from .poses import get_pose_times, find_nearest_pose

# Module-level globals for multiprocessing workers
_shm_name = None
_pts_shape = None
_pts_dtype = None
_K = None
_dist = None
_img_w = None
_img_h = None


def _init_worker(shm_name, pts_shape, pts_dtype, K, dist, img_w, img_h):
    global _shm_name, _pts_shape, _pts_dtype, _K, _dist, _img_w, _img_h
    _shm_name = shm_name
    _pts_shape = pts_shape
    _pts_dtype = pts_dtype
    _K = K
    _dist = dist
    _img_w = img_w
    _img_h = img_h


def _process_frame(args):
    """Worker: project world points into one camera frame, return indices + colors."""
    T_cw, lidar_pos, img_bgr = args
    shm = shared_memory.SharedMemory(name=_shm_name)
    pts = np.ndarray(_pts_shape, dtype=_pts_dtype, buffer=shm.buf)

    # Range filter: distance from lidar position in world frame
    diff = pts - lidar_pos
    dists_sq = np.sum(diff ** 2, axis=1)
    nearby = (dists_sq > 0.09) & (dists_sq < 100.0)
    nearby_idx = np.where(nearby)[0]
    pts_near = pts[nearby]

    # World → camera (no point cloud transform, pose goes into projection)
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    pts_cam = (R_cw @ pts_near.T).T + t_cw
    front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[front]
    valid_idx = nearby_idx[front]

    if len(pts_cam) == 0:
        shm.close()
        return None

    # Project to image
    pts_2d, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), _K, _dist)
    pts_2d = pts_2d.reshape(-1, 2)

    in_b = ((pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < _img_w - 1) &
            (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < _img_h - 1))
    b2d = pts_2d[in_b].astype(np.int32)
    bidx = valid_idx[in_b]

    if len(b2d) == 0:
        shm.close()
        return None

    pc = img_bgr[b2d[:, 1], b2d[:, 0]].astype(np.float64)
    shm.close()
    return bidx, pc


def colorize_map(cfg: PipelineConfig, map_pts: np.ndarray, images: list,
                 poses: list, R_lc: np.ndarray, t_lc: np.ndarray,
                 print_fn=print) -> tuple:
    """
    Colorize map points using camera images.

    Points stay in world frame; per-frame camera pose (T_cw) handles projection.

    Args:
        images: list of (timestamp, bgr_image)
        poses: list of (timestamp, T_wl)
        R_lc, t_lc: camera→lidar extrinsic

    Returns:
        (colors (N,3) uint8, valid_mask (N,) bool)
    """
    R_cl = R_lc.T
    t_cl = -R_lc.T @ t_lc

    # T_cl as 4x4
    T_cl = np.eye(4)
    T_cl[:3, :3] = R_cl
    T_cl[:3, 3] = t_cl

    pose_times = get_pose_times(poses)

    # Match images to poses, compute T_cw = T_cl @ T_lw for each frame
    frame_data = []
    for img_ts, img_bgr in images:
        idx = find_nearest_pose(img_ts, pose_times, cfg.max_time_offset)
        if idx is None:
            continue
        _, T_wl = poses[idx]
        T_lw = np.linalg.inv(T_wl)
        T_cw = T_cl @ T_lw
        lidar_pos = T_wl[:3, 3]
        frame_data.append((T_cw, lidar_pos, img_bgr))

    print_fn(f"[Color] {len(frame_data)} frames matched, {cfg.n_workers} workers")

    # Shared memory for map points
    shm = shared_memory.SharedMemory(create=True, size=map_pts.nbytes)
    shm_pts = np.ndarray(map_pts.shape, dtype=map_pts.dtype, buffer=shm.buf)
    shm_pts[:] = map_pts[:]

    colors = np.zeros((len(map_pts), 3), dtype=np.float64)
    counts = np.zeros(len(map_pts), dtype=np.int32)

    with Pool(cfg.n_workers, initializer=_init_worker,
              initargs=(shm.name, map_pts.shape, map_pts.dtype,
                        cfg.K, cfg.dist_coeffs, cfg.img_w, cfg.img_h)) as pool:
        results = pool.map(_process_frame, frame_data, chunksize=4)

    for r in results:
        if r is None:
            continue
        bidx, pc = r
        np.add.at(colors, bidx, pc)
        np.add.at(counts, bidx, 1)

    shm.close()
    shm.unlink()

    valid = counts > 0
    colors[valid] /= counts[valid, None]
    n = valid.sum()
    print_fn(f"  Colored {n}/{len(map_pts)} points ({100 * n / len(map_pts):.1f}%)")

    return colors[valid].astype(np.uint8), valid
