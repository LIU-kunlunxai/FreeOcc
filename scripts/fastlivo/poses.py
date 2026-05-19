import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation

import sys, os as _os
_sys_path = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)
from scripts.fastlivo.io_pcd import load_pcd_xyz, voxel_downsample


def load_poses_tum(filepath: str) -> list:
    """Load TUM-format poses. Returns list of (timestamp, T_wl 4x4)."""
    data = np.loadtxt(filepath)
    poses = []
    for row in data:
        R = Rotation.from_quat(row[4:8]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = row[1:4]
        poses.append((row[0], T))
    return poses


def get_pose_times(poses: list) -> np.ndarray:
    """Extract timestamps array from poses list."""
    return np.array([p[0] for p in poses])


def find_nearest_pose(query_ts: float, pose_times: np.ndarray, max_dt: float = 0.15):
    """Find index of nearest pose within max_dt. Returns None if too far."""
    idx = np.argmin(np.abs(pose_times - query_ts))
    if abs(pose_times[idx] - query_ts) > max_dt:
        return None
    return idx


def load_global_map(pcd_dir: str, voxel_size: float = 0.05,
                    print_fn=print) -> np.ndarray:
    """Load and merge per-frame PCDs (already in world frame) with voxel filter."""
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
    print_fn(f"  Loading {len(pcd_files)} PCDs...")

    all_points = []
    for i, f in enumerate(pcd_files):
        pts = load_pcd_xyz(f)
        all_points.append(pts)
        if (i + 1) % 100 == 0:
            print_fn(f"  {i + 1}/{len(pcd_files)}")

    all_points = np.vstack(all_points).astype(np.float32)
    print_fn(f"  Raw: {len(all_points)} points")
    filtered = voxel_downsample(all_points, voxel_size)
    print_fn(f"  After {voxel_size}m voxel: {len(filtered)} points")
    return filtered
