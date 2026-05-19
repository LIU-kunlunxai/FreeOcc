from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import os
import yaml


@dataclass
class PipelineConfig:
    # Paths
    bag_path: str = os.path.expanduser("~/Documents/——lazyall/bag_20260518_150039")
    map_pcd_dir: str = os.path.expanduser("~/ros2_ws/src/fast_livo2/Log/pcd")
    pose_file: str = os.path.expanduser("~/ros2_ws/src/fast_livo2/Log/result/unitree_g1.txt")
    output_dir: str = os.path.expanduser("~/Documents/——lazyall/output")

    # Camera intrinsics
    fx: float = 386.50219727
    fy: float = 385.93804932
    cx: float = 321.4967041
    cy: float = 241.84043884
    dist_coeffs: np.ndarray = field(
        default_factory=lambda: np.array([-0.05556279, 0.06078598, -0.00067143, 0.00018534, -0.01935727])
    )
    img_w: int = 640
    img_h: int = 480

    # Initial extrinsic: camera -> lidar (R_lc, t_lc)
    R_lc_init: np.ndarray = field(default_factory=lambda: np.array([
        [0.00431, -0.01066, 0.99993],
        [-0.99999, 0.00029, 0.00432],
        [-0.00034, -0.99994, -0.01066]
    ]))
    t_lc_init: np.ndarray = field(default_factory=lambda: np.array([0.03095, -0.02314, 0.15048]))

    # Bag topics
    color_topic: str = "/camera/body/color/image_raw"
    depth_topic: str = "/camera/body/depth/image_raw"
    h265_topic: str = "/camera/body/color/h265"

    # Calibration
    calib_voxel_size: float = 0.05
    calib_max_depth: float = 8.0
    calib_max_iter: int = 50
    calib_max_dist: float = 0.2
    calib_cam_pts_limit: int = 15000
    n_workers: int = field(default_factory=os.cpu_count)

    # Colorization
    color_voxel_size: float = 0.01
    color_max_range: float = 10.0
    color_min_range: float = 0.3

    # Pose matching
    max_time_offset: float = 0.15

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    @property
    def R_cl_init(self) -> np.ndarray:
        return self.R_lc_init.T

    @property
    def t_cl_init(self) -> np.ndarray:
        return -self.R_lc_init.T @ self.t_lc_init

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                if k in ("dist_coeffs", "R_lc_init", "t_lc_init"):
                    setattr(cfg, k, np.array(v))
                else:
                    setattr(cfg, k, v)
        return cfg

    def save_yaml(self, path: str):
        d = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False)
