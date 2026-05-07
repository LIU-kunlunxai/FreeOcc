import glob
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset

from .geom import matrix_to_lie


def get_dataset(cfg: DictConfig, device="cuda:0"):
    return dataset_dict[cfg.data.dataset](cfg, device=device)


class BaseDataset(TorchDataset):
    """Shared dataset utilities for the supported FreeOcc data loaders."""

    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super().__init__()
        self.name = cfg.data.dataset
        self.stereo = cfg.mode == "stereo"
        self.device = device
        self.png_depth_scale = cfg.data.get("png_depth_scale", None)
        self.stride = cfg.get("stride", 1)

        self.input_folder = cfg.data.input_folder
        self.color_paths = sorted(glob.glob(self.input_folder))

        self.t_start = cfg.get("t_start", 0)
        self.t_stop = cfg.get("t_stop", None)
        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
        self.color_paths = self.color_paths[:: self.stride]

        self.has_dyn_masks = False
        self.background_value = 0
        self.dilate_masks = True
        self.dilation_kernel_size = 1
        self.return_stat_masks = cfg.get("with_dyn", False)
        self.n_img = len(self.color_paths)

        self.depth_paths = None
        self.mask_paths = None
        self.poses = None
        self.relative_poses = False
        self.image_timestamps = None

        self.H, self.W = int(cfg.data.cam.H), int(cfg.data.cam.W)
        self.H_out, self.W_out = int(cfg.data.cam.H_out), int(cfg.data.cam.W_out)
        self.fx, self.fy = float(cfg.data.cam.fx), float(cfg.data.cam.fy)
        self.cx, self.cy = float(cfg.data.cam.cx), float(cfg.data.cam.cy)
        self.H_edge, self.W_edge = int(cfg.data.cam.H_edge), int(cfg.data.cam.W_edge)
        self.distortion = np.array(cfg.data.cam.distortion) if "distortion" in cfg.data.cam else None

    def __len__(self) -> int:
        return self.n_img

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(
            glob.glob(os.path.join(path, "*.txt")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )

        # ScanNet-like pose folders sometimes contain invalid tail poses.
        last_valid_frame = -1
        for i, pose_path in enumerate(pose_paths):
            with open(pose_path, "r") as f:
                lines = f.readlines()
            rows = [list(map(float, line.split(" "))) for line in lines]
            c2w = np.array(rows).reshape(4, 4)
            if np.isnan(c2w).any() or np.isinf(c2w).any():
                last_valid_frame = i + 1
                break
            self.poses.append(c2w)

        return last_valid_frame

    def depthloader(self, index: int):
        if self.depth_paths is None:
            return None

        depth_path = self.depth_paths[index]
        if depth_path.endswith(".png"):
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth_data /= self.png_depth_scale
        elif depth_path.endswith(".npy"):
            depth_data = np.load(depth_path).astype(np.float32)
        else:
            raise TypeError(depth_path)

        depth_data[np.isnan(depth_data)] = 0.0
        return depth_data

    def _get_image(self, index: int) -> torch.Tensor:
        color_path = self.color_paths[index]
        color_data = cv2.imread(color_path)

        H_out_with_edge = self.H_out + self.H_edge * 2
        W_out_with_edge = self.W_out + self.W_edge * 2
        color_data = cv2.resize(color_data, (W_out_with_edge, H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :]
        color_data = color_data.unsqueeze(dim=0)

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]

        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]

        return color_data

    def _get_depth(self, index: int):
        depth_data = self.depthloader(index)
        H_out_with_edge = self.H_out + self.H_edge * 2
        W_out_with_edge = self.W_out + self.W_edge * 2
        outsize = (H_out_with_edge, W_out_with_edge)

        if depth_data is not None:
            depth_data = torch.from_numpy(depth_data).float()
            depth_data = F.interpolate(depth_data[None, None], outsize, mode="nearest")[0, 0]
            if self.H_edge > 0:
                edge = self.H_edge
                depth_data = depth_data[edge:-edge, :]
            if self.W_edge > 0:
                edge = self.W_edge
                depth_data = depth_data[:, edge:-edge]

        return depth_data

    def __getitem__(self, index: int):
        color_path = self.color_paths[index]
        color_data = cv2.imread(color_path)

        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx, self.cx, self.fy, self.cy
            color_data = cv2.undistort(color_data, K, self.distortion)

        H_out_with_edge = self.H_out + self.H_edge * 2
        W_out_with_edge = self.W_out + self.W_edge * 2
        outsize = (H_out_with_edge, W_out_with_edge)

        color_data = cv2.resize(color_data, (W_out_with_edge, H_out_with_edge))
        color_data = torch.from_numpy(color_data).permute(2, 0, 1)[[2, 1, 0], :, :]
        color_data = color_data.unsqueeze(dim=0)

        depth_data = self.depthloader(index)
        if depth_data is not None:
            depth_data = torch.from_numpy(depth_data).float()
            depth_data = F.interpolate(depth_data[None, None], outsize, mode="nearest")[0, 0]
            if self.H_edge > 0:
                edge = self.H_edge
                depth_data = depth_data[edge:-edge, :]
            if self.W_edge > 0:
                edge = self.W_edge
                depth_data = depth_data[:, edge:-edge]

        intrinsic = torch.as_tensor([self.fx, self.fy, self.cx, self.cy]).float()
        intrinsic[0] *= W_out_with_edge / self.W
        intrinsic[1] *= H_out_with_edge / self.H
        intrinsic[2] *= W_out_with_edge / self.W
        intrinsic[3] *= H_out_with_edge / self.H

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
            intrinsic[3] -= edge

        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]
            intrinsic[2] -= edge

        if self.poses is not None:
            pose = matrix_to_lie(torch.tensor(self.poses[index])).float()
        else:
            pose = None

        if self.has_dyn_masks and self.mask_paths is not None:
            mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)
            mask = mask == self.background_value
            if self.dilate_masks:
                mask = np.uint8(~mask)
                kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                mask = ~mask.astype(bool)
            mask = np.uint8(mask)
            mask = cv2.resize(mask, (W_out_with_edge, H_out_with_edge))
            mask = torch.from_numpy(mask).bool()
            if self.H_edge > 0:
                edge = self.H_edge
                mask = mask[edge:-edge, :]
            if self.W_edge > 0:
                edge = self.W_edge
                mask = mask[:, edge:-edge]

        if self.return_stat_masks:
            if not self.has_dyn_masks:
                raise Warning(
                    "Warning. Dataset does not have any dynamic masks, please provide some if you want to return them!"
                )
            return index, color_data, depth_data, intrinsic, pose, mask

        return index, color_data, depth_data, intrinsic, pose


class Dataset(BaseDataset):
    """ScanNet/Replica-style dataset loader."""

    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super().__init__(cfg, device)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "color", "*.jpg")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "depth", "*.png")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )
        self.n_valid = self.load_poses(os.path.join(self.input_folder, "pose"))
        self.poses = self.poses[: self.n_valid]
        self.depth_paths = self.depth_paths[: self.n_valid]
        self.color_paths = self.color_paths[: self.n_valid]

        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
            self.depth_paths = self.depth_paths[self.t_start : self.t_stop]
            self.poses = self.poses[self.t_start : self.t_stop]

        self.color_paths = self.color_paths[:: self.stride]
        self.depth_paths = self.depth_paths[:: self.stride]
        self.poses = self.poses[:: self.stride]

        self.n_img = len(self.color_paths)
        print("INFO: {} images got!".format(self.n_img))

    def switch_to_rgbd_gt(self):
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "depth_gt", "*.png")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )
        self.depth_paths = self.depth_paths[: self.n_valid]
        if self.t_stop is not None:
            self.depth_paths = self.depth_paths[self.t_start : self.t_stop]
        self.depth_paths = self.depth_paths[:: self.stride]


class RealSense(BaseDataset):
    """RealSense loader with a ScanNet-like folder layout."""

    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super().__init__(cfg, device)
        self.cfg = cfg

        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "color", "*.jpg"))
            + glob.glob(os.path.join(self.input_folder, "color", "*.png")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "depth", "*.png")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )

        n = min(len(self.color_paths), len(self.depth_paths))
        self.color_paths = self.color_paths[:n]
        self.depth_paths = self.depth_paths[:n]

        self._load_intrinsics_fallback()
        self.png_depth_scale = float(getattr(cfg.data, "png_depth_scale", 1000.0))

        pose_dir = os.path.join(self.input_folder, "pose")
        if os.path.isdir(pose_dir) and len(glob.glob(os.path.join(pose_dir, "*.txt"))) > 0:
            self.n_valid = self.load_poses(pose_dir)
            if self.n_valid < 0:
                self.n_valid = n
            self.n_valid = min(self.n_valid, n)
            self.poses = self.poses[: self.n_valid]
            self.color_paths = self.color_paths[: self.n_valid]
            self.depth_paths = self.depth_paths[: self.n_valid]
        else:
            self.n_valid = n
            self.poses = [np.eye(4, dtype=np.float32) for _ in range(self.n_valid)]

        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
            self.depth_paths = self.depth_paths[self.t_start : self.t_stop]
            self.poses = self.poses[self.t_start : self.t_stop]

        self.color_paths = self.color_paths[:: self.stride]
        self.depth_paths = self.depth_paths[:: self.stride]
        self.poses = self.poses[:: self.stride]

        self.n_img = len(self.color_paths)
        print(f"INFO: {self.n_img} images got! (RealSense)")

    def _load_intrinsics_fallback(self):
        cam = getattr(self.cfg.data, "cam", None)
        have_cfg_intr = False
        if cam is not None:
            keys = ["fx", "fy", "cx", "cy", "H", "W"]
            have_cfg_intr = all(getattr(cam, k, None) is not None for k in keys)

        if have_cfg_intr:
            return

        intr_path = os.path.join(self.input_folder, "intrinsic", "intrinsic_color.txt")
        if os.path.isfile(intr_path):
            K = np.loadtxt(intr_path).astype(np.float32)
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])

            if len(self.color_paths) > 0:
                im = cv2.imread(self.color_paths[0], cv2.IMREAD_COLOR)
                H, W = im.shape[:2]
            else:
                H, W = 1080, 1920

            self.cfg.data.cam.fx = fx
            self.cfg.data.cam.fy = fy
            self.cfg.data.cam.cx = cx
            self.cfg.data.cam.cy = cy
            self.cfg.data.cam.H = int(H)
            self.cfg.data.cam.W = int(W)
            return

        meta_path = os.path.join(self.input_folder, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            intr = meta.get("intrinsics", {})
            fx = float(intr.get("fx", 0.0))
            fy = float(intr.get("fy", 0.0))
            cx = float(intr.get("cx", 0.0))
            cy = float(intr.get("cy", 0.0))
            H = int(intr.get("h", meta.get("color", {}).get("h", 0)))
            W = int(intr.get("w", meta.get("color", {}).get("w", 0)))

            if H == 0 or W == 0:
                if len(self.color_paths) > 0:
                    im = cv2.imread(self.color_paths[0], cv2.IMREAD_COLOR)
                    H, W = im.shape[:2]
                else:
                    H, W = 1080, 1920

            self.cfg.data.cam.fx = fx
            self.cfg.data.cam.fy = fy
            self.cfg.data.cam.cx = cx
            self.cfg.data.cam.cy = cy
            self.cfg.data.cam.H = int(H)
            self.cfg.data.cam.W = int(W)
            return

        if len(self.color_paths) > 0:
            im = cv2.imread(self.color_paths[0], cv2.IMREAD_COLOR)
            H, W = im.shape[:2]
            self.cfg.data.cam.H = int(H)
            self.cfg.data.cam.W = int(W)


dataset_dict = {
    "scannet": Dataset,
    "replica": Dataset,
    "realsense": RealSense,
}
