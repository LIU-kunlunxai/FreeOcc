#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Example:
python dataloader.py \
  --scene-name scene0000_00 \
  --scene-occ-root ./data/sample_scene_occ \
  --occscannet-root ./data/sample_occscannet

"""

import os
import glob
import pickle
import argparse

import numpy as np
import cv2
import torch


def load_full_scene_occ(
    scene_name: str,
    scene_occ_root: str,
    voxel_size: float = 0.08,
    to_torch: bool = False,
):

    pkg_path = os.path.join(scene_occ_root, "global_occ_package", f"{scene_name}.pkl")
    if not os.path.isfile(pkg_path):
        raise FileNotFoundError(f"global_occ_package not found: {pkg_path}")

    with open(pkg_path, "rb") as f:
        scene_pkg = pickle.load(f)

    scene_dim = np.asarray(scene_pkg["scene_dim"], dtype=np.int32)        # (3,)
    global_labels = np.asarray(scene_pkg["global_labels"])                # (Nx,Ny,Nz)
    global_pts = np.asarray(scene_pkg["global_pts"], dtype=np.float32)    # (Nx,Ny,Nz,3)
    global_mask = np.asarray(scene_pkg["global_mask"], dtype=bool)        # (Nx,Ny,Nz)

    scene_origin = np.array(
        [
            global_pts[..., 0].min(),
            global_pts[..., 1].min(),
            global_pts[..., 2].min(),
        ],
        dtype=np.float32,
    )

    scene_size = voxel_size * scene_dim.astype(np.float32)

    out = {
        "scene_name": scene_name,
        "scene_dim": scene_dim,
        "scene_size": scene_size,
        "occ_labels": global_labels,
        "occ_points": global_pts,
        "occ_mask": global_mask,
        "scene_origin": scene_origin,
    }

    if "valid_img_paths" in scene_pkg:
        out["valid_img_paths"] = scene_pkg["valid_img_paths"]

    if to_torch:
        for k, v in list(out.items()):
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v)

    return out


def load_occscannet_scene_images(
    scene_name: str,
    occscannet_root: str,
    max_frames: int | None = None,
    load_depth: bool = False,
):

    gather_dir = os.path.join(occscannet_root, "gathered_data", scene_name)
    posed_dir = os.path.join(occscannet_root, "posed_images", scene_name)

    if not os.path.isdir(gather_dir):
        raise FileNotFoundError(f"gathered_data dir not found: {gather_dir}")
    if not os.path.isdir(posed_dir):
        raise FileNotFoundError(f"posed_images dir not found: {posed_dir}")

    pkl_paths = glob.glob(os.path.join(gather_dir, "*.pkl"))
    if len(pkl_paths) == 0:
        raise RuntimeError(f"No pkl files found in {gather_dir}")

    def _frame_id(p):
        base = os.path.basename(p)   # '00032.pkl'
        return int(os.path.splitext(base)[0])

    pkl_paths = sorted(pkl_paths, key=_frame_id)

    if max_frames is not None and max_frames > 0:
        pkl_paths = pkl_paths[:max_frames]

    frames = []
    for pkl_path in pkl_paths:
        frame_id = os.path.splitext(os.path.basename(pkl_path))[0]

        with open(pkl_path, "rb") as f:
            mono = pickle.load(f)

        cam_pose = mono.get("cam_pose", None)
        intrinsic = mono.get("intrinsic", None)
        voxel_origin = mono.get("voxel_origin", None)

        rgb_path = os.path.join(posed_dir, f"{frame_id}.jpg")
        if os.path.isfile(rgb_path):
            rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                print(f"[WARN] Fail to read image: {rgb_path}")
                rgb = None
            else:
                rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        else:
            print(f"[WARN] RGB image not found: {rgb_path}")
            rgb = None

        depth_path = None
        depth = None
        if load_depth:
            depth_path = os.path.join(posed_dir, f"{frame_id}.png")
            if os.path.isfile(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            else:
                print(f"[WARN] Depth image not found: {depth_path}")
                depth_path = None

        frames.append(
            dict(
                frame_id=frame_id,
                rgb=rgb,
                depth=depth,
                cam_pose=cam_pose,
                intrinsic=intrinsic,
                voxel_origin=voxel_origin,
                rgb_path=rgb_path,
                depth_path=depth_path,
                pkl_path=pkl_path,
            )
        )

    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Load full-scene occupancy GT (EmbodiedOcc-ScanNet) "
                    "and RGB frames (Occ-ScanNet)."
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        required=True,
        help="Scene name, for example scene0000_00.",
    )
    parser.add_argument(
        "--scene-occ-root",
        type=str,
        default="./data/scene_occ",
        help="Root directory containing global_occ_package.",
    )
    parser.add_argument(
        "--occscannet-root",
        type=str,
        default="./data/occscannet",
        help="Occ-ScanNet root containing gathered_data and posed_images.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.08,
        help="Voxel size in meters. The default follows EmbodiedOcc / Occ-ScanNet.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to load. Use <= 0 to load all frames.",
    )
    parser.add_argument(
        "--to-torch",
        action="store_true",
        help="Convert full-scene occupancy GT arrays to torch.Tensor.",
    )
    parser.add_argument(
        "--load-depth",
        action="store_true",
        help="Also load PNG depth files from posed_images when present.",
    )
    parser.add_argument(
        "--valid-img-root-in-pkl",
        type=str,
        default="",
        help=(
            "Optional source root embedded in valid_img_paths inside the GT pkl. "
            "When set, it is replaced by --occscannet-root for local loading."
        ),
    )

    args = parser.parse_args()

    # ---- 1. Load full-scene occupancy GT ----
    scene_data = load_full_scene_occ(
        scene_name=args.scene_name,
        scene_occ_root=args.scene_occ_root,
        voxel_size=args.voxel_size,
        to_torch=args.to_torch,
    )

    if "valid_img_paths" in scene_data and args.valid_img_root_in_pkl:
        scene_data["valid_img_paths"] = [
            t.replace(args.valid_img_root_in_pkl, args.occscannet_root)
            for t in scene_data["valid_img_paths"]
        ]

    print("=== Global OCC (EmbodiedOcc-ScanNet) ===")
    print("scene_name:", scene_data["scene_name"])
    print("scene_dim:", scene_data["scene_dim"])
    print("scene_size (m):", scene_data["scene_size"])
    print("labels shape:", scene_data["occ_labels"].shape)
    print("points shape:", scene_data["occ_points"].shape)
    print("mask shape:", scene_data["occ_mask"].shape)
    print("origin:", scene_data["scene_origin"])
    if "valid_img_paths" in scene_data:
        print("num valid_img_paths (from pkl):", len(scene_data["valid_img_paths"]))

    # ---- 2. Load RGB frames for the scene from Occ-ScanNet ----
    max_frames = None if args.max_frames <= 0 else args.max_frames
    frames = load_occscannet_scene_images(
        scene_name=args.scene_name,
        occscannet_root=args.occscannet_root,
        max_frames=max_frames,
        load_depth=args.load_depth,
    )

    print("\n=== Occ-ScanNet posed_images & gathered_data ===")
    print(f"#frames loaded: {len(frames)}")
    if len(frames) > 0 and frames[0]["rgb"] is not None:
        print("first frame_id:", frames[0]["frame_id"])
        print("first rgb shape:", frames[0]["rgb"].shape)
        if frames[0]["cam_pose"] is not None:
            print("first cam_pose:\n", frames[0]["cam_pose"])
        if frames[0]["intrinsic"] is not None:
            print("first intrinsic:\n", frames[0]["intrinsic"])

    print("\n[Done]")


if __name__ == "__main__":
    main()
