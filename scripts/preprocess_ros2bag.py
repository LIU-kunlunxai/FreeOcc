#!/usr/bin/env python3
"""
FreeOcc ROS2 Bag Preprocessor

从 ROS2 bag + LiDAR SLAM 位姿 + 外参标定文件中提取并组织 FreeOcc 所需的完整数据集。

用法:
    python scripts/preprocess_ros2bag.py \
        --bag_dir /path/to/bag_20260518_150039 \
        --output_dir /path/to/output_sequence \
        --camera body \
        --project_lidar_depth

输入:
    - ROS2 bag (db3): color/image_raw, depth/image_raw, camera_info, livox/lidar
    - LiDAR SLAM 位姿: output/poses_lidar.txt (timestamp + 7维 lie 向量 c2w)
    - 外参: output/tfs.json

输出目录结构:
    output_sequence/
        color/         0.jpg, 1.jpg, ...
        depth/         0.png, 1.png, ...
        pose/          0.txt, 1.txt, ...  (4×4 camera-to-world 矩阵)
        config.yaml    内参等配置信息
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ──────────────────────────────────────────────
# 几何工具 (避免依赖 lietorch / pytorch3d)
# ──────────────────────────────────────────────

def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """lietorch quaternion [x, y, z, w] → 3×3 rotation matrix (Hamilton convention)."""
    x, y, z, w = q
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])
    return R


def lievec_to_matrix(lie: np.ndarray) -> np.ndarray:
    """7维 lie 向量 [tx, ty, tz, qx, qy, qz, qw] → 4×4 齐次矩阵 (c2w)."""
    t = lie[:3]
    q = lie[3:7]
    R = quat_xyzw_to_rotmat(q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def matrix_to_c2w_txt(T: np.ndarray) -> str:
    """4×4 矩阵 → FreeOcc pose/*.txt 格式 (行主序, 空格分隔)."""
    lines = []
    for row in T:
        lines.append(" ".join(f"{v:.12f}" for v in row))
    return "\n".join(lines)


def compose_pose(T_wl: np.ndarray, T_lc: np.ndarray) -> np.ndarray:
    """
    组合位姿: T_wc = T_wl @ T_lc
    T_wl: LiDAR → world (c2w for LiDAR)
    T_lc: camera → LiDAR (extrinsic, from ROS TF)
    返回: camera → world (c2w for camera)
    """
    return T_wl @ T_lc


# ──────────────────────────────────────────────
# ROS2 Bag 读取 (使用 rosbags 库)
# ──────────────────────────────────────────────

def read_camera_info(bag_dir: Path, camera: str) -> dict:
    """从 bag 中读取第一帧 camera_info, 提取内参."""
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores

    topic = f"/camera/{camera}/color/camera_info"
    store = get_typestore(Stores.ROS2_HUMBLE)

    with Reader(bag_dir) as reader:
        for conn, ts, raw in reader.messages():
            if conn.topic == topic:
                msg = store.deserialize_cdr(raw, conn.msgtype)
                K = np.array(msg.k).reshape(3, 3)
                return {
                    "height": msg.height,
                    "width": msg.width,
                    "fx": K[0, 0],
                    "fy": K[1, 1],
                    "cx": K[0, 2],
                    "cy": K[1, 2],
                    "K": K,
                    "d": list(msg.d),
                    "distortion_model": msg.distortion_model,
                }

    raise RuntimeError(f"未找到 topic: {topic}")


def extract_images(bag_dir: Path, camera: str):
    """从 bag 提取 RGB 图像 + 深度图像 + 时间戳."""
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores

    color_topic = f"/camera/{camera}/color/image_raw"
    depth_topic = f"/camera/{camera}/depth/image_raw"
    store = get_typestore(Stores.ROS2_HUMBLE)

    color_frames = {}   # timestamp_ns -> np.ndarray [H, W, 3] BGR uint8
    depth_frames = {}   # timestamp_ns -> np.ndarray [H, W] float32 (meters)

    with Reader(bag_dir) as reader:
        for conn, ts, raw in reader.messages():
            if conn.topic == color_topic:
                msg = store.deserialize_cdr(raw, conn.msgtype)
                img = _image_msg_to_numpy(msg)
                color_frames[ts] = img
            elif conn.topic == depth_topic:
                msg = store.deserialize_cdr(raw, conn.msgtype)
                depth = _depth_msg_to_numpy(msg)
                depth_frames[ts] = depth

    # 对齐时间戳: 只保留同时有 color 和 depth 的帧
    common_ts = sorted(set(color_frames.keys()) & set(depth_frames.keys()))
    if not common_ts:
        # 如果没对齐, 只用 color
        common_ts = sorted(color_frames.keys())
        depths = [None] * len(common_ts)
    else:
        depths = [depth_frames[ts] for ts in common_ts]

    colors = [color_frames[ts] for ts in common_ts]
    return common_ts, colors, depths


def _image_msg_to_numpy(msg) -> np.ndarray:
    """sensor_msgs/Image → numpy [H, W, 3] BGR uint8."""
    import numpy as np
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if msg.encoding == "rgb8":
        img = data.reshape(msg.height, msg.width, 3)
    elif msg.encoding == "bgr8":
        img = data.reshape(msg.height, msg.width, 3)
    elif msg.encoding == "bgra8":
        img = data.reshape(msg.height, msg.width, 4)[:, :, :3]
    elif msg.encoding == "rgba8":
        img = data.reshape(msg.height, msg.width, 4)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "yuv422":
        img = data.reshape(msg.height, msg.width, 2)
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)
    else:
        raise ValueError(f"不支持的图像编码: {msg.encoding}")

    return img


def _depth_msg_to_numpy(msg) -> np.ndarray:
    """sensor_msgs/Image (depth) → numpy [H, W] float32 (meters)."""
    import numpy as np
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if msg.encoding == "16UC1":
        depth = data.view(np.uint16).reshape(msg.height, msg.width).astype(np.float32)
        depth /= 1000.0  # 毫米 → 米
    elif msg.encoding == "32FC1":
        depth = data.view(np.float32).reshape(msg.height, msg.width)
    else:
        raise ValueError(f"不支持的深度编码: {msg.encoding}")

    return depth


# ──────────────────────────────────────────────
# LiDAR 深度投影 (可选)
# ──────────────────────────────────────────────

def extract_lidar_scans(bag_dir: Path):
    """从 bag 提取 LiDAR 点云 (时间戳 + XYZ points in livox_frame)."""
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores

    store = get_typestore(Stores.ROS2_HUMBLE)
    scans = []  # [(timestamp_ns, points_Nx3)]

    with Reader(bag_dir) as reader:
        for conn, ts, raw in reader.messages():
            if conn.topic == "/livox/lidar":
                msg = store.deserialize_cdr(raw, conn.msgtype)
                points = _pointcloud2_to_xyz(msg)
                scans.append((ts, points))

    return scans


def _pointcloud2_to_xyz(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 → numpy [N, 3] (x, y, z)."""
    import numpy as np
    # PointCloud2 fields: x, y, z are float32 at offsets 0, 4, 8
    data = np.frombuffer(msg.data, dtype=np.uint8)
    n_points = msg.width * msg.height

    if msg.point_step == 16 and len(data) == n_points * 16:
        pts = np.frombuffer(data, dtype=np.float32).reshape(n_points, 4)
        return pts[:, :3]
    else:
        # 通用解析
        xyz = np.zeros((n_points, 3), dtype=np.float32)
        for i in range(n_points):
            offset = i * msg.point_step
            xyz[i, 0] = np.frombuffer(data[offset:offset+4], dtype=np.float32)[0]
            xyz[i, 1] = np.frombuffer(data[offset+4:offset+8], dtype=np.float32)[0]
            xyz[i, 2] = np.frombuffer(data[offset+8:offset+12], dtype=np.float32)[0]
        return xyz


def project_lidar_to_camera(
    lidar_scans: list,
    camera_timestamps: list,
    T_lc: np.ndarray,
    intrinsics: dict,
    time_window_ns: int = 50_000_000,  # 50ms
) -> list:
    """
    将 LiDAR 点云投影到相机平面生成深度图.

    对每个相机时间戳, 在时间窗口内收集 LiDAR 扫描,
    变换到相机坐标系后投影, 取每像素最近深度.
    """
    H, W = intrinsics["height"], intrinsics["width"]
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    # T_lc 是 camera→lidar, 变换 lidar 点: p_cam = T_lc @ p_lidar
    # 实际上我们需要的是 lidar→camera 的变换 = inv(T_lc)
    # Wait, T_lc 是 camera_frame → livox_frame
    # 要将 lidar 点 P_l 变换到 camera 坐标系: P_c = inv(T_lc) @ P_l
    T_cl = np.linalg.inv(T_lc)  # livox→camera

    depth_maps = []

    for cam_ts in camera_timestamps:
        # 找到时间窗口内的 LiDAR 扫描
        nearby_points = []
        for lidar_ts, pts in lidar_scans:
            if abs(lidar_ts - cam_ts) < time_window_ns and len(pts) > 0:
                nearby_points.append(pts)

        if not nearby_points:
            depth_maps.append(None)
            continue

        all_pts_l = np.vstack(nearby_points)

        # 变换到相机坐标系
        N = all_pts_l.shape[0]
        pts_l_h = np.hstack([all_pts_l, np.ones((N, 1))])
        pts_c = (T_cl @ pts_l_h.T).T[:, :3]

        # 在前方的点
        valid = pts_c[:, 2] > 0.1
        pts_c = pts_c[valid]

        if len(pts_c) == 0:
            depth_maps.append(None)
            continue

        # 投影
        u = (fx * pts_c[:, 0] / pts_c[:, 2] + cx).round().astype(int)
        v = (fy * pts_c[:, 1] / pts_c[:, 2] + cy).round().astype(int)
        z = pts_c[:, 2]

        # 图像范围内
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]

        # 取每个像素最近的点
        depth_img = np.zeros((H, W), dtype=np.float32)
        idx = np.argsort(z)  # 远处 → 近处, 近的会覆盖远的
        for i in idx:
            depth_img[v[i], u[i]] = z[i]

        depth_maps.append(depth_img)

    return depth_maps


# ──────────────────────────────────────────────
# 位姿读取与匹配
# ──────────────────────────────────────────────

def read_tf_extrinsic(tfs_json_path: Path) -> np.ndarray:
    """从 tfs.json 读取外参, 返回 4×4 矩阵 (camera→livox frame)."""
    with open(tfs_json_path) as f:
        tfs = json.load(f)

    lie_vec = np.array(tfs["T_lidar_camera"], dtype=np.float64)
    # tfs.json 中 from_frame=livox, to_frame=camera_body
    # ROS TF: 该变换把 camera_frame 的点转到 livox_frame
    # 所以 T_lc = camera → livox
    return lievec_to_matrix(lie_vec)


def read_lidar_poses(poses_file: Path) -> dict:
    """
    读取 LiDAR SLAM 位姿文件.

    格式: timestamp_ns tx ty tz qx qy qz qw (每行一个位姿, c2w, lietorch 规范)
    返回: {timestamp_ns: 4×4 c2w matrix}
    """
    poses = {}
    with open(poses_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            ts = int(parts[0])
            lie = np.array([float(x) for x in parts[1:8]], dtype=np.float64)
            T_wl = lievec_to_matrix(lie)  # LiDAR → world (c2w)
            poses[ts] = T_wl
    return poses


def match_camera_to_lidar_poses(
    camera_timestamps: list,
    lidar_poses: dict,
    max_dt_ns: int = 100_000_000,  # 100ms
) -> list:
    """
    为每个相机帧匹配最近的 LiDAR 位姿.

    返回: [(camera_ts, T_wc), ...]  T_wc 是 4×4 camera→world 矩阵
    没有匹配到位姿的相机帧返回 (ts, None).
    """
    lidar_tss = sorted(lidar_poses.keys())
    if not lidar_tss:
        return [(ts, None) for ts in camera_timestamps]

    matched = []
    for cam_ts in camera_timestamps:
        # 二分查找最近的 LiDAR 时间戳
        idx = np.searchsorted(lidar_tss, cam_ts)
        if idx == 0:
            nearest = lidar_tss[0]
        elif idx == len(lidar_tss):
            nearest = lidar_tss[-1]
        else:
            left = lidar_tss[idx - 1]
            right = lidar_tss[idx]
            nearest = left if (cam_ts - left) < (right - cam_ts) else right

        dt = abs(cam_ts - nearest)
        if dt < max_dt_ns:
            matched.append((cam_ts, lidar_poses[nearest]))
        else:
            matched.append((cam_ts, None))
            print(f"  警告: 相机帧 {cam_ts} 无匹配 LiDAR 位姿 (最近 Δt={dt/1e6:.1f}ms)")

    return matched


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FreeOcc ROS2 Bag 预处理器")
    parser.add_argument("--bag_dir", type=str, required=True,
                        help="ROS2 bag 目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--camera", type=str, default="body",
                        choices=["body", "head"],
                        help="使用哪个相机 (默认: body, 因为外参标定了 body camera)")
    parser.add_argument("--poses_file", type=str, default=None,
                        help="LiDAR SLAM 位姿文件 (默认: {bag_dir}/output/poses_lidar.txt)")
    parser.add_argument("--extrinsic_file", type=str, default=None,
                        help="外参 tfs.json (默认: {bag_dir}/output/tfs.json)")
    parser.add_argument("--project_lidar_depth", action="store_true",
                        help="将 LiDAR 点云投影为深度图 (替代相机深度)")
    parser.add_argument("--time_window_ms", type=float, default=50.0,
                        help="LiDAR 扫描时间窗口 (毫秒, 默认 50)")
    parser.add_argument("--max_pose_dt_ms", type=float, default=100.0,
                        help="相机帧与 LiDAR 位姿的最大时间差 (毫秒, 默认 100)")

    args = parser.parse_args()

    bag_dir = Path(args.bag_dir)
    output_dir = Path(args.output_dir)
    camera = args.camera

    if not bag_dir.exists():
        print(f"错误: bag 目录不存在: {bag_dir}")
        sys.exit(1)

    # 默认路径
    poses_file = args.poses_file or str(bag_dir / "output" / "poses_lidar.txt")
    extrinsic_file = args.extrinsic_file or str(bag_dir / "output" / "tfs.json")
    poses_file = Path(poses_file)
    extrinsic_file = Path(extrinsic_file)

    print("=" * 60)
    print("FreeOcc ROS2 Bag 预处理器")
    print("=" * 60)
    print(f"  Bag 目录:    {bag_dir}")
    print(f"  输出目录:    {output_dir}")
    print(f"  相机:        {camera}")
    print(f"  位姿文件:    {poses_file}")
    print(f"  外参文件:    {extrinsic_file}")
    print(f"  LiDAR 深度:  {'是' if args.project_lidar_depth else '否 (使用相机深度)'}")
    print()

    # ── Step 1: 读取内参 ──
    print("[1/6] 读取相机内参...")
    intrinsics = read_camera_info(bag_dir, camera)
    print(f"  分辨率: {intrinsics['width']}×{intrinsics['height']}")
    print(f"  内参: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
          f"cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")

    # ── Step 2: 提取图像 ──
    print("[2/6] 提取图像...")
    cam_timestamps, color_imgs, depth_imgs = extract_images(bag_dir, camera)
    print(f"  提取到 {len(cam_timestamps)} 帧 (color+depth)")

    if len(cam_timestamps) == 0:
        print("错误: 未提取到任何图像!")
        sys.exit(1)

    # ── Step 3: 读取外参 ──
    print("[3/6] 读取外参...")
    if not extrinsic_file.exists():
        print(f"  错误: 外参文件不存在: {extrinsic_file}")
        sys.exit(1)
    T_lc = read_tf_extrinsic(extrinsic_file)  # camera→livox
    print(f"  T_lidar_camera:\n{T_lc}")

    # ── Step 4: 读取 LiDAR 位姿 ──
    print("[4/6] 读取 LiDAR SLAM 位姿...")
    if not poses_file.exists():
        print(f"  错误: 位姿文件不存在: {poses_file}")
        sys.exit(1)
    lidar_poses = read_lidar_poses(poses_file)
    print(f"  读取到 {len(lidar_poses)} 个 LiDAR 位姿")

    # ── Step 5: 时间对齐 ──
    print("[5/6] 时间对齐...")
    max_pose_dt_ns = int(args.max_pose_dt_ms * 1_000_000)
    matched = match_camera_to_lidar_poses(cam_timestamps, lidar_poses, max_pose_dt_ns)

    # 过滤掉没有位姿的帧
    valid_frames = [(i, ts, T_wl) for i, (ts, T_wl) in enumerate(matched) if T_wl is not None]
    print(f"  成功匹配 {len(valid_frames)}/{len(cam_timestamps)} 帧")
    if len(valid_frames) == 0:
        print("错误: 没有成功匹配的帧!")
        sys.exit(1)

    # ── Step 5.5: LiDAR 深度投影 (可选) ──
    if args.project_lidar_depth:
        print("[5.5/6] LiDAR 深度投影...")
        lidar_scans = extract_lidar_scans(bag_dir)
        print(f"  读取到 {len(lidar_scans)} 个 LiDAR 扫描")
        time_window_ns = int(args.time_window_ms * 1_000_000)
        lidar_depths = project_lidar_to_camera(
            lidar_scans, cam_timestamps, T_lc, intrinsics, time_window_ns
        )
        # 用 LiDAR 深度替换相机深度 (如果 LiDAR 深度有效)
        replaced_count = 0
        for i, lidar_depth in enumerate(lidar_depths):
            if lidar_depth is not None and (lidar_depth > 0).sum() > 100:
                depth_imgs[i] = lidar_depth
                replaced_count += 1
        print(f"  替换了 {replaced_count}/{len(cam_timestamps)} 帧的深度图")

    # ── Step 6: 写入输出 ──
    print("[6/6] 写入 FreeOcc 数据集...")
    (output_dir / "color").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth").mkdir(parents=True, exist_ok=True)
    (output_dir / "pose").mkdir(parents=True, exist_ok=True)

    for out_idx, (i, cam_ts, T_wl) in enumerate(valid_frames):
        # 计算相机 c2w 位姿
        T_wc = compose_pose(T_wl, T_lc)

        # color
        color_img = color_imgs[i]
        color_path = output_dir / "color" / f"{out_idx}.jpg"
        cv2.imwrite(str(color_path), color_img)

        # depth (保存为 16-bit PNG, 单位毫米)
        depth_img = depth_imgs[i]
        depth_path = output_dir / "depth" / f"{out_idx}.png"
        if depth_img is not None and depth_img.size > 0:
            depth_mm = (depth_img * 1000).astype(np.uint16)
            cv2.imwrite(str(depth_path), depth_mm)
        else:
            # 空白深度
            blank = np.zeros((intrinsics["height"], intrinsics["width"]), dtype=np.uint16)
            cv2.imwrite(str(depth_path), blank)

        # pose (4x4 camera-to-world 矩阵, 行主序)
        pose_path = output_dir / "pose" / f"{out_idx}.txt"
        pose_path.write_text(matrix_to_c2w_txt(T_wc))

    print(f"  写入 {len(valid_frames)} 帧完成")
    print(f"  {output_dir}/color/{len(valid_frames)} 个 .jpg")
    print(f"  {output_dir}/depth/{len(valid_frames)} 个 .png")
    print(f"  {output_dir}/pose/{len(valid_frames)} 个 .txt")

    # ── 生成 config.yaml ──
    config_path = output_dir / "config_info.yaml"
    config_path.write_text(f"""# FreeOcc 配置信息 (自动生成)
# 复制以下参数到运行命令中

# 数据路径
data.input_folder: {output_dir.resolve()}

# 相机内参
data.cam.H: {intrinsics['height']}
data.cam.W: {intrinsics['width']}
data.cam.H_out: {intrinsics['height']}
data.cam.W_out: {intrinsics['width']}
data.cam.fx: {intrinsics['fx']}
data.cam.fy: {intrinsics['fy']}
data.cam.cx: {intrinsics['cx']}
data.cam.cy: {intrinsics['cy']}

# 深度图缩放 (16-bit PNG 毫米 → 米)
data.png_depth_scale: 1000.0
""")

    # ── 打印运行命令 ──
    print()
    print("=" * 60)
    print("预处理完成!")
    print("=" * 60)
    print()
    print("运行 FreeOcc:")
    print(f"  cd /home/liu/workspace/FreeOcc")
    print(f"  python run.py \\")
    print(f"    mode=rgbd \\")
    print(f"    use_gt_poses=True \\")
    print(f"    data.input_folder={output_dir.resolve()} \\")
    print(f"    data.cam.H={intrinsics['height']} \\")
    print(f"    data.cam.W={intrinsics['width']} \\")
    print(f"    data.cam.H_out={intrinsics['height']} \\")
    print(f"    data.cam.W_out={intrinsics['width']} \\")
    print(f"    data.cam.fx={intrinsics['fx']} \\")
    print(f"    data.cam.fy={intrinsics['fy']} \\")
    print(f"    data.cam.cx={intrinsics['cx']} \\")
    print(f"    data.cam.cy={intrinsics['cy']} \\")
    print(f"    data.png_depth_scale=1000.0 \\")
    print(f"    mapping.online_opt.filter.bin_th=0.03 \\")
    print(f"    mapping.loss.supervise_with_prior=False")
    print()


if __name__ == "__main__":
    main()
