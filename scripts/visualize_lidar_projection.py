#!/usr/bin/env python3
"""可视化所有帧的 LiDAR→相机投影 + 相机原始深度图对比，用于检查外参是否准确。

用法:
    python scripts/visualize_lidar_projection.py \
        --bag_dir /path/to/bag \
        --output_dir /path/to/output_frames \
        [--camera body] \
        [--stride 1] \
        [--time_window_ms 200]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2


def quat_xyzw_to_rotmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def lievec_to_matrix(lie):
    t, q = lie[:3], lie[3:7]
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_rotmat(q)
    T[:3, 3] = t
    return T


def project_points_to_image(pts_lidar, T_cl, fx, fy, cx, cy, H, W, z_min=0.1, z_max=100):
    N = len(pts_lidar)
    pts_h = np.hstack([pts_lidar, np.ones((N, 1))])
    pts_c = (T_cl @ pts_h.T).T[:, :3]

    valid = (pts_c[:, 2] > z_min) & (pts_c[:, 2] < z_max)
    pts_c = pts_c[valid]

    u = fx * pts_c[:, 0] / pts_c[:, 2] + cx
    v = fy * pts_c[:, 1] / pts_c[:, 2] + cy
    z = pts_c[:, 2]

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[in_bounds].astype(int), v[in_bounds].astype(int), z[in_bounds]


def rasterize_depth(u_arr, v_arr, z_arr, H, W):
    depth = np.zeros((H, W), dtype=np.float32)
    order = np.argsort(z_arr)[::-1]
    u_np, v_np, z_np = np.array(u_arr), np.array(v_arr), np.array(z_arr)
    for k in order:
        depth[v_np[k], u_np[k]] = z_np[k]
    return depth


def make_overlay(color_bgr, u_arr, v_arr, z_arr):
    overlay = color_bgr.copy()
    u_np, v_np, z_np = np.array(u_arr), np.array(v_arr), np.array(z_arr)
    if len(z_np) == 0:
        return overlay
    z_min, z_max = z_np.min(), z_np.max()
    for k in range(len(u_np)):
        ratio = (z_np[k] - z_min) / (z_max - z_min + 1e-6)
        if ratio < 0.5:
            b, g, r = 0, int(510 * ratio), 255
        else:
            b, g, r = 0, 255, int(510 * (1 - ratio))
        cv2.circle(overlay, (int(u_np[k]), int(v_np[k])), 1, (b, g, r), -1)
    return overlay


def make_depth_vis(depth, z_min, z_max, colormap=None):
    """深度图着色。colormap: cv2.COLORMAP_JET 等, 或 None=灰度(近白远黑)。"""
    vis = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    mask = depth > 0
    if mask.sum() == 0:
        return vis
    z_norm = np.clip((depth[mask] - z_min) / (z_max - z_min + 1e-6), 0, 1)
    if colormap is None:
        gray_vals = ((1 - z_norm) * 255).astype(np.uint8)  # 近白远黑
        yy, xx = np.where(mask)
        vis[yy, xx] = np.stack([gray_vals] * 3, axis=-1)
    else:
        cmapped = cv2.applyColorMap((z_norm * 255).astype(np.uint8), colormap)
        cm_flat = cmapped.reshape(-1, 3)
        yy, xx = np.where(mask)
        vis[yy, xx] = cm_flat[np.arange(len(yy)) % len(cm_flat)]
    return vis


def depth_msg_to_numpy(msg) -> np.ndarray:
    """sensor_msgs/Image (depth) → [H, W] float32 米."""
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == "16UC1":
        depth = data.view(np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
    elif msg.encoding == "32FC1":
        depth = data.view(np.float32).reshape(msg.height, msg.width)
    else:
        raise ValueError(f"不支持的深度编码: {msg.encoding}")
    return depth


def main():
    parser = argparse.ArgumentParser(description="可视化所有帧的 LiDAR→相机投影 + 相机深度对比")
    parser.add_argument("--bag_dir", type=str, required=True, help="ROS2 bag 目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出帧图片目录")
    parser.add_argument("--camera", type=str, default="body", choices=["body", "head"])
    parser.add_argument("--extrinsic_file", type=str, default=None,
                        help="外参 tfs.json (默认: {bag_dir}/output/tfs.json)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--time_window_ms", type=float, default=200.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--camera_depth_scale", type=float, default=1.0,
                        help="相机深度缩放因子 (如果相机深度值需要缩放)")
    args = parser.parse_args()

    bag_dir = Path(args.bag_dir)
    camera = args.camera
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 外参 ──
    extrinsic_file = args.extrinsic_file or str(bag_dir / "output" / "tfs.json")
    with open(extrinsic_file) as f:
        tfs = json.load(f)
    T_lc = lievec_to_matrix(np.array(tfs["T_lidar_camera"], dtype=np.float64))
    T_cl = np.linalg.inv(T_lc)
    print(f"T_camera→livox:\n{T_lc}\n")
    print(f"T_livox→camera:\n{T_cl}\n")

    # ── 读 bag ──
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores
    store = get_typestore(Stores.ROS2_HUMBLE)

    color_topic = f"/camera/{camera}/color/image_raw"
    depth_topic = f"/camera/{camera}/depth/image_raw"
    camera_info_topic = f"/camera/{camera}/color/camera_info"

    print("读取 bag (一趟扫描)...")
    color_frames = {}       # ts → np.ndarray [H, W, 3] BGR
    depth_frames = {}       # ts → np.ndarray [H, W] float32 meters
    lidar_scans = []        # [(ts, points_Nx3)]
    cam_info = None

    with Reader(bag_dir) as reader:
        for conn, ts, raw in reader.messages():
            if conn.topic == camera_info_topic and cam_info is None:
                msg = store.deserialize_cdr(raw, conn.msgtype)
                K = np.array(msg.k).reshape(3, 3)
                cam_info = {
                    "height": msg.height, "width": msg.width,
                    "fx": K[0, 0], "fy": K[1, 1],
                    "cx": K[0, 2], "cy": K[1, 2],
                }
            elif conn.topic == color_topic:
                msg = store.deserialize_cdr(raw, conn.msgtype)
                data = np.frombuffer(msg.data, dtype=np.uint8)
                if msg.encoding in ("rgb8", "bgr8"):
                    color_frames[ts] = data.reshape(msg.height, msg.width, 3).copy()
                else:
                    raise ValueError(f"不支持的图像编码: {msg.encoding}")
            elif conn.topic == depth_topic:
                msg = store.deserialize_cdr(raw, conn.msgtype)
                depth = depth_msg_to_numpy(msg)
                if args.camera_depth_scale != 1.0:
                    depth *= args.camera_depth_scale
                depth_frames[ts] = depth
            elif conn.topic == "/livox/lidar":
                msg = store.deserialize_cdr(raw, conn.msgtype)
                data = np.frombuffer(msg.data, dtype=np.uint8)
                n_pts = msg.width * msg.height
                if msg.point_step == 16:
                    pts = np.frombuffer(data, dtype=np.float32).reshape(n_pts, 4)[:, :3]
                else:
                    pts = np.zeros((n_pts, 3), dtype=np.float32)
                    for k in range(n_pts):
                        off = k * msg.point_step
                        pts[k] = tuple(np.frombuffer(data[off + j*4:off + j*4 + 4], dtype=np.float32)[0]
                                       for j in range(3))
                lidar_scans.append((ts, pts))

    if cam_info is None:
        print("ERROR: 未读取到 camera_info!")
        sys.exit(1)

    H, W = cam_info["height"], cam_info["width"]
    fx, fy = cam_info["fx"], cam_info["fy"]
    cx, cy = cam_info["cx"], cam_info["cy"]
    print(f"内参: {W}x{H}, fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    color_ts_sorted = sorted(color_frames.keys())
    print(f"共 {len(color_ts_sorted)} 帧相机图像, "
          f"{len(depth_frames)} 帧相机深度, {len(lidar_scans)} 次 LiDAR 扫描")

    end_idx = args.end or len(color_ts_sorted)
    selected_indices = list(range(args.start, min(end_idx, len(color_ts_sorted)), args.stride))

    lidar_ts_arr = np.array([s[0] for s in lidar_scans], dtype=np.int64)
    depth_ts_arr = np.array(sorted(depth_frames.keys()), dtype=np.int64) if depth_frames else np.array([])
    time_window_ns = int(args.time_window_ms * 1_000_000)

    def _find_nearest_cam_depth(target_ts):
        """对 color 帧做最近邻深度帧匹配 (color/depth 时间戳不重合)."""
        if len(depth_ts_arr) == 0:
            return None
        idx = np.searchsorted(depth_ts_arr, target_ts)
        candidates = []
        if idx < len(depth_ts_arr):
            candidates.append(depth_ts_arr[idx])
        if idx > 0:
            candidates.append(depth_ts_arr[idx - 1])
        nearest = min(candidates, key=lambda ts: abs(ts - target_ts))
        if abs(nearest - target_ts) < time_window_ns:
            return depth_frames[nearest]
        return None

    # ── 逐帧处理 ──
    for out_idx, frame_idx in enumerate(selected_indices):
        cam_ts = color_ts_sorted[frame_idx]
        color_img = color_frames[cam_ts]
        cam_depth = _find_nearest_cam_depth(cam_ts)  # [H,W] float32 meters, or None

        # ── LiDAR 投影 ──
        mask = np.abs(lidar_ts_arr - cam_ts) < time_window_ns
        nearby_indices = np.where(mask)[0]

        lidar_depth = np.zeros((H, W), dtype=np.float32)

        if len(nearby_indices) > 0:
            all_pts_l = np.vstack([lidar_scans[i][1] for i in nearby_indices])
            u, v, z = project_points_to_image(all_pts_l, T_cl, fx, fy, cx, cy, H, W)
            n_pts_total = len(all_pts_l)

            if len(u) > 0:
                lidar_depth = rasterize_depth(u, v, z, H, W)
                z_min_lidar = float(z.min())
                z_max_lidar = float(z.max())
                overlay = make_overlay(color_img, u, v, z)
                n_pixels = int((lidar_depth > 0).sum())
            else:
                overlay = color_img.copy()
                z_min_lidar = z_max_lidar = 1.0
                n_pts_total = n_pixels = 0
        else:
            overlay = color_img.copy()
            z_min_lidar = z_max_lidar = 1.0
            n_pts_total = n_pixels = 0

        # ── 确定统一的深度显示范围 ──
        # 取 LiDAR 和相机深度的共同范围
        z_min = z_max = 10.0
        if n_pixels > 0:
            z_min, z_max = z_min_lidar, z_max_lidar
        if cam_depth is not None:
            cam_valid = cam_depth > 0
            if cam_valid.sum() > 0:
                z_min = min(z_min, cam_depth[cam_valid].min())
                z_max = max(z_max, cam_depth[cam_valid].max())

        # ── 各列可见化 ──
        # 列1: LiDAR 叠加图
        col1 = overlay

        # 列2: LiDAR 深度图
        col2 = make_depth_vis(lidar_depth, z_min, z_max)

        # 列3: 相机深度图
        if cam_depth is not None and (cam_depth > 0).sum() > 0:
            col3 = make_depth_vis(cam_depth, z_min, z_max)
            has_cam = True
        else:
            col3 = np.zeros((H, W, 3), dtype=np.uint8)
            has_cam = False

        # ── 合成 4 列 ──
        # 每列加顶部标签条
        label_h = 18

        def add_label(col_img, label, width):
            labeled = np.zeros((H + label_h, width, 3), dtype=np.uint8)
            labeled[:label_h, :] = [40, 40, 40]
            labeled[label_h:, :] = col_img
            cv2.putText(labeled, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            return labeled

        color_img_labeled = add_label(color_img, "RGB", W)
        col1_labeled = add_label(col1, "LiDAR overlay (r=近→g→b=远)", W)
        col2_labeled = add_label(col2, "LiDAR depth (近白远黑)", W)
        col3_labeled = add_label(col3, "Camera depth (近白远黑)" if has_cam else "Camera depth (N/A)", W)

        result = np.hstack([color_img_labeled, col1_labeled, col2_labeled, col3_labeled])

        # 缩放到合适宽度
        target_w = 2560
        scale = target_w / result.shape[1]
        result = cv2.resize(result, (target_w, int(result.shape[0] * scale)))

        # 底部信息条
        cv2.putText(result,
                    f"Frame {frame_idx} | LiDAR: {n_pts_total} pts → {n_pixels} px "
                    f"| depth range [{z_min:.2f}, {z_max:.2f}]m",
                    (4, result.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        out_path = output_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(out_path), result)

        if (out_idx + 1) % 10 == 0 or out_idx == 0:
            print(f"  [{out_idx+1}/{len(selected_indices)}] frame_{frame_idx:04d}.png "
                  f"— LiDAR {n_pts_total}pts/{n_pixels}px, "
                  f"cam_depth={'OK' if has_cam else 'N/A'}, "
                  f"range=[{z_min:.1f}~{z_max:.1f}]m")

    print(f"\n完成! {len(selected_indices)} 张对比图保存到: {output_dir}")
    print(f"四列: RGB | LiDAR叠加 | LiDAR深度 | 相机深度")


if __name__ == "__main__":
    main()
