import numpy as np
import cv2
import os
import tempfile


def read_color_images(bag_path: str, topic: str) -> list:
    """Read raw color images from bag. Returns list of (timestamp_sec, bgr_image)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image as RosImage

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    images = []
    while reader.has_next():
        _, data, t = reader.read_next()
        msg = deserialize_message(data, RosImage)
        ts = t / 1e9
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append((ts, img))
    return images


def read_depth_images(bag_path: str, topic: str) -> list:
    """Read uint16 depth images from bag. Returns list of (timestamp_sec, depth_mm)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image as RosImage

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    frames = []
    while reader.has_next():
        _, data, t = reader.read_next()
        msg = deserialize_message(data, RosImage)
        ts = t / 1e9
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        frames.append((ts, depth))
    return frames


def read_h265_images(bag_path: str, topic: str) -> list:
    """Read H265 compressed images from bag. Returns list of (timestamp_sec, bgr_image)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    h265_packets = []
    timestamps = []
    while reader.has_next():
        _, data, t = reader.read_next()
        msg = deserialize_message(data, CompressedImage)
        h265_packets.append(bytes(msg.data))
        timestamps.append(t / 1e9)

    if not h265_packets:
        return []

    tmp = tempfile.mktemp(suffix='.h265')
    with open(tmp, 'wb') as f:
        for pkt in h265_packets:
            f.write(pkt)

    cap = cv2.VideoCapture(tmp)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())
    cap.release()
    os.unlink(tmp)

    n = min(len(frames), len(timestamps))
    return [(timestamps[i], frames[i]) for i in range(n)]


def depth_to_pointcloud(depth_mm: np.ndarray, fx: float, fy: float,
                        cx: float, cy: float, max_depth: float = 8.0,
                        min_depth: float = 0.3) -> np.ndarray:
    """Convert depth image (uint16 mm) to camera-frame point cloud (N, 3)."""
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = (depth_m > min_depth) & (depth_m < max_depth)
    v, u = np.where(valid)
    z = depth_m[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.column_stack([x, y, z]).astype(np.float32)
