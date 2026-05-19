import numpy as np
import os


def load_pcd_xyz(filepath: str) -> np.ndarray:
    """Load XYZ from binary PCD file. Returns (N, 3) float32 array."""
    with open(filepath, "rb") as f:
        n_points = 0
        sizes = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("POINTS"):
                n_points = int(line.split()[1])
            elif line.startswith("SIZE"):
                sizes = [int(x) for x in line.split()[1:]]
            elif line == "DATA binary":
                break
        point_size = sum(sizes)
        data = f.read(n_points * point_size)

    if point_size == 12:
        dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    else:
        dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('_', f'V{point_size - 12}')])
    arr = np.frombuffer(data, dtype=dt, count=n_points)
    return np.column_stack([arr['x'], arr['y'], arr['z']])


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel grid downsample. Returns subset of original points."""
    vi = np.floor(points / voxel_size).astype(np.int32)
    _, idx = np.unique(vi, axis=0, return_index=True)
    return points[idx]


def save_pcd_binary(filepath: str, points: np.ndarray, colors: np.ndarray = None):
    """Save point cloud as binary PCD. colors: (N,3) uint8 BGR."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    n = len(points)

    if colors is not None:
        dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
        data = np.zeros(n, dtype=dt)
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]
        cols = colors.astype(np.uint32)
        data['rgb'] = (cols[:, 2] << 16) | (cols[:, 1] << 8) | cols[:, 0]
        fields_line = "FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1"
    else:
        dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        data = np.zeros(n, dtype=dt)
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]
        fields_line = "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1"

    header = (
        f"# .PCD v0.7 - Point Cloud Data file format\n"
        f"VERSION 0.7\n"
        f"{fields_line}\n"
        f"WIDTH {n}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA binary\n"
    )

    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())
