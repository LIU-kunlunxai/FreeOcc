#!/usr/bin/env python3
"""FreeOcc 离线地图统一接口 — 替代所有散乱脚本"""

import os, sys, yaml, pickle, numpy as np, cv2
from plyfile import PlyData

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semantic_query import SemanticQuery


class FreeOccMap:
    """加载 FreeOcc 语义地图 + LiDAR 几何，提供统一查询/可视化/导出接口"""

    def __init__(self, occ_ply: str, lidar_pcd: str = None,
                 name_file: str = "src/scannet_utils/kunlunxai_name0521.txt",
                 transform: str = None, lidar_source: str = None,
                 conf_threshold: float = 0.0):
        """occ_ply: 语义体素, lidar_pcd: 目标LiDAR, transform: T_occ→lidar, lidar_source: 源LiDAR(ICP用)"""
        self.occ_path = occ_ply
        self.sq = SemanticQuery(occ_ply, name_file, conf_threshold=conf_threshold)

        # T 矩阵：传了就加载，没传但有俩 LiDAR 就自动 ICP
        T_mat = None
        if transform and os.path.exists(transform):
            T_mat = np.loadtxt(transform)
            print(f"[T] 加载: {transform}")
        elif lidar_pcd and lidar_source and os.path.exists(lidar_pcd) and os.path.exists(lidar_source):
            T_mat = self._auto_icp(lidar_source, lidar_pcd)

        self.T = T_mat
        self._has_T = T_mat is not None
        self._lidar_pts = None
        if lidar_pcd and os.path.exists(lidar_pcd):
            self._lidar_pts = self._load_points(lidar_pcd)
            # 自动裁剪：FreeOcc 点必须在 LiDAR 地图范围内
            margin = 0.5  # bbox 扩展边距
            self._lidar_bbox = np.array([
                self._lidar_pts[:,0].min() - margin, self._lidar_pts[:,0].max() + margin,
                self._lidar_pts[:,1].min() - margin, self._lidar_pts[:,1].max() + margin,
                self._lidar_pts[:,2].min() - margin, self._lidar_pts[:,2].max() + margin,
            ])
            self._clip_occ_to_lidar()
        else:
            self._lidar_bbox = None

    def _auto_icp(self, src_path: str, tgt_path: str) -> np.ndarray:
        """自动 ICP 对齐两个 LiDAR 地图，返回 T_src→tgt 4x4 矩阵"""
        import open3d as o3d
        print(f"[ICP] 自动对齐: {src_path} → {tgt_path}")
        src = o3d.io.read_point_cloud(src_path).voxel_down_sample(0.1)
        tgt = o3d.io.read_point_cloud(tgt_path).voxel_down_sample(0.1)
        result = o3d.pipelines.registration.registration_icp(
            src, tgt, 1.0, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(f"[ICP] Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}")
        return result.transformation

    # ── 内部 ──
    def _load_points(self, path):
        if path.endswith(".ply"):
            v = PlyData.read(path)["vertex"].data
            return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
        else:
            import open3d as o3d
            return np.asarray(o3d.io.read_point_cloud(path).points, dtype=np.float32)

    def _to_lidar(self, xyz: np.ndarray):
        if not self._has_T: return xyz
        h = np.hstack([xyz, np.ones((len(xyz), 1))])
        return (self.T @ h.T).T[:, :3]

    def _clip_occ_to_lidar(self):
        """裁剪语义体素到 LiDAR 地图范围内"""
        xyz = self._to_lidar(self.sq.xyz) if self._has_T else self.sq.xyz
        b = self._lidar_bbox
        inside = (xyz[:,0] >= b[0]) & (xyz[:,0] <= b[1]) & \
                 (xyz[:,1] >= b[2]) & (xyz[:,1] <= b[3]) & \
                 (xyz[:,2] >= b[4]) & (xyz[:,2] <= b[5])
        n_before = len(xyz)
        self.sq.xyz = self.sq.xyz[inside]
        self.sq.labels = self.sq.labels[inside]
        print(f"[裁剪] LiDAR范围内: {inside.sum()}/{n_before} 体素")

    # ── 语义查询 ──
    def query(self, cls: str, near: str = None, proximity: float = 1.5):
        """返回 Instance 列表，center 可转 lidar 坐标"""
        return self.sq.query(cls, near_name=near, proximity=proximity)

    def list_classes(self):
        return self.sq.list_classes()

    # ── Costmap ──
    def export_costmap(self, output_prefix: str, resolution: float = 0.05,
                       z_floor: float = 0.0, z_thresh: float = 0.3, z_ceiling: float = 2.5):
        """从 LiDAR PCD 生成 2D costmap (.pgm + .yaml)"""
        if self._lidar_pts is None:
            raise RuntimeError("需要 lidar_pcd 参数")
        pts = self._lidar_pts
        pts = pts[(pts[:, 2] > z_floor + 0.05) & (pts[:, 2] < z_ceiling)]

        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        margin = 20
        w = int((x_max - x_min) / resolution) + 1 + 2 * margin
        h = int((y_max - y_min) / resolution) + 1 + 2 * margin

        grid = np.zeros((h, w), dtype=np.uint8)
        xi = ((pts[:, 0] - x_min) / resolution + margin).astype(int)
        yi = ((pts[:, 1] - y_min) / resolution + margin).astype(int)
        v = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        obstacle = (pts[v, 2] - z_floor) > z_thresh
        np.add.at(grid, (yi[v][obstacle], xi[v][obstacle]), 1)

        costmap = np.full((h, w), 255, dtype=np.uint8)
        costmap[grid == 0] = 254
        costmap[grid > 0] = 0

        pgm = output_prefix + ".pgm"
        cv2.imwrite(pgm, costmap)
        with open(output_prefix + ".yaml", "w") as f:
            f.write(f"image: {os.path.basename(pgm)}\nresolution: {resolution}\n"
                    f"origin: [{x_min - margin*resolution:.4f}, {y_min - margin*resolution:.4f}, 0.0]\n"
                    f"negate: 0\noccupied_thresh: 0.45\nfree_thresh: 0.55\n")
        return pgm

    # ── 标注 ──
    COLORS = [(0,255,0),(0,0,255),(255,0,0),(255,255,0),(255,0,255),(0,255,255),
              (128,255,0),(255,128,0),(0,128,255),(255,0,128),(128,0,255),(0,255,128)]

    def annotate(self, costmap_img: str, costmap_yaml: str, output: str,
                 queries: list = None, all: bool = False, exclude: list = None,
                 near: str = None, proximity: float = 1.5):
        """在 costmap 上标注语义，支持 --all 和 --exclude"""
        img = cv2.imread(costmap_img)
        if img is None: raise FileNotFoundError(costmap_img)
        with open(costmap_yaml) as f:
            ym = yaml.safe_load(f)
        res, ox, oy = ym["resolution"], ym["origin"][0], ym["origin"][1]
        h, w = img.shape[:2]

        if all:
            queries = [n for _, n in self.list_classes()
                       if not (exclude and any(e.lower() in n.lower() for e in exclude))]
        if not queries:
            queries = []

        for qi, qname in enumerate(queries):
            instances = self.sq.query(qname, near_name=near, proximity=proximity)
            for rank, inst in enumerate(instances):
                x, y, z = self._to_lidar(inst.center[np.newaxis])[0] if self._has_T else inst.center
                px = int((x - ox) / res); py = h - 1 - int((y - oy) / res)
                px, py = np.clip(px, 0, w-1), np.clip(py, 0, h-1)
                c = self.COLORS[(qi + rank) % len(self.COLORS)]
                cv2.circle(img, (px, py), 6, c, -1)
                cv2.putText(img, f"{inst.cls_name}#{rank+1}", (px+8, py+4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

        cv2.imwrite(output, img)
        print(f"Saved: {output}")

    # ── 融合 ──
    def fuse(self, output: str, radius: float = 0.3, max_pts: int = 500000,
             cls_name: str = None):
        """LiDAR几何 + 语义颜色 → fused.ply. cls_name 过滤只显示单类."""
        if self._lidar_pts is None:
            raise RuntimeError("需要 lidar_pcd")
        # 用 sq 内部的 xyz/labels (可能已裁剪), 而非重读 PLY
        occ_xyz = self.sq.xyz.copy()
        occ_xyz = self._to_lidar(occ_xyz) if self._has_T else occ_xyz
        occ_labels = self.sq.labels

        ply = PlyData.read(self.occ_path)
        occ_colors = np.stack([ply["vertex"]["red"], ply["vertex"]["green"],
                               ply["vertex"]["blue"]], axis=1)
        if len(occ_colors) != len(occ_xyz):
            occ_colors = occ_colors[:len(occ_xyz)]

        # 单类过滤
        if cls_name:
            cls_idx = self.sq._find_cls(cls_name)
            cls_mask = occ_labels == cls_idx
            occ_xyz = occ_xyz[cls_mask]
            occ_colors = occ_colors[cls_mask]
            print(f"[融合] 只保留 {cls_name}: {cls_mask.sum()} 体素")

        from scipy.spatial import cKDTree
        tree = cKDTree(occ_xyz)
        dists, idx = tree.query(self._lidar_pts, k=1)
        matched = dists < radius

        colors = np.zeros((len(self._lidar_pts), 3), dtype=np.uint8)
        colors[matched] = occ_colors[idx[matched]]

        # 子采样
        if len(self._lidar_pts) > max_pts:
            mi = np.where(matched)[0]; ui = np.where(~matched)[0]
            nm = min(max_pts//2, len(mi)); nu = min(max_pts-nm, len(ui))
            keep = np.concatenate([
                np.random.choice(mi, nm, replace=False),
                np.random.choice(ui, nu, replace=False)])
            pts, colors = self._lidar_pts[keep], colors[keep]
        else:
            pts = self._lidar_pts

        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
                    f"property float x\nproperty float y\nproperty float z\n"
                    f"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for i in range(len(pts)):
                f.write(f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} "
                        f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")
        print(f"Saved: {output}")


# ── CLI ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FreeOcc 离线地图统一查询/可视化工具")
    parser.add_argument("--occ", default="occ_voxel_sem_label.ply",
                        help="语义体素 PLY 路径 (occ_voxel_sem_label.ply)")
    parser.add_argument("--lidar", default=None,
                        help="目标 LiDAR 地图 PCD 路径 (供 costmap/标注/融合)")
    parser.add_argument("--lidar-source", default=None,
                        help="源 LiDAR 地图 PCD 路径 (FreeOcc 坐标系)，用于不传 --transform 时自动 ICP")
    parser.add_argument("--name-file", default="src/scannet_utils/kunlunxai_name0521.txt",
                        help="语义类别文件")
    parser.add_argument("--transform", default=None,
                        help="坐标系变换矩阵 T_occ→lidar (不传且给了俩 LiDAR 则自动 ICP)")
    parser.add_argument("--action", default="annotate",
                        choices=["query","list","costmap","annotate","fuse"],
                        help="操作: query=查语义, list=列类别, costmap=出2D代价图, annotate=标注, fuse=融合可视化")
    parser.add_argument("--query-cls", nargs="+",
                        help="查询的类别名 (如 door / chair)")
    parser.add_argument("--near-cls", default=None,
                        help="空间参考类 (如 table, 查找XX旁边的YY)")
    parser.add_argument("--proximity", type=float, default=1.5,
                        help="空间参考距离(米)")
    parser.add_argument("--all", action="store_true",
                        help="标注所有类别")
    parser.add_argument("--exclude", nargs="+", default=[],
                        help="标注排除的类别 (如 floor wall)")
    parser.add_argument("--costmap-img", default=None,
                        help="costmap 底图路径 (.pgm)")
    parser.add_argument("--costmap-yaml", default=None,
                        help="costmap 参数路径 (.yaml)")
    parser.add_argument("--output", default="output.png",
                        help="输出文件路径")
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="costmap 分辨率(米)")
    parser.add_argument("--conf", type=float, default=0.0,
                        help="语义置信度阈值 (0~1, 越大越严格, 默认0不设限)")
    args = parser.parse_args()

    m = FreeOccMap(args.occ, args.lidar, args.name_file, args.transform, args.lidar_source, conf_threshold=args.conf)

    if args.action == "query":
        insts = m.query(args.query_cls[0], args.near_cls, args.proximity)
        for ri, inst in enumerate(insts):
            center = m._to_lidar(inst.center[np.newaxis])[0] if m._has_T else inst.center
            dist = f"距 {args.near_cls} {inst.dist_to_ref:.2f}m" if inst.dist_to_ref else ""
            print(f"  {inst.cls_name}#{ri+1}: "
                  f"({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) {inst.n_voxels}v {dist}")
    elif args.action == "list":
        for i, n in m.list_classes():
            print(f"  cls{i}: {n}")
    elif args.action == "costmap":
        m.export_costmap(args.output.replace(".pgm","").replace(".yaml",""), args.resolution)
    elif args.action == "annotate":
        m.annotate(args.costmap_img, args.costmap_yaml, args.output,
                   queries=args.query_cls, all=args.all, exclude=args.exclude,
                   near=args.near_cls, proximity=args.proximity)
    elif args.action == "fuse":
        cls = args.query_cls[0] if args.query_cls else None
        m.fuse(args.output, cls_name=cls)
