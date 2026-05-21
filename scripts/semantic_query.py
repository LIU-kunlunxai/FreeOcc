#!/usr/bin/env python3
"""语义查询类 — 可被其他脚本 import 使用"""

import os, sys
import numpy as np
from plyfile import PlyData
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Instance:
    """单个语义实例"""
    cls_id: int
    cls_name: str
    pts: np.ndarray       # [N, 3] 体素坐标
    center: np.ndarray    # [3] 中心坐标
    n_voxels: int
    dist_to_ref: Optional[float] = None  # 到参考类别的距离


class SemanticQuery:
    def __init__(self, ply_path: str, name_file: str = "src/scannet_utils/kunlunxai_name.txt"):
        with open(name_file) as f:
            self.names = [l.strip() for l in f if l.strip()]
        ply = PlyData.read(ply_path)
        v = ply["vertex"].data
        self.xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
        self._load_labels(v)

    def _load_labels(self, v):
        ov_keys = sorted([k for k in v.dtype.names if k.startswith("ov_feat_")],
                         key=lambda x: int(x.split("_")[-1]))
        if ov_keys:
            ov = np.stack([v[k] for k in ov_keys], axis=1)
            self.labels = np.argmax(ov, axis=1)
        else:
            import cv2
            rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
            self.labels = np.zeros(len(self.xyz), dtype=int)
            for i in range(len(self.names)):
                hue = i / max(len(self.names), 1) * 180
                c = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
                self.labels[np.all(rgb == c, axis=1)] = i

    def _find_cls(self, name: str) -> int:
        matches = [i for i, n in enumerate(self.names) if name.lower() in n.lower()]
        if not matches:
            raise ValueError(f"类别未找到: {name}. 可用: {self.names}")
        return matches[0]

    def list_classes(self):
        return list(enumerate(self.names))

    def query(self, cls_name: str, near_name: Optional[str] = None,
              proximity: float = 1.5, eps: float = 0.3, min_samples: int = 5
              ) -> List[Instance]:
        """查询指定类别的实例，可选空间过滤"""
        from sklearn.cluster import DBSCAN

        cls_idx = self._find_cls(cls_name)
        pts = self.xyz[self.labels == cls_idx]
        if len(pts) == 0:
            return []

        cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
        instances = []
        for cid in sorted(set(cluster.labels_)):
            if cid == -1:
                continue
            c_pts = pts[cluster.labels_ == cid]
            instances.append(Instance(
                cls_id=cls_idx, cls_name=self.names[cls_idx],
                pts=c_pts, center=c_pts.mean(axis=0), n_voxels=len(c_pts)
            ))

        if near_name:
            from scipy.spatial import cKDTree
            near_idx = self._find_cls(near_name)
            near_pts = self.xyz[self.labels == near_idx]
            tree = cKDTree(near_pts)
            filtered = []
            for inst in instances:
                d, _ = tree.query(inst.center)
                if d < proximity:
                    inst.dist_to_ref = float(d)
                    filtered.append(inst)
            filtered.sort(key=lambda x: x.dist_to_ref or 999)
            return filtered

        return instances

    def print_instances(self, instances: List[Instance]):
        if not instances:
            print("  无匹配")
            return
        for rank, inst in enumerate(instances):
            extra = ""
            if inst.dist_to_ref is not None:
                extra = f", 距参考类 {inst.dist_to_ref:.2f}m"
            print(f"  {inst.cls_name} #{rank+1}: {inst.n_voxels} voxels, "
                  f"中心=({inst.center[0]:.2f}, {inst.center[1]:.2f}, {inst.center[2]:.2f}){extra}")
