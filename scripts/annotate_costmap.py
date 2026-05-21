#!/usr/bin/env python3
"""在 costmap 上标注语义查询结果，直接调用 SemanticQuery"""

import argparse, os, numpy as np, cv2, yaml
from semantic_query import SemanticQuery

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="语义 PLY")
    parser.add_argument("--name-file", default="src/scannet_utils/kunlunxai_name.txt")
    parser.add_argument("--costmap-img", required=True)
    parser.add_argument("--costmap-yaml", required=True)
    parser.add_argument("--query", nargs="+", default=None, help="查询类别(可多个), 与 --all 二选一")
    parser.add_argument("--all", action="store_true", help="标注所有已知类别")
    parser.add_argument("--exclude", nargs="+", default=[], help="排除的类别名 (如 floor)")
    parser.add_argument("--near", default=None)
    parser.add_argument("--proximity", type=float, default=1.5)
    parser.add_argument("--transform", default=None, help="4x4 变换矩阵")
    parser.add_argument("--output", default="costmap_annotated.png")
    args = parser.parse_args()

    sq = SemanticQuery(args.ply, args.name_file)

    queries = args.query or []
    if args.all:
        queries = [n for _, n in sq.list_classes()
                   if not any(e.lower() in n.lower() for e in args.exclude)]
        print(f"标注 {len(queries)} 类 (排除: {args.exclude})")
    if not queries:
        print("请用 --query 指定类别，或用 --all 标注全部")
        return

    T = np.loadtxt(args.transform) if args.transform else None

    img = cv2.imread(args.costmap_img)
    if img is None:
        raise FileNotFoundError(args.costmap_img)

    with open(args.costmap_yaml) as f:
        y = yaml.safe_load(f)
    res = y["resolution"]
    ox, oy = y["origin"][0], y["origin"][1]
    h, w = img.shape[:2]

    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
              (0, 128, 255), (255, 0, 128), (128, 0, 255), (0, 255, 128)]

    for qi, qname in enumerate(queries):
        instances = sq.query(qname, near_name=args.near, proximity=args.proximity)
        n_inst = len(instances)
        # 只标注中心点 (太多实例会画不下)
        for rank, inst in enumerate(instances):
            if not args.all and n_inst > 10 and rank >= 5:
                break  # 单类太多实例时只画前 5 个
            x, y, z = inst.center
            if T is not None:
                p = np.array([x, y, z, 1.0])
                p = T @ p
                x, y, z = p[0], p[1], p[2]
            px = int((x - ox) / res)
            py = h - 1 - int((y - oy) / res)
            px = np.clip(px, 0, w - 1)
            py = np.clip(py, 0, h - 1)
            c = colors[(qi + rank) % len(colors)]
            cv2.circle(img, (px, py), 6, c, -1)
            label = f"{inst.cls_name}#{rank+1}"
            cv2.putText(img, label, (px+8, py+4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

    cv2.imwrite(args.output, img)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
