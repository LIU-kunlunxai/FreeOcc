#!/usr/bin/env python3
"""Logit 模式查询 CLI"""
import argparse
from semantic_query import SemanticQuery

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--name-file", default="src/scannet_utils/kunlunxai_name.txt")
    parser.add_argument("--query", default=None)
    parser.add_argument("--near", default=None)
    parser.add_argument("--proximity", type=float, default=1.5)
    parser.add_argument("--no-ply", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    sq = SemanticQuery(args.ply, args.name_file)

    if args.list:
        for i, n in sq.list_classes():
            print(f"  cls{i}: {n}")
        return

    if not args.query:
        print("请用 --query 指定类别名，或 --list")
        return

    instances = sq.query(args.query, near_name=args.near, proximity=args.proximity)

    print(f"查询: {args.query}")
    if args.near:
        print(f"  空间过滤: near {args.near} (<{args.proximity}m)")
    print(f"  实例数: {len(instances)}")
    sq.print_instances(instances)

    if not args.no_ply and instances:
        for inst in instances:
            out_i = args.ply.replace(".ply", f"_{inst.cls_name}_{instances.index(inst)+1}.ply")
            with open(out_i, "w") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {inst.n_voxels}\n")
                f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
                for p in inst.pts:
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            print(f"    → {out_i}")

if __name__ == "__main__":
    main()
