#!/usr/bin/env python3
"""从 FreeOcc PLY 生成语义着色的高斯点云，可在 CloudCompare/MeshLab 中查看。"""
import argparse, sys, os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plyfile import PlyData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="FreeOcc PLY 路径")
    parser.add_argument("--output", required=True, help="输出 PLY 路径")
    parser.add_argument("--max-points", type=int, default=200000, help="最多保留点数")
    args = parser.parse_args()

    # 读 PLY
    ply = PlyData.read(args.input)
    v = ply["vertex"].data
    N = len(v)
    names = v.dtype.names
    print(f"[INFO] {N} Gaussians, fields: {names}")

    # 语义特征
    ov = sorted([k for k in names if k.startswith("ov_feat_")],
                key=lambda x: int(x.split("_")[-1]))
    n_cls = len(ov)
    print(f"[INFO] {n_cls} semantic classes")

    is_clip_mode = (n_cls > 20)  # > 20 维 → CLIP 特征，不是类别 logits

    if n_cls == 0 or is_clip_mode:
        if is_clip_mode:
            print(f"[INFO] CLIP mode ({n_cls}-dim), using RGB colors (use query_occ.py for text search)")
        else:
            print("[WARN] No ov_feat, using RGB colors")
        labels = np.zeros(N, dtype=int)
        # 用 RGB 着色
        rgb = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
        rgb = 1.0 / (1.0 + np.exp(-rgb))  # sigmoid
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    else:
        sem = np.stack([v[k] for k in ov], axis=1)  # [N, C]
        labels = np.argmax(sem, axis=1)  # [N]
        rgb = None

    # 读类名
    name_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "src", "scannet_utils", "kunlunxai_name.txt")
    if os.path.exists(name_file):
        with open(name_file) as f:
            class_names = [l.strip() for l in f if l.strip()]
    else:
        class_names = [f"class_{i}" for i in range(n_cls)]

    if not is_clip_mode:
        print("[INFO] Class distribution (top 10):")
        for i in range(min(n_cls, len(class_names))):
            cnt = (labels == i).sum()
            print(f"  {cnt:8d}  cls{i} {class_names[i] if i < len(class_names) else ''}")
        print(f"  {'─'*30}")

        # 调色板 (class logit 模式)
        palette = []
        for i in range(n_cls):
            hue = i / max(n_cls, 1) * 180
            c = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
            palette.append(c[::-1].tolist())  # BGR→RGB
        palette = np.array(palette, dtype=np.uint8)

    # 随机子采样
    if N > args.max_points:
        idx = np.random.choice(N, args.max_points, replace=False)
    else:
        idx = np.arange(N)

    # 保存 ASCII PLY (兼容性好)
    xyz = np.stack([v["x"][idx], v["y"][idx], v["z"][idx]], axis=1)
    if is_clip_mode:
        colors = rgb[idx]
    else:
        colors = palette[labels[idx]]

    with open(args.output, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(idx)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(idx)):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                    f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")

    print(f"[INFO] Saved {len(idx)} points → {args.output}")

    # 图例
    legend_h = n_cls * 22 + 20
    legend = np.ones((legend_h, 280, 3), dtype=np.uint8) * 255
    for i in range(min(n_cls, len(class_names))):
        y0 = 15 + i * 22
        cv2.rectangle(legend, (10, y0), (40, y0+16), palette[i][::-1].tolist(), -1)
        cv2.putText(legend, f"cls{i} {class_names[i]}", (55, y0+13),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    legend_path = args.output.replace(".ply", "_legend.png")
    cv2.imwrite(legend_path, legend)
    print(f"[INFO] Legend: {legend_path}")


if __name__ == "__main__":
    main()
