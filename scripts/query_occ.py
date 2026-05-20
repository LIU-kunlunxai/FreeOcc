#!/usr/bin/env python3
"""在线开集语义查询 — 给定文本，返回 3D 坐标 + 可视化热力图

用法:
    # 加载体素特征并查询
    python scripts/query_occ.py \
        --ply /path/to/mesh/final_rgbd.ply \
        --query "door" \
        --topk 10

    # 也可直接查询 occ 体素 PLY (如已有 occ_voxel_sem_label.ply 且带 ov_feat)
    python scripts/query_occ.py \
        --ply /path/to/mesh/final_rgbd.ply \
        --query "fire extinguisher" \
        --output heatmap.ply
"""

import argparse, os, sys, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="FreeOcc PLY (带 ov_feat)")
    parser.add_argument("--feat-cache", default=None, help="预缓存的体素特征 .pkl")
    parser.add_argument("--query", required=True, help="查询文本")
    parser.add_argument("--topk", type=int, default=50, help="返回前 K 个结果")
    parser.add_argument("--output", default=None, help="输出热力图 PLY")
    parser.add_argument("--max-gaussians", type=int, default=500000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── 1. 加载特征 ──
    if args.feat_cache and os.path.exists(args.feat_cache):
        print(f"[INFO] Loading feature cache: {args.feat_cache}")
        with open(args.feat_cache, "rb") as f:
            data = pickle.load(f)
        features = torch.from_numpy(data["features"].astype(np.float32)).to(device)
        centers = data["centers"]
        grid_size = data["grid_size"]
    else:
        # 直接从 PLY 读取
        from plyfile import PlyData
        ply = PlyData.read(args.ply)
        v = ply["vertex"].data
        N = len(v)
        names = v.dtype.names

        ov_keys = sorted([k for k in names if k.startswith("ov_feat_")],
                         key=lambda x: int(x.split("_")[-1]))
        if not ov_keys:
            print("[ERROR] PLY has no ov_feat_* fields")
            sys.exit(1)

        print(f"[INFO] Loading {N} Gaussians, {len(ov_keys)}-dim features")

        # 子采样
        if N > args.max_gaussians:
            idx = np.random.choice(N, args.max_gaussians, replace=False)
        else:
            idx = np.arange(N)

        xyz = np.stack([v["x"][idx], v["y"][idx], v["z"][idx]], axis=1).astype(np.float32)
        feats = np.stack([v[k][idx] for k in ov_keys], axis=1).astype(np.float32)

        centers = xyz
        features = torch.from_numpy(feats).to(device)
        grid_size = None

    print(f"[INFO] {features.shape[0]} points, {features.shape[1]}-dim features")

    # ── 2. 编码查询文本 ──
    print(f"[INFO] Query: \"{args.query}\"")
    from trident import Trident
    import open_clip

    # 用 Trident 里的 CLIP 模型编码文本
    model = open_clip.create_model("ViT-B/16", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B/16")
    model.eval().to(device)

    text = tokenizer([args.query]).to(device)
    with torch.no_grad():
        query_emb = model.encode_text(text)
        query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)

    # ── 3. 余弦相似度 ──
    features_norm = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
    scores = (features_norm @ query_emb.T).squeeze(-1)  # [N]

    topk = min(args.topk, len(scores))
    top_vals, top_idx = torch.topk(scores, topk)

    print(f"\n  Top-{topk} results:")
    print(f"  {'Rank':<5} {'Score':<8} {'X':<10} {'Y':<10} {'Z':<10}")
    for i in range(min(20, topk)):
        j = top_idx[i].item()
        print(f"  {i+1:<5} {top_vals[i].item():.4f}   {centers[j,0]:.2f}     {centers[j,1]:.2f}     {centers[j,2]:.2f}")

    # ── 4. 热力图 PLY ──
    if args.output and len(centers) > 0:
        import cv2
        scores_np = scores.cpu().numpy()
        s_min, s_max = scores_np.min(), scores_np.max()
        if s_max - s_min < 1e-6:
            s_norm = np.zeros_like(scores_np)
        else:
            s_norm = (scores_np - s_min) / (s_max - s_min)

        jet = cv2.applyColorMap((s_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        colors = jet.reshape(-1, 3)[:, ::-1].astype(np.float64) / 255.0  # BGR→RGB

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        o3d.io.write_point_cloud(args.output, pcd)
        print(f"\n[INFO] Heatmap saved: {args.output}")

    print("[DONE]")


if __name__ == "__main__":
    main()
