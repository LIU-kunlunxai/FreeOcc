#!/usr/bin/env python3
"""多段 PLY 合并到统一语义占用体素中，支持离线建图 + 在线查询。"""

import argparse, os, sys, pickle, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plyfile import PlyData
from src.gs2occ.localagg_prob.local_aggregate_prob import LocalAggregator


def load_gs_from_ply(path: str, device: torch.device):
    """加载 FreeOcc PLY 的 3DGS 参数."""
    ply = PlyData.read(path)
    v = ply["vertex"].data
    names = v.dtype.names
    n = len(v)

    xyz = torch.from_numpy(np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)).to(device)

    def _sigmoid(x): return 1.0 / (1.0 + torch.exp(-x))

    opa = _sigmoid(torch.from_numpy(v["opacity"].astype(np.float32)).to(device))
    if opa.ndim == 1: opa = opa.unsqueeze(-1)

    scl = torch.from_numpy(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)).to(device)
    scales = torch.exp(scl).clamp(min=1e-6)

    quat = torch.from_numpy(np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)).to(device)
    w, x, y, zz = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
    norm = torch.sqrt(w*w + x*x + y*y + zz*zz + 1e-12)
    w, x, y, zz = w/norm, x/norm, y/norm, zz/norm
    R = torch.zeros((quat.shape[0], 3, 3), device=device)
    R[:,0,0]=1-2*(y*y+zz*zz); R[:,0,1]=2*(x*y-zz*w); R[:,0,2]=2*(x*zz+y*w)
    R[:,1,0]=2*(x*y+zz*w); R[:,1,1]=1-2*(x*x+zz*zz); R[:,1,2]=2*(y*zz-x*w)
    R[:,2,0]=2*(x*zz-y*w); R[:,2,1]=2*(y*zz+x*w); R[:,2,2]=1-2*(x*x+y*y)
    cov = R @ torch.diag_embed(scales**2) @ R.transpose(1,2)

    # 语义特征
    ov_keys = sorted([k for k in names if k.startswith("ov_feat_")],
                     key=lambda x: int(x.split("_")[-1]))
    sem = torch.from_numpy(np.stack([v[k] for k in ov_keys], axis=1).astype(np.float32)).to(device) if ov_keys else None

    return xyz, scales, cov, opa, sem, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plies", nargs="+", required=True, help="多个 FreeOcc PLY 路径")
    parser.add_argument("--output", required=True, help="输出占位体素 PLY")
    parser.add_argument("--feat-out", default=None, help="输出 CLIP 特征缓存 (.npy)")
    parser.add_argument("--grid-size", type=float, default=0.1)
    parser.add_argument("--thr", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 1. 加载所有 PLY 高斯
    all_xyz, all_sc, all_cov, all_opa, all_sem = [], [], [], [], []
    total = 0
    for p in args.plies:
        xyz, sc, cov, opa, sem, n = load_gs_from_ply(p, device)
        print(f"  {p}: {n} Gaussians")
        all_xyz.append(xyz); all_sc.append(sc); all_cov.append(cov)
        all_opa.append(opa); all_sem.append(sem)
        total += n

    xyz = torch.cat(all_xyz); scales = torch.cat(all_sc); cov = torch.cat(all_cov)
    opa = torch.cat(all_opa); sem = torch.cat(all_sem) if all_sem[0] is not None else None
    print(f"  Total: {total} Gaussians")

    # 2. 全局 bbox
    lo = torch.quantile(xyz, 0.01, dim=0) - 0.5
    hi = torch.quantile(xyz, 0.99, dim=0) + 0.5
    print(f"  BBox: [{lo.tolist()} → {hi.tolist()}]")

    # 3. 构建体素网格
    bsz = hi - lo
    H = int((bsz[0]/args.grid_size).ceil()) + 1
    W = int((bsz[1]/args.grid_size).ceil()) + 1
    D = int((bsz[2]/args.grid_size).ceil()) + 1
    print(f"  Grid: {H}×{W}×{D} = {H*W*D} voxels")

    xs = torch.arange(H, device=device, dtype=torch.float32)
    ys = torch.arange(W, device=device, dtype=torch.float32)
    zs = torch.arange(D, device=device, dtype=torch.float32)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    pts = torch.stack([X, Y, Z], dim=-1) * args.grid_size + lo[None,None,None,:]
    pts = pts.reshape(-1, 3)

    # 4. 过滤 bbox 内的高斯
    m = (xyz[:,0]>=lo[0])&(xyz[:,0]<=hi[0])&(xyz[:,1]>=lo[1])&(xyz[:,1]<=hi[1])&(xyz[:,2]>=lo[2])&(xyz[:,2]<=hi[2])
    xyz_f, sc_f, cov_f, opa_f = xyz[m], scales[m], cov[m], opa[m]
    sem_f = sem[m] if sem is not None else None
    print(f"  Gaussians in bbox: {m.sum().item()}")

    # 5. LocalAggregator 投影
    agg = LocalAggregator(scale_multiplier=3.0, H=H, W=W, D=D,
                          pc_min=lo.cpu().tolist(), grid_size=args.grid_size,
                          radii_min=1).to(device)

    with torch.no_grad():
        logits, bin_logits, density = agg(
            pts=pts.unsqueeze(0), means3D=xyz_f.unsqueeze(0),
            opas=opa_f.unsqueeze(0),
            semantics=sem_f.unsqueeze(0) if sem_f is not None else torch.ones(1, xyz_f.shape[0], 1, device=device),
            scales=sc_f.unsqueeze(0), cov3D=cov_f.unsqueeze(0),
            metas=None, origin_use=lo)

    occ_3d = 1.0 - torch.exp(-density.reshape(H, W, D))
    occ_np = occ_3d.cpu().numpy()

    # 6. 保存
    mask = occ_np > args.thr
    n_occ = int(mask.sum())
    print(f"  Occupied voxels: {n_occ}")

    idx = np.argwhere(mask)
    centers = lo.cpu().numpy() + (idx.astype(np.float64) + 0.5) * args.grid_size

    # 密度着色
    vals = occ_np[mask]
    vals = np.clip(vals / (vals.max()+1e-6), 0, 1)
    colors = np.repeat(vals[:,None], 3, axis=1).astype(np.float64)

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    o3d.io.write_point_cloud(args.output, pcd)
    print(f"  Saved: {args.output}")

    # 7. 导出体素语义特征 (用于在线开集查询)
    if args.feat_out and sem_f is not None:
        # logits: [1, H*W*D, C]
        logits_3d = logits.reshape(H, W, D, -1).cpu().numpy()
        feats_occ = logits_3d[mask]  # [N_occ, C]
        data = {
            "features": feats_occ.astype(np.float16),
            "centers": centers,
            "grid_size": args.grid_size,
            "occ_values": occ_np[mask],
            "bbox_min": lo.cpu().numpy(),
            "grid_dims": (H, W, D),
        }
        with open(args.feat_out, "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved features: {args.feat_out} ({feats_occ.shape})")

    print("[DONE]")


if __name__ == "__main__":
    main()
