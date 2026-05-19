import os
from typing import List, Optional, Tuple
import gc
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torch.multiprocessing as mp
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from .gaussian_splatting.gui import gui_utils
from .gaussian_splatting.scene.gaussian_model import GaussianModel
from .gaussian_splatting.camera_utils import Camera
from .gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, focal2fov
from .utils.multiprocessing_utils import clone_obj
from .geom import lie_to_matrix
from .gs2occ.localagg_prob.local_aggregate_prob import LocalAggregator
from .scannet_utils.dataloader import load_full_scene_occ
from .scannet_utils.eval_utils import SSCMetricsTorch
from dataclasses import dataclass


def _quat_mul_wxyz(q1, q2):
    # q = q1 ⊗ q2, both (...,4) in (w,x,y,z)
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def _rotmat_to_quat_wxyz(R):
    """R: (...,3,3) -> q: (...,4) (w,x,y,z)"""
    # vectorized, stable-ish branch
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = torch.empty(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)

    # case 1: trace > 0
    mask = t > 0
    if mask.any():
        tr = t[mask]
        r = torch.sqrt(1.0 + tr)
        q_w = 0.5 * r
        s = 0.5 / r
        Rm = R[mask]
        q_x = (Rm[..., 2, 1] - Rm[..., 1, 2]) * s
        q_y = (Rm[..., 0, 2] - Rm[..., 2, 0]) * s
        q_z = (Rm[..., 1, 0] - Rm[..., 0, 1]) * s
        q[mask] = torch.stack([q_w, q_x, q_y, q_z], dim=-1)

    # case 2: trace <= 0 -> pick major diagonal
    mask2 = ~mask
    if mask2.any():
        Rm = R[mask2]
        d0, d1, d2 = Rm[..., 0, 0], Rm[..., 1, 1], Rm[..., 2, 2]
        # select max diag
        idx = torch.stack([d0, d1, d2], dim=-1).argmax(dim=-1)

        q_sub = torch.empty(Rm.shape[:-2] + (4,), device=Rm.device, dtype=Rm.dtype)

        # idx==0
        m0 = idx == 0
        if m0.any():
            r = torch.sqrt(1.0 + d0[m0] - d1[m0] - d2[m0])
            s = 0.5 / r
            R0 = Rm[m0]
            q_x = 0.5 * r
            q_w = (R0[..., 2, 1] - R0[..., 1, 2]) * s
            q_y = (R0[..., 0, 1] + R0[..., 1, 0]) * s
            q_z = (R0[..., 0, 2] + R0[..., 2, 0]) * s
            q_sub[m0] = torch.stack([q_w, q_x, q_y, q_z], dim=-1)

        # idx==1
        m1 = idx == 1
        if m1.any():
            r = torch.sqrt(1.0 + d1[m1] - d0[m1] - d2[m1])
            s = 0.5 / r
            R1 = Rm[m1]
            q_y = 0.5 * r
            q_w = (R1[..., 0, 2] - R1[..., 2, 0]) * s
            q_x = (R1[..., 0, 1] + R1[..., 1, 0]) * s
            q_z = (R1[..., 1, 2] + R1[..., 2, 1]) * s
            q_sub[m1] = torch.stack([q_w, q_x, q_y, q_z], dim=-1)

        # idx==2
        m2 = idx == 2
        if m2.any():
            r = torch.sqrt(1.0 + d2[m2] - d0[m2] - d1[m2])
            s = 0.5 / r
            R2 = Rm[m2]
            q_z = 0.5 * r
            q_w = (R2[..., 1, 0] - R2[..., 0, 1]) * s
            q_x = (R2[..., 0, 2] + R2[..., 2, 0]) * s
            q_y = (R2[..., 1, 2] + R2[..., 2, 1]) * s
            q_sub[m2] = torch.stack([q_w, q_x, q_y, q_z], dim=-1)
        q[mask2] = q_sub

    return q

@dataclass
class FrameGaussians:
    means: Optional[torch.Tensor] = None      # (Ni,3) world
    rgbs: Optional[torch.Tensor] = None       # (Ni,3) float[0,1]
    scales: Optional[torch.Tensor] = None     # (Ni,3) positive
    rotations: Optional[torch.Tensor] = None  # (Ni,4) quat (w,x,y,z) or (x,y,z,w), treated as a 4D normalized quaternion here
    opacities: Optional[torch.Tensor] = None  # (Ni,1)
    mask: Optional[torch.Tensor] = None       # (Ni,) flattened pixel indices for stable sampling
    cam_id: Optional[int] = None
    ov_feat: Optional[torch.Tensor] = None

    def cuda(self):
        self.means = self.means.cuda() if self.means is not None else None
        self.rgbs = self.rgbs.cuda() if self.rgbs is not None else None
        self.scales = self.scales.cuda() if self.scales is not None else None
        self.rotations = self.rotations.cuda() if self.rotations is not None else None
        self.opacities = self.opacities.cuda() if self.opacities is not None else None
        self.mask = self.mask.cuda() if self.mask is not None else None
        self.ov_feat = self.ov_feat.cuda() if self.ov_feat is not None else None
        return self

    def cpu(self):
        self.means = self.means.cpu() if self.means is not None else None
        self.rgbs = self.rgbs.cpu() if self.rgbs is not None else None
        self.scales = self.scales.cpu() if self.scales is not None else None
        self.rotations = self.rotations.cpu() if self.rotations is not None else None
        self.opacities = self.opacities.cpu() if self.opacities is not None else None
        self.mask = self.mask.cpu() if self.mask is not None else None
        self.ov_feat = self.ov_feat.cpu() if self.ov_feat is not None else None
        return self

# -------------------------
# View container
# -------------------------
@dataclass
class View:
    image_chw: torch.Tensor  # (3,H,W) float[0,1], original resolution
    w2c: torch.Tensor        # (4,4) world->cam
    fx: float
    fy: float
    cx: float
    cy: float
    znear: float = 0.01
    zfar: float = 100.0
    ts: int = None

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

def gaussians_to_occ(
    points: torch.Tensor,
    colors: torch.Tensor,
    scales: torch.Tensor,
    rot_mat: torch.Tensor,
    opacity: torch.Tensor,
    semantics: torch.Tensor,
    scene_data: dict,
    scale_multiplier: float = 3.0,
    radii_min: int = 1,
    owner=None,
) -> torch.Tensor:
    """
    Use voxel centers from the global occupancy package as query points.
    Build Gaussians from PLY points and use LocalAggregator to estimate each voxel density.

    Returns:
        densitynp: [X, Y, Z] numpy array
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build Gaussian parameters
    points = points.to(device)
    N = points.shape[0]

    print(f"[INFO] #Gaussians = {N}")

    means3D = points
    N = means3D.shape[0]

    sc2 = scales.square()                 # (N,3)

    # (R @ diag(sc2)) is equivalent to scaling the columns of R by sc2.
    R_scaled = rot_mat * sc2[:, None, :]  # (N,3,3)
    cov3D = R_scaled @ rot_mat.transpose(-1, -2)

    # 2) Use GT occupancy voxel centers as query points.
    occ_pts = scene_data["occ_points"]          # (X,Y,Z,3) numpy or torch
    occ_dim = scene_data["scene_dim"]           # (3,)  [X,Y,Z]

    scene_size = scene_data["scene_size"]       # float
    scene_origin = scene_data["scene_origin"]   # [3]
    voxel_size = scene_size[0] / occ_dim[0]

    if isinstance(occ_pts, np.ndarray):
        occ_pts_t = torch.from_numpy(occ_pts.reshape(-1, 3)).to(device)
    else:
        occ_pts_t = occ_pts.reshape(-1, 3).to(device)

    H, W, D = int(occ_dim[0]), int(occ_dim[1]), int(occ_dim[2])

    # 3) Build LocalAggregator.
    agg = LocalAggregator(
        scale_multiplier=scale_multiplier,
        H=H,
        W=W,
        D=D,
        pc_min=scene_origin.tolist(),  # Keep this consistent with origin_use.
        grid_size=voxel_size,
        radii_min=radii_min,
    ).to(device)

    # 4) Add batch dimensions and call forward.
    pts_b = occ_pts_t.unsqueeze(0)        # [1,N_pts,3]

    origin_use = torch.from_numpy(scene_origin).cuda().float()

    origin_t = origin_use.view(1, 3)
    size_t = torch.from_numpy(scene_size).to(origin_t)
    max_corner = origin_t + size_t

    # means3D_b: [1, N_gs, 3]
    inside_box = (means3D > origin_t) & (means3D < max_corner)
    in_range_mask = inside_box.all(dim=-1)  # [N_gs] bool

    n_in = int(in_range_mask.sum().item())
    print(f"[INFO] Gaussians inside scene bbox: {n_in}/{in_range_mask.numel()}")

    if owner is not None:
        owner.last_gauss_inside = int(n_in)
        owner.last_gauss_total = int(in_range_mask.numel())

    # Guard: if no Gaussians are inside the bbox, return all zeros to avoid LocalAggregator failures.
    if n_in == 0:
        print("[WARN] gaussians_to_occ: no Gaussians inside scene bbox, return empty occupancy.")
        sem_pred_empty = torch.zeros((1, H * W * D), dtype=torch.long, device=device)
        return sem_pred_empty

    means3D_b = means3D[in_range_mask].unsqueeze(0)      # [1,N_gs,3]
    scales_b = scales[in_range_mask].unsqueeze(0)        # [1,N_gs,3]
    cov3D_b = cov3D[in_range_mask].unsqueeze(0)          # [1,N_gs,3,3]
    opas_b = opacity[in_range_mask].unsqueeze(0)         # [1,N_gs,1]
    semantics_b = semantics[in_range_mask].unsqueeze(0)  # [1,N_gs,C]
    with torch.no_grad():
        logits, bin_logits, density = agg(
            pts=pts_b.contiguous(),
            means3D=means3D_b.contiguous(),
            opas=opas_b.contiguous(),
            semantics=semantics_b.contiguous(),
            scales=scales_b.contiguous(),
            cov3D=cov3D_b.contiguous(),
            metas=None,
            origin_use=origin_use,
        )

    sem_pred = torch.argmax(logits, dim=1) + 1
    occ_pred = (bin_logits > 0.5).long()
    sem_pred[occ_pred == 0] = 0

    return sem_pred

# -------------------------
# Projection (supports principal point cx,cy)
# -------------------------
def projection_from_intrinsics(
    fx: float, fy: float, cx: float, cy: float,
    W: int, H: int, znear: float, zfar: float,
    device: torch.device
) -> torch.Tensor:
    """
    Off-center perspective projection matrix from pinhole intrinsics.
    This helps if cx/cy != W/2,H/2.
    """
    fx = float(fx); fy = float(fy); cx = float(cx); cy = float(cy)
    W = float(W); H = float(H)

    # Frustum bounds at near plane (camera coords: +x right, +y down, +z forward)
    left   = (-cx) / fx * znear
    right  = (W - cx) / fx * znear
    top    = (cy) / fy * znear
    bottom = -(H - cy) / fy * znear

    P = torch.zeros((4, 4), device=device, dtype=torch.float32)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P

def render_one_view(
    g,
    view: View,
    GaussianRasterizer,
    GaussianRasterizationSettings,
    bg_rgb=(0.0, 0.0, 0.0),
    scale_modifier: float = 1.0,
    debug: bool = False,
):

    means3D = g.get_xyz
    device = means3D.device
    gt = view.image_chw.to(device=device, dtype=torch.float32)
    _, H, W = gt.shape

    screenspace_points = torch.zeros_like(means3D, requires_grad=True)
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = W / (2.0 * float(view.fx))
    tanfovy = H / (2.0 * float(view.fy))

    world_view_transform = view.w2c.to(device=device, dtype=torch.float32).transpose(0, 1).contiguous()
    proj_raw = projection_from_intrinsics(
        view.fx, view.fy, view.cx, view.cy,
        W=W, H=H,
        znear=view.znear, zfar=view.zfar,
        device=device,
    ).transpose(0, 1).contiguous()
    full_proj = (world_view_transform.unsqueeze(0).bmm(proj_raw.unsqueeze(0))).squeeze(0).contiguous()
    campos = torch.inverse(world_view_transform)[3, :3].contiguous()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(H),
        image_width=int(W),
        tanfovx=float(tanfovx),
        tanfovy=float(tanfovy),
        bg=torch.tensor(bg_rgb, device=device, dtype=torch.float32),
        scale_modifier=float(scale_modifier),
        viewmatrix=world_view_transform,
        projmatrix=full_proj,
        projmatrix_raw=proj_raw,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=bool(debug),
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = g.get_xyz
    means2D = screenspace_points
    colors_precomp = g._features_dc[..., 0].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    # Use the current learnable scale/rotation/opacity directly.
    scales = g.get_scaling
    rotations = g.get_rotation
    opacities = g.get_opacity

    color, radii, depth, opacity_img, n_touched = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        shs=None,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        theta=None,
        rho=None,
    )

    return color.clamp(0.0, 1.0), radii, depth, opacity_img

def optimize_scale_rotation(
    g,
    views: List[View],
    GaussianRasterizer,
    GaussianRasterizationSettings,
    iters: int = 20,
    lr_scale: float = 1e-4,
    lr_rot: float = 1e-4,
    lr_opa: float = 1e-4,
    views_per_iter: int = 1,
    bg_rgb=(0.0, 0.0, 0.0),
    scale_modifier: float = 1.0,
    seed: int = 0,
    verbose_every: int = 20,
):
    device = g._surface_xyz.device if hasattr(g, "_surface_xyz") else g._xyz.device
    torch.manual_seed(seed)

    # freeze everything except scaling/rotation
    if hasattr(g, "_surface_xyz"):
        g._surface_xyz.requires_grad_(False)
    if hasattr(g, "_features_dc"):
        g._features_dc.requires_grad_(False)
    if hasattr(g, "_features_rest"):
        g._features_rest.requires_grad_(False)

    g._opacity.requires_grad_(True)
    g._scaling.requires_grad_(True)

    # no rot update
    g._rotation.requires_grad_(False)

    optim = torch.optim.Adam(
        [
            {"params": [g._scaling], "lr": lr_scale},
            # {"params": [g._rotation], "lr": lr_rot},
            {"params": [g._opacity], "lr": lr_opa},
        ],
        betas=(0.9, 0.999),
        eps=1e-15,
    )

    num_views = len(views)

    for it in range(1, iters + 1):
        optim.zero_grad(set_to_none=True)

        if views_per_iter >= num_views:
            idxs = list(range(num_views))
        else:
            perm = torch.randperm(num_views, device=device)
            idxs = perm[:views_per_iter].tolist()
            # idxs = torch.randint(0, num_views, (views_per_iter,), device=device).tolist()

        loss = 0.0
        for idx in idxs:
            v = views[idx]
            # Detach all input tensors to avoid backpropagating through stale graphs multiple times.
            v_detached = View(
                image_chw=v.image_chw.detach(),
                w2c=v.w2c.detach(),
                fx=float(v.fx),
                fy=float(v.fy),
                cx=float(v.cx),
                cy=float(v.cy),
                znear=float(v.znear),
                zfar=float(v.zfar),
            )

            pred, radii, render_depth, render_alpha = render_one_view(
                g=g,
                view=v_detached,
                GaussianRasterizer=GaussianRasterizer,
                GaussianRasterizationSettings=GaussianRasterizationSettings,
                bg_rgb=bg_rgb,
                scale_modifier=scale_modifier,
                debug=False,
            )
            gt = v_detached.image_chw.to(device=pred.device, dtype=torch.float32)

            loss = loss + ((pred - gt).abs() * render_alpha).sum() / render_alpha.sum().clamp(0.1)

        loss = loss / float(len(idxs))

        loss.backward()
        optim.step()

        # Keep quaternions normalized.
        with torch.no_grad():
            g._rotation.copy_(F.normalize(g._rotation, dim=-1, eps=1e-8))

    return g

def extract_occ_points_from_labels(labels, scene_data, min_label=0):
    # labels: [X,Y,Z] or [N_vox], flattened in the same order as occ_points.
    occ_pts = scene_data['occ_points']  # [N_vox,3] or [X,Y,Z,3], depending on the stored format.

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.asarray(labels)

    if isinstance(occ_pts, torch.Tensor):
        occ_pts = occ_pts.detach().cpu().numpy()
    else:
        occ_pts = np.asarray(occ_pts)

    labels_flat = labels.reshape(-1)
    pts_flat = occ_pts.reshape(-1, 3)
    assert labels_flat.shape[0] == pts_flat.shape[0], \
        f"labels_flat={labels_flat.shape}, pts_flat={pts_flat.shape}"

    mask = labels_flat > min_label
    return pts_flat[mask]

def extract_gt_occupied_points(scene_data: dict, min_label: int = 0) -> np.ndarray:
    """
    Extract points with label > min_label from scene_data['occ_points'] and scene_data['occ_labels'].
    Returns:
        gt_pts_occ: [N_gt, 3] numpy
    """
    labels = scene_data["occ_labels"]
    pts = scene_data["occ_points"]

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.asarray(labels)

    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    else:
        pts = np.asarray(pts)

    mask = labels > min_label
    gt_pts_occ = pts[mask]

    print(f"[INFO] GT occupied points: {gt_pts_occ.shape[0]}")
    return gt_pts_occ

def estimate_global_scale(pred_pts, gt_pts, method: str = "bbox") -> float:
    """
    Estimate a global scale s so that pred_pts * s matches the scale of gt_pts.
    This ignores rotation and translation and only compares overall scene size.

    Args:
        pred_pts: [N_pred, 3] numpy / torch
        gt_pts:   [N_gt, 3]   numpy / torch
        method:   "bbox" or "var"

    Returns:
        scale: float
    """
    if isinstance(pred_pts, torch.Tensor):
        pred_pts = pred_pts.detach().cpu().numpy()
    else:
        pred_pts = np.asarray(pred_pts)

    if isinstance(gt_pts, torch.Tensor):
        gt_pts = gt_pts.detach().cpu().numpy()
    else:
        gt_pts = np.asarray(gt_pts)

    assert pred_pts.ndim == 2 and pred_pts.shape[1] == 3
    assert gt_pts.ndim == 2 and gt_pts.shape[1] == 3

    eps = 1e-8

    if method == "var":
        # Align using the average radius around the mean center.
        cp = pred_pts.mean(axis=0)
        cg = gt_pts.mean(axis=0)
        r2_pred = np.sum((pred_pts - cp) ** 2, axis=1).mean()
        r2_gt = np.sum((gt_pts - cg) ** 2, axis=1).mean()
        scale = np.sqrt((r2_gt + eps) / (r2_pred + eps))
    else:
        # Default: use the ratio of bbox diagonal lengths.
        def _bbox_diag(pts):
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            return float(np.linalg.norm(maxs - mins))

        d_pred = _bbox_diag(pred_pts)
        d_gt = _bbox_diag(gt_pts)
        scale = (d_gt + eps) / (d_pred + eps)

    return float(scale)

class GaussianMapper(object):
    """
    SLAM from Rendering with 3D Gaussian Splatting.
    """

    def __init__(self, cfg, slam, gui_qs=None):
        self.cfg = cfg
        self.slam = slam
        self.video = slam.video
        self.device = cfg.device
        self.mode = cfg.mode
        self.output = slam.output
        self.delay = cfg.mapping.delay  # Delay between tracking and mapping
        self.warmup = cfg.mapping.warmup
        self.batch_mode = cfg.mapping.online_opt.batch_mode  # Take a batch of all unupdated frames at once

        # Given an external mask for dyn. objects, remove these from the optimization
        self.filter_dyn = cfg.get("with_dyn", False)

        self.loss_params = cfg.mapping.loss  # Losses

        self.pipeline_params = cfg.mapping.pipeline_params
        self.sh_degree = 0
        # Change the downsample factor for initialization depending on cfg.tracking.upsample, so we always have points
        if not self.cfg.tracking.upsample:
            cfg.mapping.input.pcd_downsample_init /= 8
            cfg.mapping.input.pcd_downsample /= 8

        # Online Tracker
        self.use_gt_poses = cfg.get("use_gt_poses", False)
        self.update_params = cfg.mapping.online_opt
        self.mapping_iters = self.update_params.iters
        # How to filter the Tracking map before Rendering
        self.filter_params = self.update_params.filter

        # SLAM-side covisibility threshold for deciding whether to initialize new Gaussians for a keyframe
        # If not provided in config, default to 0.2 (20%)
        self.covisibility_th = getattr(cfg.mapping, "covisibility_th", 0.2)
        self.skipped_kfs = set()

        self.gaussians = GaussianModel(self.sh_degree, config=cfg.mapping.input)

        self.z_near = 0.0001
        self.z_far = 10000.0
        self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)

        if gui_qs is not None:
            self.q_main2vis = gui_qs
            self.use_gui = True
        else:
            self.use_gui = False

        self.last_idx = 0
        self.cameras, self.new_cameras = [], []
        self.iteration_info = []
        self.cam2buffer, self.buffer2cam = {}, {}
        self.info(f"[DEBUG][SLAM_vs_3DGS] buffer2cam_size={len(self.buffer2cam)}")
        self.uid2index = {}
        self.count = 0

        self.cam2gaussian = {}
        self.stride = int(getattr(self.cfg.mapping, "frame_gaussians_stride", 1))
        self.scene_occ_root = getattr(
            self.cfg.mapping,
            "scene_occ_root",
            "/data",  # Default ScanNet path
        )
        self.enable_occ_eval = bool(getattr(self.cfg.mapping, "enable_occ_eval", True))
        if self.enable_occ_eval:
            self.gt_scene_data, self.gt_occ_pts = self.get_gt_occ()
        else:
            self.gt_scene_data, self.gt_occ_pts = None, None
            self.info("[OCC] enable_occ_eval=False, skip loading GT occupancy.")

        self.gui_vis_mode = getattr(self.cfg.mapping, "gui_vis_mode", "rgb")
        # Allowed values: "ov3d_label" | "rgb" | "ov_label" | "ov_overlay"
        self.gui_overlay_alpha = float(getattr(self.cfg.mapping, "gui_overlay_alpha", 0.5))

        # Cached Sim(3) alignment parameters, equivalent to evo -as: pred_world -> gt_world.
        self.align_s = None           # scalar scale
        self.align_R = None           # 3x3 rotation matrix, torch.Tensor
        self.align_t = None           # 3D translation vector, torch.Tensor

        self.ov_name_path = getattr(self.cfg.mapping, "ov_name_path", "./src/scannet_utils/scannet_name.txt")
        

        # --- GUI incremental mesh export (save aligned semantic gaussians per update) ---
        self.save_mesh_each_update = bool(getattr(self.cfg.mapping, "save_mesh_each_update", False))
        self.save_mesh_each_update_every = int(getattr(self.cfg.mapping, "save_mesh_each_update_every", 1))  # save once per N updates
        self._mesh_export_counter = 0

    def info(self, msg: str):
        print(colored("[Gaussian Mapper] " + msg, "magenta"))

    def __len__(self):
        """Return the number of camera frames in the scene."""
        return len(self.cameras)

    def camera_from_video(self, idx):
        """Extract Camera objects from a part of the video."""
        if self.video.disps_clean[idx].sum() < 1:  # Sanity check:
            self.info(f"Warning. Trying to intialize from empty frame {idx}!")
            return None

        color, depth, depth_prior, intrinsics, w2c_lie, stat_mask, ts = self.video.get_mapping_item(idx, self.device)
        w2c = lie_to_matrix(w2c_lie)
        return self.camera_from_frame(idx, color, w2c, intrinsics, depth, mask=stat_mask)

    def camera_from_frame(
        self,
        idx: int,
        image: torch.Tensor,
        w2c: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_init: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """Given the image, depth, intrinsic and pose, creates a Camera object.
        The depth for supervision and initialization does not need to be the same, e.g. we could initialize
        the Gaussians with a sparse, but certain depth map and supervise with a dense prior.
        We also use an optional mask for the objective function, e.g. for supervising only the static parts of the scene
        explainable by the camera motion."""
        fx, fy, cx, cy = intrinsics

        height, width = image.shape[-2:]
        fovx, fovy = focal2fov(fx, width), focal2fov(fy, height)
        projection_matrix = getProjectionMatrix2(self.z_near, self.z_far, cx, cy, fx, fy, width, height)
        projection_matrix = projection_matrix.transpose(0, 1).to(device=self.device)

        return Camera(
            idx,
            image.contiguous(),
            depth_init,
            depth,
            w2c,
            projection_matrix,
            (fx, fy, cx, cy),
            (fovx, fovy),
            (height, width),
            device=self.device,
            mask=mask,
        )

    def _compute_slam_covisibility(self, new_idx: int) -> float:
        """Compute SLAM-side covisibility between candidate keyframe (new_idx)
        and existing keyframes in self.cameras.

        Covisibility = max over ref kf of:
            (# new valid pixels that project into ref FOV AND depth-consistent) / (# new valid pixels)
        """
        if len(self.cam2gaussian) == 0:
            return 0.0

        device = self.device

        # ---- load new frame exactly like mapping uses ----
        # returns: image, est_depth, depth_prior, intrinsics, w2c_lie, static_mask, ts
        color_n, depth_n, depth_prior_n, intr_n, w2c_lie_n, stat_mask_n, ts_n = \
            self.video.get_mapping_item(new_idx, device=device)

        # depth_n: [H,W] or [1,H,W]
        if depth_n.ndim == 3:
            depth_n = depth_n[0]
        ht_n, wd_n = depth_n.shape[-2], depth_n.shape[-1]

        fx_n, fy_n, cx_n, cy_n = [float(x) for x in intr_n]

        # valid pixels in new frame
        valid_new = (depth_n > 0)
        if stat_mask_n is not None:
            if stat_mask_n.ndim == 3:
                stat_mask_n = stat_mask_n[0]
            valid_new = valid_new & stat_mask_n.bool()

        n_total = int(valid_new.sum().item())
        if n_total < 100:
            return 0.0

        # pixel grid
        yy, xx = torch.meshgrid(
            torch.arange(ht_n, device=device),
            torch.arange(wd_n, device=device),
            indexing="ij",
        )

        z = depth_n[valid_new]
        x = (xx[valid_new].to(z.dtype) - cx_n) * z / fx_n
        y = (yy[valid_new].to(z.dtype) - cy_n) * z / fy_n
        pts_cam_new = torch.stack([x, y, z], dim=-1)  # [N,3]

        # pose: w2c from mapping item -> c2w
        w2c_new = lie_to_matrix(w2c_lie_n).to(device=device)          # [4,4] cam_from_world
        c2w_new = torch.linalg.inv(w2c_new)                           # [4,4] world_from_cam

        pts_h = torch.cat(
            [pts_cam_new, torch.ones((pts_cam_new.shape[0], 1), device=device, dtype=pts_cam_new.dtype)],
            dim=-1
        )  # [N,4]
        pts_world = (c2w_new @ pts_h.t()).t()[:, :3]  # [N,3]

        max_cov = 0.0

        # Iterate over reference keyframes
        # for ref_cam in self.cameras:
        for ref_cam_uid in self.cam2gaussian.keys():

            ref_idx = self.cam2buffer.get(ref_cam_uid, None)
            if ref_idx is None:
                continue

            # ---- load ref frame exactly like mapping uses ----
            color_r, depth_r, depth_prior_r, intr_r, w2c_lie_r, stat_mask_r, ts_r = \
                self.video.get_mapping_item(ref_idx, device=device)

            if depth_r.ndim == 3:
                depth_r = depth_r[0]
            ht_r, wd_r = depth_r.shape[-2], depth_r.shape[-1]
            fx_r, fy_r, cx_r, cy_r = [float(x) for x in intr_r]

            # world -> ref cam directly via w2c
            w2c_ref = lie_to_matrix(w2c_lie_r).to(device=device)       # [4,4] cam_from_world
            R = w2c_ref[:3, :3]
            t = w2c_ref[:3, 3]

            pts_ref = (R @ pts_world.t() + t[:, None]).t()  # [N,3]
            z_r = pts_ref[:, 2]

            valid_z = z_r > 0
            if valid_z.sum().item() == 0:
                continue

            x_r = pts_ref[:, 0]
            y_r = pts_ref[:, 1]
            u = fx_r * x_r / z_r + cx_r
            v = fy_r * y_r / z_r + cy_r

            in_fov = valid_z & (u >= 0) & (u <= wd_r - 1) & (v >= 0) & (v <= ht_r - 1)
            if in_fov.sum().item() == 0:
                continue

            # sample ref depth and check consistency
            u_i = u[in_fov].long().clamp(0, wd_r - 1)
            v_i = v[in_fov].long().clamp(0, ht_r - 1)

            depth_ref = depth_r[v_i, u_i]  # [M]
            valid_ref = depth_ref > 0
            if valid_ref.sum().item() == 0:
                continue

            depth_new_proj = z_r[in_fov][valid_ref]
            depth_ref = depth_ref[valid_ref]

            rel_err = torch.abs(depth_new_proj - depth_ref) / torch.clamp(depth_ref, min=1e-6)
            depth_consistent = rel_err < 0.1  # 10% threshold

            cov = float(depth_consistent.sum().item()) / float(n_total)
            if cov > max_cov:
                max_cov = cov

        return float(max_cov)

    def get_new_cameras(self, delay=0):
        """Get all new cameras from the video."""

        # import sys, pdb; sys.stdin = open(0); sys.stdout = open(1, "w", buffering=1); pdb.set_trace()

        # Only add a batch of cameras in batch_mode
        if self.batch_mode:
            to_add = range(self.last_idx, self.cur_idx - delay)
            to_add = to_add[: self.update_params.batch_size]
        else:
            to_add = range(self.last_idx, self.cur_idx - delay)

        for idx in to_add:
            if hasattr(self, "skipped_kfs") and idx in self.skipped_kfs:
                continue

            color, depth, depth_prior, intrinsics, w2c_lie, stat_mask, ts = self.video.get_mapping_item(
                idx, device=self.device
            )
            w2c = lie_to_matrix(w2c_lie)

            # HOTFIX Sanity check for when we dont have any good depth
            if (depth > 0).sum() < 100:
                depth = None
            if (depth_prior > 0).sum() < 100:
                depth_prior = None

            # SLAM-side covisibility gating using DepthVideo.disps_up
            cov = self._compute_slam_covisibility(idx)

            if len(self.cam2buffer) and cov >= self.covisibility_th:
                self.info(
                    f"Skip KF {idx} for Gaussian init due to high SLAM covisibility: cov={cov:.3f} >= th={self.covisibility_th:.3f}"
                )
                self.skipped_kfs.add(idx)
                continue

            cam = self.camera_from_frame(
                idx, color, w2c, intrinsics, depth_init=depth, depth=depth_prior, mask=stat_mask
            )

            # Insert camera into index mapping
            # Keep cam2buffer / buffer2cam as uid -> uid so that GaussianModel.unique_kfIDs stays consistent
            if cam.uid not in self.cam2buffer:
                self.cam2buffer[cam.uid] = cam.uid
                self.buffer2cam[cam.uid] = cam.uid

            H, W = color.shape[-2], color.shape[-1]

            init_scale = float(getattr(self.cfg.mapping, "frame_gaussians_init_scale", 0.01))
            opacity_value = float(getattr(self.cfg.mapping, "frame_gaussians_opacity_value", 0.999))

            fg = FrameGaussians()
            fg.cam_id = int(cam.uid)
            fg.timestamp = ts

            device = 'cpu'

            # All-true mask: all pixels will be used when sampling by mask later.
            fg.mask = torch.ones((H // self.stride, W // self.stride), device=device, dtype=torch.bool)

            # ---- replace your current fg.scales / fg.rotations init with this ----
            Hd, Wd = H // self.stride, W // self.stride
            device_cpu = torch.device("cpu")

            # import sys, pdb; sys.stdin = open(0); sys.stdout = open(1, "w", buffering=1); pdb.set_trace()

            K = intrinsics.detach().to(device_cpu).float()  # (3,3)
            fx, fy, cx, cy = K

            # downsample grid pixel centers (still in original pixel units)
            u = (torch.arange(Wd, device=device_cpu, dtype=torch.float32) + 0.5) * float(self.stride)
            v = (torch.arange(Hd, device=device_cpu, dtype=torch.float32) + 0.5) * float(self.stride)
            vv, uu = torch.meshgrid(v, u, indexing="ij")  # (Hd,Wd)

            # ray dir in CAMERA frame (OpenCV convention: z forward)
            x = (uu - cx) / fx
            y = (vv - cy) / fy
            dirs_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # (Hd,Wd,3)
            dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)

            def quat_from_z_to_dir_cam(d: torch.Tensor) -> torch.Tensor:
                """
                d: (N,3) unit vectors in camera frame
                return: (N,4) quaternion (w,x,y,z), rotating local +Z to d
                """
                a = torch.tensor([0.0, 0.0, 1.0], device=d.device, dtype=d.dtype).expand_as(d)  # +Z
                v = torch.cross(a, d, dim=-1)                              # (N,3)
                w = 1.0 + (a * d).sum(dim=-1, keepdim=True)               # (N,1)
                q = torch.cat([w, v], dim=-1)                             # (N,4)

                # handle near-opposite (rare here, but safe)
                mask = (w.squeeze(-1) < 1e-6)
                if mask.any():
                    q[mask] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=d.device, dtype=d.dtype)  # 180° about X

                q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                return q

            # rotations in CAMERA frame
            rot = quat_from_z_to_dir_cam(dirs_cam.reshape(-1, 3)).reshape(Hd, Wd, 4)

            # anisotropic scales: make local Z the longest axis (aligned with ray)
            init_scale = float(getattr(self.cfg.mapping, "frame_gaussians_init_scale", 0.01))
            ray_aspect = float(getattr(self.cfg.mapping, "frame_gaussians_ray_aspect", 1.0))  # tune: 3~10

            s_perp = init_scale
            s_ray  = init_scale * ray_aspect

            sc = torch.empty((Hd, Wd, 3), device=device_cpu, dtype=torch.float32)
            sc[..., 0] = s_perp
            sc[..., 1] = s_perp
            sc[..., 2] = s_ray

            fg.scales = sc
            fg.rotations = rot
            # ---- end replacement ----

            fg.opacities = torch.full((H // self.stride, W // self.stride, 1), opacity_value, device=device, dtype=torch.float32)
            self.cam2gaussian[cam.uid] = fg

            self.new_cameras.append(cam)

    def frame_updater(self, delay=0):
        """Gets the list of frames and updates the depth and pose based on the video.

        NOTE: This assumes, that optical flow based tracking overall is more reliable for sparse supervision
        We only use the Renderer for great scene representation and potential densification / correction

        NOTE chen: in some cases we might have keyframes & non-keyframes in self.cameras. We can distinguish keyframes by using
        index mapping since this is a unique mapping from cam.uid to the position in the video buffer.
        """
        all_cameras = self.cameras + self.new_cameras
        with self.video.get_lock():
            (dirty_index,) = torch.where(self.video.mapping_dirty.clone())
            # Only update up to the current frame in Mapper
            dirty_index = dirty_index[dirty_index < self.cur_idx - delay]

        # Check if the dirty indices from video buffer are also in our cam2buffer as values
        # -> Only update already inserted cameras
        to_update = dirty_index[
            torch.isin(dirty_index, torch.tensor(list(self.cam2buffer.values()), device=self.device))
        ]

        for idx in to_update:
            # TODO can the stat_mask potentially change as well?
            color, depth, depth_prior, intrinsics, w2c_lie, stat_mask, ts = self.video.get_mapping_item(
                idx, device=self.device
            )
            w2c = lie_to_matrix(w2c_lie)

            # Use uid2index to map from video-buffer index (uid) to index in self.cameras/new_cameras
            cam_idx = self.uid2index.get(int(idx), None)
            if cam_idx is None or cam_idx >= len(all_cameras):
                # This buffer index does not correspond to any current camera (e.g. was skipped by covisibility gating)
                # -> skip updating it here
                continue

            cam = all_cameras[cam_idx]
            # update intrinsics in case we use opt_intrinsics
            cam.update_intrinsics(intrinsics, color.shape[-2:], self.z_near, self.z_far)
            cam.depth = depth.detach()
            R = w2c[:3, :3].unsqueeze(0).detach()
            T = w2c[:3, 3].detach()
            cam.update_RT(R, T)

        self.video.mapping_dirty[to_update] = False

    def get_ram_usage(self) -> Tuple[float, float]:
        free_mem, total_mem = torch.cuda.mem_get_info(device=self.device)
        used_mem = 1 - (free_mem / total_mem)
        return used_mem, free_mem

    def _last_call(self, mapping_queue: mp.Queue, received_item: mp.Event):
        """We already build up the map based on the SLAM system and finetuned over it.
        Depending on compute budget this has been done scarcely.
        This call runs many more iterations for refinement and densification to get a high quality map.

        Since the SLAM system operates on keyframes, but we have many more views in our video stream,
        we can use additional supervision from non-keyframes to get higher detail.
        """
        print("\n[Gaussian Mapper] >>> ENTER _last_call <<<\n", flush=True)
        # Free memory before doing refinement
        torch.cuda.empty_cache()
        gc.collect()

        final_aligned_gaussians = self.get_aligned_gaussians()
        ply_path = f"{self.output}/mesh/final_{self.mode}.ply"
        final_aligned_gaussians.save_ply(ply_path)
        self.info(f"Mesh saved at {ply_path} (from get_current_gaussians)")

        self.info(f"{len(self.iteration_info)} iterations, {len(self.cameras)/len(self.iteration_info)} cams/it")

        if mapping_queue is not None:
            mapping_queue.put("done")
        if received_item is not None:
            received_item.wait()

    def update_gui(self, last_new_cam: Camera) -> None:
        # Debug: make sure this is called (multiprocess safe)
        # print("[GUI] update_gui called", flush=True)

        # -------------------------
        # 1) Default: latest frame RGB + depth (RGB mode keeps this)
        # -------------------------
        if len(self.new_cameras) > 0 and last_new_cam is not None:
            rgb_img = last_new_cam.original_image
            # unify to float [0,1]
            if isinstance(rgb_img, torch.Tensor):
                if rgb_img.ndim == 3 and rgb_img.shape[0] != 3 and rgb_img.shape[-1] == 3:
                    rgb_img = rgb_img.permute(2, 0, 1).contiguous()
                rgb_img = rgb_img.float()
                if rgb_img.max() > 1.0:
                    rgb_img = rgb_img / 255.0
            else:
                # fallback: leave as-is
                pass

            # latest depth (only for rgb mode; ov_* will override later)
            if last_new_cam.depth is not None:
                if not self.loss_params.supervise_with_prior:
                    gtdepth_np = last_new_cam.depth.detach().clone().cpu().numpy()
                else:
                    gtdepth_np = last_new_cam.depth_prior.detach().clone().cpu().numpy()
            else:
                gtdepth_np = None
        else:
            rgb_img, gtdepth_np = None, None
            last_new_cam = self.cameras[-1]

        img = rgb_img  # default gtcolor

        # -------------------------
        # 2) OV modes: use SAME uid for (rgb/label/depth)
        # -------------------------
        mode = getattr(self, "gui_vis_mode", "rgb")
        if mode in ["ov_label", "ov_overlay"]:
            label_cache = getattr(self, "ov_label_cache", None)

            uid_now = int(last_new_cam.uid)
            uid_used = uid_now
            label_hw = None if label_cache is None else label_cache.get(uid_used, None)

            # fallback: choose nearest cached uid (<= uid_now), else latest cached uid
            if label_hw is None and label_cache is not None and len(label_cache) > 0:
                cands = [k for k in label_cache.keys() if k <= uid_now]
                uid_used = max(cands) if len(cands) > 0 else max(label_cache.keys())
                label_hw = label_cache.get(uid_used, None)

            if label_hw is not None:
                # --- semantic rgb (cpu) ---
                pal = self._get_palette_auto()
                sem_rgb = self._label_to_rgb(label_hw, pal)  # [3,H,W] cpu float

                # --- fetch SAME-frame rgb + depth via get_mapping_item(uid_used) ---
                rgb_used = None
                depth_used = None
                try:
                    color2, depth2, depth_prior2, *_ = self.video.get_mapping_item(uid_used, device=self.device)

                    # rgb_used: cpu float [0,1] [3,H,W]
                    if isinstance(color2, torch.Tensor):
                        if color2.ndim == 3 and color2.shape[0] != 3 and color2.shape[-1] == 3:
                            color2 = color2.permute(2, 0, 1).contiguous()
                        color2 = color2.float()
                        if color2.max() > 1.0:
                            color2 = color2 / 255.0
                        rgb_used = color2.detach().cpu()

                    # depth_used: choose depth or prior, then numpy
                    if not self.loss_params.supervise_with_prior:
                        depth_used = depth2
                    else:
                        depth_used = depth_prior2

                    if depth_used is not None:
                        gtdepth_np = depth_used.detach().clone().cpu().numpy()
                    else:
                        gtdepth_np = None

                except Exception:
                    # if get_mapping_item fails, fall back:
                    rgb_used = rgb_img.detach().cpu() if isinstance(rgb_img, torch.Tensor) else rgb_img
                    # keep gtdepth_np as latest (already set)

                # --- choose displayed gtcolor ---
                if mode == "ov_label":
                    img = sem_rgb
                else:  # ov_overlay
                    alpha = float(getattr(self, "gui_overlay_alpha", 0.5))
                    if rgb_used is None:
                        rgb_used = rgb_img.detach().cpu() if isinstance(rgb_img, torch.Tensor) else rgb_img
                    img = (1 - alpha) * rgb_used + alpha * sem_rgb

        # Build the current (merged) Gaussians for visualization.
        # Always visualize the result from get_current_gaussians(); do not fall back to self.gaussians.
        res = self.get_current_gaussians()
        if not (isinstance(res, tuple) and len(res) >= 1 and isinstance(res[0], GaussianModel)):
            # Return without sending any Gaussians to the GUI to avoid mixing with stale models.
            self.info("[GUI] get_current_gaussians() did not return a valid GaussianModel, skip GUI update.")
            return
        gaussians_vis = res[0]

        self.q_main2vis.put_nowait(
            gui_utils.GaussianPacket(
                gaussians=clone_obj(gaussians_vis),
                current_frame=last_new_cam.detach(),
                keyframes=[cam.detach() for cam in self.cameras],
                kf_window=None,
                gtcolor=img,
                gtdepth=gtdepth_np,
            )
        )

    def _init_ov_model(self):
        if getattr(self, "ov_model", None) is not None:
            return

        import sys, os
        trident_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "thirdparty", "Trident")
        if trident_path not in sys.path:
            sys.path.insert(0, trident_path)

        from trident import Trident
        proj_root = os.path.dirname(os.path.dirname(__file__))
        sam_checkpoint = os.path.join(proj_root, "pretrained", "sam_vit_b_01ec64.pth")
        dino_checkpoint = os.path.join(proj_root, "pretrained", "dino_vitbase16_pretrain.pth")

        self.ov_model = Trident(
            clip_type='openai',
            model_type='ViT-B/16',
            vfm_model='dino',
            name_path=self.ov_name_path,
            sam_refinement=True,
            coarse_thresh=0.2,
            minimal_area=225,
            debug=False,              # Keep debug off; it is slow.
            sam_ckpt=sam_checkpoint,
            vfm_ckpt=dino_checkpoint,
            sam_model_type="vit_b",
            slide_crop=16 * 24,
            slide_stride=16 * 8,
        ).cuda()
        self.ov_model.eval()

        # cache：uid -> logits tensor (CPU)
        if not hasattr(self, "ov_cache"):
            self.ov_cache = {}

    def _get_palette_11(self) -> np.ndarray:
        """11-class palette for 3DSSG / OV segmentation."""
        pal = np.array([
            [220, 45, 45],  # 0 ceiling
            [ 40,160, 40],  # 1 floor
            [155,210,225],  # 2 wall
            [115,155,210],  # 3 window
            [195,200, 75],  # 4 chair
            [255,180,110],  # 5 bed
            [140,105,180],  # 6 sofa
            [ 25,110,180],  # 7 table
            [150,180, 55],  # 8 television
            [255,140,  0],  # 9 furniture
            [195,180,220],  # 10 objects
        ], dtype=np.uint8)
        return pal
    
    def _infer_num_classes_from_ov_name_path(self) -> int:
        """Infer semantic class count from ov_name_path / name file content.

        Priority:
        1) filename contains 'replica' -> 101
        2) filename contains 'scannet' -> 11
        3) fallback: count lines in the txt (ignore empty lines)
        """
        p = str(getattr(self, "ov_name_path", "")).lower()
        if "replica" in p:
            return 101
        if "realsense" in p:
            return 50
        if "scannet" in p:
            return 11

        # Fallback: count lines in txt
        try:
            with open(getattr(self, "ov_name_path"), "r") as f:
                lines = [ln.strip() for ln in f.readlines()]
            lines = [ln for ln in lines if len(ln) > 0]
            # If file has 101 lines, return 101, etc.
            if len(lines) > 0:
                return int(len(lines))
        except Exception:
            pass

        # Conservative fallback
        return 11


    def _get_palette_n(self, n: int, seed: int = 0) -> np.ndarray:
        """Generate a deterministic palette with stronger separation for n colors (uint8 RGB).

        Strategy:
        - Spread hues using golden-ratio step (avoids adjacent hues being too similar)
        - Cycle a few (S,V) pairs to increase discriminability when n is moderately large
        """
        if n <= 0:
            return np.zeros((0, 3), dtype=np.uint8)

        n = int(n)

        # --- golden ratio hue stepping ---
        # Use a fixed irrational step so neighbors are far apart on average
        phi = 0.618033988749895  # golden ratio conjugate
        h0 = (float(seed) * 0.3123) % 1.0

        # (S,V) cycles: distinct lightness/saturation levels, still vivid
        sv_cycle = [
            (0.85, 0.95),
            (0.65, 0.95),
            (0.85, 0.80),
            (0.60, 0.85),
        ]

        hsv = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            hsv[i, 0] = (h0 + phi * i) % 1.0
            s, v = sv_cycle[i % len(sv_cycle)]
            hsv[i, 1] = s
            hsv[i, 2] = v

        # Convert HSV -> RGB (vectorized, no extra deps)
        h = hsv[:, 0] * 6.0
        i = np.floor(h).astype(np.int32)
        f = h - i

        s = hsv[:, 1]
        v = hsv[:, 2]

        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        r = np.zeros(n, dtype=np.float32)
        g = np.zeros(n, dtype=np.float32)
        b = np.zeros(n, dtype=np.float32)

        i_mod = i % 6
        m = (i_mod == 0); r[m], g[m], b[m] = v[m], t[m], p[m]
        m = (i_mod == 1); r[m], g[m], b[m] = q[m], v[m], p[m]
        m = (i_mod == 2); r[m], g[m], b[m] = p[m], v[m], t[m]
        m = (i_mod == 3); r[m], g[m], b[m] = p[m], q[m], v[m]
        m = (i_mod == 4); r[m], g[m], b[m] = t[m], p[m], v[m]
        m = (i_mod == 5); r[m], g[m], b[m] = v[m], p[m], q[m]

        rgb = np.stack([r, g, b], axis=1)
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        return rgb


    def _get_palette_auto(self) -> np.ndarray:
        """Auto-select palette based on ov_name_path."""
        n = int(self._infer_num_classes_from_ov_name_path())

        if n == 11:
            return self._get_palette_11()

        # For Replica (101) or any other count, use generated palette
        return self._get_palette_n(n)

    def _colors_from_label_hw(self, label_hw: torch.Tensor, valid_mask: torch.Tensor, palette: np.ndarray) -> torch.Tensor:
        """
        label_hw: [H,W] CPU (uint16/long)
        valid_mask: [H,W] bool on GPU/CPU, matching the post-stride valid mask
        return: [N,3] float on GPU, N = valid_mask.sum()
        """
        if isinstance(label_hw, np.ndarray):
            label_hw = torch.from_numpy(label_hw)
        label_hw = label_hw.to(torch.long)  # CPU

        # Gather labels for valid pixels. valid_mask may live on GPU.
        vm = valid_mask.detach().cpu()
        lab = label_hw[vm]  # [N] CPU long

        pal = torch.from_numpy(palette).to(torch.float32) / 255.0  # [11,3] CPU float
        lab = torch.clamp(lab, 0, pal.shape[0] - 1)
        rgb = pal[lab]  # [N,3] CPU float

        return rgb.to(device=self.device, dtype=torch.float32)  # GPU float

    def _label_to_rgb(self, label_hw: torch.Tensor, palette: np.ndarray) -> torch.Tensor:
        """
        label_hw: [H,W] (cpu) uint16/long
        return: [3,H,W] float32 in [0,1] (cpu)
        """
        if label_hw is None:
            return None
        if isinstance(label_hw, np.ndarray):
            label_hw = torch.from_numpy(label_hw)

        lab = label_hw.detach().cpu().numpy()
        lab = np.clip(lab, 0, palette.shape[0] - 1)
        rgb = palette[lab]                       # [H,W,3] uint8
        rgb = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0
        return rgb


    def get_ov_pred(self, image, **kwargs):
        assert image.shape[1] >= 256, 'minimum requirements'
        assert image.shape[2] >= 256, 'minimum requirements'
        seg_logits = self.ov_model.predict(
            image, data_samples=None, **kwargs)
        return seg_logits
    
    def _estimate_sim3_from_points(self, pred_pts: np.ndarray, gt_pts: np.ndarray):
        """
        Estimate the Sim(3) transform (s,R,t) with the Umeyama algorithm:
            gt ≈ s * R @ pred + t
        This is equivalent to evo -as alignment (SE(3) + scale).
        """
        assert pred_pts.shape == gt_pts.shape and pred_pts.shape[1] == 3
        N = pred_pts.shape[0]
        assert N >= 3, f"Need at least 3 points to estimate Sim3, got {N}"

        mu_x = pred_pts.mean(axis=0)
        mu_y = gt_pts.mean(axis=0)

        Xc = pred_pts - mu_x
        Yc = gt_pts - mu_y

        # Covariance matrix.
        Sigma = (Yc.T @ Xc) / N  # 3x3

        U, D, Vt = np.linalg.svd(Sigma)
        det = np.linalg.det(U @ Vt)
        S = np.eye(3)
        if det < 0:
            S[2, 2] = -1.0

        R = U @ S @ Vt  # 3x3

        var_x = (Xc ** 2).sum(axis=1).mean()
        if var_x < 1e-12:
            s = 1.0
        else:
            s = (np.trace(np.diag(D) @ S) / var_x)

        t = mu_y - s * (R @ mu_x)

        return s, R, t

    def _compute_pose_alignment(self, min_pairs: int = 5, debug_k: int = 3):
        """
        Estimate Sim(3) using keyframes that actually contributed to 3DGS construction:
        - Prediction side: self.video.poses stores w2c Lie(7), so camera centers must be C = -R^T t.
        - GT side: self.video.poses_gt stores c2w (t,q), so the camera center is t.

        Outputs self.align_s / self.align_R / self.align_t such that:
            C_gt ≈ s * R @ C_pred + t
        """

        device = self.device

        # -----------------------------
        # Helper: set identity and return
        # -----------------------------
        def _set_identity(reason: str):
            self.info(f"[Sim3] Use identity Sim3 ({reason}).")
            self.align_s = 1.0
            self.align_R = torch.eye(3, device=device)
            self.align_t = torch.zeros(3, device=device)

            # optional stats for logging/json
            self.last_sim3_scale = 1.0
            self.last_sim3_post_rmse = 0.0
            self.last_sim3_pre_rmse = 0.0
            self.last_sim3_pairs = 0
            return

        # 1) Only use frames that actually contributed to 3DGS: keys in cam2gaussian.
        if len(self.cam2gaussian) == 0:
            return _set_identity("cam2gaussian empty")
        
        if getattr(self, "gt_scene_data", None) is None:
            return _set_identity("GT occ not loaded (gt_scene_data is None)")

        used_indices = sorted(list(self.cam2gaussian.keys()))
        idxs = torch.tensor(used_indices, device=device, dtype=torch.long)

        with self.video.get_lock():
            # NOTE: video.poses is used as w2c Lie by get_mapping_item().
            poses_w2c_lie = self.video.poses[idxs].clone()     # [N,7]  (w2c lie)
            ts = self.video.timestamp[idxs].clone()

            poses_gt_attr = getattr(self.video, "poses_gt", None)
            if poses_gt_attr is None:
                return _set_identity("video.poses_gt is missing")

            try:
                poses_gt_c2w = poses_gt_attr[idxs].clone()  # [N,7] (c2w t,q)
            except Exception:
                return _set_identity("video.poses_gt cannot be indexed by idxs")

        # Handle all-zero / invalid GT, which is common in real-world runs.
        if poses_gt_c2w.numel() == 0:
            return _set_identity("video.poses_gt is empty")

        # Treat near-zero GT translation as missing GT.
        if torch.norm(poses_gt_c2w[:, :3], dim=-1).max().item() < 1e-6:
            return _set_identity("video.poses_gt translation is all ~0")
        

        # 2) Drop timestamp=0 frames that still have identity poses.
        ts_mask = ts > 0

        # Prediction side: poses_w2c_lie[:3] is not the camera center, but it is useful as a coarse all-zero filter.
        # GT side: poses_gt_c2w[:3] is the camera center and can be used directly.
        raw_t_est = poses_w2c_lie[:, :3]
        raw_t_gt  = poses_gt_c2w[:, :3]
        est_mask = torch.norm(raw_t_est, dim=-1) > 1e-6
        gt_mask  = torch.norm(raw_t_gt,  dim=-1) > 1e-6

        n_total = idxs.shape[0]
        n_ts    = ts_mask.sum().item()
        n_est   = est_mask.sum().item()
        n_gt    = gt_mask.sum().item()
        self.info(
            f"[Sim3][DEBUG] cam2gaussian_n={n_total}, ts>0={n_ts}, "
            f"raw_est_t!=0={n_est}, gt_center!=0={n_gt}"
        )

        valid_mask = ts_mask & est_mask & gt_mask
        n_valid = valid_mask.sum().item()
        if n_valid < min_pairs:
            self.info(f"[Sim3] Not enough valid pose pairs after filtering: {n_valid} < {min_pairs}. Use identity Sim3.")
            self.align_s = 1.0
            self.align_R = torch.eye(3, device=device)
            self.align_t = torch.zeros(3, device=device)
            return

        poses_w2c_lie = poses_w2c_lie[valid_mask]
        poses_gt_c2w  = poses_gt_c2w[valid_mask]
        idxs_valid    = idxs[valid_mask]

        # 3) Compute camera centers.
        # Prediction: poses_w2c_lie -> w2c matrix -> C = -R^T t.
        # GT: poses_gt_c2w is c2w (t,q), so center = t.
        try:
            # The project usually provides a batched lie_to_matrix.
            w2c = lie_to_matrix(poses_w2c_lie)  # [M,4,4]
        except Exception:
            # Fallback to lietorch.
            import lietorch
            w2c = lietorch.SE3(poses_w2c_lie).matrix()  # [M,4,4]

        R_w2c = w2c[:, :3, :3]
        t_w2c = w2c[:, :3, 3]
        C_pred = -(R_w2c.transpose(1, 2) @ t_w2c.unsqueeze(-1)).squeeze(-1)  # [M,3]

        C_gt = poses_gt_c2w[:, :3]  # [M,3]  (gt c2w center)

        # -------- Debug: check consistency between get_mapping_item and video.poses, plus raw t vs camera center --------
        # Print only the first debug_k samples.
        k = min(debug_k, C_pred.shape[0])
        for j in range(k):
            uid = int(idxs_valid[j].item())
            try:
                # Get w2c through the mapping path, matching map construction.
                _ret = self.video.get_mapping_item(uid, use_gt=False, device=device)
                # Compatible with return lengths from return_raw_depth=True/False.
                w2c_lie_map = _ret[4]
                try:
                    w2c_map = lie_to_matrix(w2c_lie_map)
                except Exception:
                    import lietorch
                    w2c_map = lietorch.SE3(w2c_lie_map).matrix()
                Rm = w2c_map[:3, :3]
                tm = w2c_map[:3, 3]
                C_map = -(Rm.T @ tm)
                # Raw t is not the center; it is only the first three values of the w2c Lie vector.
                raw_t = poses_w2c_lie[j, :3]
                self.info(
                    f"[Sim3][DEBUG] uid={uid} "
                    f"C_pred={C_pred[j].detach().cpu().numpy()} "
                    f"C_map={C_map.detach().cpu().numpy()} "
                    f"||C_pred-C_map||={float(torch.norm(C_pred[j]-C_map).item()):.3e} "
                    f"raw_t(w2c_lie[:3])={raw_t.detach().cpu().numpy()}"
                )
            except Exception as e:
                self.info(f"[Sim3][DEBUG] uid={uid} mapping_item check failed: {repr(e)}")

        # 4) Move to CPU / numpy for Umeyama.
        C_pred_np = C_pred.detach().cpu().numpy()
        C_gt_np   = C_gt.detach().cpu().numpy()

        # subsample
        max_pairs = 400
        if C_pred_np.shape[0] > max_pairs:
            step = C_pred_np.shape[0] // max_pairs
            C_pred_np = C_pred_np[::step]
            C_gt_np   = C_gt_np[::step]

        # Debug: pre-alignment RMSE.
        pre_rmse = float(np.sqrt(np.mean(np.sum((C_gt_np - C_pred_np) ** 2, axis=1))))
        self.info(f"[Sim3][DEBUG] pre-align center RMSE = {pre_rmse:.6f} (before Sim3)")

        # 5) Umeyama
        s, R, t = self._estimate_sim3_from_points(C_pred_np, C_gt_np)

        self.align_s = float(s)
        self.align_R = torch.from_numpy(R).to(device=device, dtype=torch.float32)
        self.align_t = torch.from_numpy(t).to(device=device, dtype=torch.float32)

        # Debug: post-alignment RMSE.
        C_pred_aligned = (s * (C_pred_np @ R.T)) + t[None, :]  # Row-vector numpy convention, so use @R.T.
        post_rmse = float(np.sqrt(np.mean(np.sum((C_gt_np - C_pred_aligned) ** 2, axis=1))))

        self.info(
            f"[Sim3] Estimated s={self.align_s:.6f} from {C_pred_np.shape[0]} pose pairs. "
            f"post-align RMSE={post_rmse:.6f}. "
            f"sample: C_pred[0]={C_pred_np[0]}, C_gt[0]={C_gt_np[0]}"
        )
        self.last_sim3_scale = float(s)
        self.last_sim3_post_rmse = float(post_rmse)
        self.last_sim3_pre_rmse = float(pre_rmse)
        self.last_sim3_pairs = int(C_pred_np.shape[0])

    def get_current_occ(self):
        if self.gt_scene_data is None:
            raise RuntimeError(
                "get_current_occ() requires GT occ data, but enable_occ_eval=False or GT not loaded."
            )
        
        scene_data = self.gt_scene_data
        gt_occ_pts = self.gt_occ_pts  # Only kept for later visualization; Sim3 alignment uses poses only.

        # Use the 3DGS representation tied to SLAM xyz from get_current_gaussians().
        g, frame_slices, views = self.get_current_gaussians()
        self.gaussians = g

        # Ensure the global Sim(3) alignment has been estimated from SLAM/GT trajectories.
        self._compute_pose_alignment()

        xyz = g.get_xyz                            # [N,3], currently in the SLAM world frame
        rotation = quaternion_to_matrix(g.get_rotation)
        R_a = self.align_R.to(rotation)                   # [3,3]
        rotation_aligned = R_a.unsqueeze(0) @ rotation    # [N,3,3]
        scale = g.get_scaling
        opacity = g.get_opacity
        feats = g.get_features.flatten(1)
        semantics = g.ov_feat

        # Apply Sim(3): gt_world ~= s * R @ pred_world + t.
        # xyz: [N,3]; align_R: [3,3]; align_t: [3]
        s = self.align_s
        R = self.align_R
        t = self.align_t

        # Apply R @ xyz^T, then transpose back to [N,3].
        points_aligned = (R @ xyz.t()).t()  # [N,3]
        points_aligned = points_aligned * s + t.unsqueeze(0)  # [N,3]
        scale_aligned = scale * s  # Scale Gaussian size as well.

        # Feed into gaussians_to_occ. GT occupancy coordinates are already in the scene_data world frame.
        pred_occ = gaussians_to_occ(
            points_aligned,
            feats,
            scale_aligned,
            rotation_aligned,
            opacity,
            semantics,
            scene_data,
            owner=self
        )

        return pred_occ

    def get_gt_occ(self):
        scene_name = self.slam.dataset.input_folder.strip('/').split('/')[-1]
        scene_occ_root = self.scene_occ_root
        self.info(f"[OCC] load_full_scene_occ: scene={scene_name}, root={scene_occ_root}")
        scene_data = load_full_scene_occ(
            scene_name=scene_name,
            scene_occ_root=scene_occ_root,
            voxel_size=0.08,
            to_torch=False,
        )
        gt_occ_pts = extract_gt_occupied_points(scene_data, min_label=0)  # label > 0
        return scene_data, gt_occ_pts

    def _to_jsonable(self, x):
        """Convert tensors/ndarrays to JSON-serializable python types."""
        import numpy as np
        import torch

        if x is None:
            return None
        if isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        if isinstance(x, (list, tuple)):
            return [self._to_jsonable(v) for v in x]
        if isinstance(x, dict):
            return {str(k): self._to_jsonable(v) for k, v in x.items()}
        return str(x)

    def _atomic_write_json(self, path: str, obj):
        """Atomically write JSON to disk to avoid partial/corrupted files."""
        import os, json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._to_jsonable(obj), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def get_current_metrics(self, save_pts_name=None):
        """Evaluate SSC metrics (requires GT occ)."""
        if self.gt_scene_data is None:
            self.info("[OCC] GT occ not available; skip metrics.")
            return {
                "enabled": False,
                "reason": "enable_occ_eval=False or GT occ not loaded",
            }

        scene_data = self.gt_scene_data

        evaluator = SSCMetricsTorch(12)
        pred_occ = self.get_current_occ()

        gt_occ = torch.from_numpy(scene_data['occ_labels']).to(device=pred_occ.device)
        gt_occ = gt_occ.long()[None]

        pred_occ = pred_occ.view(gt_occ.shape)
        evaluator.add_batch(pred_occ, gt_occ)

        metrics = evaluator.get_stats(distributed=False)
        # Keep logging for runtime inspection.
        self.info("\n[RESULT] OCC evaluation:")
        for k, v in metrics.items():
            self.info(f"  {k}: {v}")

        # import sys, pdb; sys.stdin = open(0); sys.stdout = open(1, "w", buffering=1); pdb.set_trace()

        # Convert to a JSON-compatible dict. iou_ssc tensors are converted to lists.
        metrics_json = {
            # occ metrics
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "iou": float(metrics.get("iou", 0.0)),
            "iou_ssc": self._to_jsonable(metrics.get("iou_ssc", None)),
            "iou_ssc_mean": float(metrics.get("iou_ssc_mean", 0.0)),

            # gauss bbox stats (written by gaussians_to_occ(..., owner=self))
            "gauss_inside": int(getattr(self, "last_gauss_inside", 0)),
            "gauss_total": int(getattr(self, "last_gauss_total", 0)),

            # sim3 alignment stats (written by _compute_pose_alignment)
            "sim3_scale": float(getattr(self, "last_sim3_scale", 0.0)),
            "sim3_post_rmse": float(getattr(self, "last_sim3_post_rmse", 0.0)),
            "sim3_pre_rmse": float(getattr(self, "last_sim3_pre_rmse", 0.0)),
            "sim3_pairs": int(getattr(self, "last_sim3_pairs", 0)),

            "tp": evaluator.tps.cpu().numpy().tolist(),
            "fp": evaluator.fps.cpu().numpy().tolist(),
            "fn": evaluator.fns.cpu().numpy().tolist(),

            "ctp": evaluator.completion_tp.cpu().item(),
            "cfp": evaluator.completion_fp.cpu().item(),
            "cfn": evaluator.completion_fn.cpu().item(),
        }

        self.last_metrics = metrics_json
        return metrics_json

    def _update(self, delay_to_tracking=True, iters: int = 10, release_cache: bool = False):
        """Update our rendered map by:
        i) Pull a filtered update from the sparser SLAM map
        ii) Add new Gaussians based on new views
        iii) Run a bunch of optimization steps to update Gaussians and camera poses
        iv) Prune the resulting Gaussians based on visibility and size
        v) Maybe send back a filtered update to the SLAM system
        """
        # self.info("Currently has: {}} gaussians gaussians".format(len(self.gaussians)))

        ### Filter map based on multiview_consistency and uncertainty
        filter_kwargs = dict(self.update_params.filter)
        if self.use_gt_poses:
            filter_kwargs["uncertainty"] = False  # No BA confidence available in GT mode
        self.video.filter_map(**filter_kwargs)

        if delay_to_tracking:
            delay = self.delay
        else:
            delay = 0

        ### Add new cameras based on video index
        self.get_new_cameras(delay=delay)  # Add new cameras
        if len(self.new_cameras) != 0:
            self.last_idx = self.new_cameras[-1].uid + 1
            self.info(f"Added {len(self.new_cameras)} new cameras: {[cam.uid for cam in self.new_cameras]}")

        if len(self.new_cameras) == 0 and self.count > 0:
            self.info("No new cameras and mapping has run before, skip heavy optimization step ...")
            return

        ### Update the frames based on Tracker
        self.frame_updater(delay=delay)  # Update all changed cameras with new information from SLAM system

        ### Update visualization
        if self.use_gui:
            self.update_gui(self.new_cameras[-1])

        self.iteration_info.append(len(self.new_cameras))
        # Keep track of added cameras
        self.cameras += self.new_cameras
        self.new_cameras = []

        # Rebuild uid2index so that it maps video-buffer indices (cam.uid) -> index in self.cameras
        self.uid2index = {}
        for local_idx, cam in enumerate(self.cameras):
            self.uid2index[int(cam.uid)] = local_idx
        # NOTE: cam2buffer / buffer2cam remain uid -> uid and are not modified here

        self.update_frame_gaussians()

        torch.cuda.empty_cache()
        gc.collect()

    def _maybe_save_aligned_gaussians_per_update(self, frame_id: int):
        """When GUI visualization is enabled, save aligned semantic gaussians each update (no eval)."""
        # Only do this when GUI/mapping visualization is on
        if not bool(getattr(self.cfg, "run_mapping_gui", False)):
            return
        if not getattr(self, "save_mesh_each_update", False):
            return

        self._mesh_export_counter += 1
        every = max(1, int(getattr(self, "save_mesh_each_update_every", 1)))
        if (self._mesh_export_counter % every) != 0:
            return

        try:
            os.makedirs(f"{self.output}/mesh", exist_ok=True)

            # If GT occ is not loaded (real-world), skip alignment and save raw gaussians
            if getattr(self, "gt_scene_data", None) is None:
                res = self.get_current_gaussians()
                if not (isinstance(res, tuple) and len(res) >= 1 and isinstance(res[0], GaussianModel)):
                    self.info("[mesh-each-update] get_current_gaussians() invalid, skip saving.")
                    return
                gaussians_to_save = res[0]
                ply_path = f"{self.output}/mesh/frame_{int(frame_id):06d}_{self.mode}_raw.ply"
                gaussians_to_save.save_ply(ply_path)
                self.info(f"[mesh-each-update] saved (raw, no GT occ): {ply_path}")
                return

            # Reuse last_call logic: align -> save ply (no metrics)
            aligned_gaussians = self.get_aligned_gaussians()

            # Name: frame_<frame_id>_<mode>.ply
            ply_path = f"{self.output}/mesh/frame_{int(frame_id):06d}_{self.mode}.ply"
            aligned_gaussians.save_ply(ply_path)

            self.info(f"[mesh-each-update] saved: {ply_path}")
        except Exception as e:
            self.info(f"[mesh-each-update] failed on frame {frame_id}: {type(e).__name__}: {e}")

    def update_frame_gaussians(self):
        """
        Maintain a sliding window of per-frame Gaussians in self.cam2gaussian,
        jointly optimize ONLY scale & rotation via optimize_scale_rotation(),
        then write back the updated scale/rot to each frame's cached gaussians.

        Returns:
            updated_uids: List[int]
        """

        device = torch.device(self.device)

        # -----------------------------
        # Configs (provide sane defaults)
        # -----------------------------
        win = int(getattr(self.cfg.mapping, "frame_gaussians_window", 10))

        iters = int(getattr(self.cfg.mapping, "frame_gaussians_opt_iters", 20))
        lr_scale = float(getattr(self.cfg.mapping, "frame_gaussians_lr_scale", 1e-3))
        lr_rot = float(getattr(self.cfg.mapping, "frame_gaussians_lr_rot", 1e-3))
        lr_opa = float(getattr(self.cfg.mapping, "frame_gaussians_lr_opa", 1e-5))
        views_per_iter = int(getattr(self.cfg.mapping, "frame_gaussians_views_per_iter", 10))

        result = self.get_current_gaussians(window_size=win)
        if not isinstance(result, tuple) or len(result) != 3 or len(result[1]) == 0:
            return
        g, frame_slices, views = result

        # -----------------------------
        # Optimize scale & rotation
        # -----------------------------
        g = optimize_scale_rotation(
            g,
            views=views,
            GaussianRasterizer=GaussianRasterizer,
            GaussianRasterizationSettings=GaussianRasterizationSettings,
            iters=iters,
            lr_scale=lr_scale,
            lr_rot=lr_rot,
            lr_opa=lr_opa,
            views_per_iter=min(views_per_iter, len(views)),
            bg_rgb=(0, 0, 0),
            scale_modifier=1.0,
            seed=0,
            verbose_every=1,
        )

        new_scales = g.get_scaling.detach()
        new_rots = g.get_rotation.detach()
        new_opacities = g.get_opacity.detach()

        # -----------------------------
        # Write back to each frame
        # -----------------------------
        updated_uids = []
        for uid, (s, e) in frame_slices.items():
            fg = self.cam2gaussian[uid]

            mask = fg.mask  # [H/stride, W/stride] bool

            # Move updated parameter slices from GPU back to CPU before writing into fg.
            upd_scales = new_scales[s:e].detach().cpu()
            upd_rots   = new_rots[s:e].detach().cpu()
            upd_opas   = new_opacities[s:e].detach().cpu()

            # Normally mask.sum() == e - s.
            if upd_scales.shape[0] != mask.sum().item():
                # Warn to simplify future debugging.
                self.info(f"[FrameGaussians][WARN] uid={uid} mask.sum()={mask.sum().item()} slice_len={upd_scales.shape[0]}")
                # Fallback: match by the minimum count.
                n = min(mask.sum().item(), upd_scales.shape[0])
                if n == 0:
                    fg.cpu()
                    self.cam2gaussian[uid] = fg
                    updated_uids.append(uid)
                    continue

                # Flatten indices and update the first n valid entries.
                flat_idx = mask.view(-1).nonzero(as_tuple=False).view(-1)[:n]

                fg_scales_flat = fg.scales.view(-1, fg.scales.shape[-1])
                fg_rots_flat   = fg.rotations.view(-1, fg.rotations.shape[-1])
                fg_opas_flat   = fg.opacities.view(-1, fg.opacities.shape[-1])

                fg_scales_flat[flat_idx] = upd_scales[:n]
                fg_rots_flat[flat_idx]   = upd_rots[:n]
                fg_opas_flat[flat_idx]   = upd_opas[:n]
            else:
                # Ideal case: one-to-one correspondence, write back using mask directly.
                fg.scales[mask]    = upd_scales
                fg.rotations[mask] = upd_rots
                fg.opacities[mask] = upd_opas

            fg.cpu()
            self.cam2gaussian[uid] = fg
            updated_uids.append(uid)

        # self.get_current_metrics()
        self._maybe_save_aligned_gaussians_per_update(frame_id=self.cur_idx)

    def __call__(self, mapping_queue: mp.Queue, received_item: mp.Event, the_end: bool = False):

        self.cur_idx = self.video.counter.value
        print(f"[Gaussian Mapper] __call__(the_end={the_end}) cur_idx={self.video.counter.value} last_idx={self.last_idx} delay={self.delay}, cur_idx={self.cur_idx}, warmup={self.warmup}", flush=True)
        
        self._init_ov_model()

        # 1) Normal online update: new frames are available and warmup has passed.
        if (not the_end) and (self.last_idx + self.delay < (self.cur_idx + 1)) and ((self.cur_idx + 1) > self.warmup):
            self._update(delay_to_tracking=True, iters=self.mapping_iters, release_cache=False)
            self.count += 1
            return False

        # 2) Final stage: the last keyframe batch is not fully consumed yet, keep updating.
        if the_end and (self.last_idx + self.delay < self.cur_idx) and ((self.cur_idx + 1) > self.warmup):
            self._update(delay_to_tracking=True, iters=self.mapping_iters, release_cache=False)
            self.count += 1

            # force
            self.last_idx = self.cur_idx
            return False

        # 3) Final stage: the last keyframe batch is ready, run the final update and last_call.
        if the_end and (self.last_idx + self.delay) >= self.cur_idx:
            print("\n[Gaussian Mapper] >>> ENTER _last_call <<<\n", flush=True)

            # Disable delay_to_tracking for the final round so the last batch is fully consumed.
            self._update(iters=self.mapping_iters + 10, delay_to_tracking=False, release_cache=False)
            self.count += 1

            self._last_call(mapping_queue=mapping_queue, received_item=received_item)
            return True

    def get_aligned_gaussians(self):

        scene_data = self.gt_scene_data
        gt_occ_pts = self.gt_occ_pts  # Only kept for later visualization; Sim3 alignment uses poses only.

        # Use the 3DGS representation tied to SLAM xyz from get_current_gaussians().
        g, frame_slices, views = self.get_current_gaussians()
        self.gaussians = g

        # Ensure the global Sim(3) alignment has been estimated from SLAM/GT trajectories.
        self._compute_pose_alignment()

        xyz = g.get_xyz                            # [N,3], currently in the SLAM world frame
        rotation = quaternion_to_matrix(g.get_rotation)
        R_a = self.align_R.to(rotation)                   # [3,3]
        rotation_aligned = R_a.unsqueeze(0) @ rotation    # [N,3,3]
        scale = g.get_scaling
        opacity = g.get_opacity
        # feats = g.get_features.flatten(1)
        semantics = g.ov_feat
        rot_q = g.get_rotation
        rot_m = quaternion_to_matrix(rot_q)  # [N,3,3]

        # Apply Sim(3): gt_world ~= s * R @ pred_world + t.
        # xyz: [N,3]; align_R: [3,3]; align_t: [3]
        s = self.align_s
        R = self.align_R
        t = self.align_t

        # Apply R @ xyz^T, then transpose back to [N,3].
        points_aligned = (R @ xyz.t()).t()  # [N,3]
        points_aligned = points_aligned * s + t.unsqueeze(0)  # [N,3]
        scale_aligned = scale * s  # Scale Gaussian size as well.
        rotation_aligned_m = R[None, :, :] @ rot_m  # [N,3,3]
        rotation_aligned_q = matrix_to_quaternion(rotation_aligned_m)

        final_aligned_gaussians = GaussianModel(
            sh_degree=0, use_surface_points=False)

        final_aligned_gaussians._xyz = points_aligned
        final_aligned_gaussians._scaling = final_aligned_gaussians.scaling_inverse_activation(scale_aligned)
        final_aligned_gaussians._rotation = rotation_aligned_q
        final_aligned_gaussians._opacity = final_aligned_gaussians.inverse_opacity_activation(opacity)
        final_aligned_gaussians._features_dc = g._features_dc
        final_aligned_gaussians.ov_feat = semantics

        return final_aligned_gaussians

    def get_current_gaussians(self, window_size=None):
        # -----------------------------
        # Choose window frames (prefer keyframes)
        # -----------------------------
        # keyframes in mapping are those in cam2buffer (you keep uid->uid for keyframes)
        kf_uids = [int(cam.uid) for cam in self.cameras if int(cam.uid) in self.cam2buffer]
        kf_uids = sorted(kf_uids)

        if window_size is not None:
            if len(kf_uids) == 0:
                return [], {}, []
            win_uids = kf_uids[-window_size:]
        else:
            win_uids = kf_uids

        # -----------------------------
        # Refresh / build per-frame cached gaussians
        #   - keep stable sampling via fg.mask (flatten pixel indices)
        # -----------------------------
        frame_slices = {}  # uid -> (start, end)
        pts_all, rgb_all, sc_all, rot_all, op_all, feat_all = [], [], [], [], [], []
        ts_all = []
        views = []
        mask_all = []
        view_id_all = []

        for uid in win_uids:
            fg = self.cam2gaussian[uid]
            fg.cuda()
            color, depth, depth_prior, intrinsics, w2c_lie, stat_mask, _ts = self.video.get_mapping_item(uid, device=self.device)

            # --- OV logits cache per uid ---
            if fg.ov_feat is None:
                if uid in self.ov_cache:
                    fg.ov_feat = self.ov_cache[uid].to(self.device, non_blocking=True)
                else:
                    with torch.no_grad():
                        logits = self.get_ov_pred(color)[0].detach()   # [C,H,W] on GPU
                    self.ov_cache[uid] = logits.cpu()                 # store on CPU
                    fg.ov_feat = logits                               # keep on GPU for this call

            # --- NEW: label cache per uid (cheap) ---
            if not hasattr(self, "ov_label_cache"):
                self.ov_label_cache = {}

            if uid not in self.ov_label_cache:
                with torch.no_grad():
                    label = fg.ov_feat.argmax(dim=0)  # [H,W] on GPU
                self.ov_label_cache[uid] = label.to(torch.long).cpu()

            ts_all.append(_ts)

            color = color[:, ::self.stride, ::self.stride]
            depth = depth[::self.stride, ::self.stride]
            feat = fg.ov_feat[:, ::self.stride, ::self.stride]

            # pick depth source
            d = depth

            # normalize shapes
            color = color / 255.0
            stat_mask = stat_mask[::self.stride, ::self.stride]

            d = d.to(device=self.device)
            if d.ndim == 3:
                d = d[0]
            H, W = int(d.shape[-2]), int(d.shape[-1])

            fx, fy, cx, cy = [float(x) / self.stride for x in intrinsics]

            # build valid mask
            valid = (d > 0)
            valid = valid & stat_mask
            valid = valid & fg.mask

            # --- early skip frames with no valid points ---
            if valid.sum().item() == 0:
                fg.mask = valid.detach()
                fg.cpu()
                self.cam2gaussian[uid] = fg
                continue

            yy, xx = torch.meshgrid(
                torch.arange(H, device=d.device, dtype=d.dtype),
                torch.arange(W, device=d.device, dtype=d.dtype),
                indexing="ij",
            )

            # compute pts_cam from pix_idx (avoid meshgrid)
            z = depth
            x = (xx - cx) / fx * z
            y = (yy - cy) / fy * z
            pts_cam = torch.stack([x, y, z], dim=-1)[valid]

            # colors
            rgb = color.permute(1, 2, 0)[valid]
            feat = feat.permute(1, 2, 0)[valid]
            # pose
            w2c = lie_to_matrix(w2c_lie).to(device=self.device)
            c2w = torch.linalg.inv(w2c)
            pts_h = torch.cat([pts_cam, torch.ones((pts_cam.shape[0], 1), device=pts_cam.device, dtype=pts_cam.dtype)], dim=-1)
            pts_w = (c2w @ pts_h.T).T[:, :3]

            mode = getattr(self, "gui_vis_mode", "rgb")
            if mode in ["ov3d_label"]:
                label_hw = None
                if hasattr(self, "ov_label_cache"):
                    label_hw = self.ov_label_cache.get(uid, None)

                if label_hw is None:
                    # Fallback: use argmax over fg.ov_feat.
                    with torch.no_grad():
                        label_hw = fg.ov_feat.argmax(dim=0).to(torch.uint16).cpu()

                # Align labels to the configured stride.
                if self.stride != 1:
                    label_hw_s = label_hw[::self.stride, ::self.stride]
                else:
                    label_hw_s = label_hw

                pal = self._get_palette_auto()
                sem_rgb = self._colors_from_label_hw(label_hw_s, valid, pal)  # [N,3] GPU float

                rgb = sem_rgb
            R_c2w = c2w[:3, :3]                          # cam->world
            q_c2w = _rotmat_to_quat_wxyz(R_c2w)          # (4,) wxyz
            q_c2w = F.normalize(q_c2w, dim=-1, eps=1e-8)

            Ni = pts_w.shape[0]

            # --- sanitize per-frame params BEFORE concatenation ---
            # Ensure fg.scales > 0
            scales_lin = fg.scales[valid].to(color)
            scales_lin = torch.clamp(scales_lin, min=1e-6)
            scales = torch.log(scales_lin)

            rotations_cam = fg.rotations[valid].to(color)
            if rotations_cam.ndim == 1:
                rotations_cam = rotations_cam.view(1, -1)

            if rotations_cam.shape[-1] != 4:
                rotations_world = torch.zeros((Ni, 4), device=pts_w.device, dtype=torch.float32)
                rotations_world[:, 0] = 1.0
            else:
                rotations_cam = F.normalize(rotations_cam, dim=-1, eps=1e-8)
                # local->world = (cam->world) ⊗ (local->cam)
                qcw = q_c2w.view(1, 4).expand(rotations_cam.shape[0], 4).to(rotations_cam)
                rotations_world = _quat_mul_wxyz(qcw, rotations_cam)
                rotations_world = F.normalize(rotations_world, dim=-1, eps=1e-8)

            rotations = rotations_world

            # Ensure opacity in (0,1) and finite before inverse_sigmoid
            opa_lin = fg.opacities[valid].to(color)
            opa_lin = torch.clamp(opa_lin, 1e-6, 1.0 - 1e-6)
            opacities = inverse_sigmoid(opa_lin)

            # Filter per-frame invalid rows (NaN/Inf) to avoid poisoning merged g
            finite = torch.isfinite(pts_w).all(dim=-1)
            finite = finite & torch.isfinite(rgb).all(dim=-1)
            finite = finite & torch.isfinite(scales).all(dim=-1)
            finite = finite & torch.isfinite(rotations).all(dim=-1)
            finite = finite & torch.isfinite(opacities).all(dim=-1)

            if finite.sum().item() == 0:
                fg.mask = valid.detach()
                fg.cpu()
                self.cam2gaussian[uid] = fg
                continue

            Ni = pts_w.shape[0]

            fg.mask = valid.detach()

            # build View for optimization
            views.append(
                View(
                    image_chw=color.detach(),
                    w2c=w2c.detach(),
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    znear=0.01, zfar=100.0,
                    ts=_ts
                )
            )

            # append to global concat lists
            start = sum([p.shape[0] for p in pts_all]) if len(pts_all) > 0 else 0

            view_idx = len(views)

            view_id_all.append(torch.full((Ni,), view_idx, device=finite.device, dtype=torch.long))  # NEW
            mask_all.append(finite)
            pts_all.append(pts_w)
            rgb_all.append(rgb)
            feat_all.append(feat)
            sc_all.append(scales)
            rot_all.append(rotations)
            op_all.append(opacities)
            end = start + Ni
            frame_slices[uid] = (start, end)

            fg.cpu()
            self.cam2gaussian[uid] = fg

        # If nothing collected, return empty
        if len(pts_all) == 0:
            return [], {}, []

        # -----------------------------
        # Build a lightweight gaussian container for optimize_scale_rotation()
        # -----------------------------
        pts = torch.cat(pts_all, dim=0)
        rgb = torch.cat(rgb_all, dim=0)
        feat = torch.cat(feat_all, dim=0)
        scales0 = torch.cat(sc_all, dim=0)
        rots0 = torch.cat(rot_all, dim=0)
        op0 = torch.cat(op_all, dim=0)
        masks = torch.cat(mask_all, dim=0)
        view_ids = torch.cat(view_id_all, dim=0)

        g = GaussianModel(sh_degree=0, config=self.cfg.mapping.input, use_surface_points=True)
        g._surface_xyz = nn.Parameter(pts, requires_grad=False)
        g._features_dc = nn.Parameter(rgb[:, :, None], requires_grad=False)
        g._features_rest = nn.Parameter(torch.empty((Ni, 0, 1), device=self.device, dtype=torch.float32), requires_grad=False)
        g._scaling = nn.Parameter(scales0, requires_grad=False)
        g._rotation = nn.Parameter(F.normalize(rots0, dim=-1, eps=1e-8), requires_grad=False)
        g._opacity = nn.Parameter(op0, requires_grad=False)
        g._mask = nn.Parameter(masks, requires_grad=False)
        g.max_radii2D = torch.zeros((Ni,), device=self.device, dtype=torch.float32)
        g.xyz_gradient_accum = torch.zeros((Ni, 1), device=self.device, dtype=torch.float32)
        g.ov_feat = feat
        g._view_ids = view_ids
        g.views = views
        return g, frame_slices, views
