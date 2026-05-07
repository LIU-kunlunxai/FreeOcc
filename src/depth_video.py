from typing import Optional, List
import torch
import torch.multiprocessing as mp
import lietorch
import droid_backends
from .droid_net import cvx_upsample
from .geom import projective_ops as pops
from .geom import matrix_to_lie, check_and_correct_transform


class DepthVideo:
    """
    Data structure of multiple buffers to keep track of indices, poses, disparities, images, external disparities and more
    """

    def __init__(self, cfg):
        self.cfg = cfg

        ### Intrinsics / Calibration ###
        if cfg.data.cam.camera_model == "pinhole":
            self.n_intr = 4
            self.model_id = 0
        elif cfg.data.cam.camera_model == "mei":
            self.n_intr = 5
            self.model_id = 1
        else:
            raise Exception("Camera model not implemented! Choose either pinhole or mei model.")
        self.opt_intr = cfg.opt_intr

        self.ready = mp.Value("i", 0)
        self.counter = mp.Value("i", 0)

        ht = cfg.data.cam.H_out
        self.ht = ht
        wd = cfg.data.cam.W_out
        self.wd = wd
        self.stereo = cfg.mode == "stereo"
        device = cfg.device
        self.device = device
        c = 1 if not self.stereo else 2
        self.scale_factor = 8
        s = self.scale_factor
        self.buffer_size = cfg.tracking.buffer
        # Whether we upsample the predictions or not
        self.upsampled = cfg.tracking.upsample

        buffer = cfg.tracking.buffer
        ### state attributes -> Raw map ###
        self.timestamp = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        # List for keeping track of updated frames for visualization
        self.dirty = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        # List for keeping track of updated frames for Map Renderer
        self.mapping_dirty = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()

        # NOTE chen: simply storing images in original uint8 and only normalizing them when needed SAVES a SHITTON of memory
        self.images = torch.zeros(buffer, 3, ht, wd, device=device, dtype=torch.uint8).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=device, dtype=torch.float).share_memory_()
        self.poses = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # c2w quaterion
        self.poses_gt = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # c2w quaterion

        # Measure the change of poses before and after backend optimization, so we can track large map changes
        self.pose_changes = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # c2w quaterion
        self.scale_changes = torch.ones(buffer, device=device, dtype=torch.float).share_memory_()  # Float

        self.disps = torch.ones(buffer, ht // s, wd // s, device=device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        # Estimated confidence weights for Optimization reduced for each node from factor graph
        self.confidence = torch.zeros(buffer, ht // s, wd // s, device=device, dtype=torch.float)
        self.confidence_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()

        self.disps_sens_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht // s, wd // s, device=device, dtype=torch.float).share_memory_()
        self.mode = cfg.mode

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht // s, wd // s, dtype=torch.half, device=device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht // s, wd // s, dtype=torch.half, device=device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht // s, wd // s, dtype=torch.half, device=device).share_memory_()

        ### Initialize poses to identity transformation
        self.poses[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        self.poses_gt[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        self.pose_changes[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)

        ### Additional flags for multi-view filter -> Clean map for rendering ###
        if self.cfg.tracking.upsample:
            self.disps_clean = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        else:
            self.disps_clean = torch.zeros(buffer, ht // s, wd // s, device=device, dtype=torch.float).share_memory_()
        # Poses that have been finetuned by the Rendering module, we could reassign these back to the DepthVideo
        self.poses_clean = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # w2c quaterion
        self.poses_clean[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)

        # Keep track of which frames have been filtered
        self.filtered_id = torch.tensor([0], dtype=torch.int, device=device).share_memory_()

        self.static_masks = torch.ones(buffer, ht, wd, device=device, dtype=torch.bool).share_memory_()

        # DEBUG flag: whether to compute and print SLAM xyz / BA-after xyz per BA call
        self.debug_xyz = getattr(cfg, "debug_xyz", False)

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        s = self.scale_factor
        self.timestamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None and self.cfg.mode != "mono":
            depth_up = item[4]
            self.disps_sens_up[index] = torch.where(depth_up > 0, 1.0 / depth_up, depth_up)
            self.disps_sens[index] = self.disps_sens_up[index][..., int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]
            self.disps[index] = self.disps_sens[index]

        if item[5] is not None:
            # NOTE chen: we always work with the downscaled images/disps for optimization so store intrinsics at that scale
            self.intrinsics[index] = item[5] / s

        if len(item) > 6 and item[6] is not None:
            self.fmaps[index] = item[6]

        if len(item) > 7 and item[7] is not None:
            self.nets[index] = item[7]

        if len(item) > 8 and item[8] is not None:
            self.inps[index] = item[8]

        if len(item) > 9 and item[9] is not None:
            self.poses_gt[index] = item[9].to(self.poses_gt.device)

        if len(item) > 10 and item[10] is not None:
            self.static_masks[index] = item[10]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """index the depth video"""

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index > 0:
                index = self.counter.value + index
            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index],
            )

        return item

    def remove(self, index) -> None:
        """Given a list of indices, we want to reset these items to the initial values.

        Example use case: In the trajectory filler we use a small intermediate buffer at the end of the video
        to optimize intermediate poses before returning them. Keeping the overall video buffer clean afterwards should
        be a priority.
        """
        self.timestamp[index] = torch.zeros_like(self.timestamp[index], dtype=torch.float, device=self.device)
        self.images[index] = torch.zeros_like(self.images[index], dtype=torch.uint8, device=self.device)
        self.intrinsics[index] = torch.zeros_like(self.intrinsics[index], dtype=torch.float, device=self.device)

        zero_poses = torch.zeros_like(self.poses[index], dtype=torch.float, device=self.device)
        zero_poses[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)
        self.poses[index] = zero_poses
        self.poses_gt[index] = zero_poses

        self.disps[index] = torch.ones_like(self.disps[index], dtype=torch.float, device=self.device)
        self.disps_up[index] = torch.zeros_like(self.disps_up[index], dtype=torch.float, device=self.device)
        self.disps_sens[index] = torch.zeros_like(self.disps_sens[index], dtype=torch.float, device=self.device)
        self.disps_sens_up[index] = torch.zeros_like(self.disps_sens_up[index], dtype=torch.float, device=self.device)
        self.fmaps[index] = torch.zeros_like(self.fmaps[index], dtype=torch.half, device=self.device)
        self.nets[index] = torch.zeros_like(self.nets[index], dtype=torch.half, device=self.device)
        self.inps[index] = torch.zeros_like(self.inps[index], dtype=torch.half, device=self.device)

        self.static_masks[index] = torch.ones_like(self.static_masks[index], dtype=torch.bool, device=self.device)

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    def pad_indices(self, indices: torch.Tensor, radius: int = 3) -> torch.Tensor:
        """Given a sequence of indices, we want to include surrounding indices as well within a radius.
        This is useful, as when we want to compute multi-view consistency we need to consider surrounding frames.
        Since we dont want do this for the whole map all the time, we only pad a given list of indices.
        """
        padded_indices = []
        # Dont pad past the sequence, because we might not have visited these frames at all
        last_frame = indices.max().item()
        for ix in indices:
            padded_indices.append(
                torch.arange(max(0, ix - radius), min(last_frame + 1, ix + radius + 1), device=indices.device)
            )
        padded_indices = torch.cat(padded_indices)
        return torch.unique(padded_indices)

    def filter_map(
        self,
        idx: Optional[torch.Tensor] = None,
        radius: int = 2,
        mv_count_th: int = 2,
        bin_th: float = 0.1,
        min_disp_th: float = 0.01,  # Relative to distribution, i.e. reject the lowest 1%
        hard_disp_th: float = 1e-3,  # Absolute, i.e. reject depth > 1000m
        conf_th: float = 0.1,
        multiview: bool = True,
        uncertainty: bool = True,
        return_mask: bool = False,
        set_video: bool = True,  # Whether to actually set the cleaned disps in the video
    ) -> None:
        """Filter the map based on consistency across multiple views and uncertainty.
        Normally this is done based on only a selected few views. We extend the selection with a local neighborhood
        to achieve a better estimate of consistent points.
        """

        with self.get_lock():
            if idx is None:
                (dirty_index,) = torch.where(self.mapping_dirty.clone())
                dirty_index = dirty_index
            else:
                dirty_index = idx

            if len(dirty_index) == 0:
                return

            # Check for multiview consistency not in the whole map, but also not only in a few local frames
            # -> Pad to neighborhoods, so we can get many consistent points
            dirty_index = self.pad_indices(dirty_index, radius=radius)

            if self.upsampled:
                disps = torch.index_select(self.disps_up, 0, dirty_index).clone()
                intrinsics = self.intrinsics[0] * self.scale_factor
                if uncertainty:
                    conf = torch.index_select(self.confidence_up, 0, dirty_index).clone()
            else:
                disps = torch.index_select(self.disps, 0, dirty_index).clone()
                intrinsics = self.intrinsics[0]
                if uncertainty:
                    conf = torch.index_select(self.confidence, 0, dirty_index).clone()

        mask = torch.ones_like(disps, dtype=torch.bool)
        if multiview:
            # Only take pixels where multiple points are consistent across views and they do not have an outlier disparity
            thresh = bin_th * torch.ones_like(disps.mean(dim=[1, 2]))
            if self.upsampled:
                count = droid_backends.depth_filter(self.poses, self.disps_up, intrinsics, dirty_index, thresh)
            else:
                count = droid_backends.depth_filter(self.poses, self.disps, intrinsics, dirty_index, thresh)
            mv_mask = count >= mv_count_th
            mask = mask & mv_mask

        if uncertainty:
            conf_mask = conf > conf_th
            mask = mask & conf_mask

        # Filter away spurious points with very low disparities (relative to distribution and hard threshold)
        inliers = torch.logical_and(disps > min_disp_th * disps.mean(dim=[1, 2], keepdim=True), disps > hard_disp_th)
        mask = mask & inliers

        disps[~mask] = 0.0  # Filter away invalid points
        # Assign clean values to datastructure
        if set_video:
            self.disps_clean[dirty_index] = disps
            self.filtered_id = max(dirty_index.max().item(), self.filtered_id)

        if return_mask:
            return mask

    def get_mapping_item(self, index, use_gt=False, device="cuda:0", return_raw_depth: bool = False):
        """Get a part of the video to transfer to the Rendering module

        If return_raw_depth=True, also return the depth constructed directly from self.disps[index]
        (without any external filtering), which can be used to compute SLAM-side point clouds for
        debugging / comparison against 3DGS.
        """

        s = self.scale_factor
        with self.get_lock():
            if self.upsampled:
                image = self.images[index].clone()  # [H, W, 3]
                static_mask = self.static_masks[index].clone().to(device)  # [H, W]
                intrinsics = self.intrinsics[0].clone().contiguous().to(device) * s  # [4]
                disp_prior = self.disps_sens_up[index].contiguous().clone().to(device)  # [H, W]
            else:
                # Color is always stored in the original resolution, downsample here to match
                image = self.images[index, ..., int(s // 2 - 1) :: s, int(s // 2 - 1) :: s].clone()
                image = image.contiguous().to(device)  # [C, H // s, W // s]
                static_mask = self.static_masks[index, ..., int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]
                static_mask = static_mask.contiguous().to(device)  # [H // s, W // s]
                intrinsics = self.intrinsics[0].clone().contiguous().to(device)  # [4]
                disp_prior = self.disps_sens[index].contiguous().clone().to(device)  # [H // s, W // s]

            est_disp = self.disps_clean[index].clone().contiguous().to(device)  # [H, W]
            est_depth = torch.where(est_disp > 0, 1.0 / est_disp, est_disp)
            # Some modes do not have any disps_sens
            if self.cfg.mode == "rgbd":
                depth_prior = torch.where(disp_prior > 0, 1.0 / disp_prior, disp_prior)  # Prior depth
            else:
                depth_prior = est_depth

            # Optional: raw depth directly from tracked disparities (before cleaning)
            raw_depth = None
            if return_raw_depth:
                raw_disp = self.disps[index].clone().contiguous().to(device)
                raw_depth = torch.where(raw_disp > 0, 1.0 / raw_disp, raw_disp)

            # [7, 1]
            if use_gt:
                c2w = lietorch.SE3(self.poses_gt[index].clone()).to(device)
                w2c = c2w.inv().vec()
            else:
                w2c = self.poses[index].clone().to(device)

            ts = self.timestamp[index]

            if return_raw_depth:
                return image, est_depth, depth_prior, intrinsics, w2c, static_mask, raw_depth, ts
            else:
                return image, est_depth, depth_prior, intrinsics, w2c, static_mask, ts

    def set_mapping_item(self, index: List[torch.Tensor], poses: List[torch.Tensor], depths: List[torch.Tensor]):
        """Set a part of the video from the Rendering module"""
        # Sanity check for when we did not render anything
        if (len(poses) == 0 and len(depths) == 0) or len(index) == 0:
            return

        # We may get only poses or only depths, so we need to check for both
        if len(poses) != 0:
            has_poses = True
            poses = torch.stack(poses)
            assert len(poses) == len(index), "Index should match the number of poses!"
        else:
            has_poses = False

        if len(depths) != 0:
            has_depths = True
            depths = torch.stack(depths)
            assert len(depths) == len(index), "Index should match the number of depths!"
            valid = depths > 0
        else:
            has_depths = False

        with self.get_lock():

            if has_depths:
                disps = torch.where(valid, 1.0 / depths, depths)
                disps.clamp_(min=1e-3)  # Sanity for optimization

                s = self.scale_factor
                # We work with the original resolution in Rendering
                if self.upsampled:
                    self.disps_up[index] = torch.where(
                        valid[:, 0], disps[:, 0].clone().detach().to(self.device), self.disps_up[index]
                    )
                    self.disps[index] = torch.where(
                        valid[:, 0, int(s // 2 - 1) :: s, int(s // 2 - 1) :: s],
                        disps[:, 0, int(s // 2 - 1) :: s, int(s // 2 - 1) :: s].clone().detach().to(self.device),
                        self.disps[index],
                    )
                # We work with downscaled resolution. This means, we need to upsample the disparities in tracking
                else:
                    self.disps[index] = torch.where(
                        valid[:, 0], disps[:, 0].clone().detach().to(self.device), self.disps[index]
                    )

            if has_poses:
                w2c = poses.clone().detach().to(self.device)  # [4, 4] homogenous matrix
                w2c_vec = matrix_to_lie(w2c)  # [7, 1] Lie element
                # Since Matrix2Lie is not unique, we need to check for sign flips!
                vid_w2c_vec = self.poses[index].clone()
                corrected = check_and_correct_transform(w2c_vec, vid_w2c_vec)
                # Sanity check: Always leave the first pose fixed
                valid_idx = torch.tensor(index)[torch.tensor(index) > 0]
                self.poses[valid_idx] = corrected[torch.tensor(index) > 0]

        self.dirty[index] = True  # Mark frames for visualization

    @staticmethod
    def format_indices(ii, jj, device="cuda"):
        """to device, long, {-1}"""
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)
        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device=device, dtype=torch.long).reshape(-1)
        jj = jj.to(device=device, dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        disps_up = cvx_upsample(self.disps[ix].unsqueeze(dim=-1), mask)  # [b, h, w, 1]
        self.disps_up[ix] = disps_up.squeeze()  # [b, h, w]

        confidence_up = cvx_upsample(self.confidence[ix].unsqueeze(-1), mask)  # [b, h, w, 1]
        self.confidence_up[ix] = confidence_up.squeeze()  # [b, h, w]

    def normalize(self):
        """normalize depth and poses"""
        with self.get_lock():
            cur_ix = self.counter.value
            s = self.disps[:cur_ix].mean()
            self.disps[:cur_ix] /= s
            self.poses[:cur_ix, :3] *= s  # [tx, ty, tz, qx, qy, qz, qw]
            self.dirty[:cur_ix] = True

    def reproject(self, ii, jj):
        """project points from ii -> jj"""
        ii, jj = DepthVideo.format_indices(ii, jj, self.device)
        Gs = lietorch.SE3(self.poses[None, ...])

        coords, valid_mask = pops.general_projective_transform(
            poses=Gs,
            depths=self.disps[None, ...],
            intrinsics=self.intrinsics[None, ...],
            ii=ii,
            jj=jj,
            model_id=self.model_id,
            jacobian=False,
            return_depth=False,
        )

        return coords, valid_mask

    def reduce_confidence(self, weights: torch.Tensor, ii: torch.Tensor, strategy: str = "avg") -> torch.Tensor:
        """Given the factor graph for the scene, we optimize poses at different camera locations.
        Each location/pose is a node in the graph and can have multiple edges to other nodes.
        For each edge we have a predicted confidence weight map given the learned feature correlations.
        We are interested in viewing these uncertainties or use them later on, as they should correlate
        with moving objects and pixels/points that do not contribute to a good reconstruction.
        Given the indices of an optimization window we reduce edges to get a single uncertainty estimate
        for each frame.

        args:
        ---
        weight [torch.Tensor]: Weight tensor of shape [len(nodes), 2, ht // 8, wd // 8]. Optimization weights for bundle adjustment.
            Each point is a vector [u_x, u_y] in [0, 1], which measures the confidence for x- and y-components.
        ii [torch.Tensor]: Indices of source nodes (We go from i to j, i.e. we have edges e_ij) with same length as jj.
        strategy [str]: How to reduce across edges. Choices: (avg, max). Given multiple confidence weight maps for
            each pixel, it is unclear how to correctly reduce this. In the end these are optimal to compute correct camera motion
            and static scene maps. Which edge contributes more to this goal is not straight-forward.
        """
        frames = ii.unique()
        idx = []
        for frame in frames:
            idx.append(frame == ii)

        frame_weights = [weights[ix] for ix in idx]
        if strategy == "avg":
            reduced = [weight.mean(dim=0) for weight in frame_weights]
        elif strategy == "max":
            reduced = [weight.max(dim=0) for weight in frame_weights]
        else:
            raise Exception("Invalid reduction strategy: {}! Use either 'avg' or 'max'".format(strategy))
        return torch.stack(reduced), frames

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """frame distance metric, where distance = sqrt((u(ii) - u(jj->ii))^2 + (v(ii) - v(jj->ii))^2)"""
        return_matrix = False
        with self.get_lock():
            N = self.counter.value

        if ii is None:
            return_matrix = True
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")

        ii, jj = DepthVideo.format_indices(ii, jj)

        intrinsic_common_id = 0  # we assume the intrinsic within one scene is the same
        if bidirectional:
            poses = self.poses[: self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[intrinsic_common_id], ii, jj, beta, self.model_id
            )

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[intrinsic_common_id], jj, ii, beta, self.model_id
            )

            d = 0.5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[intrinsic_common_id], ii, jj, beta, self.model_id
            )

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(
        self,
        target,
        weight,
        eta,
        ii,
        jj,
        t0=1,
        t1=None,
        iters=2,
        lm=1e-4,
        ep=0.1,
        motion_only=False,
        lock: Optional[mp.Lock] = None,
    ):
        """Wrapper for dense bundle adjustment. This is used both in Frontend and Backend.

        When self.debug_xyz is True, this will additionally:
            - For all frames in [t0, t1), build SLAM point clouds from poses / disps BEFORE BA
            - Run BA
            - Rebuild point clouds from updated poses / disps AFTER BA
            - Print the number of valid 3D points and the max L2 distance between
              corresponding xyz coordinates (per frame).

        This provides an explicit "SLAM xyz" and "SLAM BA-after xyz" for comparison with 3DGS.
        """
        # Use an external lock if wanted
        if lock is None:
            lock = self.get_lock()
        intrinsic_common_id = 0  # we assume the intrinsic within one scene is the same

        debug_before_xyz = None
        debug_intr = None

        if self.debug_xyz:
            with self.get_lock():
                if t1 is None:
                    t1_eff = max(ii.max().item(), jj.max().item()) + 1
                else:
                    t1_eff = t1
                # snapshot poses & disps BEFORE BA
                poses_before = self.poses[t0:t1_eff].clone()
                disps_before = self.disps[t0:t1_eff].clone()
                intr = self.intrinsics[intrinsic_common_id].clone()

            debug_intr = intr
            # convert to depth, then to xyz in camera frame (z = 1/disp)
            # NOTE: we do not apply any filtering here; this is the raw SLAM map
            B = poses_before.shape[0]
            ht, wd = disps_before.shape[-2], disps_before.shape[-1]
            yy, xx = torch.meshgrid(
                torch.arange(ht, device=disps_before.device),
                torch.arange(wd, device=disps_before.device),
                indexing="ij",
            )
            fx, fy, cx, cy = intr
            # build per-pixel x,y,z in camera coordinates
            debug_before_xyz = []
            for b in range(B):
                d = disps_before[b]
                depth = torch.where(d > 0, 1.0 / d, d)
                valid = depth > 0
                z = depth[valid]
                x = (xx[valid] - cx) * z / fx
                y = (yy[valid] - cy) * z / fy
                xyz_cam = torch.stack([x, y, z], dim=-1)  # [N,3]
                debug_before_xyz.append(xyz_cam)

        # Use actual mp.Lock for securing BA if given
        with lock:
            torch.cuda.synchronize()

            # Store the uncertainty maps for source frames, that will get updated
            confidence, idx = self.reduce_confidence(weight, ii)
            # Uncertainties are for [x, y] directions -> Take norm to get single scalar
            self.confidence[idx] = torch.norm(confidence, dim=1)

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            droid_backends.ba(
                self.poses,
                self.disps,
                self.intrinsics[intrinsic_common_id],
                self.disps_sens,
                target,
                weight,
                eta,
                ii,
                jj,
                t0,
                t1,
                iters,
                self.model_id,
                lm,
                ep,
                motion_only,
                self.opt_intr,
            )

            self.disps.clamp_(min=1e-3)  # Always make sure that Disparities are non-negative!!!
            # Reassigning intrinsics after optimization
            if self.opt_intr:
                self.intrinsics[: self.counter.value] = self.intrinsics[intrinsic_common_id]

            self.mapping_dirty[t0:t1] = True
            torch.cuda.synchronize()

        # After BA, if requested, compute BA-updated xyz and print comparison
        if self.debug_xyz and debug_before_xyz is not None and debug_intr is not None:
            with self.get_lock():
                poses_after = self.poses[t0:t1].clone()
                disps_after = self.disps[t0:t1].clone()
            intr = debug_intr
            B = disps_after.shape[0]
            ht, wd = disps_after.shape[-2], disps_after.shape[-1]
            yy, xx = torch.meshgrid(
                torch.arange(ht, device=disps_after.device),
                torch.arange(wd, device=disps_after.device),
                indexing="ij",
            )
            fx, fy, cx, cy = intr

            for b in range(B):
                d = disps_after[b]
                depth = torch.where(d > 0, 1.0 / d, d)
                valid = depth > 0
                z = depth[valid]
                x = (xx[valid] - cx) * z / fx
                y = (yy[valid] - cy) * z / fy
                xyz_cam_after = torch.stack([x, y, z], dim=-1)

                xyz_cam_before = debug_before_xyz[b]
                N_before = xyz_cam_before.shape[0]
                N_after = xyz_cam_after.shape[0]
                N = min(N_before, N_after)
                if N == 0:
                    print(f"[DEBUG][SLAM_BA][frame {t0 + b}] no valid points before/after, skip")
                    continue
                # Compare first N points (in scanline order)
                diff = xyz_cam_after[:N] - xyz_cam_before[:N]
                max_err = float(torch.norm(diff, dim=-1).max().item())
                mean_err = float(torch.norm(diff, dim=-1).mean().item())
                # print(
                #     f"[DEBUG][SLAM_BA] frame={t0 + b} N_before={N_before} N_after={N_after} "
                #     f"max_l2_diff={max_err:.6e} mean_l2_diff={mean_err:.6e}"
                # )
