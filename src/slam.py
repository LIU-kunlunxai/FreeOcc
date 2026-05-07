import os
import gc
from time import sleep, perf_counter
from typing import List, Optional, Tuple
from tqdm import tqdm
import logging
from termcolor import colored
from collections import OrderedDict
from omegaconf import DictConfig

import cv2

import torch
import torch.multiprocessing as mp
from lietorch import SE3

from .droid_net import DroidNet
from .frontend import FrontendWrapper
from .backend import BackendWrapper
from .depth_video import DepthVideo
from .visualization import droid_visualization, depth2rgb, uncertainty2rgb
from .trajectory_filler import PoseTrajectoryFiller
from .loop_detection import LoopDetector, merge_candidates
from .gaussian_mapping import GaussianMapper

from .gaussian_splatting.gui import gui_utils, slam_gui
from .utils import clone_obj, get_all_queue

# A logger for this file
log = logging.getLogger(__name__)


class SLAM:
    
    def __init__(self, cfg: DictConfig, dataset=None, output_folder: Optional[str] = None):
        super(SLAM, self).__init__()

        self.cfg = cfg
        self.device = cfg.get("device", torch.device("cuda:0"))
        self.mode = cfg.get("mode", "mono")
        self.use_gt_poses = cfg.get("use_gt_poses", False)

        self.create_out_dirs(output_folder)
        self.update_cam(cfg)  # Update output resolution, intrinsics, etc.

        ### Main SLAM neural network
        if not self.use_gt_poses:
            self.net = DroidNet()
            self.load_pretrained(cfg.tracking.pretrained)
            self.net.to(self.device).eval()
            self.net.share_memory()
        else:
            self.net = None
            self.info("GT poses mode: skipping DroidNet loading")

        # Insert a dummy delay to snychronize frontend and backend as needed
        self.sleep_time = cfg.get("sleep_delay", 0.1)

        # Delete backend when hitting this threshold, so we can keep going with just frontend
        self.max_ram_usage = cfg.get("max_ram_usage", 0.8)
        self.plot_uncertainty = cfg.get("plot_uncertainty", False)  # Show the optimization uncertainty maps

        ### Main SLAM components
        self.dataset = dataset
        self.video = DepthVideo(cfg)  # store images, depth, poses, intrinsics (shared between process)

        if not self.use_gt_poses:
            self.traj_filler = PoseTrajectoryFiller(self.cfg, net=self.net, video=self.video, device=self.device)
            self.frontend = FrontendWrapper(cfg, self)
        else:
            self.traj_filler = None
            self.frontend = None
            self.info("GT poses mode: skipping Frontend and TrajFiller")

        if self.cfg.run_backend and not self.use_gt_poses:
            self.backend = BackendWrapper(cfg, self)
            self.backend_warmup = self.backend.warmup
        else:
            self.backend = None
            self.backend_warmup = 9999

        if cfg.run_loop_detection and not self.use_gt_poses:
            self.loop_detector = LoopDetector(self.cfg.loop_closure, self.net, self.video, self.device)
        else:
            self.loop_detector = None

        if cfg.run_mapping_gui and cfg.run_mapping:
            self.q_main2vis = mp.Queue()
            self.gaussian_mapper = GaussianMapper(cfg, self, gui_qs=(self.q_main2vis))
            if not self.use_gt_poses:
                self.mapping_warmup = min(self.frontend.window, self.gaussian_mapper.warmup)
            else:
                self.mapping_warmup = self.gaussian_mapper.warmup
            self.params_gui = gui_utils.ParamsGUI(
                pipe=cfg.mapping.pipeline_params,
                background=self.gaussian_mapper.background,
                gaussians=self.gaussian_mapper.gaussians,
                q_main2vis=self.q_main2vis,
            )
        elif cfg.run_mapping:
            self.gaussian_mapper = GaussianMapper(cfg, self)
            if not self.use_gt_poses:
                self.mapping_warmup = min(self.frontend.window, self.gaussian_mapper.warmup)
            else:
                self.mapping_warmup = self.gaussian_mapper.warmup
        else:
            self.gaussian_mapper = None
            self.mapping_warmup = 9999

        self.sanity_checks()

        # Indoor ScanNet/Replica/RealSense depth visualization range.
        self.max_depth_visu = 10.0

        ### Multi-threading stuff
        # Objects for communicating between processes
        self.input_pipe = mp.Queue()  # Communicate data from Stream -> main thread
        self.mapping_queue = mp.Queue()  # Communicate data between Mapping <-> main thread
        self.ba_lock = mp.Lock()  # Block the bundle adjustment optimization explicitly
        self.received_mapping = mp.Event()  # Ensure we have received the mapping state before moving on
        self.loop_queue = mp.Queue()  # Communicate loop candidates between Loop Detection -> Backend thread

        # Manage life time of individual processes
        self.num_running_thread = torch.zeros((1)).int().share_memory_()
        self.all_trigered = torch.zeros((1)).int().share_memory_()
        self.backend_can_start = torch.zeros((1)).int().share_memory_()  # When to trigger after warmup
        self.mapping_can_start = torch.zeros((1)).int().share_memory_()  # When to trigger after warmup
        self.all_finished = torch.zeros((1)).int().share_memory_()
        self.tracking_finished = torch.zeros((1)).int().share_memory_()
        self.backend_finished = torch.zeros((1)).int().share_memory_()
        self.gaussian_mapping_finished = torch.zeros((1)).int().share_memory_()
        # NOTE chen: we use this flag to avoid frontend to start a new optimization window while the Renderer is working
        # Flag to signal that the current optimization is done or not
        self.mapping_done = torch.zeros((1)).int().share_memory_()
        self.mapping_done += 1  # Set to 1, so Tracking does not wait for mapping
        self.loop_detection_finished = torch.zeros((1)).int().share_memory_()
        self.mapping_visualizing_finished = torch.zeros((1)).int().share_memory_()

        if not self.cfg.run_mapping_gui:
            self.mapping_visualizing_finished += 1

        # Synchronization objects
        self.backend_freq = self.cfg.get("backend_every", 10)  # Run the backend every k frontend calls
        self.mapping_freq = self.cfg.get("mapper_every", 5)  # Run the Renderer/Mapper every k frontend calls
        self.sema_backend = mp.Semaphore(1)  # Semaphore allows to keep concurrency
        self.sema_mapping = mp.Semaphore(1)  # Semaphore allows to keep concurrency
        self.communication_lock = mp.Lock()
        self.cond_mapping = mp.Condition(lock=self.communication_lock)  # Conditional for fine-grained synchronization

    def info(self, msg: str, logger=None) -> None:
        if logger is not None:
            logger.info(colored("[Main]: " + msg, "green"))
        else:
            print(colored("[Main]: " + msg, "green"))

    def sanity_checks(self) -> None:
        """Perform sanity checks to see if the system is misconfigured, this is just supposed
        to protect the user when running the system"""
        assert (
            self.cfg.run_frontend or self.use_gt_poses
        ), "All other systems rely on the Frontend Tracker. This component must run at all times (unless use_gt_poses=True)!"
        if self.cfg.mode == "stereo":
            # NOTE chen: I noticed, that this is really not impemented, i.e.
            # we would need to do some changes in motion_filter, depth_video, BA, etc. to even store the right images, fmaps, etc.
            # NOTE chen: Teed definitely had the idea on dataloader level to return (image_left, image_right) as one sample for images
            raise NotImplementedError(colored("Stereo mode not supported yet!", "red"))
        if self.cfg.run_mapping_gui:
            assert self.cfg.run_mapping, colored(
                """If you want to use the Mapping GUI, you also need to run the Mapping process!""",
                "red",
            )
        if self.cfg.run_loop_detection:
            assert self.cfg.run_backend, colored(
                """We only do loop closure optimization in the backend, which optimizes the global map. 
                Use the loop detector always together with the backend enabled!""",
                "red",
            )

        if self.cfg.run_backend and self.cfg.run_mapping:
            if self.cfg.mapper_every > self.cfg.backend_every:
                print(colored("Warning. Mapping is run less often than backend!", "red"))

    def create_out_dirs(self, output_folder: Optional[str] = None) -> None:
        if output_folder is not None:
            self.output = output_folder
        else:
            self.output = "./outputs/"

        os.makedirs(self.output, exist_ok=True)
        os.makedirs(f"{self.output}/evaluation", exist_ok=True)

    def update_cam(self, cfg: DictConfig) -> None:
        """Update the camera intrinsics according to the pre-processing config, such as resize or edge crop"""
        # resize the input images to crop_size(variable name used in lietorch)
        H, W = float(cfg.data.cam.H), float(cfg.data.cam.W)
        fx, fy = cfg.data.cam.fx, cfg.data.cam.fy
        cx, cy = cfg.data.cam.cx, cfg.data.cam.cy

        h_edge, w_edge = cfg.data.cam.H_edge, cfg.data.cam.W_edge
        H_out, W_out = cfg.data.cam.H_out, cfg.data.cam.W_out

        self.fx = fx * (W_out + w_edge * 2) / W
        self.fy = fy * (H_out + h_edge * 2) / H
        self.cx = cx * (W_out + w_edge * 2) / W
        self.cy = cy * (H_out + h_edge * 2) / H
        self.H, self.W = H_out, W_out

        self.cx = self.cx - w_edge
        self.cy = self.cy - h_edge

    def load_pretrained(self, pretrained: str) -> None:
        self.info(f"Load pretrained checkpoint from {pretrained}!")

        state_dict = OrderedDict([(k.replace("module.", ""), v) for (k, v) in torch.load(pretrained).items()])
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)

    def tracking(
        self,
        rank: int,
        communication_lock: mp.Lock,
        stream,
        cond_mapping: mp.Condition,
        sema_backend: mp.Semaphore,
        sema_mapping: mp.Semaphore,
        input_queue: mp.Queue,
        ba_lock: mp.Lock = None,
    ) -> None:
        """Main driver of framework by looping over the input stream"""

        def maybe_notify_other_threads_to_start() -> None:
            # Check to notify other threads that they can start
            if self.cfg.run_backend and self.frontend.count > self.backend_warmup and self.backend_can_start < 1:
                self.backend.info("Backend warmup over!")
                self.backend_can_start += 1
            if self.cfg.run_mapping and self.frontend.count > self.mapping_warmup and self.mapping_can_start < 1:
                self.gaussian_mapper.info("Mapper warmup over!")
                self.mapping_can_start += 1

        def synchronize_with_other_threads(sema_backend: mp.Semaphore, sema_mapping: mp.Semaphore) -> None:
            # Synchronize other parallel threads
            if self.frontend.count % self.backend_freq == 0 and self.frontend.count > (self.backend_warmup + 1):
                sema_backend.release()  # Signal Backend that it can run again
            if self.frontend.count % self.mapping_freq == 0 and self.frontend.count > (self.mapping_warmup + 1):
                sema_mapping.release()  # Signal Mapping that it can run again
                self.mapping_done -= 1  # Reset flag for next Mapping call

        self.info("Frontend tracking thread started!")
        self.all_trigered += 1

        # Wait up for other threads to start
        while self.all_trigered < self.num_running_thread:
            pass

        # Main Loop which drives the whole system
        for frame in tqdm(stream):

            old_count = self.frontend.count  # Memoize current state

            if self.cfg.with_dyn and stream.has_dyn_masks:
                timestamp, image, depth, intrinsic, gt_pose, static_mask = frame
            else:
                timestamp, image, depth, intrinsic, gt_pose = frame
                static_mask = None
            if self.mode != "rgbd":
                depth = None

            # Transmit the incoming stream to another visualization thread
            if self.cfg.show_stream:
                input_queue.put(image)
                input_queue.put(depth)

            # If the Renderer / Mapping is currently optimizing, wait until that it is finished
            if self.frontend.count > self.mapping_warmup:
                with communication_lock:
                    cond_mapping.wait_for(lambda: self.mapping_done > 0)

            self.frontend(timestamp, image, depth, intrinsic, gt_pose, static_mask=static_mask, lock=ba_lock)

            maybe_notify_other_threads_to_start()
            # Check if we actually inserted a new frame and optimized
            if self.frontend.count > old_count:
                synchronize_with_other_threads(sema_backend, sema_mapping)

        self.info(f"Ran Frontend {self.frontend.count} times!")
        del self.frontend
        torch.cuda.empty_cache()
        gc.collect()

        self.tracking_finished += 1
        self.all_finished += 1
        self.info("Frontend Tracking done!")

        # Release the Semaphores to avoid deadlock
        # HACK Increase the counter just by a lot, so it will never go to 0 again
        for i in range(1000):
            sema_backend.release()
            sema_mapping.release()

    def tracking_gt(
        self,
        rank: int,
        communication_lock: mp.Lock,
        stream,
        cond_mapping: mp.Condition,
        sema_backend: mp.Semaphore,
        sema_mapping: mp.Semaphore,
        input_queue: mp.Queue,
        ba_lock: mp.Lock = None,
    ) -> None:
        """GT poses mode: populate DepthVideo directly from dataset without running SLAM.
        Keyframe selection based on camera center distance (similar to SLAM motion filter).
        GT poses + sensor depth are written directly into the video buffer."""
        from lietorch import SE3

        self.info("GT Tracking thread started (using ground truth poses)!")
        self.all_trigered += 1

        # Wait for all threads to start
        while self.all_trigered < self.num_running_thread:
            pass

        s = self.video.scale_factor
        buffer_size = self.video.buffer_size
        # Motion threshold for keyframe selection (meters of camera center movement)
        gt_motion_thresh = self.cfg.get("gt_motion_thresh", 0.1)

        frame_count = 0      # inserted keyframes
        stream_count = 0     # total frames iterated
        last_kf_center = None

        for frame in tqdm(stream, desc="GT Tracking"):
            if self.cfg.with_dyn and stream.has_dyn_masks:
                timestamp, image, depth, intrinsic, gt_pose, static_mask = frame
            else:
                timestamp, image, depth, intrinsic, gt_pose = frame
                static_mask = None

            stream_count += 1

            # Motion-based keyframe selection using camera center distance
            if gt_pose is not None and last_kf_center is not None:
                cam_center = gt_pose[:3]  # c2w translation = camera center in world
                dist = (cam_center - last_kf_center).norm().item()
                if dist < gt_motion_thresh:
                    continue  # Not enough motion, skip

            # Buffer overflow protection
            if frame_count >= buffer_size:
                self.info(f"GT Tracking: buffer full ({buffer_size}), stopping at frame {stream_count}")
                break

            if self.mode != "rgbd":
                depth = None

            # Show stream if requested
            if self.cfg.show_stream:
                input_queue.put(image)
                input_queue.put(depth)

            # Convert GT c2w pose to w2c for video.poses buffer
            if gt_pose is not None:
                c2w = SE3.InitFromVec(gt_pose.float())
                w2c_lie = c2w.inv().vec()  # [7] w2c lie vector
                last_kf_center = gt_pose[:3].clone()  # Update last keyframe center
            else:
                w2c_lie = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float)

            # Directly populate the video buffer (bypassing motion filter)
            with self.video.get_lock():
                idx = self.video.counter.value

                self.video.timestamp[idx] = timestamp
                self.video.images[idx] = image[0]  # [1, 3, H, W] -> [3, H, W]
                self.video.intrinsics[idx] = intrinsic / s
                self.video.poses[idx] = w2c_lie.to(self.device)

                if gt_pose is not None:
                    self.video.poses_gt[idx] = gt_pose.to(self.device)

                if depth is not None:
                    depth_up = depth
                    disps_sens_up = torch.where(depth_up > 0, 1.0 / depth_up, depth_up)
                    self.video.disps_sens_up[idx] = disps_sens_up
                    self.video.disps_sens[idx] = disps_sens_up[
                        ..., int(s // 2 - 1) :: int(s), int(s // 2 - 1) :: int(s)
                    ]
                    # Fill disps and disps_up with sensor depth (for filter_map)
                    self.video.disps[idx] = self.video.disps_sens[idx]
                    self.video.disps_up[idx] = disps_sens_up

                if static_mask is not None:
                    self.video.static_masks[idx] = static_mask

                self.video.mapping_dirty[idx] = True
                self.video.counter.value = idx + 1

            frame_count += 1

            # Signal mapper at appropriate intervals
            if frame_count > self.mapping_warmup and self.mapping_can_start < 1:
                self.gaussian_mapper.info("Mapper warmup over!")
                self.mapping_can_start += 1

            if self.mapping_can_start >= 1 and frame_count % self.mapping_freq == 0:
                # Wait for mapper to finish previous iteration
                if frame_count > self.mapping_warmup:
                    with communication_lock:
                        cond_mapping.wait_for(lambda: self.mapping_done > 0)
                sema_mapping.release()
                self.mapping_done -= 1

        self.info(f"GT Tracking: {frame_count} keyframes from {stream_count} frames (motion_thresh={gt_motion_thresh}m)")
        torch.cuda.empty_cache()
        gc.collect()

        self.tracking_finished += 1
        self.all_finished += 1
        self.info("GT Tracking done!")

        # Release semaphores to avoid deadlock
        for i in range(1000):
            sema_backend.release()
            sema_mapping.release()

    def loop_detection(self, rank: int, loop_queue: mp.Queue, run: bool = False) -> None:
        """Run a loop detector in parallel, which either measures optical flow between the current frame
        and all past frames or visual similarity by comparing feature descriptors.
        When two frames could be the same, we add a bidirectional edge to the backend optimization graph.
        """

        if run:
            assert self.loop_detector is not None, "Loop Detection is not enabled, but we are running it!"
            # Initialize network during worker process, since torch.hub models need to, see https://github.com/Lightning-AI/pytorch-lightning/issues/17637
            if self.loop_detector.method == "eigen" and self.loop_detector.net is None:
                self.loop_detector.net = self.loop_detector.load_eigen()

        self.info("Loop Detection thread started!")
        self.all_trigered += 1

        # Run as long as Frontend tracking gives use new frames
        while self.tracking_finished < 1 and run:
            candidates = self.loop_detector.check()
            if candidates is not None:
                self.loop_detector.info("Sending loop candidates ...")
                loop_queue.put(candidates)

        # Free memory
        del self.loop_detector
        torch.cuda.empty_cache()
        gc.collect()

        self.loop_detection_finished += 1
        self.all_finished += 1
        self.info("Loop Detection done!")

    def get_ram_usage(self) -> Tuple[float, float]:
        free_mem, total_mem = torch.cuda.mem_get_info(device=self.device)
        used_mem = 1 - (free_mem / total_mem)
        return used_mem, free_mem

    def ram_safeguard_backend(self, max_ram: float = 0.9, min_ram: float = 0.5, count_to_set: int = 0) -> None:
        """There are some scenes, where we might get into trouble with memory.
        In order to keep the system going, we simply dont use the backend until we can afford it again.
        """
        used_mem, free_mem = self.get_ram_usage()
        if used_mem > max_ram and self.backend is not None:
            print(colored(f"[Main]: Warning: Deleting Backend due to high memory usage [{used_mem} %]!", "red"))
            print(colored(f"[Main]: Warning: Warning: Got only {free_mem/ 1024 ** 3} GB left!", "red"))
            old_count = self.backend.count
            del self.backend
            self.backend = None
            gc.collect()
            torch.cuda.empty_cache()
            return old_count

        # NOTE chen: if we deleted the backend due to memory issues we likely have not a lot of capacity left for backend
        # only use backend again once we have some slack -> 50% free RAM (12GB in use)
        if self.backend is None and used_mem <= min_ram:
            self.info("Reinstantiating Backend ...")
            self.backend = BackendWrapper(self.cfg, self)
            self.backend.count = count_to_set  # Reset with memoized count
            self.backend.to(self.device)

        return count_to_set

    def get_potential_loop_update(self, loop_queue: mp.Queue):
        """Extract the loop candidates from the Queue and merge them into one set of edges."""
        try:
            new_loops = get_all_queue(loop_queue)  # Empty the whole Queue at once
            new_loops = merge_candidates(new_loops)  # In case we had multiple loop updates
            self.backend.info("Received loop candidates!")
            candidates = clone_obj(new_loops)
            del new_loops
            loop_ii, loop_jj = candidates
        except Exception as e:
            self.info("Could not get anything from the Queue! :(")
            print(colored(e, "red"))
            loop_ii, loop_jj = None, None
        return loop_ii, loop_jj

    def backend_op(
        self,
        add_ii: Optional[torch.Tensor] = None,
        add_jj: Optional[torch.Tensor] = None,
        lock: Optional[mp.Lock] = None,
    ) -> None:
        """Simple wrapper to call backend depending on which method we want to use. Reason being, that we also use the
        frontend factor graph for GO-SLAM's loop aware Bundle Adjustment.
        """
        # Use GO-SLAM's loop closure bundle adjustment (This will also consider edges with very small motion, but only in a loop window)
        if self.backend.enable_loop:
            self.backend(local_graph=self.frontend.optimizer.graph, add_ii=add_ii, add_jj=add_jj, lock=lock)
        else:
            # Use vanilla DROID Graph for Bundle adjustment
            self.backend(add_ii=add_ii, add_jj=add_jj, lock=lock)

    def final_backend_op(
        self,
        t_start=0,
        t_end=-1,
        steps: int = 6,
        add_ii=None,
        add_jj=None,
        n_repeat: int = 2,
        lock: Optional[mp.Lock] = None,
    ) -> None:
        """Make two final refinements over the whole global map. This explicitly calls the optimizer function,
        so this does not have a backend window limit, i.e. this actually runs over the whole map."""
        for i in range(n_repeat):
            # Use vanilla Graph for Bundle adjustment
            _, n_edges = self.backend.optimizer.dense_ba(
                t_start=t_start, t_end=t_end, steps=steps, add_ii=add_ii, add_jj=add_jj, lock=lock
            )
            msg = "Full BA: [{}, {}]; Using {} edges!".format(t_start, t_end, n_edges)
            self.backend.info(msg)

    def global_ba(
        self,
        rank: int,
        ba_lock: mp.Lock,
        sema_backend: mp.Semaphore,
        loop_queue: Optional[mp.Queue] = None,
        run: bool = False,
    ) -> None:
        self.info("Backend thread started!")
        self.all_trigered += 1

        memoized_backend_count = 0
        all_lc_candidates = []
        all_loop_ii, all_loop_jj = None, None

        ### Online Loop
        while self.tracking_finished < 1 and run:
            if self.backend_can_start < 1:
                continue

            sema_backend.acquire()  # Aquire the semaphore (If the counter == 0, then this thread will be blocked)
            sleep(self.sleep_time)  # Let multiprocessing cool down a little bit

            ## Only run backend if we have enough RAM for it
            memoized_backend_count = self.ram_safeguard_backend(
                max_ram=self.max_ram_usage, count_to_set=memoized_backend_count
            )
            # Backend got deactivated due to OOM
            if self.backend is None:
                continue

            ## If we run an additional loop detector -> Pull in visually similar candidate edges as well
            loop_ii, loop_jj = None, None
            if self.cfg.run_loop_detection and loop_queue is not None:
                if not loop_queue.empty():
                    loop_ii, loop_jj = self.get_potential_loop_update(loop_queue)
                    # memoize all loop candidates to always give all candidates to the backend as edges!
                    if loop_ii is not None and loop_jj is not None:
                        all_lc_candidates.append((loop_ii, loop_jj))
                        all_loop_ii, all_loop_jj = merge_candidates(all_lc_candidates)

            ### Actual Backend call
            self.backend_op(add_ii=all_loop_ii, add_jj=all_loop_jj, lock=ba_lock)

        sleep(self.sleep_time)  # Let other threads finish their last optimization
        # Try to instantiate again if needed
        if run:
            memoized_backend_count = self.ram_safeguard_backend(
                max_ram=self.max_ram_usage, count_to_set=memoized_backend_count
            )
        if run:
            if self.backend is not None:
                self.info(f"Ran Backend {self.backend.count} times before refinement!")
            else:
                self.info(
                    """Backend Refinement does not fit into memory! Please run the system in a 
                    different configuration for this to work. (Maybe use lower resolution)"""
                )

        ### Run one last time after tracking finished
        if run and self.backend is not None and self.backend.do_refinement:
            with self.video.get_lock():
                t_end = self.video.counter.value

                msg = "Optimize full map: [{}, {}]!".format(0, t_end)
                self.backend.info(msg)
                self.final_backend_op(t_start=0, t_end=t_end, add_ii=all_loop_ii, add_jj=all_loop_jj, lock=ba_lock)

        if self.backend is not None:
            del self.backend
            torch.cuda.empty_cache()
            gc.collect()

        self.backend_finished += 1
        self.all_finished += 1
        self.info("Backend done!")

    def gaussian_mapping(
        self,
        rank: int,
        communication_lock: mp.Lock,
        cond_mapping: mp.Condition,
        sema_mapping: mp.Semaphore,
        mapping_queue: mp.Queue,
        received_mapping: mp.Event,
        run: bool,
    ) -> None:
        self.info("Gaussian Mapping Triggered!")
        self.all_trigered += 1

        while (self.tracking_finished + self.backend_finished) < 2 and run:

            # Wait for warmup phase to finish
            if self.mapping_can_start < 1:
                continue

            sema_mapping.acquire()  # Aquire the semaphore (If the counter == 0, then this thread will be blocked)

            self.gaussian_mapper(mapping_queue, received_mapping)

            # Notify leading Tracking thread, that we finished the current Render optimization
            # -> This avoids both Frontend and Renderer to run at the same time and conflict on depth_video
            # (Notify Tracking thread only when it is still alive, else we get a deadlock)
            if self.tracking_finished < 1:
                with communication_lock:
                    self.mapping_done += 1
                    cond_mapping.notify()

            sleep(self.sleep_time)  # Let system cool off a little

        # Run for one last time after everything finished
        if self.gaussian_mapper is not None and run:
            self.info(f"Ran Gaussian Mapper {self.gaussian_mapper.count} times before Refinement!")

        finished = False
        while not finished and run:
            finished = self.gaussian_mapper(mapping_queue, received_mapping, True)

        self.gaussian_mapping_finished += 1
        # Let the user still interact with the GUI
        while self.mapping_visualizing_finished < 1:
            pass

        self.all_finished += 1
        self.info("Gaussian Mapping Done!")

    def visualizing(self, rank: int, run=True) -> None:
        """Vanilla Point Cloud Visualizer in Open3D"""
        self.info("Visualization thread started!")
        self.all_trigered += 1
        finished = False

        while (self.tracking_finished + self.backend_finished < 2) and run and not finished:
            finished = droid_visualization(self.video, device=self.device, save_root=self.output)

        self.all_finished += 1
        self.info("Visualization done!")

    def mapping_gui(self, rank: int, run=True) -> None:
        """Gaussian Splatting Visualizer in Open3D"""
        # import sys, pdb; sys.stdin = open(0); sys.stdout = open(1, "w", buffering=1); pdb.set_trace()

        self.info("Mapping GUI thread started!")
        self.all_trigered += 1
        finished = False

        while (self.tracking_finished + self.backend_finished < 2) and run and not finished:
            finished = slam_gui.run(self.params_gui)

        # Wait for Gaussian Mapper to be finished so nothing new is put into the queue anymore
        while self.gaussian_mapping_finished < 1:
            pass

        # empty all the guis that are in params_gui so this will for sure get empty
        if run:  # NOTE Leon: It crashes if we dont check this
            while not self.params_gui.q_main2vis.empty():
                obj = self.params_gui.q_main2vis.get()
                a = clone_obj(obj)
                del obj

        self.mapping_visualizing_finished += 1
        self.all_finished += 1
        self.info("Mapping GUI done!")

    def show_stream(self, rank, input_queue: mp.Queue, run=True) -> None:
        """Show the input RGBD stream (+ confidence of network) in separate windows"""
        self.info("OpenCV Image stream thread started!")
        self.all_trigered += 1

        while (self.tracking_finished + self.backend_finished < 2) and run:
            if not input_queue.empty():
                try:
                    rgb = input_queue.get()
                    depth = input_queue.get()

                    rgb_image = rgb[0, [2, 1, 0], ...].permute(1, 2, 0).clone().cpu()
                    cv2.imshow("RGB", rgb_image.numpy())
                    if self.mode == "rgbd" and depth is not None:
                        # Create normalized depth map with intensity plot
                        depth_image = depth2rgb(depth.clone().cpu(), max_depth=self.max_depth_visu)[0]
                        # Convert to BGR for cv2
                        cv2.imshow("depth", depth_image[..., ::-1])
                    cv2.waitKey(1)
                except Exception as e:
                    # print(colored(e, "red"))
                    pass

            if self.plot_uncertainty:
                # Plot the uncertainty on top
                with self.video.get_lock():
                    t_cur = max(0, self.video.counter.value - 1)
                    if self.cfg.tracking.get("upsample", False):
                        uncertanity_cur = self.video.confidence_up[t_cur].clone()
                    else:
                        uncertanity_cur = self.video.confidence[t_cur].clone()
                uncertainty_img = uncertainty2rgb(uncertanity_cur)[0]
                cv2.imshow("Uncertainty", uncertainty_img[..., ::-1])
                cv2.waitKey(1)

        self.all_finished += 1
        self.info("Show stream Done!")

    def terminate(
        self,
        processes: List[mp.Process],
        stream=None,
    ) -> None:
        """Shut down all processes."""

        if self.video.opt_intr:
            self.info("Final estimated video intrinsics: {}".format(self.video.intrinsics[0]), logger=log)

        self.info("Initiating termination ...", logger=log)

        for i, p in enumerate(processes):
            p.terminate()
            p.join()
            self.info("Terminated process {}".format(p.name))
        self.info("Terminate: Done!", logger=log)

    def run(self, stream) -> None:
        """Main SLAM function to manage the multi-threaded application."""
        # Choose tracking target based on mode
        tracking_target = self.tracking_gt if self.use_gt_poses else self.tracking
        tracking_name = "GT Tracking" if self.use_gt_poses else "Frontend Tracking"
        # In GT mode, disable backend and loop detection
        run_backend = self.cfg.run_backend and not self.use_gt_poses
        run_loop = self.cfg.run_loop_detection and not self.use_gt_poses

        processes = [
            # NOTE The OpenCV thread always needs to be 0 to work somehow
            mp.Process(target=self.show_stream, args=(0, self.input_pipe, self.cfg.show_stream), name="OpenCV Stream"),
            mp.Process(
                target=tracking_target,
                args=(
                    1,
                    self.communication_lock,
                    stream,
                    self.cond_mapping,
                    self.sema_backend,
                    self.sema_mapping,
                    self.input_pipe,
                    self.ba_lock,
                ),
                name=tracking_name,
            ),
            mp.Process(
                target=self.global_ba,
                args=(2, self.ba_lock, self.sema_backend, self.loop_queue, run_backend),
                name="Backend",
            ),
            mp.Process(
                target=self.loop_detection,
                args=(3, self.loop_queue, run_loop),
                name="Loop Detector",
            ),
            mp.Process(
                target=self.visualizing,
                args=(4, self.cfg.run_mapping_gui and self.cfg.run_mapping),
                name="Visualizing",
            ),
            mp.Process(
                target=self.gaussian_mapping,
                args=(
                    5,
                    self.communication_lock,
                    self.cond_mapping,
                    self.sema_mapping,
                    self.mapping_queue,
                    self.received_mapping,
                    self.cfg.run_mapping,
                ),
                name="Gaussian Mapping",
            ),
            mp.Process(
                target=self.mapping_gui,
                args=(6, self.cfg.run_mapping_gui and self.cfg.run_mapping),
                name="Mapping GUI",
            ),
        ]

        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        start_time = perf_counter()
        self.info(str(start_time), logger=log)

        # Wait for all processes to have finished before terminating and for final mapping update to be transmitted
        if self.cfg.run_mapping:
            while self.mapping_queue.empty():
                pass
            mapping_result = self.mapping_queue.get()
            self.info("Received final mapping update!", logger=log)
            del mapping_result
            torch.cuda.empty_cache()
            gc.collect()
            self.received_mapping.set()

        while self.all_finished < self.num_running_thread:
            pass

        self.info("##########", logger=log)
        end_time = perf_counter()
        self.info("Total time elapsed: {:.2f} minutes".format((end_time - start_time) / 60), logger=log)
        self.info(str(start_time), logger=log)
        self.info(str(end_time), logger=log)
        if (end_time - start_time) > 1e-10:
            self.info("Total FPS: {:.2f}".format(len(stream) / (end_time - start_time)), logger=log)
        self.info("##########", logger=log)
        try:
            import json, os, time

            metrics_path = os.path.join(self.output, "metrics_final.json")
            if os.path.isfile(metrics_path):
                with open(metrics_path, "r") as f:
                    m = json.load(f)

                fps = float(len(stream) / (end_time - start_time)) if (end_time - start_time) > 1e-10 else 0.0
                m["total_fps"] = fps
                m["total_time_min"] = float((end_time - start_time) / 60.0)
                m["slam_start_time"] = float(start_time)
                m["slam_end_time"] = float(end_time)

                tmp = metrics_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(m, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, metrics_path)

                self.info(f"Appended total_fps to {metrics_path}", logger=log)
        except Exception as e:
            self.info(f"WARNING: failed to append total_fps to metrics json: {repr(e)}", logger=log)

        self.terminate(processes, stream)
