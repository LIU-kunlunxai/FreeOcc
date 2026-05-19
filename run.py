import shutil
from termcolor import colored
import ipdb
import yaml
import os
import random
import shlex
import time
import urllib.request
import hydra
import logging
from omegaconf import OmegaConf

import numpy as np
import torch
from tqdm import tqdm

from src.slam import SLAM
from src.datasets import get_dataset

"""
Run the SLAM system on a given dataset or on image folder.
You can configure the system using .yaml configs. See docs for reference ...
"""

# A logger for this file
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAM_VIT_B_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_VIT_B_CHECKPOINT = os.path.join(PROJECT_ROOT, "pretrained", "sam_vit_b_01ec64.pth")


def sys_print(msg: str) -> None:
    log.info(colored(msg, "white", "on_grey", attrs=["bold"]))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def ensure_checkpoint(path: str, url: str, name: str) -> None:
    """Download a checkpoint before runtime starts if it is missing."""
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        sys_print(f"{name} checkpoint found: {path}")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.part"
    manual_cmd = "mkdir -p {} && curl -L {} -o {}".format(
        shlex.quote(os.path.dirname(path)),
        shlex.quote(url),
        shlex.quote(path),
    )

    sys_print(f"{name} checkpoint not found: {path}")
    sys_print(f"Downloading {name} checkpoint before SLAM starts.")
    sys_print(f"URL: {url}")
    sys_print(f"Manual command: {manual_cmd}")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    try:
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            start = time.time()
            downloaded = 0

            with open(tmp_path, "wb") as f:
                with tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {name}",
                    dynamic_ncols=True,
                ) as pbar:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))

                        elapsed = max(time.time() - start, 1e-6)
                        speed = downloaded / elapsed
                        if total > 0 and speed > 0:
                            remaining = max(total - downloaded, 0)
                            pbar.set_postfix_str(
                                f"speed={speed / 1024 / 1024:.2f}MB/s ETA={remaining / speed:.0f}s"
                            )
                        else:
                            pbar.set_postfix_str(f"speed={speed / 1024 / 1024:.2f}MB/s")

        if os.path.getsize(tmp_path) == 0:
            raise RuntimeError("Downloaded file is empty")

        os.replace(tmp_path, path)
        sys_print(f"{name} checkpoint ready: {path}")
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"Failed to download {name} checkpoint from {url}. "
            f"Please run manually: {manual_cmd}"
        ) from exc


def ensure_runtime_checkpoints(cfg) -> None:
    """Prepare large runtime checkpoints before SLAM processes start."""
    if bool(cfg.get("run_mapping", False)):
        ensure_checkpoint(SAM_VIT_B_CHECKPOINT, SAM_VIT_B_CHECKPOINT_URL, "SAM ViT-B")


def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".",
        "..",
        ".git*",
        "*pycache*",
        "*build",
        "*ext",
        "*thirdparty",
        "*.fuse*",
        "*_drive_*",
        "*pretrained*",
        "*output*",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.mp4",
        "*.gif",
        "*media*",
        "*.so",
        "*.pyc",
        "*.Python",
        "*.eggs*",
        "*.DS_Store*",
        "*.idea*",
        "*.pth",
        "*__pycache__*",
        "*.ply",
        "*exps*",
    )

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree(".", backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))


def get_in_the_wild_heuristics(ht: int, wd: int, strategy: str = "generic") -> torch.Tensor:
    """We do not have camera intrinsics on in-the-wild data. In order for this to converge, we
    need a good initialize guess. There are two strategies to do this: i) generc ii) Teeds from DeepV2D
    """
    if strategy == "generic":
        fx = fy = (wd + ht) / 2
        cx, cy = wd / 2, ht / 2
    else:
        fx = fy = wd * 1.2
        cx, cy = wd / 2, ht / 2
    return fx, fy, cx, cy


@hydra.main(version_base=None, config_path="./configs/", config_name="slam")
def run_slam(cfg):

    output_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if cfg.get("output_folder") and cfg.output_folder != "/data/FreeOcc/outputs/":
        output_folder = cfg.output_folder
    log.info(OmegaConf.to_yaml(cfg))
    # Save the cfg to yaml file
    with open(os.path.join(output_folder, "config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(cfg), f, default_flow_style=False)

    setup_seed(43)
    torch.multiprocessing.set_start_method("spawn")
    # Save state for reproducibility
    backup_source_code(os.path.join(output_folder, "code"))

    sys_print(f"\n\n** Running {cfg.data.input_folder} in {cfg.mode} mode!!! **\n\n")

    if cfg.data.cam.fx is None or cfg.data.cam.fy is None:
        sys_print("Using generic intrinsics for in-the-wild data")
        cfg.data.cam.fx, cfg.data.cam.fy, cfg.data.cam.cx, cfg.data.cam.cy = get_in_the_wild_heuristics(
            ht=cfg.data.cam.H, wd=cfg.data.cam.W
        )
    dataset = get_dataset(cfg, device=cfg.device)
    ensure_runtime_checkpoints(cfg)
    slam = SLAM(cfg, dataset=dataset, output_folder=output_folder)

    sys_print(f"Running on {len(dataset)} frames")
    slam.run(dataset)
    sys_print("Done!")


if __name__ == "__main__":
    run_slam()
