#!/bin/bash
# FreeOcc 运行脚本 - holodata 数据集
set -euo pipefail

source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate freeocc
cd /root/workspace/FreeOcc

export TORCH_CUDA_ARCH_LIST="8.0"
export CUDA_HOME=/usr/local/cuda-12.4

python run.py \
  mode=rgbd \
  use_gt_poses=True \
  run_visualization=False \
  run_mapping_gui=False \
  output_folder=/root/workspace/freeocc_output \
  data.input_folder=/root/workspace/holodata/bag_20260519_152901_dataset \
  data.cam.H=480 \
  data.cam.W=640 \
  data.cam.H_out=480 \
  data.cam.W_out=640 \
  data.cam.fx=386.502 \
  data.cam.fy=385.938 \
  data.cam.cx=321.497 \
  data.cam.cy=241.840 \
  data.png_depth_scale=1000.0 \
  mapping.online_opt.filter.bin_th=0.03 \
  mapping.loss.supervise_with_prior=False \
  mapping.enable_occ_eval=False
