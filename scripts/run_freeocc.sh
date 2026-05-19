#!/bin/bash
# FreeOcc 一键运行脚本
# 用法: bash run_freeocc.sh /path/to/dataset
set -euo pipefail

DATA_DIR="${1:?用法: bash run_freeocc.sh /path/to/dataset}"

source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate freeocc
cd /root/workspace/FreeOcc

export TORCH_CUDA_ARCH_LIST="8.0"
export CUDA_HOME=/usr/local/cuda-12.4

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_ROOT=/root/workspace/freeocc_output/$TIMESTAMP
OCC_OUTPUT=$OUTPUT_ROOT/occupancy

echo "============================================"
echo "Step 1/2: FreeOcc mono mapping"
echo "  Data:   $DATA_DIR"
echo "  Output: $OUTPUT_ROOT"
echo "============================================"

python run.py \
  mode=mono \
  mono_depth=metric3d-vit_giant2 \
  use_gt_poses=True \
  run_visualization=False \
  run_mapping_gui=False \
  output_folder="$OUTPUT_ROOT" \
  data.input_folder="$DATA_DIR" \
  data.cam.H=480 \
  data.cam.W=640 \
  data.cam.H_out=480 \
  data.cam.W_out=640 \
  data.cam.fx=386.502 \
  data.cam.fy=385.938 \
  data.cam.cx=321.497 \
  data.cam.cy=241.840 \
  mapping.online_opt.filter.bin_th=0.02 \
  mapping.online_opt.filter.uncertainty=True \
  mapping.online_opt.filter.conf_th=0.05 \
  mapping.enable_occ_eval=False

echo ""
echo "============================================"
echo "Step 2/2: Occupancy + Semantic visualization"
echo "============================================"

PLY="$OUTPUT_ROOT/mesh/final_mono.ply"

if [ ! -f "$PLY" ]; then
  echo "ERROR: PLY not found at $PLY"
  exit 1
fi

echo "PLY: $PLY"
echo "Occupancy output: $OCC_OUTPUT"
mkdir -p "$OCC_OUTPUT"

python /root/workspace/FreeOcc/scripts/occ_from_ply.py \
  --input "$PLY" \
  --output "$OCC_OUTPUT" \
  --grid-size 0.1 \
  --thr 0.15 \
  --max-gaussians 300000

echo ""
echo "============================================"
echo "Done!"
echo "============================================"
echo "Output: $OUTPUT_ROOT/"
echo ""
echo "  PLY:       $PLY"
echo "  Occupancy: $OCC_OUTPUT/"
ls -la "$OCC_OUTPUT/"*.png "$OCC_OUTPUT/"*.ply 2>/dev/null || true
