#!/bin/bash
# FreeOcc 一键运行脚本
# 用法: bash run_freeocc.sh /path/to/dataset
set -eo pipefail

DATA_DIR="${1:?用法: bash run_freeocc.sh /path/to/dataset}"

source /home/hello/miniconda3/etc/profile.d/conda.sh
conda activate freeocc
FREEOC_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$FREEOC_DIR"

export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_HOME=$CONDA_PREFIX
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 清理上一次 crash 残留的 GPU 显存
pkill -9 -f 'python run.py' 2>/dev/null || true
pkill -9 -f spawn_main 2>/dev/null || true
sleep 2

FREEOC_DIR=$(cd "$(dirname "$0")/.." && pwd)
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_ROOT="$FREEOC_DIR/../freeocc_output/$TIMESTAMP"
OCC_OUTPUT=$OUTPUT_ROOT/occupancy

# 优先用 LiDAR 深度
if [ -d "$DATA_DIR/depth_lidar" ]; then
  if [ -d "$DATA_DIR/depth" ] && [ ! -L "$DATA_DIR/depth" ]; then
    mv "$DATA_DIR/depth" "$DATA_DIR/depth_camera"
  fi
  [ -e "$DATA_DIR/depth" ] && rm -f "$DATA_DIR/depth"
  ln -s "$DATA_DIR/depth_lidar" "$DATA_DIR/depth"
  echo "[INFO] Using LiDAR depth (depth_lidar → depth)"
fi

echo "============================================"
echo "Step 1/2: FreeOcc mapping"
echo "  Data:   $DATA_DIR"
echo "  Output: $OUTPUT_ROOT"
echo "============================================"

python run.py \
  mode=rgbd \
  sync_method=relaxed \
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
  data.png_depth_scale=1000.0 \
  mapping.ov_name_path=./src/scannet_utils/kunlunxai_name0521.txt \
  mapping.store_clip_features=False \
  mapping.loss.supervise_with_prior=False \
  mapping.online_opt.filter.bin_th=0.02 \
  mapping.online_opt.filter.uncertainty=True \
  mapping.online_opt.filter.conf_th=0.05 \
  mapping.enable_occ_eval=False

echo ""
echo "============================================"
echo "Step 2/2: Occupancy + Voxel features"
echo "============================================"

PLY="$OUTPUT_ROOT/mesh/final_rgbd.ply"
mkdir -p "$OCC_OUTPUT"
FEAT_CACHE="$OCC_OUTPUT/voxel_features.pkl"

# 检测单 PLY 还是多 PLY
PLY_LIST=$(ls "$OUTPUT_ROOT/mesh"/final_rgbd_*.ply 2>/dev/null || true)

if [ -n "$PLY_LIST" ]; then
  echo "Found windowed PLYs: $(echo "$PLY_LIST" | wc -l) files"
  python "$FREEOC_DIR/scripts/merge_sessions.py" \
    --plies $PLY_LIST \
    --output "$OCC_OUTPUT/merged_occ.ply" \
    --grid-size 0.1 --thr 0.15 \
    --feat-out "$FEAT_CACHE"
elif [ -f "$PLY" ]; then
  # 单 PLY 模式
  echo "PLY: $PLY"
  python "$FREEOC_DIR/scripts/occ_from_ply.py" \
    --input "$PLY" \
    --output "$OCC_OUTPUT" \
    --grid-size 0.1 \
    --thr 0.15 \
    --max-gaussians 300000 \
    --save-features "$FEAT_CACHE"
else
  echo "ERROR: No PLY found"
  exit 1
fi

echo ""
echo "============================================"
echo "Done!"
echo "============================================"
echo "Output: $OUTPUT_ROOT/"
echo ""
echo "  PLY:         $PLY"
echo "  Occupancy:   $OCC_OUTPUT/"
echo "  Voxel Feats: $FEAT_CACHE"
ls -la "$OCC_OUTPUT/"*.png "$OCC_OUTPUT/"*.ply "$FEAT_CACHE" 2>/dev/null || true
