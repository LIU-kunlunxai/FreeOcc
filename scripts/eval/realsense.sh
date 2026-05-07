#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODE=${1:-${MODE:-rgbd}}
EXPNAME=${2:-${EXPNAME:-realsense_visualization}}
DATA_ROOT=${DATA_ROOT:-/data/datasets/slam/realsense/datasets}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/FreeOcc/outputs}
SCENES_FILE=${SCENES_FILE:-${SCRIPT_DIR}/scenes/realsense_example.txt}
RUN_MAPPING_GUI=${RUN_MAPPING_GUI:-True}
SAVE_MESH_EACH_UPDATE=${SAVE_MESH_EACH_UPDATE:-False}
SAVE_MESH_EACH_UPDATE_EVERY=${SAVE_MESH_EACH_UPDATE_EVERY:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

mapfile -t SCENE_LIST < <(load_scenes "${SCENES:-}" "${SCENES_FILE}")
require_nonempty_scene_list "${#SCENE_LIST[@]}" "${SCENES_FILE}"

echo "Start RealSense reconstruction."
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "MODE=${MODE}"
echo "EXPNAME=${EXPNAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "RUN_MAPPING_GUI=${RUN_MAPPING_GUI}"
echo "SAVE_MESH_EACH_UPDATE=${SAVE_MESH_EACH_UPDATE}"

for sc in "${SCENE_LIST[@]}"; do
  echo "=============================="
  echo "Running ${sc}"

  SCENE_DIR="${DATA_ROOT}/${sc}"
  require_dir "${SCENE_DIR}" "Scene folder"

  INTR_FILE="${SCENE_DIR}/intrinsic/intrinsic_color.txt"
  if ! intrinsics=$(read_intrinsics "${INTR_FILE}" 2>/dev/null); then
    die "Intrinsic file not found or unreadable: ${INTR_FILE}"
  fi
  read -r FX FY CX CY <<<"${intrinsics}"
  echo "intrinsics: fx=${FX}, fy=${FY}, cx=${CX}, cy=${CY}"

  if ! image_size=$(read_image_size "${SCENE_DIR}"); then
    die "Failed to infer image size for ${SCENE_DIR}"
  fi
  read -r H W <<<"${image_size}"
  echo "resolution: H=${H}, W=${W}"

  CMD=(python run.py
    data=realsense/base
    data.input_folder="${SCENE_DIR}"
    mode="${MODE}"
    stride=1
    run_mapping_gui="${RUN_MAPPING_GUI}"
    show_stream=False
    backend_every=8
    mapper_every=15
    hydra.job.name="${EXPNAME}"
    hydra.run.dir="${OUTPUT_ROOT}/${EXPNAME}/${sc}_${MODE}"
    data.cam.H="${H}"
    data.cam.W="${W}"
    data.cam.fx="${FX}"
    data.cam.fy="${FY}"
    data.cam.cx="${CX}"
    data.cam.cy="${CY}"
    data.png_depth_scale=1000.0
    tracking=realsense
    mapping=realsense
    mapping.save_mesh_each_update="${SAVE_MESH_EACH_UPDATE}"
    mapping.save_mesh_each_update_every="${SAVE_MESH_EACH_UPDATE_EVERY}"
    device=cuda:0
  )

  echo "[CMD] ${CMD[*]}"
  "${CMD[@]}"
  echo "${sc} done."
done
