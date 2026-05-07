#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODE=${1:-${MODE:-rgbd}}
EXPNAME=${2:-${EXPNAME:-ours_visualization}}
DATA_ROOT=${DATA_ROOT:-/data/datasets/slam/Replica_OCC/Replica_OCC/sequences}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/FreeOcc/outputs}
EXP_PATH=${EXP_PATH:-${OUTPUT_ROOT}/${EXPNAME}}
SCENE_OCC_ROOT=${SCENE_OCC_ROOT:-/data/datasets/slam/Replica_OCC/Replica_OCC}
SCENES_FILE=${SCENES_FILE:-${SCRIPT_DIR}/scenes/replica_all.txt}
RUN_OCC_EVAL=${RUN_OCC_EVAL:-True}
EVAL_DEVICE=${EVAL_DEVICE:-${DEVICE:-cuda}}
RUN_MAPPING_GUI=${RUN_MAPPING_GUI:-False}
SAVE_MESH_EACH_UPDATE=${SAVE_MESH_EACH_UPDATE:-False}
SAVE_MESH_EACH_UPDATE_EVERY=${SAVE_MESH_EACH_UPDATE_EVERY:-1}
TRACKING_CONFIG=${TRACKING_CONFIG:-replica}
MAPPING_CONFIG=${MAPPING_CONFIG:-replica}
H=${H:-680}
W=${W:-1200}

mapfile -t SCENE_LIST < <(load_scenes "${SCENES:-}" "${SCENES_FILE}")
require_nonempty_scene_list "${#SCENE_LIST[@]}" "${SCENES_FILE}"

echo "Start Replica reconstruction."
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "EXP_PATH=${EXP_PATH}"
echo "MODE=${MODE}"
echo "EXPNAME=${EXPNAME}"
echo "SCENE_OCC_ROOT=${SCENE_OCC_ROOT}"
echo "RUN_OCC_EVAL=${RUN_OCC_EVAL}"
echo "EVAL_DEVICE=${EVAL_DEVICE}"
echo "RUN_MAPPING_GUI=${RUN_MAPPING_GUI}"
echo "SAVE_MESH_EACH_UPDATE=${SAVE_MESH_EACH_UPDATE}"

mkdir -p "${EXP_PATH}"
SUCCESS_SCENES=()

for sc in "${SCENE_LIST[@]}"; do
  echo "=============================="
  echo "Running ${sc}"

  SCENE_DIR="${DATA_ROOT}/${sc}"
  RUN_DIR="${EXP_PATH}/${sc}_${MODE}"
  PLY_PATH="${RUN_DIR}/mesh/final_${MODE}.ply"

  if [ -f "${PLY_PATH}" ]; then
    echo "found existing ply, skip reconstruction: ${PLY_PATH}"
    SUCCESS_SCENES+=("${sc}")
    continue
  fi

  require_dir "${SCENE_DIR}" "Scene folder"

  INTR_FILE="${SCENE_DIR}/intrinsic/intrinsic_color.txt"
  FX="" FY="" CX="" CY=""
  if intrinsics=$(read_intrinsics "${INTR_FILE}" 2>/dev/null); then
    read -r FX FY CX CY <<<"${intrinsics}"
    echo "intrinsics: fx=${FX}, fy=${FY}, cx=${CX}, cy=${CY}"
  else
    echo "warning: intrinsic file not found, using config defaults: ${INTR_FILE}"
  fi

  CMD=(python run.py
    data=replica/base
    stride=1
    data.input_folder="${SCENE_DIR}"
    mode="${MODE}"
    backend_every=8
    mapper_every=15
    run_mapping_gui="${RUN_MAPPING_GUI}"
    hydra.job.name="${EXPNAME}"
    tracking="${TRACKING_CONFIG}"
    mapping="${MAPPING_CONFIG}"
    mapping.save_mesh_each_update="${SAVE_MESH_EACH_UPDATE}"
    mapping.save_mesh_each_update_every="${SAVE_MESH_EACH_UPDATE_EVERY}"
    hydra.run.dir="${RUN_DIR}"
  )

  if [ -n "${FX}" ]; then
    CMD+=(data.cam.fx="${FX}" data.cam.fy="${FY}" data.cam.cx="${CX}" data.cam.cy="${CY}" data.cam.H="${H}" data.cam.W="${W}")
  fi

  "${CMD[@]}"
  if [ -f "${PLY_PATH}" ]; then
    SUCCESS_SCENES+=("${sc}")
    echo "${sc} done. ply: ${PLY_PATH}"
  else
    echo "warning: ${sc} finished but ply was not found: ${PLY_PATH}"
  fi
done

case "${RUN_OCC_EVAL}" in
  True|true|1|yes|YES|on|ON)
    ;;
  *)
    echo "RUN_OCC_EVAL=${RUN_OCC_EVAL}; skip Replica OCC evaluation."
    exit 0
    ;;
esac

if [ "${#SUCCESS_SCENES[@]}" -eq 0 ]; then
  echo "No reconstructed or existing ply found; skip Replica OCC evaluation."
  exit 0
fi

SCENES_EVAL_STR=$(join_scenes "${SUCCESS_SCENES[@]}")
EVAL_LOG=${EVAL_LOG:-${EXP_PATH}/eval_occ_replica_${MODE}.log}

echo "Start Replica OCC evaluation."
echo "scene_occ_root=${SCENE_OCC_ROOT}"
echo "scenes=${SCENES_EVAL_STR}"
echo "eval_log=${EVAL_LOG}"

PYTHONPATH=".:${PYTHONPATH:-}" \
python scripts/src/eval_occ_replica.py \
  --exp_path "${EXP_PATH}" \
  --scene_occ_root "${SCENE_OCC_ROOT}" \
  --mode "${MODE}" \
  --scenes "${SCENES_EVAL_STR}" \
  --device "${EVAL_DEVICE}" \
  --dump_npz |& tee "${EVAL_LOG}"
