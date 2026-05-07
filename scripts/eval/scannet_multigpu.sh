#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODE=${MODE:-rgbd}
EXPNAME=${EXPNAME:-embodied_scannet_all}
GPUS_RAW=${1:-${GPUS:-0}}
GPUS_RAW="${GPUS_RAW//,/ }"
read -r -a GPU_LIST <<< "${GPUS_RAW}"
[ "${#GPU_LIST[@]}" -gt 0 ] || die "No GPU specified. Example: $0 0,1,2,3"

DATA_ROOT=${DATA_ROOT:-/data/datasets/slam/scannet200/sequences/test_online}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/FreeOcc/outputs}
EXP_PATH=${EXP_PATH:-${OUTPUT_ROOT}/${EXPNAME}}
SCENE_OCC_ROOT=${SCENE_OCC_ROOT:-/data/datasets/slam/scannet200/scene_occ}
SCENES_FILE=${SCENES_FILE:-${SCRIPT_DIR}/scenes/scannet_16.txt}
MAX_RETRY=${MAX_RETRY:-10}
TIMEOUT_MIN=${TIMEOUT_MIN:-20}
TIMEOUT_SEC=${TIMEOUT_SEC:-$((TIMEOUT_MIN * 60))}
EVAL_DEVICE=${EVAL_DEVICE:-cuda}
RUN_MAPPING_GUI=${RUN_MAPPING_GUI:-False}
SAVE_MESH_EACH_UPDATE=${SAVE_MESH_EACH_UPDATE:-False}
SAVE_MESH_EACH_UPDATE_EVERY=${SAVE_MESH_EACH_UPDATE_EVERY:-1}
H=${H:-968}
W=${W:-1296}

ERR_PATTERNS=(
  "malloc(): invalid size (unsorted)"
  "double free or corruption"
  "corrupted size vs. prev_size"
  "Segmentation fault"
  "CUDA error"
  "RuntimeError:"
  "OutOfMemoryError:"
)

mapfile -t SCENE_LIST < <(load_scenes "${SCENES:-}" "${SCENES_FILE}")
require_nonempty_scene_list "${#SCENE_LIST[@]}" "${SCENES_FILE}"

LOG_DIR=${LOG_DIR:-logs/${EXPNAME}_$(date +%Y%m%d_%H%M%S)_${MODE}}
mkdir -p "${LOG_DIR}"

SUMMARY_CSV="${LOG_DIR}/summary.csv"
EVAL_LOG="${LOG_DIR}/eval_occ_scannet.log"
SUCCESS_SCENES_FILE="${LOG_DIR}/success_scenes.txt"
QUEUE_FILE="${LOG_DIR}/scene_queue.txt"
QUEUE_LOCK="${LOG_DIR}/queue.lock"
SUMMARY_LOCK="${LOG_DIR}/summary.lock"
SUCCESS_LOCK="${LOG_DIR}/success.lock"

echo "scene,status,retry_count,run_dir,log_path,ply_path,gpu" > "${SUMMARY_CSV}"
: > "${SUCCESS_SCENES_FILE}"
: > "${QUEUE_FILE}"
printf '%s\n' "${SCENE_LIST[@]}" > "${QUEUE_FILE}"

build_cmd_array() {
  local scene="$1"
  local scene_dir="$2"
  local fx="$3"
  local fy="$4"
  local cx="$5"
  local cy="$6"

  CMD=(python run.py
    data=scannet/base
    stride=1
    data.input_folder="${scene_dir}"
    mode="${MODE}"
    backend_every=8
    mapper_every=15
    run_mapping_gui="${RUN_MAPPING_GUI}"
    hydra.job.name="${EXPNAME}"
    tracking=scannet
    mapping=scannet
    mapping.save_mesh_each_update="${SAVE_MESH_EACH_UPDATE}"
    mapping.save_mesh_each_update_every="${SAVE_MESH_EACH_UPDATE_EVERY}"
    hydra.run.dir="${EXP_PATH}/${scene}_${MODE}"
  )

  if [ -n "${fx}" ]; then
    CMD+=(data.cam.fx="${fx}" data.cam.fy="${fy}" data.cam.cx="${cx}" data.cam.cy="${cy}" data.cam.H="${H}" data.cam.W="${W}")
  fi
}

append_summary_csv() {
  local line="$1"
  (
    flock -x 200
    echo "${line}" >> "${SUMMARY_CSV}"
  ) 200>"${SUMMARY_LOCK}"
}

append_success_scene() {
  local scene="$1"
  (
    flock -x 200
    echo "${scene}" >> "${SUCCESS_SCENES_FILE}"
  ) 200>"${SUCCESS_LOCK}"
}

pop_next_scene() {
  (
    flock -x 200
    if [ ! -s "${QUEUE_FILE}" ]; then
      echo ""
      exit 0
    fi
    local scene
    IFS= read -r scene < "${QUEUE_FILE}" || true
    tail -n +2 "${QUEUE_FILE}" > "${QUEUE_FILE}.tmp" 2>/dev/null || true
    mv -f "${QUEUE_FILE}.tmp" "${QUEUE_FILE}"
    echo "${scene}"
  ) 200>"${QUEUE_LOCK}"
}

cleanup_and_exit() {
  local code="${1:-130}"
  echo
  echo "[TRAP] Caught signal. Cleaning up..."

  local gpu state_file sid log_path
  for gpu in "${GPU_LIST[@]}"; do
    state_file="${LOG_DIR}/.state_gpu${gpu}"
    if [ -f "${state_file}" ]; then
      sid=$(awk -F= '$1 == "SID" { print $2; exit }' "${state_file}" 2>/dev/null || true)
      log_path=$(awk -F= '$1 == "LOG" { print $2; exit }' "${state_file}" 2>/dev/null || true)
      kill_session_by_sid "${sid}" "${log_path}"
    fi
  done

  if [ -n "${WORKER_PIDS:-}" ]; then
    kill -TERM ${WORKER_PIDS} 2>/dev/null || true
    sleep 1
    kill -KILL ${WORKER_PIDS} 2>/dev/null || true
  fi

  exit "${code}"
}
trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM

run_attempt() {
  local gpu="$1"
  local scene="$2"
  local attempt="$3"
  local scene_dir="${DATA_ROOT}/${scene}"
  local run_dir="${EXP_PATH}/${scene}_${MODE}"
  local ply_path="${run_dir}/mesh/final_${MODE}.ply"
  local attempt_log="${LOG_DIR}/${scene}.gpu${gpu}.attempt_${attempt}.log"
  local state_file="${LOG_DIR}/.state_gpu${gpu}"
  : > "${attempt_log}"

  local fx="" fy="" cx="" cy=""
  local intr_file="${scene_dir}/intrinsic/intrinsic_color.txt"
  if intrinsics=$(read_intrinsics "${intr_file}" 2>/dev/null); then
    read -r fx fy cx cy <<<"${intrinsics}"
  fi

  build_cmd_array "${scene}" "${scene_dir}" "${fx}" "${fy}" "${cx}" "${cy}"

  echo "[GPU=${gpu}][${scene}] attempt ${attempt}/${MAX_RETRY}"
  echo "[GPU=${gpu}][${scene}] run_dir: ${run_dir}"
  echo "[GPU=${gpu}][${scene}] log    : ${attempt_log}"
  echo "[GPU=${gpu}][${scene}] ply    : ${ply_path}"

  rm -f "${ply_path}" 2>/dev/null || true

  CUDA_VISIBLE_DEVICES="${gpu}" setsid stdbuf -oL -eL timeout -k 15s "${TIMEOUT_SEC}" "${CMD[@]}" >> "${attempt_log}" 2>&1 &
  local run_pid=$!

  local sid=""
  for _ in 1 2 3 4 5; do
    sid=$(ps -o sid= -p "${run_pid}" 2>/dev/null | tr -d ' ')
    [ -n "${sid}" ] && break
    sleep 0.1
  done
  if [ -z "${sid}" ]; then
    echo "[GPU=${gpu}][${scene}] ERROR: failed to get SID for pid=${run_pid}" | tee -a "${attempt_log}"
    wait "${run_pid}" 2>/dev/null || true
    return 201
  fi

  {
    echo "SID=${sid}"
    echo "LOG=${attempt_log}"
    echo "SCENE=${scene}"
    echo "PID=${run_pid}"
  } > "${state_file}"

  local detected=0
  local success=0
  local offset=1
  local line=""

  while true; do
    local total_lines
    total_lines=$(wc -l < "${attempt_log}" 2>/dev/null || echo 0)
    if [ "${total_lines}" -ge "${offset}" ]; then
      while [ "${offset}" -le "${total_lines}" ]; do
        line=$(sed -n "${offset}p" "${attempt_log}" 2>/dev/null)
        offset=$((offset + 1))
        echo "${line}"

        if match_any_error "${line}" "${ERR_PATTERNS[@]}"; then
          echo "[GPU=${gpu}][${scene}] fatal pattern detected, retrying: ${line}" | tee -a "${attempt_log}"
          detected=1
          kill_session_by_sid "${sid}" "${attempt_log}"
          break
        fi
      done
    fi

    [ "${detected}" -eq 1 ] && break

    if [ -f "${ply_path}" ]; then
      echo "[GPU=${gpu}][${scene}] ply found, waiting for stable file size..." | tee -a "${attempt_log}"
      if wait_file_stable "${ply_path}" 3 1 90; then
        local size vertex_count
        size=$(stat -c%s "${ply_path}" 2>/dev/null || echo 0)
        vertex_count=$(ply_header_vertex_count "${ply_path}" || echo -1)
        echo "[GPU=${gpu}][${scene}] ply stable (size=${size}, vertex=${vertex_count}); attempt succeeded." | tee -a "${attempt_log}"
        success=1
        kill_session_by_sid "${sid}" "${attempt_log}"
        break
      fi
    fi

    if ! kill -0 "${run_pid}" 2>/dev/null; then
      break
    fi

    sleep 0.2
  done

  : > "${state_file}"

  if [ "${detected}" -eq 1 ]; then
    wait "${run_pid}" 2>/dev/null || true
    return 123
  fi

  if [ "${success}" -eq 1 ]; then
    wait "${run_pid}" 2>/dev/null || true
    return 0
  fi

  wait "${run_pid}"
  return $?
}

process_one_scene() {
  local gpu="$1"
  local scene="$2"
  local scene_dir="${DATA_ROOT}/${scene}"
  local run_dir="${EXP_PATH}/${scene}_${MODE}"
  local ply_path="${run_dir}/mesh/final_${MODE}.ply"

  echo "=============================="
  echo "[GPU=${gpu}] Running ${scene}"

  if [ -f "${ply_path}" ]; then
    echo "[GPU=${gpu}][${scene}] found existing ply, skip reconstruction."
    append_success_scene "${scene}"
    append_summary_csv "${scene},ok,0,${run_dir},SKIPPED,${ply_path},${gpu}"
    return 0
  fi

  if [ ! -d "${scene_dir}" ]; then
    echo "[GPU=${gpu}] Scene folder not found: ${scene_dir}" >&2
    append_summary_csv "${scene},failed,0,${run_dir},SCENE_NOT_FOUND,,${gpu}"
    return 1
  fi

  local status="failed"
  local retry=0
  local rc=0
  while [ "${retry}" -lt "${MAX_RETRY}" ]; do
    retry=$((retry + 1))
    run_attempt "${gpu}" "${scene}" "${retry}"
    rc=$?
    if [ "${rc}" -eq 0 ]; then
      status="ok"
      break
    fi
    echo "[GPU=${gpu}][${scene}] attempt failed (rc=${rc})."
    [ "${retry}" -lt "${MAX_RETRY}" ] && sleep 2
  done

  if [ "${status}" = "ok" ] && [ -f "${ply_path}" ]; then
    append_success_scene "${scene}"
  else
    status="failed"
  fi

  append_summary_csv "${scene},${status},${retry},${run_dir},${LOG_DIR}/${scene}.gpu${gpu}.attempt_${retry}.log,${ply_path},${gpu}"
  echo "[GPU=${gpu}][${scene}] done. status=${status}. retries=${retry}"
}

gpu_worker() {
  local gpu="$1"
  echo "[WORKER] GPU=${gpu} started."

  local scene
  while true; do
    scene=$(pop_next_scene)
    if [ -z "${scene}" ]; then
      echo "[WORKER] GPU=${gpu} queue empty."
      break
    fi
    process_one_scene "${gpu}" "${scene}"
  done
}

echo "Start ScanNet reconstruction with multi-GPU retries."
echo "DATA_ROOT=${DATA_ROOT}"
echo "EXP_PATH=${EXP_PATH}"
echo "GPUS=${GPU_LIST[*]}"
echo "LOG_DIR=${LOG_DIR}"
echo "RUN_MAPPING_GUI=${RUN_MAPPING_GUI}"
echo "SAVE_MESH_EACH_UPDATE=${SAVE_MESH_EACH_UPDATE}"

WORKER_PIDS=""
for gpu in "${GPU_LIST[@]}"; do
  gpu_worker "${gpu}" &
  WORKER_PIDS="${WORKER_PIDS} $!"
done

wait ${WORKER_PIDS}

echo "Stage-1 done. Summary: ${SUMMARY_CSV}"

SCENES_EVAL_STR=$(awk '!seen[$0]++' "${SUCCESS_SCENES_FILE}" | xargs || true)
echo "Start Stage-2 OCC evaluation."
echo "scene_occ_root=${SCENE_OCC_ROOT}"
echo "scenes=${SCENES_EVAL_STR}"

if [ -z "${SCENES_EVAL_STR}" ]; then
  echo "[Stage-2] No successful scenes, skip offline evaluation." | tee "${EVAL_LOG}"
else
  PYTHONPATH=".:${PYTHONPATH:-}" \
  python scripts/src/eval_occ_scannet.py \
    --exp_path "${EXP_PATH}" \
    --scene_occ_root "${SCENE_OCC_ROOT}" \
    --mode "${MODE}" \
    --scenes "${SCENES_EVAL_STR}" \
    --device "${EVAL_DEVICE}" \
    --dump_npz |& tee "${EVAL_LOG}"
fi
