#!/usr/bin/env bash

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

repo_root_from_script() {
  local script_dir="$1"
  cd "${script_dir}/../.." && pwd
}

parse_scenes_from_stdin() {
  awk '
    {
      sub(/#.*/, "")
      gsub(/,/, " ")
      for (i = 1; i <= NF; i++) {
        if ($i != "") {
          print $i
        }
      }
    }
  ' | awk '!seen[$0]++'
}

load_scenes() {
  local scenes_value="${1:-}"
  local scenes_file="${2:-}"

  if [ -n "${scenes_value}" ]; then
    printf '%s\n' "${scenes_value}" | parse_scenes_from_stdin
    return 0
  fi

  if [ -z "${scenes_file}" ]; then
    die "Set SCENES or SCENES_FILE."
  fi
  if [ ! -f "${scenes_file}" ]; then
    die "Scene list not found: ${scenes_file}"
  fi

  parse_scenes_from_stdin < "${scenes_file}"
}

join_scenes() {
  printf '%s\n' "$@" | awk 'NF { printf "%s%s", sep, $0; sep=" " } END { print "" }'
}

require_nonempty_scene_list() {
  local count="$1"
  local source_desc="$2"
  if [ "${count}" -eq 0 ]; then
    die "No scenes resolved from ${source_desc}."
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [ ! -d "${path}" ]; then
    die "${label} not found: ${path}"
  fi
}

read_intrinsics() {
  local intr_file="$1"
  if [ ! -f "${intr_file}" ]; then
    return 1
  fi

  python - "${intr_file}" <<'PY'
import sys
import numpy as np

K = np.loadtxt(sys.argv[1])
print(K[0, 0], K[1, 1], K[0, 2], K[1, 2])
PY
}

read_image_size() {
  local scene_dir="$1"

  python - "${scene_dir}" <<'PY'
import glob
import os
import sys

scene_dir = sys.argv[1]
patterns = [
    os.path.join(scene_dir, "color", "*.jpg"),
    os.path.join(scene_dir, "color", "*.png"),
]
images = []
for pattern in patterns:
    images.extend(sorted(glob.glob(pattern)))
if not images:
    raise SystemExit(f"No color images found under {os.path.join(scene_dir, 'color')}")

try:
    import cv2
    im = cv2.imread(images[0], cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to read image: {images[0]}")
    h, w = im.shape[:2]
except Exception:
    from PIL import Image
    with Image.open(images[0]) as im:
        w, h = im.size

print(h, w)
PY
}

wait_file_stable() {
  local path="$1"
  local stable_need="${2:-3}"
  local sleep_sec="${3:-1}"
  local timeout_sec="${4:-60}"

  local start
  start=$(date +%s)
  local last_size=-1
  local stable_count=0

  while true; do
    if [ -f "${path}" ]; then
      local size
      size=$(stat -c%s "${path}" 2>/dev/null || echo 0)
      if [ "${size}" -gt 0 ] && [ "${size}" -eq "${last_size}" ]; then
        stable_count=$((stable_count + 1))
      else
        stable_count=0
        last_size="${size}"
      fi

      if [ "${stable_count}" -ge "${stable_need}" ]; then
        return 0
      fi
    else
      last_size=-1
      stable_count=0
    fi

    local now
    now=$(date +%s)
    if [ $((now - start)) -ge "${timeout_sec}" ]; then
      return 1
    fi

    sleep "${sleep_sec}"
  done
}

ply_header_vertex_count() {
  local path="$1"

  python - "${path}" <<'PY' 2>/dev/null || true
import re
import sys

pattern = re.compile(r"^element\s+vertex\s+(\d+)\s*$")
with open(sys.argv[1], "rb") as f:
    for _ in range(200):
        line = f.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="ignore").strip()
        match = pattern.match(text)
        if match:
            print(match.group(1))
            raise SystemExit(0)
        if text == "end_header":
            break
print(-1)
PY
}

kill_session_by_sid() {
  local sid="$1"
  local log_path="${2:-}"

  if [ -z "${sid}" ]; then
    return 0
  fi

  if [ -n "${log_path}" ]; then
    echo "[KILL] TERM sid=${sid}" | tee -a "${log_path}"
  else
    echo "[KILL] TERM sid=${sid}"
  fi
  pkill -TERM -s "${sid}" 2>/dev/null || true
  sleep 1

  if [ -n "${log_path}" ]; then
    echo "[KILL] KILL sid=${sid}" | tee -a "${log_path}"
  else
    echo "[KILL] KILL sid=${sid}"
  fi
  pkill -KILL -s "${sid}" 2>/dev/null || true
}

match_any_error() {
  local line="$1"
  shift

  local pattern
  for pattern in "$@"; do
    [[ "${line}" == *"${pattern}"* ]] && return 0
  done
  return 1
}
