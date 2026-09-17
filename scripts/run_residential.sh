#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${DREAM_PYTHON:-python3.11}"
asset_dir="${DREAM_ASSET_DIR:-$project_root/.runtime/render_assets}"
model_cache="${DREAM_MODEL_CACHE:-$project_root/.runtime/models}"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$project_root"
exec "$python_bin" -m dream_sim.study \
  --controller "${DREAM_CONTROLLER:-continuous_return}" \
  --task-manifest configs/residential50-easy-grasp/task_manifest.json \
  --asset-dir "$asset_dir" --model-cache "$model_cache" \
  --robot-time-limit-seconds 1800 --wall-timeout-seconds 0 \
  --wait-for-slot-seconds 1200 \
  --parallel "$@"
