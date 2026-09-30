#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: launch.sh {8gpu|16die} {0|1|2} /path/to/MiniMax-H3/Ref2VA" >&2
  exit 2
fi

topology=$1
stage_id=$2
model=$3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$root/../../.." && pwd)

case "$topology" in
  8gpu) deploy="$root/8gpu-local.yaml" ;;
  16die) deploy="$root/16die-local.yaml" ;;
  *) echo "Unknown topology: $topology" >&2; exit 2 ;;
esac
case "$stage_id" in
  0|1|2) ;;
  *) echo "Stage must be 0, 1, or 2" >&2; exit 2 ;;
esac

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-14400}

python_bin=${PYTHON_BIN:-python}
master_address=${OMNI_MASTER_ADDRESS:-127.0.0.1}
master_port=${OMNI_MASTER_PORT:-36000}
api_port=${H3_API_PORT:-18091}
args=(serve "$model" --omni --trust-remote-code --task-type ref2va
  --deploy-config "$deploy" --stage-id "$stage_id"
  --omni-master-address "$master_address" --omni-master-port "$master_port"
  --stage-init-timeout 1800 --init-timeout 1800)
if [[ "$stage_id" == 0 ]]; then
  args+=(--host 0.0.0.0 --port "$api_port")
else
  args+=(--headless)
fi

exec "$python_bin" -m vllm_omni.entrypoints.cli.main "${args[@]}"
