#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: submit_8gpu.sh /path/to/Ref2VA /path/to/reference.png /path/to/new-run-dir" >&2
  exit 2
fi

model=$1
reference=$2
run_dir=$3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$run_dir"
test ! -e "$run_dir/queue.pid"
test ! -e "$run_dir/queue.log"

python_bin=${PYTHON_BIN:-python}
nohup gpu run --gpus 8 --timeout 4h --note "MiniMax-H3 USP4 CP2 split preflight" -- \
  env MODEL="$model" REF_IMAGE="$reference" RUN_DIR="$run_dir" \
      PYTHON_BIN="$python_bin" \
  bash "$root/smoke_8gpu.sh" > "$run_dir/queue.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run_dir/queue.pid"
