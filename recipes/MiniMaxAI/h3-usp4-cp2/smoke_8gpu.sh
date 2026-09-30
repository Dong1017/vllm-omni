#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
model=${MODEL:?Set MODEL to the complete Ref2VA partition}
reference=${REF_IMAGE:?Set REF_IMAGE to a readable PNG}
run_dir=${RUN_DIR:?Set RUN_DIR to a new output directory}
mkdir -p "$run_dir"

export H3_API_PORT=${H3_API_PORT:-18091}
export OMNI_MASTER_PORT=${OMNI_MASTER_PORT:-36000}
export PYTHON_BIN=${PYTHON_BIN:-python}

git -C "$root/../../.." rev-parse HEAD > "$run_dir/source_head.txt"
git -C "$root/../../.." status --short > "$run_dir/source_status.txt"
cp "$root/8gpu-local.yaml" "$run_dir/deploy.yaml"
free -h > "$run_dir/host_memory_before.txt"
nvidia-smi > "$run_dir/gpu_before.txt"

bash "$root/launch.sh" 8gpu 0 "$model" > "$run_dir/stage0.log" 2>&1 &
p0=$!
sleep 10
bash "$root/launch.sh" 8gpu 1 "$model" > "$run_dir/stage1.log" 2>&1 &
p1=$!
bash "$root/launch.sh" 8gpu 2 "$model" > "$run_dir/stage2.log" 2>&1 &
p2=$!
trap 'kill -TERM "$p0" "$p1" "$p2" 2>/dev/null || true; wait "$p0" "$p1" "$p2" 2>/dev/null || true' EXIT

for attempt in $(seq 1 180); do
  if curl --fail --silent --max-time 2 "http://127.0.0.1:$H3_API_PORT/health" > /dev/null; then
    break
  fi
  if ! kill -0 "$p0" 2>/dev/null || ! kill -0 "$p1" 2>/dev/null || ! kill -0 "$p2" 2>/dev/null; then
    echo "A stage exited before /health; inspect stage0.log, stage1.log, stage2.log" >&2
    exit 1
  fi
  sleep 10
done
curl --fail --silent --max-time 2 "http://127.0.0.1:$H3_API_PORT/health" > "$run_dir/health.txt"
nvidia-smi > "$run_dir/gpu_after_load.txt"

curl --fail-with-body --max-time 14400 -sS \
  -D "$run_dir/headers.txt" \
  -w 'http_code=%{http_code}\nwall_seconds=%{time_total}\n' \
  "http://127.0.0.1:$H3_API_PORT/v1/videos/sync" \
  -F 'prompt=Preserve the person and setting, with natural ambient sound.' \
  -F "input_reference=@${reference};type=image/png" \
  -F 'width=512' -F 'height=512' -F 'aspect_ratio=1:1' -F 'fps=24' \
  -F 'num_inference_steps=2' -F 'flow_shift=12' -F 'seed=42' \
  -F 'extra_params={"task":"ref2va","duration":4.4,"audio_flow_shift":3.0}' \
  -o "$run_dir/ref2va.mp4" > "$run_dir/request.txt"
ffprobe -v error -show_entries format=duration:stream=index,codec_type,width,height,nb_frames \
  -of json "$run_dir/ref2va.mp4" > "$run_dir/output_ffprobe.json"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_type -of csv=p=0 \
  "$run_dir/ref2va.mp4" | grep -qx video
ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of csv=p=0 \
  "$run_dir/ref2va.mp4" | grep -qx audio
nvidia-smi > "$run_dir/gpu_after_request.txt"
