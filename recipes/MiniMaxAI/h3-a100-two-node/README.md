# MiniMax-H3 on A100 nodes 82 and 83: stage-split smoke test

This experiment uses the already proven Mooncake Store TCP control/data plane.
Node `10.90.67.82` owns the API and encoder TP8; node `10.90.67.83` owns DiT
USP8. The three-stage run additionally places an independent VAE decoder
patch8 on node 82 by default, or on node 83. Both machines use their **local**
CUDA devices `0-7`.
The two-stage YAML inherits the repository's
`vllm_omni/deploy/minimax_h3_disaggregated.yaml`; the three-stage YAML inherits
the decoder-split YAML added by this branch. The overlays change placement,
transport, memory limits, and two-step smoke-test settings. The repository also has a NIXL overlay,
but these runs use Mooncake Store because it is the transport already tested on
nodes 82 and 83.

These are 16-physical-GPU *deployments*, not DiT USP16. The diffusion executor
currently supports local `mp`; its `ray` and `external_launcher` backends raise
`NotImplementedError`. Mooncake moves **stage payloads**, not DiT collectives.
Run `02` to validate encoder/DiT separation, `03` for decoder/DiT separation,
and `02dp` for two TP8 encoder replicas with a separate DiT USP8 stage. `02dp`
reuses the `02` YAML; no second copy is needed. None of these runs validates
DiT16.

The decoder split is the H200 four-GPU functionally checked change ported to
the repository's vLLM 0.30 main line. Check A100 capacity and end-to-end output
for each two-host topology independently.

## Before starting

On **both** nodes, use the same branch and source installation inside the
existing PyTorch 26.01 / vLLM 0.30 environment. `MODEL` must point to the same
complete task partition on both machines. The requests below use `Ref2VA`, so
`FL2VA` is not required. Keep the local native-checkpoint path workaround that
passed `02` if the repository root has `modular_model_index.json` but no
`fastvideo_inference.json`. Check `free -h`, `df -h`, `nvidia-smi -L`,
and `python -m pip show vllm vllm-omni mooncake` first. The previously reported
USP8+DLO host-memory footprint was about 650 GB; node 83 must have adequate
available RAM before loading its eight DiT workers.

```bash
git fetch git@github.com:Dong1017/vllm-omni.git experiment/h3-a100-two-node-20260924
git switch -c experiment/h3-a100-two-node-20260924 FETCH_HEAD
python -m pip install -e . --no-deps
export REPO="$PWD"
export MODEL=/data/models/MiniMax-H3/Ref2VA
export RUN_DIR=/path/to/new/h3-a100-run-02
```

If this branch is already checked out, run the same `git fetch` followed by
`git merge --ff-only FETCH_HEAD` instead of `git switch`. `RUN_DIR` is local to
each node and holds the rendered YAML and logs. The shell script keeps the
selected model partition and parallel YAML consistent across nodes, and sets
the connector's *local* `host` field for each process.

On node 82, start the Mooncake Master without `--root_fs_dir` for this TCP-only
experiment. That option requires the same shared filesystem on both nodes.

```bash
mooncake_master \
  --rpc_port=50051 \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080 \
  --metrics_port=9003 \
  --cluster_id=h3-a100-01
```

Keep Master in its own terminal. Its RPC port `50051`, metadata port `8080`,
Omni Master port `36000`, and API port `18091` must be reachable as in the
earlier cross-host test. Add both IPs to `no_proxy`/`NO_PROXY`; `launch.sh`
does this for each stage process. Do not run `02`, `02dp`, and `03`
simultaneously.

## Issue 2: encoder 8 + DiT 8 across the two hosts

In separate terminals, start stage 0 on node 82 and stage 1 on node 83.
Stage 0 starts the Omni Master and waits for stage 1 registration. Its wait
limit is 1800 seconds.

```bash
# 82
NODE_IP=10.90.67.82 bash "$REPO/recipes/MiniMaxAI/h3-a100-two-node/launch.sh" 02 0 "$MODEL" \
  2>&1 | tee "$RUN_DIR/stage0.log"
```

```bash
# 83; set REPO, MODEL and RUN_DIR for this node first
NODE_IP=10.90.67.83 bash "$REPO/recipes/MiniMaxAI/h3-a100-two-node/launch.sh" 02 1 "$MODEL" \
  2>&1 | tee "$RUN_DIR/stage1.log"
```

Once `http://10.90.67.82:18091/health` is ready, issue a two-step Ref2VA
request from node 82. Use a real PNG reference image and a new output path for
each retry:

```bash
export REF_IMAGE=/path/to/reference.png
curl --fail-with-body --max-time 14400 -sS \
  http://10.90.67.82:18091/v1/videos/sync \
  -F 'prompt=Preserve the person and setting, with natural ambient sound.' \
  -F "input_reference=@${REF_IMAGE};type=image/png" \
  -F 'width=512' -F 'height=512' -F 'aspect_ratio=1:1' -F 'fps=24' \
  -F 'num_inference_steps=2' -F 'flow_shift=12' -F 'seed=42' \
  -F 'extra_params={"task":"ref2va","duration":4.4,"audio_flow_shift":3.0}' \
  -o "$RUN_DIR/issue2-ref2va.mp4"
ffprobe -v error -show_streams "$RUN_DIR/issue2-ref2va.mp4" > "$RUN_DIR/issue2-streams.txt"
```

Check both logs for stage registration, encoder TP8 and DiT USP8; check the
video and audio streams. A valid result demonstrates cross-host stage
separation with Qwen3-VL TP8. It does not show that Qwen3-VL can use TP16.

## Issue 2 follow-up: encoder DP2 × TP8 with DiT USP8

Run this separately from `02` and `03`, using a fresh `RUN_DIR` on both nodes,
for example `export RUN_DIR=/home/d00949542/h3-a100-run-02dp` in every terminal.
`02dp` reuses `02-encoder8-dit8.yaml.in`. Node 82 runs the API and encoder
replica 0 (TP8); node 83 runs a second, headless encoder replica (TP8) and
DiT (USP8). Each process launches one local replica; the head dynamically
attaches the second stage-0 replica and uses round-robin routing. The encoder
and DiT on node 83 **share GPUs 0-7**, so check its free GPU and host memory
before loading both. This verifies encoder DP2 × TP8 alongside DiT USP8; it
does not test DiT USP16 or a throughput gain from the single DiT stage.

Keep the working Mooncake Master. In separate terminals, start:

```bash
# 82: API and encoder replica 0
NODE_IP=10.90.67.82 bash "$REPO/recipes/MiniMaxAI/h3-a100-two-node/launch.sh" 02dp 0 "$MODEL" \
  2>&1 | tee "$RUN_DIR/stage0-82.log"
```

```bash
# 83: DiT
NODE_IP=10.90.67.83 bash "$REPO/recipes/MiniMaxAI/h3-a100-two-node/launch.sh" 02dp 1 "$MODEL" \
  2>&1 | tee "$RUN_DIR/stage1-83.log"
```

```bash
# 83: encoder replica 1; same YAML, but headless
NODE_IP=10.90.67.83 bash "$REPO/recipes/MiniMaxAI/h3-a100-two-node/launch.sh" 02dp 0 "$MODEL" \
  2>&1 | tee "$RUN_DIR/stage0-83.log"
```

Wait for the 82 log to report `Stage 0 replica 1 registered` and
`attached remote replica stage=0 replica=1`, and the 83 log to report
`Stage 0 replica id=1 up`. Then send two Ref2VA requests with different seeds
from node 82. Round-robin should send one to each encoder replica:

```bash
export REF_IMAGE=/path/to/reference.png
send_dp_request() {
  local seed=$1
  curl --fail-with-body --max-time 14400 -sS \
    http://10.90.67.82:18091/v1/videos/sync \
    -F 'prompt=Preserve the person and setting, with natural ambient sound.' \
    -F "input_reference=@${REF_IMAGE};type=image/png" \
    -F 'width=512' -F 'height=512' -F 'aspect_ratio=1:1' -F 'fps=24' \
    -F 'num_inference_steps=2' -F 'flow_shift=12' -F "seed=$seed" \
    -F 'extra_params={"task":"ref2va","duration":4.4,"audio_flow_shift":3.0}' \
    -o "$RUN_DIR/issue2-dp-seed${seed}.mp4"
}
send_dp_request 42 > "$RUN_DIR/seed42-curl.log" 2>&1 & p42=$!
send_dp_request 43 > "$RUN_DIR/seed43-curl.log" 2>&1 & p43=$!
wait "$p42"; rc42=$?
wait "$p43"; rc43=$?
test "$rc42" -eq 0 && test "$rc43" -eq 0
for seed in 42 43; do
  ffprobe -v error -show_streams "$RUN_DIR/issue2-dp-seed${seed}.mp4" > "$RUN_DIR/issue2-dp-seed${seed}-streams.txt"
done
```

Keep the 82 and 83 encoder logs plus both GPU-memory traces. Two registered
replicas and two valid outputs establish the DP2 × TP8 deployment; confirm
request IDs reached both encoder processes before claiming both served traffic.

## Issue 3: separate VAE decoder 8

Stop the issue 2 processes after saving their logs; keep Mooncake Master.
Set a **new** `RUN_DIR` on each node. Start stage 0 on 82, stage 1 on 83, and
stage 2 on 82 in three terminals:

```bash
# 82: API / encoder
NODE_IP=10.90.67.82 bash "$REPO/recipes/MiniMaxAI/h3-a100-two-node/launch.sh" 03 0 "$MODEL" \
  2>&1 | tee "$RUN_DIR/stage0.log"
```

```bash
# 83: DiT, without VAE decoder
NODE_IP=10.90.67.83 bash "$REPO/recipes/MiniMaxAI/h3-a100-two-node/launch.sh" 03 1 "$MODEL" \
  2>&1 | tee "$RUN_DIR/stage1.log"
```

```bash
# 82: independent VAE video patch8 + audio decoder
NODE_IP=10.90.67.82 bash "$REPO/recipes/MiniMaxAI/h3-a100-two-node/launch.sh" 03 2 "$MODEL" \
  2>&1 | tee "$RUN_DIR/stage2.log"
```

Stage 2 may instead run on node 83: change only `NODE_IP` to `10.90.67.83`
and launch it there. The YAML still selects local GPUs 0-7. On 83 it shares
those cards with DiT, so record GPU and host memory for both stages. This is
process separation, regardless of which node hosts stage 2.

Repeat the issue 2 Ref2VA request with a fresh output name. Confirm that stage
1 logs show no VAE decoder load, stage 2 logs show patch-parallel size 8,
the logs do not say `decoding rank-locally for this shape`, and the returned
MP4 contains video and audio. Record peak GPU and host memory before calling
the deployment viable.

## Issue 5: Ref2VA host gap, same three-stage deployment

Use the issue 3 server and a real reference image accessible to the API node.
The request below checks end-to-end Ref2VA and preserves the wall time. It
does **not** by itself attribute host idle time; use the existing H200 profiler
procedure for a CPU/GPU timeline if the gap remains.

```bash
export REF_IMAGE=/path/to/reference.png
/usr/bin/time -p curl --fail-with-body --max-time 14400 -sS \
  http://10.90.67.82:18091/v1/videos/sync \
  -F 'prompt=Preserve the person and setting, with natural ambient sound.' \
  -F 'width=512' -F 'height=512' -F 'aspect_ratio=1:1' -F 'fps=24' \
  -F 'num_inference_steps=2' -F 'flow_shift=12' -F 'seed=42' \
  -F 'extra_params={"task":"ref2va","duration":4.4,"audio_flow_shift":3.0}' \
  -F "input_reference=@${REF_IMAGE};type=image/png" \
  -o "$RUN_DIR/issue5-ref2va.mp4" \
  2> "$RUN_DIR/issue5-wall-time.txt"
ffprobe -v error -show_streams "$RUN_DIR/issue5-ref2va.mp4" > "$RUN_DIR/issue5-streams.txt"
```

Keep both nodes' stage logs, rendered `deploy-stage*.yaml`, `free -h` before
and after load, GPU memory snapshots, MP4 output, and exact `git rev-parse
HEAD`. Distinguish model load failures, connector errors, DiT failures, and
shutdown errors when reading the logs.
