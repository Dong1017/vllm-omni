# MiniMax-H3 on A100 nodes 82 and 83: stage-split smoke test

This experiment uses the already proven Mooncake Store TCP control/data plane.
Node `10.90.67.82` owns the API and encoder TP8; node `10.90.67.83` owns DiT
USP8. The three-stage run additionally places an independent VAE decoder
patch8 on node 82. Both machines use their **local** CUDA devices `0-7`.

This is a 16-physical-GPU *deployment*, not DiT USP16. The diffusion executor
currently supports local `mp`; its `ray` and `external_launcher` backends raise
`NotImplementedError`. Mooncake moves **stage payloads**, not DiT collectives.
Therefore this run can validate issue 2's encoder/DiT separation and issue 3's
decoder/DiT separation, but cannot validate DiT16 or DP2×TP8 encoder replicas.
The latter needs its own topology after the cross-host stage test.

The decoder split is the H200 four-GPU functionally checked change ported to
the repository's vLLM 0.30 main line. The two-host MiniMax-H3 path, connector
payload size, A100 capacity, and end-to-end output remain to be checked here.

## Before starting

On **both** nodes, use the same branch and source installation inside the
existing PyTorch 26.01 / vLLM 0.30 environment. `MODEL` must resolve to the
same complete local MiniMax-H3 checkpoint on both machines, including FL2VA
and Ref2VA for the Ref2VA request. Check `free -h`, `df -h`, `nvidia-smi -L`,
and `python -m pip show vllm vllm-omni mooncake` first. The previously reported
USP8+DLO host-memory footprint was about 650 GB; node 83 must have adequate
available RAM before loading its eight DiT workers.

```bash
git fetch git@github.com:Dong1017/vllm-omni.git experiment/h3-a100-two-node-20260924
git switch -c experiment/h3-a100-two-node-20260924 FETCH_HEAD
python -m pip install -e . --no-deps
export REPO="$PWD"
export MODEL=/path/to/complete/MiniMax-H3
export RUN_DIR=/path/to/new/h3-a100-run-02
```

If this branch is already checked out, run the same `git fetch` followed by
`git merge --ff-only FETCH_HEAD` instead of `git switch`. `RUN_DIR` is local to each node and holds the rendered
YAML and logs. The shell script keeps the model path and all stage arguments
identical across nodes; it replaces only the connector's *local* `host` field.

On node 82, start the same Mooncake Master configuration used by the Qwen2.5
Omni cross-host test. One documented invocation is:

```bash
mkdir -p "$RUN_DIR/mooncake-storage"
mooncake_master \
  --rpc_port=50051 \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080 \
  --metrics_port=9003 \
  --root_fs_dir="$RUN_DIR/mooncake-storage/" \
  --cluster_id=h3-a100-01
```

Keep Master in its own terminal. Its RPC port `50051`, metadata port `8080`,
Omni Master port `36000`, and API port `18091` must be reachable as in the
earlier cross-host test. Add both IPs to `no_proxy`/`NO_PROXY`; `launch.sh`
does this for each stage process. Do not run the two- and three-stage
experiments simultaneously.

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

Once `http://10.90.67.82:18091/health` is ready, issue a two-step functional
request from node 82. Use a new output path for each retry:

```bash
curl --fail-with-body --max-time 14400 -sS \
  http://10.90.67.82:18091/v1/videos/sync \
  -F 'prompt=A quiet night street, with natural ambient sound.' \
  -F 'width=512' -F 'height=512' -F 'fps=24' \
  -F 'num_inference_steps=2' -F 'flow_shift=12' -F 'seed=42' \
  -F 'extra_params={"task":"t2va","duration":4.4,"audio_flow_shift":3.0}' \
  -o "$RUN_DIR/issue2-t2va.mp4"
ffprobe -v error -show_streams "$RUN_DIR/issue2-t2va.mp4" > "$RUN_DIR/issue2-streams.txt"
```

Check both logs for stage registration, encoder TP8 and DiT USP8; check the
video and audio streams. A valid result demonstrates cross-host stage
separation with Qwen3-VL TP8. It does not show that Qwen3-VL can use TP16.

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

Repeat the issue 2 T2VA request with a fresh output name. Confirm that stage
1 logs show no VAE decoder load, stage 2 logs show patch-parallel size 8,
and the returned MP4 contains video and audio. Both stages are separate
processes even though encoder and decoder share node 82's GPUs; record their
peak GPU and host memory before calling the deployment viable.

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
  -F 'width=512' -F 'height=512' -F 'fps=24' \
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
