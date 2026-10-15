# Noisy PP: CausVid Wan DiT benchmark

This opt-in harness runs the native `StageWanTransformer`, cell executor,
Latest KV selection, experimental FlowEuler sampler and paged FlashAttention from
[PR #8282](https://github.com/vllm-project/vllm-omni/pull/8282).
It publishes KV per transformer block using CUDA IPC on the same host and
NCCL between hosts. Native attention, model weights and sampling stay in use.
The fused variant also caches request-local conditioning projections and
preserves eager BF16 rounding in gated residual kernels. Normalization and
modulation use the native AdaLayerNorm, including its CUDA fusion.

These are benchmark-scoped adapters, installed and restored by context
managers. The diffusion frontend does not gain a multinode serving executor.
Each process handles one request at a time; this harness does not measure
concurrent serving or VAE/end-to-end video throughput.

## Reproduce

This benchmark depends on the native cell APIs in PR #8282. Until that PR
lands, apply this draft's benchmark commits to the pinned dependency in a
disposable development checkout. Start from this draft branch:

```bash
benchmark_head=$(git rev-parse HEAD)
git fetch https://github.com/vllm-project/vllm-omni.git refs/pull/8282/head
git switch -c codex/wan-benchmark-reproduce 80d6d53c3a5994dd5b86b157b042bd89d131c447
git cherry-pick "e4af781dc71ffdf6962c469aaa6a8b87ab7908f6..${benchmark_head}"
```

The benchmark and `test_wan_hybrid_kv.py` require that combined checkout;
`main` alone does not yet provide the dependency's native cell APIs.
This pins the reviewed CausVid dependency without merging later `main` changes.
Later dependency revisions need separate compatibility validation.

Use a Linux/CUDA environment compatible with the PR #8282 branch, with vLLM,
PyTorch, paged FlashAttention, Triton, `safetensors` and `cuda.bindings` installed.
The tested environments and raw measurements are linked under
[Recorded validation](#recorded-validation).

`OMNI_MODEL` must point to the complete `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` root,
including `transformer/config.json` and transformer safetensors. Passing only
the transformer directory is unsupported. The model configuration must have
30 blocks, 12 attention heads and head size 128.

Set `CAUSVID_CKPT` to the locally downloaded `tianweiy/CausVid`
`autoregressive_checkpoint/model.pt`. The native remapper overlays the student
after loading the Diffusers DiT shell. Both variants use identical weights.
Runtime module paths and pipeline registration follow the pinned dependency.
The default is T=3 plus one clean stage, shift=8. This uses the dependency's
experimental FlowEuler path; it does not implement official DMD sampling parity.

The compressed fixture contains BF16 T5 embeddings for `a cat walking on grass`,
shape `[1,512,4096]`. It excludes T5 loading and encoding from DiT measurements.
Expand it to an absolute path visible from both nodes:

```bash
gzip -dc benchmarks/ar_diffusion/fixtures/conditioning.pt.gz > /tmp/omni-condition.pt
sha256sum /tmp/omni-condition.pt
# 198358abb9eb6e80296d18bb78f82924a4fef1fa1a44ff26af34e75cfda9c5af
```

For two nodes, set `OMNI_NODE_RANK` to `0` or `1`, `OMNI_MASTER_ADDR` to the
first node's reachable address, `OMNI_CONDITION` to the expanded fixture path,
and `OMNI_OUTPUT` to an empty shared results directory. Run on both nodes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=0 \
python -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=4 --node-rank="$OMNI_NODE_RANK" \
  --master-addr="$OMNI_MASTER_ADDR" --master-port=29879 \
  -m benchmarks.ar_diffusion.benchmark_noisy_pp \
  --model "$OMNI_MODEL" --causvid-ckpt "$CAUSVID_CKPT" \
  --condition "$OMNI_CONDITION" --out "$OMNI_OUTPUT" \
  --groups 2 --steps 3 --shift 8 --chunks 128 --skip 64 --measure 32 \
  --variants baseline hybrid_fused --warmup 1 --repeat 3
```

Adjust the network interface to the actual deployment. Each node uses four
GPUs; the stage-major placement has four stages with two block partitions per
stage (`K=15`). CUDA IPC requires distinct physical GPUs for local ranks.

For one node (`K=30`), use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m benchmarks.ar_diffusion.benchmark_noisy_pp \
  --model "$OMNI_MODEL" --causvid-ckpt "$CAUSVID_CKPT" \
  --condition "$OMNI_CONDITION" --out "$OMNI_OUTPUT" \
  --groups 1 --steps 3 --shift 8 --chunks 128 --skip 64 --measure 32 \
  --variants baseline hybrid_fused --warmup 1 --repeat 3
```

Run from the checkout root and use a distinct output directory for each
topology or source revision. `hybrid` enables the transport and conditioning
cache without pointwise fusion; `cached` enables only the conditioning cache.
Only Latest-KV with one fixed denoise/clean stage per rank is supported.
Each scheduled cell and its attention context refer to one transformer block;
the hybrid ring indexes the native plan without expanding the schedule again.

## Timing and output contract

Each request has 128 chunks of three latent frames at BF16 shape
`[1,16,3,60,104]`, seed zero, history six chunks and no sink. The complete latent
is `[1,16,384,60,104]`; decoding it would produce 1533 frames at 832x480.

Steady FPS is a pixel-frame equivalent of pure DiT throughput. The final
denoise rank records one CUDA completion event per chunk. After 64 skipped
chunks, 32 measured chunks contribute 384 frame equivalents:

```text
FPS = 384000 / (completion_ms[95] - completion_ms[63])
```

The remaining 32 chunks complete the request. Text encoding, model loading,
VAE, hashing and video encoding are excluded. A full request warms each
variant, followed by three full measured requests; the report uses medians.
The harness retains all latents, checks finiteness and hashes the complete
output after timing. All variants and repeats must match the supplied hash
or the first request's hash. Latest KV semantics depend on `K`, so hashes are
compared within a topology rather than across `K=15` and `K=30`.

The default hybrid ring has `(history+1) * stages = 28` version positions per layer;
it owns the only KV pool in hybrid runs. The baseline allocates its native
version pool separately. `kv_capacity` reports version positions per layer
for the selected pool, identified by `kv_pool`; `kv_reserved_bytes` includes
the hybrid ticket storage when applicable.
Buffers use producer-ready and consumer-release tickets to prevent a reused
ring position from overwriting KV that attention still reads. Failed IPC
requests keep mappings alive until the worker job exits.

Source snapshots, per-request events and hashes, GPU memory statistics and
median summaries are written under the output directory. Per-block transport
rounds and read labels derive directly from the native `CellPlan`; no second
scheduler or general KV manager is included. Original transport
source attribution and adaptation hashes are listed in
[provenance.json](provenance.json).

## Tests

CPU tests cover native KV labels, last-reader release identities, physical
pool addresses, allocation boundaries, exported-buffer lifetime on failure,
cache expiration and restoration of modules after normal and failed requests.
CUDA tests compare fused kernels against separate eager BF16 operations,
including strided residual tensors. They do not require model weights:

```bash
python -m pytest -o addopts='' \
  tests/diffusion/ar_diffusion/test_wan_hybrid_kv.py \
  tests/diffusion/ar_diffusion/test_wan_native_optimizations.py \
  -m 'core_model and cpu' --run-level=core_model -q
python -m pytest -o addopts='' \
  tests/diffusion/ar_diffusion/test_wan_native_pointwise.py \
  -m 'core_model and cuda' --run-level=core_model -q
```

The repository-wide test plugins also require a vLLM version compatible with
the selected Omni branch. For isolated unit/kernel checks when those unrelated
plugins cannot load, add `--noconftest` and omit `--run-level`.

## Recorded validation

### CausVid/cell validation (October 15)

On four H200 GPUs with dependency `80d6d53`, BF16, T3/S4/G1, shift8, q3/H6,
128 chunks and skip64/measure32, one warmup plus three measured requests gave:

| Variant | Median DiT FPS | KV reservation per rank |
| --- | ---: | ---: |
| Native baseline | 129.320568 | 8.03 GiB |
| `hybrid_fused` | 138.301577 (+6.94%) | 22.49 GiB |

All eight complete latent hashes matched. The 23 CPU contracts and 3 CUDA
kernel checks passed. Eight-chunk smokes also passed for all four variants
at T3/G1 and for baseline/hybrid_fused at T1/G2 (four GPUs in both cases).
The fused block now reuses native AdaLayerNorm; bypassing its CUDA path had
caused a hash mismatch. This is single-node DiT validation, not E2E or DMD parity.
Raw events, source snapshots and the independent timing/hash audit are under
`/data/dxw/outputs/pr8728-causvid-cell-20261015/results/` on the validation host.

### Historical RF validation (October 10)

The following measurements used the earlier RF checkpoint, T=4, shift=5 and
chunk APIs, rather than the current CausVid benchmark.

Independent five-H200 validation measured baseline **120.589275 DiT FPS** and
`hybrid_fused` **133.678281 DiT FPS** (+10.85%), with matching full latent hashes
across one warmup and three measured requests per variant. The four-variant
smoke, 23 CPU contracts and 3 CUDA kernel checks also passed. Performance is
bound to dependency `86490bab` plus benchmark `ef5f8c8c`; the later import fixes
were checked with another five-GPU smoke. These are single-node DiT results.

The complete evidence and audit scripts are preserved at the fixed archive
commit `72d0a1a0ae92157d3aa1d5156eadad5dfb2696fe`, outside the active benchmark:

- [Independent H200 report and raw evidence](https://github.com/Dong1017/vllm-omni/tree/72d0a1a0ae92157d3aa1d5156eadad5dfb2696fe/benchmarks/ar_diffusion/results/h200-validation-20261010)
- [Original contributor's H800 report and raw evidence](https://github.com/Dong1017/vllm-omni/tree/72d0a1a0ae92157d3aa1d5156eadad5dfb2696fe/benchmarks/ar_diffusion/results/h800-q3-20261009)
- [H800 archive audit script](https://github.com/Dong1017/vllm-omni/blob/72d0a1a0ae92157d3aa1d5156eadad5dfb2696fe/benchmarks/ar_diffusion/audit_results.py)

The first baseline request establishes a reference hash, which every variant
and repeat must match. Record that hash and optionally pass `--expected-sha`
for subsequent runs in the same environment. Archived H800 hashes are not
cross-environment references.
