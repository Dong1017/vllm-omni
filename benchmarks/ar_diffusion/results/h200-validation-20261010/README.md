# Independent single-node H200 validation

Run on October 10, 2026 against native dependency
`86490babe358740cf98f019d1972f3b364a329d4` plus benchmark draft head
`ef5f8c8c54c42a3d84deb51e791b7fa4ce928b1f`.
These are our own measurements, separate from the archived H800 contributor
results. The reproduction applies the benchmark commits to the pinned dependency;
it does not run against `main` alone or the later review-fix branch.

## Result

| Variant | Median DiT FPS | Samples | Relative to baseline |
| --- | ---: | --- | ---: |
| Native baseline | 120.589275 | 120.320924 / 120.661543 / 120.589275 | 1.0000x |
| Hybrid IPC + conditioning cache + exact BF16 fusion | 133.678281 | 133.659401 / 134.708570 / 133.678281 | 1.1085x (+10.85%) |

One node, five H200 GPUs, S=5, G=1, K=30, T=4 plus clean, history=6,
BF16, eager, q=3, latent chunk `[1,16,3,60,104]`, 832x480.
Each variant ran one full warmup and three measured 128-chunk requests.
FPS is `384000 / (gpu_ms[95] - gpu_ms[63])`, excluding T5 encoding, VAE,
video encoding and the serving frontend. The complete latent has 384 frames
and would decode to 1533 pixel frames; pixel-frame equivalents in the timing
window are the throughput denominator.

All eight full outputs have SHA256
`ca3e3a7c429f24ed69b62cd7a4423b10baf01ed22f965a3fcaa40f7ac0201b6a`.
An independent audit recomputed every FPS and median from completion events
and checked all 14 captured source files. This hash differs from the archived
H800 environment; no cross-environment pixel identity is claimed.

KV reservation per rank rises from 9,488,793,600 bytes (8.84 GiB, 11 positions)
to 30,191,624,400 bytes (28.12 GiB, 35 positions, including tickets).
This is a throughput gain with a larger KV allocation.
No two-node execution, communication-free VS, video quality evaluation,
concurrent serving or E2E speedup is established by this run.

## Reproduce

Use the combined checkout described in the parent README and expand its
conditioning fixture (SHA256
`198358abb9eb6e80296d18bb78f82924a4fef1fa1a44ff26af34e75cfda9c5af`).
The model was `/data/models/waveserve-wan2.1-1.3b-diffusers-rf-dev`;
`model-sha256.txt` records its transformer configuration and weights.
`environment.txt` records torch 2.13.0+cu130, vLLM 0.31.0,
diffusers 0.40.0, Transformers, Triton and CUDA bindings.
`hardware.json` records device UUIDs, SM 9.0, 132 SMs and memory;
the driver's reported L20X name is a known host label error.

Request five GPUs through the host's scheduler and let it set
`CUDA_VISIBLE_DEVICES`. From the checkout root, with absolute paths:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -m torch.distributed.run --standalone --nproc-per-node=5 \
  -m benchmarks.ar_diffusion.run_multinode_wan \
  --model "$OMNI_MODEL" --condition "$OMNI_CONDITION" --out "$OMNI_OUTPUT" \
  --groups 1 --steps 4 --chunks 128 --skip 64 --measure 32 \
  --variants baseline hybrid_fused --warmup 1 --repeat 3
```

The first baseline establishes the environment's reference hash, and every
later variant/repeat must match it. A 4-chunk smoke with skip1/measure2,
warmup0/repeat1 also completed for all four variants with identical outputs;
its cold timings are not used for performance claims.
Targeted checks passed: 23 CPU contracts and 3 CUDA BF16 kernel comparisons.
Import ordering and the invalid `--run-level=L1/L2` documentation were corrected
after these measurements. The import-adjusted harness is separately checked
with another real five-GPU smoke.

## Evidence

- `summary.json`: medians, samples, latent/conditioning hashes and per-rank memory.
- `audit.json`: independently recomputed results and source checks.
- `evidence.zip`: full raw request events, source snapshots, smoke results,
  test logs, environment, device properties and model hashes.

After expanding `evidence.zip`, independently recompute the result with:

```bash
python audit-pr8728-h200.py full
```

Shared-host measurements are not isolated from other users' work. Full upstream
CI remains a separate gate.
