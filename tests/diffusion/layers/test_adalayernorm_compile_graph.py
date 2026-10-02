# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compile and CUDA-graph smoke tests for the fused AdaLayerNorm fast path.

The serving stack reaches AdaLayerNorm.forward_cuda through CustomOp dispatch
from compiled diffusion workloads (lazy regional torch.compile on repeated
blocks) and can capture CUDA graphs around pipeline stages. These smokes
verify that the supported fused path can be compiled under torch.compile and
captured/replayed inside a CUDA graph without crashing and with outputs
matching the native fallback. They are smoke tests, not performance
measurements.
"""

import pytest
import torch

from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _make(bs=2, seq=512, hidden=3072, dtype=torch.bfloat16, seed=0):
    m = AdaLayerNorm(hidden).to(device="cuda", dtype=dtype)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(bs, seq, hidden, generator=g, device="cuda", dtype=dtype)
    scale = torch.randn(bs, 1, hidden, generator=g, device="cuda", dtype=dtype)
    shift = torch.randn(bs, 1, hidden, generator=g, device="cuda", dtype=dtype)
    return m, x, scale, shift


def test_compile_smoke():
    # torch.compile over the guarded forward_cuda: with the is_compiling
    # guard the compiled region takes the NATIVE chain, so this compares
    # inductor's own LN+modulation fusion against the eager 3-kernel chain.
    # Cross-implementation BF16 outputs differ at rounding level (up to a few
    # ulps at O(1-8) magnitudes), hence the loose tolerance. The smoke
    # verifies the compiled workload stays functional - no crash, finite
    # outputs, no request failure.
    m, x, scale, shift = _make()
    compiled = torch.compile(m.forward_cuda, dynamic=False)
    out1 = compiled(x, scale, shift)
    out2 = compiled(x, scale, shift)
    native = m.forward_native(x, scale, shift)
    torch.testing.assert_close(out1.float(), native.float(), atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(out2.float(), native.float(), atol=5e-2, rtol=2e-2)


def test_cuda_graph_capture_replay_smoke():
    # CUDA graph capture must prove the FUSED path itself is captured: call
    # _adaln_fused_forward inside the capture and assert it returned a tensor
    # (None would mean the guard/fallback silently degraded the graph to the
    # native chain). Warm the JIT on a side stream so capture contains no
    # compile-time allocations, then replay and compare against the native
    # fallback on the same static input buffers.
    from vllm_omni.diffusion.layers.adalayernorm import _adaln_fused_forward

    m, x, scale, shift = _make()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fused_warm = _adaln_fused_forward(m, x, scale, shift)
            assert fused_warm is not None
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph = _adaln_fused_forward(m, x, scale, shift)
    assert out_graph is not None
    graph.replay()
    torch.accelerator.synchronize()
    native = m.forward_native(x, scale, shift)
    torch.testing.assert_close(out_graph.float(), native.float(), atol=2e-2, rtol=2e-2)

    # Replay must consume NEW input values: mutate the captured buffers and
    # verify the replay output tracks them (a no-op replay would pass the
    # check above trivially).
    x2, scale2, shift2 = torch.randn_like(x), torch.randn_like(scale), torch.randn_like(shift)
    x.copy_(x2)
    scale.copy_(scale2)
    shift.copy_(shift2)
    graph.replay()
    torch.accelerator.synchronize()
    native2 = m.forward_native(x, scale, shift)
    torch.testing.assert_close(out_graph.float(), native2.float(), atol=2e-2, rtol=2e-2)
    assert not torch.equal(native, native2), "replay must consume the new input"
