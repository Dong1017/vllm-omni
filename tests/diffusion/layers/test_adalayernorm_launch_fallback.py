# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Direct tests for the synchronous launch-failure fallback (C3a).

Simulates a backend-specific Triton launch failure (the launcher raises
synchronously) and verifies the two-layer response: the failed config is
recorded, the caller gets the correct native result, the same config is not
retried, and a different constexpr variant stays eligible. Uses a unique
(hidden size, eps) so the module-global failed-key set cannot collide with
other tests in the same session.
"""

import pytest
import torch
from triton.runtime.jit import JITFunction

from vllm_omni.diffusion.layers.adalayernorm import (
    _FAILED_ADALN_KEYS,
    AdaLayerNorm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

# Unique channels + eps so the failed key cannot collide with other tests.
_HIDDEN = 2048
_EPS = 1e-5


def _expected_key(x):
    return (x.device.index, _HIDDEN, x.dtype, False, False, False, False)


def _make():
    m = AdaLayerNorm(_HIDDEN, eps=_EPS).to(device="cuda", dtype=torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(31)
    x = torch.randn(1, 512, _HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1, _HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(1, _HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    return m, x, scale, shift


def test_launch_failure_fallback_once(monkeypatch):
    m, x, scale, shift = _make()
    native = m.forward_native(x, scale, shift)

    calls = {"n": 0}

    def raising_run(self, *args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("simulated synchronous launch failure")

    monkeypatch.setattr(JITFunction, "run", raising_run)

    # Phase 1: the launch raises synchronously; the caller must receive the
    # correct native result and the failed key must be recorded.
    out = m.forward_cuda(x, scale, shift)
    torch.testing.assert_close(out.float(), native.float())
    assert _expected_key(x) in _FAILED_ADALN_KEYS
    assert calls["n"] >= 1  # the fused launcher was attempted exactly this phase

    # Phase 2: with the launcher restored, the same config must NOT retry the
    # launcher (failed-key cache) and still return the correct native result.
    calls["n"] = 0
    out2 = m.forward_cuda(x, scale, shift)
    assert calls["n"] == 0
    torch.testing.assert_close(out2.float(), native.float())


def test_launch_failure_different_variant_still_eligible(monkeypatch):
    # A failed constexpr variant must not disable a different one: after the
    # shared-modulation key failed, a per-sample-scale call (B=2 with
    # (B, 1, C) modulation -> per_sample=True) with the same channels still
    # attempts (and uses) the launcher.
    m, _, _, _ = _make()
    g = torch.Generator(device="cuda").manual_seed(37)
    x2 = torch.randn(2, 256, _HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    ps_scale = torch.randn(2, 1, _HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    ps_shift = torch.randn(2, 1, _HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    key_ps = (x2.device.index, _HIDDEN, x2.dtype, False, False, True, False)
    assert key_ps not in _FAILED_ADALN_KEYS

    calls = {"n": 0}
    real_run = JITFunction.run

    def counting_run(self, *args, **kwargs):
        calls["n"] += 1
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "run", counting_run)
    out = m.forward_cuda(x2, ps_scale, ps_shift)
    native = m.forward_native(x2, ps_scale, ps_shift)
    # bf16 outputs across two equivalent implementations (kernel tree-sum vs
    # Welford moments) differ at bf16 rounding level, not bit-exact: use the
    # repo's bf16 tolerance, not fp32 defaults.
    torch.testing.assert_close(out.float(), native.float(), atol=2e-2, rtol=2e-2)
    assert calls["n"] >= 1  # the per-sample variant attempted the launcher
    monkeypatch.undo()
