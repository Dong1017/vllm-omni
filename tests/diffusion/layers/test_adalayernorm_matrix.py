"""P0 correctness matrix for the fused AdaLayerNorm CUDA path.

Covers: fp32/bf16/fp16, B>1 (shared modulation), C in {1536,3072,4096},
varying L, elementwise_affine False/True with NON-DEFAULT weight/bias,
eps variants, legal non-contiguous inputs, broadcast scale/shift shapes
((C,), (1,C), (1,1,C)), and fallback completeness (dtypes/shapes that must
route to forward_native, incl. the frozen B>1 (B,C) RuntimeError).
"""

import pytest
import torch

from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm

TOL_STRICT = {"bf16": (2e-2, 2e-2), "fp16": (1e-2, 1e-2), "fp32": (1e-3, 1e-3)}
TOL_LOOSE = {"bf16": (5e-2, 2e-2), "fp16": (2e-2, 1e-2), "fp32": (5e-3, 5e-3)}
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def tol_key(dtype):
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return "fp32"


def fp32_reference(x, scale, shift, eps, weight=None, bias=None):
    xf = x.float()
    w = weight.float() if weight is not None else None
    b = bias.float() if bias is not None else None
    xn = torch.nn.functional.layer_norm(xf, (x.shape[-1],), w, b, eps)
    return xn * (1 + scale.float()[:, None]) + shift.float()[:, None]


def make_module(hidden, elementwise_affine, eps, device, dtype, nondefault_affine=False):
    m = AdaLayerNorm(hidden, elementwise_affine=elementwise_affine, eps=eps)
    if elementwise_affine:
        m = m.to(dtype)
        if nondefault_affine:
            with torch.no_grad():
                g = torch.Generator(device="cpu").manual_seed(42)
                m.layernorm.weight.copy_(0.5 + torch.rand(hidden, generator=g))
                m.layernorm.bias.copy_(0.2 * torch.randn(hidden, generator=g))
    return m.to(device)


def make_inputs(bs, seq, hidden, dtype, device, seed=0, mod_shape=None):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(bs, seq, hidden, generator=g, device=device, dtype=dtype)
    ms = mod_shape or (1, hidden)
    scale = torch.randn(ms, generator=g, device=device, dtype=dtype)
    shift = torch.randn(ms, generator=g, device=device, dtype=dtype)
    return x, scale, shift


def assert_close(a, b, dtype, loose=False):
    atol, rtol = (TOL_LOOSE if loose else TOL_STRICT)[tol_key(dtype)]
    torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize(
    "bs,seq,hidden", [(1, 4096, 3072), (2, 1024, 1536), (3, 128, 4096), (1, 1, 3072), (1, 8192, 1536)]
)
def test_matrix_main(dtype, affine, bs, seq, hidden):
    device = "cuda"
    m = make_module(hidden, affine, 1e-6, device, dtype)
    x, scale, shift = make_inputs(bs, seq, hidden, dtype, device)
    out_cuda = m.forward_cuda(x, scale, shift)
    out_native = m.forward_native(x, scale, shift)
    assert out_cuda.shape == x.shape and out_cuda.dtype == dtype
    assert_close(out_cuda, out_native, dtype)
    w = m.layernorm.weight if affine else None
    b = m.layernorm.bias if affine else None
    assert_close(out_cuda, fp32_reference(x, scale, shift, 1e-6, w, b), dtype, loose=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_weight_bias_nondefault(dtype):
    # weight/bias 语义真实验证（非恒等初始化）：out = (w*ln(x)+b)*(1+scale)+shift
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, True, 1e-6, device, dtype, nondefault_affine=True)
    x, scale, shift = make_inputs(1, 1024, hidden, dtype, device, seed=13)
    out_cuda = m.forward_cuda(x, scale, shift)
    out_native = m.forward_native(x, scale, shift)
    assert_close(out_cuda, out_native, dtype)
    ref = fp32_reference(x, scale, shift, 1e-6, m.layernorm.weight, m.layernorm.bias)
    assert_close(out_cuda, ref, dtype, loose=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("eps", [1e-5, 1e-3, 1e-2])
def test_matrix_eps_variants(dtype, eps):
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, eps, device, dtype)
    x, scale, shift = make_inputs(1, 512, hidden, dtype, device, seed=3)
    assert_close(m.forward_cuda(x, scale, shift), m.forward_native(x, scale, shift), dtype)
    assert_close(
        m.forward_cuda(x, scale, shift),
        fp32_reference(x, scale, shift, eps),
        dtype,
        loose=True,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mod_shape", ["C", "1x1xC", "1xC"])
def test_matrix_broadcast_scale_shift(dtype, mod_shape):
    # (C,) / (1,1,C) / (1,C) 都是对 x (B,L,C) 的合法 per-channel 广播
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, _, _ = make_inputs(2, 512, hidden, dtype, device, seed=5)
    g = torch.Generator(device=device).manual_seed(6)
    if mod_shape == "C":
        scale = torch.randn(hidden, generator=g, device=device, dtype=dtype)
        shift = torch.randn(hidden, generator=g, device=device, dtype=dtype)
    elif mod_shape == "1x1xC":
        scale = torch.randn(1, 1, hidden, generator=g, device=device, dtype=dtype)
        shift = torch.randn(1, 1, hidden, generator=g, device=device, dtype=dtype)
    else:
        scale = torch.randn(1, hidden, generator=g, device=device, dtype=dtype)
        shift = torch.randn(1, hidden, generator=g, device=device, dtype=dtype)
    out_cuda = m.forward_cuda(x, scale, shift)
    out_native = m.forward_native(x, scale, shift)
    assert_close(out_cuda, out_native, dtype)
    s = scale.float().reshape(1, 1, hidden) if scale.ndim == 1 else scale.float()
    sh = shift.float().reshape(1, 1, hidden) if shift.ndim == 1 else shift.float()
    ref = fp32_reference(x, s.reshape(-1, hidden)[0:1], sh.reshape(-1, hidden)[0:1], 1e-6)
    assert_close(out_cuda, ref, dtype, loose=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matrix_noncontiguous_fallback(dtype):
    # 非连续输入必须回落 native 且结果正确（kernel 只服务连续快路径）
    device = "cuda"
    hidden = 1536
    m = make_module(hidden, False, 1e-6, device, dtype)
    x_big, scale, shift = make_inputs(1, 1024, hidden, dtype, device, seed=7)
    x = x_big[:, ::2, :]
    assert not x.is_contiguous()
    out_cuda = m.forward_cuda(x, scale, shift)
    out_native = m.forward_native(x, scale, shift)
    assert_close(out_cuda, out_native, dtype)
    assert_close(out_cuda, fp32_reference(x.contiguous(), scale, shift, 1e-6), dtype, loose=True)


def test_matrix_fallback_fp64():
    # fp64 不在 kernel 支持列表 → 必须走 native 且正确（fallback 完整性）
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, torch.float64)
    x, scale, shift = make_inputs(1, 256, hidden, torch.float64, device, seed=9)
    out_cuda = m.forward_cuda(x, scale, shift)
    out_native = m.forward_native(x, scale, shift)
    torch.testing.assert_close(out_cuda, out_native)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matrix_determinism(dtype):
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, scale, shift = make_inputs(1, 4096, hidden, dtype, device, seed=11)
    assert torch.equal(m.forward_cuda(x, scale, shift), m.forward_cuda(x, scale, shift))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matrix_multi_batch_shared_modulation(dtype):
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, scale, shift = make_inputs(4, 512, hidden, dtype, device, seed=15)
    out = m.forward_cuda(x, scale, shift)
    for b in range(4):
        single = m.forward_cuda(x[b : b + 1], scale, shift)
        assert_close(out[b : b + 1], single, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matrix_per_sample_modulation_batch_gt1_raises(dtype):
    # 冻结语义：B>1 且 scale (B,C) 保持 RuntimeError（fused 与 native 一致）
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, scale, shift = make_inputs(4, 64, hidden, dtype, device, seed=17, mod_shape=(4, hidden))
    with pytest.raises(RuntimeError):
        m.forward_native(x, scale, shift)
    with pytest.raises(RuntimeError):
        m.forward_cuda(x, scale, shift)
