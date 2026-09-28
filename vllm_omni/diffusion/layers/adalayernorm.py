from importlib.util import find_spec
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.diffusion.layers.norm import LayerNorm

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)

_HAS_MINDIESD = find_spec("mindiesd") is not None

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton is a hard dep on CUDA builds
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _adaln_scale_shift_layernorm_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        weight_ptr,
        bias_ptr,
        eps,
        channels,
        has_weight: tl.constexpr,
        has_bias: tl.constexpr,
        is_half: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """One program per LayerNorm row (contiguous x): out = ln(x) * (1 + scale) + shift.

        Reductions accumulate in fp32 (matches the golden path, which computes
        F.layer_norm on x.float()). For half-precision outputs the golden chain
        rounds after every torch op (LN result, 1+scale, product, sum), so the
        kernel replicates exactly those roundings - keeping the output
        bit-faithful to the frozen semantics. fp32 outputs have no intermediate
        rounding in the golden path and use a plain fp32 chain. BLOCK_C covers
        the whole normalized dim (masked); it is never autotuned. Non-contiguous
        x never reaches this kernel (routed to forward_native upstream).
        """
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, block_c)
        mask = cols < channels
        x = tl.load(x_ptr + row * channels + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / channels
        xm = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xm * xm, axis=0) / channels
        xn = xm * tl.rsqrt(var + eps)
        if has_weight:
            xn = xn * tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if has_bias:
            xn = xn + tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sh = tl.load(shift_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out_dtype = out_ptr.dtype.element_ty
        if is_half:
            # Golden bf16/fp16 chain: round after each op, exactly as the
            # sequence of torch elementwise kernels does (fp32 internal math).
            ln16 = xn.to(out_dtype)
            t1 = (1.0 + s).to(out_dtype)
            t2 = (ln16.to(tl.float32) * t1.to(tl.float32)).to(out_dtype)
            y = (t2.to(tl.float32) + sh).to(out_dtype)
        else:
            y = xn * (1.0 + s) + sh
        # out is always freshly allocated contiguous (B, L, C)
        tl.store(out_ptr + row * channels + cols, y.to(out_dtype), mask=mask)

    # Deterministic per-channel-count launch configs: BLOCK_C = next_pow2(C)
    # (covers the reduction dim, never autotuned), num_warps from a static
    # size heuristic. No search is performed.
    _ADALN_CONFIGS: dict = {}
    _ADALN_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

    def _adaln_is_channelwise(t: torch.Tensor, x: torch.Tensor) -> bool:
        """True iff t broadcasts per-channel against x (B, L, C), i.e. shape
        (..., C) with every leading dim == 1 (or 1-D (C,)). Anything else
        (per-row/per-sample modulation or an invalid broadcast) must fall back
        to forward_native, which reproduces torch semantics exactly - including
        the frozen B>1 (B, C) RuntimeError.
        """
        shape = t.shape
        if t.ndim < 1 or t.ndim > x.ndim or shape[-1] != x.shape[-1]:
            return False
        for d in shape[:-1]:
            if d != 1:
                return False
        return True

    def _adaln_fused_forward(module: "AdaLayerNorm", x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        batch, seq_len, channels = x.shape
        rows = batch * seq_len
        out = torch.empty((batch, seq_len, channels), dtype=x.dtype, device=x.device)
        ln = module.layernorm
        weight = ln.weight
        bias = ln.bias
        has_weight = weight is not None
        has_bias = bias is not None
        is_half = x.dtype is not torch.float32
        cfg = _ADALN_CONFIGS.get(channels)
        if cfg is None:
            block_c = triton.next_power_of_2(channels)
            num_warps = 4 if block_c <= 1024 else (8 if block_c <= 4096 else 16)
            cfg = (block_c, num_warps)
            _ADALN_CONFIGS[channels] = cfg
        block_c = cfg[0]
        # Dummy pointer args for disabled branches: never dereferenced because
        # the loads are constexpr-pruned when HAS_WEIGHT/HAS_BIAS is False.
        args = (
            x,
            scale,
            shift,
            out,
            weight if has_weight else scale,
            bias if has_bias else scale,
            module.eps,
            channels,
            has_weight,
            has_bias,
            is_half,
            block_c,
        )
        _adaln_scale_shift_layernorm_kernel[(rows,)](*args, num_warps=cfg[1])
        return out


class AdaLayerNorm(CustomOp):
    """
    AdaLayerNorm:
        out = layernorm(x) * (1 + scale) + shift
    """

    def __init__(self, hidden_size: int, elementwise_affine: bool = False, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.hidden_size = hidden_size
        self.layernorm = LayerNorm(self.hidden_size, elementwise_affine=self.elementwise_affine, eps=self.eps)

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        # Fused Triton path for the canonical per-channel modulation case with
        # contiguous, 16B-aligned tensors (fresh torch allocations always are).
        # Everything else (non-contiguous/unaligned tensors, odd dtypes or
        # shapes, per-row or per-sample scale/shift) falls back to
        # forward_native, which reproduces torch broadcasting exactly -
        # including the frozen B>1 (B, C) RuntimeError.
        if (
            _HAS_TRITON
            and x.is_cuda
            and x.ndim == 3
            and x.numel() > 0
            and x.is_contiguous()
            and scale.dtype is x.dtype
            and shift.dtype is x.dtype
            and scale.is_contiguous()
            and shift.is_contiguous()
            and x.data_ptr() % 16 == 0
            and scale.data_ptr() % 16 == 0
            and shift.data_ptr() % 16 == 0
            and x.dtype in _ADALN_DTYPES
            and _adaln_is_channelwise(scale, x)
            and _adaln_is_channelwise(shift, x)
        ):
            return _adaln_fused_forward(self, x, scale, shift)
        return self.forward_native(x, scale, shift)

    def forward_hip(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, scale, shift)

    def forward_npu(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        if _HAS_MINDIESD:
            try:
                from mindiesd import layernorm_scale_shift

                output = layernorm_scale_shift(self.layernorm, x, scale, shift, fused=True)

                return output
            except ImportError as e:
                logger.warning_once(f"mindiesd import failed, falling back to torch_npu: {e}")

        import torch_npu

        output = (
            torch_npu.npu_layer_norm_eval(x, normalized_shape=[self.hidden_size], eps=self.eps) * (1 + scale) + shift
        )

        return output

    def forward_native(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.layernorm(x) * (1 + scale) + shift


class AdaLayerNormZero(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.emb = None
        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(
            embedding_dim,
            6 * embedding_dim,
            bias=bias,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        if isinstance(emb, tuple):
            emb = emb[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(
            embedding_dim,
            3 * embedding_dim,
            bias=bias,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        if isinstance(emb, tuple):
            emb = emb[0]
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = False,
        eps: float = 1e-6,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(
            conditioning_embedding_dim,
            embedding_dim * 2,
            bias=bias,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        if isinstance(emb, tuple):
            emb = emb[0]
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
