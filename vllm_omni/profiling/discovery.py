# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Artifact 发现（T06/T07/T08）：按文件结构检测 backend，失败必须显式报错，不得猜测。
# CUDA 命名遵循 vllm_omni/profiler/omni_torch_profiler.py 的 {stem}_rank{N}{suffix} 约定。

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 超过该大小的 trace 默认拒绝整体 json.load（AC-30），提示走流式/HTA 替代路径
DEFAULT_MAX_TRACE_BYTES = 2 * 1024**3

_CUDA_TRACE_PATTERNS = ("trace_rank*.json", "trace_rank*.json.gz", "stage*_rank*.json", "stage*_rank*.json.gz")
_CUDA_SIDECAR_PATTERNS = (
    "ops_rank*.xlsx",
    "stacks_cpu_rank*.txt",
    "stacks_cuda_rank*.txt",
    "memory_snapshot_rank*.pickle",
)
_ASCEND_CSV_NAMES = ("step_trace_time.csv", "kernel_details.csv", "operator_details.csv", "api_statistic.csv")
_ASCEND_GLOB_NAMES = ("op_statistic*.csv", "op_summary*.csv")
_ASCEND_DB_NAMES = ("analysis.db",)


class BackendDetectionError(ValueError):
    pass


class ArtifactSizeError(ValueError):
    pass


@dataclass
class CudaArtifacts:
    trace_files: list[Path] = field(default_factory=list)
    sidecar_files: list[Path] = field(default_factory=list)


@dataclass
class AscendArtifacts:
    step_trace: list[Path] = field(default_factory=list)
    op_statistic: list[Path] = field(default_factory=list)
    op_summary: list[Path] = field(default_factory=list)
    kernel_details: list[Path] = field(default_factory=list)
    profiler_db: list[Path] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    root: Path
    backend: str  # "cuda" / "ascend"
    cuda: CudaArtifacts = field(default_factory=CudaArtifacts)
    ascend: AscendArtifacts = field(default_factory=AscendArtifacts)


def _match(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        found.extend(p for p in root.rglob(pat) if p.is_file())
    return sorted(set(found))


def detect_backend(root: Path) -> str | None:
    """根据文件结构判断 backend；无法判断时返回 None（由调用方显式报错）。"""
    if root.is_file():
        name = root.name
        if name.endswith((".json", ".json.gz")) and "rank" in name:
            return "cuda"
        if name in _ASCEND_CSV_NAMES or name == _ASCEND_DB_NAMES[0] or name.startswith(("op_statistic", "op_summary")):
            return "ascend"
        return None
    cuda_hits = _match(root, _CUDA_TRACE_PATTERNS)
    ascend_hits = _match(root, _ASCEND_CSV_NAMES + _ASCEND_GLOB_NAMES + _ASCEND_DB_NAMES) + _match(
        root, ("ASCEND_PROFILER_OUTPUT",)
    )
    ascend_dirs = [p for p in ascend_hits if p.is_dir()]
    ascend_files = [p for p in ascend_hits if p.is_file()]
    has_ascend = bool(ascend_files or ascend_dirs)
    if cuda_hits and has_ascend:
        raise BackendDetectionError(
            f"discovery: both CUDA and Ascend markers under {root}; "
            f"cuda={[p.name for p in cuda_hits[:3]]} ascend={[p.name for p in (ascend_files + ascend_dirs)[:3]]}; "
            "backend=auto requires an unambiguous artifact set"
        )
    if cuda_hits:
        return "cuda"
    if has_ascend:
        return "ascend"
    return None


def discover(root: Path, backend: str) -> DiscoveryResult:
    root = Path(root)
    if not root.exists():
        raise BackendDetectionError(f"discovery: input path does not exist: {root}")
    result = DiscoveryResult(root=root, backend=backend)
    if backend == "cuda":
        result.cuda.trace_files = _match(root, _CUDA_TRACE_PATTERNS)
        result.cuda.sidecar_files = _match(root, _CUDA_SIDECAR_PATTERNS)
        if not result.cuda.trace_files:
            raise BackendDetectionError(
                f"discovery: no torch profiler trace (trace_rank*/stage*_rank*.json[.gz]) under {root}"
            )
    elif backend == "ascend":
        result.ascend.step_trace = _match(root, ("step_trace_time.csv",))
        result.ascend.op_statistic = _match(root, ("op_statistic*.csv",))
        result.ascend.op_summary = _match(root, ("op_summary*.csv",))
        result.ascend.kernel_details = _match(root, ("kernel_details.csv",))
        result.ascend.profiler_db = _match(root, ("analysis.db", "*.db"))
        if not any(
            [
                result.ascend.step_trace,
                result.ascend.op_statistic,
                result.ascend.op_summary,
                result.ascend.kernel_details,
                result.ascend.profiler_db,
            ]
        ):
            raise BackendDetectionError(
                f"discovery: no known Ascend artifact (step_trace_time/op_statistic*/op_summary*/"
                f"kernel_details.csv/analysis.db) under {root}"
            )
    else:
        raise BackendDetectionError(f"discovery: unsupported backend {backend!r}")
    return result


def check_size(path: Path, max_bytes: int = DEFAULT_MAX_TRACE_BYTES) -> int:
    size = path.stat().st_size
    if size > max_bytes:
        raise ArtifactSizeError(
            f"artifact {path} is {size} bytes (> limit {max_bytes}); "
            "refusing full-file json.load. Use a smaller trace, raise the limit explicitly, "
            "or use HTA/nsys DB based analysis for large traces"
        )
    return size
