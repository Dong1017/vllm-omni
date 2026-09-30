# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Artifact 发现测试（T09）：backend 检测、歧义显式失败、size guard

from pathlib import Path

import pytest

from vllm_omni.profiling.discovery import (
    ArtifactSizeError,
    BackendDetectionError,
    check_size,
    detect_backend,
    discover,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_cuda_dir(tmp_path: Path) -> Path:
    (tmp_path / "trace_rank0.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ops_rank0.xlsx").write_bytes(b"x")
    return tmp_path


def _make_ascend_dir(tmp_path: Path) -> Path:
    out = tmp_path / "ASCEND_PROFILER_OUTPUT"
    out.mkdir()
    (out / "op_summary_0.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (out / "step_trace_time.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    return tmp_path


def test_detect_cuda_dir(tmp_path):
    assert detect_backend(_make_cuda_dir(tmp_path)) == "cuda"


def test_detect_ascend_dir(tmp_path):
    assert detect_backend(_make_ascend_dir(tmp_path)) == "ascend"


def test_detect_single_trace_file(tmp_path):
    f = tmp_path / "trace_rank3.json.gz"
    f.write_bytes(b"x")
    assert detect_backend(f) == "cuda"


def test_detect_single_ascend_csv(tmp_path):
    f = tmp_path / "op_summary_0.csv"
    f.write_text("a\n", encoding="utf-8")
    assert detect_backend(f) == "ascend"


def test_detect_ambiguous_fails_explicitly(tmp_path):
    _make_cuda_dir(tmp_path)
    (tmp_path / "op_summary_0.csv").write_text("a\n", encoding="utf-8")
    with pytest.raises(BackendDetectionError, match="both CUDA and Ascend"):
        detect_backend(tmp_path)


def test_detect_unknown_returns_none(tmp_path):
    (tmp_path / "random.txt").write_text("x", encoding="utf-8")
    assert detect_backend(tmp_path) is None


def test_discover_cuda_lists_traces(tmp_path):
    root = _make_cuda_dir(tmp_path)
    result = discover(root, "cuda")
    assert result.backend == "cuda"
    assert [p.name for p in result.cuda.trace_files] == ["trace_rank0.json"]
    assert [p.name for p in result.cuda.sidecar_files] == ["ops_rank0.xlsx"]


def test_discover_cuda_without_trace_fails(tmp_path):
    (tmp_path / "ops_rank0.xlsx").write_bytes(b"x")
    with pytest.raises(BackendDetectionError, match="no torch profiler trace"):
        discover(tmp_path, "cuda")


def test_discover_ascend_lists_csvs(tmp_path):
    root = _make_ascend_dir(tmp_path)
    result = discover(root, "ascend")
    assert result.ascend.op_summary
    assert result.ascend.step_trace


def test_discover_unknown_backend_fails(tmp_path):
    with pytest.raises(BackendDetectionError, match="unsupported backend"):
        discover(tmp_path, "tpu")


def test_discover_missing_path_fails(tmp_path):
    with pytest.raises(BackendDetectionError, match="does not exist"):
        discover(tmp_path / "nope", "cuda")


def test_size_guard(tmp_path):
    f = tmp_path / "trace_rank0.json"
    f.write_text("{}", encoding="utf-8")
    assert check_size(f, max_bytes=1000) == 2
    with pytest.raises(ArtifactSizeError, match="refusing full-file"):
        check_size(f, max_bytes=1)
