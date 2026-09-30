# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# CLI 测试（T18）：坏路径、不支持格式、backend 不匹配、空 profile、部分 artifact（AC-31）

import json
from pathlib import Path

import pytest

from tests.profiling.test_cuda_torch_profiler import _trace_dict
from vllm_omni.profiling.cli import main

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture()
def cuda_dir(tmp_path: Path) -> Path:
    (tmp_path / "trace_rank0.json").write_text(json.dumps(_trace_dict()), encoding="utf-8")
    return tmp_path


def test_analyze_bad_path(tmp_path, capsys):
    rc = main(["analyze", "--backend", "cuda", "--input", str(tmp_path / "nope"), "--output", str(tmp_path / "out")])
    assert rc == 2
    assert "does not exist" in capsys.readouterr().err


def test_analyze_backend_mismatch_ascend_requested(tmp_path, cuda_dir, capsys):
    rc = main(["analyze", "--backend", "ascend", "--input", str(cuda_dir), "--output", str(tmp_path / "out")])
    assert rc == 2
    # CUDA 目录用 ascend 分析 -> discovery 显式报错（AC-31），不静默输出空报告
    assert "no known Ascend artifact" in capsys.readouterr().err


def test_analyze_ascend_end_to_end(tmp_path):
    from tests.profiling.test_ascend_csv import write_ascend_fixtures

    write_ascend_fixtures(tmp_path)
    out = tmp_path / "out"
    rc = main(["analyze", "--backend", "ascend", "--input", str(tmp_path), "--output", str(out)])
    assert rc == 0
    # P0-1：Ascend 无 rank 证据 -> 输出目录回退到 device_id
    assert (out / "device0" / "evidence.json").exists()
    assert (out / "device0" / "summary.md").exists()


def test_analyze_backend_cuda_on_ascend_dir(tmp_path, capsys):
    out = tmp_path / "ASCEND_PROFILER_OUTPUT"
    out.mkdir()
    (out / "op_summary_0.csv").write_text("a\n", encoding="utf-8")
    rc = main(["analyze", "--backend", "cuda", "--input", str(tmp_path), "--output", str(tmp_path / "o2")])
    assert rc == 2
    assert "no torch profiler trace" in capsys.readouterr().err


def test_analyze_auto_detects_cuda(tmp_path, cuda_dir):
    out = tmp_path / "out"
    rc = main(["analyze", "--backend", "auto", "--input", str(cuda_dir), "--output", str(out)])
    assert rc == 0
    # AC-01：统一入口产物
    for name in ("evidence.json", "summary.md", "operators.csv", "provenance.json"):
        assert (out / name).exists(), name


def test_analyze_multi_rank_no_cross_rank_merge(tmp_path):
    d = tmp_path / "prof"
    d.mkdir()
    (d / "trace_rank0.json").write_text(json.dumps(_trace_dict()), encoding="utf-8")
    (d / "trace_rank1.json").write_text(json.dumps(_trace_dict()), encoding="utf-8")
    out = tmp_path / "out"
    rc = main(["analyze", "--backend", "cuda", "--input", str(d), "--output", str(out)])
    assert rc == 0
    assert (out / "rank0" / "evidence.json").exists()
    assert (out / "rank1" / "evidence.json").exists()
    assert not (out / "evidence.json").exists()  # 禁止跨 rank 合并输出（AC-05）


def test_analyze_corrupt_trace_fails(tmp_path):
    d = tmp_path / "prof"
    d.mkdir()
    (d / "trace_rank0.json").write_text("{not json", encoding="utf-8")
    rc = main(["analyze", "--backend", "cuda", "--input", str(d), "--output", str(tmp_path / "out")])
    assert rc == 2


def test_analyze_empty_trace_produces_unavailable(tmp_path):
    d = tmp_path / "prof"
    d.mkdir()
    (d / "trace_rank0.json").write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
    out = tmp_path / "out"
    rc = main(["analyze", "--backend", "cuda", "--input", str(d), "--output", str(out)])
    assert rc == 0
    evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["workload"]["wall_ms"] is None
    assert evidence["timeline"]["device_busy_ms"] is None


def test_query_views_on_generated_evidence(tmp_path, cuda_dir, capsys):
    out = tmp_path / "out"
    main(["analyze", "--backend", "cuda", "--input", str(cuda_dir), "--output", str(out)])
    evidence = out / "evidence.json"
    for view in ("summary", "operators", "shapes", "runtime", "memory", "communication", "diagnosis", "provenance"):
        rc = main(["query", "--evidence", str(evidence), "--view", view])
        assert rc == 0, view
        assert capsys.readouterr().out


def test_query_missing_evidence_file(tmp_path, capsys):
    rc = main(["query", "--evidence", str(tmp_path / "nope.json"), "--view", "summary"])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


def test_query_invalid_view_rejected_by_argparse(tmp_path, cuda_dir):
    out = tmp_path / "out"
    main(["analyze", "--backend", "cuda", "--input", str(cuda_dir), "--output", str(out)])
    with pytest.raises(SystemExit):
        main(["query", "--evidence", str(out / "evidence.json"), "--view", "made_up_view"])
