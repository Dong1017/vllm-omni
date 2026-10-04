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


# ---- M4.2a final-review：NCU 接线 / provenance identity / multi-trace fail-fast ----


def _write_ncu_csv(path: Path, launch_value: str) -> None:
    header = (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
        '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"'
    )
    rows = [
        f'"0","1234","python","host","ampere_sgemm","0","7","(128, 1, 1)","(1, 1, 1)","0","8.0",'
        f'"LaunchStats","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","{launch_value}"'
    ]
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_ncu_csv_attaches_hardware_provenance(tmp_path):
    trace = tmp_path / "trace_rank0.json"
    trace.write_text(json.dumps(_trace_dict()), encoding="utf-8")
    ncu = tmp_path / "ncu.csv"
    _write_ncu_csv(ncu, "72.5")
    out = tmp_path / "out"
    rc = main(
        [
            "analyze",
            "--backend",
            "cuda",
            "--input",
            str(trace.parent),
            "--output",
            str(out),
            "--ncu-csv",
            str(ncu),
        ]
    )
    assert rc == 0
    evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    hw = evidence["hardware"]
    assert hw["backend"] == "cuda"
    assert hw["provenance_ref"]
    # serialized provenance 实际包含 hardware 记录（P0-3）
    prov_ids = {r["id"] for r in evidence["provenance"]}
    hw_ref = hw["provenance_ref"]
    assert hw_ref.split(":", 1)[1] in prov_ids


def test_ncu_csv_metric_evidence_binding(tmp_path):
    trace = tmp_path / "trace_rank0.json"
    trace.write_text(json.dumps(_trace_dict()), encoding="utf-8")
    ncu = tmp_path / "ncu.csv"
    _write_ncu_csv(ncu, "72.5")
    out = tmp_path / "out"
    ncu_path = str(ncu)
    main(
        [
            "analyze",
            "--backend",
            "cuda",
            "--input",
            str(trace.parent),
            "--output",
            str(out),
            "--ncu-csv",
            ncu_path,
        ]
    )
    evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    keys = [k for k in evidence["metric_evidence"] if k.startswith("hardware:")]
    assert keys, "hardware metric keys must be bound"
    # 每个 ref 都能 resolve 到 serialized provenance 中的条目
    prov_ids = {r["id"] for r in evidence["provenance"]}
    for k, refs in evidence["metric_evidence"].items():
        if k.startswith("hardware:"):
            for ref in refs:
                assert ref.split(":", 1)[1] in prov_ids, (k, ref)


def test_two_different_ncu_csv_contents_give_different_refs(tmp_path):
    trace = tmp_path / "trace_rank0.json"
    trace.write_text(json.dumps(_trace_dict()), encoding="utf-8")
    ncu_a = tmp_path / "a.csv"
    ncu_b = tmp_path / "b.csv"
    _write_ncu_csv(ncu_a, "72.5")
    _write_ncu_csv(ncu_b, "91.0")
    refs = []
    for csv_path in (ncu_a, ncu_b):
        out = tmp_path / f"out_{csv_path.stem}"
        main(
            [
                "analyze",
                "--backend",
                "cuda",
                "--input",
                str(trace.parent),
                "--output",
                str(out),
                "--ncu-csv",
                str(csv_path),
            ]
        )
        evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
        refs.append(evidence["hardware"]["provenance_ref"])
    assert refs[0] != refs[1]  # P0-3：不同 NCU 内容 -> 不同 hardware provenance ref


def test_hardware_provenance_resolution_invariant(tmp_path):
    # final-review P0-3 item 5：hardware.provenance_ref 和全部 hardware
    # metric_evidence refs 的 (run_id, evidence_id) 都在 serialized provenance 中
    trace = tmp_path / "trace_rank0.json"
    trace.write_text(json.dumps(_trace_dict()), encoding="utf-8")
    ncu = tmp_path / "ncu.csv"
    _write_ncu_csv(ncu, "72.5")
    out = tmp_path / "out"
    main(["analyze", "--backend", "cuda", "--input", str(trace.parent), "--output", str(out), "--ncu-csv", str(ncu)])
    evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    hw = evidence["hardware"]
    hw_ref = hw["provenance_ref"]
    assert hw_ref  # hardware provenance_ref 存在
    hw_run_id, hw_eid = hw_ref.split(":", 1)
    prov_by_id = {r["id"]: r for r in evidence["provenance"]}
    assert hw_eid in prov_by_id, "hardware provenance record must exist"
    assert prov_by_id[hw_eid]["run_id"] == hw_run_id
    # 每个 hardware metric_evidence ref 都能 resolve
    for k, refs in evidence["metric_evidence"].items():
        if not k.startswith("hardware:"):
            continue
        for ref in refs:
            run_id, eid = ref.split(":", 1)
            assert eid in prov_by_id, f"ref {ref} does not resolve"
