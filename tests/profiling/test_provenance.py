# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Provenance 测试：id 确定性、必填字段 fail fast、可检索（AC-07 地基）

import pytest

from vllm_omni.profiling.provenance import ProvenanceStore

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _populate(store: ProvenanceStore) -> None:
    store.add(
        source_file="trace_rank0.json",
        source_type="torch_profiler_trace",
        parser="cuda.torch_profiler",
        unit="us",
        rank=0,
        query="traceEvents[type=kernel]",
        aggregation="sum(name)",
    )
    store.add(
        source_file="op_summary_0.csv",
        source_type="ascend_csv",
        parser="ascend.csv.op_summary",
        unit="us",
        rank=0,
    )


def test_ids_are_sequential_and_deterministic():
    s1, s2 = ProvenanceStore(), ProvenanceStore()
    _populate(s1)
    _populate(s2)
    assert [r.id for r in s1.records] == ["ev_000001", "ev_000002"]
    assert [r.id for r in s1.records] == [r.id for r in s2.records]


def test_records_keep_all_fields():
    store = ProvenanceStore()
    rec = store.add(
        source_file="analysis.db",
        source_type="profiler_db",
        parser="ascend.profiler_db",
        unit="us",
        rank=3,
        query="SELECT ...",
        aggregation="top_k(20)",
    )
    assert rec.id == "ev_000001"
    assert rec.source_file == "analysis.db"
    assert rec.source_type == "profiler_db"
    assert rec.rank == 3
    assert rec.query == "SELECT ..."
    assert rec.parser == "ascend.profiler_db"
    assert rec.aggregation == "top_k(20)"
    assert rec.unit == "us"


def test_rank_none_means_unspecified():
    store = ProvenanceStore()
    rec = store.add(source_file="f.csv", source_type="ascend_csv", parser="p", unit="ms")
    assert rec.rank is None


@pytest.mark.parametrize("field", ["source_file", "parser", "unit"])
def test_required_fields_fail_fast(field):
    store = ProvenanceStore()
    kwargs = {
        "source_file": "f",
        "source_type": "t",
        "parser": "p",
        "unit": "us",
    }
    kwargs[field] = ""
    with pytest.raises(ValueError, match=field if field != "source_file" else "source_file"):
        store.add(**kwargs)


def test_get_unknown_id_raises():
    store = ProvenanceStore()
    with pytest.raises(KeyError, match="ev_000009"):
        store.get("ev_000009")


def test_run_id_stamped_on_all_records():
    store = ProvenanceStore(run_id="abc123def456")
    _populate(store)
    assert all(r.run_id == "abc123def456" for r in store.records)
    # 无 run_id 时为 None，不编造
    bare = ProvenanceStore()
    _populate(bare)
    assert all(r.run_id is None for r in bare.records)


def test_ref_form():
    store = ProvenanceStore(run_id="abc123def456")
    rec = store.add(source_file="f", source_type="t", parser="p", unit="us")
    assert store.ref(rec.id) == "abc123def456:ev_000001"
    bare = ProvenanceStore()
    rec2 = bare.add(source_file="f", source_type="t", parser="p", unit="us")
    assert bare.ref(rec2.id) == "ev_000001"  # 无 run_id 退化为本地 id


def test_sha256_file_streams(tmp_path):
    from vllm_omni.profiling.provenance import compute_run_id, sha256_file

    f = tmp_path / "blob.bin"
    f.write_bytes(b"hello world")
    digest = sha256_file(f)
    assert len(digest) == 64
    # 内容不变 -> hash 不变；内容变 -> hash 变
    assert sha256_file(f) == digest
    f.write_bytes(b"hello world!")
    assert sha256_file(f) != digest
    # run_id 由 source hash 派生，确定性且区分内容
    rid1 = compute_run_id([digest], backend="cuda", parser="p")
    rid2 = compute_run_id([digest], backend="cuda", parser="p")
    rid3 = compute_run_id([sha256_file(f)], backend="cuda", parser="p")
    assert rid1 == rid2 and len(rid1) == 12
    assert rid3 != rid1
