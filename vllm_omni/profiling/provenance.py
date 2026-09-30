# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Provenance 模型（T02）：每个 derived metric 可追溯到 file/parser/query/aggregation/rank/unit。
# id 由 store 按注册顺序生成（ev_000001...），同一输入两次分析得到相同 id（AC-08）。
# 跨 run 引用（决议 4）：每条记录携带 run_id 与 source_sha256，
# 全局引用形如 "run_id:ev_000001"，可回溯到 exact profiling run 与 exact raw artifact。

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvidenceRecord:
    id: str
    source_file: str
    source_type: str  # 如 "torch_profiler_trace" / "ascend_csv" / "profiler_db"
    parser: str  # 解析器名，如 "cuda.torch_profiler"
    unit: str  # "us" / "ms" / "bytes" / "ratio" / "count" 等，禁止空串
    rank: int | None = None  # None = 不区分 rank 或 aggregate
    query: str | None = None  # SQL 或等价的确定性查询描述
    aggregation: str | None = None  # 如 "sum" / "top_k(20)"；None = 原样字段
    run_id: str | None = None  # 所属 profiling run 的稳定标识
    source_sha256: str | None = None  # source 文件内容 hash（流式计算）
    depends_on: list[str] = field(default_factory=list)  # derived 记录的 input evidence ids（P1-2）


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """流式计算文件 sha256，不整体载入内存（AC-30）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


def compute_run_id(source_hashes: list[str], backend: str, parser: str) -> str:
    """由 source 内容 hash + backend + parser 派生 run id（确定性，AC-08）。"""
    h = hashlib.sha256()
    h.update(f"backend={backend};parser={parser};".encode())
    for digest in sorted(source_hashes):
        h.update(digest.encode("utf-8"))
        h.update(b";")
    return h.hexdigest()[:12]


class ProvenanceStore:
    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id
        self._records: list[EvidenceRecord] = []

    def add(
        self,
        source_file: str,
        source_type: str,
        parser: str,
        unit: str,
        rank: int | None = None,
        query: str | None = None,
        aggregation: str | None = None,
        source_sha256: str | None = None,
        depends_on: list[str] | None = None,
    ) -> EvidenceRecord:
        if not source_file:
            raise ValueError("provenance: source_file is required")
        if not parser:
            raise ValueError("provenance: parser is required")
        if not unit:
            raise ValueError("provenance: unit is required")
        rec = EvidenceRecord(
            id=f"ev_{len(self._records) + 1:06d}",
            source_file=source_file,
            source_type=source_type,
            rank=rank,
            query=query,
            parser=parser,
            aggregation=aggregation,
            unit=unit,
            run_id=self.run_id,
            source_sha256=source_sha256,
            depends_on=list(depends_on) if depends_on else [],
        )
        self._records.append(rec)
        return rec

    @property
    def records(self) -> list[EvidenceRecord]:
        return list(self._records)

    def get(self, evidence_id: str) -> EvidenceRecord:
        for rec in self._records:
            if rec.id == evidence_id:
                return rec
        raise KeyError(f"provenance: unknown evidence id {evidence_id!r}")

    def ref(self, evidence_id: str) -> str:
        """跨 run 可用的全局引用形式 run_id:ev_id。"""
        if self.run_id is None:
            return evidence_id
        return f"{self.run_id}:{evidence_id}"
