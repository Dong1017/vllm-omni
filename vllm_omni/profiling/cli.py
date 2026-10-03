# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# CLI（T16/T17）：analyze 与 query 两个子命令。
# 失败必须显式报错退出非 0（AC-31），不静默生成空报告。

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vllm_omni.profiling.analysis import build_all_observations, enrich
from vllm_omni.profiling.backends import (
    UnsupportedProfilerDBError,
    analyze_ascend_csv,
    analyze_ascend_db,
    analyze_ascend_trace_db,
    analyze_cuda_torch_profiler,
)
from vllm_omni.profiling.discovery import BackendDetectionError, detect_backend, discover
from vllm_omni.profiling.report import write_outputs
from vllm_omni.profiling.schema import OptimizationEvidence

QUERY_VIEWS = (
    "summary",
    "operators",
    "shapes",
    "runtime",
    "memory",
    "communication",
    "diagnosis",
    "provenance",
)


def _fmt(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "unavailable"


def cmd_analyze(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    backend = args.backend
    if backend == "auto":
        detected = detect_backend(input_path)
        if detected is None:
            raise BackendDetectionError(
                f"backend=auto could not identify the artifact set under {input_path}; pass --backend explicitly"
            )
        backend = detected
    if backend == "ascend":
        outputs_local: dict[str, Path] = {}
        result = discover(input_path, "ascend")
        ascend_files = (
            result.ascend.step_trace
            + result.ascend.op_statistic
            + result.ascend.op_summary
            + result.ascend.kernel_details
            + result.ascend.operator_details
            + result.ascend.api_statistic
        )
        if not ascend_files:
            # DB-only 目录：analysis.db 存在时走 profiler DB adapter（T2）
            if result.ascend.profiler_db:
                for db_path in result.ascend.profiler_db:
                    # 按 schema 路由：analysis.db -> M2 视图；大库 -> M4.1b timeline correlation
                    try:
                        ev, _store = analyze_ascend_db(db_path)
                    except UnsupportedProfilerDBError:
                        ev, _store = analyze_ascend_trace_db(db_path)
                    ev = enrich(ev)
                    if ev.run.rank is not None:
                        leaf = f"rank{ev.run.rank}"
                    elif ev.run.device_id:
                        leaf = f"device{ev.run.device_id}"
                    else:
                        leaf = "rank_unknown"
                    target = output_dir / leaf
                    written = write_outputs(ev, target)
                    for name, path in written.items():
                        outputs_local[f"{target.name}/{name}"] = path
                    print(f"analyzed {db_path} (device_id={ev.run.device_id or 'unknown'}) -> {target}")
                for key in sorted(outputs_local):
                    print(f"  wrote {outputs_local[key]}")
                return 0
            raise BackendDetectionError(f"discovery: no known Ascend artifacts under {input_path}")
        worker_dirs = sorted({p.parent for p in ascend_files})
        for worker_dir in worker_dirs:
            ev, _store = analyze_ascend_csv(worker_dir)
            ev = enrich(ev)
            # P0-1：Ascend CSV 无 rank 证据；目录名回退到 device_id
            if ev.run.rank is not None:
                leaf = f"rank{ev.run.rank}"
            elif ev.run.device_id:
                leaf = f"device{ev.run.device_id}"
            else:
                leaf = "rank_unknown"
            target = output_dir / leaf
            written = write_outputs(ev, target)
            for name, path in written.items():
                outputs_local[f"{target.name}/{name}"] = path
            print(f"analyzed {worker_dir} (device_id={ev.run.device_id or 'unknown'}) -> {target}")
        db_only = {p.parent for p in result.ascend.profiler_db} - set(worker_dirs)
        for d in sorted(db_only):
            print(f"note: skipped {d} (no known Ascend DB schema matched)")
        for key in sorted(outputs_local):
            print(f"  wrote {outputs_local[key]}")
        return 0
    result = discover(input_path, backend)
    outputs: dict[str, Path] = {}
    traces = result.cuda.trace_files
    for trace in traces:
        ev, _store = analyze_cuda_torch_profiler(trace)
        ev = enrich(ev)
        # 单 rank 输入用标准文件名；多 rank 用带 rank 后缀的文件名，
        # 禁止跨 rank 相加 duration（AC-05），跨 rank 聚合属于后续版本
        if len(traces) == 1:
            target = output_dir
        else:
            rank = ev.run.rank
            target = output_dir / (f"rank{rank}" if rank is not None else "rank_unknown")
        written = write_outputs(ev, target)
        for name, path in written.items():
            outputs[f"{target.name}/{name}"] = path
        print(f"analyzed {trace.name} (rank={ev.run.rank}) -> {target}")
    for key in sorted(outputs):
        print(f"  wrote {outputs[key]}")
    return 0


def _load_evidence(path: str) -> OptimizationEvidence:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"evidence file not found: {p}")
    return OptimizationEvidence.from_json(p.read_text(encoding="utf-8"))


def cmd_query(args: argparse.Namespace) -> int:
    ev = _load_evidence(args.evidence)
    view = args.view
    top = args.top
    if view == "summary":
        from vllm_omni.profiling.report import render_summary_md

        print(render_summary_md(ev))
    elif view == "operators":
        # P0-3：framework 层 TopK 只用 device_self_ms（inclusive 不得用于排序）
        rows = sorted(
            [o for o in ev.operators if o.layer == "framework"],
            key=lambda o: (o.device_self_ms is None, -(o.device_self_ms or 0.0), -(o.host_inclusive_ms or 0.0)),
        )[:top]
        print(f"{'operator':<50} {'calls':>8} {'dev_self_ms':>12} {'host_incl_ms':>12} shape")
        for o in rows:
            shape = o.shapes[0] if o.shapes else ""
            print(
                f"{o.name:<50} {o.calls if o.calls is not None else '-':>8} "
                f"{_fmt(o.device_self_ms):>12} {_fmt(o.host_inclusive_ms):>12} {shape}"
            )
    elif view == "shapes":
        if not ev.shapes:
            print("unavailable")
        else:
            print(
                f"{'operator':<50} {'shape':<40} {'calls':>8} {'total_ms':>12} "
                f"{'dev_frac':>10} denominator / profiler_ratio"
            )
            for s in ev.shapes[:top]:
                denom = s.fraction_denominator or "-"
                ratio = _fmt(s.profiler_ratio, 6) if s.profiler_ratio is not None else "-"
                print(
                    f"{s.operator:<50} {s.shape:<40} {s.calls if s.calls is not None else '-':>8} "
                    f"{_fmt(s.total_ms):>12} {_fmt(s.device_time_fraction, 6):>10} {denom} / {ratio}"
                )
    elif view == "runtime":
        r = ev.runtime
        print(f"launch_count       = {r.launch_count if r.launch_count is not None else 'unavailable'}")
        print(f"api_summed_ms      = {_fmt(r.api_summed_ms)}")
        print(f"launch_summed_ms   = {_fmt(r.launch_summed_ms)}")
        print(f"sync_summed_ms     = {_fmt(r.synchronization_summed_ms)}")
        print(f"alloc_summed_ms    = {_fmt(r.allocation_summed_ms)}")
        print(f"other_summed_ms    = {_fmt(r.other_summed_ms)}")
    elif view == "memory":
        m = ev.memory
        if (
            m.allocation_count is None
            and m.allocated_bytes is None
            and m.churn_ratio is None
            and m.host_device_copy_ms is None
        ):
            print("unavailable")
        else:
            print(f"allocation_count    = {m.allocation_count}")
            print(f"free_count          = {m.free_count}")
            print(f"allocated_bytes     = {m.allocated_bytes}")
            print(f"churn_ratio         = {_fmt(m.churn_ratio, 6)}")
            print(f"host_device_copy_ms = {_fmt(m.host_device_copy_ms)} (summed)")
    elif view == "communication":
        c = ev.communication
        if c.calls is None and c.total_ms is None:
            print("unavailable")
        else:
            print(f"calls        = {c.calls}")
            print(f"total_ms     = {_fmt(c.total_ms)}")
            print(f"overlap_ms   = {_fmt(c.overlap_ms)}")
            print(f"overlap_ratio= {_fmt(c.overlap_ratio, 6)}")
            for name, count in sorted((c.collectives or {}).items()):
                print(f"  {name}: {count}")
    elif view == "diagnosis":
        print("--- observations (what is happening) ---")
        for o in build_all_observations(ev):
            print(f"{o.id} [{o.kind}] {o.metric_key}={o.value} evidence={', '.join(o.evidence_ids) or '-'}")
            print(f"      {o.note}")
        print("--- diagnosis candidates (conservative; why needs the agent) ---")
        if not ev.diagnosis.candidates:
            print("unknown")
        for c in ev.diagnosis.candidates:
            print(f"{c.class_:<28} confidence={c.confidence:<7} evidence={', '.join(c.evidence_ids)}")
    elif view == "provenance":
        for r in ev.provenance:
            agg = r.aggregation or "-"
            query = r.query or "-"
            rank = r.rank if r.rank is not None else "-"
            print(
                f"{r.id} | {r.source_file} | {r.source_type} | rank={rank} | "
                f"parser={r.parser} | query={query} | aggregation={agg} | unit={r.unit}"
            )
    else:
        print(f"error: unknown view {view!r}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vllm_omni.profiling",
        description="Offline profiling evidence framework: raw artifacts -> OptimizationEvidence",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="analyze a profiler output directory")
    p_analyze.add_argument("--backend", choices=("cuda", "ascend", "auto"), required=True)
    p_analyze.add_argument("--input", required=True, help="profiler output directory or trace file")
    p_analyze.add_argument("--output", required=True, help="output directory for evidence files")
    p_analyze.set_defaults(func=cmd_analyze)

    p_query = sub.add_parser("query", help="query views from an evidence.json")
    p_query.add_argument("--evidence", required=True, help="path to evidence.json")
    p_query.add_argument("--view", choices=QUERY_VIEWS, required=True)
    p_query.add_argument("--top", type=int, default=20)
    p_query.set_defaults(func=cmd_query)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (BackendDetectionError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
