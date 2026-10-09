# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# CLI（T16/T17）：analyze 与 query 两个子命令。
# 失败必须显式报错退出非 0（AC-31），不静默生成空报告。

from __future__ import annotations

import argparse
import hashlib
import json
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
from vllm_omni.profiling.backends.ncu_csv import parse_ncu_csv
from vllm_omni.profiling.discovery import BackendDetectionError, detect_backend, discover
from vllm_omni.profiling.distributed import (
    build_distributed_summary,
    build_rank_timeline_from_ascend_db,
    resolve_clock_validation,
    write_distributed_outputs,
)
from vllm_omni.profiling.provenance import ProvenanceStore, compute_run_id
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
    # final-review fail-fast：多 CUDA trace + 单 NCU capture 的归属未定义
    if args.ncu_csv and len(traces) > 1:
        raise BackendDetectionError(
            f"ncu-csv: {len(traces)} CUDA traces found under {input_path}; "
            "per-rank/capture association is not defined until M4.2b — "
            "analyze one trace at a time"
        )
    for trace in traces:
        ev, store = analyze_cuda_torch_profiler(trace)
        ev = enrich(ev)
        # 单 rank 输入用标准文件名；多 rank 用带 rank 后缀的文件名，
        # 禁止跨 rank 相加 duration（AC-05），跨 rank 聚合属于后续版本
        if len(traces) == 1:
            target = output_dir
        else:
            rank = ev.run.rank
            target = output_dir / (f"rank{rank}" if rank is not None else "rank_unknown")
        if getattr(args, "ncu_csv", None):
            # P0-3：NCU evidence 独立 run_id namespace（由 NCU CSV sha 派生，
            # 不复用 torch-trace run_id）；record 追加进 ev.provenance，
            # hardware.provenance_ref 与 metric_evidence 均引用该 namespace
            hw_sha = hashlib.sha256(Path(args.ncu_csv).read_bytes()).hexdigest()
            hw_run_id = compute_run_id([hw_sha], backend="cuda", parser="cuda.ncu_csv")
            hw_store = ProvenanceStore(run_id=hw_run_id)
            rec_hw = hw_store.add(
                source_file=Path(args.ncu_csv).name,
                source_type="ncu_csv",
                parser="cuda.ncu_csv",
                unit="mixed",
                aggregation=None,
                source_sha256=hw_sha,
            )
            hw_ref = hw_store.ref(rec_hw.id)
            ev.hardware = parse_ncu_csv(Path(args.ncu_csv))
            ev.hardware.provenance_ref = hw_ref
            ev.provenance.append(rec_hw)
            for e in ev.hardware.entries:
                ev.metric_evidence[f"hardware:{e.scope_id}:{e.metric_name}"] = [hw_ref]
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
        for obs_item in build_all_observations(ev):
            print(
                f"{obs_item.id} [{obs_item.kind}] {obs_item.metric_key}={obs_item.value} "
                f"evidence={', '.join(obs_item.evidence_ids) or '-'}"
            )
            print(f"      {obs_item.note}")
        print("--- diagnosis candidates (conservative; why needs the agent) ---")
        if not ev.diagnosis.candidates:
            print("unknown")
        for cand in ev.diagnosis.candidates:
            print(f"{cand.class_:<28} confidence={cand.confidence:<7} evidence={', '.join(cand.evidence_ids)}")
    elif view == "provenance":
        for rec in ev.provenance:
            agg = rec.aggregation or "-"
            query = rec.query or "-"
            rank = rec.rank if rec.rank is not None else "-"
            print(
                f"{rec.id} | {rec.source_file} | {rec.source_type} | rank={rank} | "
                f"parser={rec.parser} | query={query} | aggregation={agg} | unit={rec.unit}"
            )
    else:
        print(f"error: unknown view {view!r}", file=sys.stderr)
        return 2
    return 0


def cmd_analyze_distributed(args: argparse.Namespace) -> int:
    """M4.3：多 rank 空泡分析（backend-neutral reducer + Ascend rank-timeline adapter）。"""
    rank_timelines = []
    seen_paths: set[Path] = set()
    if args.rank_input:
        for item in args.rank_input:
            rank_str, _, path_str = item.partition("=")
            if not rank_str.isdigit() or not path_str:
                raise ValueError(f"rank-input must be N=<db path>, got {item!r}")
            path = Path(path_str)
            if path in seen_paths:
                raise ValueError(f"distributed: repeated rank DB path {path}")
            seen_paths.add(path)
            timeline = build_rank_timeline_from_ascend_db(path)
            if timeline.rank_id != int(rank_str):
                raise ValueError(
                    f"distributed: --rank-input {rank_str}={path} but the DB's RANK_DEVICE_MAP "
                    f"says rankId={timeline.rank_id}; the DB carries its own rank identity"
                )
            rank_timelines.append(timeline)
    if args.input:
        input_path = Path(args.input)
        result = discover(input_path, "ascend")
        for db_path in sorted(result.ascend.profiler_db):
            if db_path in seen_paths:
                continue
            seen_paths.add(db_path)
            rank_timelines.append(build_rank_timeline_from_ascend_db(db_path))
    if not rank_timelines:
        raise BackendDetectionError(
            "distributed: no rank DBs provided; pass --rank-input N=<db path> "
            "(repeatable) or --input <dir containing rank DBs>"
        )
    clock_metadata = json.loads(Path(args.clock_metadata).read_text(encoding="utf-8")) if args.clock_metadata else None
    clock_validation, clock_meta = resolve_clock_validation(rank_timelines, args.shared_clock, clock_metadata)
    summary = build_distributed_summary(
        rank_timelines, clock_validation, clock_meta, Path(args.windows) if args.windows else None
    )
    written = write_distributed_outputs(summary, Path(args.output), intervals_csv=args.intervals_csv)
    common_window = next(w for w in summary["windows"] if w["window"]["name"] == "common_coverage_window")
    print(
        f"analyzed-distributed ranks={summary['rank_ids']} clock={clock_validation} "
        f"common_no_device_task_ms={common_window['distributed']['common_no_device_task']['duration_ms']:.3f} "
        f"common_no_compute_ms={common_window['distributed']['common_no_compute']['duration_ms']:.3f}"
    )
    for key in sorted(written):
        print(f"  wrote {written[key]}")
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
    p_analyze.add_argument(
        "--ncu-csv",
        default=None,
        help="optional NCU --csv export (long format) to attach as hardware evidence (CUDA path)",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    p_query = sub.add_parser("query", help="query views from an evidence.json")
    p_query.add_argument("--evidence", required=True, help="path to evidence.json")
    p_query.add_argument("--view", choices=QUERY_VIEWS, required=True)
    p_query.add_argument("--top", type=int, default=20)
    p_query.set_defaults(func=cmd_query)

    p_dist = sub.add_parser(
        "analyze-distributed",
        help="multi-rank common no-task / no-compute analysis (M4.3, Ascend trace DBs)",
    )
    p_dist.add_argument("--backend", choices=("ascend",), default="ascend")
    p_dist.add_argument(
        "--input",
        default=None,
        help="directory containing multi-rank Ascend profiler DBs (rank identity read from each DB)",
    )
    p_dist.add_argument(
        "--rank-input",
        action="append",
        default=None,
        metavar="N=<db path>",
        help="explicit rank DB (repeatable); the DB's RANK_DEVICE_MAP rankId must match N",
    )
    p_dist.add_argument(
        "--shared-clock",
        action="store_true",
        help="explicit assertion that rank timestamps share one clock origin (recorded as "
        "clock_validation=explicit_assertion; without it or clock metadata the command fails)",
    )
    p_dist.add_argument(
        "--clock-metadata",
        default=None,
        help="optional JSON with non-empty 'evidence' string -> clock_validation=validated",
    )
    p_dist.add_argument(
        "--windows",
        default=None,
        help='optional JSON of named phase windows: {"name": [start_ns, end_ns], ...}',
    )
    p_dist.add_argument("--output", required=True, help="output directory for distributed summary files")
    p_dist.add_argument(
        "--intervals-csv", action="store_true", help="also write distributed_intervals.csv (common layers)"
    )
    p_dist.set_defaults(func=cmd_analyze_distributed)
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
