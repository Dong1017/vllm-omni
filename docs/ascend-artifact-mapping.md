# Ascend artifact → OptimizationEvidence mapping (M2-T0)

Source of truth: real torch_npu profiler output at `outputs/h3_16die_validation_20260922/ref2va_15s_280ts/` (ASCEND_PROFILER_OUTPUT, produced 2026-08-18), plus a second MindStudio/msprof format family at `npu_lingbot/profiling_csv/` (op_summary/op_statistic/api_statistic/task_time). Survey is observational; nothing below is guessed — each row cites the file it came from.

## rank vs device identity (P0-1, verified 2026-09-30)

`Device_id` columns are **device numbers, not distributed ranks**. Verified against the trace DB (read-only query):

```text
RANK_DEVICE_MAP: rankId=0, deviceId=8
NPU_INFO:        id=8, name='Ascend910B'
```

Therefore this sample's process rank is **0**, running on device 8 (Ascend910B). The CSV family carries no rank column → adapters set `run.rank=None` and `run.device_id="<Device_id>"`. True rank requires the big DB (`RANK_DEVICE_MAP`) or `profiler_metadata.json` parallel-group info — planned with the dispatch-chain increment. Earlier evidence that wrote `rank=8` was wrong and has been regenerated.

## Artifact inventory (torch_npu family)

```text
ASCEND_PROFILER_OUTPUT/
├── step_trace_time.csv        (293 B)   step timeline breakdown
├── op_statistic.csv           (5.9 KB)  device op-type aggregation with Ratio(%)
├── kernel_details.csv         (59 MB)   per device task: aclnn name, core type, start/duration, shapes
├── operator_details.csv       (48 MB)   per PyTorch op: host vs device self/total durations, call stack
├── api_statistic.csv          (11 KB)   host-side API statistics, Level=acl → CANN API layer
├── analysis.db                (372 KB)  StepTraceTime + CommAnalyzer{Time,Bandwidth,Matrix}
├── ascend_pytorch_profiler_0.db (459 MB) full trace DB (see dispatch chain below)
├── trace_view.json            (1.6 GB)  chrome-like trace — NOT loaded by the framework (AC-30)
├── profiler_metadata.json / profiler_info_0.json   env, HCCL groups, profiler config
```

## Data-quality findings (affect parsing)

- `kernel_details.csv` Start Time field carries a stray trailing `\t` before the delimiter (`1787060308130367.891\t,`) — parser strips whitespace on all values.
- `operator_details.csv` has no call-count column; per-op `calls` is unavailable from this file.
- `profiler_info_0.json` records `record_shapes: false` — shapes exist only in device-side files (kernel_details / op_summary), sourced from CANN.
- Timestamps are wall-clock epoch in us (e.g. 1787060308130367.891); durations in us.

## Mapping → common schema (MVP adapter scope)

| Ascend source | Evidence destination | Notes / 口径 |
|---|---|---|
| `step_trace_time.csv` `Stage` | `workload.wall_ms` | measured, us→ms |
| `Step` `Computing` | `timeline.compute_ms` | CANN's own classification; recorded as measured with provenance |
| `Communication` | `timeline.communication_ms` | verified: Communication = NotOverlapped + Overlapped (exact in sample) |
| `Communication(Not Overlapped)`, `Overlapped` | `communication.total_ms/overlap_ms` | overlap_ratio left None: denominator semantics not documented → no guessing |
| `Free` | `timeline.exposed_non_device_busy_gap` | CANN's exposed idle |
| `Bubble`, `Preparing` | `backend_metrics.ascend.*` | step-trace specifics, not common semantics |
| — | `timeline.device_busy_ms` | **unavailable from CSVs** (no reliable busy union at step level); derived later from DB `OVERLAP_ANALYSIS` (T2) |
| `kernel_details.csv` rows | `operators[]` layer=`device`, grouped by `Name` | calls=count, total/avg/median from `Duration(us)`, common shapes from `Input Shapes`; start/end → device-busy **union** (real intervals) |
| `kernel_details.csv` `Accelerator Core` | `backend_metrics.ascend.core_type_*` | AI_CORE / AI_VECTOR_CORE / MIX_AIC / AI_CPU counts and time shares, never mapped to CUDA counters |
| `operator_details.csv` rows | `operators[]` layer=`framework`, grouped by `Name` | `total_host_ms`←Host Total, `total_device_ms`←Device Total; `calls` unavailable (no count column) |
| `api_statistic.csv` rows with `Level=acl` | `operators[]` layer=`runtime`, grouped by `API Name`; `runtime.api_summed_ms` = sum of all acl Time | P0-2: each acl API classified into exactly one of launch/synchronization/allocation/other by name keyword (`launch` / `synchronize` / `malloc`,`free`); partitions sum to api_summed. `runtime_api_summed > wall` is normal (multi-thread overlap) — never a bottleneck fraction |
| `op_statistic.csv` rows | `backend_metrics.ascend.op_statistic` (verbatim) + `profiler_ratio` candidates | its `Ratio(%)` denominator is undocumented → per decision D2 it must NOT enter `device_time_fraction` |
| `analysis.db` `CommAnalyzerTime` | `communication.collectives` / per-collective durations (T2 DB adapter) | hccl_op_name, elapse/transit/wait/sync/idle per call |
| `analysis.db` `StepTraceTime` | same as step_trace_time.csv | see data note below: stage/free differ from CSV |
| `profiler_metadata.json` | provenance / run metadata | HCCL groups, env; parallel-group info is a future rank source |
| `Bubble` / `Preparing` (step_trace columns) | `backend_metrics.ascend.step_trace_bubble_us` / `step_trace_preparing_us` | verbatim sums, no common-schema mapping (P1-4A) |

## Dispatch chain (M2-T2, real sample present)

`ascend_pytorch_profiler_0.db` (459 MB, schema inspected read-only) contains the full chain:

```text
PYTORCH_API(name, startNs/endNs, connectionId, inputShapes)   → framework
CANN_API(name, startNs/endNs, connectionId)                   → runtime
TASK(startNs/endNs, deviceId, streamId, globalTaskId)         → device
CONNECTION_IDS / STRING_IDS / ENUM_*                          → join + name tables
COMPUTE_TASK_INFO(globalTaskId → shapes, blockNum)            → task details
COMMUNICATION_OP / COMMUNICATION_TASK_INFO                    → HCCL details
OVERLAP_ANALYSIS(startNs, endNs, type)                        → real busy-union source
```

`connectionId` joins the three layers (one PyTorch API → CANN API → one or more TASKs). Names are likely stored as STRING_IDS references (verified during T2 implementation, not assumed).

## Denominator discipline (decision D2 applied)

- `Ratio(%)` in op_statistic: denominator undocumented → `profiler_ratio` only.
- `device_time_fraction` (if computed) uses only framework-derived denominators with `fraction_denominator` recorded.
- No field maps `Ascend MTE/UB` onto any CUDA counter or onto common `timeline` semantics (decision D3).

## Data note: analysis.db vs step_trace_time.csv discrepancy (verified 2026-09-29)

The same real sample reports different numbers in the two sources:

| metric | step_trace_time.csv | analysis.db StepTraceTime |
|---|---:|---:|
| stage | 111048989.75 us | 108815501.753 us |
| computing | 85704020.577 us | 85704020.577 us (identical) |
| communication | 4504063.418 us | 4504063.418 us (identical) |
| free | 21471450.19187618 us | 19237962.204 us |

`computing`/`communication` match exactly; `stage`/`free` differ by ~2%. Cause unknown; possible causes include bubble/preparing accounting and exclude-receive differences between the two export paths — no evidence yet to pick one, so none is asserted. The framework records whichever single source it reads, never merges or cross-corrects. Both variants are frozen as golden facts with their source named.

## Golden facts (to freeze after first adapter run on this sample)

Candidate facts (to be manually confirmed and frozen as regression fixtures in `tests/profiling/fixtures/`):

1. `wall_ms` from Stage = 111048989.75 us → 111048.99 ms (step_trace_time.csv row 1)
2. `communication_ms` = 4504063.418 us → 4504.06 ms; Overlapped 630544.446 + NotOverlapped 3873518.972 = total (exact identity holds)
3. `exposed_non_device_busy_gap` = Free = 21471450.19 us → 21471.45 ms
4. top op_statistic row: `alltoallAicpuKernel`, AI_CPU, count=200, total=51294753.473 us, Ratio=35.982%
5. device task top by total from kernel_details (aggregated) — extract during T1 run
6. framework top by Device Total from operator_details — extract during T1 run
