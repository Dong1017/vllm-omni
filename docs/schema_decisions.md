# Profiling evidence framework — schema decisions

Companion to `docs/profiling-framework-design.md`. Each decision below is an architecture decision with its rationale and the tests that guard it. Current schema version: **0.4** (0.1/0.2/0.3 remain readable via explicit migration).

## D5. Runtime API time is a mutually exclusive partition (v0.3, P0-2)

`dispatch_ms` (v0.1/0.2) summed ALL runtime API events — including malloc/free and synchronize calls — so `dispatch_ms=151.7s > wall=111.0s` was observed on real data (multi-thread overlap makes summed API time exceed wall legitimately). Replaced by:

```text
runtime.api_summed_ms            = all runtime API events summed
runtime.launch_summed_ms         + runtime.synchronization_summed_ms
                                 + runtime.allocation_summed_ms
                                 + runtime.other_summed_ms       (mutually exclusive, sum == api_summed)
```

Rules: every event classified into exactly one category (keyword rules per backend, documented in adapters/mapping doc); total and categories share one source/time domain; `api_summed / wall` is FORBIDDEN as a bottleneck fraction; `host_dispatch_bound` requires exposed-gap correlation or equivalent critical-path evidence (M3 rule constraint).

## D6. Operator time semantics are per-layer explicit (v0.3, P0-3)

Nested framework attribution cannot be compared with per-event sums. `OperatorEvidence` therefore carries:

```text
framework layer:  host_self_ms, host_inclusive_ms, device_self_ms, device_inclusive_ms
runtime/device:   summed_host_ms, summed_device_ms
```

Mechanical rule for M3: TopK/fraction only over values with equal semantics — framework TopK by `device_self_ms` only; `device_inclusive_ms` (e.g. Ascend `operator_details` Device Total ≈ 397s vs wall 111s on the real sample) is attribution-only. Migration 0.2→0.3 is backend-aware: CUDA's external-id kernel sum → `device_self_ms` (leaf attribution); Ascend's `Device Total` → `device_inclusive_ms`.

## D7. rank and device_id are distinct (v0.3, P0-1)

`Device_id` columns in Ascend CSVs are device numbers, not distributed ranks. Verified on the real sample via the trace DB: `RANK_DEVICE_MAP (rankId=0, deviceId=8)`. `run.rank` is set only from independent rank evidence; otherwise `None`. Device number lives in `run.device_id`. Rank provenance source: big-DB `RANK_DEVICE_MAP` (planned increment) or `profiler_metadata.json` parallel groups.

## D8. Writer-side validation (v0.3, P0-4)

Schema validation runs at construction (`__post_init__`), serialization (`OptimizationEvidence.__post_init__` tree walk), and deserialization (`_build`) — an invalid in-memory object (e.g. `DiagnosisCandidate(class_="made_up_class", confidence="0.82")`) cannot be constructed or serialized. Reader-side checks remain as defense-in-depth.

## D9. Observation → Diagnosis contract (v0.4, M3, revised at MVP-gate audit)

Architecture (ckpt-2/ckpt-3 audits): `Evidence → deterministic Observation → conservative DiagnosisCandidate`. Framework answers *what is happening?*; the optimization agent answers *why / what to change*. Quality metric: **false diagnoses avoided**, not diagnosis classes covered.

**Observation layer** (module `observation.py`) — the only place raw metrics are interpreted:

- Kind vocabulary: `exposed_gap_present`, `runtime_launch_heavy`, `runtime_sync_heavy`, `runtime_allocation_heavy`, `runtime_other_heavy`, `communication_exposed`, `communication_present_no_overlap_breakdown`, `compute_activity_dominant`, `device_busy_dominant`, `no_evidence_available`, `no_salient_observation`.
- Every observation binds exact metric keys + global evidence refs (`run_id:ev_id`, P1-1); `*_bound` vocabulary is forbidden here (P1-3).
- Observe thresholds: gap/wall ≥ 10%, category/api_summed ≥ 20%, busy/wall ≥ 75%, compute/busy ≥ 75%.
- **Same-source discipline applies to every cross-metric ratio** (MVP-gate Fix 1/2): `gap/wall`, `busy/wall`, `(total−overlap)/wall` are computed only when `metric_evidence` intersection proves all inputs share a source/time domain. `communication_exposed` additionally cites ALL its inputs (total + overlap + wall). Breakdown missing or source-incompatible → `communication_present_no_overlap_breakdown`, never `communication_exposed`.

**Diagnosis layer** (module `diagnosis.py`) — consumes Observations only (P0-1):

- The **only** supported `*_bound` diagnosis in M3 is `communication_bound`: requires the `communication_exposed` observation (which already carries all input refs), mechanical same-source verification over total/overlap/wall/exposed_gap, not-overlapped ≥ 15% wall AND ≥ 50% of gap. 2× both lines → `high`, else `medium`.
- `host_dispatch_bound` / `synchronization_bound` / `allocation_bound` are **not output** in M3: summed runtime-API category shares cannot prove causality for the exposed gap (overlapping activity outside the gap). They await M4 temporal-correlation evidence (API interval ∩ exposed-gap interval via Nsight/msprof timelines).
- `compute_bound` is **not output** in M3: compute-activity dominance does not prove hardware-throughput bottleneck. It awaits M4 hardware evidence (NCU/roofline/DRAM/SM for CUDA; AIC/AIV/MTE/Cube for Ascend).
- `memory_bound` / `load_imbalance`: no evidence sources exist → no rules.
- No candidate justified → `unknown` with `confidence=low` and the observation's evidence refs — a valid, honest output.
- Hard constraint (D5): runtime-API-summed/wall is never a bottleneck fraction.

**Hotspot ranking** (module `analysis.py`): per-layer, same-semantics TopK with explicit `fraction_denominator` (D2 extended); inclusive values never enter ranking or fractions; row-level evidence ids are normalized to global `run_id:ev_id` form.

**v0.4 fields**: `metric_evidence` (P1-1) and `EvidenceRecord.depends_on` (P1-2). Migrations 0.1/0.2/0.3 → 0.4 leave these empty — historical files cannot gain bindings without re-analysis, and none are fabricated. Summary rendering resolves per-metric sources through `metric_evidence` + provenance records.

Guarded by: `test_observation.py` (same-source gating, naming discipline, split insufficient cases), `test_diagnosis.py` (the five checkpoint cases, communication-only bound, mechanically-enforced same-source), `test_serialization.py` (migration chain + P0-5 mutation rejection).

## D10. Timeline correlation upgrades heavy observations to bounds (v0.5, M4.1, revised at final review)

ckpt-3 审计 P0-2 留下的缺口：`runtime_launch_heavy` 等 summed 占比观察不能证明该分类 API 时间真的与 exposed gap 相关。M4.1 引入时间相关性证据；最终 review 修订两点口径。

- **Schema**：`RuntimeStats.gap_overlap_ms: dict[str, float] | None` —— 各 runtime 分类 API 与 device exposed-gap 区间（trace 窗口 − device busy union）的交集时长（ms）。仅当 source 自带时间轴（当前：CUDA torch-profiler trace；后续：Ascend 大库 CANN_API 区间）时由 adapter 计算；CSV 类无时间轴 source 保持 `None`，禁止用 summed 值冒充。
- **P0-1 分子口径**：`overlap = Σ per-event (event ∩ gap-union)`（逐事件求交再求和，summed 语义），与分母 `*_summed_ms`（逐事件时长求和）一致；禁止先把同分类 runtime 区间 union 再求交——并发重叠事件会被合并少算。
- **P0-2 双条件**：相关性（correlation）与影响（coverage）是两个独立条件，缺一不判：
  - `correlation = gap_overlap_ms[cat] / <cat>_summed_ms`（分类时间落 gap 的比例）≥ 0.50；
  - `gap_coverage = gap_overlap_ms[cat] / exposed_non_device_busy_ms`（该分类实际解释的 gap 比例）≥ 0.10。
  - 大 gap + 极少量 100% 相关的 API 时间 → coverage 不足 → 不构成 `*_bound`（有负例测试）。
- **观察层**：`runtime_launch_gap_correlated` / `runtime_sync_gap_correlated` / `runtime_allocation_gap_correlated` —— heavy（share ≥ 20%）且 correlation ≥ 30%，且发出前机械校验同源（分类时长与 `runtime.gap_overlap_ms.<cat>` 同记录；gap 与 wall 同记录）。
- **诊断层**：`host_dispatch_bound` / `synchronization_bound` / `allocation_bound` 恢复输出，证据链三条观察缺一不可，外加三层机械同源校验（分类↔gap_overlap、gap_overlap↔exposed、exposed↔wall）。
- **metric key**：按分类细分 `runtime.gap_overlap_ms.launch` / `.synchronization` / `.allocation` / `.other`；`metric_evidence` 同时绑定 runtime 区间记录与 device/gap 派生记录（完整依赖，P1-2）。
- Ascend CSV 路径无 API 时间轴 → 相关性观察缺席 → 三个 bound 仍然不输出（保守行为不变，等 M4 大库 CANN_API 区间接入）。
- Timeline-domain 前提：记录级 same-source（metric_evidence 交集）是 timeline domain 的代理；成立前提是单 adapter 单文件单 clock（torch profiler 统一时钟），跨文件/跨 clock 混用在 adapter 结构上不可达。

Guarded by: `test_intervals.py`（intersect / total_event_overlap 重叠事件 summed 语义）、`test_cuda_torch_profiler.py`（gap_overlap 数值、正/负升格链、覆盖不足负例）、`test_observation.py`（同源门控、命名纪律）、`test_diagnosis.py`（coverage 负例）、`test_serialization.py`（0.4→0.5 additive migration，历史文件 gap_overlap=None 不编造）。

## D11. Ascend trace-DB correlation contract (v0.5, M4.1b)

The `ascend_pytorch_profiler_0.db` main DB exposes the three-layer chain over a **single ns capture clock**:

```text
PYTORCH_API(type=50001, name→STRING_IDS=aten::)  → framework
CANN_API(name→STRING_IDS=acl*)                   → runtime
TASK(globalTaskId→COMPUTE_TASK_INFO.name)        → device
```

All three join on `connectionId` (verified on the real 459MB sample: TASK.connectionId ⊆ CANN_API.connectionId at 297330/297332 distinct keys). Rules:

- Correlation/linkage uses connectionId equality only; no name or timestamp-proximity guessing.
- Wall comes from `SESSION_TIME_INFO` (absolute session bounds); exposed gap = session window − TASK busy union. The gap window must be the **absolute** session interval — deriving it from the relative wall duration misplaces the window against absolute-interval data (found during checkpoint review: overlap was identically zero).
- CANN categories use the same keyword rule as the CSV adapter; `gap_overlap_ms` per category via per-event overlap (summed semantics, D10).
- `rank` comes from `RANK_DEVICE_MAP.rankId`; `device_id`/`device` from `deviceId`/`NPU_INFO`. Multi-rank DBs fail fast.
- Large-DB safety: gap derivation via one-pass `complement()`; overlap via bisect-stepped `total_event_overlap()` — a quadratic per-removal subtract loop hung on the real ~200k-interval busy union.
- Conservative degradation: no timeline/tables → no correlation observations, no `*_bound`; the CSV path keeps `gap_overlap_ms=None` (no API timeline there).

## D12. Hardware evidence contract (v0.6, M4.2a)

`OptimizationEvidence.hardware` carries an `ncu --csv` long-format capture as typed evidence:

```text
HardwareEvidence(backend, source, provenance_ref, entries[])
HardwareMetricEntry(scope=kernel, metric_name, metric_unit, metric_value,
                    canonical_value, canonical_unit)
```

Discipline (inherited from the Yotta feasibility final review, revised at the M4.2a gate):

- The canonical hardware-counter source is the raw NCU `(metric_name, metric_unit, metric_value)` tuple; `metric_unit` is preserved verbatim because ncu auto-scales value/unit and a unitless value is uninterpretable.
- **Unit scaling uses SI factors per the Nsight Compute CLI spec: byte-family K/M/G = 1000/1e6/1e9 (NOT 1024).** The collector contract for M4.2b explicitly passes `--print-units base` (ncu `--csv` implies it by default; explicit is more stable).
- `canonical_*` fields hold ONLY unit conversion (SI factors). Unparsable values (`n/a`) and unknown units leave `canonical_*` as None — unavailable, never 0.
- Metric value is the parsed numeric (thousand separators removed); the raw comma-format string is not preserved.
- **Unitless interpretation is registry-gated**: an empty unit is interpretable only for metrics in `METRIC_REGISTRY` (the curated 20-counter shortlist with architecture fallbacks); unknown metric + empty unit stays unresolved.
- **Observations are registry-semantic-driven**: only the saturation family (dram/sm/l1tex/lts throughput, tensor-pipe active, issue-active pct counters) emits `hardware_resource_saturation` at ≥ 80% pct. Hit-rate/occupancy/counters-descriptive families emit nothing (a 90% hit rate is not a problem signal). `compute_bound` / `memory_bound` remain closed until M4.2b validates counters, attribution, and units on a real capture.
- Kernel→operator attribution is deliberately out of scope for M4.2a.

## Terminology note: run_id is an analysis identity

`run_id` is content-addressed: `hash(sorted source hashes + backend + parser)`. Two identical captures would produce the same `run_id`. It identifies an **analysis/evidence set**, not a capture instance. If the future Optimization IR needs capture-instance identity, that becomes a separate field — not extended now.

## D1. Operator abstraction is three-layer, not CUDA's two-layer

```text
framework_op            e.g. aten::layer_norm (CUDA) / PyTorch API (Ascend)
    ↓
runtime_op              e.g. cudaLaunchKernel (CUDA) / aclnnXxx CANN API (Ascend)
    ↓
device_task(s)          e.g. triton_xxx kernel / AI Core TASK   (one-to-many)
```

Represented in `OptimizationEvidence.operators[]` as flat rows with:

- `layer`: `framework` | `runtime` | `device`;
- `parent_id`: id of the row one layer up; multiple rows may share a parent (one-to-many);
- aggregated rows fill `parent_id` only when the attribution is unique; otherwise `None`.

Rules:

- The runtime layer may be entirely unavailable; this is normal, not an error.
- CUDA must not invent runtime semantics it does not have; CUDA genuinely has `cuda_runtime` events, so its runtime layer is populated from them.
- Ascend's `PyTorch API → CANN API → TASK` chain must keep its three levels; collapsing it into one duration is forbidden.
- Rows are name-aggregated; instance-level trees belong to a future Optimization IR, not this contract.

Guarded by: `test_cuda_torch_profiler.py` (three-layer rows, parent linkage, one-to-many), `test_schema.py` (layer enum validation), `test_serialization.py` (0.1 migration infers layer).

## D2. Shape ratio fields are split by denominator

A ratio without an explicit denominator cannot be compared across backends. `workload_fraction` (v0.1) was CUDA's "summed device time of this (operator, shape) / summed device time of all device tasks". It is replaced by:

| field | semantics |
|---|---|
| `calls` | occurrences (count) |
| `device_time_fraction` | computed by the framework, only when the denominator is explicit |
| `fraction_denominator` | required whenever `device_time_fraction` is set; describes the denominator |
| `profiler_ratio` | verbatim ratio reported by the profiler (e.g. Ascend `op_statistic` `Ratio(%)`), never re-interpreted |

Invariant: `device_time_fraction is not None ⇔ fraction_denominator is not None`. Unsupported fields stay `unavailable` (None). A profiler ratio whose denominator differs from the framework denominator must land in `profiler_ratio`, never in `device_time_fraction`.

Guarded by: `test_schema.py` (invariant + passthrough), `test_cuda_torch_profiler.py` (denominator set, fraction values).

## D3. Common timeline excludes memory semantics

The common `timeline` keeps only hardware-neutral, high-level intervals:

```text
device_busy_ms, exposed_non_device_busy_ms, compute_ms, communication_ms, unknown_ms
```

`memory_ms` (v0.1, CUDA memcpy/memset union) was removed: mapping `CUDA memcpy == Ascend MTE/UB transfer` is a cross-hardware pseudo-mapping. Memory-adjacent evidence lives elsewhere:

- `memory.host_device_copy_ms` — host↔device copy, summed (backend-neutral concept);
- `backend_metrics.cuda.*` — DRAM/L2 etc.;
- `backend_metrics.ascend.*` — MTE/UB etc.

If a backend cannot reliably compute a common field, it stays `unavailable`.

Guarded by: `test_serialization.py` (0.1 migration moves `timeline.memory_ms` to `backend_metrics.cuda.legacy_memory_union_ms` because the union denominator differs from the summed semantics of `host_device_copy_ms`; `runtime.copy_ms` moves to `memory.host_device_copy_ms` as its semantics match), `test_cuda_torch_profiler.py`.

## D4. Evidence identity is valid across runs

Downstream optimization actions will reference evidence long after the profiling run. Every `EvidenceRecord` carries:

- `run_id` — deterministic, derived from `sha256(sorted source hashes) + backend + parser` (12 hex chars); identical inputs produce identical `run_id` (AC-08);
- `source_sha256` — content hash of the source artifact, streamed (AC-30);
- existing fields: source_file, rank, parser, query, aggregation, unit.

`RunInfo.run_id` carries the same value. Global reference form: `run_id:ev_000001` (`ProvenanceStore.ref`). This supports, without implementing it now:

```text
Optimization Action → evidence reference → exact profiling run → exact raw artifact
```

Guarded by: `test_provenance.py` (run_id propagation, ref form, streaming hash), `test_cuda_torch_profiler.py` (run_id stability across two parses).

## Migration rules (0.1 → 0.2)

Applied explicitly by `_migrate_0_1` in `schema.py`; no silent drops:

| v0.1 field | v0.2 destination | note |
|---|---|---|
| `operators[].op_type` | `operators[].layer` | `aten_op→framework`, `kernel→device`; `parent_id=None` |
| `timeline.memory_ms` | `backend_metrics.cuda.legacy_memory_union_ms` | union denominator; renaming to the summed field would lie about the metric |
| `runtime.copy_ms` | `memory.host_device_copy_ms` | both summed, semantics match |
| `shapes[].workload_fraction` | `shapes[].device_time_fraction` + `fraction_denominator="summed_device_time_all_device_tasks"` | v0.1 CUDA writer's only denominator |
| `run.run_id`, `evidence.run_id/source_sha256` | absent → `None` | old files simply carry no cross-run identity |

Unknown schema versions fail explicitly (AC-31).
