# Profiling evidence framework — T00 design note

Status: T00 output (repository inventory + landing decisions). Pre-implementation note; a module design doc following the `docs/design/module/*.md` frontmatter convention will be authored when the module lands.

- Base SHA (frozen at T00, 2026-09-29): `235c43032646bce93950b6746dd7fe6b6d8e06ba` (upstream/main). Do not auto-track main during milestones.
- Working branch: `profiling/evidence-framework` (dedicated worktree). This framework shares no dependency with Noisy Chunk PP; never develop it on the `noisy-pp-integration` branch or modify that worktree.
- Specs: task spec lives outside this repo (`profiling-agent/SPEC.md`, `ACCEPTANCE.md`, `TODO.md`, plus the 2026-09-29 execution rulings). This note records only what the repo needs.

## 1. Landing spot

New package `vllm_omni/profiling/` (offline analysis framework; spec section 6 structure). Verified: no on-disk collision (`profiling/`, `schema.py`, `discovery.py`, `cli.py` do not exist anywhere in the repo).

Boundary rule:

| Package | Responsibility |
|---|---|
| `vllm_omni/profiler/` (existing, singular) | Runtime trace **collection** (OmniTorchProfilerWrapper, platform dispatch). Unchanged by this framework. |
| `vllm_omni/profiling/` (new) | Offline **analysis**: raw artifacts → `OptimizationEvidence` → reports/query views. |

Precedent for package/scripts same-name coexistence: `vllm_omni/benchmarks/` vs top-level `benchmarks/`. The singular/plural split here mirrors collection vs analysis, and both `docs/design/module/profiling.md` (collection) and the new module doc must cross-reference each other.

Entry points and conventions:

- CLI: `python -m vllm_omni.profiling analyze|query` via `__main__.py`; argparse with kebab-case flags, `main()` + `if __name__ == "__main__":` (convention examples: `benchmarks/tts/bench_tts.py`, `tools/nightly/generate_nightly_perf_excel.py`).
- Every new `.py` file needs the SPDX header (pre-commit `check-spdx-header`); docs pass markdownlint.
- Tests: `tests/profiling/` (package-mirror convention), module-level `pytestmark = [pytest.mark.core_model, pytest.mark.cpu]` so Buildkite "Simple Test" runs them GPU-free. Fixture-based tests only; hardware never required (AC-24).
- AC-29 (offline only): `vllm_omni/__init__.py` imports only version/patch/transformers_utils — verified it will not pull in `profiling/`. Keep `profiling/__init__.py` free of heavy/optional imports; adapters import optional deps lazily at use site, following the repo idiom `try: import torch_npu  # noqa: F401 / except ImportError: pass` (`vllm_omni/benchmarks/data_modules/seed_tts_eval.py:141`) and warn-and-degrade (`vllm_omni/profiler/omni_torch_profiler.py:345`).

## 2. Reuse (do not re-implement)

| Existing component | Path | Use |
|---|---|---|
| `ProfilerConfig` | upstream `vllm.config.ProfilerConfig`, wired via `vllm_omni/config/omni_config.py`, `stage_config.py:436` | The "profiler_config" of the spec. No new config dataclass. |
| Artifact layout/naming | `vllm_omni/profiler/omni_torch_profiler.py` (session dirs, `trace_rank{N}.json[.gz]`, `get_results()`) | CUDA discovery must match these exact conventions. |
| NPU collection path | `vllm_omni/platforms/npu/profiler.py` (torch_npu.profiler, tensorboard_trace_handler → `{dir}/{worker_name}`) | Defines where Ascend artifacts appear. Note: NPU path does not compute `key_averages`; offline analysis reads exported CSV/DB instead. |
| Chrome-trace parsing reference | `.claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py` | Reference for busy/idle unions, top ops, NCCL stats. It is an agent skill, not a package: write fresh module code in the framework (repo SPDX/license consistency), do not import or copy verbatim. |
| Evidence-JSON precedent | `benchmarks/tts/evidence/*.json` (`measurement_scope`, `environment`, `workload`, `limits[]`) | Aligns with the `OptimizationEvidence` envelope philosophy; provenance requirements come from the framework spec. |
| Test fakes pattern | `tests/profile/test_omni_torch_profiler.py` (fakes + monkeypatch + tmp_path, no GPU) | Template for adapter tests. |

## 3. Do not duplicate

- Runtime collection: `OmniTorchProfilerWrapper`, `NPUTorchProfilerWrapper`, vLLM `CudaProfilerWrapper` (nsys), `vllm_omni/diffusion/profiler/diffusion_pipeline_profiler.py`, `vllm_omni/engine/orchestrator_monitor.py`. The framework consumes their outputs; never re-instruments the serving runtime.
- Workload generation: `benchmarks/`, `vllm bench serve --omni` stay as-is.
- GUI drill-down: Perfetto / Nsight / MindStudio remain the human tools (spec non-goals).
- HTA / nsys / NCU / msprof: optional adapters wrap them when installed; never vendor or hard-depend (AC-11). No `hta` usage exists in the repo today (grep: 0 hits).

## 4. Dependencies

- Hard imports of the framework: stdlib only (`json`, `csv`, `sqlite3`, `argparse`, `gzip`, `pathlib`).
- Optional (lazy, degrade with warning): `pandas` (already an optional repo dep), `hta`, later `msprof` tooling. No new pyproject dependencies for MVP; an extras entry can be added later if justified.
- Parsing surfaces: CUDA torch-profiler Chrome JSON (gzip-aware); Ascend `step_trace_time.csv`, `op_statistic*.csv`, `op_summary*.csv`, `kernel_details.csv`, `analysis.db` (SQLite, read-only; explicit unsupported-schema error per spec section 10.3). Nsight input is SQLite/CSV export only — never parse `.nsys-rep`.

## 5. Execution discipline (2026-09-29 rulings)

1. **Real profiling artifacts are an acceptance dependency, not an implementation dependency.** No artifact → continue schema/parser development and mark `prepared_not_run`; never mark `real-data validated` without one. Milestone reporting uses exactly: `implemented` / `tested` / `real-data validated` / `prepared_not_run` / `blocked`.
2. **Checkpoints.** T00 → M1 (CUDA vertical slice) → *Checkpoint 1: schema review* (human pass over actual `evidence.json`, `summary.md`, provenance sample, query output, measured/derived/unavailable examples, raw→evidence mapping, fields most likely to be challenged by Ascend) → M2 (Ascend vertical slice; purpose is cross-backend contract validation, not full file-format coverage; profiler-DB adapter = `prepared_not_run`/`blocked-on-sample` without a real DB) → *Checkpoint 2: cross-XPU contract review* (does Ascend force unreasonable distortion of the common contract?) → M3 (deterministic diagnosis, built on observed common semantics of both backends' evidence) → MVP gate → M4 (optional enrichment) → v1 gate. Schema stays adjustable through M2.
3. **Golden facts start at M1.** Every real profiler sample gets ≥5 manually confirmed facts saved as regression fixtures; adapters must reproduce them (doubles as the manual cross-check required by acceptance). Ascend gets its own set at M2.
4. **Confidence values** are `high` / `medium` / `low` or a rule-derived score; never pseudo-precise LLM-style decimals.
5. **Diagnosis scope**: classification only (`what is happening?`); root cause and change proposals belong to the downstream optimization agent.

## 6. Key file map (inventory appendix)

```text
vllm_omni/profiler/omni_torch_profiler.py     collection wrapper, artifact naming, get_results()
vllm_omni/platforms/{xpu,npu}/profiler.py     platform profiler classes (lazy torch_npu import)
vllm_omni/worker/base.py:49-98                profiler wiring per worker/stage
vllm_omni/entrypoints/serve/profile/          /start_profile //stop_profile endpoints
vllm_omni/config/omni_config.py               ProfilerConfig wiring
docs/contributing/profiling.md                user-facing profiling guide (artifact inventory)
docs/design/module/profiling.md               module design doc (collection)
benchmarks/tts/evidence/*.json                committed evidence-JSON precedent
.claude/skills/diffusion-perf-opt/            Chrome-trace analyzer reference (agent skill)
tests/profile/test_omni_torch_profiler.py     GPU-free test pattern with fakes
pyproject.toml [tool.pytest.ini_options]      markers: core_model, cpu, gpu, ...
```
