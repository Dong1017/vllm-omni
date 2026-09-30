# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Backend adapters：每个 adapter 把一种 raw artifact 转成 OptimizationEvidence。
# 可选依赖一律延迟导入（AC-11/AC-29）。

from vllm_omni.profiling.backends.ascend_csv import analyze_ascend_csv
from vllm_omni.profiling.backends.ascend_db import UnsupportedProfilerDBError, analyze_ascend_db
from vllm_omni.profiling.backends.cuda_torch_profiler import analyze_cuda_torch_profiler

__all__ = ["UnsupportedProfilerDBError", "analyze_ascend_csv", "analyze_ascend_db", "analyze_cuda_torch_profiler"]
