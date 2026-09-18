# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layer 1: noisy-PP slot planning as a pure function.

No KV, no model, no device. Takes the execution geometry (world size,
chunk count, step count, source policy) and returns the slot timetable
all PP stages share. This is the USP2 slot model specialized to noisy
chunk work; both prior experiment branches' planners are subsumed here:

- Sheng's carry-queue planner (``plan_chunk_pipeline`` with ``carry``)
  for world >= 1 generality.
- Dong's source-policy decoupling: the timetable is independent of which
  KV versions tasks consume; ``source_policy`` only constrains the
  dependency edges (latest permits interleaved waves, clean forces
  chunk-serial ordering).

Run-to-run, the same plan must be computable on every rank without
communication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SourcePolicy = Literal["latest", "clean"]

#: Step index reserved for the t=0 context (clean) pass; it publishes KV
#: and never updates the sample.
CLEAN_PASS = -1


@dataclass(frozen=True)
class NoisyPPTask:
    """One unit of chunk-pipeline work on one PP stage."""

    chunk: int
    step: int  # 0..num_denoise_steps-1 for denoise; CLEAN_PASS for the context pass
    kind: Literal["denoise", "clean"]
    num_denoise_steps: int = 0  # dependency-key translation for clean tasks

    def dep_key(self) -> tuple[int, int]:
        """Key under which this task counts as completed for dependency
        checks: the clean pass completes (chunk, num_denoise_steps)."""
        if self.kind == "clean":
            return (self.chunk, self.num_denoise_steps)
        return (self.chunk, self.step)

    def sort_key(self) -> tuple[int, int]:
        return (self.chunk, self.step)


@dataclass(frozen=True)
class NoisyPPSlot:
    """One timetable slot: at most one task per PP stage."""

    index: int
    tasks: tuple[NoisyPPTask | None, ...]


def plan_noisy_pp(
    *,
    world: int,
    chunks: int,
    num_denoise_steps: int,
    enable_clean_pass: bool,
    source_policy: SourcePolicy,
    history_chunks: int = 6,
) -> list[NoisyPPSlot]:
    """Build the slot timetable shared by all PP stages.

    ``latest`` policy: waves of ``world`` chunks advance step-locked
    (stepwise). The consumer of a version is whoever's slot follows the
    version's completion; exact version selection happens at runtime in
    layer 2.

    ``clean`` policy: chunks run strictly one after another (serial Self
    Forcing); a chunk's first denoise step depends on every window
    predecessor's clean pass having completed.
    """
    if world < 1:
        raise ValueError("noisy PP requires pipeline_parallel_size >= 1")
    if chunks < 1:
        raise ValueError("noisy PP requires at least one chunk")
    if num_denoise_steps < 1:
        raise ValueError("noisy PP requires a positive number of denoise steps")
    if source_policy == "clean" and world > 1:
        # Self Forcing semantics are defined for serial execution; a
        # multi-stage clean run has no reference contract yet.
        raise ValueError("clean source policy currently requires world == 1")

    steps = num_denoise_steps + 1 if enable_clean_pass else num_denoise_steps
    jobs: list[tuple[int, int]] = []
    if source_policy == "clean" or world == 1:
        jobs = [(chunk, step) for chunk in range(chunks) for step in range(steps)]
    else:
        # Stepwise: waves of `world` chunks, step-locked inside a wave.
        jobs = [
            (chunk, step)
            for first in range(0, chunks, world)
            for step in range(steps)
            for chunk in range(first, min(first + world, chunks))
        ]

    def task_of(chunk: int, step: int) -> NoisyPPTask:
        if enable_clean_pass and step == num_denoise_steps:
            return NoisyPPTask(chunk=chunk, step=CLEAN_PASS, kind="clean", num_denoise_steps=num_denoise_steps)
        return NoisyPPTask(chunk=chunk, step=step, kind="denoise", num_denoise_steps=num_denoise_steps)

    slots: list[NoisyPPSlot] = []
    completed: set[tuple[int, int]] = set()
    carry: list[NoisyPPTask | None] = [None] * (world - 1)
    cursor = 0
    slot_index = 0
    while cursor < len(jobs) or any(item is not None for item in carry):
        launched: NoisyPPTask | None = None
        chunk_serial = source_policy == "clean" or world == 1
        if cursor < len(jobs) and not (chunk_serial and any(item is not None for item in carry)):
            chunk, step = jobs[cursor]
            dependencies: set[tuple[int, int]] = set()
            if step > 0 and not (enable_clean_pass and step == num_denoise_steps):
                dependencies.add((chunk, step - 1))
            if source_policy == "clean" and chunk > 0 and step == 0:
                # Serial Self Forcing: the window's clean passes must be done.
                window_start = max(0, chunk - history_chunks)
                dependencies.update((c, num_denoise_steps) for c in range(window_start, chunk))
            if dependencies <= completed:
                launched = task_of(chunk, step)
                cursor += 1
        slot_tasks: list[NoisyPPTask | None] = [None] * world
        if launched is not None:
            slot_tasks[0] = launched
        for stage, pending in enumerate(carry, start=1):
            slot_tasks[stage] = pending
        if all(item is None for item in slot_tasks):
            raise RuntimeError("noisy PP plan cannot satisfy the next task's dependencies")
        slots.append(NoisyPPSlot(index=slot_index, tasks=tuple(slot_tasks)))
        slot_index += 1
        finished = slot_tasks[-1]
        if finished is not None:
            completed.add(finished.dep_key())
        # Tasks advance one stage per slot: stages 0..world-2 carry forward.
        carry = slot_tasks[:-1]
    return slots
