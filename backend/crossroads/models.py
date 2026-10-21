"""Normalized data model for Crossroads.

A *trajectory* is a long-horizon agent run (a Claude Code session or a SWE-bench
trajectory) normalized into a flat list of *steps*. Each step is one decision the
agent made: a tool call with its observation/outcome.

Pivotality here is a **criticality index** derived from *real, observable* per-step
signals (tool errors, re-reads, edits, test failures, compaction proximity,
redundancy features). It is NOT a counterfactual rollout probability. The true
counterfactual quantity P(success | resample at step i) requires re-executing the
agent from step i; that engine is defined as an interface in `engine.py` and is
reported as "not attached" until a real re-executor is wired in. We never
fabricate rollout numbers.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Source(str, Enum):
    claude = "claude"
    swe = "swe"


class Outcome(str, Enum):
    success = "success"
    fail = "fail"
    unknown = "unknown"


class Phase(str, Enum):
    """Coarse activity phase, inferred from the tool + args of a step."""
    explore = "explore"
    read = "read"
    edit = "edit"
    test = "test"
    debug = "debug"
    other = "other"


class StepSignals(BaseModel):
    """Real, observable signals on a single step. All are evidence, not guesses."""
    is_error: bool = False
    is_edit: bool = False
    is_test: bool = False
    test_failed: bool = False
    is_reread: bool = False
    is_redundant_reread: bool = False
    file_path: Optional[str] = None
    file_access_count: int = 1
    near_compaction: bool = False
    in_loop: bool = False

    surrogate_prior: Optional[float] = None
    surrogate_reasons: list[str] = Field(default_factory=list)
    redundancy_score: Optional[float] = None
    usefulness_label: Optional[int] = None


class CacheBreakdown(BaseModel):
    """Per-step token decomposition from the Anthropic usage object."""
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_creation_5m_tokens: int = 0
    cache_creation_1h_tokens: int = 0
    fresh_input_tokens: int = 0
    output_tokens: int = 0


class Step(BaseModel):
    index: int
    phase: Phase = Phase.other
    tool: str = ""
    summary: str = ""
    thought: str = ""
    observation_preview: str = ""
    tokens: int = 0
    cache: CacheBreakdown = Field(default_factory=CacheBreakdown)
    cost_usd: float = 0.0
    reread_type: Optional[str] = None
    signals: StepSignals = Field(default_factory=StepSignals)


    text_chars: int = 0
    thinking_chars: int = 0
    tool_args_chars: int = 0
    ts: Optional[str] = None

    invalidation_cause: Optional[str] = None
    invalidation_drop: float = 0.0



    criticality: float = 0.0
    criticality_components: dict[str, float] = Field(default_factory=dict)


class CompactionEvent(BaseModel):
    """An 'amnesia wall' — a point where the agent compacted/summarized its memory."""
    at_index: int
    summary_preview: str = ""


class SessionMetrics(BaseModel):
    """Session-level cost / efficiency metrics (real, trace-based)."""
    cost_usd: float = 0.0
    cost_no_cache_usd: float = 0.0
    effective_savings: float = 0.0
    cache_hit_rate: float = 0.0
    paradox_gap: float = 0.0
    tokens_cache_read: int = 0
    tokens_cache_create: int = 0
    tokens_output: int = 0
    tokens_fresh_input: int = 0
    explore_steps: int = 0
    produce_steps: int = 0
    explore_produce_ratio: float = 0.0
    reread_churn: int = 0
    reread_recovery: int = 0
    reread_verification: int = 0
    waste_step_fraction: float = 0.0
    waste_tokens_est: int = 0


class TrajectoryMeta(BaseModel):
    id: str
    source: Source
    title: str
    method: Optional[str] = None
    model: Optional[str] = None
    n_steps: int = 0
    outcome: Outcome = Outcome.unknown
    n_compactions: int = 0
    n_errors: int = 0
    n_edits: int = 0
    total_tokens: int = 0


    root_cause_index: Optional[int] = None
    metrics: SessionMetrics = Field(default_factory=SessionMetrics)


class Trajectory(BaseModel):
    meta: TrajectoryMeta
    steps: list[Step]
    compactions: list[CompactionEvent] = Field(default_factory=list)




class SpineResponse(BaseModel):
    meta: TrajectoryMeta
    steps: list[Step]
    compactions: list[CompactionEvent]


class FanThread(BaseModel):
    """One row of the Counterfactual Fan. Until a real re-execution engine is
    attached, threads are derived from *observed* downstream signals of THIS run,
    not from resampling. `is_actual` marks the realized path."""
    label: str
    outcome: Outcome
    is_actual: bool = False
    phase_sparkline: list[Phase] = Field(default_factory=list)
    note: str = ""


class FanResponse(BaseModel):
    step_index: int
    engine_attached: bool
    engine_note: str

    recover_rate: Optional[float] = None
    recover_ci: Optional[tuple[float, float]] = None
    threads: list[FanThread] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class LadderRung(BaseModel):
    rung: int
    name: str
    semantics: str

    applicable: bool
    evidence: str = ""

    estimated_success: Optional[float] = None


class LadderResponse(BaseModel):
    step_index: int
    engine_attached: bool
    engine_note: str
    rungs: list[LadderRung]




class PeriEventPoint(BaseModel):
    offset: int
    context_tokens: int
    cost_usd: float
    is_reread: bool = False


class RecoveryRead(BaseModel):
    offset: int
    file_path: str
    reread_type: str
    cost_usd: float


class CompactionLensResponse(BaseModel):
    wall_index: int
    summary_preview: str

    pre_peak_tokens: int
    post_trough_tokens: int
    drop_fraction: float
    series: list[PeriEventPoint]

    recovery_reads: list[RecoveryRead]
    recovery_read_count: int
    recovery_cost_usd: float

    recovery_by_type: dict[str, int]
