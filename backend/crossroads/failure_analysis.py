"""Failure analysis for a trajectory — two tiers.

Tier 1 (heuristic, always available):
  Identifies WHERE the session failed and classifies the failure mode from
  observable signals. No LLM needed — runs instantly on the Step model.

Tier 2 (LLM judge, opt-in):
  Sends trajectory context to a judge model for semantic analysis: WHY it
  failed, what the agent misunderstood, what it should have done differently.
  Gated behind explicit user action (never auto-runs).

Failure modes detected heuristically:
  - loop_to_death: agent enters a loop it never exits (3+ repeats of same action)
  - error_cascade: 3+ consecutive errors without meaningful recovery
  - lost_after_compaction: agent loses critical state after a compaction wall
    and never recovers the knowledge it needs
  - wrong_approach_persisted: agent tries the same failing strategy repeatedly
    (edits that fail tests, re-edits, same test fails again)
  - abandoned: session ends abruptly (no clear resolution, last steps are
    exploratory or the user stops engaging)
  - context_exhaustion: session hits compaction multiple times with declining
    productivity (fewer edits, more re-reads per turn)
"""

from __future__ import annotations

from .llm import extract_json, sampling_kwargs

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .models import Phase, Step, Trajectory


class FailureMode(str, Enum):
    loop_to_death = "loop_to_death"
    error_cascade = "error_cascade"
    lost_after_compaction = "lost_after_compaction"
    wrong_approach_persisted = "wrong_approach_persisted"
    abandoned = "abandoned"
    context_exhaustion = "context_exhaustion"
    unknown = "unknown"
    none = "none"


class FailureEvidence(BaseModel):
    """One piece of evidence supporting the failure diagnosis."""
    step_index: int
    signal: str
    detail: str


class CausalChainLink(BaseModel):
    """One link in the inferred causal chain leading to failure."""
    step_index: int
    role: str
    summary: str


class Difficulty(BaseModel):
    """A difficulty the agent overcame during the session."""
    kind: str
    step_index: int
    detail: str


class HeuristicAnalysis(BaseModel):
    """Tier 1: heuristic failure/quality analysis (always available, no LLM)."""
    failed: bool
    failure_mode: FailureMode
    confidence: float
    failure_step: Optional[int] = None
    description: str
    evidence: list[FailureEvidence] = Field(default_factory=list)
    causal_chain: list[CausalChainLink] = Field(default_factory=list)
    contributing_factors: list[str] = Field(default_factory=list)
    recovery_attempts: int = 0
    wasted_after_failure: float = 0.0
    difficulties_overcome: list[Difficulty] = Field(default_factory=list)
    efficiency_score: Optional[float] = None
    shortcuts_taken: list[str] = Field(default_factory=list)


class JudgeDecisionPoint(BaseModel):
    step: int
    decision: str
    better_alternative: str
    confidence: float = 0.0


class JudgeAnalysis(BaseModel):
    """Tier 2: LLM judge semantic analysis (opt-in)."""
    available: bool = False
    ran: bool = False
    why_it_failed: Optional[str] = None
    what_agent_misunderstood: Optional[str] = None
    what_should_have_happened: Optional[str] = None
    what_went_well: Optional[str] = None
    what_could_be_better: Optional[str] = None
    failure_category: Optional[str] = None
    severity: Optional[str] = None
    confidence: float = 0.0
    reasoning: Optional[str] = None
    failure_step_judge: Optional[int] = None
    key_decision_points: list[JudgeDecisionPoint] = Field(default_factory=list)
    model_used: Optional[str] = None
    cost_usd: float = 0.0
    error: Optional[str] = None


class FailureAnalysisResponse(BaseModel):
    session_id: str
    n_steps: int
    heuristic: HeuristicAnalysis
    judge: JudgeAnalysis = Field(default_factory=JudgeAnalysis)
    note: str




def _detect_loop_to_death(steps: list[Step]) -> Optional[tuple[int, list[FailureEvidence]]]:
    """Detect a loop the agent never exits in the final third of the run."""
    n = len(steps)
    if n < 10:
        return None

    final_third_start = n * 2 // 3
    final_steps = steps[final_third_start:]


    sigs = [(s.tool, _norm(s.summary)) for s in final_steps]
    from collections import Counter
    sig_counts = Counter(sigs)


    if not sig_counts:
        return None
    most_common_sig, count = sig_counts.most_common(1)[0]
    if count < 4:
        return None


    loop_steps = [s for s in final_steps if (s.tool, _norm(s.summary)) == most_common_sig]
    first_loop_step = loop_steps[0].index

    evidence = [
        FailureEvidence(
            step_index=s.index,
            signal="repeated_action",
            detail=f"{s.tool}: {s.summary[:60]}",
        )
        for s in loop_steps[:5]
    ]

    return first_loop_step, evidence


def _detect_error_cascade(steps: list[Step]) -> Optional[tuple[int, list[FailureEvidence]]]:
    """Detect 3+ consecutive errors without meaningful recovery."""
    max_run = 0
    max_run_start = -1
    current_run = 0
    run_start = -1

    for s in steps:
        if s.signals.is_error or s.signals.test_failed:
            if current_run == 0:
                run_start = s.index
            current_run += 1
            if current_run > max_run:
                max_run = current_run
                max_run_start = run_start

        elif s.signals.is_edit:
            current_run = 0

    if max_run < 3:
        return None

    cascade_steps = [s for s in steps if max_run_start <= s.index < max_run_start + max_run]
    evidence = [
        FailureEvidence(
            step_index=s.index,
            signal="error" if s.signals.is_error else "test_failed",
            detail=f"{s.tool}: {s.observation_preview[:80]}",
        )
        for s in cascade_steps[:6]
    ]

    return max_run_start, evidence


def _detect_lost_after_compaction(steps: list[Step], compaction_indices: set[int]) -> Optional[tuple[int, list[FailureEvidence]]]:
    """Detect when the agent loses critical info after compaction and can't recover."""
    if not compaction_indices:
        return None

    for wall in sorted(compaction_indices):

        post_wall = [s for s in steps if wall <= s.index < wall + 15]
        if len(post_wall) < 5:
            continue



        rereads = [s for s in post_wall if s.signals.is_reread or s.signals.is_redundant_reread]
        errors = [s for s in post_wall if s.signals.is_error]

        if len(rereads) >= 3 or (len(rereads) >= 2 and len(errors) >= 2):
            evidence = []
            for s in rereads[:3]:
                evidence.append(FailureEvidence(
                    step_index=s.index,
                    signal="reread_after_compaction",
                    detail=f"re-reading {s.signals.file_path or s.summary[:40]} (seen {s.signals.file_access_count}x)",
                ))
            for s in errors[:2]:
                evidence.append(FailureEvidence(
                    step_index=s.index,
                    signal="error_after_compaction",
                    detail=f"{s.tool}: {s.observation_preview[:60]}",
                ))
            return wall, evidence

    return None


def _detect_wrong_approach(steps: list[Step]) -> Optional[tuple[int, list[FailureEvidence]]]:
    """Detect edit-test-fail cycles where the agent keeps trying the same approach."""
    cycles = []
    i = 0
    while i < len(steps):
        s = steps[i]
        if s.signals.is_edit:

            for j in range(i + 1, min(i + 6, len(steps))):
                if steps[j].signals.test_failed:
                    cycles.append((s.index, steps[j].index))
                    break
        i += 1

    if len(cycles) < 3:
        return None


    edit_files = []
    for edit_idx, _ in cycles:
        step = next((s for s in steps if s.index == edit_idx), None)
        if step and step.signals.file_path:
            edit_files.append(step.signals.file_path)

    from collections import Counter
    if edit_files:
        most_edited, count = Counter(edit_files).most_common(1)[0]
        if count >= 3:
            first_cycle_step = cycles[0][0]
            evidence = [
                FailureEvidence(
                    step_index=edit_idx,
                    signal="failed_edit_cycle",
                    detail=f"edited then test failed at step {fail_idx}",
                )
                for edit_idx, fail_idx in cycles[:5]
            ]
            return first_cycle_step, evidence


    if len(cycles) >= 4:
        evidence = [
            FailureEvidence(
                step_index=edit_idx,
                signal="failed_edit_cycle",
                detail=f"edited then test failed at step {fail_idx}",
            )
            for edit_idx, fail_idx in cycles[:5]
        ]
        return cycles[0][0], evidence

    return None


def _detect_abandoned(steps: list[Step]) -> Optional[tuple[int, list[FailureEvidence]]]:
    """Detect if the session just stops without resolution."""
    if len(steps) < 5:
        return None


    tail = steps[-5:]
    has_edit = any(s.signals.is_edit for s in tail)
    has_error = any(s.signals.is_error or s.signals.test_failed for s in tail)


    total_errors = sum(1 for s in steps if s.signals.is_error or s.signals.test_failed)
    if not has_edit and total_errors > 3 and not has_error:
        evidence = [
            FailureEvidence(
                step_index=s.index,
                signal="trailing_exploration",
                detail=f"{s.tool}: {s.summary[:60]} (no resolution)",
            )
            for s in tail
        ]
        return tail[0].index, evidence

    return None


def _detect_context_exhaustion(steps: list[Step], compaction_indices: set[int]) -> Optional[tuple[int, list[FailureEvidence]]]:
    """Detect declining productivity across multiple compactions."""
    if len(compaction_indices) < 2:
        return None

    walls = sorted(compaction_indices)

    segments = []
    for i, wall in enumerate(walls):
        next_wall = walls[i + 1] if i + 1 < len(walls) else len(steps)
        segment = [s for s in steps if wall <= s.index < next_wall]
        if not segment: continue
        edits = sum(1 for s in segment if s.signals.is_edit)
        density = edits / len(segment)
        segments.append((wall, next_wall, density))

    if len(segments) < 2:
        return None


    if all(segments[i][2] >= segments[i + 1][2] for i in range(len(segments) - 1)):
        if segments[-1][2] < 0.05:
            evidence = [
                FailureEvidence(
                    step_index=wall,
                    signal="declining_productivity",
                    detail=f"edit density dropped to {density:.0%} after compaction at step {wall}",
                )
                for wall, _, density in segments
            ]
            return walls[1], evidence

    return None


def _norm(s: str) -> str:
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:50]


def _build_causal_chain(steps: list[Step], failure_step: int, mode: FailureMode) -> list[CausalChainLink]:
    """Build a causal chain from the trigger to the failure symptom."""
    chain = []


    pre_failure = [s for s in steps if s.index < failure_step and s.criticality > 0.3]
    if pre_failure:

        trigger_candidates = [s for s in pre_failure if s.signals.is_edit or s.signals.is_error]
        if trigger_candidates:
            trigger = trigger_candidates[0]
            chain.append(CausalChainLink(
                step_index=trigger.index,
                role="trigger",
                summary=f"{trigger.tool}: {trigger.summary[:60]}",
            ))


    trigger_idx = chain[0].step_index if chain else 0
    mid_steps = [s for s in steps if trigger_idx < s.index < failure_step and s.criticality > 0.4]
    for s in mid_steps[:3]:
        chain.append(CausalChainLink(
            step_index=s.index,
            role="escalation",
            summary=f"{s.tool}: {s.summary[:60]}",
        ))


    chain.append(CausalChainLink(
        step_index=failure_step,
        role="point_of_no_return",
        summary=f"Failure mode: {mode.value}",
    ))


    post_failure = [s for s in steps if s.index > failure_step]
    symptom_candidates = sorted(post_failure, key=lambda s: s.criticality, reverse=True)
    if symptom_candidates:
        symptom = symptom_candidates[0]
        chain.append(CausalChainLink(
            step_index=symptom.index,
            role="symptom",
            summary=f"{symptom.tool}: {symptom.summary[:60]}",
        ))

    return chain


def _count_recovery_attempts(steps: list[Step], failure_step: int) -> int:
    """Count edit attempts after the failure point (recovery efforts)."""
    return sum(1 for s in steps if s.index > failure_step and s.signals.is_edit)


def _cost_after_step(steps: list[Step], idx: int) -> float:
    """Total cost spent after a given step."""
    return sum(s.cost_usd for s in steps if s.index > idx)




def _detect_difficulties(steps: list[Step], compaction_indices: set[int]) -> list[Difficulty]:
    """Detect difficulties the agent overcame during the session."""
    difficulties = []
    n = len(steps)


    if n >= 10:
        final_third_start = n * 2 // 3
        for start in range(0, final_third_start - 4):
            sigs = [(s.tool, _norm(s.summary)) for s in steps[start:start + 6]]
            from collections import Counter
            sig_counts = Counter(sigs)
            most_common, count = sig_counts.most_common(1)[0]
            if count >= 3:
                post = steps[start + 6:start + 10] if start + 6 < n else []
                if post and not any((s.tool, _norm(s.summary)) == most_common for s in post):
                    difficulties.append(Difficulty(
                        kind="loop_recovered",
                        step_index=start,
                        detail=f"entered loop ({count} repeats) at step {start}, then broke out",
                    ))
                    break


    run = 0
    run_start = -1
    for s in steps:
        if s.signals.is_error or s.signals.test_failed:
            if run == 0:
                run_start = s.index
            run += 1
        else:
            if run >= 3 and s.signals.is_edit:
                difficulties.append(Difficulty(
                    kind="error_recovered",
                    step_index=run_start,
                    detail=f"hit {run} consecutive errors at step {run_start}, then recovered with an edit",
                ))
            run = 0


    for wall in sorted(compaction_indices):
        post_wall = [s for s in steps if wall <= s.index < wall + 10]
        rereads = [s for s in post_wall if s.signals.is_reread or s.signals.is_redundant_reread]
        if len(rereads) >= 2:
            later_edits = [s for s in steps if s.index > wall + 10 and s.signals.is_edit]
            if later_edits:
                difficulties.append(Difficulty(
                    kind="compaction_recovered",
                    step_index=wall,
                    detail=f"lost context at compaction (step {wall}), re-read {len(rereads)} files, then continued productively",
                ))

    return difficulties


def _compute_session_efficiency(steps: list[Step], compaction_indices: set[int]) -> tuple[float, list[str]]:
    """Compute efficiency score and detect shortcuts for a Claude session."""
    n = len(steps)
    if n == 0:
        return 1.0, []

    shortcuts = []

    total_errors = sum(1 for s in steps if s.signals.is_error or s.signals.test_failed)
    total_rereads = sum(1 for s in steps if s.signals.is_redundant_reread)
    total_edits = sum(1 for s in steps if s.signals.is_edit)


    edits_without_test = 0
    for i, s in enumerate(steps):
        if s.signals.is_edit:
            next_few = steps[i + 1:i + 5]
            has_test = any(ns.signals.is_test for ns in next_few)
            if not has_test:
                edits_without_test += 1
    if edits_without_test > 0 and total_edits > 0 and edits_without_test / total_edits > 0.5:
        shortcuts.append(f"{edits_without_test}/{total_edits} edits had no test verification")


    error_penalty = min(total_errors * 0.03, 0.3)
    reread_penalty = min(total_rereads * 0.02, 0.25)
    compaction_penalty = len(compaction_indices) * 0.05


    sigs = [(s.tool, _norm(s.summary)) for s in steps]
    from collections import Counter
    sig_counts = Counter(sigs)
    max_repeat = sig_counts.most_common(1)[0][1] if sig_counts else 0
    loop_penalty = min((max_repeat - 3) * 0.04, 0.2) if max_repeat > 3 else 0

    efficiency = max(0.0, min(1.0, 1.0 - error_penalty - reread_penalty - compaction_penalty - loop_penalty))
    return round(efficiency, 2), shortcuts




def analyze_failure(traj: Trajectory) -> FailureAnalysisResponse:
    """Run heuristic failure analysis on a trajectory."""
    steps = traj.steps
    compaction_indices = {c.at_index for c in traj.compactions}


    detectors = [
        (FailureMode.loop_to_death, lambda: _detect_loop_to_death(steps)),
        (FailureMode.error_cascade, lambda: _detect_error_cascade(steps)),
        (FailureMode.wrong_approach_persisted, lambda: _detect_wrong_approach(steps)),
        (FailureMode.lost_after_compaction, lambda: _detect_lost_after_compaction(steps, compaction_indices)),
        (FailureMode.context_exhaustion, lambda: _detect_context_exhaustion(steps, compaction_indices)),
        (FailureMode.abandoned, lambda: _detect_abandoned(steps)),
    ]

    detected_mode = FailureMode.none
    failure_step = None
    evidence = []
    contributing = []

    for mode, detect in detectors:
        result = detect()
        if result is not None:
            step_idx, ev = result
            if detected_mode == FailureMode.none:
                detected_mode = mode
                failure_step = step_idx
                evidence = ev
            else:
                contributing.append(f"{mode.value} (from step {step_idx})")


    error_rate = sum(1 for s in steps if s.signals.is_error or s.signals.test_failed) / max(len(steps), 1)
    has_late_errors = any(
        s.signals.is_error or s.signals.test_failed
        for s in steps[len(steps) * 2 // 3:]
    )
    failed = detected_mode != FailureMode.none or (error_rate > 0.15 and has_late_errors)

    if not failed:
        detected_mode = FailureMode.none


    if failed and detected_mode == FailureMode.none:
        detected_mode = FailureMode.unknown

        final_third = steps[len(steps) * 2 // 3:]
        if final_third:
            worst = max(final_third, key=lambda s: s.criticality)
            failure_step = worst.index
            evidence = [FailureEvidence(
                step_index=worst.index,
                signal="high_criticality",
                detail=f"criticality={worst.criticality:.2f}, components: {worst.criticality_components}",
            )]


    confidence = 0.0
    description = ""
    if detected_mode == FailureMode.none:
        confidence = 0.8
        description = "No clear failure pattern detected. The session may have completed successfully or the failure mode is not covered by heuristics."
    elif detected_mode == FailureMode.loop_to_death:
        confidence = 0.85
        description = f"The agent entered a repetitive loop at step {failure_step} and never broke out. It kept performing the same action without making progress."
    elif detected_mode == FailureMode.error_cascade:
        confidence = 0.8
        description = f"A cascade of {len(evidence)} consecutive errors began at step {failure_step}. The agent failed to recover between errors."
    elif detected_mode == FailureMode.wrong_approach_persisted:
        confidence = 0.75
        description = f"The agent persisted with a failing approach starting at step {failure_step} — editing, testing, failing, and repeating without changing strategy."
    elif detected_mode == FailureMode.lost_after_compaction:
        confidence = 0.7
        description = f"After the compaction wall at step {failure_step}, the agent lost critical context and spent significant effort re-reading files it had already processed."
    elif detected_mode == FailureMode.context_exhaustion:
        confidence = 0.65
        description = "Multiple compactions eroded the agent's productivity. Edit density declined across compaction boundaries, suggesting progressive context loss."
    elif detected_mode == FailureMode.abandoned:
        confidence = 0.55
        description = f"The session appears abandoned at step {failure_step}. The agent stopped making edits despite unresolved errors earlier in the run."
    else:
        confidence = 0.5
        description = f"The session shows failure signals but the mode doesn't match known patterns. Highest-criticality step in the final third is step {failure_step}."

    causal_chain = []
    recovery_attempts = 0
    wasted = 0.0
    if failure_step is not None:
        causal_chain = _build_causal_chain(steps, failure_step, detected_mode)
        recovery_attempts = _count_recovery_attempts(steps, failure_step)
        wasted = _cost_after_step(steps, failure_step)


    difficulties = _detect_difficulties(steps, compaction_indices)
    efficiency, shortcuts = _compute_session_efficiency(steps, compaction_indices)


    if not failed:
        desc_parts = []
        if efficiency >= 0.8:
            desc_parts.append("Clean execution")
        elif efficiency >= 0.5:
            desc_parts.append("Completed with moderate difficulty")
        else:
            desc_parts.append("Completed despite significant difficulties")
        if difficulties:
            desc_parts.append(f" — overcame {len(difficulties)} obstacle{'s' if len(difficulties) != 1 else ''}")
        if shortcuts:
            desc_parts.append(f". Took {len(shortcuts)} shortcut{'s' if len(shortcuts) != 1 else ''}")
        description = "".join(desc_parts) + "."

    heuristic = HeuristicAnalysis(
        failed=failed,
        failure_mode=detected_mode,
        confidence=confidence,
        failure_step=failure_step,
        description=description,
        evidence=evidence,
        causal_chain=causal_chain,
        contributing_factors=contributing,
        recovery_attempts=recovery_attempts,
        wasted_after_failure=round(wasted, 4),
        difficulties_overcome=difficulties,
        efficiency_score=efficiency,
        shortcuts_taken=shortcuts,
    )

    note = (
        "Tier 1 (heuristic): failure mode and causal chain inferred from observable signals — no LLM needed. Tier 2 (semantic): call the /judge endpoint to get an LLM-based analysis of WHY the agent failed and what it should have done."


        if failed
        else "Tier 1 (heuristic): execution quality analysis. Tier 2 (semantic): call the /judge endpoint for deeper analysis of what went well and what could be better."
    )


    return FailureAnalysisResponse(
        session_id=traj.meta.id,
        n_steps=len(steps),
        heuristic=heuristic,
        judge=JudgeAnalysis(available=False),
        note=note,
    )





from .settings import JUDGE_API_KEY as _JUDGE_API_KEY, JUDGE_BASE_URL as _JUDGE_BASE_URL, JUDGE_MODEL as _JUDGE_MODEL

def _extract_judge_context(traj: Trajectory, failure_step: Optional[int]) -> str:
    """Build a condensed context string for the LLM judge."""
    steps = traj.steps
    lines = [
        f"Session: {traj.meta.title}",
        f"Model: {traj.meta.model or 'unknown'}",
        f"Steps: {len(steps)}, Errors: {traj.meta.n_errors}, Edits: {traj.meta.n_edits}",
        f"Compactions: {traj.meta.n_compactions}",
        "",
        "=== FULL TRAJECTORY ===",
    ]


    important_steps = set()
    if failure_step is not None:
        for i in range(max(0, failure_step - 5), min(len(steps), failure_step + 10)):
            important_steps.add(i)
    for s in steps:
        if s.criticality > 0.4:
            important_steps.add(s.index)

    for s in steps:
        if s.index in important_steps:
            marker = " <<<FAILURE POINT>>>" if s.index == failure_step else ""
            err_flag = " [ERROR]" if s.signals.is_error else ""
            test_flag = " [TEST FAILED]" if s.signals.test_failed else ""
            loop_flag = " [IN LOOP]" if s.signals.in_loop else ""
            lines.append(
                f"Step {s.index} [{s.phase.value}]: {s.tool} | {s.summary[:100]}"
                f"{err_flag}{test_flag}{loop_flag}{marker}"
            )
            if s.thought:
                lines.append(f"  Agent thought: {s.thought[:300]}")
            if s.observation_preview:
                lines.append(f"  Result: {s.observation_preview[:300]}")
            if s.criticality > 0.3:
                lines.append(f"  Criticality: {s.criticality:.2f} {s.criticality_components}")
            lines.append("")
        elif s.signals.is_error or s.signals.test_failed or s.signals.is_edit:
            lines.append(
                f"Step {s.index} [{s.phase.value}]: {s.tool} | {s.summary[:80]}"
                f" [{'ERROR' if s.signals.is_error else 'EDIT' if s.signals.is_edit else 'TEST FAIL'}]"
            )
        else:
            lines.append(f"Step {s.index} [{s.phase.value}]: {s.tool} | {s.summary[:50]}")

    return "\n".join(lines)


JUDGE_SYSTEM_PROMPT_FAILED = """You are an expert at analyzing AI coding agent trajectories to understand why they fail.

You will receive the full trajectory of an agent session and a heuristic pre-analysis. Your job is to provide the SEMANTIC explanation — not just what signals fired, but WHY the agent failed at the task level.

Think step by step about:
1. What was the agent trying to accomplish?
2. Where did its understanding break down?
3. What was the critical decision that led to failure?
4. Could it have recovered, or was this a fundamental limitation?

Be specific — reference exact step numbers and explain the reasoning gap."""

JUDGE_SYSTEM_PROMPT_QUALITY = """You are an expert at analyzing AI coding agent trajectories for execution quality.

You will receive the full trajectory of an interactive Claude Code session that completed (the user's task was accomplished). Your job is to analyze HOW it was accomplished — the quality of execution, not just the outcome.

Think step by step about:
1. What did the agent do well? (efficient exploration, good strategy, quick recovery from errors)
2. What difficulties did it overcome? (user interrupts, errors, compaction, wrong approaches it corrected)
3. What could it have done better? (unnecessary re-reads, over-exploration, missed shortcuts)
4. Did it require excessive user intervention? Was the user guiding it or was it autonomous?
5. Were there any risky shortcuts? (skipped tests, minimal validation, partial fixes)

Be specific — reference exact step numbers."""

JUDGE_USER_TEMPLATE_FAILED = """Analyze this agent trajectory for failure:

{context}

=== HEURISTIC PRE-ANALYSIS ===
Failure mode detected: {failure_mode}
Failure step (heuristic): {failure_step}
Heuristic description: {description}

=== INSTRUCTIONS ===
Provide your analysis as JSON with these fields:
- "reasoning": your step-by-step analysis of what happened (2-4 sentences)
- "why_it_failed": the root semantic reason, not just "it errored" (1-2 sentences)
- "what_agent_misunderstood": what the agent got wrong about the task/codebase (1-2 sentences)
- "what_should_have_happened": the ideal path it should have taken (1-2 sentences)
- "what_went_well": anything the agent did correctly before failing (1-2 sentences, or null)
- "what_could_be_better": what would have prevented the failure (1-2 sentences)
- "failure_category": one of "planning" / "execution" / "knowledge" / "tool-use"
- "severity": one of "recoverable" / "hard" / "impossible"
- "confidence": 0.0-1.0, your confidence in this analysis
- "failure_step": the step number where you think the critical failure occurred
- "key_decision_points": array of {{"step": N, "decision": "what agent did", "better_alternative": "what it should have done", "confidence": 0.0-1.0}}

Respond ONLY with valid JSON, no markdown fences."""

JUDGE_USER_TEMPLATE_QUALITY = """Analyze this agent trajectory for execution quality:

{context}

=== HEURISTIC PRE-ANALYSIS ===
Efficiency score: {efficiency_score}
Difficulties overcome: {difficulties}
Shortcuts taken: {shortcuts}
Description: {description}

=== INSTRUCTIONS ===
The session completed — the user's task was accomplished. Analyze execution quality.

Provide your analysis as JSON with these fields:
- "reasoning": your step-by-step analysis of how the agent solved this (3-5 sentences)
- "what_went_well": what the agent did efficiently or cleverly (2-3 sentences)
- "what_could_be_better": unnecessary steps, missed shortcuts, over-exploration (2-3 sentences)
- "why_it_failed": null
- "what_agent_misunderstood": any misconceptions along the way, even though it succeeded (1-2 sentences, or null)
- "what_should_have_happened": the ideal minimal path (1-2 sentences)
- "failure_category": null
- "severity": null
- "confidence": 0.0-1.0
- "failure_step": null
- "key_decision_points": array of {{"step": N, "decision": "what agent did", "better_alternative": "what would have been more efficient", "confidence": 0.0-1.0}}

Respond ONLY with valid JSON, no markdown fences."""


import json as _json
import re as _re


def _extract_json_text(raw: str) -> str:
    """Extract JSON object from model output that may contain <think> tags, markdown fences, or preamble."""
    text = raw.strip()

    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_start < brace_end:
        text = text[brace_start:brace_end + 1]
    return text


async def run_judge(traj: Trajectory, model: str = _JUDGE_MODEL) -> JudgeAnalysis:
    """Call the LLM judge and return structured results. Results are cached."""
    from .judge_cache import get_cached_judge, save_judge_result

    cached = get_cached_judge(traj.meta.id)
    if cached is not None:
        decision_points = [
            JudgeDecisionPoint(**dp) for dp in cached.get("key_decision_points", [])
            if isinstance(dp, dict) and "step" in dp
        ]
        return JudgeAnalysis(
            available=True, ran=True,
            why_it_failed=cached.get("why_it_failed"),
            what_agent_misunderstood=cached.get("what_agent_misunderstood"),
            what_should_have_happened=cached.get("what_should_have_happened"),
            what_went_well=cached.get("what_went_well"),
            what_could_be_better=cached.get("what_could_be_better"),
            failure_category=cached.get("failure_category"),
            severity=cached.get("severity"),
            confidence=cached.get("confidence", 0.0),
            reasoning=cached.get("reasoning"),
            failure_step_judge=cached.get("failure_step_judge"),
            key_decision_points=decision_points,
            model_used=cached.get("model_used"),
            cost_usd=cached.get("cost_usd", 0.0),
        )

    try:
        from openai import AsyncOpenAI
    except ImportError:
        return JudgeAnalysis(
            available=False, ran=False,
            error="openai package not installed. Run: pip install openai",
        )

    heuristic_result = analyze_failure(traj).heuristic
    context = _extract_judge_context(traj, heuristic_result.failure_step)

    if heuristic_result.failed:
        system_prompt = JUDGE_SYSTEM_PROMPT_FAILED
        user_msg = JUDGE_USER_TEMPLATE_FAILED.format(
            context=context,
            failure_mode=heuristic_result.failure_mode.value,
            failure_step=heuristic_result.failure_step,
            description=heuristic_result.description,
        )
    else:
        system_prompt = JUDGE_SYSTEM_PROMPT_QUALITY
        user_msg = JUDGE_USER_TEMPLATE_QUALITY.format(
            context=context,
            efficiency_score=heuristic_result.efficiency_score,
            difficulties="; ".join(d.detail for d in heuristic_result.difficulties_overcome) or "none detected",
            shortcuts="; ".join(heuristic_result.shortcuts_taken) or "none detected",
            description=heuristic_result.description,
        )

    try:
        client = AsyncOpenAI(base_url=_JUDGE_BASE_URL, api_key=_JUDGE_API_KEY)
        response = await client.chat.completions.create(
            model=model,
            max_tokens=4096,
            **sampling_kwargs(model, 0.3),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
    except Exception as e:
        return JudgeAnalysis(
            available=True, ran=True,
            error=f"Judge call failed: {str(e)[:200]}. Check OPENAI_API_KEY and OPENAI_BASE_URL.",
            model_used=model,
        )

    msg = response.choices[0].message
    raw_text = msg.content or ""
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0

    cost = 0.0

    text = _extract_json_text(raw_text)

    try:
        data = extract_json(text)
    except _json.JSONDecodeError:
        return JudgeAnalysis(
            available=True, ran=True,
            reasoning=raw_text[:500],
            error="Failed to parse judge response as JSON. Raw response saved in reasoning.",
            model_used=model,
            cost_usd=0.0,
        )

    decision_points = []
    for dp in data.get("key_decision_points", []):
        if isinstance(dp, dict) and "step" in dp:
            decision_points.append(JudgeDecisionPoint(
                step=dp["step"],
                decision=dp.get("decision", ""),
                better_alternative=dp.get("better_alternative", ""),
                confidence=dp.get("confidence", 0.0),
            ))

    judge_result = {
        "available": True, "ran": True,
        "why_it_failed": data.get("why_it_failed"),
        "what_agent_misunderstood": data.get("what_agent_misunderstood"),
        "what_should_have_happened": data.get("what_should_have_happened"),
        "what_went_well": data.get("what_went_well"),
        "what_could_be_better": data.get("what_could_be_better"),
        "failure_category": data.get("failure_category"),
        "severity": data.get("severity"),
        "confidence": data.get("confidence", 0.0),
        "reasoning": data.get("reasoning"),
        "failure_step_judge": data.get("failure_step"),
        "key_decision_points": [
            {"step": dp.step, "decision": dp.decision, "better_alternative": dp.better_alternative, "confidence": dp.confidence}
            for dp in decision_points
        ],
        "model_used": model, "cost_usd": 0.0, "error": None,
    }
    save_judge_result(traj.meta.id, "claude", judge_result)

    return JudgeAnalysis(
        available=True,
        ran=True,
        why_it_failed=data.get("why_it_failed"),
        what_agent_misunderstood=data.get("what_agent_misunderstood"),
        what_should_have_happened=data.get("what_should_have_happened"),
        what_went_well=data.get("what_went_well"),
        what_could_be_better=data.get("what_could_be_better"),
        failure_category=data.get("failure_category"),
        severity=data.get("severity"),
        confidence=data.get("confidence", 0.0),
        reasoning=data.get("reasoning"),
        failure_step_judge=data.get("failure_step"),
        key_decision_points=decision_points,
        model_used=model,
        cost_usd=0.0,
    )


class JudgeRunResponse(BaseModel):
    """Full response: heuristic + judge combined."""
    session_id: str
    n_steps: int
    heuristic: HeuristicAnalysis
    judge: JudgeAnalysis
    note: str


async def stream_judge(traj: Trajectory, model: str = _JUDGE_MODEL):
    """Stream the judge response token-by-token as SSE events.

    Yields SSE lines:
      data: {"type": "token", "text": "..."}
      data: {"type": "done", "judge": {...}}   (final parsed result)
      data: {"type": "error", "message": "..."}

    Results are cached in SQLite so re-runs don't waste tokens.
    """
    from .judge_cache import get_cached_judge, save_judge_result

    cached = get_cached_judge(traj.meta.id)
    if cached is not None:
        yield f"data: {_json.dumps({'type': 'done', 'judge': cached})}\n\n"
        return

    try:
        from openai import AsyncOpenAI
    except ImportError:
        yield f"data: {_json.dumps({'type': 'error', 'message': 'openai package not installed'})}\n\n"
        return

    heuristic_result = analyze_failure(traj).heuristic
    context = _extract_judge_context(traj, heuristic_result.failure_step)

    if heuristic_result.failed:
        system_prompt = JUDGE_SYSTEM_PROMPT_FAILED
        user_msg = JUDGE_USER_TEMPLATE_FAILED.format(
            context=context,
            failure_mode=heuristic_result.failure_mode.value,
            failure_step=heuristic_result.failure_step,
            description=heuristic_result.description,
        )
    else:
        system_prompt = JUDGE_SYSTEM_PROMPT_QUALITY
        user_msg = JUDGE_USER_TEMPLATE_QUALITY.format(
            context=context,
            efficiency_score=heuristic_result.efficiency_score,
            difficulties="; ".join(d.detail for d in heuristic_result.difficulties_overcome) or "none detected",
            shortcuts="; ".join(heuristic_result.shortcuts_taken) or "none detected",
            description=heuristic_result.description,
        )

    try:
        client = AsyncOpenAI(base_url=_JUDGE_BASE_URL, api_key=_JUDGE_API_KEY)
        stream = await client.chat.completions.create(
            model=model,
            max_tokens=4096,
            **sampling_kwargs(model, 0.3),
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
    except Exception as e:
        yield f"data: {_json.dumps({'type': 'error', 'message': f'Judge call failed: {str(e)[:200]}'})}\n\n"
        return

    full_text = ""
    async for chunk in stream:
        if not chunk.choices: continue
        delta = chunk.choices[0].delta
        if delta.content:
            full_text += delta.content
            yield f"data: {_json.dumps({'type': 'token', 'text': delta.content})}\n\n"

    text = _extract_json_text(full_text)

    try:
        data = extract_json(text)
        decision_points = []
        for dp in data.get("key_decision_points", []):
            if isinstance(dp, dict) and "step" in dp:
                decision_points.append({
                    "step": dp["step"],
                    "decision": dp.get("decision", ""),
                    "better_alternative": dp.get("better_alternative", ""),
                    "confidence": dp.get("confidence", 0.0),
                })

        judge_result = {
            "available": True, "ran": True,
            "why_it_failed": data.get("why_it_failed"),
            "what_agent_misunderstood": data.get("what_agent_misunderstood"),
            "what_should_have_happened": data.get("what_should_have_happened"),
            "what_went_well": data.get("what_went_well"),
            "what_could_be_better": data.get("what_could_be_better"),
            "failure_category": data.get("failure_category"),
            "severity": data.get("severity"),
            "confidence": data.get("confidence", 0.0),
            "reasoning": data.get("reasoning"),
            "failure_step_judge": data.get("failure_step"),
            "key_decision_points": decision_points,
            "model_used": model,
            "cost_usd": 0.0, "error": None,
        }
    except _json.JSONDecodeError:
        judge_result = {
            "available": True, "ran": True,
            "reasoning": full_text[:500],
            "error": "Failed to parse as JSON",
            "model_used": model, "cost_usd": 0.0,
            "why_it_failed": None, "what_agent_misunderstood": None,
            "what_should_have_happened": None, "what_went_well": None,
            "what_could_be_better": None, "failure_category": None,
            "severity": None, "confidence": 0.0, "failure_step_judge": None,
            "key_decision_points": [],
        }

    save_judge_result(traj.meta.id, "claude", judge_result)
    yield f"data: {_json.dumps({'type': 'done', 'judge': judge_result})}\n\n"
