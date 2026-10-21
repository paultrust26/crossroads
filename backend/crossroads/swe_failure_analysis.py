"""Failure analysis for SWE-bench trajectories.

Same two-tier approach as Claude session analysis:
  Tier 1 (heuristic): detects failure patterns from step data
  Tier 2 (LLM judge): semantic analysis of why it failed

SWE-bench has an advantage: we KNOW the outcome (resolved=True/False), so
failure detection is definitive, not inferred.
"""

from __future__ import annotations

import json as _json
import re
from collections import Counter
from typing import Optional

from pydantic import BaseModel, Field


def _extract_json_text(raw: str) -> str:
    """Extract JSON object from model output that may contain <think> tags, markdown fences, or preamble."""
    text = raw.strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

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


from .swe_parser import (
    SweConversationBlock,
    SweConversationResponse,
    SweSpineResponse,
    SweSpineStep,
    build_swe_conversation,
    build_swe_spine,
)


class SweFailureMode(str):
    loop_to_death = "loop_to_death"
    error_cascade = "error_cascade"
    wrong_file = "wrong_file"
    wrong_approach = "wrong_approach"
    incomplete_fix = "incomplete_fix"
    never_edited = "never_edited"
    test_blindness = "test_blindness"
    unknown = "unknown"


class SweFailureEvidence(BaseModel):
    step_index: int
    signal: str
    detail: str


class SweCausalChainLink(BaseModel):
    step_index: int
    role: str
    summary: str


class SweJudgeDecisionPoint(BaseModel):
    step: int
    decision: str
    better_alternative: str
    confidence: float = 0.0


class SweDifficulty(BaseModel):
    """A difficulty the agent overcame (relevant for resolved sessions)."""
    kind: str
    step_index: int
    detail: str


class SweHeuristicAnalysis(BaseModel):
    failed: bool
    failure_mode: str
    confidence: float
    failure_step: Optional[int] = None
    description: str
    evidence: list[SweFailureEvidence] = Field(default_factory=list)
    causal_chain: list[SweCausalChainLink] = Field(default_factory=list)
    contributing_factors: list[str] = Field(default_factory=list)
    recovery_attempts: int = 0
    total_steps: int = 0
    total_edits: int = 0
    difficulties_overcome: list[SweDifficulty] = Field(default_factory=list)
    efficiency_score: Optional[float] = None
    shortcuts_taken: list[str] = Field(default_factory=list)


class SweJudgeAnalysis(BaseModel):
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
    key_decision_points: list[SweJudgeDecisionPoint] = Field(default_factory=list)
    model_used: Optional[str] = None
    cost_usd: float = 0.0
    error: Optional[str] = None


class SweFailureAnalysisResponse(BaseModel):
    session_id: str
    task_id: str
    method: str
    resolved: bool
    n_steps: int
    heuristic: SweHeuristicAnalysis
    judge: SweJudgeAnalysis = Field(default_factory=SweJudgeAnalysis)
    note: str


_ERROR_MARKERS = (
    "error", "Error", "Traceback", "command not found", "No such file",
    "failed", "FAILED", "FileNotFoundError", "ModuleNotFoundError",
    "SyntaxError", "IndentationError",
)


def _has_error(obs: str) -> bool:
    if not obs:
        return False
    for m in _ERROR_MARKERS:
        if m in obs:
            return True
    return False


def _norm_sig(s: str) -> str:
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:50]


def _detect_loop(steps: list[SweSpineStep]) -> Optional[tuple[int, list[SweFailureEvidence]]]:
    """Detect repeated actions in the final half."""
    n = len(steps)
    if n < 8:
        return None

    half_start = n // 2
    final_steps = steps[half_start:]
    sigs = [(s.tool, _norm_sig(s.observation_preview)) for s in final_steps]
    sig_counts = Counter(sigs)

    if not sig_counts:
        return None

    most_common, count = sig_counts.most_common(1)[0]
    if count < 4:
        return None

    loop_steps = [s for s in final_steps if (s.tool, _norm_sig(s.observation_preview)) == most_common]
    evidence = [
        SweFailureEvidence(step_index=s.index, signal="repeated_action", detail=f"{s.tool}: {s.observation_preview[:60]}")
        for s in loop_steps[:5]
    ]
    return loop_steps[0].index, evidence


def _detect_error_cascade(steps: list[SweSpineStep]) -> Optional[tuple[int, list[SweFailureEvidence]]]:
    """Detect 3+ consecutive error steps."""
    max_run = 0
    max_start = -1
    run = 0
    run_start = -1

    for s in steps:
        if _has_error(s.observation_preview):
            if run == 0:
                run_start = s.index
            run += 1
            if run > max_run:
                max_run = run
                max_start = run_start

        elif s.is_edit:
            run = 0

    if max_run < 3:
        return None

    error_steps = [s for s in steps if max_start <= s.index < max_start + max_run]
    evidence = [
        SweFailureEvidence(step_index=s.index, signal="error", detail=s.observation_preview[:80])
        for s in error_steps[:5]
    ]
    return max_start, evidence


def _detect_wrong_file(steps: list[SweSpineStep]) -> Optional[tuple[int, list[SweFailureEvidence]]]:
    """Detect when edits target files that are probably wrong (many reads of other files, edits on one)."""
    edit_files = []
    read_files = []
    for s in steps:
        if not s.file_path:
            continue
        if s.is_edit:
            edit_files.append(s.file_path)
        else:
            read_files.append(s.file_path)

    if not edit_files:
        return None

    edit_counter = Counter(edit_files)
    main_edit_file = edit_counter.most_common(1)[0][0]


    read_counter = Counter(read_files)
    if main_edit_file not in read_counter or read_counter[main_edit_file] < 2:
        if len(read_counter) > 3:
            first_edit = next((s for s in steps if s.is_edit and s.file_path == main_edit_file), None)
            if first_edit:
                evidence = [
                    SweFailureEvidence(
                        step_index=first_edit.index,
                        signal="wrong_file_target",
                        detail=f"edited {main_edit_file} (read {read_counter.get(main_edit_file, 0)}x) while reading {len(read_counter)} other files",
                    )
                ]
                return first_edit.index, evidence

    return None


def _detect_never_edited(steps: list[SweSpineStep]) -> Optional[tuple[int, list[SweFailureEvidence]]]:
    """Session never made any edits."""
    edits = [s for s in steps if s.is_edit]
    if edits:
        return None
    if not steps:
        return None

    last_step = steps[-1]
    evidence = [
        SweFailureEvidence(
            step_index=last_step.index,
            signal="no_edits",
            detail=f"session completed {len(steps)} steps without any file edits",
        )
    ]
    return last_step.index, evidence


def _detect_wrong_approach(steps: list[SweSpineStep]) -> Optional[tuple[int, list[SweFailureEvidence]]]:
    """Detect edit-error cycles (edit, observe error, edit same file, error again)."""
    cycles = 0
    edit_steps = []
    i = 0
    while i < len(steps):
        s = steps[i]
        if s.is_edit:

            for j in range(i + 1, min(i + 4, len(steps))):
                if _has_error(steps[j].observation_preview):
                    cycles += 1
                    edit_steps.append(s.index)
                    break
        i += 1

    if cycles < 3:
        return None

    evidence = [
        SweFailureEvidence(step_index=idx, signal="failed_edit_cycle", detail="edit followed by error")
        for idx in edit_steps[:5]
    ]
    return edit_steps[0], evidence


def _build_swe_causal_chain(steps: list[SweSpineStep], failure_step: int, mode: str) -> list[SweCausalChainLink]:
    """Build causal chain for SWE trajectory."""
    chain = []


    pre = [s for s in steps if s.index < failure_step and (s.is_edit or _has_error(s.observation_preview))]
    if pre:
        trigger = pre[0]
        chain.append(SweCausalChainLink(
            step_index=trigger.index,
            role="trigger",
            summary=f"{trigger.tool}: {trigger.observation_preview[:60]}",
        ))


    trigger_idx = chain[0].step_index if chain else 0
    mid = [s for s in steps if trigger_idx < s.index < failure_step and (s.is_edit or _has_error(s.observation_preview))]
    for s in mid[:3]:
        chain.append(SweCausalChainLink(
            step_index=s.index,
            role="escalation",
            summary=f"{s.tool}: {s.observation_preview[:60]}",
        ))


    chain.append(SweCausalChainLink(
        step_index=failure_step,
        role="point_of_no_return",
        summary=f"Failure mode: {mode}",
    ))


    post = [s for s in steps if s.index > failure_step and _has_error(s.observation_preview)]
    if post:
        chain.append(SweCausalChainLink(
            step_index=post[-1].index,
            role="symptom",
            summary=f"{post[-1].tool}: {post[-1].observation_preview[:60]}",
        ))

    return chain


def _detect_difficulties_overcome(steps: list[SweSpineStep]) -> list[SweDifficulty]:
    """Detect difficulties the agent overcame during a resolved session."""
    difficulties = []


    n = len(steps)
    if n >= 8:
        for start in range(0, n - 4):
            sigs = [(s.tool, _norm_sig(s.observation_preview)) for s in steps[start:start + 6]]
            sig_counts = Counter(sigs)
            most_common, count = sig_counts.most_common(1)[0]
            if count >= 3:

                post = steps[start + 6:start + 10] if start + 6 < n else []
                if post and not any((s.tool, _norm_sig(s.observation_preview)) == most_common for s in post):
                    difficulties.append(SweDifficulty(
                        kind="loop_recovered",
                        step_index=start,
                        detail=f"entered repetitive loop ({count} repeats) then broke out",
                    ))
                    break


    run = 0
    run_start = -1
    for s in steps:
        if _has_error(s.observation_preview):
            if run == 0:
                run_start = s.index
            run += 1
        else:
            if run >= 3 and s.is_edit:
                difficulties.append(SweDifficulty(
                    kind="error_recovered",
                    step_index=run_start,
                    detail=f"hit {run} consecutive errors starting at step {run_start}, then recovered",
                ))
            run = 0


    edit_files = []
    for s in steps:
        if s.is_edit and s.file_path:
            edit_files.append((s.index, s.file_path))
    if len(edit_files) >= 2:
        first_file = edit_files[0][1]
        for idx, f in edit_files[1:]:
            if f != first_file:

                had_errors = any(
                    _has_error(s.observation_preview)
                    for s in steps if edit_files[0][0] < s.index < idx
                )
                if had_errors:
                    difficulties.append(SweDifficulty(
                        kind="wrong_file_corrected",
                        step_index=idx,
                        detail=f"initially edited {first_file.split('/')[-1]}, hit errors, then pivoted to {f.split('/')[-1]}",
                    ))
                break

    return difficulties


def _compute_efficiency(steps: list[SweSpineStep]) -> tuple[float, list[str]]:
    """Compute an efficiency score (0-1) and detect shortcuts.

    Efficiency = how direct was the path. A perfect run with no wasted steps = 1.0.
    """
    n = len(steps)
    if n == 0:
        return 1.0, []

    total_edits = sum(1 for s in steps if s.is_edit)
    total_errors = sum(1 for s in steps if _has_error(s.observation_preview))
    total_reads = sum(1 for s in steps if not s.is_edit)

    shortcuts = []


    first_edit_idx = next((s.index for s in steps if s.is_edit), n)
    reads_before_edit = sum(1 for s in steps if s.index < first_edit_idx and not s.is_edit)
    if first_edit_idx < 3 and total_edits > 0:
        shortcuts.append(f"jumped to editing at step {first_edit_idx} with only {reads_before_edit} reads")


    test_tools = ("bash", "execute", "run", "test")
    ran_tests = any(
        any(t in s.tool.lower() for t in test_tools) and "test" in s.observation_preview.lower()
        for s in steps
    )
    if not ran_tests and total_edits > 0:
        shortcuts.append("no test execution detected — fix may be unverified")


    error_penalty = min(total_errors * 0.05, 0.4)
    read_ratio = total_reads / max(n, 1)
    read_penalty = max(0, (read_ratio - 0.6) * 0.5)


    sigs = [(s.tool, _norm_sig(s.observation_preview)) for s in steps]
    sig_counts = Counter(sigs)
    max_repeat = sig_counts.most_common(1)[0][1] if sig_counts else 0
    loop_penalty = min((max_repeat - 2) * 0.05, 0.3) if max_repeat > 2 else 0

    efficiency = max(0.0, min(1.0, 1.0 - error_penalty - read_penalty - loop_penalty))
    return round(efficiency, 2), shortcuts




def analyze_swe_failure(session_id: str) -> SweFailureAnalysisResponse:
    """Run heuristic analysis on a SWE-bench session (both resolved and failed)."""
    spine = build_swe_spine(session_id=session_id)
    steps = spine.steps
    failed = not spine.resolved


    detectors = [
        ("loop_to_death", lambda: _detect_loop(steps)),
        ("error_cascade", lambda: _detect_error_cascade(steps)),
        ("wrong_approach", lambda: _detect_wrong_approach(steps)),
        ("wrong_file", lambda: _detect_wrong_file(steps)),
        ("never_edited", lambda: _detect_never_edited(steps)),
    ]

    detected_mode = "none" if not failed else "unknown"
    failure_step = None
    evidence = []
    contributing = []

    for mode, detect in detectors:
        result = detect()
        if result is None:
            continue
        step_idx, ev = result
        if failed and detected_mode == "unknown":
            detected_mode = mode
            failure_step = step_idx
            evidence = ev
        else:
            contributing.append(f"{mode} (from step {step_idx})")


    difficulties = []
    efficiency = None
    shortcuts = []
    if not failed:
        difficulties = _detect_difficulties_overcome(steps)
        efficiency, shortcuts = _compute_efficiency(steps)
        desc_parts = []
        if efficiency is not None and efficiency >= 0.8:
            desc_parts.append("Clean resolution")
        elif efficiency is not None and efficiency >= 0.5:
            desc_parts.append("Resolved with moderate difficulty")
        else:
            desc_parts.append("Resolved despite significant difficulties")
        if difficulties:
            desc_parts.append(f" — overcame {len(difficulties)} obstacle{'s' if len(difficulties) != 1 else ''}")
        if shortcuts:
            desc_parts.append(f". Took {len(shortcuts)} shortcut{'s' if len(shortcuts) != 1 else ''}")
        description = "".join(desc_parts) + "."
    else:
        descriptions = {
            "loop_to_death": f"The agent entered a loop at step {failure_step} repeating the same action without progress.",
            "error_cascade": f"A cascade of consecutive errors starting at step {failure_step}. The agent failed to recover.",
            "wrong_file": f"The agent edited the wrong file starting at step {failure_step}. It explored many files but edited one it barely understood.",
            "wrong_approach": f"The agent persisted with a failing approach from step {failure_step} — edit, error, repeat without changing strategy.",
            "never_edited": f"The agent never made any edits in {len(steps)} steps. It explored but never attempted a fix.",
            "unknown": "The task was not resolved. The failure mode doesn't match known heuristic patterns.",
        }
        description = descriptions.get(detected_mode, "Unknown failure.")

    confidence_map = {
        "loop_to_death": 0.85,
        "error_cascade": 0.8,
        "wrong_file": 0.65,
        "wrong_approach": 0.75,
        "never_edited": 0.9,
        "none": 1.0,
        "unknown": 0.4,
    }

    causal_chain = []
    if failure_step is not None:
        causal_chain = _build_swe_causal_chain(steps, failure_step, detected_mode)

    recovery_attempts = sum(1 for s in steps if s.index > (failure_step or 0) and s.is_edit) if failed else 0

    heuristic = SweHeuristicAnalysis(
        failed=failed,
        failure_mode=detected_mode,
        confidence=confidence_map.get(detected_mode, 0.5),
        failure_step=failure_step,
        description=description,
        evidence=evidence,
        causal_chain=causal_chain,
        contributing_factors=contributing,
        recovery_attempts=recovery_attempts,
        total_steps=spine.n_steps,
        total_edits=sum(1 for s in steps if s.is_edit),
        difficulties_overcome=difficulties,
        efficiency_score=efficiency,
        shortcuts_taken=shortcuts,
    )

    note = (
        "Tier 1 (heuristic) analysis complete. Click 'Run judge' for semantic Tier 2 analysis."
        if failed
        else "Session resolved. Tier 1 shows execution quality. Click 'Run judge' for deeper analysis of what went well and what could be better."
    )

    return SweFailureAnalysisResponse(
        session_id=session_id,
        task_id=spine.task_id,
        method=spine.method,
        resolved=not failed,
        n_steps=spine.n_steps,
        heuristic=heuristic,
        note=note,
    )






from .settings import JUDGE_API_KEY as _JUDGE_API_KEY, JUDGE_BASE_URL as _JUDGE_BASE_URL, JUDGE_MODEL as _JUDGE_MODEL

SWE_JUDGE_SYSTEM_FAILED = """You are an expert at analyzing SWE-bench agent trajectories to understand why they fail to resolve issues.

You will see the full trajectory of a coding agent attempting to fix a GitHub issue. The task was NOT resolved. Your job is to explain semantically WHY — not just "it errored" but what fundamental mistake or misunderstanding led to failure.

Consider:
1. Did the agent understand the problem correctly?
2. Did it look in the right places?
3. Did it attempt the right kind of fix?
4. Where was the critical decision that derailed it?
5. Could it have recovered, or was this doomed from a specific point?

Be precise — reference step numbers."""

SWE_JUDGE_SYSTEM_RESOLVED = """You are an expert at analyzing SWE-bench agent trajectories for execution quality.

You will see the full trajectory of a coding agent that SUCCESSFULLY resolved a GitHub issue. Your job is to analyze the quality of execution — not just whether it succeeded, but HOW.

Consider:
1. What did the agent do well? (efficient exploration, good strategy, quick pivots)
2. What difficulties did it overcome? (errors, wrong approaches it corrected)
3. What could it have done better? (unnecessary steps, missed shortcuts, over-exploration)
4. Did it take shortcuts that might be risky? (skipped tests, minimal validation)
5. Was the approach direct or wasteful?

Be precise — reference step numbers."""

SWE_JUDGE_USER_FAILED = """Analyze this failed SWE-bench trajectory:

Task: {task_id}
Method: {method}
Resolved: False
Steps: {n_steps}

=== HEURISTIC PRE-ANALYSIS ===
Failure mode: {failure_mode}
Failure step: {failure_step}
Description: {description}

=== TRAJECTORY ===
{trajectory_text}

=== INSTRUCTIONS ===
Respond as JSON:
{{
  "reasoning": "your step-by-step analysis (3-5 sentences)",
  "why_it_failed": "root semantic reason (1-2 sentences)",
  "what_agent_misunderstood": "what the agent got wrong (1-2 sentences)",
  "what_should_have_happened": "ideal path (1-2 sentences)",
  "what_went_well": "anything the agent did correctly before failing (1-2 sentences, or null)",
  "what_could_be_better": "what would have prevented the failure (1-2 sentences)",
  "failure_category": "planning|execution|knowledge|tool-use",
  "severity": "recoverable|hard|impossible",
  "confidence": 0.0-1.0,
  "failure_step": step_number_where_it_went_wrong,
  "key_decision_points": [
    {{"step": N, "decision": "what agent did", "better_alternative": "what it should have done", "confidence": 0.0-1.0}}
  ]
}}

Respond ONLY with valid JSON."""

SWE_JUDGE_USER_RESOLVED = """Analyze this RESOLVED SWE-bench trajectory for execution quality:

Task: {task_id}
Method: {method}
Resolved: True
Steps: {n_steps}

=== HEURISTIC PRE-ANALYSIS ===
Efficiency score: {efficiency_score}
Difficulties overcome: {difficulties}
Shortcuts taken: {shortcuts}
Description: {description}

=== TRAJECTORY ===
{trajectory_text}

=== INSTRUCTIONS ===
The task was resolved. Analyze execution quality — what went well, what was wasteful, and what could be improved.

Respond as JSON:
{{
  "reasoning": "your step-by-step analysis of how the agent solved this (3-5 sentences)",
  "what_went_well": "what the agent did efficiently or cleverly (2-3 sentences)",
  "what_could_be_better": "unnecessary steps, missed shortcuts, over-exploration (2-3 sentences)",
  "why_it_failed": null,
  "what_agent_misunderstood": "any misconceptions the agent had along the way, even if it succeeded (1-2 sentences, or null)",
  "what_should_have_happened": "the ideal minimal path to resolution (1-2 sentences)",
  "failure_category": null,
  "severity": null,
  "confidence": 0.0-1.0,
  "failure_step": null,
  "key_decision_points": [
    {{"step": N, "decision": "what agent did", "better_alternative": "what would have been more efficient", "confidence": 0.0-1.0}}
  ]
}}

Respond ONLY with valid JSON."""


def _build_trajectory_text(session_id: str) -> str:
    """Build full trajectory text for the judge from conversation blocks."""
    try:
        conv = build_swe_conversation(session_id=session_id)
    except ValueError:
        return "(conversation data not available)"

    lines = []
    for b in conv.blocks:
        edit_flag = " [EDIT]" if b.is_edit else ""
        err_flag = " [ERROR]" if _has_error(b.served_observation) else ""
        lines.append(f"Step {b.index}: {b.tool}{edit_flag}{err_flag}")
        if b.file_path:
            lines.append(f"  File: {b.file_path}")
        if b.thought:
            lines.append(f"  Thought: {b.thought[:300]}")
        obs = b.served_observation or b.raw_observation
        if obs:
            lines.append(f"  Observation: {obs[:400]}")
        lines.append("")

    return "\n".join(lines)


async def run_swe_judge(session_id: str, model: str = _JUDGE_MODEL) -> SweJudgeAnalysis:
    """Run the LLM judge on a SWE session. Results are cached."""
    from .judge_cache import get_cached_judge, save_judge_result

    cached = get_cached_judge(session_id)
    if cached is not None:
        decision_points = [
            SweJudgeDecisionPoint(**dp) for dp in cached.get("key_decision_points", [])
            if isinstance(dp, dict) and "step" in dp
        ]
        return SweJudgeAnalysis(
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
        return SweJudgeAnalysis(
            available=False, ran=False,
            error="openai package not installed. Run: pip install openai",
        )


    analysis = analyze_swe_failure(session_id)
    h = analysis.heuristic

    trajectory_text = _build_trajectory_text(session_id)

    if analysis.resolved:
        system_prompt = SWE_JUDGE_SYSTEM_RESOLVED
        user_msg = SWE_JUDGE_USER_RESOLVED.format(
            task_id=analysis.task_id,
            method=analysis.method,
            n_steps=analysis.n_steps,
            efficiency_score=h.efficiency_score,
            difficulties="; ".join(d.detail for d in h.difficulties_overcome) or "none detected",
            shortcuts="; ".join(h.shortcuts_taken) or "none detected",
            description=h.description,
            trajectory_text=trajectory_text,
        )
    else:
        system_prompt = SWE_JUDGE_SYSTEM_FAILED
        user_msg = SWE_JUDGE_USER_FAILED.format(
            task_id=analysis.task_id,
            method=analysis.method,
            n_steps=analysis.n_steps,
            failure_mode=h.failure_mode,
            failure_step=h.failure_step,
            description=h.description,
            trajectory_text=trajectory_text,
        )

    try:
        client = AsyncOpenAI(base_url=_JUDGE_BASE_URL, api_key=_JUDGE_API_KEY)
        response = await client.chat.completions.create(
            model=model,
            max_tokens=4096,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
    except Exception as e:
        return SweJudgeAnalysis(
            available=True, ran=True,
            error=f"Judge call failed: {str(e)[:200]}. Check OPENAI_API_KEY and OPENAI_BASE_URL.",
            model_used=model,
        )

    msg = response.choices[0].message
    raw_text = msg.content or ""

    text = _extract_json_text(raw_text)

    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        return SweJudgeAnalysis(
            available=True, ran=True,
            reasoning=raw_text[:500],
            error="Failed to parse judge JSON. Raw response in reasoning.",
            model_used=model,
            cost_usd=0.0,
        )

    decision_points = []
    for dp in data.get("key_decision_points", []):
        if isinstance(dp, dict) and "step" in dp:
            decision_points.append(SweJudgeDecisionPoint(
                step=dp["step"],
                decision=dp.get("decision", ""),
                better_alternative=dp.get("better_alternative", ""),
                confidence=dp.get("confidence", 0.0),
            ))


    judge_dict = {
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
    save_judge_result(session_id, "swe", judge_dict)

    return SweJudgeAnalysis(
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


async def stream_swe_judge(session_id: str, model: str = _JUDGE_MODEL):
    """Stream the SWE judge response token-by-token as SSE events.

    Results are cached in SQLite so re-runs don't waste tokens.
    """
    from .judge_cache import get_cached_judge, save_judge_result

    cached = get_cached_judge(session_id)
    if cached is not None:
        yield f"data: {_json.dumps({'type': 'done', 'judge': cached})}\n\n"
        return

    try:
        from openai import AsyncOpenAI
    except ImportError:
        yield f"data: {_json.dumps({'type': 'error', 'message': 'openai package not installed'})}\n\n"
        return

    analysis = analyze_swe_failure(session_id)
    h = analysis.heuristic
    trajectory_text = _build_trajectory_text(session_id)

    if analysis.resolved:
        system_prompt = SWE_JUDGE_SYSTEM_RESOLVED
        user_msg = SWE_JUDGE_USER_RESOLVED.format(
            task_id=analysis.task_id,
            method=analysis.method,
            n_steps=analysis.n_steps,
            efficiency_score=h.efficiency_score,
            difficulties="; ".join(d.detail for d in h.difficulties_overcome) or "none detected",
            shortcuts="; ".join(h.shortcuts_taken) or "none detected",
            description=h.description,
            trajectory_text=trajectory_text,
        )
    else:
        system_prompt = SWE_JUDGE_SYSTEM_FAILED
        user_msg = SWE_JUDGE_USER_FAILED.format(
            task_id=analysis.task_id,
            method=analysis.method,
            n_steps=analysis.n_steps,
            failure_mode=h.failure_mode,
            failure_step=h.failure_step,
            description=h.description,
            trajectory_text=trajectory_text,
        )

    try:
        client = AsyncOpenAI(base_url=_JUDGE_BASE_URL, api_key=_JUDGE_API_KEY)
        stream = await client.chat.completions.create(
            model=model,
            max_tokens=4096,
            temperature=0.3,
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
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            full_text += delta.content
            yield f"data: {_json.dumps({'type': 'token', 'text': delta.content})}\n\n"

    text = _extract_json_text(full_text)

    try:
        data = _json.loads(text)
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
            "model_used": model, "cost_usd": 0.0, "error": None,
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

    save_judge_result(session_id, "swe", judge_result)
    yield f"data: {_json.dumps({'type': 'done', 'judge': judge_result})}\n\n"
