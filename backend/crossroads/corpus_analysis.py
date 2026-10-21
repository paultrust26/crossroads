"""Corpus-level failure analysis and token waste mitigations.

Two views:
  1. Corpus failures — all sessions ranked by failure risk, batch judge streaming
  2. Corpus mitigations — waste patterns + actionable tactics (skills, pre-hooks, prompts)
"""

from __future__ import annotations

import json as _json
from typing import Optional

from pydantic import BaseModel, Field

from .failure_analysis import FailureMode, analyze_failure, run_judge
from .judge_cache import get_all_cached, get_cached_judge, save_judge_result
from .overview import SessionSummary, build_overview
from .store import discover, get_trajectory
from .swe_failure_analysis import analyze_swe_failure, run_swe_judge
from .swe_parser import build_swe_overview




class CorpusFailureEntry(BaseModel):
    session_id: str
    source: str
    project: str
    title: str = ""
    risk_score: float
    failure_mode: str
    confidence: float
    failure_step: Optional[int] = None
    n_steps: int
    n_errors: int = 0
    description: str
    judged: bool = False
    judge_summary: Optional[str] = None
    failure_category: Optional[str] = None
    severity: Optional[str] = None


class CorpusFailureResponse(BaseModel):
    entries: list[CorpusFailureEntry]
    n_total: int
    n_failed: int
    n_claude: int
    n_swe: int
    top_modes: dict[str, int] = Field(default_factory=dict)


_SEVERITY_WEIGHT = {
    "loop_to_death": 0.95,
    "error_cascade": 0.85,
    "wrong_approach_persisted": 0.8,
    "wrong_approach": 0.8,
    "lost_after_compaction": 0.75,
    "context_exhaustion": 0.7,
    "wrong_file": 0.65,
    "never_edited": 0.6,
    "abandoned": 0.5,
    "unknown": 0.4,
    "none": 0.0,
}


def _risk_score(failure_mode: str, confidence: float, n_errors: int, n_steps: int) -> float:
    mode_weight = _SEVERITY_WEIGHT.get(failure_mode, 0.4)
    error_rate = n_errors / max(n_steps, 1)
    return round(mode_weight * confidence + error_rate * 0.3, 4)


def build_corpus_failures(source: str = "all") -> CorpusFailureResponse:
    entries: list[CorpusFailureEntry] = []
    cached = get_all_cached()

    if source in ("all", "claude"):
        overview = build_overview()
        for sess in overview.sessions:
            traj = get_trajectory(sess.id, max_steps=400)
            if traj is None:
                continue
            try:
                analysis = analyze_failure(traj)
            except Exception:
                continue
            h = analysis.heuristic
            risk = _risk_score(h.failure_mode.value, h.confidence, sess.n_errors, sess.n_steps)
            entry = CorpusFailureEntry(
                session_id=sess.id,
                source="claude",
                project=sess.project,
                title=sess.title,
                risk_score=risk,
                failure_mode=h.failure_mode.value,
                confidence=h.confidence,
                failure_step=h.failure_step,
                n_steps=sess.n_steps,
                n_errors=sess.n_errors,
                description=h.description,
            )
            if sess.id in cached:
                j = cached[sess.id]
                entry.judged = True
                entry.judge_summary = j.get("why_it_failed")
                entry.failure_category = j.get("failure_category")
                entry.severity = j.get("severity")
            entries.append(entry)

    if source in ("all", "swe"):
        swe_overview = build_swe_overview()
        for sess in swe_overview.sessions:
            if not sess.has_trajectory:
                continue
            try:
                analysis = analyze_swe_failure(sess.id)
            except Exception:
                continue
            h = analysis.heuristic
            risk = _risk_score(h.failure_mode, h.confidence, sess.n_errors, sess.n_steps)
            entry = CorpusFailureEntry(
                session_id=sess.id,
                source="swe",
                project=sess.project,
                title=f"{sess.task_id} ({sess.method})",
                risk_score=risk,
                failure_mode=h.failure_mode,
                confidence=h.confidence,
                failure_step=h.failure_step,
                n_steps=sess.n_steps,
                n_errors=sess.n_errors,
                description=h.description,
            )
            if sess.id in cached:
                j = cached[sess.id]
                entry.judged = True
                entry.judge_summary = j.get("why_it_failed")
                entry.failure_category = j.get("failure_category")
                entry.severity = j.get("severity")
            entries.append(entry)

    entries.sort(key=lambda e: e.risk_score, reverse=True)

    mode_counts: dict[str, int] = {}
    for e in entries:
        if e.failure_mode != "none":
            mode_counts[e.failure_mode] = mode_counts.get(e.failure_mode, 0) + 1

    n_claude = sum(1 for e in entries if e.source == "claude")
    n_swe = sum(1 for e in entries if e.source == "swe")

    return CorpusFailureResponse(
        entries=entries,
        n_total=len(entries),
        n_failed=sum(1 for e in entries if e.failure_mode != "none"),
        n_claude=n_claude,
        n_swe=n_swe,
        top_modes=dict(sorted(mode_counts.items(), key=lambda kv: -kv[1])),
    )


async def stream_batch_judge(session_ids: list[str], source: str = "claude"):
    """Stream batch judge results as SSE — one per session, sequential.

    Uses SQLite cache: skips LLM call if result already exists.
    """
    total = len(session_ids)
    completed = 0

    for sid in session_ids:

        cached = get_cached_judge(sid)
        if cached is not None:
            completed += 1
            yield f"data: {_json.dumps({'type': 'result', 'session_id': sid, 'index': completed, 'total': total, 'judge': cached, 'cached': True})}\n\n"
            continue

        try:
            if source == "swe":
                judge_result = await run_swe_judge(sid)
                result_dict = judge_result.model_dump()
            else:
                traj = get_trajectory(sid, max_steps=400)
                if traj is None:
                    yield f"data: {_json.dumps({'type': 'skip', 'session_id': sid, 'reason': 'not found'})}\n\n"
                    continue
                judge_result = await run_judge(traj)
                result_dict = judge_result.model_dump()


            save_judge_result(sid, source, result_dict)

            completed += 1
            yield f"data: {_json.dumps({'type': 'result', 'session_id': sid, 'index': completed, 'total': total, 'judge': result_dict})}\n\n"
        except Exception as e:
            completed += 1
            yield f"data: {_json.dumps({'type': 'error', 'session_id': sid, 'message': str(e)[:200]})}\n\n"

    yield f"data: {_json.dumps({'type': 'done', 'total': completed})}\n\n"




class MitigationTactic(BaseModel):
    id: str
    target: str
    pattern: str
    severity: str
    title: str
    description: str
    artifact_type: str
    artifact_content: str
    artifact_filename: Optional[str] = None
    impact_estimate: str = ""


class CorpusWasteSummary(BaseModel):
    total_waste_usd: float
    waste_share_of_spend: float
    n_invalidations: int
    by_cause: dict[str, int] = Field(default_factory=dict)
    by_cause_cost: dict[str, float] = Field(default_factory=dict)
    top_waste_sessions: list[dict] = Field(default_factory=list)
    mean_reread_rate: float = 0.0
    sessions_with_high_reread: int = 0
    sessions_with_errors: int = 0
    error_cascade_count: int = 0


class CorpusMitigationResponse(BaseModel):
    waste: CorpusWasteSummary
    tactics: list[MitigationTactic]
    source: str


def _generate_claude_tactics(sessions: list[SessionSummary], waste: CorpusWasteSummary) -> list[MitigationTactic]:
    tactics: list[MitigationTactic] = []
    total_cost = sum(s.cost_usd for s in sessions) or 1.0


    ttl_count = waste.by_cause.get("ttl_expiry", 0)
    ttl_cost = waste.by_cause_cost.get("ttl_expiry", 0.0)
    if ttl_count > 0:
        pct = round(ttl_cost / waste.total_waste_usd * 100, 0) if waste.total_waste_usd else 0
        tactics.append(MitigationTactic(
            id="ttl-batch-ops",
            target="claude_code",
            pattern=f"TTL expiry accounts for {ttl_count} invalidations (${ttl_cost:.2f}, {pct:.0f}% of waste)",
            severity="high" if pct > 40 else "medium",
            title="Batch operations to avoid cache TTL expiry",
            description="The 5-minute Anthropic cache window expires during idle gaps. A pre-hook can remind the agent to batch related operations and avoid pauses.",
            artifact_type="skill",
            artifact_content="""---
description: Avoid cache TTL expiry by batching operations
---

# Cache-Aware Operation Batching

When working on multi-step tasks:
- Complete related file reads in sequence (don't pause between them)
- If you need to think, do it within the same turn — don't leave idle gaps
- Group related edits together rather than spacing them across turns
- After reading files for context, proceed immediately to the edit

The API cache expires after 5 minutes of inactivity. Every expiry forces a full context rebuild (~$0.02-0.05). Batch operations to stay within the window.""",
            artifact_filename=".claude/skills/avoid-ttl-waste.md",
            impact_estimate=f"Could save ~${ttl_cost:.2f}/session corpus ({pct:.0f}% of waste)",
        ))


    comp_count = waste.by_cause.get("compaction", 0)
    comp_cost = waste.by_cause_cost.get("compaction", 0.0)
    if comp_count > 0:
        pct = round(comp_cost / waste.total_waste_usd * 100, 0) if waste.total_waste_usd else 0
        tactics.append(MitigationTactic(
            id="compaction-resilience",
            target="claude_code",
            pattern=f"Compaction causes {comp_count} cache rebuilds (${comp_cost:.2f}, {pct:.0f}% of waste)",
            severity="high" if pct > 30 else "medium",
            title="Front-load critical context to survive compaction",
            description="After context compression, the agent loses cached state and must re-read files. Pin critical information in CLAUDE.md so it survives truncation.",
            artifact_type="skill",
            artifact_content="""---
description: Maintain critical state across context compactions
---

# Compaction Resilience

Before starting complex multi-file changes:
1. Write a brief plan to CLAUDE.md with file paths and what each needs
2. After each significant edit, update the plan with what's done vs remaining
3. If you notice you're re-reading files you already processed, check CLAUDE.md first

This ensures critical navigation state survives context compression. The plan acts as external memory that persists across compaction boundaries.""",
            artifact_filename=".claude/skills/compaction-resilience.md",
            impact_estimate=f"Could save ~${comp_cost:.2f}/session corpus ({pct:.0f}% of waste)",
        ))


    if waste.mean_reread_rate > 0.15:
        high_rr_sessions = waste.sessions_with_high_reread
        tactics.append(MitigationTactic(
            id="reduce-rereads",
            target="claude_code",
            pattern=f"Mean re-read rate is {waste.mean_reread_rate:.0%} across {high_rr_sessions} sessions",
            severity="medium",
            title="Pre-hook: workspace summary to reduce redundant reads",
            description="A pre-hook that generates a brief workspace summary before the session starts, giving the agent a map so it doesn't re-read files for orientation.",
            artifact_type="pre_hook",
            artifact_content="""#!/bin/bash
# .claude/hooks/pre-session-summary.sh
# Generates a workspace map that reduces redundant file reads

echo "## Workspace Summary (auto-generated)"
echo ""
echo "### Key files (by recent git activity):"
git log --oneline -20 --name-only --pretty=format: | sort | uniq -c | sort -rn | head -10 | awk '{print "- "$2" ("$1" recent changes)"}'
echo ""
echo "### Project structure:"
find . -name "*.py" -o -name "*.ts" -o -name "*.tsx" | head -30 | sort
echo ""
echo "### Recent changes:"
git diff --stat HEAD~3 2>/dev/null || echo "(no recent commits)"
""",
            artifact_filename=".claude/hooks/pre-session-summary.sh",
            impact_estimate=f"Could reduce re-reads in {high_rr_sessions} sessions",
        ))


    if waste.sessions_with_errors > 2:
        tactics.append(MitigationTactic(
            id="error-circuit-breaker",
            target="claude_code",
            pattern=f"{waste.sessions_with_errors} sessions have significant error cascades",
            severity="medium" if waste.sessions_with_errors < 5 else "high",
            title="Skill: circuit-breaker for repeated failures",
            description="When the agent hits 3+ consecutive errors, it should stop and reassess rather than continuing the same approach. This skill encodes that pattern.",
            artifact_type="skill",
            artifact_content="""---
description: Stop and reassess after repeated failures
---

# Error Circuit Breaker

If you encounter 3 or more consecutive errors (failed edits, test failures, command errors):

1. STOP the current approach immediately
2. Re-read the error messages carefully — look for the ROOT cause, not symptoms
3. Consider: "Am I editing the right file? Is my mental model correct?"
4. If the same test keeps failing after 3 attempts, read the test itself to understand what it expects
5. If a command keeps erroring, verify the environment/path assumptions

Never persist with a failing approach beyond 3 attempts. Change strategy instead.""",
            artifact_filename=".claude/skills/error-circuit-breaker.md",
            impact_estimate=f"Could prevent wasted recovery cycles in {waste.sessions_with_errors} sessions",
        ))


    sys_count = waste.by_cause.get("system_change", 0)
    sys_cost = waste.by_cause_cost.get("system_change", 0.0)
    if sys_count > 0:
        pct = round(sys_cost / waste.total_waste_usd * 100, 0) if waste.total_waste_usd else 0
        tactics.append(MitigationTactic(
            id="avoid-system-changes",
            target="claude_code",
            pattern=f"System changes caused {sys_count} invalidations (${sys_cost:.2f})",
            severity="low" if pct < 15 else "medium",
            title="Avoid mid-session permission/tool changes",
            description="Changing tool permissions, switching models, or modifying system configuration mid-session invalidates the entire cached prefix.",
            artifact_type="skill",
            artifact_content="""---
description: Avoid cache-breaking system changes mid-session
---

# Minimize System Changes

Avoid these mid-session actions that invalidate the entire cache:
- Don't change tool permissions after the session starts
- Don't switch models mid-conversation
- Don't modify .claude/settings.json during active work
- Set up all permissions and configurations BEFORE starting the task

Each system change forces a full context rebuild (~$0.02-0.10 depending on session length).""",
            artifact_filename=".claude/skills/avoid-system-changes.md",
            impact_estimate=f"Could save ~${sys_cost:.2f} from {sys_count} invalidations",
        ))

    return tactics


def _generate_swe_tactics(waste: CorpusWasteSummary) -> list[MitigationTactic]:
    tactics: list[MitigationTactic] = []


    tactics.append(MitigationTactic(
        id="observation-compression",
        target="swe",
        pattern="Observation tokens dominate SWE trajectories",
        severity="high",
        title="Compress tool observations",
        description="Compress long tool observations before serving them to the model, keeping errors and key lines, to reduce context pollution without losing critical information.",
        artifact_type="config",
        artifact_content="""# Observation compression configuration
# Apply tiered compression to tool observations

observation_compression:
  enabled: true
  strategy: "tiered"
  tiers:
    - name: "full"
      max_tokens: 200
      description: "Short outputs kept verbatim"
    - name: "summary"
      max_tokens: 2000
      description: "Medium outputs get key-line extraction"
    - name: "skeleton"
      max_tokens: null
      description: "Large outputs get structural skeleton only"

  # Tool-specific rules
  tools:
    bash:
      default_tier: "summary"
      preserve_patterns: ["error", "Error", "FAIL", "warning"]
      max_lines: 50
    str_replace_editor_view:
      default_tier: "skeleton"
      preserve_lines: [1, 2, 3]  # keep file header
      context_around_matches: 3
    browse:
      default_tier: "summary"
      max_chars: 4000""",
        artifact_filename="observation_compression.yaml",
        impact_estimate="Fewer observation tokens per task; check the resolve rate on your own tasks",
    ))


    tactics.append(MitigationTactic(
        id="planning-prompt",
        target="swe",
        pattern="Agents that fail often lack upfront planning — they dive into edits without understanding the codebase",
        severity="medium",
        title="Inject planning prompt before task execution",
        description="Prepend a planning instruction that forces the agent to read relevant files and form a hypothesis before making any edits.",
        artifact_type="prompt",
        artifact_content="""## Planning Phase (REQUIRED before any edits)

Before making ANY file edits, you MUST complete these steps:

1. **Understand the issue**: Re-read the issue description. Identify the specific behavior that needs to change.

2. **Locate relevant code**: Use search/browse to find:
   - The file(s) where the bug manifests
   - The test file(s) that verify the expected behavior
   - Any related utility functions or base classes

3. **Form a hypothesis**: State in one sentence what you think causes the issue and what change will fix it.

4. **Verify the hypothesis**: Read the specific function/method you plan to change. Confirm your hypothesis matches the code.

5. **Only then**: Make your edit. Run the tests. If they fail, go back to step 2 — do NOT retry the same edit.

Common mistakes to avoid:
- Editing a file you haven't read yet
- Making multiple unrelated changes at once
- Continuing to edit after 3 failed test runs without re-reading code""",
        artifact_filename="planning_prompt_injection.md",
        impact_estimate="May improve the resolve rate on tasks where planning failures dominate",
    ))


    tactics.append(MitigationTactic(
        id="repo-map-injection",
        target="swe",
        pattern="Agents waste steps browsing irrelevant files — a repo map reduces exploration waste",
        severity="medium",
        title="Inject repository structure map into context",
        description="Pre-compute a condensed repository map (key files, their roles, recent changes) and inject it at the start of the trajectory to reduce aimless exploration.",
        artifact_type="prompt",
        artifact_content="""## Repository Context (auto-injected)

{repo_map}

### Key Entry Points
Based on the issue, the most relevant files are likely:
{relevant_files}

### Testing
- Test command: `{test_command}`
- Relevant test files: {test_files}

Use this map to navigate directly to relevant code instead of browsing the full tree.""",
        artifact_filename="repo_map_template.md",
        impact_estimate="May reduce exploration steps",
    ))

    return tactics


def build_corpus_mitigations(source: str = "claude") -> CorpusMitigationResponse:
    """Build waste summary and generate mitigation tactics."""
    if source == "claude":
        overview = build_overview()
        sessions = overview.sessions
        total_cost = sum(s.cost_usd for s in sessions) or 1.0

        by_cause: dict[str, int] = {}
        by_cause_cost: dict[str, float] = {}
        for s in sessions:
            for c, v in (s.waste_by_cause or {}).items():
                by_cause[c] = by_cause.get(c, 0) + v
            for c, v in (s.waste_cost_by_cause or {}).items():
                by_cause_cost[c] = round(by_cause_cost.get(c, 0.0) + v, 4)

        total_waste = sum(s.waste_cost_usd for s in sessions)
        n_invalidations = sum(s.n_invalidations for s in sessions)
        mean_rr = sum(s.reread_rate for s in sessions) / max(len(sessions), 1)
        high_rr_sessions = sum(1 for s in sessions if s.reread_rate > 0.2)
        error_sessions = sum(1 for s in sessions if s.n_errors > 3)

        top_waste = sorted(sessions, key=lambda s: s.waste_cost_usd, reverse=True)
        top_waste_list = [
            {
                "session_id": s.id,
                "project": s.project,
                "waste_usd": round(s.waste_cost_usd, 4),
                "n_invalidations": s.n_invalidations,
                "reread_rate": round(s.reread_rate, 4),
                "n_steps": s.n_steps,
                "waste_causes": dict(s.waste_by_cause) if s.waste_by_cause else {},
            }
            for s in top_waste
        ]

        waste_summary = CorpusWasteSummary(
            total_waste_usd=round(total_waste, 2),
            waste_share_of_spend=round(total_waste / total_cost, 4),
            n_invalidations=n_invalidations,
            by_cause=by_cause,
            by_cause_cost=by_cause_cost,
            top_waste_sessions=top_waste_list,
            mean_reread_rate=round(mean_rr, 4),
            sessions_with_high_reread=high_rr_sessions,
            sessions_with_errors=error_sessions,
        )

        tactics = _generate_claude_tactics(sessions, waste_summary)

    else:

        swe_overview = build_swe_overview()
        sessions_swe = swe_overview.sessions
        total_obs_raw = sum(s.obs_tokens_raw for s in sessions_swe)
        total_obs_served = sum(s.obs_tokens_served for s in sessions_swe)
        savings = total_obs_raw - total_obs_served
        error_sessions = sum(1 for s in sessions_swe if s.n_errors > 3)

        waste_summary = CorpusWasteSummary(
            total_waste_usd=0.0,
            waste_share_of_spend=round(savings / max(total_obs_raw, 1), 4),
            n_invalidations=0,
            by_cause={"observation_bloat": len(sessions_swe)},
            by_cause_cost={},
            top_waste_sessions=[],
            mean_reread_rate=0.0,
            sessions_with_high_reread=0,
            sessions_with_errors=error_sessions,
        )

        tactics = _generate_swe_tactics(waste_summary)

    return CorpusMitigationResponse(
        waste=waste_summary,
        tactics=tactics,
        source=source,
    )
