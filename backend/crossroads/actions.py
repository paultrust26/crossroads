"""Actions — surface patterns from analysis into actionable CLAUDE.md rules or hooks.

Extracts:
  1. Repeated user instructions across sessions (things users keep telling the agent)
  2. Top failure patterns → prevention rules for CLAUDE.md
  3. Token waste patterns → efficiency rules for CLAUDE.md
  4. Anomaly patterns → safety rules for CLAUDE.md
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class ActionSuggestion(BaseModel):
    id: str
    source: str
    category: str
    severity: str
    title: str
    description: str
    frequency: int = 1
    evidence: list[str] = Field(default_factory=list)
    artifact: str = ""
    artifact_type: str = "claude_md"


class ActionsResponse(BaseModel):
    suggestions: list[ActionSuggestion]
    n_sessions_analyzed: int
    n_queries_analyzed: int




_INSTRUCTION_PATTERNS = [
    (r"\balways\b", "always"),
    (r"\bnever\b", "never"),
    (r"\bdon'?t\b", "dont"),
    (r"\bdo not\b", "dont"),
    (r"\bmake sure\b", "make_sure"),
    (r"\bremember\b", "remember"),
    (r"\bplease\s+(always|never|don'?t|do not|make sure)\b", "please_directive"),
    (r"\bfrom now on\b", "from_now_on"),
    (r"\bevery time\b", "every_time"),
    (r"\bwhenever\b", "whenever"),
    (r"\bin the future\b", "in_future"),
    (r"\buse .+ instead of\b", "use_instead"),
    (r"\bprefer\b", "prefer"),
    (r"\bavoid\b", "avoid"),
    (r"\bstop\b", "stop"),
    (r"\bkeep\b.+\b(short|concise|brief|minimal)\b", "keep_concise"),
]

_NOISE_QUERIES = {
    "yes", "no", "y", "n", "ok", "okay", "sure", "thanks", "thank you",
    "continue", "go ahead", "done", "good",
}


def _is_instruction_like(query: str) -> bool:
    """Check if a query contains directive language."""
    q = query.lower().strip()
    if len(q) < 10 or q in _NOISE_QUERIES:
        return False
    for pat, _ in _INSTRUCTION_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return True
    return False


def _normalize_instruction(text: str) -> str:
    """Normalize for dedup: lowercase, collapse whitespace, strip punctuation."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t[:200]


def _cluster_instructions(queries: list[str]) -> list[dict[str, Any]]:
    """Group similar instructions by normalized form."""
    clusters: dict[str, list[str]] = {}
    for q in queries:
        norm = _normalize_instruction(q)

        key = norm[:60]
        found = False
        for existing_key in list(clusters.keys()):
            if _similarity(key, existing_key) > 0.6:
                clusters[existing_key].append(q)
                found = True
                break
        if not found:
            clusters[key] = [q]

    results = []
    for key, examples in clusters.items():
        if len(examples) >= 2:
            results.append({
                "text": examples[0][:200],
                "count": len(examples),
                "examples": examples[:5],
            })
    return sorted(results, key=lambda r: r["count"], reverse=True)


def _similarity(a: str, b: str) -> float:
    """Simple word-overlap Jaccard similarity."""
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def extract_user_instructions(session_paths: list[Path]) -> list[dict[str, Any]]:
    """Extract directive-like queries from all sessions."""
    from .parse_claude import _iter_records

    all_instructions: list[str] = []
    n_queries = 0

    for path in session_paths:
        try:
            for rec in _iter_records(path):
                if rec.get("type") != "user":
                    continue
                msg = rec.get("message", {})
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                text = None
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                        continue
                    parts = [b.get("text", "") for b in content if isinstance(b, dict) if b.get("type") == "text"]
                    text = "\n".join(parts)
                if not text or not text.strip():
                    continue
                t = text.strip()
                if t.lower() in _NOISE_QUERIES:
                    continue
                n_queries += 1
                if _is_instruction_like(t):
                    all_instructions.append(t[:300])
        except Exception:
            continue

    clusters = _cluster_instructions(all_instructions)
    return clusters




def build_failure_actions(failure_data: dict[str, Any]) -> list[ActionSuggestion]:
    """Generate actions from corpus failure patterns."""
    actions: list[ActionSuggestion] = []
    top_modes = failure_data.get("top_modes", {})

    mode_rules = {
        "loop_to_death": {
            "title": "Add loop detection rule",
            "description": "Agent frequently enters loops repeating the same failed approach",
            "artifact": "- If you've tried the same approach 3 times without success, stop and try a fundamentally different strategy. Do not repeat failed commands with minor variations.",
        },
        "error_cascade": {
            "title": "Add error recovery rule",
            "description": "Agent keeps going after encountering errors instead of diagnosing the root cause",
            "artifact": "- When you encounter an error, diagnose the root cause before attempting a fix. Do not make changes to suppress errors without understanding why they occur.",
        },
        "wrong_file": {
            "title": "Add file verification rule",
            "description": "Agent frequently edits the wrong file or looks in the wrong location",
            "artifact": "- Before editing a file, verify it's the correct one by checking imports, class names, or function signatures. Use grep to confirm the location of the code you need to change.",
        },
        "wrong_approach": {
            "title": "Add approach validation rule",
            "description": "Agent pursues incorrect solution strategies",
            "artifact": "- Before implementing a solution, briefly verify your understanding of the problem. If unsure about the correct approach, read more context first rather than guessing.",
        },
        "incomplete_fix": {
            "title": "Add completeness check rule",
            "description": "Agent submits incomplete fixes that don't fully address the issue",
            "artifact": "- After making a fix, verify completeness: check all call sites, run related tests, and confirm the fix handles edge cases mentioned in the issue.",
        },
        "test_blindness": {
            "title": "Add test awareness rule",
            "description": "Agent ignores or doesn't properly use test results",
            "artifact": "- Always run relevant tests after making changes. If tests fail, read the full error output carefully before attempting another fix.",
        },
    }

    for mode, count in sorted(top_modes.items(), key=lambda x: x[1], reverse=True):
        if mode in mode_rules and count >= 2:
            r = mode_rules[mode]
            actions.append(ActionSuggestion(
                id=f"failure-{mode}",
                source="failure_pattern",
                category="claude_md_rule",
                severity="high" if count >= 5 else "medium",
                title=r["title"],
                description=f"{r['description']} ({count} occurrences across sessions)",
                frequency=count,
                evidence=[f"{count} sessions exhibited '{mode.replace('_', ' ')}' pattern"],
                artifact=r["artifact"],
                artifact_type="claude_md",
            ))

    return actions


def build_waste_actions(waste_data: dict[str, Any]) -> list[ActionSuggestion]:
    """Generate actions from token waste patterns."""
    actions: list[ActionSuggestion] = []
    by_cause = waste_data.get("by_cause", {})
    by_cause_cost = waste_data.get("by_cause_cost", {})
    total_waste = waste_data.get("total_waste_usd", 0)

    cause_rules = {
        "ttl_expiry": {
            "title": "Batch operations to avoid cache expiry",
            "description": "Cache expires during idle gaps between operations",
            "artifact": "- Complete related file operations in sequence without pauses. Group reads together, then edits together. Don't leave idle gaps between related steps.",
            "hook": "# .claude/hooks/pre-tool.sh\n# Warn when gap between tool calls exceeds cache TTL\necho 'Reminder: batch related operations to avoid cache TTL expiry' >&2",
        },
        "edit_invalidation": {
            "title": "Minimize unnecessary cache invalidations from edits",
            "description": "Edits to files cause cache invalidation of previously read content",
            "artifact": "- Read all the context you need before starting edits. Once you begin editing, avoid re-reading files you've already read unless the edit changed them.",
        },
        "compaction_regrowth": {
            "title": "Reduce post-compaction re-reading",
            "description": "After context compaction, agent re-reads files it had already processed",
            "artifact": "- After context is compacted, focus on the task at hand rather than re-reading all previously accessed files. Only re-read files directly relevant to your next action.",
        },
        "redundant_reads": {
            "title": "Avoid reading the same file multiple times",
            "description": "Agent re-reads files it has already seen without any intervening changes",
            "artifact": "- Do not re-read a file you've already read in this session unless it has been modified since your last read. Trust your memory of file contents.",
        },
    }

    for cause, count in sorted(by_cause.items(), key=lambda x: x[1], reverse=True):
        cost = by_cause_cost.get(cause, 0)
        if cause in cause_rules and count >= 2:
            r = cause_rules[cause]
            pct = (cost / total_waste * 100) if total_waste > 0 else 0
            actions.append(ActionSuggestion(
                id=f"waste-{cause}",
                source="waste_pattern",
                category="claude_md_rule",
                severity="high" if pct > 30 else "medium" if pct > 10 else "low",
                title=r["title"],
                description=f"{r['description']} ({count} invalidations, ${cost:.2f}, {pct:.0f}% of waste)",
                frequency=count,
                evidence=[f"{count} occurrences costing ${cost:.2f}"],
                artifact=r["artifact"],
                artifact_type="claude_md",
            ))


    mean_reread = waste_data.get("mean_reread_rate", 0)
    if mean_reread > 0.3:
        actions.append(ActionSuggestion(
            id="waste-high-reread",
            source="waste_pattern",
            category="claude_md_rule",
            severity="high" if mean_reread > 0.5 else "medium",
            title="Reduce file re-read rate",
            description=f"Average re-read rate is {mean_reread * 100:.0f}% — nearly 1 in {int(1 / mean_reread) if mean_reread > 0 else 0} reads is redundant",
            frequency=int(mean_reread * 100),
            evidence=[f"Mean re-read rate: {mean_reread * 100:.0f}%"],
            artifact="- Before reading a file, check if you've already read it in this session. Only re-read if the file was modified since your last read or if context was compacted.",
            artifact_type="claude_md",
        ))

    return actions


def build_anomaly_actions(anomaly_data: dict[str, Any]) -> list[ActionSuggestion]:
    """Generate actions from anomaly audit patterns."""
    actions: list[ActionSuggestion] = []
    sessions = anomaly_data.get("sessions", [])

    kind_counts: Counter = Counter()
    for s in sessions:
        for k in s.get("anomaly_kinds", []):
            kind_counts[k] += 1

    kind_rules = {
        "scope_creep": {
            "title": "Stay within requested scope",
            "description": "Agent frequently does extra work beyond what was asked",
            "artifact": "- Only do what the user explicitly asked. Do not refactor, rename, or reorganize code that isn't directly related to the task. If you see improvements, mention them but don't implement without permission.",
        },
        "unauthorized_deletion": {
            "title": "Never delete without explicit permission",
            "description": "Agent deletes files or code without being asked to",
            "artifact": "- Never delete files, functions, or significant code blocks unless the user explicitly asked for deletion. If code seems unused, ask before removing it.",
        },
        "ignored_instruction": {
            "title": "Respect explicit user constraints",
            "description": "Agent ignores explicit instructions like 'don't touch X'",
            "artifact": "- When the user says 'don't', 'avoid', 'leave X alone', or similar, treat this as a hard constraint. Do not modify the specified files or components even if you think it would help.",
        },
        "unasked_refactor": {
            "title": "No unrequested refactoring",
            "description": "Agent refactors code that wasn't part of the request",
            "artifact": "- Do not refactor, rename variables, or reorganize imports unless the user specifically asked for it. A bug fix should only fix the bug, not improve surrounding code.",
        },
        "intent_mismatch": {
            "title": "Verify understanding before acting",
            "description": "Agent's actions don't match what the user intended",
            "artifact": "- If the user's request is ambiguous, ask for clarification rather than guessing. Re-read the user's message carefully before starting work.",
        },
    }

    for kind, count in kind_counts.most_common():
        if kind in kind_rules and count >= 2:
            r = kind_rules[kind]
            actions.append(ActionSuggestion(
                id=f"anomaly-{kind}",
                source="anomaly_pattern",
                category="claude_md_rule",
                severity="high" if count >= 5 else "medium",
                title=r["title"],
                description=f"{r['description']} ({count} occurrences)",
                frequency=count,
                evidence=[f"{count} sessions flagged for '{kind.replace('_', ' ')}'"],
                artifact=r["artifact"],
                artifact_type="claude_md",
            ))

    return actions


def build_user_instruction_actions(clusters: list[dict[str, Any]]) -> list[ActionSuggestion]:
    """Generate actions from repeated user instructions."""
    actions: list[ActionSuggestion] = []

    for i, cluster in enumerate(clusters[:10]):
        text = cluster["text"]
        count = cluster["count"]
        examples = cluster.get("examples", [])


        artifact = f"- {text.strip().rstrip('.')}"
        if not artifact.endswith("."):
            artifact += "."

        actions.append(ActionSuggestion(
            id=f"user-instruction-{i}",
            source="user_instruction",
            category="claude_md_rule",
            severity="high" if count >= 5 else "medium" if count >= 3 else "low",
            title=f'User repeatedly says: "{text[:60]}{"…" if len(text) > 60 else ""}"',
            description=f"Found {count} times across sessions. Consider adding to CLAUDE.md so the agent follows this by default.",
            frequency=count,
            evidence=examples[:3],
            artifact=artifact,
            artifact_type="claude_md",
        ))

    return actions


def build_actions(session_paths: list[Path], failure_data: dict, waste_data: dict, anomaly_data: dict) -> ActionsResponse:
    """Build all action suggestions."""
    suggestions: list[ActionSuggestion] = []


    clusters = extract_user_instructions(session_paths)
    suggestions.extend(build_user_instruction_actions(clusters))


    suggestions.extend(build_failure_actions(failure_data))


    suggestions.extend(build_waste_actions(waste_data))


    suggestions.extend(build_anomaly_actions(anomaly_data))


    sev_order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: (sev_order.get(s.severity, 3), -s.frequency))

    n_queries = sum(1 for _ in _count_queries(session_paths))

    return ActionsResponse(
        suggestions=suggestions,
        n_sessions_analyzed=len(session_paths),
        n_queries_analyzed=n_queries,
    )


def _count_queries(paths: list[Path]):
    """Cheap query counter."""
    from .parse_claude import _iter_records
    for path in paths:
        try:
            for rec in _iter_records(path):
                if rec.get("type") == "user":
                    msg = rec.get("message", {})
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, list):
                            if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content): continue
                            yield 1
                        elif isinstance(content, str) and content.strip():
                            yield 1
        except Exception:
            continue
