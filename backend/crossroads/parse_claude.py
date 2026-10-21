"""Parser for Claude Code session transcripts (~/.claude/projects/<proj>/<id>.jsonl).

This is the primary target substrate for Crossroads. Each line is a JSON record.
We care about:
  * type == "assistant" with a message.content list containing `tool_use` blocks
    -> these are the agent's *decisions* (steps).
  * type == "user" with content containing `tool_result` blocks
    -> the observation/outcome for the preceding tool_use (matched by tool_use_id).
  * compaction boundaries: records with isCompactSummary == true, or
    message.content / subtype == "compact_boundary" -> 'amnesia walls'.

Outcome of a Claude session is generally unknown (no pass/fail oracle), so we set
Outcome.unknown and let the criticality index (fragility/recoverability surrogate)
carry the analysis. We never fabricate an outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .analyze import (
    detect_loops,
    detect_rereads,
    infer_phase,
    is_test_text,
    mark_compaction_proximity,
)
from .settings import SESSIONS_DIR, project_label
from .metrics import classify_rereads, compute_session_metrics, step_cost_usd
from .models import (
    CacheBreakdown,
    CompactionEvent,
    Outcome,
    Source,
    Step,
    StepSignals,
    Trajectory,
    TrajectoryMeta,
)

CLAUDE_ROOT = SESSIONS_DIR


_EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
_ERROR_MARKERS = ("error", "Error", "ENOENT", "Traceback", "command not found",
                  "No such file", "failed", "can't open")


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _text_of(content: Any) -> str:
    """Flatten message.content (str or list of blocks) into searchable text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict): continue
            if b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif b.get("type") == "thinking":
                parts.append(b.get("thinking", ""))
        return "\n".join(parts)
    return ""


def _arg_text(inp: Any) -> str:
    try:
        return json.dumps(inp, ensure_ascii=False)
    except Exception:
        return str(inp)


def _summary(tool: str, inp: dict[str, Any]) -> str:
    if not isinstance(inp, dict):
        return tool
    for key in ("command", "file_path", "path", "pattern", "query", "url", "prompt"):
        if key in inp and inp[key]:
            val = str(inp[key]).replace("\n", " ")
            return f"{tool}: {val[:80]}"
    return tool


def _primary_file(tool: str, inp: dict[str, Any]) -> str | None:
    if not isinstance(inp, dict):
        return None
    for key in ("file_path", "path", "notebook_path"):
        if inp.get(key):
            return str(inp[key])
    return None


def _result_is_error(block: dict[str, Any]) -> tuple[bool, str]:
    if block.get("is_error"):
        return True, _result_text(block)
    txt = _result_text(block)
    head = txt[:400]
    if any(m in head for m in _ERROR_MARKERS):
        return True, txt
    return False, txt


def _result_text(block: dict[str, Any]) -> str:
    c = block.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "\n".join(parts)
    return ""


def _cache_from_usage(usage: dict[str, Any]) -> dict[str, int]:
    """Extract the token breakdown from an Anthropic usage object."""
    cc = usage.get("cache_creation", {}) or {}
    return {
        "cache_read_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
        "cache_creation_tokens": int(usage.get("cache_creation_input_tokens", 0) or 0),
        "cache_creation_5m_tokens": int(cc.get("ephemeral_5m_input_tokens", 0) or 0),
        "cache_creation_1h_tokens": int(cc.get("ephemeral_1h_input_tokens", 0) or 0),
        "fresh_input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
    }


def _is_compaction(rec: dict[str, Any]) -> bool:
    if rec.get("isCompactSummary") is True:
        return True
    if rec.get("subtype") == "compact_boundary":
        return True
    msg = rec.get("message")
    if isinstance(msg, dict) and msg.get("isCompactSummary"):
        return True
    return False


def parse_claude_session(path: Path, max_steps: int | None = None) -> Trajectory:
    """Parse one Claude session JSONL into a normalized Trajectory."""
    path = Path(path)



    results: dict[str, dict[str, Any]] = {}
    for rec in _iter_records(path):
        if rec.get("type") != "user":
            continue
        msg = rec.get("message", {})
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tid = b.get("tool_use_id")
                    if tid:
                        results[tid] = b

    steps: list[Step] = []
    compactions: list[CompactionEvent] = []
    model: str | None = None
    total_tokens = 0
    last_thought = ""

    step_idx = 0
    for rec in _iter_records(path):
        if _is_compaction(rec):

            msg = rec.get("message", {})
            if isinstance(msg, dict):
                preview = _text_of(msg.get("content", ""))
            else:
                preview = _text_of(rec.get("summary", ""))
            compactions.append(CompactionEvent(at_index=step_idx, summary_preview=preview[:1200]))
            continue

        if rec.get("type") != "assistant":
            continue
        msg = rec.get("message", {})
        if not isinstance(msg, dict):
            continue
        if model is None:
            model = msg.get("model")
        content = msg.get("content")
        usage = msg.get("usage", {}) or {}
        cache_msg = _cache_from_usage(usage)
        out_tok = cache_msg["output_tokens"]
        ts_here = rec.get("timestamp")


        msg_text_chars = msg_think_chars = 0
        if isinstance(content, list):
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "text":
                    msg_text_chars += len(blk.get("text", ""))
                elif blk.get("type") == "thinking":
                    msg_think_chars += len(blk.get("thinking", ""))


        thought_here = _text_of(content)
        if thought_here.strip():
            last_thought = thought_here.strip()

        if not isinstance(content, list):
            continue
        first_tool_use_in_msg = True
        for b in content:
            if not isinstance(b, dict) or b.get("type") != "tool_use":
                continue
            tool = b.get("name", "")
            inp = b.get("input", {}) or {}
            tid = b.get("id")
            arg_text = _arg_text(inp)

            res = results.get(tid, {})
            is_err, obs_text = _result_is_error(res) if res else (False, "")

            is_edit = tool in _EDIT_TOOLS
            is_test = is_test_text(arg_text) or is_test_text(obs_text[:200])
            test_failed = is_test and is_err
            obs_tok = _approx_tokens(obs_text)
            total_tokens += out_tok + obs_tok



            if first_tool_use_in_msg:
                cache = CacheBreakdown(**cache_msg)
                text_chars = msg_text_chars
                thinking_chars = msg_think_chars
                first_tool_use_in_msg = False
            else:
                cache = CacheBreakdown()
                text_chars = thinking_chars = 0

            phase = infer_phase(tool, arg_text, is_edit, is_test, is_err)
            sig = StepSignals(
                is_error=is_err,
                is_edit=is_edit,
                is_test=is_test,
                test_failed=test_failed,
                file_path=_primary_file(tool, inp),
            )
            step = Step(
                index=step_idx,
                phase=phase,
                tool=tool,
                summary=_summary(tool, inp),
                thought=last_thought,
                observation_preview=obs_text,
                tokens=out_tok + obs_tok,
                cache=cache,
                signals=sig,
                text_chars=text_chars,
                thinking_chars=thinking_chars,
                tool_args_chars=len(arg_text),
                ts=ts_here,
            )
            step.cost_usd = round(step_cost_usd(step, model), 6)
            steps.append(step)
            last_thought = ""
            step_idx += 1
            if max_steps and step_idx >= max_steps:
                break
        if max_steps and step_idx >= max_steps:
            break


    seen_walls: set[int] = set()
    deduped: list[CompactionEvent] = []
    for c in compactions:
        if c.at_index in seen_walls:
            continue
        seen_walls.add(c.at_index)
        deduped.append(c)
    compactions = deduped


    detect_rereads(steps)
    detect_loops(steps)
    mark_compaction_proximity(steps, compactions)

    meta = TrajectoryMeta(
        id=path.stem,
        source=Source.claude,
        title=_session_title(path, steps),
        model=model,
        n_steps=len(steps),
        outcome=Outcome.unknown,
        n_compactions=len(compactions),
        n_errors=sum(1 for s in steps if s.signals.is_error),
        n_edits=sum(1 for s in steps if s.signals.is_edit),
        total_tokens=total_tokens,
    )
    traj = Trajectory(meta=meta, steps=steps, compactions=compactions)


    classify_rereads(traj)
    traj.meta.metrics = compute_session_metrics(traj)
    return traj


def _approx_tokens(text: str) -> int:
    return len(text) // 4 if text else 0


def _session_title(path: Path, steps: list[Step]) -> str:
    proj = project_label(path.parent.name)
    return f"{proj} · {path.stem[:8]} ({len(steps)} steps)"




def list_sessions(root: Path = CLAUDE_ROOT, min_steps: int = 20, limit: int = 60) -> list[dict[str, Any]]:
    """Cheap listing: scan session files, report size-based step estimate without
    full parse. Used to populate the trajectory picker."""
    out = []
    if not root.exists():
        return out
    files = sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_size, reverse=True)
    for p in files:
        if "subagents" in p.parts:
            continue
        size = p.stat().st_size
        if size < 200_000:
            continue
        out.append({
            "id": p.stem,
            "path": str(p),
            "project": p.parent.name,
            "size_bytes": size,
        })
        if len(out) >= limit:
            break
    return out
