"""Cross-session overview: a fast single-pass summary of every discoverable
Claude session, plus portfolio-level aggregates.

This deliberately does NOT use the full parser (which reads each file twice and
builds the whole Step list). A long session is ~30 ms to scan here, so the overview
of dozens of sessions is computed on demand and cached.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .settings import project_label
from .metrics import per_token_prices
from .parse_claude import list_sessions

_EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
_EXPLORE_TOOLS = {"Read", "Grep", "Glob", "WebFetch"}
_ERROR_MARKERS = (
    "error", "Error", "ENOENT", "Traceback", "command not found", "No such file",
    "failed", "can't open",
)
_PREV_MIN = 50_000
_DROP_TO = 0.3
_TTL_SECS = 300


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


class SessionSummary(BaseModel):
    id: str
    project: str
    title: str
    model: str | None = None
    started_at: str | None = None
    duration_secs: float = 0.0
    size_mb: float = 0.0
    n_steps: int = 0
    n_edits: int = 0
    n_errors: int = 0
    n_compactions: int = 0
    explore_steps: int = 0
    produce_steps: int = 0
    cost_usd: float = 0.0

    tokens_cache_read: int = 0
    tokens_cache_create: int = 0
    tokens_fresh_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    cache_read_share: float = 0.0

    n_reads: int = 0
    n_rereads: int = 0
    reread_rate: float = 0.0
    n_recovery_rereads: int = 0

    cost_cache_read: float = 0.0
    cost_cache_create: float = 0.0
    cost_fresh: float = 0.0
    cost_output: float = 0.0

    out_text_chars: int = 0
    out_thinking_chars: int = 0
    out_toolargs_chars: int = 0

    tool_stats: dict = Field(default_factory=dict)

    n_invalidations: int = 0
    waste_tokens: int = 0
    waste_cost_usd: float = 0.0
    waste_by_cause: dict = Field(default_factory=dict)
    waste_cost_by_cause: dict = Field(default_factory=dict)


class OverviewResponse(BaseModel):
    sessions: list[SessionSummary]

    n_sessions: int
    total_cost_usd: float
    total_steps: int
    total_compactions: int

    tokens_cache_read: int
    tokens_cache_create: int
    tokens_fresh_input: int
    tokens_output: int

    total_reads: int
    total_rereads: int
    mean_reread_rate: float
    total_recovery_rereads: int
    models: dict[str, int]
    projects: dict[str, int]


_CACHE: dict[str, SessionSummary] = {}


def _clean_project(p: str) -> str:
    return project_label(p)


def summarize_session(path: Path) -> SessionSummary:
    key = path.stem
    if key in _CACHE:
        return _CACHE[key]

    model = None
    first_ts = None
    last_ts = None
    steps = edits = errors = comps = explore = produce = 0
    cr = cc = fresh = out = 0
    reads = rereads = recovery = 0
    out_text = out_think = out_args = 0
    seen_files: set = set()
    steps_since_wall = 10_000
    tool_stats: dict = {}
    pr = per_token_prices("claude-sonnet-4.5")


    prev_cr = 0
    prev_ts = None
    inval_n = 0
    inval_tokens = 0
    inval_by_cause: dict = {}

    with path.open(errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            if d.get("isCompactSummary") or d.get("subtype") == "compact_boundary":
                comps += 1
                steps_since_wall = 0
            ty = d.get("type")
            msg = d.get("message") if isinstance(d.get("message"), dict) else None

            if ty == "assistant" and msg:
                if model is None:
                    model = msg.get("model")
                    pr = per_token_prices(model)
                u = msg.get("usage", {}) or {}
                content = msg.get("content")
                curr_cr = int(u.get("cache_read_input_tokens", 0) or 0)
                curr_cc = int(u.get("cache_creation_input_tokens", 0) or 0)
                curr_ts = _parse_ts(d.get("timestamp"))


                if prev_cr > _PREV_MIN and curr_cr < prev_cr * _DROP_TO:
                    secs = (curr_ts - prev_ts).total_seconds() if curr_ts and prev_ts else None
                    if secs is not None and secs > _TTL_SECS:
                        cause = "ttl_expiry"
                    elif steps_since_wall <= 2:
                        cause = "compaction"
                    else:
                        cause = "system_change"
                    inval_n += 1
                    inval_tokens += curr_cc
                    inval_by_cause[cause] = inval_by_cause.get(cause, 0) + 1
                prev_cr = curr_cr
                if curr_ts:
                    prev_ts = curr_ts
                    if first_ts is None:
                        first_ts = curr_ts
                    last_ts = curr_ts

                first = True
                if isinstance(content, list):
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        bt = b.get("type")
                        if bt == "text":
                            out_text += len(b.get("text", ""))
                        elif bt == "thinking":
                            out_think += len(b.get("thinking", ""))
                        if bt != "tool_use":
                            continue
                        steps += 1
                        steps_since_wall += 1
                        name = b.get("name", "")
                        inp = b.get("input", {}) or {}
                        try:
                            out_args += len(json.dumps(inp, ensure_ascii=False))
                        except (TypeError, ValueError):
                            pass
                        if name in _EDIT_TOOLS:
                            edits += 1
                            produce += 1
                        elif name in _EXPLORE_TOOLS:
                            explore += 1
                        if name == "Read":
                            fp = inp.get("file_path") or inp.get("path")
                            if fp:
                                reads += 1
                                if fp in seen_files:
                                    rereads += 1
                                    if steps_since_wall <= 20:
                                        recovery += 1
                                seen_files.add(fp)

                        t = tool_stats.setdefault(name or "?", {"calls": 0, "cr": 0, "cc": 0})
                        t["calls"] += 1
                        if first:
                            cr += curr_cr
                            cc += curr_cc
                            fresh += int(u.get("input_tokens", 0) or 0)
                            out += int(u.get("output_tokens", 0) or 0)
                            t["cr"] += curr_cr
                            t["cc"] += curr_cc
                            first = False
            elif ty == "user" and msg:
                content = msg.get("content")
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error"):
                            errors += 1

    cost = cr * pr["cache_read"] + cc * pr["cache_create"] + out * pr["output"] + fresh * pr["input"]
    total_tokens = cr + cc + fresh + out
    inval_cost = inval_tokens * pr["cache_create"]
    inval_cost_by_cause = {
        c: round(inval_by_cause[c] / max(1, inval_n) * inval_cost, 4) for c in inval_by_cause
    }

    proj = _clean_project(path.parent.name)
    dur = (last_ts - first_ts).total_seconds() if first_ts and last_ts else 0.0
    summary = SessionSummary(
        id=key,
        project=proj,
        title=f"{proj} · {key[:8]}",
        model=model,
        started_at=first_ts.isoformat() if first_ts else None,
        duration_secs=round(dur, 1),
        size_mb=round(path.stat().st_size / 1e6, 1),
        n_steps=steps,
        n_edits=edits,
        n_errors=errors,
        n_compactions=comps,
        explore_steps=explore,
        produce_steps=produce,
        cost_usd=round(cost, 4),
        tokens_cache_read=cr,
        tokens_cache_create=cc,
        tokens_fresh_input=fresh,
        tokens_output=out,
        tokens_total=total_tokens,
        cache_read_share=round(cr / total_tokens, 4) if total_tokens else 0.0,
        n_reads=reads,
        n_rereads=rereads,
        reread_rate=round(rereads / reads, 4) if reads else 0.0,
        n_recovery_rereads=recovery,
        cost_cache_read=round(cr * pr["cache_read"], 4),
        cost_cache_create=round(cc * pr["cache_create"], 4),
        cost_fresh=round(fresh * pr["input"], 4),
        cost_output=round(out * pr["output"], 4),
        out_text_chars=out_text,
        out_thinking_chars=out_think,
        out_toolargs_chars=out_args,
        tool_stats=tool_stats,
        n_invalidations=inval_n,
        waste_tokens=inval_tokens,
        waste_cost_usd=round(inval_cost, 4),
        waste_by_cause=inval_by_cause,
        waste_cost_by_cause=inval_cost_by_cause,
    )
    _CACHE[key] = summary
    return summary


def build_overview() -> OverviewResponse:
    sessions = []
    for info in list_sessions():
        try:
            sessions.append(summarize_session(Path(info["path"])))
        except OSError:
            continue

    sessions.sort(key=lambda s: s.cost_usd, reverse=True)

    n = len(sessions)
    models: dict[str, int] = {}
    projects: dict[str, int] = {}
    for s in sessions:
        m = s.model or "unknown"
        models[m] = models.get(m, 0) + 1
        projects[s.project] = projects.get(s.project, 0) + 1

    total_reads = sum(s.n_reads for s in sessions)
    total_rereads = sum(s.n_rereads for s in sessions)

    return OverviewResponse(
        sessions=sessions,
        n_sessions=n,
        total_cost_usd=round(sum(s.cost_usd for s in sessions), 2),
        total_steps=sum(s.n_steps for s in sessions),
        total_compactions=sum(s.n_compactions for s in sessions),
        tokens_cache_read=sum(s.tokens_cache_read for s in sessions),
        tokens_cache_create=sum(s.tokens_cache_create for s in sessions),
        tokens_fresh_input=sum(s.tokens_fresh_input for s in sessions),
        tokens_output=sum(s.tokens_output for s in sessions),
        total_reads=total_reads,
        total_rereads=total_rereads,
        mean_reread_rate=round(total_rereads / total_reads, 4) if total_reads else 0.0,
        total_recovery_rereads=sum(s.n_recovery_rereads for s in sessions),
        models=dict(sorted(models.items(), key=lambda kv: -kv[1])),
        projects=dict(sorted(projects.items(), key=lambda kv: -kv[1])),
    )




class CorpusAction(BaseModel):
    tool: str
    calls: int
    call_share: float
    read_share: float
    create_share: float
    benefit_ratio: float


class CorpusResponse(BaseModel):
    n_sessions: int

    tokens_cache_read: int
    tokens_cache_create: int
    tokens_fresh_input: int
    tokens_output: int
    vol_cache_read: float
    vol_cache_create: float
    vol_fresh: float
    vol_output: float

    cost_total_usd: float
    cost_cache_read_share: float
    cost_cache_create_share: float
    cost_fresh_share: float
    cost_output_share: float

    out_text_share: float
    out_thinking_share: float
    out_toolargs_share: float

    actions: list[CorpusAction]

    waste_cost_usd: float
    waste_tokens: int
    waste_share_of_create: float
    n_invalidations: int
    waste_by_cause: dict[str, int]
    waste_cost_by_cause: dict[str, float]

    total_steps: int
    total_errors: int
    error_rate: float
    sessions_with_errors: int


def build_corpus() -> CorpusResponse:
    sessions = []
    for info in list_sessions():
        try:
            sessions.append(summarize_session(Path(info["path"])))
        except OSError:
            continue

    cr = sum(s.tokens_cache_read for s in sessions)
    cc = sum(s.tokens_cache_create for s in sessions)
    fr = sum(s.tokens_fresh_input for s in sessions)
    ou = sum(s.tokens_output for s in sessions)
    vol_total = cr + cc + fr + ou or 1

    cost_cr = sum(s.cost_cache_read for s in sessions)
    cost_cc = sum(s.cost_cache_create for s in sessions)
    cost_fr = sum(s.cost_fresh for s in sessions)
    cost_ou = sum(s.cost_output for s in sessions)
    cost_total = cost_cr + cost_cc + cost_fr + cost_ou or 1e-9

    o_text = sum(s.out_text_chars for s in sessions)
    o_think = sum(s.out_thinking_chars for s in sessions)
    o_args = sum(s.out_toolargs_chars for s in sessions)
    o_total = o_text + o_think + o_args or 1


    merged: dict[str, dict] = {}
    for s in sessions:
        for tool, t in (s.tool_stats or {}).items():
            m = merged.setdefault(tool, {"calls": 0, "cr": 0, "cc": 0})
            m["calls"] += t.get("calls", 0)
            m["cr"] += t.get("cr", 0)
            m["cc"] += t.get("cc", 0)
    n_calls = sum(m["calls"] for m in merged.values()) or 1
    actions = [
        CorpusAction(
            tool=tool,
            calls=m["calls"],
            call_share=m["calls"] / n_calls,
            read_share=m["cr"] / (cr or 1),
            create_share=m["cc"] / (cc or 1),
            benefit_ratio=(m["cr"] / m["cc"]) if m["cc"] else float(m["cr"] > 0) * 999.0,
        )
        for tool, m in merged.items()
    ]
    actions.sort(key=lambda a: a.calls, reverse=True)

    waste_tokens = sum(s.waste_tokens for s in sessions)
    waste_cost = sum(s.waste_cost_usd for s in sessions)
    by_cause: dict[str, int] = {}
    cost_by_cause: dict[str, float] = {}
    for s in sessions:
        for c, v in (s.waste_by_cause or {}).items():
            by_cause[c] = by_cause.get(c, 0) + v
        for c, v in (s.waste_cost_by_cause or {}).items():
            cost_by_cause[c] = round(cost_by_cause.get(c, 0.0) + v, 4)

    total_steps = sum(s.n_steps for s in sessions)
    total_errors = sum(s.n_errors for s in sessions)

    return CorpusResponse(
        n_sessions=len(sessions),
        tokens_cache_read=cr, tokens_cache_create=cc, tokens_fresh_input=fr, tokens_output=ou,
        vol_cache_read=cr / vol_total, vol_cache_create=cc / vol_total,
        vol_fresh=fr / vol_total, vol_output=ou / vol_total,
        cost_total_usd=round(cost_total, 2),
        cost_cache_read_share=cost_cr / cost_total, cost_cache_create_share=cost_cc / cost_total,
        cost_fresh_share=cost_fr / cost_total, cost_output_share=cost_ou / cost_total,
        out_text_share=o_text / o_total, out_thinking_share=o_think / o_total,
        out_toolargs_share=o_args / o_total,
        actions=actions,
        waste_cost_usd=round(waste_cost, 2),
        waste_tokens=waste_tokens,
        waste_share_of_create=round(waste_tokens / (cc or 1), 4),
        n_invalidations=sum(s.n_invalidations for s in sessions),
        waste_by_cause=dict(sorted(by_cause.items(), key=lambda kv: -kv[1])),
        waste_cost_by_cause=cost_by_cause,
        total_steps=total_steps,
        total_errors=total_errors,
        error_rate=round(total_errors / total_steps, 4) if total_steps else 0.0,
        sessions_with_errors=sum(1 for s in sessions if s.n_errors > 0),
    )
