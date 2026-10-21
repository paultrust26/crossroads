"""Token-economics analyses, mirroring the "Where Do the Tokens Go?" paper.

Three exploratory lenses, all computed from real trace signals (no LLM judge):
  1. token_budget       — RQ1: where tokens go (input types + output subtypes),
                          and cache dynamics by action type (paper Tables 1 & 3).
  2. waste / invalidation — the single largest avoidable cost: cache *invalidation*,
                          where the cached prefix collapses and must be rebuilt.
  3. failures           — free failure signals (tool errors, retries, failed tests).

Semantic questions (task complexity, interruption type, outcome) need an LLM judge;
those are NOT computed here — `judge.estimate_*` reports what running one would cost.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from .metrics import per_token_prices
from .models import Step, Trajectory



class TokenBudget(BaseModel):

    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    fresh_input_tokens: int = 0
    output_tokens: int = 0

    cache_read_share: float = 0.0
    cache_create_share: float = 0.0
    fresh_input_share: float = 0.0
    output_share: float = 0.0

    output_text_share: float = 0.0
    output_thinking_share: float = 0.0
    output_toolargs_share: float = 0.0

    cost_total_usd: float = 0.0
    cost_cache_read_share: float = 0.0
    cost_cache_create_share: float = 0.0
    cost_fresh_share: float = 0.0
    cost_output_share: float = 0.0


class ActionCache(BaseModel):
    """Cache dynamics for one action/tool type (paper Table 3)."""
    tool: str
    calls: int
    call_share: float
    cache_read: int
    cache_create: int
    read_share: float
    create_share: float
    benefit_ratio: float


class TokenBudgetResponse(BaseModel):
    budget: TokenBudget
    actions: list[ActionCache]


def _model_of(traj: Trajectory) -> str | None:
    return traj.meta.model


def token_budget(traj: Trajectory) -> TokenBudgetResponse:
    cr = cc = fr = out = 0
    text_c = think_c = args_c = 0
    by_tool: dict[str, dict] = {}

    for st in traj.steps:
        c = st.cache
        cr += c.cache_read_tokens
        cc += c.cache_creation_tokens
        fr += c.fresh_input_tokens
        out += c.output_tokens
        text_c += st.text_chars
        think_c += st.thinking_chars
        args_c += st.tool_args_chars

        t = by_tool.setdefault(st.tool or "?", {"calls": 0, "cr": 0, "cc": 0})
        t["calls"] += 1
        t["cr"] += c.cache_read_tokens
        t["cc"] += c.cache_creation_tokens

    total = (cr + cc + fr + out) or 1
    out_chars = (text_c + think_c + args_c) or 1

    pr = per_token_prices(_model_of(traj))
    cost_cr = cr * pr["cache_read"]
    cost_cc = cc * pr["cache_create"]
    cost_fr = fr * pr["input"]
    cost_out = out * pr["output"]
    cost_total = (cost_cr + cost_cc + cost_fr + cost_out) or 1e-9

    budget = TokenBudget(
        cache_read_tokens=cr, cache_create_tokens=cc, fresh_input_tokens=fr, output_tokens=out,
        cache_read_share=cr / total, cache_create_share=cc / total,
        fresh_input_share=fr / total, output_share=out / total,
        output_text_share=text_c / out_chars,
        output_thinking_share=think_c / out_chars,
        output_toolargs_share=args_c / out_chars,
        cost_total_usd=round(cost_total, 4),
        cost_cache_read_share=cost_cr / cost_total,
        cost_cache_create_share=cost_cc / cost_total,
        cost_fresh_share=cost_fr / cost_total,
        cost_output_share=cost_out / cost_total,
    )

    n_calls = sum(t["calls"] for t in by_tool.values()) or 1
    tot_cr = cr or 1
    tot_cc = cc or 1
    actions = [
        ActionCache(
            tool=tool,
            calls=t["calls"],
            call_share=t["calls"] / n_calls,
            cache_read=t["cr"],
            cache_create=t["cc"],
            read_share=t["cr"] / tot_cr,
            create_share=t["cc"] / tot_cc,
            benefit_ratio=(t["cr"] / t["cc"]) if t["cc"] else float(t["cr"] > 0) * 999.0,
        )
        for tool, t in by_tool.items()
    ]
    actions.sort(key=lambda a: a.calls, reverse=True)
    return TokenBudgetResponse(budget=budget, actions=actions)




class InvalidationEvent(BaseModel):
    step_index: int
    cause: str
    drop_fraction: float
    rebuild_tokens: int
    rebuild_cost_usd: float
    seconds_gap: float | None = None


class WasteResponse(BaseModel):
    n_invalidations: int
    waste_tokens: int
    waste_cost_usd: float
    waste_share_of_create: float
    by_cause: dict[str, int]
    by_cause_cost: dict[str, float]
    events: list[InvalidationEvent]

    top_cost_steps: list[dict]



_PREV_MIN = 50_000
_DROP_TO = 0.3
_TTL_SECS = 300


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def detect_invalidations(traj: Trajectory) -> list[InvalidationEvent]:
    pr = per_token_prices(_model_of(traj))
    walls = {c.at_index for c in traj.compactions}
    events: list[InvalidationEvent] = []

    prev_cr = 0
    prev_ts: datetime | None = None
    for st in traj.steps:
        curr_cr = st.cache.cache_read_tokens
        curr_ts = _parse_ts(st.ts)
        if prev_cr > _PREV_MIN and curr_cr < prev_cr * _DROP_TO:
            secs = None
            if curr_ts and prev_ts:
                secs = (curr_ts - prev_ts).total_seconds()

            near_wall = any(abs(st.index - w) <= 2 for w in walls)
            if secs is not None and secs > _TTL_SECS:
                cause = "ttl_expiry"
            elif near_wall:
                cause = "compaction"
            else:
                cause = "system_change"
            rebuild = st.cache.cache_creation_tokens
            events.append(InvalidationEvent(
                step_index=st.index,
                cause=cause,
                drop_fraction=round(1 - curr_cr / prev_cr, 4) if prev_cr else 0.0,
                rebuild_tokens=rebuild,
                rebuild_cost_usd=round(rebuild * pr["cache_create"], 6),
                seconds_gap=round(secs, 1) if secs is not None else None,
            ))

            st.invalidation_cause = cause
            st.invalidation_drop = round(1 - curr_cr / prev_cr, 4) if prev_cr else 0.0
        prev_cr = curr_cr
        if curr_ts:
            prev_ts = curr_ts
    return events


def waste(traj: Trajectory) -> WasteResponse:
    pr = per_token_prices(_model_of(traj))
    events = detect_invalidations(traj)

    total_create = sum(st.cache.cache_creation_tokens for st in traj.steps) or 1
    waste_tokens = sum(e.rebuild_tokens for e in events)
    waste_cost = sum(e.rebuild_cost_usd for e in events)

    by_cause: dict[str, int] = {}
    by_cause_cost: dict[str, float] = {}
    for e in events:
        by_cause[e.cause] = by_cause.get(e.cause, 0) + 1
        by_cause_cost[e.cause] = round(by_cause_cost.get(e.cause, 0.0) + e.rebuild_cost_usd, 4)

    top = sorted(traj.steps, key=lambda s: s.cost_usd, reverse=True)[:10]
    top_cost_steps = [
        {
            "step_index": s.index,
            "tool": s.tool,
            "summary": s.summary[:80],
            "cost_usd": round(s.cost_usd, 4),
            "cache_create_tokens": s.cache.cache_creation_tokens,
            "invalidation_cause": s.invalidation_cause,
        }
        for s in top
    ]

    return WasteResponse(
        n_invalidations=len(events),
        waste_tokens=waste_tokens,
        waste_cost_usd=round(waste_cost, 4),
        waste_share_of_create=round(waste_tokens / total_create, 4),
        by_cause=by_cause,
        by_cause_cost=by_cause_cost,
        events=events,
        top_cost_steps=top_cost_steps,
    )




class FailureCluster(BaseModel):
    kind: str
    count: int
    cost_usd: float
    example_steps: list[int]


class FailureResponse(BaseModel):
    n_error_steps: int
    error_rate: float
    error_cost_usd: float
    clusters: list[FailureCluster]

    errors_by_third: list[int]
    note: str


def failures(traj: Trajectory) -> FailureResponse:
    steps = traj.steps
    n = len(steps) or 1
    err_steps = [s for s in steps if s.signals.is_error]
    failed_tests = [s for s in steps if s.signals.test_failed]
    err_cost = sum(s.cost_usd for s in err_steps)


    retries = 0
    run = 0
    retry_examples: list[int] = []
    for s in steps:
        if s.signals.is_error:
            run += 1
            if run == 3:
                retries += 1
                retry_examples.append(s.index)
        else:
            run = 0

    clusters: list[FailureCluster] = []
    if err_steps:
        clusters.append(FailureCluster(
            kind="tool_error", count=len(err_steps),
            cost_usd=round(err_cost, 4),
            example_steps=[s.index for s in err_steps[:5]],
        ))
    if failed_tests:
        clusters.append(FailureCluster(
            kind="failed_test", count=len(failed_tests),
            cost_usd=round(sum(s.cost_usd for s in failed_tests), 4),
            example_steps=[s.index for s in failed_tests[:5]],
        ))
    if retries:
        clusters.append(FailureCluster(
            kind="retry_run", count=retries, cost_usd=0.0, example_steps=retry_examples[:5],
        ))

    third = max(1, n // 3)
    by_third = [0, 0, 0]
    for s in err_steps:
        by_third[min(2, s.index // third)] += 1

    return FailureResponse(
        n_error_steps=len(err_steps),
        error_rate=round(len(err_steps) / n, 4),
        error_cost_usd=round(err_cost, 4),
        clusters=clusters,
        errors_by_third=by_third,
        note=(
            "These are free, trace-derived failure signals (tool errors, failed tests, retry "
            "runs). Classifying *why* a step failed, or whether an outcome was a "
            "success/partial/failure, requires an LLM judge — see the judge panel for what "
            "that would cost."
        ),
    )
