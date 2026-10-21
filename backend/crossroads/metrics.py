"""Real cost / cache / efficiency metrics for agent trajectories.

Pricing tables and the cost + reread-classification logic follow the
"Where Do the Tokens Go?" analysis, adapted to operate on Crossroads' Step model
directly (no pandas). Everything here is trace-based: no LLM judge.

Prices are USD per **million** tokens (Anthropic prompt-caching tiers):
  base_input  — fresh (uncached) input
  cache_5m    — 5-minute cache *write* (cache creation)
  cache_1h    — 1-hour cache write
  cache_hit   — cached input *read*
  output      — output tokens
"""

from __future__ import annotations

from .models import Phase, Step, Trajectory


MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-fable-5": {"base_input": 10.0, "cache_5m": 12.5, "cache_1h": 20.0, "cache_hit": 1.0, "output": 50.0},
    "claude-mythos-5": {"base_input": 10.0, "cache_5m": 12.5, "cache_1h": 20.0, "cache_hit": 1.0, "output": 50.0},
    "claude-opus-4.8": {"base_input": 5.0, "cache_5m": 6.25, "cache_1h": 10.0, "cache_hit": 0.5, "output": 25.0},
    "claude-opus-4.7": {"base_input": 5.0, "cache_5m": 6.25, "cache_1h": 10.0, "cache_hit": 0.5, "output": 25.0},
    "claude-opus-4.6": {"base_input": 5.0, "cache_5m": 6.25, "cache_1h": 10.0, "cache_hit": 0.5, "output": 25.0},
    "claude-opus-4.5": {"base_input": 5.0, "cache_5m": 6.25, "cache_1h": 10.0, "cache_hit": 0.5, "output": 25.0},
    "claude-opus-4.1": {"base_input": 15.0, "cache_5m": 18.75, "cache_1h": 30.0, "cache_hit": 1.5, "output": 75.0},
    "claude-opus-4": {"base_input": 15.0, "cache_5m": 18.75, "cache_1h": 30.0, "cache_hit": 1.5, "output": 75.0},
    "claude-sonnet-4.6": {"base_input": 3.0, "cache_5m": 3.75, "cache_1h": 6.0, "cache_hit": 0.3, "output": 15.0},
    "claude-sonnet-4.5": {"base_input": 3.0, "cache_5m": 3.75, "cache_1h": 6.0, "cache_hit": 0.3, "output": 15.0},
    "claude-sonnet-4": {"base_input": 3.0, "cache_5m": 3.75, "cache_1h": 6.0, "cache_hit": 0.3, "output": 15.0},
    "claude-haiku-4.5": {"base_input": 1.0, "cache_5m": 1.25, "cache_1h": 2.0, "cache_hit": 0.1, "output": 5.0},
}
_DEFAULT = "claude-sonnet-4.5"


def normalize_model_name(model: str | None) -> str:
    m = (model or "").lower().strip()
    if "fable" in m:
        return "claude-fable-5"
    if "mythos" in m:
        return "claude-mythos-5"
    if "haiku" in m:
        return "claude-haiku-4.5"
    if "opus-4-8" in m or "opus-4.8" in m:
        return "claude-opus-4.8"
    if "opus-4-7" in m or "opus-4.7" in m:
        return "claude-opus-4.7"
    if "opus-4-6" in m or "opus-4.6" in m:
        return "claude-opus-4.6"
    if "opus-4-5" in m or "opus-4.5" in m:
        return "claude-opus-4.5"
    if "opus-4-1" in m or "opus-4.1" in m:
        return "claude-opus-4.1"
    if "opus" in m:
        return "claude-opus-4.8"
    if "sonnet-4-6" in m or "sonnet-4.6" in m:
        return "claude-sonnet-4.6"
    if "sonnet" in m:
        return "claude-sonnet-4.5"
    return _DEFAULT


def per_token_prices(model: str | None) -> dict[str, float]:
    p = MODEL_PRICING.get(normalize_model_name(model), MODEL_PRICING[_DEFAULT])
    return {
        "cache_read": p["cache_hit"] / 1e6,
        "cache_create": p["cache_5m"] / 1e6,
        "output": p["output"] / 1e6,
        "input": p["base_input"] / 1e6,
    }


def step_cost_usd(step: Step, model: str | None) -> float:
    """Dollar cost of one step from its token breakdown."""
    pr = per_token_prices(model)
    c = step.cache
    return (
        c.cache_read_tokens * pr["cache_read"]
        + c.cache_creation_tokens * pr["cache_create"]
        + c.output_tokens * pr["output"]
        + c.fresh_input_tokens * pr["input"]
    )





def classify_rereads(traj: Trajectory) -> None:
    """Annotate each Read step that re-reads a file with a reread_type, in place.

    RECOVERY        — a compaction occurred between the two reads (re-acquiring
                      context the agent lost). Cost of forgetting.
    VERIFICATION    — the file was edited between reads (checking your own work).
    CHECK_EXTERNAL  — a bash command likely modified the file between reads.
    CHURN           — neither; a short-gap or unexplained re-read (waste).
    REFERENCE_DIST  — a long-gap legitimate re-reference.
    """
    wall_indices = {c.at_index for c in traj.compactions}
    last_read_at: dict[str, int] = {}

    steps = traj.steps
    for st in steps:
        fp = st.signals.file_path
        is_read = st.tool == "Read" or st.phase == Phase.read
        if not fp or not is_read:
            continue
        if fp in last_read_at:
            prev = last_read_at[fp]
            rtype = _classify_one(steps, prev, st.index, fp, wall_indices)
            st.reread_type = rtype
            st.signals.is_reread = True
            if rtype == "CHURN":
                st.signals.is_redundant_reread = True
        last_read_at[fp] = st.index


def _classify_one(steps: list[Step], prev: int, cur: int, fp: str, walls: set[int]) -> str:
    between = steps[prev + 1:cur]
    if any(prev < w <= cur for w in walls):
        return "RECOVERY"
    for b in between:
        if b.signals.is_edit and b.signals.file_path == fp:
            return "VERIFICATION"
    for b in between:
        if b.tool == "Bash" and fp in b.summary and any(op in b.summary for op in (">", ">>", "tee", "mv ", "cp ")):
            return "CHECK_EXTERNAL"
    gap = cur - prev
    if gap > 50:
        return "REFERENCE_DIST"
    return "CHURN"





_EXPLORE_TOOLS = {"Read", "Grep", "Glob", "WebFetch"}
_PRODUCE_TOOLS = {"Edit", "MultiEdit", "Write", "NotebookEdit"}


def compute_session_metrics(traj: Trajectory) -> "SessionMetrics":
    from .models import SessionMetrics

    model = traj.meta.model
    cost = 0.0
    cache_read = cache_create = output = fresh = 0
    explore = produce = 0
    churn = recovery = verification = 0
    waste_tokens = 0

    for st in traj.steps:
        cost += step_cost_usd(st, model)
        c = st.cache
        cache_read += c.cache_read_tokens
        cache_create += c.cache_creation_tokens
        output += c.output_tokens
        fresh += c.fresh_input_tokens

        if st.tool in _EXPLORE_TOOLS or (st.tool == "Bash" and st.phase == Phase.explore):
            explore += 1
        elif st.tool in _PRODUCE_TOOLS:
            produce += 1

        if st.reread_type == "CHURN":
            churn += 1
            waste_tokens += st.cache.output_tokens + _obs_tokens(st)
        elif st.reread_type == "RECOVERY":
            recovery += 1
        elif st.reread_type == "VERIFICATION":
            verification += 1

    total_input = cache_read + cache_create + fresh
    cache_hit_rate = cache_read / total_input if total_input else 0.0


    pr = per_token_prices(model)
    cost_no_cache = (
        (cache_read + cache_create + fresh) * pr["input"] + output * pr["output"]
    )
    effective_savings = 1 - cost / cost_no_cache if cost_no_cache else 0.0

    paradox_gap = cache_hit_rate - effective_savings

    total_steps = max(1, len(traj.steps))
    waste_frac = churn / total_steps

    return SessionMetrics(
        cost_usd=round(cost, 4),
        cost_no_cache_usd=round(cost_no_cache, 4),
        effective_savings=round(effective_savings, 4),
        cache_hit_rate=round(cache_hit_rate, 4),
        paradox_gap=round(paradox_gap, 4),
        tokens_cache_read=cache_read,
        tokens_cache_create=cache_create,
        tokens_output=output,
        tokens_fresh_input=fresh,
        explore_steps=explore,
        produce_steps=produce,
        explore_produce_ratio=round(explore / produce, 2) if produce else float(explore),
        reread_churn=churn,
        reread_recovery=recovery,
        reread_verification=verification,
        waste_step_fraction=round(waste_frac, 4),
        waste_tokens_est=waste_tokens,
    )


def _obs_tokens(step: Step) -> int:

    return max(0, step.tokens - step.cache.output_tokens)
