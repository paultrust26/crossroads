"""Compaction Lens — peri-event analysis of a single compaction wall.

Aligns token/context cost around the
compaction event (t=0), measure the drop, and quantify the *recovery cost* — the
re-reads the agent is forced into afterward to re-acquire what it forgot.

Trace-based only: context size is approximated by the per-step input token count
(cache_read + cache_creation + fresh_input), which is what the model actually had
to (re)process at that step. No LLM judge required.
"""

from __future__ import annotations

from .models import (
    CompactionLensResponse,
    PeriEventPoint,
    RecoveryRead,
    Trajectory,
)

WINDOW = 25
RECOVERY_WINDOW = 20


def _context_tokens(step) -> int:
    c = step.cache
    return c.cache_read_tokens + c.cache_creation_tokens + c.fresh_input_tokens


def build_compaction_lens(traj: Trajectory, wall_index: int) -> CompactionLensResponse | None:
    walls = {c.at_index: c for c in traj.compactions}
    if wall_index not in walls:
        return None
    wall = walls[wall_index]
    steps = traj.steps
    n = len(steps)

    lo = max(0, wall_index - WINDOW)
    hi = min(n, wall_index + WINDOW)

    series = []
    for i in range(lo, hi):
        st = steps[i]
        series.append(
            PeriEventPoint(
                offset=i - wall_index,
                context_tokens=_context_tokens(st),
                cost_usd=round(st.cost_usd, 6),
                is_reread=st.signals.is_reread,
            )
        )


    pre_vals = [p.context_tokens for p in series if p.offset < 0]
    post_vals = [p.context_tokens for p in series if 0 <= p.offset <= RECOVERY_WINDOW]
    pre_peak = max(pre_vals) if pre_vals else 0
    post_trough = min(post_vals) if post_vals else 0
    drop = (1 - post_trough / pre_peak) if pre_peak else 0.0


    recovery_reads = []
    by_type = {}
    recovery_cost = 0.0
    for i in range(wall_index, min(n, wall_index + RECOVERY_WINDOW)):
        st = steps[i]
        if not st.reread_type: continue
        recovery_reads.append(
            RecoveryRead(
                offset=i - wall_index,
                file_path=st.signals.file_path or "",
                reread_type=st.reread_type,
                cost_usd=round(st.cost_usd, 6),
            )
        )
        by_type[st.reread_type] = by_type.get(st.reread_type, 0) + 1
        recovery_cost += st.cost_usd

    return CompactionLensResponse(
        wall_index=wall_index,
        summary_preview=wall.summary_preview,
        pre_peak_tokens=pre_peak,
        post_trough_tokens=post_trough,
        drop_fraction=round(drop, 4),
        series=series,
        recovery_reads=recovery_reads,
        recovery_read_count=len(recovery_reads),
        recovery_cost_usd=round(recovery_cost, 6),
        recovery_by_type=by_type,
    )
