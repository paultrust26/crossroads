"""Time-series module — bins step data into wall-clock intervals.

Every step has a `ts` (ISO timestamp). This module computes time-binned
aggregates for cost, tokens, cache dynamics, waste events, topics, and file
access patterns. The frontend renders these as area charts / sparklines.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel

from .models import Phase, Step, Trajectory


class TimeBin(BaseModel):
    t: float
    t_label: str


class CostTimeSeries(BaseModel):
    bins: list[TimeBin]
    cumulative_cost: list[float]
    per_bin_cost: list[float]
    cumulative_tokens: list[int]


class TokenMixBin(BaseModel):
    t: float
    t_label: str
    cache_read: int
    cache_create: int
    fresh_input: int
    output: int


class TokenMixTimeSeries(BaseModel):
    bins: list[TokenMixBin]
    total_duration_secs: float


class WasteTimeSeries(BaseModel):
    bins: list[TimeBin]
    cumulative_waste_usd: list[float]
    invalidation_events: list[dict]


class PhaseTimeSeries(BaseModel):
    bins: list[TimeBin]
    phases: dict[str, list[int]]


class FileAccessEvent(BaseModel):
    t: float
    t_label: str
    file_path: str
    short_path: str
    phase: str
    is_edit: bool
    is_error: bool
    is_reread: bool
    step_index: int


class FileLifeline(BaseModel):
    path: str
    short_path: str
    events: list[FileAccessEvent]
    total_accesses: int
    edits: int


class FileTimelineResponse(BaseModel):
    duration_secs: float
    files: list[FileLifeline]
    all_events: list[FileAccessEvent]


class TopicTimeBin(BaseModel):
    t: float
    t_label: str
    topic_counts: dict[int, int]


class TopicEvolutionResponse(BaseModel):
    bins: list[TopicTimeBin]
    total_duration_secs: float


class FullTimeSeriesResponse(BaseModel):
    duration_secs: float
    n_bins: int
    cost: CostTimeSeries
    token_mix: TokenMixTimeSeries
    waste: WasteTimeSeries
    phases: PhaseTimeSeries
    file_timeline: FileTimelineResponse


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _short_path(p: str) -> str:
    parts = p.split("/")
    if len(parts) <= 3:
        return p
    return "…/" + "/".join(parts[-3:])


def _fmt_time(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _compute_bins(steps: list[Step], n_bins: int = 40) -> tuple[list[float], float, datetime | None]:
    """Compute bin edges. Returns (bin_edges_in_secs, duration, start_time)."""
    timestamps = []
    for st in steps:
        ts = _parse_ts(st.ts)
        if ts:
            timestamps.append((st.index, ts))

    if len(timestamps) < 2:
        return [], 0.0, None

    start = timestamps[0][1]
    end = timestamps[-1][1]
    duration = (end - start).total_seconds()
    if duration <= 0:
        return [], 0.0, None

    bin_width = duration / n_bins
    edges = [i * bin_width for i in range(n_bins + 1)]
    return edges, duration, start


def _step_offset(step: Step, start: datetime) -> float | None:
    ts = _parse_ts(step.ts)
    if not ts:
        return None
    return (ts - start).total_seconds()


def build_full_timeseries(traj: Trajectory, n_bins: int = 40) -> FullTimeSeriesResponse | None:
    edges, duration, start = _compute_bins(traj.steps, n_bins)
    if not edges or start is None:
        return None

    bin_width = duration / n_bins


    cost_per_bin = [0.0] * n_bins
    tok_bins = [{"cr": 0, "cc": 0, "fr": 0, "out": 0} for _ in range(n_bins)]
    waste_per_bin = [0.0] * n_bins
    phase_per_bin = {p.value: [0] * n_bins for p in Phase}
    waste_events = []
    file_events = []

    for st in traj.steps:
        offset = _step_offset(st, start)
        if offset is None:
            continue

        bin_idx = min(int(offset / bin_width), n_bins - 1)
        bin_idx = max(0, bin_idx)

        cost_per_bin[bin_idx] += st.cost_usd
        tok_bins[bin_idx]["cr"] += st.cache.cache_read_tokens
        tok_bins[bin_idx]["cc"] += st.cache.cache_creation_tokens
        tok_bins[bin_idx]["fr"] += st.cache.fresh_input_tokens
        tok_bins[bin_idx]["out"] += st.cache.output_tokens
        phase_per_bin[st.phase.value][bin_idx] += 1

        if st.invalidation_cause:
            waste_per_bin[bin_idx] += st.cost_usd
            waste_events.append({
                "t": round(offset, 1),
                "cause": st.invalidation_cause,
                "cost_usd": round(st.cost_usd, 4),
                "step_index": st.index,
            })

        if st.signals.file_path:
            file_events.append(FileAccessEvent(
                t=round(offset, 1),
                t_label=_fmt_time(offset),
                file_path=st.signals.file_path,
                short_path=_short_path(st.signals.file_path),
                phase=st.phase.value,
                is_edit=st.signals.is_edit,
                is_error=st.signals.is_error,
                is_reread=st.signals.is_reread,
                step_index=st.index,
            ))


    time_bins = [TimeBin(t=round(i * bin_width, 1), t_label=_fmt_time(i * bin_width)) for i in range(n_bins)]


    cum_cost = []
    running = 0.0
    for c in cost_per_bin:
        running += c
        cum_cost.append(round(running, 4))


    cum_tokens = []
    running_tok = 0
    for tb in tok_bins:
        running_tok += tb["cr"] + tb["cc"] + tb["fr"] + tb["out"]
        cum_tokens.append(running_tok)


    cum_waste = []
    running_w = 0.0
    for w in waste_per_bin:
        running_w += w
        cum_waste.append(round(running_w, 4))


    token_mix_bins = [
        TokenMixBin(
            t=round(i * bin_width, 1),
            t_label=_fmt_time(i * bin_width),
            cache_read=tok_bins[i]["cr"],
            cache_create=tok_bins[i]["cc"],
            fresh_input=tok_bins[i]["fr"],
            output=tok_bins[i]["out"],
        )
        for i in range(n_bins)
    ]


    file_by_path: dict[str, list[FileAccessEvent]] = defaultdict(list)
    for ev in file_events:
        file_by_path[ev.file_path].append(ev)

    lifelines = []
    for path, events in sorted(file_by_path.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
        lifelines.append(FileLifeline(
            path=path,
            short_path=_short_path(path),
            events=events,
            total_accesses=len(events),
            edits=sum(1 for e in events if e.is_edit),
        ))

    return FullTimeSeriesResponse(
        duration_secs=round(duration, 1),
        n_bins=n_bins,
        cost=CostTimeSeries(
            bins=time_bins,
            cumulative_cost=cum_cost,
            per_bin_cost=[round(c, 4) for c in cost_per_bin],
            cumulative_tokens=cum_tokens,
        ),
        token_mix=TokenMixTimeSeries(bins=token_mix_bins, total_duration_secs=round(duration, 1)),
        waste=WasteTimeSeries(bins=time_bins, cumulative_waste_usd=cum_waste, invalidation_events=waste_events),
        phases=PhaseTimeSeries(bins=time_bins, phases=phase_per_bin),
        file_timeline=FileTimelineResponse(
            duration_secs=round(duration, 1),
            files=lifelines,
            all_events=file_events,
        ),
    )
