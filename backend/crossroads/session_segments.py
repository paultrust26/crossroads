"""Session-level topic segmentation.

Detects when a user switches tasks within a session and groups steps into
contiguous "segments" — each representing a coherent work topic. Uses TF-IDF
similarity between consecutive user queries to detect boundaries.

This is used by the failure analysis to attribute failures to specific topics
within a session, so users can see which sub-task failed even in long sessions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .conversation import build_conversation

_BOUNDARY_THRESHOLD = 0.08
_MIN_QUERY_CHARS = 10


class Segment(BaseModel):
    index: int
    label: str
    start_group: int
    end_group: int
    start_step: int
    end_step: int
    n_queries: int
    n_tool_calls: int
    cost_usd: float
    has_errors: bool
    n_errors: int
    outcome: str


class SegmentationResponse(BaseModel):
    session_id: str
    n_segments: int
    segments: list[Segment]
    boundaries: list[int]


def segment_session(session_path: Path, session_id: str = "") -> SegmentationResponse:
    conv = build_conversation(session_path, max_groups=300)
    groups = conv.groups

    if len(groups) < 2:
        seg = Segment(
            index=0, label=groups[0].query[:60] if groups else "empty",
            start_group=0, end_group=0,
            start_step=0, end_step=groups[0].start_step + groups[0].n_tool_calls - 1 if groups else 0,
            n_queries=len(groups), n_tool_calls=sum(g.n_tool_calls for g in groups),
            cost_usd=sum(g.cost_usd for g in groups),
            has_errors=False, n_errors=0, outcome="unclear",
        )
        return SegmentationResponse(
            session_id=session_id, n_segments=1,
            segments=[seg], boundaries=[],
        )

    queries = [g.query for g in groups]
    texts = [q if len(q) >= _MIN_QUERY_CHARS else "continuation" for q in queries]

    vec = TfidfVectorizer(max_features=300, stop_words="english", min_df=1)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return _single_segment(session_id, groups)

    similarities = []
    for i in range(1, len(groups)):
        sim = cosine_similarity(X[i - 1:i], X[i:i + 1])[0, 0]
        similarities.append(float(sim))

    boundaries = []
    for i, sim in enumerate(similarities):
        if sim < _BOUNDARY_THRESHOLD:
            boundaries.append(i + 1)

    if not boundaries:
        return _single_segment(session_id, groups)

    seg_starts = [0] + boundaries
    seg_ends = boundaries + [len(groups)]
    segments = []

    for idx, (start, end) in enumerate(zip(seg_starts, seg_ends)):
        seg_groups = groups[start:end]
        first_step = seg_groups[0].start_step
        last_step = seg_groups[-1].start_step + seg_groups[-1].n_tool_calls - 1
        n_errors = 0
        for g in seg_groups:
            for b in g.blocks:
                if b.kind == "tool" and ("error" in b.text.lower()[:100] or "traceback" in b.text.lower()[:100]):
                    n_errors += 1

        label = seg_groups[0].query[:60]
        if len(label) == 60:
            label = label.rsplit(" ", 1)[0] + "…"

        outcome = "unclear"
        last_g = seg_groups[-1]
        if any("error" in b.text.lower()[:100] for b in last_g.blocks if b.kind == "tool"):
            outcome = "failed"
        elif any(b.kind == "text" and ("done" in b.text.lower()[:200] or "complete" in b.text.lower()[:200]) for b in last_g.blocks):
            outcome = "success"

        segments.append(Segment(
            index=idx, label=label,
            start_group=start, end_group=end - 1,
            start_step=first_step, end_step=last_step,
            n_queries=len(seg_groups),
            n_tool_calls=sum(g.n_tool_calls for g in seg_groups),
            cost_usd=round(sum(g.cost_usd for g in seg_groups), 4),
            has_errors=n_errors > 0, n_errors=n_errors,
            outcome=outcome,
        ))

    return SegmentationResponse(
        session_id=session_id, n_segments=len(segments),
        segments=segments, boundaries=boundaries,
    )


def _single_segment(session_id: str, groups) -> SegmentationResponse:
    n_errors = 0
    for g in groups:
        for b in g.blocks:
            if b.kind == "tool" and ("error" in b.text.lower()[:100] or "traceback" in b.text.lower()[:100]):
                n_errors += 1

    label = groups[0].query[:60] if groups else "empty"
    seg = Segment(
        index=0, label=label,
        start_group=0, end_group=len(groups) - 1,
        start_step=groups[0].start_step if groups else 0,
        end_step=groups[-1].start_step + groups[-1].n_tool_calls - 1 if groups else 0,
        n_queries=len(groups),
        n_tool_calls=sum(g.n_tool_calls for g in groups),
        cost_usd=round(sum(g.cost_usd for g in groups), 4),
        has_errors=n_errors > 0, n_errors=n_errors,
        outcome="unclear",
    )
    return SegmentationResponse(
        session_id=session_id, n_segments=1,
        segments=[seg], boundaries=[],
    )
