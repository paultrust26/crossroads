"""Compaction Diff — entity-level analysis of what was preserved vs dropped.

The Compaction Lens (compaction.py) shows *how much* context was lost (token drop %).
This module answers *what* was lost semantically: files, symbols, constraints, goals.

Algorithm:
  1. Extract entities from pre-compaction context (the agent's thought/observations
     in the window before the wall) using regex + code-aware patterns.
  2. Extract entities from the compact summary (summary_preview on the wall).
  3. Compute preservation rate, rank dropped entities by importance (frequency × recency).
  4. Return a structured diff: preserved / dropped / mutated.

No LLM judge required — all extraction is pattern-based.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from pydantic import BaseModel

from .models import CompactionEvent, Step, Trajectory





_FILE_RE = re.compile(
    r"""(?:^|[\s\"'`(,])(/[\w./-]+\.\w{1,6}|[\w./-]+/[\w.-]+\.\w{1,6})""",
    re.MULTILINE,
)

_SYMBOL_RE = re.compile(
    r"\b(?:def|class|function|const|let|var|export)\s+(\w{2,})"
)

_CONSTRAINT_RE = re.compile(
    r"(?:must|don't|do not|never|always|should not|TODO|FIXME|IMPORTANT|NOTE)[:\s]+([^\n.]{8,80})",
    re.IGNORECASE,
)

_IDENT_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z]\w*_\w+)\b")


class Entity(BaseModel):
    name: str
    kind: str
    frequency: int = 1
    recency: float = 0.0
    importance: float = 0.0


class CompactionDiffResponse(BaseModel):
    wall_index: int
    summary_preview: str
    n_pre_entities: int
    n_post_entities: int
    preservation_rate: float
    preserved: list[Entity]
    dropped: list[Entity]
    summary_only: list[Entity]


def _extract_entities(text: str) -> list[tuple[str, str]]:
    """Extract (name, kind) pairs from text."""
    entities = []

    for m in _FILE_RE.finditer(text):
        path = m.group(1).strip()
        if len(path) > 4 and not path.startswith("//"):
            entities.append((path, "file"))

    for m in _SYMBOL_RE.finditer(text):
        entities.append((m.group(1), "symbol"))

    for m in _CONSTRAINT_RE.finditer(text):
        constraint = m.group(1).strip()
        if len(constraint) > 10:
            entities.append((constraint[:60], "constraint"))

    for m in _IDENT_RE.finditer(text):
        ident = m.group(1)
        if len(ident) >= 4 and ident not in ("self", "None", "True", "False", "this"):
            entities.append((ident, "identifier"))

    return entities


def _build_entity_set(
    raw_entities: list[tuple[str, str]], n_steps_in_window: int
) -> dict[str, Entity]:
    """Deduplicate and score entities by frequency and recency."""
    counts = Counter()
    kinds = {}
    last_pos = {}

    for i, (name, kind) in enumerate(raw_entities):
        key = name.lower() if kind != "constraint" else name
        counts[key] += 1
        kinds[key] = kind
        last_pos[key] = i

    total = len(raw_entities) or 1
    result = {}
    for key, count in counts.items():
        recency = last_pos[key] / total if total else 0.0
        importance = (count / total) * 0.6 + recency * 0.4
        result[key] = Entity(
            name=key if kinds[key] != "file" else key,
            kind=kinds[key],
            frequency=count,
            recency=round(recency, 3),
            importance=round(importance, 4),
        )
    return result


def build_compaction_diff(
    traj: Trajectory, wall_index: int, pre_window: int = 30
) -> Optional[CompactionDiffResponse]:
    """Compute the entity-level diff for a compaction wall."""
    walls = {c.at_index: c for c in traj.compactions}
    if wall_index not in walls:
        return None
    wall = walls[wall_index]
    steps = traj.steps


    lo = max(0, wall_index - pre_window)
    pre_text_parts = []
    for i in range(lo, wall_index):
        st = steps[i]
        if st.thought:
            pre_text_parts.append(st.thought)
        if st.observation_preview:
            pre_text_parts.append(st.observation_preview)
        if st.summary:
            pre_text_parts.append(st.summary)
    pre_text = "\n".join(pre_text_parts)


    post_text = wall.summary_preview


    pre_raw = _extract_entities(pre_text)
    post_raw = _extract_entities(post_text)

    pre_entities = _build_entity_set(pre_raw, pre_window)
    post_entities = _build_entity_set(post_raw, 1)


    pre_keys = set(pre_entities.keys())
    post_keys = set(post_entities.keys())

    preserved_keys = pre_keys & post_keys
    dropped_keys = pre_keys - post_keys
    new_keys = post_keys - pre_keys

    preserved = [pre_entities[k] for k in preserved_keys]
    dropped = sorted(
        [pre_entities[k] for k in dropped_keys],
        key=lambda e: e.importance,
        reverse=True,
    )
    summary_only = [post_entities[k] for k in new_keys]

    n_pre = len(pre_keys) or 1
    preservation_rate = len(preserved_keys) / n_pre

    return CompactionDiffResponse(
        wall_index=wall_index,
        summary_preview=wall.summary_preview,
        n_pre_entities=len(pre_keys),
        n_post_entities=len(post_keys),
        preservation_rate=round(preservation_rate, 3),
        preserved=preserved[:30],
        dropped=dropped[:40],
        summary_only=summary_only[:20],
    )
