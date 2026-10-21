"""Topic modeling + 2D projection of agent queries — a ConceptViz-style map.

Inspired by ConceptViz (arXiv:2509.20376): embed items, cluster into labeled
semantic regions, project to 2D, and let the user explore. Here the items are the
*user queries* in agent sessions: "the map of what people ask agents to do."

Pipeline (TF-IDF, fully local, no model download):
  1. TF-IDF vectorize the query texts.
  2. KMeans cluster → topics; auto-label each from its top distinguishing terms.
  3. t-SNE → 2D coordinates for the scatter.
  4. (per session) topic-to-topic transitions for a flow graph.

All deterministic (fixed seeds) so the map is stable across reloads.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from .conversation import build_conversation

_MIN_QUERY_CHARS = 8
_SEED = 0


class QueryPoint(BaseModel):
    session_id: str
    group_index: int
    x: float
    y: float
    topic: int
    query: str
    cost_usd: float
    output_tokens: int
    n_tool_calls: int
    n_compactions: int = 0


class Topic(BaseModel):
    id: int
    label: str
    terms: list[str]
    size: int
    total_cost_usd: float
    mean_output_tokens: float


class TransitionEdge(BaseModel):
    src: int
    dst: int
    count: int


class QueryMapResponse(BaseModel):
    n_queries: int
    topics: list[Topic]
    points: list[QueryPoint]
    transitions: list[TransitionEdge]
    note: str = ""


def _label_from_terms(terms: list[str]) -> str:
    clean = [t for t in terms if not re.fullmatch(r"[0-9a-f]{4,}", t) and len(t) > 1]
    return " · ".join(clean[:3]) if clean else "misc"


def _n_topics(n_items: int) -> int:
    return max(2, min(10, n_items // 12))


def _build_map(
    texts: list[str],
    meta: list[dict],
    with_transitions: bool,
) -> QueryMapResponse:
    n = len(texts)
    if n < 4:
        return QueryMapResponse(n_queries=n, topics=[], points=[], transitions=[],
                                note="Too few queries to map (need ≥4).")


    vec = TfidfVectorizer(max_features=600, stop_words="english",
                          min_df=2 if n > 20 else 1, token_pattern=r"[A-Za-z][A-Za-z_]{1,}")
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return QueryMapResponse(n_queries=n, topics=[], points=[], transitions=[],
                                note="Queries share no common vocabulary to cluster.")
    terms = vec.get_feature_names_out()


    k = _n_topics(n)
    km = KMeans(n_clusters=k, n_init=4, random_state=_SEED)
    labels = km.fit_predict(X)

    topic_terms: dict[int, list[str]] = {}
    for c in range(k):
        order = km.cluster_centers_[c].argsort()[::-1]
        topic_terms[c] = [terms[i] for i in order[:6]]


    dense = X.toarray()
    if dense.shape[1] > 50:
        from sklearn.decomposition import TruncatedSVD
        dense = TruncatedSVD(n_components=50, random_state=_SEED).fit_transform(X)
    perp = max(5, min(30, n // 3))
    coords = TSNE(n_components=2, perplexity=perp, random_state=_SEED,
                  init="pca", learning_rate="auto").fit_transform(dense)

    cmin, cmax = coords.min(0), coords.max(0)
    span = np.where(cmax - cmin == 0, 1, cmax - cmin)
    coords = (coords - cmin) / span

    points = []
    for i in range(n):
        m = meta[i]
        points.append(QueryPoint(
            session_id=m["session_id"], group_index=m["group_index"],
            x=round(float(coords[i, 0]), 4), y=round(float(coords[i, 1]), 4),
            topic=int(labels[i]), query=texts[i][:160],
            cost_usd=round(m["cost_usd"], 4), output_tokens=m["output_tokens"],
            n_tool_calls=m["n_tool_calls"], n_compactions=m.get("n_compactions", 0),
        ))

    topics = []
    for c in range(k):
        idx = [i for i in range(n) if labels[i] == c]
        topics.append(Topic(
            id=c, label=_label_from_terms(topic_terms[c]), terms=topic_terms[c],
            size=len(idx),
            total_cost_usd=round(sum(meta[i]["cost_usd"] for i in idx), 4),
            mean_output_tokens=round(sum(meta[i]["output_tokens"] for i in idx) / max(1, len(idx)), 1),
        ))

    transitions = []
    if with_transitions:
        counts: dict[tuple[int, int], int] = {}

        for i in range(1, n):
            a, b = int(labels[i - 1]), int(labels[i])
            if a != b:
                counts[(a, b)] = counts.get((a, b), 0) + 1
        transitions = [TransitionEdge(src=a, dst=b, count=c) for (a, b), c in counts.items()]
        transitions.sort(key=lambda e: e.count, reverse=True)

    return QueryMapResponse(n_queries=n, topics=topics, points=points, transitions=transitions)


def _group_text(g, kind: str) -> str:
    """Text to embed for a prompt group: the user query, or the agent's response
    (its reasoning + user-facing text blocks, excluding tool args)."""
    if kind == "response":
        parts = [b.text for b in g.blocks if b.kind in ("thinking", "text")]
        return "\n".join(parts)
    return g.query


def session_query_map(path: Path, kind: str = "query") -> QueryMapResponse:
    conv = build_conversation(path)
    texts: list[str] = []
    meta: list[dict] = []
    for g in conv.groups:
        text = _group_text(g, kind)
        if len(text) < _MIN_QUERY_CHARS:
            continue
        texts.append(text)
        meta.append({
            "session_id": path.stem, "group_index": g.index,
            "cost_usd": g.cost_usd, "output_tokens": g.output_tokens,
            "n_tool_calls": g.n_tool_calls,
        })
    return _build_map(texts, meta, with_transitions=True)


def swe_query_map(data_dir: Path, kind: str = "query", cap: int = 2000) -> QueryMapResponse:
    """Build a topic map from SWE-bench trajectories.
    kind='query' uses agent thoughts; kind='response' uses observations."""
    from .swe_parser import list_swe_sessions, _find_trajectory

    sessions = [s for s in list_swe_sessions(data_dir) if s.has_trajectory]
    texts: list[str] = []
    meta: list[dict] = []

    for sess in sessions:
        try:
            _, _, data, trajectory = _find_trajectory(data_dir, sess.id)
        except ValueError:
            continue
        for step in trajectory:
            if kind == "response":
                text = step.get("served_observation") or step.get("raw_observation") or ""
            else:
                text = step.get("thought") or ""
            if len(text) < _MIN_QUERY_CHARS:
                continue
            texts.append(text)
            meta.append({
                "session_id": sess.id,
                "group_index": step.get("step", 0),
                "cost_usd": 0.0,
                "output_tokens": step.get("output_tokens", 0),
                "n_tool_calls": 1,
                "n_compactions": 0,
            })

    if len(texts) > cap:
        rng = np.random.default_rng(_SEED)
        keep = rng.choice(len(texts), size=cap, replace=False)
        keep.sort()
        texts = [texts[i] for i in keep]
        meta = [meta[i] for i in keep]

    res = _build_map(texts, meta, with_transitions=False)
    if res.n_queries:
        res.note = f"{res.n_queries} agent thoughts across {len(sessions)} sessions" + (" (sampled)" if len(texts) >= cap else "")
    return res


def corpus_query_map(paths: list[tuple[str, Path, int]], cap: int = 1500, kind: str = "query") -> QueryMapResponse:
    """paths: list of (session_id, path, n_compactions). Pools queries (or responses)
    across sessions, sampling down to `cap` for a responsive map."""
    texts: list[str] = []
    meta: list[dict] = []
    for sid, path, ncomp in paths:
        try:
            conv = build_conversation(path, max_groups=80)
        except OSError:
            continue
        for g in conv.groups:
            text = _group_text(g, kind)
            if len(text) < _MIN_QUERY_CHARS:
                continue
            texts.append(text)
            meta.append({
                "session_id": sid, "group_index": g.index,
                "cost_usd": g.cost_usd, "output_tokens": g.output_tokens,
                "n_tool_calls": g.n_tool_calls, "n_compactions": ncomp,
            })


    if len(texts) > cap:
        rng = np.random.default_rng(_SEED)
        keep = rng.choice(len(texts), size=cap, replace=False)
        keep.sort()
        texts = [texts[i] for i in keep]
        meta = [meta[i] for i in keep]

    res = _build_map(texts, meta, with_transitions=False)
    if res.n_queries:
        res.note = f"{res.n_queries} queries across sessions" + (" (sampled)" if len(meta) >= cap else "")
    return res
