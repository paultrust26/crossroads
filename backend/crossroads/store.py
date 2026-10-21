"""Trajectory store: discovery + lazy parse + in-memory cache.

Parsing a 20MB session takes a moment, so we cache parsed+scored Trajectory objects
by id. Discovery is cheap (file stat only)."""

from __future__ import annotations

from pathlib import Path

from .models import Trajectory
from .parse_claude import list_sessions, parse_claude_session
from .scoring import score_trajectory

_CACHE: dict[str, Trajectory] = {}
_PATHS: dict[str, Path] = {}


def discover() -> list[dict]:
    sessions = list_sessions()
    for s in sessions:
        _PATHS[s["id"]] = Path(s["path"])
    return sessions


def get_path(traj_id: str) -> Path | None:
    if traj_id not in _PATHS:
        discover()
    p = _PATHS.get(traj_id)
    return p if p and p.exists() else None


def get_trajectory(traj_id: str, max_steps: int | None = None) -> Trajectory | None:
    if traj_id in _CACHE:
        return _CACHE[traj_id]

    if traj_id not in _PATHS:
        discover()
    path = _PATHS.get(traj_id)
    if not path or not path.exists():
        return None
    traj = parse_claude_session(path, max_steps=max_steps)
    score_trajectory(traj)
    _CACHE[traj_id] = traj
    return traj
