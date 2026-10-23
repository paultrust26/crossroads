"""SWE-bench trajectory parser.

Parses two JSONL formats:
1. Multi-method: {"task_id": "...", "baseline": {...}, "my_method": {...}, ...}
   where each method dict may or may not contain a "trajectory" list.
2. Flat (one method per line): {"task_id": "...", "condition"/"method": "...",
   "trajectory": [...], ...}

Only sessions that have trajectory data are surfaced (summary-only entries
without step-level data are included in overviews for aggregate stats but
excluded from spine/conversation views).
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .settings import SWE_DIR




_DATA_DIR = SWE_DIR


_EDIT_COMMANDS = {"create", "str_replace", "insert"}
_ERROR_MARKERS = (
    "error", "Error", "Traceback", "command not found",
    "No such file", "ENOENT", "Permission denied", "not found",
    "FileNotFoundError", "ModuleNotFoundError",
)




class SweSessionSummary(BaseModel):
    """Summary of one task_id + method combination."""
    id: str
    task_id: str
    method: str
    project: str
    resolved: bool
    n_steps: int
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    obs_tokens_raw: int = 0
    obs_tokens_served: int = 0
    savings_pct: float = 0.0
    duration_s: float = 0.0
    cost_usd: float = 0.0
    has_trajectory: bool = False
    tier_distribution: dict[str, int] = Field(default_factory=dict)
    tools_used: dict[str, int] = Field(default_factory=dict)
    n_edits: int = 0
    n_errors: int = 0


class SweOverviewResponse(BaseModel):
    """Overview of all discoverable SWE sessions in a data directory."""
    sessions: list[SweSessionSummary]
    n_sessions: int
    total_tokens: int
    mean_steps: float
    resolve_rate: float
    methods: dict[str, int] = Field(default_factory=dict)
    projects: dict[str, int] = Field(default_factory=dict)


class SweSpineStep(BaseModel):
    """One step in the spine (timeline) view."""
    index: int
    tool: str
    thought: str = ""
    observation_preview: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw_obs_tokens: int = 0
    served_obs_tokens: int = 0
    tier: str = ""
    file_path: Optional[str] = None
    is_edit: bool = False


class SweSpineResponse(BaseModel):
    """Step-level spine data for a single session."""
    session_id: str
    task_id: str
    method: str
    resolved: bool
    n_steps: int
    steps: list[SweSpineStep]


class SweConversationBlock(BaseModel):
    """One step rendered for conversation view."""
    index: int
    tool: str
    thought: str = ""
    raw_observation: str = ""
    served_observation: str = ""
    raw_obs_tokens: int = 0
    served_obs_tokens: int = 0
    tier: str = ""
    file_path: Optional[str] = None
    is_edit: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    args: dict = Field(default_factory=dict)


class SweConversationResponse(BaseModel):
    """Full conversation view for a single session."""
    session_id: str
    task_id: str
    method: str
    resolved: bool
    n_steps: int
    total_tokens: int = 0
    savings_pct: float = 0.0
    blocks: list[SweConversationBlock]




def _extract_project(task_id: str) -> str:
    """Extract project from task_id like 'arrow-py__arrow.xxx' -> 'arrow-py/arrow'."""


    parts = task_id.split(".")
    if parts:
        org_repo = parts[0]
        segments = org_repo.split("__")
        if len(segments) >= 2:
            return f"{segments[0]}/{segments[1]}"
        return org_repo
    return task_id


def _extract_file_path(args: dict) -> Optional[str]:
    """Extract file path from step args."""
    return args.get("path") or args.get("file_path")


def _is_edit_step(tool: str, args: dict) -> bool:
    """Determine if a step is an edit (modifying a file)."""
    if tool == "str_replace_editor":
        command = args.get("command", "")
        return command in _EDIT_COMMANDS
    return False


def _has_error(observation: str) -> bool:
    """Check if an observation contains error markers."""
    if not observation:
        return False
    for marker in _ERROR_MARKERS:
        if marker in observation:
            return True
    return False


def _step_obs_tokens(step: dict, kind: str) -> int:
    """Return a step's raw or served observation tokens.

    Trajectory files name these ``obs_tokens_raw`` / ``obs_tokens_served``; the
    older ``raw_obs_tokens`` / ``served_obs_tokens`` names are also accepted.

    Args:
        step: One trajectory step.
        kind: ``"raw"`` or ``"served"``.

    Returns:
        The token count, or 0 when the step has none.
    """
    value = step.get(f"obs_tokens_{kind}", step.get(f"{kind}_obs_tokens", 0))
    return int(value or 0)


def _parse_method_data(
    task_id: str,
    method: str,
    data: dict,
) -> SweSessionSummary:
    """Parse a single method's data dict into a session summary."""
    trajectory = data.get("trajectory", [])
    has_traj = bool(trajectory)


    tools: Counter = Counter()
    tiers: Counter = Counter()
    n_edits = 0
    n_errors = 0

    for step in trajectory:
        tool = step.get("tool", "unknown")
        tools[tool] += 1
        tier = step.get("tier", "")
        if tier:
            tiers[tier] += 1
        args = step.get("args", {}) or {}
        if _is_edit_step(tool, args):
            n_edits += 1

        obs = step.get("served_observation") or step.get("raw_observation") or ""
        if tool == "bash" and _has_error(obs):
            n_errors += 1


    if not has_traj:

        pass

    obs_raw = int(data.get("obs_tokens_raw") or sum(_step_obs_tokens(st, "raw") for st in trajectory))
    obs_served = int(data.get("obs_tokens_served") or sum(_step_obs_tokens(st, "served") for st in trajectory))

    return SweSessionSummary(
        id=f"{task_id}__{method}",
        task_id=task_id,
        method=method,
        project=_extract_project(task_id),
        resolved=bool(data.get("resolved", False)),
        n_steps=int(data.get("num_steps", len(trajectory))),
        total_tokens=int(data.get("total_tokens", 0)),
        input_tokens=int(data.get("input_tokens", 0)),
        output_tokens=int(data.get("output_tokens", 0)),
        obs_tokens_raw=obs_raw,
        obs_tokens_served=obs_served,
        savings_pct=float(data.get("savings_pct") or (100.0 * (1 - obs_served / obs_raw) if obs_raw else 0.0)),
        duration_s=float(data.get("duration_s", 0.0)),
        cost_usd=float(data.get("cost_usd") or data.get("agent_cost_usd") or 0.0),
        has_trajectory=has_traj,
        tier_distribution=dict(tiers),
        tools_used=dict(tools),
        n_edits=n_edits,
        n_errors=n_errors,
    )




_SESSIONS_CACHE: dict[str, list[SweSessionSummary]] = {}


def _load_all_sessions(data_dir: Path) -> list[SweSessionSummary]:
    """Load and cache all session summaries from the data directory."""
    cache_key = str(data_dir)
    if cache_key in _SESSIONS_CACHE:
        return _SESSIONS_CACHE[cache_key]

    sessions: list[SweSessionSummary] = []

    for jsonl_path in sorted(data_dir.glob("*_results.jsonl")):
        with jsonl_path.open("r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                task_id = record.get("task_id", "")
                if not task_id:
                    continue


                if "trajectory" in record or "condition" in record or "method" in record:

                    method = record.get("condition") or record.get("method") or "unknown"
                    sessions.append(_parse_method_data(task_id, method, record))
                else:

                    for key, value in record.items():
                        if key == "task_id":
                            continue
                        if isinstance(value, dict):
                            sessions.append(_parse_method_data(task_id, key, value))

    _SESSIONS_CACHE[cache_key] = sessions
    return sessions


def list_swe_sessions(data_dir: Path = _DATA_DIR) -> list[SweSessionSummary]:
    """Return all discoverable sessions (one per task_id + method combo).

    Includes sessions both with and without trajectory data. Those without
    trajectory will have has_trajectory=False.
    """
    return _load_all_sessions(data_dir)


def build_swe_overview(data_dir: Path = _DATA_DIR) -> SweOverviewResponse:
    """Build a portfolio-level overview of all SWE sessions."""
    sessions = _load_all_sessions(data_dir)

    n = len(sessions)
    total_tokens = sum(s.total_tokens for s in sessions)
    total_steps = sum(s.n_steps for s in sessions)
    resolved_count = sum(1 for s in sessions if s.resolved)

    methods: Counter = Counter()
    projects: Counter = Counter()
    for s in sessions:
        methods[s.method] += 1
        projects[s.project] += 1

    return SweOverviewResponse(
        sessions=sessions,
        n_sessions=n,
        total_tokens=total_tokens,
        mean_steps=round(total_steps / n, 1) if n else 0.0,
        resolve_rate=round(resolved_count / n, 4) if n else 0.0,
        methods=dict(methods.most_common()),
        projects=dict(projects.most_common()),
    )


def _find_trajectory(data_dir: Path, session_id: str) -> tuple[str, str, dict, list]:
    """Locate the trajectory for a given session_id.

    Returns (task_id, method, record_or_method_data, trajectory_list).
    Raises ValueError if not found.
    """

    parts = session_id.split("__")
    if len(parts) < 2:
        raise ValueError(f"Invalid session_id format: {session_id}")



    method = parts[-1]
    task_id = "__".join(parts[:-1])

    for jsonl_path in sorted(data_dir.glob("*_results.jsonl")):
        with jsonl_path.open("r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if record.get("task_id") != task_id:
                    continue


                if "trajectory" in record or "condition" in record or "method" in record:
                    rec_method = record.get("condition") or record.get("method") or "unknown"
                    if rec_method == method:
                        trajectory = record.get("trajectory", [])
                        if trajectory:
                            return task_id, method, record, trajectory


                elif method in record and isinstance(record[method], dict):
                    method_data = record[method]
                    trajectory = method_data.get("trajectory", [])
                    if trajectory:
                        return task_id, method, method_data, trajectory

    raise ValueError(
        f"No trajectory found for session_id={session_id!r} "
        f"(task_id={task_id!r}, method={method!r})"
    )


def build_swe_spine(data_dir: Path = _DATA_DIR, session_id: str = "") -> SweSpineResponse:
    """Build step-level spine data for timeline visualization."""
    task_id, method, data, trajectory = _find_trajectory(data_dir, session_id)

    steps: list[SweSpineStep] = []
    for step_data in trajectory:
        args = step_data.get("args", {}) or {}
        tool = step_data.get("tool", "unknown")
        served_obs = step_data.get("served_observation", "") or ""

        steps.append(SweSpineStep(
            index=int(step_data.get("step", len(steps))),
            tool=tool,
            thought=step_data.get("thought", "") or "",
            observation_preview=served_obs[:200],
            input_tokens=int(step_data.get("input_tokens", 0)),
            output_tokens=int(step_data.get("output_tokens", 0)),
            raw_obs_tokens=_step_obs_tokens(step_data, "raw"),
            served_obs_tokens=_step_obs_tokens(step_data, "served"),
            tier=step_data.get("tier", "") or "",
            file_path=_extract_file_path(args),
            is_edit=_is_edit_step(tool, args),
        ))

    return SweSpineResponse(
        session_id=session_id,
        task_id=task_id,
        method=method,
        resolved=bool(data.get("resolved", False)),
        n_steps=len(steps),
        steps=steps,
    )


def build_swe_conversation(
    data_dir: Path = _DATA_DIR,
    session_id: str = "",
) -> SweConversationResponse:
    """Build full conversation view with complete observations."""
    task_id, method, data, trajectory = _find_trajectory(data_dir, session_id)

    blocks: list[SweConversationBlock] = []
    for step_data in trajectory:
        args = step_data.get("args", {}) or {}
        tool = step_data.get("tool", "unknown")

        blocks.append(SweConversationBlock(
            index=int(step_data.get("step", len(blocks))),
            tool=tool,
            thought=step_data.get("thought", "") or "",
            raw_observation=step_data.get("raw_observation", "") or "",
            served_observation=step_data.get("served_observation", "") or "",
            raw_obs_tokens=_step_obs_tokens(step_data, "raw"),
            served_obs_tokens=_step_obs_tokens(step_data, "served"),
            tier=step_data.get("tier", "") or "",
            file_path=_extract_file_path(args),
            is_edit=_is_edit_step(tool, args),
            input_tokens=int(step_data.get("input_tokens", 0)),
            output_tokens=int(step_data.get("output_tokens", 0)),
            args=args,
        ))

    total_tokens = int(data.get("total_tokens", 0))
    savings_pct = float(data.get("savings_pct", 0.0))

    return SweConversationResponse(
        session_id=session_id,
        task_id=task_id,
        method=method,
        resolved=bool(data.get("resolved", False)),
        n_steps=len(blocks),
        total_tokens=total_tokens,
        savings_pct=savings_pct,
        blocks=blocks,
    )


def clear_cache() -> None:
    """Clear the session cache (useful for testing or after data changes)."""
    _SESSIONS_CACHE.clear()
