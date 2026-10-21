"""File/Repo Heatmap — aggregate file access patterns across a session.

Shows which files were accessed (read/edit/test), how frequently, and whether
the accesses were productive (led to edits/test passes) or wasteful (churn,
loops, redundant re-reads). Provides data for a treemap/heatmap visualization.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from pydantic import BaseModel

from .models import Phase, Step, Trajectory


class FileEntry(BaseModel):
    path: str
    short_path: str
    reads: int = 0
    edits: int = 0
    errors: int = 0
    total_accesses: int = 0
    cost_usd: float = 0.0
    in_loop_accesses: int = 0
    reread_count: int = 0
    redundant_reread_count: int = 0
    first_access: int = 0
    last_access: int = 0
    phases: dict[str, int]


class DirectoryEntry(BaseModel):
    path: str
    files: int = 0
    total_accesses: int = 0
    cost_usd: float = 0.0
    edits: int = 0
    errors: int = 0


class FileMapResponse(BaseModel):
    n_files: int
    n_directories: int
    total_file_accesses: int
    files: list[FileEntry]
    directories: list[DirectoryEntry]

    most_accessed: list[FileEntry]
    most_costly: list[FileEntry]
    most_churned: list[FileEntry]
    hotspot_directories: list[DirectoryEntry]


def _short_path(p: str) -> str:
    parts = p.split("/")
    if len(parts) <= 3:
        return p
    return "…/" + "/".join(parts[-3:])


def _dir_of(p: str) -> str:
    parts = p.rsplit("/", 1)
    return parts[0] if len(parts) > 1 else "."


def build_filemap(traj: Trajectory) -> FileMapResponse:
    """Aggregate file access patterns from trajectory steps."""
    file_data = defaultdict(lambda: {
        "reads": 0, "edits": 0, "errors": 0, "total": 0,
        "cost": 0.0, "in_loop": 0, "rereads": 0, "redundant_rereads": 0,
        "first": 9999999, "last": 0, "phases": defaultdict(int),
    })

    for st in traj.steps:
        fp = st.signals.file_path
        if not fp:
            continue

        d = file_data[fp]
        d["total"] += 1
        d["cost"] += st.cost_usd
        d["first"] = min(d["first"], st.index)
        d["last"] = max(d["last"], st.index)
        d["phases"][st.phase.value] += 1

        if st.signals.is_edit:
            d["edits"] += 1
        elif st.phase in (Phase.read, Phase.explore):
            d["reads"] += 1

        if st.signals.is_error:
            d["errors"] += 1
        if st.signals.in_loop:
            d["in_loop"] += 1
        if st.signals.is_reread:
            d["rereads"] += 1
        if st.signals.is_redundant_reread:
            d["redundant_rereads"] += 1


    files = []
    for path, d in file_data.items():
        files.append(FileEntry(
            path=path,
            short_path=_short_path(path),
            reads=d["reads"],
            edits=d["edits"],
            errors=d["errors"],
            total_accesses=d["total"],
            cost_usd=round(d["cost"], 4),
            in_loop_accesses=d["in_loop"],
            reread_count=d["rereads"],
            redundant_reread_count=d["redundant_rereads"],
            first_access=d["first"] if d["first"] != 9999999 else 0,
            last_access=d["last"],
            phases=dict(d["phases"]),
        ))


    dir_data = defaultdict(lambda: {
        "files": set(), "total": 0, "cost": 0.0, "edits": 0, "errors": 0,
    })
    for f in files:
        d_path = _dir_of(f.path)
        dd = dir_data[d_path]
        dd["files"].add(f.path)
        dd["total"] += f.total_accesses
        dd["cost"] += f.cost_usd
        dd["edits"] += f.edits
        dd["errors"] += f.errors

    directories = []
    for d_path, dd in dir_data.items():
        directories.append(DirectoryEntry(
            path=d_path,
            files=len(dd["files"]),
            total_accesses=dd["total"],
            cost_usd=round(dd["cost"], 4),
            edits=dd["edits"],
            errors=dd["errors"],
        ))


    most_accessed = sorted(files, key=lambda f: f.total_accesses, reverse=True)[:15]
    most_costly = sorted(files, key=lambda f: f.cost_usd, reverse=True)[:15]
    most_churned = sorted(files, key=lambda f: f.redundant_reread_count, reverse=True)
    most_churned = [f for f in most_churned if f.redundant_reread_count > 0][:15]
    hotspot_dirs = sorted(directories, key=lambda d: d.total_accesses, reverse=True)[:10]

    return FileMapResponse(
        n_files=len(files),
        n_directories=len(directories),
        total_file_accesses=sum(f.total_accesses for f in files),
        files=sorted(files, key=lambda f: f.total_accesses, reverse=True),
        directories=sorted(directories, key=lambda d: d.total_accesses, reverse=True),
        most_accessed=most_accessed,
        most_costly=most_costly,
        most_churned=most_churned,
        hotspot_directories=hotspot_dirs,
    )
