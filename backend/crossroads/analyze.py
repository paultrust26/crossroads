"""Shared analysis helpers used by both parsers: phase inference, loop detection,
re-read detection, compaction proximity. All operate on observable facts.
"""

from __future__ import annotations

import re

from .models import CompactionEvent, Phase, Step
from .scoring import COMPACTION_WINDOW


_TEST_RE = re.compile(r"\b(pytest|unittest|tox|nose|test_|reproduce|repro|npm test|go test)\b", re.I)
_SEARCH_RE = re.compile(r"\b(grep|rg|find|ls|glob|search)\b", re.I)


def infer_phase(tool: str, arg_text: str, is_edit: bool, is_test: bool, is_error: bool) -> Phase:
    if is_edit:
        return Phase.edit
    if is_test:
        return Phase.test
    t = (tool or "").lower()
    if t in ("read",) or ("view" in arg_text.lower() and "path" in arg_text.lower()):

        if re.search(r"\.[a-z]{1,4}\b", arg_text):
            return Phase.read
        return Phase.explore
    if _SEARCH_RE.search(arg_text) or _SEARCH_RE.search(t):
        return Phase.explore
    if is_error or (t == "bash" and not _TEST_RE.search(arg_text)):
        return Phase.debug if is_error else Phase.other
    return Phase.other


def detect_rereads(steps: list[Step]) -> None:
    """Mark re-reads and redundant re-reads in place using file_path history."""
    seen = {}
    for st in steps:
        fp = st.signals.file_path
        if not fp:
            continue
        cnt = seen.get(fp, 0) + 1
        seen[fp] = cnt
        st.signals.file_access_count = cnt
        if cnt >= 2 and st.phase in (Phase.read, Phase.explore):
            st.signals.is_reread = True

            if st.signals.redundancy_score is not None and st.signals.redundancy_score >= 0.6:
                st.signals.is_redundant_reread = True


def detect_loops(steps: list[Step], window: int = 6, min_repeats: int = 3) -> None:
    """Mark steps inside repeat regions. A loop = the same (tool, normalized-arg)
    signature occurring >= min_repeats times within a sliding window."""
    sigs = [(st.tool, _norm_sig(st.summary)) for st in steps]
    in_loop = [False] * len(steps)
    for i in range(len(steps)):
        lo = max(0, i - window)
        local = sigs[lo:i + 1]
        if local.count(sigs[i]) >= min_repeats:
            for j in range(lo, i + 1):
                if sigs[j] == sigs[i]:
                    in_loop[j] = True
    for st, flag in zip(steps, in_loop):
        if flag:
            st.signals.in_loop = True


def _norm_sig(summary: str) -> str:

    s = re.sub(r"\d+", "#", summary)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:60]


def mark_compaction_proximity(steps: list[Step], compactions: list[CompactionEvent]) -> None:
    walls = {c.at_index for c in compactions}
    for st in steps:
        for w in walls:
            if abs(st.index - w) <= COMPACTION_WINDOW:
                st.signals.near_compaction = True
                break


def is_test_text(text: str) -> bool:
    return bool(_TEST_RE.search(text or ""))
