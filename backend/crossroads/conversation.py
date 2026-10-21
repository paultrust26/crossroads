"""Conversation view — 'what is actually happening' in a session.

Groups a session into *prompt groups*: one real user query followed by all the
assistant turns (thinking / text / tool calls) it triggered, up to the next real
query. For each group we expose the actual text — the query, and the response split
into reasoning (thinking), user-facing text, and tool-call arguments — alongside
token/cost accounting. This is the qualitative companion to the economics lenses.

A "real" user query is a human message, not a tool_result and not a slash-command /
local-command wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel

from .metrics import per_token_prices


_NONQUERY_PREFIXES = ("<", "[Request interrupted", "Caveat:")
_CMD_MARKERS = ("<command-name>", "<local-command", "<bash-")


class ResponseBlock(BaseModel):
    kind: str
    text: str
    chars: int


class PromptGroup(BaseModel):
    index: int
    query: str
    query_chars: int
    start_step: int
    n_tool_calls: int
    tools_used: dict[str, int]

    thinking_chars: int
    text_chars: int
    toolargs_chars: int

    output_tokens: int
    input_tokens: int
    cost_usd: float

    blocks: list[ResponseBlock]


class ConversationResponse(BaseModel):
    n_groups: int
    model: str | None

    total_query_chars: int
    total_thinking_chars: int
    total_text_chars: int
    total_toolargs_chars: int
    groups: list[PromptGroup]


def _iter(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _user_text(msg: dict[str, Any]) -> str | None:
    """Return the human text of a user message, or None if it's a tool_result."""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):

        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
            return None
        parts = [b.get("text", "") for b in c if isinstance(b, dict) if b.get("type") == "text"]
        return "\n".join(parts) if parts else None
    return None


def _is_real_query(text: str | None) -> bool:
    if not text or not text.strip():
        return False
    t = text.lstrip()
    if t.startswith(_NONQUERY_PREFIXES):
        return False
    if any(m in text for m in _CMD_MARKERS):
        return False
    return True


def _tool_arg_summary(name: str, inp: dict[str, Any]) -> str:
    if isinstance(inp, dict):
        for k in ("command", "file_path", "path", "pattern", "query", "url", "prompt"):
            if inp.get(k):
                return f"{name}: {str(inp[k])[:120]}"
    return name


def build_conversation(path: Path, max_groups: int | None = None) -> ConversationResponse:
    path = Path(path)
    model = None
    groups: list[PromptGroup] = []
    cur: PromptGroup | None = None
    step_idx = 0

    def flush():
        nonlocal cur
        if cur is not None:
            groups.append(cur)
            cur = None

    for rec in _iter(path):
        ty = rec.get("type")
        msg = rec.get("message") if isinstance(rec.get("message"), dict) else None

        if ty == "user" and msg is not None:
            qt = _user_text(msg)
            if _is_real_query(qt):
                flush()
                if max_groups and len(groups) >= max_groups:
                    break
                cur = PromptGroup(
                    index=len(groups), query=qt.strip()[:4000], query_chars=len(qt),
                    start_step=step_idx, n_tool_calls=0, tools_used={},
                    thinking_chars=0, text_chars=0, toolargs_chars=0,
                    output_tokens=0, input_tokens=0, cost_usd=0.0, blocks=[],
                )
            continue

        if ty != "assistant" or msg is None:
            continue
        if model is None:
            model = msg.get("model")
        if cur is None:

            continue

        usage = msg.get("usage", {}) or {}
        pr = per_token_prices(model)
        cr = int(usage.get("cache_read_input_tokens", 0) or 0)
        cc = int(usage.get("cache_creation_input_tokens", 0) or 0)
        fr = int(usage.get("input_tokens", 0) or 0)
        ot = int(usage.get("output_tokens", 0) or 0)
        cur.output_tokens += ot
        cur.input_tokens += cr + cc + fr
        cur.cost_usd += cr * pr["cache_read"] + cc * pr["cache_create"] + fr * pr["input"] + ot * pr["output"]

        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "thinking":
                txt = b.get("thinking", "")
                cur.thinking_chars += len(txt)
                if txt.strip():
                    cur.blocks.append(ResponseBlock(kind="thinking", text=txt[:2000], chars=len(txt)))
            elif bt == "text":
                txt = b.get("text", "")
                cur.text_chars += len(txt)
                if txt.strip():
                    cur.blocks.append(ResponseBlock(kind="text", text=txt[:2000], chars=len(txt)))
            elif bt == "tool_use":
                name = b.get("name", "")
                inp = b.get("input", {}) or {}
                try:
                    args = json.dumps(inp, ensure_ascii=False)
                except (TypeError, ValueError):
                    args = str(inp)
                cur.toolargs_chars += len(args)
                cur.n_tool_calls += 1
                cur.tools_used[name] = cur.tools_used.get(name, 0) + 1
                cur.blocks.append(ResponseBlock(kind="tool", text=_tool_arg_summary(name, inp), chars=len(args)))
                step_idx += 1

    flush()

    for g in groups:
        g.cost_usd = round(g.cost_usd, 4)

    return ConversationResponse(
        n_groups=len(groups),
        model=model,
        total_query_chars=sum(g.query_chars for g in groups),
        total_thinking_chars=sum(g.thinking_chars for g in groups),
        total_text_chars=sum(g.text_chars for g in groups),
        total_toolargs_chars=sum(g.toolargs_chars for g in groups),
        groups=groups,
    )
