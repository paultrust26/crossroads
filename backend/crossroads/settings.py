"""Runtime settings, read from environment variables.

The LLM judge calls a hosted provider through the `openai` package: OpenAI by
default, or another provider with an OpenAI compatible API via OPENAI_BASE_URL.

  OPENAI_API_KEY           API key for the judge provider
  OPENAI_BASE_URL          Base URL of the provider (unset means api.openai.com)
  CROSSROADS_JUDGE_MODEL   Model name sent to the endpoint (default: gpt-4o-mini)
  CROSSROADS_SESSIONS_DIR  Root of agent session transcripts
                           (default: ~/.claude/projects)
  CROSSROADS_SWE_DIR       Directory of SWE-bench style trajectories (optional)
  CROSSROADS_CACHE_DIR     Where the judge result cache is stored
                           (default: ~/.cache/crossroads)
"""

from __future__ import annotations

import os
from pathlib import Path

JUDGE_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
JUDGE_BASE_URL: str | None = os.environ.get("OPENAI_BASE_URL") or None
JUDGE_MODEL: str = os.environ.get("CROSSROADS_JUDGE_MODEL", "gpt-4o-mini")

SESSIONS_DIR = Path(os.environ.get("CROSSROADS_SESSIONS_DIR", Path.home() / ".claude" / "projects")).expanduser()
SWE_DIR = Path(os.environ.get("CROSSROADS_SWE_DIR", "data/swe")).expanduser()
CACHE_DIR = Path(os.environ.get("CROSSROADS_CACHE_DIR", Path.home() / ".cache" / "crossroads")).expanduser()


def judge_client():
    """Return an AsyncOpenAI client for the configured judge endpoint.

    Returns:
        openai.AsyncOpenAI: Client built from OPENAI_API_KEY and OPENAI_BASE_URL.

    Raises:
        RuntimeError: If the openai package is missing or no API key is set.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise RuntimeError("openai package not installed. Run: pip install openai") from e
    if not JUDGE_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set; the LLM judge needs an API key")
    return AsyncOpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)


def project_label(dir_name: str) -> str:
    """Turn a Claude Code project folder name into a short readable label.

    Claude Code names project folders after the absolute path with "/" replaced
    by "-" (e.g. "-home-alice-code-myrepo"). The home directory prefix is dropped.

    Args:
        dir_name: Folder name under the sessions directory.

    Returns:
        str: The label, or the folder name unchanged if nothing is left.
    """
    home = str(Path.home()).replace("/", "-")
    label = dir_name
    if label.startswith(home):
        label = label[len(home):]
    return label.strip("-") or dir_name
