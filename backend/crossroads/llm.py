"""Helpers for chat-completion calls made by the LLM judges."""

import json
from typing import Any

# Model families that accept only their default sampling temperature (1.0).
# Sending any other temperature makes the request fail.
DEFAULT_TEMPERATURE_ONLY_PREFIXES: tuple[str, ...] = ("claude-haiku-5",)


def sampling_kwargs(model: str, temperature: float = 0.0) -> dict[str, Any]:
    """Return the sampling arguments for a chat-completion call.

    For models that accept only their default temperature, the ``temperature``
    argument is omitted so the provider uses its default.

    Args:
        model: Model name as sent to the provider.
        temperature: Requested sampling temperature for models that accept it.

    Returns:
        A dict to unpack into ``chat.completions.create``.
    """
    if model.lower().startswith(DEFAULT_TEMPERATURE_ONLY_PREFIXES):
        return {}
    return {"temperature": temperature}


def extract_json(text: str) -> Any:
    """Parse the JSON object in a model reply, even when the reply adds prose or code fences.

    Args:
        text: Raw text returned by the model.

    Returns:
        The parsed JSON value (usually a dict).

    Raises:
        ValueError: If no JSON object can be parsed from the text.
    """
    stripped = (text or "").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model reply")
    return json.loads(stripped[start:end + 1])
