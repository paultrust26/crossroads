"""LLM-judge cost *estimator* — and an explicit boundary.

Several questions in the "Where Do the Tokens Go?" framing are semantic and cannot
be answered from trace signals alone (paper §3, "Token instrumentation"): task
complexity, interruption type, correction vs refinement, success/partial/failure
outcome. Those need an LLM judge with structured prompts.

This module does NOT run a judge. It estimates what running one would cost on a
given session, so the UI can show the user the price and let them decide. Running a
judge is gated behind explicit user confirmation.
"""

from __future__ import annotations

from pydantic import BaseModel

from .metrics import MODEL_PRICING, normalize_model_name
from .models import Trajectory



JUDGE_PASSES = [
    ("complexity", "session", "Rate task ambiguity / scope / knowledge required (1–5)."),
    ("outcome", "session", "Classify SUCCESS / PARTIAL / FAILURE / ABANDONED."),
    ("interruption_types", "intervention", "Classify each user interruption's intent."),
    ("correction_handling", "intervention", "Did the agent understand each correction; tokens wasted before."),
    ("tool_justification", "tool_call", "Was each tool call necessary vs wasteful."),
    ("compaction_fidelity", "compaction", "What state each compaction preserved / lost."),
]


_PROMPT_OVERHEAD = 1200

_CTX_PER_SESSION = 8000
_CTX_PER_UNIT = 600

_OUT_PER_CALL = 250


_JUDGE_MODEL = "claude-haiku-4.5"


class JudgePassEstimate(BaseModel):
    name: str
    unit: str
    description: str
    n_units: int
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float


class JudgeEstimateResponse(BaseModel):
    judge_model: str
    available: bool
    note: str
    passes: list[JudgePassEstimate]
    total_cost_usd: float


def _unit_count(traj: Trajectory, unit: str) -> int:
    if unit == "session":
        return 1
    if unit == "tool_call":
        return len(traj.steps)
    if unit == "compaction":
        return max(1, len(traj.compactions))
    if unit == "intervention":


        return max(1, sum(1 for s in traj.steps if s.signals.is_error) + len(traj.compactions))
    return 1


def estimate_judges(traj: Trajectory, judge_model: str = _JUDGE_MODEL) -> JudgeEstimateResponse:
    pricing = MODEL_PRICING.get(normalize_model_name(judge_model), MODEL_PRICING["claude-haiku-4.5"])
    in_price = pricing["base_input"] / 1e6
    out_price = pricing["output"] / 1e6

    passes = []
    total = 0.0
    for name, unit, desc in JUDGE_PASSES:
        n = _unit_count(traj, unit)
        ctx = _CTX_PER_SESSION if unit == "session" else _CTX_PER_UNIT
        in_tok = n * (_PROMPT_OVERHEAD + ctx)
        out_tok = n * _OUT_PER_CALL
        cost = in_tok * in_price + out_tok * out_price
        total += cost
        passes.append(JudgePassEstimate(
            name=name, unit=unit, description=desc, n_units=n,
            est_input_tokens=in_tok, est_output_tokens=out_tok,
            est_cost_usd=round(cost, 4),
        ))

    return JudgeEstimateResponse(
        judge_model=judge_model,
        available=False,
        note=(
            "Estimate only — no judge runner is attached. These numbers are what an "
            "LLM-judge pass would cost on this session at the listed model's pricing. "
            "Run the LLM judge from the failure or anomaly views to populate them."
        ),
        passes=passes,
        total_cost_usd=round(total, 4),
    )
