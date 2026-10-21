"""Criticality scoring.

The Criticality Spine needs a principled y-axis. True pivotality is a counterfactual
quantity (variance in the final outcome attributable to the decision at step i) and
requires re-execution. Until an engine is attached, we compute a transparent
**surrogate criticality index** from real, observable signals on each step.

Design principles:
  * Every component is grounded in an observable fact about the run, not a guess.
  * Components are additive and individually inspectable (returned per step) so the
    UI can explain *why* a step scored high — causality, not a black box.
  * The score is a *salience prior over decision points* — exactly the cheap
    surrogate the Crossroads design doc (§4.3) uses to target where an expensive
    counterfactual engine should spend rollouts. We are honest that it is a prior.

The weights below encode a simple, defensible hypothesis: steps that introduce
irreversible change (edits), steps that fail (errors / failed tests), steps where
the agent is thrashing (redundant re-reads, loops), and steps adjacent to memory
loss (compaction) are the most likely to carry causal leverage in a long run.
"""

from __future__ import annotations

from .models import Step, Trajectory


W = {
    "edit": 0.30,
    "test_failed": 0.25,
    "error": 0.18,
    "redundant_reread": 0.15,
    "near_compaction": 0.12,
    "loop": 0.10,
    "surrogate_prior": 0.25,
}


COMPACTION_WINDOW = 3


def _normalize_prior(p: float | None) -> float:

    if p is None:
        return 0.0
    return max(0.0, min(1.0, p / 10.0))


def score_step(step: Step) -> tuple[float, dict[str, float]]:
    s = step.signals
    comp = {}

    if s.is_edit:
        comp["edit"] = W["edit"]
    if s.test_failed:
        comp["test_failed"] = W["test_failed"]
    if s.is_error:
        comp["error"] = W["error"]
    if s.is_redundant_reread:
        comp["redundant_reread"] = W["redundant_reread"]
    elif s.is_reread and s.file_access_count >= 3:

        comp["redundant_reread"] = 0.5 * W["redundant_reread"]
    if s.near_compaction:
        comp["near_compaction"] = W["near_compaction"]
    if s.in_loop:
        comp["loop"] = W["loop"]
    if s.surrogate_prior is not None:
        comp["surrogate_prior"] = W["surrogate_prior"] * _normalize_prior(s.surrogate_prior)

    raw = sum(comp.values())


    import math
    score = 1.0 - math.exp(-1.6 * raw)
    return score, comp


def score_trajectory(traj: Trajectory, salience_threshold: float = 0.45) -> None:
    """Populate `criticality` on each step and `root_cause_index` on meta, in place.

    The root-cause candidate is the *earliest* step whose criticality exceeds the
    salience threshold AND that lies upstream of the failure symptom. This is the
    surrogate analog of the doc's 'earliest step a fix would still have worked'.
    """
    for step in traj.steps:
        score, comp = score_step(step)
        step.criticality = round(score, 4)
        step.criticality_components = {k: round(v, 4) for k, v in comp.items()}



    candidates = [
        st for st in traj.steps
        if st.criticality >= salience_threshold
        and (st.signals.is_edit or st.signals.is_error or st.signals.test_failed
             or (st.signals.surrogate_prior or 0) > 0)
    ]
    traj.meta.root_cause_index = candidates[0].index if candidates else None
