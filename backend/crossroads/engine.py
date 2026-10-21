"""Counterfactual engine interface + the Fan/Ladder builders.

The honest core of Crossroads: true pivotality is P(success | resample at step i),
which requires *re-executing* the agent from step i many times. That engine is
defined here as an interface (`CounterfactualEngine`). No real re-executor is wired
in for Claude sessions yet (it needs the model + a replay harness), so the default
engine reports `attached = False` and we NEVER fabricate rollout probabilities.

What we CAN compute without an engine, from real observed signals:
  * the Counterfactual Fan's *structure* and the realized thread (what actually
    happened downstream of this step in THIS run);
  * the Intervention Ladder's *applicability* — which typed interventions the
    observed evidence says are even relevant here (e.g. rung-1 'perturb tool result'
    is applicable iff this step had a tool error).

Outcome estimates on Fan/Ladder stay None until a real engine populates them.
"""

from __future__ import annotations

from typing import Protocol

from .models import (
    FanResponse,
    FanThread,
    LadderResponse,
    LadderRung,
    Outcome,
    Step,
    Trajectory,
)


class CounterfactualEngine(Protocol):
    """A real implementation re-executes the agent from a frozen prefix under an
    intervention and returns outcome samples. Wiring one in is the path from a
    surrogate criticality index to true pivotality."""

    attached: bool
    note: str

    def recover_rate(self, traj: Trajectory, step_index: int, rung: int) -> tuple[float, tuple[float, float]] | None:
        ...


class NullEngine:
    """Default engine: no re-execution available. Honest about it."""
    attached = False
    note = (
        "No counterfactual re-execution engine attached. Rollout probabilities "
        "(P(success | resample)) require re-running the agent from this step and are "
        "intentionally not shown rather than fabricated. The Spine y-axis is a surrogate "
        "criticality index from real observed signals; the Fan shows the realized path "
        "and observed downstream outcome of THIS run."
    )

    def recover_rate(self, traj, step_index, rung):
        return None


_ENGINE: CounterfactualEngine = NullEngine()


def set_engine(engine: CounterfactualEngine) -> None:
    global _ENGINE
    _ENGINE = engine


def get_engine() -> CounterfactualEngine:
    return _ENGINE




def build_fan(traj: Trajectory, step_index: int, lookahead: int = 12) -> FanResponse:
    eng = get_engine()
    step = traj.steps[step_index]


    tail = traj.steps[step_index:step_index + lookahead]
    actual_sparkline = [s.phase for s in tail]
    downstream_outcome = _observed_downstream_outcome(traj, step_index)

    threads = [
        FanThread(
            label="realized path (this run)",
            outcome=downstream_outcome,
            is_actual=True,
            phase_sparkline=actual_sparkline,
            note=_describe_tail(tail),
        )
    ]

    evidence = _step_evidence(step)

    recover = eng.recover_rate(traj, step_index, rung=0)
    return FanResponse(
        step_index=step_index,
        engine_attached=eng.attached,
        engine_note=eng.note,
        recover_rate=recover[0] if recover else None,
        recover_ci=recover[1] if recover else None,
        threads=threads,
        evidence=evidence,
    )


def _observed_downstream_outcome(traj: Trajectory, step_index: int) -> Outcome:


    return traj.meta.outcome


def _describe_tail(tail: list[Step]) -> str:
    errs = sum(1 for s in tail if s.signals.is_error)
    edits = sum(1 for s in tail if s.signals.is_edit)
    loops = sum(1 for s in tail if s.signals.in_loop)
    bits = []
    if edits:
        bits.append(f"{edits} edit(s)")
    if errs:
        bits.append(f"{errs} error(s)")
    if loops:
        bits.append(f"{loops} step(s) in a loop")
    return "next %d steps: " % len(tail) + (", ".join(bits) if bits else "no errors/edits")


def _step_evidence(step: Step) -> list[str]:
    s = step.signals
    ev = []
    if s.is_edit:
        ev.append(f"Irreversible change: edited {s.file_path or 'a file'}.")
    if s.is_error:
        ev.append("Tool returned an error here — a fork between recovery and spiral.")
    if s.test_failed:
        ev.append("A test/repro step failed at this point.")
    if s.is_redundant_reread:
        ev.append(f"Redundant re-read (file seen {s.file_access_count}× with unchanged content) — agent uncertainty.")
    elif s.is_reread:
        ev.append(f"Re-read of a file already seen ({s.file_access_count}× total).")
    if s.near_compaction:
        ev.append("Adjacent to a compaction wall — a constraint may have been dropped near here.")
    if s.in_loop:
        ev.append("Inside a detected repeat/loop region.")
    if not ev:
        ev.append("No high-leverage signals — likely an inert step (any reasonable choice leads to the same place).")
    return ev




_RUNGS = [
    (0, "Re-sample, same context",
     "Is the failure just high-entropy sampling? Re-run the same decision unchanged."),
    (1, "Perturb a tool result",
     "Was the agent misled by a noisy/failed observation? Replace it with a clean one."),
    (2, "Inject / restore one constraint",
     "Was a dropped fact fatal? Re-inject a constraint lost (e.g. at compaction)."),
    (3, "Override the single decision",
     "Force the alternative action at this step."),
]


def build_ladder(traj: Trajectory, step_index: int) -> LadderResponse:
    eng = get_engine()
    step = traj.steps[step_index]
    s = step.signals

    rungs = []
    for rung, name, semantics in _RUNGS:
        applicable, evidence = _rung_applicability(rung, step, traj)
        est = eng.recover_rate(traj, step_index, rung)
        rungs.append(
            LadderRung(
                rung=rung,
                name=name,
                semantics=semantics,
                applicable=applicable,
                evidence=evidence,
                estimated_success=est[0] if est else None,
            )
        )
    return LadderResponse(
        step_index=step_index,
        engine_attached=eng.attached,
        engine_note=eng.note,
        rungs=rungs,
    )


def _rung_applicability(rung: int, step: Step, traj: Trajectory) -> tuple[bool, str]:
    s = step.signals
    if rung == 0:


        return True, "Always testable. Most informative when the agent had the needed context."
    if rung == 1:
        if s.is_error or s.test_failed:
            return True, "This step has a tool error / failed test whose output could be perturbed."
        return False, "No erroneous observation at this step to perturb."
    if rung == 2:
        if s.near_compaction:
            return True, "A compaction wall is within range — a constraint may have been dropped to restore."
        return False, "No compaction wall nearby; no obviously-dropped constraint to restore."
    if rung == 3:
        if s.is_edit or step.phase.value in ("edit", "debug", "explore"):
            return True, "This is an actionable decision point with a clear alternative action."
        return False, "Passive step; overriding it is unlikely to change much."
    return False, ""
