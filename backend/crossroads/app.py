"""Crossroads FastAPI backend.

Endpoints:
  GET  /api/health
  GET  /api/trajectories                       -> list discoverable Claude sessions
  GET  /api/trajectories/{id}                  -> Spine: meta + scored steps + walls
  GET  /api/trajectories/{id}/step/{i}/fan     -> Counterfactual Fan for a step
  GET  /api/trajectories/{id}/step/{i}/ladder  -> Intervention Ladder for a step
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .compaction import build_compaction_lens
from .compaction_diff import CompactionDiffResponse, build_compaction_diff
from .conversation import ConversationResponse, build_conversation
from .corpus_analysis import (
    CorpusFailureResponse,
    CorpusMitigationResponse,
    build_corpus_failures,
    build_corpus_mitigations,
    stream_batch_judge,
)
from .economics import (
    FailureResponse,
    TokenBudgetResponse,
    WasteResponse,
    failures,
    token_budget,
    waste,
)
from .engine import build_fan, build_ladder, get_engine
from .failure_analysis import (
    FailureAnalysisResponse,
    JudgeRunResponse,
    analyze_failure,
    run_judge,
    stream_judge,
)
from .anomaly_audit import AnomalyAuditResponse, analyze_anomalies, stream_anomaly_judge, stream_batch_anomaly_judge
from .filemap import FileMapResponse, build_filemap
from .judge import JudgeEstimateResponse, estimate_judges
from .timeseries import FullTimeSeriesResponse, build_full_timeseries
from .overview import CorpusResponse, OverviewResponse, build_corpus, build_overview
from .models import (
    CompactionLensResponse,
    FanResponse,
    LadderResponse,
    SpineResponse,
)
from .store import discover, get_path, get_trajectory
from .swe_failure_analysis import (
    SweFailureAnalysisResponse,
    analyze_swe_failure,
    run_swe_judge,
    stream_swe_judge,
)
from .swe_parser import (
    SweConversationResponse,
    SweOverviewResponse,
    SweSpineResponse,
    build_swe_conversation,
    build_swe_overview,
    build_swe_spine,
)
from .session_segments import SegmentationResponse, segment_session
from .topics import QueryMapResponse, corpus_query_map, session_query_map, swe_query_map

app = FastAPI(title="Crossroads", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    eng = get_engine()
    return {"ok": True, "engine_attached": eng.attached}


@app.get("/api/trajectories")
def trajectories() -> dict:
    return {"sessions": discover()}


@app.get("/api/overview", response_model=OverviewResponse)
def overview() -> OverviewResponse:
    return build_overview()


@app.get("/api/corpus", response_model=CorpusResponse)
def corpus() -> CorpusResponse:
    return build_corpus()


@app.get("/api/corpus/querymap", response_model=QueryMapResponse)
def corpus_querymap(kind: str = Query(default="query", pattern="^(query|response)$")) -> QueryMapResponse:
    from .overview import summarize_session
    paths = []
    for info in discover():
        p = get_path(info["id"])
        if p is None:
            continue
        try:
            ncomp = summarize_session(p).n_compactions
        except OSError:
            ncomp = 0
        paths.append((info["id"], p, ncomp))
    return corpus_query_map(paths, kind=kind)


@app.get("/api/trajectories/{traj_id}", response_model=SpineResponse)
def trajectory(
    traj_id: str,
    max_steps: int | None = Query(default=None, ge=1, le=5000),
) -> SpineResponse:
    traj = get_trajectory(traj_id, max_steps=max_steps)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    return SpineResponse(meta=traj.meta, steps=traj.steps, compactions=traj.compactions)


@app.get("/api/trajectories/{traj_id}/step/{i}/fan", response_model=FanResponse)
def fan(traj_id: str, i: int) -> FanResponse:
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    if not (0 <= i < len(traj.steps)):
        raise HTTPException(400, f"step {i} out of range (0..{len(traj.steps) - 1})")
    return build_fan(traj, i)


@app.get("/api/trajectories/{traj_id}/step/{i}/ladder", response_model=LadderResponse)
def ladder(traj_id: str, i: int) -> LadderResponse:
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    if not (0 <= i < len(traj.steps)):
        raise HTTPException(400, f"step {i} out of range (0..{len(traj.steps) - 1})")
    return build_ladder(traj, i)


@app.get("/api/trajectories/{traj_id}/querymap", response_model=QueryMapResponse)
def querymap(traj_id: str, kind: str = Query(default="query", pattern="^(query|response)$")) -> QueryMapResponse:
    path = get_path(traj_id)
    if path is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    return session_query_map(path, kind=kind)


@app.get("/api/trajectories/{traj_id}/conversation", response_model=ConversationResponse)
def conversation(traj_id: str, max_groups: int | None = Query(default=200, ge=1, le=2000)) -> ConversationResponse:
    path = get_path(traj_id)
    if path is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    return build_conversation(path, max_groups=max_groups)


@app.get("/api/trajectories/{traj_id}/tokens", response_model=TokenBudgetResponse)
def tokens(traj_id: str) -> TokenBudgetResponse:
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    return token_budget(traj)


@app.get("/api/trajectories/{traj_id}/waste", response_model=WasteResponse)
def waste_endpoint(traj_id: str) -> WasteResponse:
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    return waste(traj)


@app.get("/api/trajectories/{traj_id}/failures", response_model=FailureResponse)
def failures_endpoint(traj_id: str) -> FailureResponse:
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    return failures(traj)


@app.get("/api/trajectories/{traj_id}/failure-analysis", response_model=FailureAnalysisResponse)
def failure_analysis(traj_id: str) -> FailureAnalysisResponse:
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    return analyze_failure(traj)


@app.post("/api/trajectories/{traj_id}/failure-analysis/judge", response_model=JudgeRunResponse)
async def failure_judge_run(traj_id: str) -> JudgeRunResponse:
    """Run the LLM judge on a trajectory (non-streaming fallback)."""
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    heuristic_result = analyze_failure(traj).heuristic
    judge_result = await run_judge(traj)
    return JudgeRunResponse(
        session_id=traj.meta.id,
        n_steps=len(traj.steps),
        heuristic=heuristic_result,
        judge=judge_result,
        note="LLM judge analysis complete.",
    )


@app.get("/api/trajectories/{traj_id}/failure-analysis/judge/stream")
async def failure_judge_stream(traj_id: str):
    """Stream LLM judge reasoning token-by-token (SSE)."""
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    return StreamingResponse(
        stream_judge(traj),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/trajectories/{traj_id}/segments", response_model=SegmentationResponse)
def session_segments(traj_id: str) -> SegmentationResponse:
    path = get_path(traj_id)
    if path is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    return segment_session(path, session_id=traj_id)


@app.get("/api/corpus/anomaly-audit")
def corpus_anomaly_audit():
    """Run anomaly audit across all sessions (lightweight, no full parse)."""
    from .anomaly_audit import analyze_anomalies_lightweight
    sessions = discover()
    results = []
    for s in sessions:
        try:
            entry = analyze_anomalies_lightweight(s["id"], s["path"], s.get("project", ""))
            if entry:
                results.append(entry)
        except Exception:
            continue
    results.sort(key=lambda r: r["risk_score"], reverse=True)
    return {"sessions": results}


@app.get("/api/trajectories/{traj_id}/anomaly-audit", response_model=AnomalyAuditResponse)
def anomaly_audit(traj_id: str) -> AnomalyAuditResponse:
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    return analyze_anomalies(traj)


@app.get("/api/trajectories/{traj_id}/anomaly-audit/judge/stream")
async def anomaly_judge_stream(traj_id: str):
    """Stream anomaly audit judge reasoning token-by-token (SSE)."""
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    return StreamingResponse(
        stream_anomaly_judge(traj),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/corpus/anomaly-audit/batch-judge/stream")
async def corpus_anomaly_batch_judge_stream(
    session_ids: str = Query(..., description="Comma-separated session IDs"),
):
    """Stream batch anomaly judge results as SSE — one result per session."""
    ids = [s.strip() for s in session_ids.split(",") if s.strip()]
    if not ids:
        raise HTTPException(400, "No session_ids provided")
    return StreamingResponse(
        stream_batch_anomaly_judge(ids),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/trajectories/{traj_id}/judge-estimate", response_model=JudgeEstimateResponse)
def judge_estimate(traj_id: str) -> JudgeEstimateResponse:
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    return estimate_judges(traj)


@app.get("/api/trajectories/{traj_id}/compaction/{wall}", response_model=CompactionLensResponse)
def compaction_lens(traj_id: str, wall: int) -> CompactionLensResponse:
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    lens = build_compaction_lens(traj, wall)
    if lens is None:
        walls = [c.at_index for c in traj.compactions]
        raise HTTPException(400, f"step {wall} is not a compaction wall. walls: {walls}")
    return lens


@app.get("/api/trajectories/{traj_id}/compaction/{wall}/diff", response_model=CompactionDiffResponse)
def compaction_diff(traj_id: str, wall: int) -> CompactionDiffResponse:
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    diff = build_compaction_diff(traj, wall)
    if diff is None:
        walls = [c.at_index for c in traj.compactions]
        raise HTTPException(400, f"step {wall} is not a compaction wall. walls: {walls}")
    return diff


@app.get("/api/trajectories/{traj_id}/filemap", response_model=FileMapResponse)
def filemap(traj_id: str) -> FileMapResponse:
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    return build_filemap(traj)


@app.get("/api/trajectories/{traj_id}/timeseries")
def timeseries(traj_id: str, n_bins: int = Query(default=40, ge=10, le=200)):
    traj = get_trajectory(traj_id)
    if traj is None:
        raise HTTPException(404, f"trajectory '{traj_id}' not found")
    ts = build_full_timeseries(traj, n_bins=n_bins)
    if ts is None:
        raise HTTPException(400, "session has insufficient timestamps for time-series")
    return ts




@app.get("/api/corpus/failures", response_model=CorpusFailureResponse)
def corpus_failures(source: str = Query(default="all", pattern="^(all|claude|swe)$")) -> CorpusFailureResponse:
    return build_corpus_failures(source=source)


@app.get("/api/corpus/judge-cache")
def judge_cache(source: str = Query(default=None)):
    """Return all cached judge results (for restoring sidebar on refresh)."""
    from .judge_cache import get_all_cached
    return {"results": get_all_cached(source=source)}


@app.get("/api/corpus/failures/batch-judge/stream")
async def corpus_batch_judge_stream(
    session_ids: str = Query(..., description="Comma-separated session IDs"),
    source: str = Query(default="claude", pattern="^(claude|swe)$"),
):
    """Stream batch judge results as SSE — one result per session."""
    ids = [s.strip() for s in session_ids.split(",") if s.strip()]
    if not ids:
        raise HTTPException(400, "No session_ids provided")
    return StreamingResponse(
        stream_batch_judge(ids, source=source),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/corpus/mitigations", response_model=CorpusMitigationResponse)
def corpus_mitigations(source: str = Query(default="claude", pattern="^(claude|swe)$")) -> CorpusMitigationResponse:
    return build_corpus_mitigations(source=source)


@app.get("/api/actions/suggestions")
def action_suggestions():
    """Surface actionable patterns from corpus analysis for CLAUDE.md / hooks."""
    from pathlib import Path
    from .actions import (ActionsResponse, build_actions, build_failure_actions, build_waste_actions,
        build_anomaly_actions, extract_user_instructions, build_user_instruction_actions)

    session_paths = []
    for info in discover():
        p = get_path(info["id"])
        if not p: continue
        session_paths.append(Path(p))


    failure_resp = build_corpus_failures(source="claude")
    failure_data = {"top_modes": failure_resp.top_modes}


    waste_resp = build_corpus_mitigations(source="claude")
    waste_data = {
        "by_cause": waste_resp.waste.by_cause,
        "by_cause_cost": waste_resp.waste.by_cause_cost,
        "total_waste_usd": waste_resp.waste.total_waste_usd,
        "mean_reread_rate": waste_resp.waste.mean_reread_rate,
    }


    from .anomaly_audit import analyze_anomalies_lightweight
    anomaly_sessions = []
    for s in discover():
        try:
            entry = analyze_anomalies_lightweight(s["id"], s["path"], s.get("project", ""))
            if entry:
                anomaly_sessions.append(entry)
        except Exception:
            continue
    anomaly_data = {"sessions": anomaly_sessions}

    return build_actions(session_paths, failure_data, waste_data, anomaly_data)




@app.get("/api/swe/overview", response_model=SweOverviewResponse)
def swe_overview() -> SweOverviewResponse:
    return build_swe_overview()


@app.get("/api/swe/querymap", response_model=QueryMapResponse)
def swe_querymap(kind: str = Query(default="query", pattern="^(query|response)$")) -> QueryMapResponse:
    from .swe_parser import _DATA_DIR
    return swe_query_map(_DATA_DIR, kind=kind)


@app.get("/api/swe/sessions/{session_id}/spine", response_model=SweSpineResponse)
def swe_spine(session_id: str) -> SweSpineResponse:
    try:
        return build_swe_spine(session_id=session_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/api/swe/sessions/{session_id}/conversation", response_model=SweConversationResponse)
def swe_conversation(session_id: str) -> SweConversationResponse:
    try:
        return build_swe_conversation(session_id=session_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/api/swe/sessions/{session_id}/failure-analysis", response_model=SweFailureAnalysisResponse)
def swe_failure_analysis(session_id: str) -> SweFailureAnalysisResponse:
    try:
        return analyze_swe_failure(session_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/api/swe/sessions/{session_id}/failure-analysis/judge", response_model=SweFailureAnalysisResponse)
async def swe_failure_judge(session_id: str) -> SweFailureAnalysisResponse:
    """Run LLM judge on a SWE session (non-streaming fallback)."""
    try:
        analysis = analyze_swe_failure(session_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    judge_result = await run_swe_judge(session_id)
    analysis.judge = judge_result
    analysis.note = "Tier 2 (LLM judge) analysis complete."
    return analysis


@app.get("/api/swe/sessions/{session_id}/failure-analysis/judge/stream")
async def swe_failure_judge_stream(session_id: str):
    """Stream SWE LLM judge reasoning token-by-token (SSE)."""
    return StreamingResponse(
        stream_swe_judge(session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
