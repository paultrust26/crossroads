import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from crossroads.app import app  # noqa: E402
from crossroads.settings import project_label  # noqa: E402


def test_health():
    assert TestClient(app).get("/api/health").status_code == 200


def test_project_label_strips_home_prefix(monkeypatch):
    home = Path("/home/alice")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert project_label("-home-alice-code-myrepo") == "code-myrepo"


def test_project_label_keeps_unknown_names():
    assert project_label("standalone") == "standalone"


def test_swe_example_trajectory_serves_all_views():
    """The bundled SWE example loads and each SWE view builds from it."""
    from pathlib import Path

    from crossroads.swe_parser import build_swe_conversation, build_swe_overview, build_swe_spine
    from crossroads.topics import swe_query_map

    data = Path(__file__).resolve().parent.parent / "examples" / "swe"
    overview = build_swe_overview(data)
    assert overview.n_sessions >= 1
    sid = overview.sessions[0].id
    assert build_swe_spine(data, sid) is not None
    assert build_swe_conversation(data, sid) is not None
    assert swe_query_map(data) is not None


def test_query_map_handles_fewer_queries_than_t_sne_needs():
    """A query map with very few queries is returned instead of failing in t-SNE."""
    from pathlib import Path

    from crossroads.topics import swe_query_map

    result = swe_query_map(Path(__file__).resolve().parent.parent / "examples" / "swe")
    assert result.n_queries >= 1


def test_swe_step_observation_tokens_are_read_from_trajectory_files():
    """Per-step observation tokens use the obs_tokens_raw / obs_tokens_served names written by trajectory files."""
    from pathlib import Path

    from crossroads.swe_parser import build_swe_overview, build_swe_spine

    data = Path(__file__).resolve().parent.parent / "examples" / "swe"
    sessions = build_swe_overview(data).sessions
    assert sum(s.obs_tokens_raw for s in sessions) > 0
    spine = build_swe_spine(data, sessions[0].id)
    assert sum(step.raw_obs_tokens for step in spine.steps) > 0
