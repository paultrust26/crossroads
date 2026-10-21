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
