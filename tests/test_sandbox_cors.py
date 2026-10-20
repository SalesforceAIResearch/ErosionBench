"""The sandbox CORS policy admits only local origins (no wildcard, no credentials)."""
from fastapi.testclient import TestClient

from erosionbench.sandbox.server import app

client = TestClient(app)


def _preflight(origin: str):
    """Send a CORS preflight for a GET from ``origin``."""
    return client.options("/admin/health", headers={
        "Origin": origin, "Access-Control-Request-Method": "GET"})


def test_local_origins_allowed():
    for origin in ("http://localhost:3000", "http://127.0.0.1:8199", "http://localhost"):
        resp = _preflight(origin)
        assert resp.headers.get("access-control-allow-origin") == origin


def test_foreign_origin_rejected():
    resp = _preflight("https://evil.example.com")
    assert "access-control-allow-origin" not in resp.headers


def test_no_wildcard_or_credentials():
    resp = client.get("/admin/health", headers={"Origin": "http://localhost:3000"})
    assert resp.headers.get("access-control-allow-origin") != "*"
    assert "access-control-allow-credentials" not in resp.headers
