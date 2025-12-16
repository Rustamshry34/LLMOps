"""
Smoke test for /generate endpoint
docker run -p 8000:8000 nizami-1.7b
"""
import requests
import pytest

URL = "http://localhost:8000/generate"
HEALTH = "http://localhost:8000/health"


def test_health():
    r = requests.get(HEALTH)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_generate():
    payload = {"prompt": "How is D-dimer used in VTE?"}
    r = requests.post(URL, json=payload)
    assert r.status_code == 200
    assert "text" in r.json()
    assert len(r.json()["text"]) > 10
