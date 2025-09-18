import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def patch_backend_imports(monkeypatch):
    # Mock heavy deps before importing app
    mock_model = MagicMock()

    class _Vec:
        def tolist(self):
            return [0.1, 0.2, 0.3]

    mock_model.encode.return_value = _Vec()

    mock_index = MagicMock()
    mock_index.query.return_value = {
        "matches": [
            {
                "id": "http://example.com/a",
                "score": 0.9,
                "metadata": {
                    "title": "Title A",
                    "summary": "Summary A",
                    "category": "catA",
                    "published": "2025-01-01",
                },
            },
            {
                "id": "http://example.com/b",
                "score": 0.8,
                "metadata": {
                    "title": "Title B",
                    "summary": "Summary B",
                    "category": "catB",
                    "published": "2025-01-02",
                },
            },
        ]
    }
    mock_index.describe_index_stats.return_value = {"total_vector_count": 1234}

    mock_pc = MagicMock()
    mock_pc.Index.return_value = mock_index

    with (
        patch("src.backend.SentenceTransformer", return_value=mock_model),
        patch("src.backend.pinecone.Pinecone", return_value=mock_pc),
    ):
        yield


def get_client():
    from src.backend import app

    return TestClient(app)


def test_search_basic():
    client = get_client()
    resp = client.post("/search", json={"query": "hello", "top_k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"results", "metrics"}
    assert len(data["results"]) == 2
    assert data["metrics"]["top_k"] == 2
    assert data["metrics"]["total_vectors"] == 1234
    assert data["metrics"]["filtered"] is False


def test_search_with_categories():
    client = get_client()
    resp = client.post(
        "/search",
        json={"query": "hello cats", "top_k": 1, "categories": ["catA", "catB"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metrics"]["filtered"] is True


def test_search_single_category_and_stats_error(monkeypatch):
    # Force describe_index_stats to raise to hit exception path
    from src import backend as be

    client = get_client()

    # Patch get_index to a MagicMock with behaviors
    idx = MagicMock()
    idx.query.return_value = {"matches": []}
    idx.describe_index_stats.side_effect = Exception("boom")
    monkeypatch.setattr(be, "get_index", lambda: idx)

    resp = client.post(
        "/search",
        json={"query": "x", "top_k": 1, "categories": ["only"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metrics"]["filtered"] is True
    assert data["metrics"]["top_k"] == 1


def test_search_uses_default_top_k_and_empty_categories():
    client = get_client()
    resp = client.post(
        "/search",
        json={"query": "default topk", "categories": []},
    )
    assert resp.status_code == 200
    data = resp.json()
    # default top_k=5 from model
    assert data["metrics"]["top_k"] == 5
    # empty list means no filter
    assert data["metrics"]["filtered"] is False


def test_search_invalid_body():
    client = get_client()
    resp = client.post("/search", json={})
    assert resp.status_code == 422
