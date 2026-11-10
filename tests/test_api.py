import pytest
from fastapi.testclient import TestClient

import app as app_module
from src import config


class FakeGraph:
    """Stand-in for the compiled LangGraph: no network, no LLM."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def invoke(self, state):
        if self.error:
            raise self.error
        return self.result


ANSWERED = {
    "answer": "Agentic AI refers to autonomous systems.",
    "context": ["chunk one", "chunk two"],
    "sources": [
        {"source": "Ebook-Agentic-AI.pdf", "page": 3, "score": 0.848},
        {"source": "Ebook-Agentic-AI.pdf", "page": 4, "score": 0.846},
    ],
    "grounded": True,
    "score": 0.9,
}

REFUSED = {
    "answer": config.REFUSAL_MESSAGE,
    "context": ["chunk one"],
    "sources": [],
    "grounded": False,
    "score": 0.0,
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "graph", FakeGraph(result=ANSWERED))
    return TestClient(app_module.app)


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_empty_query_returns_400(client):
    r = client.post("/chat", json={"query": "   "})
    assert r.status_code == 400


def test_over_length_query_returns_422(client):
    r = client.post("/chat", json={"query": "a" * 1001})
    assert r.status_code == 422


def test_missing_query_field_returns_422(client):
    r = client.post("/chat", json={})
    assert r.status_code == 422


def test_chat_success_has_expected_fields(client):
    r = client.post("/chat", json={"query": "What is Agentic AI?"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "What is Agentic AI?"
    assert body["final_answer"] == ANSWERED["answer"]
    assert body["confidence_score"] == 0.9
    assert body["grounded"] is True
    assert body["retrieved_context_chunks"] == ANSWERED["context"]
    assert body["sources"] == ANSWERED["sources"]


def test_chat_refusal_has_no_sources(monkeypatch):
    monkeypatch.setattr(app_module, "graph", FakeGraph(result=REFUSED))
    c = TestClient(app_module.app)
    body = c.post("/chat", json={"query": "What is the capital of France?"}).json()
    assert body["final_answer"] == config.REFUSAL_MESSAGE
    assert body["grounded"] is False
    assert body["confidence_score"] == 0.0
    assert body["sources"] == []


def test_graph_failure_returns_503_without_leaking_details(monkeypatch):
    monkeypatch.setattr(
        app_module, "graph", FakeGraph(error=RuntimeError("secret internal detail"))
    )
    c = TestClient(app_module.app)
    r = c.post("/chat", json={"query": "What is Agentic AI?"})
    assert r.status_code == 503
    assert "secret internal detail" not in r.text


def test_cors_allows_configured_origin(client):
    r = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"