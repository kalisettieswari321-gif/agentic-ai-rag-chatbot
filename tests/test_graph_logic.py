import pytest

import src.graph as graph_module
from src import config


class FakeDoc:
    def __init__(self, text, page, source="Ebook-Agentic-AI.pdf"):
        self.page_content = text
        self.metadata = {"page": page, "source": source}


class FakeResponse:
    def __init__(self, content):
        self.content = content


def make_graph(monkeypatch, results, llm_outputs):
    """Build the real graph with fake Pinecone and Groq.

    results:     list of (FakeDoc, score) returned by similarity search
    llm_outputs: list of strings (returned in order) or Exceptions (raised)
    """
    calls = {"llm": 0}

    class FakeEmbeddings:
        def __init__(self, *a, **k):
            pass

    class FakeStore:
        def __init__(self, *a, **k):
            pass

        def similarity_search_with_score(self, q, k=4):
            return results

    class FakeLLM:
        def __init__(self, *a, **k):
            pass

        def invoke(self, prompt):
            out = llm_outputs[calls["llm"]]
            calls["llm"] += 1
            if isinstance(out, Exception):
                raise out
            return FakeResponse(out)

    monkeypatch.setattr(graph_module, "FastEmbedEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(graph_module, "PineconeVectorStore", FakeStore)
    monkeypatch.setattr(graph_module, "ChatGroq", FakeLLM)
    return graph_module.build_rag_graph(), calls


DOCS = [
    (FakeDoc("chunk A", 3.0), 0.85),
    (FakeDoc("chunk B", 4.0), 0.80),
]
GOOD = '{"groundedness": 0.9, "reason": "ok"}'


def test_low_retrieval_score_refuses_without_calling_llm(monkeypatch):
    low = [(FakeDoc("irrelevant", 1.0), config.MIN_RETRIEVAL_SCORE - 0.1)]
    g, calls = make_graph(monkeypatch, low, [])
    r = g.invoke({"question": "q"})
    assert r["answer"] == config.REFUSAL_MESSAGE
    assert r["grounded"] is False
    assert r["score"] == 0.0
    assert r["sources"] == []
    assert calls["llm"] == 0


def test_grounded_answer_returns_sources_and_confidence(monkeypatch):
    g, _ = make_graph(monkeypatch, DOCS, ["A real answer.", GOOD])
    r = g.invoke({"question": "q"})
    assert r["answer"] == "A real answer."
    assert r["grounded"] is True
    expected = round(config.W_RETRIEVAL * ((0.85 + 0.80) / 2) + config.W_GROUNDED * 0.9, 2)
    assert r["score"] == expected
    assert len(r["sources"]) == 2


def test_low_groundedness_is_refused_and_sources_cleared(monkeypatch):
    weak = '{"groundedness": 0.4, "reason": "unsupported"}'
    g, _ = make_graph(monkeypatch, DOCS, ["Embellished answer.", weak])
    r = g.invoke({"question": "q"})
    assert r["answer"] == config.REFUSAL_MESSAGE
    assert r["grounded"] is False
    assert r["score"] == 0.0
    assert r["sources"] == []


def test_blank_grader_reply_is_retried(monkeypatch):
    g, calls = make_graph(monkeypatch, DOCS, ["A real answer.", "", GOOD])
    r = g.invoke({"question": "q"})
    assert r["grounded"] is True
    assert calls["llm"] == 3


def test_grader_that_never_returns_json_fails_closed(monkeypatch):
    g, calls = make_graph(monkeypatch, DOCS, ["A real answer.", "", "no json", "still none"])
    r = g.invoke({"question": "q"})
    assert r["answer"] == config.REFUSAL_MESSAGE
    assert r["grounded"] is False
    assert calls["llm"] == 4


def test_model_refusal_gives_zero_score_and_no_sources(monkeypatch):
    g, _ = make_graph(monkeypatch, DOCS, [config.REFUSAL_MESSAGE])
    r = g.invoke({"question": "q"})
    assert r["answer"] == config.REFUSAL_MESSAGE
    assert r["score"] == 0.0
    assert r["sources"] == []


def test_api_error_propagates_instead_of_false_refusal(monkeypatch):
    g, _ = make_graph(
        monkeypatch, DOCS, ["A real answer.", RuntimeError("429 rate limit")]
    )
    with pytest.raises(RuntimeError):
        g.invoke({"question": "q"})


def test_page_is_returned_as_int(monkeypatch):
    g, _ = make_graph(monkeypatch, DOCS, ["A real answer.", GOOD])
    r = g.invoke({"question": "q"})
    assert [s["page"] for s in r["sources"]] == [3, 4]
    assert all(isinstance(s["page"], int) for s in r["sources"])