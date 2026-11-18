import json
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.answering import ABSTENTION, ask_ollama, build_prompt, install_ask_route, normalize_citations

class Hit(BaseModel):
    title: str = "Partial Indexes"
    url: str = "https://www.postgresql.org/docs/18/indexes-partial.html"
    section_id: str = "pg18:indexes-partial.html"
    chunk_id: str = "pg18:indexes-partial.html:0"
    text: str = "A partial index indexes only rows satisfying its predicate."

class Results:
    results = [Hit()]

def test_prompt_with_pydantic_hits():
    assert "[1] Partial Indexes" in build_prompt("What is a partial index?", Results.results)

def test_citation_normalization():
    assert normalize_citations([1, "[1]", "2", 0, 99, True, "foo"], 2) == [1, 2]

def test_ask_returns_cited_answer_from_pydantic_hits():
    app = FastAPI()
    install_ask_route(app, lambda **kw: Results())
    with patch("app.answering.ask_ollama", return_value={"answer":"An index over a subset of rows.","citations":[1]}):
        resp = TestClient(app).post("/ask", json={"question":"What is a partial index?"})
    assert resp.status_code == 200
    assert resp.json()["sources"][0]["section_id"] == "pg18:indexes-partial.html"
    assert resp.json()["answer"].endswith("[1]")

def test_invalid_citations_abstain():
    app = FastAPI()
    install_ask_route(app, lambda **kw: Results())
    with patch("app.answering.ask_ollama", return_value={"answer":"Some text.","citations":[]}):
        resp = TestClient(app).post("/ask", json={"question":"What is a partial index?"})
    assert resp.json()["answer"] == ABSTENTION

def test_ollama_json_response():
    class Reply:
        def raise_for_status(self): pass
        def json(self):
            return {"message":{"content":json.dumps({"answer":"Partial indexes index a subset.","citations":["[1]"]})}}
    with patch("app.answering.httpx.post", return_value=Reply()):
        result = ask_ollama("What is a partial index?", Results.results, base_url="http://localhost:11434", model="llama3.2:3b")
    assert result["citations"] == [1]


def test_duplicate_chunks_from_same_section_are_shown_once():
    app = FastAPI()
    class TwoChunks:
        results = [Hit(), Hit(chunk_id="pg18:indexes-partial.html:1")]
    install_ask_route(app, lambda **kw: TwoChunks())
    with patch("app.answering.ask_ollama", return_value={
        "answer": "Only rows matching the predicate are indexed.", "citations": [1, 2]
    }):
        response = TestClient(app).post("/ask", json={"question": "What is a partial index?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].endswith("[1]")
    assert len(body["sources"]) == 1
    assert body["sources"][0]["section_id"] == "pg18:indexes-partial.html"


def test_no_retrieved_results_abstains_without_calling_ollama():
    app = FastAPI()
    class Empty:
        results = []
    install_ask_route(app, lambda **kw: Empty())
    with patch("app.answering.ask_ollama") as mock_ollama:
        response = TestClient(app).post("/ask", json={"question": "No matching docs?"})
    assert response.status_code == 200
    assert response.json()["answer"] == ABSTENTION
    assert response.json()["sources"] == []
    mock_ollama.assert_not_called()


def test_out_of_range_citations_do_not_appear_in_answer():
    app = FastAPI()
    install_ask_route(app, lambda **kw: Results())
    with patch("app.answering.ask_ollama", return_value={
        "answer": "Unverified statement", "citations": [3]
    }):
        response = TestClient(app).post("/ask", json={"question": "What is a partial index?"})
    assert response.status_code == 200
    assert response.json()["answer"] == ABSTENTION
    assert response.json()["sources"] == []


def test_model_abstention_stays_abstention_even_with_a_citation():
    app = FastAPI()
    install_ask_route(app, lambda **kw: Results())
    with patch("app.answering.ask_ollama", return_value={
        "answer": ABSTENTION, "citations": [1]
    }):
        response = TestClient(app).post("/ask", json={"question": "What is the price of tea?"})
    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.json()["answer"] == ABSTENTION


def test_ollama_invalid_json_returns_502():
    from fastapi import HTTPException
    class Reply:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "not-json"}}
    with patch("app.answering.httpx.post", return_value=Reply()):
        try:
            ask_ollama("question", Results.results, base_url="http://localhost:11434", model="llama3.2:3b")
        except HTTPException as error:
            assert error.status_code == 502
        else:
            raise AssertionError("Malformed model output should fail")


def test_ollama_connection_error_returns_503():
    import httpx
    from fastapi import HTTPException
    with patch("app.answering.httpx.post", side_effect=httpx.ConnectError("offline")):
        try:
            ask_ollama("question", Results.results, base_url="http://localhost:11434", model="llama3.2:3b")
        except HTTPException as error:
            assert error.status_code == 503
        else:
            raise AssertionError("Offline model should fail")
