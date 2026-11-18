import pytest

from evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank_at_k, unique_sections
from evaluation.evaluate import load_questions, summarize, verify_gold_sources


def test_metrics_for_single_gold_section():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"b"}, 1) == 0
    assert recall_at_k(ranked, {"b"}, 3) == 1
    assert reciprocal_rank_at_k(ranked, {"b"}, 10) == 0.5
    assert hit_at_k(ranked, {"b"}, 1) == 0


def test_multiple_relevant_sections():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"b", "c"}, 2) == 0.5
    assert recall_at_k(ranked, {"b", "c"}, 3) == 1


def test_deduplication_keeps_first_rank():
    assert unique_sections(["a", "a", "b", "a", "c"]) == ["a", "b", "c"]


def test_invalid_input():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 5)
    with pytest.raises(ValueError):
        reciprocal_rank_at_k(["a"], {"a"}, 0)


def test_reject_unverified_questions(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"id":"x","question":"test","status":"draft","relevant_section_ids":["a"]}\n')
    with pytest.raises(ValueError, match="not verified"):
        load_questions(path)


def test_gold_ids_checked_against_corpus(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"section_id":"pg18:test","text":"test"}\n')
    questions = [{"relevant_section_ids": ["pg18:missing"]}]
    with pytest.raises(ValueError, match="not found"):
        verify_gold_sources(questions, corpus)


def test_summary():
    row = {f"recall@{k}": 1.0 for k in (1, 3, 5, 10)}
    row.update({"hit@5": 1.0, "mrr@10": 1.0, "latency_ms": 20})
    assert summarize([row])["mrr@10"] == 1.0
