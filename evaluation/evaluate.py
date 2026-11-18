"""Run repeatable retrieval-only experiments against existing Phase 1 collections.

Example:
    python -m evaluation.evaluate --strategy both --questions evaluation/questions.jsonl
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank_at_k, unique_sections


def load_questions(path: Path) -> list[dict]:
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if not questions:
        raise ValueError("Question file is empty")
    seen = set()
    for item in questions:
        if not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("Every question needs a nonempty string id")
        if item["id"] in seen:
            raise ValueError(f"Repeated question id: {item['id']}")
        seen.add(item["id"])
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"Question {item['id']} has no question text")
        if item.get("status") != "verified":
            raise ValueError(f"Question {item['id']} is not verified; check labels first")
        gold = item.get("relevant_section_ids")
        if not isinstance(gold, list) or not gold or not all(isinstance(s, str) for s in gold):
            raise ValueError(f"Question {item['id']} needs relevant_section_ids")
        if len(gold) != len(set(gold)):
            raise ValueError(f"Question {item['id']} contains duplicate gold sections")
    return questions


def verify_gold_sources(questions: list[dict], corpus_path: Path) -> str:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus snapshot missing: {corpus_path}")
    content = corpus_path.read_bytes()
    corpus_ids = {json.loads(line)["section_id"] for line in content.decode("utf-8").splitlines()
                  if line.strip()}
    unknown = sorted({s for item in questions for s in item["relevant_section_ids"]
                      if s not in corpus_ids})
    if unknown:
        raise ValueError(f"Gold source IDs not found in corpus: {unknown}")
    return hashlib.sha256(content).hexdigest()


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No results")
    return {
        "questions": len(rows),
        "recall@1": mean(r["recall@1"] for r in rows),
        "recall@3": mean(r["recall@3"] for r in rows),
        "recall@5": mean(r["recall@5"] for r in rows),
        "recall@10": mean(r["recall@10"] for r in rows),
        "hit@5": mean(r["hit@5"] for r in rows),
        "mrr@10": mean(r["mrr@10"] for r in rows),
        "mean_latency_ms": mean(r["latency_ms"] for r in rows),
    }


def evaluate(questions: list[dict], *, strategy: str, client, model, k: int = 10) -> tuple[dict, list[dict]]:
    # Imported here so metric unit tests do not require Qdrant or Torch.
    from app.store import collection_name

    if k < 10:
        raise ValueError("k must be at least 10 to calculate recall@10 and MRR@10")
    collection = collection_name(strategy)
    if not client.collection_exists(collection):
        raise RuntimeError(f"Missing Qdrant collection {collection}; run ingestion first")

    rows = []
    for item in questions:
        started = perf_counter()
        vector = model.encode(item["question"], normalize_embeddings=True).tolist()
        # Fetch extra chunks: many chunks can come from the same source section.
        matches = client.query_points(collection_name=collection, query=vector,
                                      limit=max(100, k), with_payload=True).points
        elapsed_ms = (perf_counter() - started) * 1000
        ranked = unique_sections([p.payload["section_id"] for p in matches if p.payload])
        relevant = set(item["relevant_section_ids"])
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "category": item.get("category", "unspecified"),
            "gold_sections": sorted(relevant),
            "ranked_sections": ranked[:k],
            "recall@1": recall_at_k(ranked, relevant, 1),
            "recall@3": recall_at_k(ranked, relevant, 3),
            "recall@5": recall_at_k(ranked, relevant, 5),
            "recall@10": recall_at_k(ranked, relevant, 10),
            "hit@5": hit_at_k(ranked, relevant, 5),
            "mrr@10": reciprocal_rank_at_k(ranked, relevant, 10),
            "latency_ms": round(elapsed_ms, 2),
        })
    return summarize(rows), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 dense retrieval")
    parser.add_argument("--questions", type=Path, default=Path("evaluation/questions.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/postgres18_sections.jsonl"))
    parser.add_argument("--strategy", choices=("fixed", "paragraph", "both"), default="both")
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_results.json"))
    args = parser.parse_args()

    questions = load_questions(args.questions)
    sha = verify_gold_sources(questions, args.corpus)
    from app.store import get_client, get_model
    client, model = get_client(), get_model()
    strategies = ("fixed", "paragraph") if args.strategy == "both" else (args.strategy,)
    output = {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
              "corpus_sha256": sha, "question_count": len(questions), "configurations": {}}
    for strategy in strategies:
        summary, rows = evaluate(questions, strategy=strategy, client=client, model=model)
        output["configurations"][strategy] = {"summary": summary, "per_question": rows}
        print(f"{strategy:10s} recall@5={summary['recall@5']:.3f} "
              f"MRR@10={summary['mrr@10']:.3f} "
              f"mean latency={summary['mean_latency_ms']:.0f} ms")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
