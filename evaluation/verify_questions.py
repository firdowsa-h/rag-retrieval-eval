"""Review gold-source labels without manually editing JSONL.

Run from the project root:
    python -m evaluation.verify_questions

Every answer is written back to questions.jsonl after the decision.
The script never treats a matching source ID as proof that the source answers the question.
"""

import argparse
import json
import os
from pathlib import Path
import textwrap


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Review RAG questions against downloaded PostgreSQL sections")
    parser.add_argument("--questions", type=Path, default=Path("evaluation/questions.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/postgres18_sections.jsonl"))
    parser.add_argument("--chars", type=int, default=1800, help="Number of characters per source preview")
    parser.add_argument("--all", action="store_true", help="Revisit questions already verified or rejected")
    args = parser.parse_args()

    questions = load_jsonl(args.questions)
    sources = {row["section_id"]: row for row in load_jsonl(args.corpus)}
    print(f"Loaded {len(questions)} questions and {len(sources)} source sections.")
    print("Y=verified  N=reject  S=skip  Q=quit  (each choice saves immediately)\n")

    for index, q in enumerate(questions):
        if not args.all and q.get("status") in ("verified", "rejected"):
            continue
        ids = q.get("relevant_section_ids", [])
        missing = [sid for sid in ids if sid not in sources]
        print("=" * 76)
        print(f"[{index + 1}/{len(questions)}] {q['id']} | {q.get('category', '-')}")
        print(textwrap.fill(q["question"], width=76))
        if missing:
            print("MISSING SOURCE IDS: " + ", ".join(missing))
            print("This cannot be verified until the source IDs are corrected.")
        else:
            for sid in ids:
                source = sources[sid]
                print(f"\nSOURCE: {sid}")
                print(f"TITLE: {source.get('title', '')}")
                print(f"URL: {source.get('url', '')}")
                content = source.get("text", "")
                print("\n" + content[:args.chars])
                if len(content) > args.chars:
                    print(f"\n... [{len(content) - args.chars} more characters; open URL for full context]")
        while True:
            try:
                choice = input("\n[Y]es / [N]o / [S]kip / [Q]uit: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nStopped. Decisions already saved.")
                return
            if choice in ("y", "n", "s", "q"):
                break
            print("Please type y, n, s or q.")
        if choice == "q":
            break
        if choice == "y" and missing:
            print("Not marked verified: one or more source IDs are missing.")
            continue
        if choice == "y":
            q["status"] = "verified"
            q.pop("review_note", None)
        elif choice == "n":
            q["status"] = "rejected"
            q["review_note"] = "Gold section does not reliably support the question; revise before evaluation."
        else:
            continue
        save_jsonl(args.questions, questions)
        print(f"Saved {q['id']} as {q['status']}.")

    statuses = {}
    for q in questions:
        s = q.get("status", "draft")
        statuses[s] = statuses.get(s, 0) + 1
    print("\nStatus summary: " + ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items())))
    print("Note: evaluation.evaluate currently requires every row in the dataset to be verified.")


if __name__ == "__main__":
    main()
