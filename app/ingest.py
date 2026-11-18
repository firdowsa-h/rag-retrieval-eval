import argparse
from pathlib import Path

from app.chunking import fixed_chunks, paragraph_chunks
from app.documents import load_sections
from app.store import collection_name, get_client, get_model, ingest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/postgres18_sections.jsonl"))
    parser.add_argument("--strategy", choices=["fixed", "paragraph", "both"], default="both")
    args = parser.parse_args()

    sections = load_sections(args.input)
    if not sections:
        raise SystemExit("No documents found. Run python -m app.fetch_docs first.")
    model = get_model()
    client = get_client()
    strategies = ["fixed", "paragraph"] if args.strategy == "both" else [args.strategy]
    for strategy in strategies:
        chunker = fixed_chunks if strategy == "fixed" else paragraph_chunks
        chunks = [chunk for section in sections for chunk in chunker(section)]
        collection = collection_name(strategy)
        ingest(client, model, chunks, collection)
        print(f"Indexed {len(chunks)} chunks in {collection}")


if __name__ == "__main__":
    main()
