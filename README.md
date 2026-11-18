# PostgreSQL documentation RAG — retrieval evaluation

A small Python project for searching a fixed set of PostgreSQL 18 documentation pages and asking questions against the retrieved text. The main experiment is retrieval quality: I compared two ways of chunking the same documents and measured how often each method returned the expected source.

This is deliberately small. It uses a local Qdrant instance, a sentence-transformer embedding model, FastAPI, and Ollama (`llama3.2:3b`). It does not need a hosted LLM or an agent framework.

## What it does

- Downloads an allowlisted set of 30 PostgreSQL 18 documentation pages and stores the extracted content as JSONL.
- Builds two Qdrant collections: `fixed` (300-word chunks with 50-word overlap) and `paragraph` (paragraph-aware grouping up to 300 words). These are *word* counts, not tokenizer counts.
- Exposes `GET /search` for dense-vector retrieval and `POST /ask` for an answer based on retrieved excerpts. `/ask` returns numbered citations and the corresponding documentation URLs.
- Scores retrieval against 30 question–source pairs using Recall@k, Hit@5 and MRR@10. Unit tests cover the metric calculations and response-handling failures.

Citations are checked against the excerpts returned by search. That only establishes that a cited excerpt exists; it does **not** prove every generated claim is entailed by the excerpt.

## Run it locally

Tested by the project author on macOS with Docker Desktop, Python in a virtual environment, and Ollama. The commands assume you are in the repository root.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
docker compose up -d
```

The local Qdrant dashboard is at <http://localhost:6333/dashboard>. Docker Compose starts Qdrant only; FastAPI runs on the host.

If you have not indexed the docs yet:

```bash
python -m app.fetch_docs
python -m app.ingest --strategy both
```

Ingestion recreates the target collections, so don't rerun it on collections you need to preserve. The first run downloads the embedding model. The corpus comes from <https://www.postgresql.org/docs/18/>; this project doesn't distribute the full downloaded documentation.

Install [Ollama](https://ollama.com/download/mac), start it, and pull the model:

```bash
ollama pull llama3.2:3b
ollama list
```

Run the API in one terminal:

```bash
uvicorn app.main:app --reload
```

Try it in another terminal, or use <http://127.0.0.1:8000/docs>:

```bash
curl -sG 'http://127.0.0.1:8000/search' \
  --data-urlencode 'q=What is a partial index?' \
  --data-urlencode 'strategy=fixed' \
  --data-urlencode 'k=5'

curl -sS -X POST 'http://127.0.0.1:8000/ask' \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is a partial index?","strategy":"fixed","k":5}' \
  | python -m json.tool
```

The `/ask` response contains the answer, the model, the retrieval strategy and a `sources` array. Source entries include a section ID, title, URL and chunk ID. The API returns an explicit insufficient-evidence answer when retrieval is empty or the model supplies no valid citations. If Ollama is unavailable, it returns a service error rather than a fabricated answer.

## Retrieval evaluation

The labelled dataset lives in `evaluation/questions.jsonl`. Each question has one or more expected **documentation section IDs**, not chunk IDs, so both chunkers can be evaluated against the same labels. The questions were checked against the downloaded source material before the benchmark.

```bash
python -m evaluation.evaluate --strategy both
python -m pytest -q
```

The evaluation script saves question-level rankings and summary scores to `results/retrieval_results.json` (generated locally; ignored by Git). A compact record of the observed benchmark is kept in [`results/benchmark-summary.json`](results/benchmark-summary.json).

### Results — 18 November 2026

30 verified question–source pairs, 30 PostgreSQL documentation pages, dense retrieval. Both chunking strategies used the same corpus and embedding model. These are results from one local evaluation run, not an average of repeated benchmarks.

| Metric | Fixed | Paragraph |
|---|---:|---:|
| Recall@1 | **73.3%** | 66.7% |
| Recall@3 | 93.3% | 93.3% |
| Recall@5 | 96.7% | 96.7% |
| Recall@10 | 100% | 100% |
| MRR@10 | **0.846** | 0.803 |
| Mean latency | 76.9 ms | 11.3 ms |

The measured corpus SHA-256 was `b8c6e691ad0c9b298dee1b4d8af67b94dc59722fd6308005d7907caa2af505b6`.

The fixed chunker ranked the expected section first for 22/30 questions; paragraph chunking did so for 20/30. Both found the expected section in the top five for 29/30. For `pg-001`, fixed-size retrieval missed the target in its top five; for `pg-028` (EXPLAIN), paragraph chunking missed it in its top five.

**Latency caveat:** the fixed run included a ~1,047 ms first query, so its mean latency isn't a fair head-to-head speed comparison. A proper timing study would warm both configurations and collect repeated p50/p95 measurements.

**Evaluation caveat:** metrics are computed over **unique documentation sections**. Multiple matching chunks from one page count as one section. The dataset is small, is not held out from development, and predominantly contains answerable questions. These results do not establish broad generalisation or answer faithfulness.

## A few things I learned

Fixed-size chunks placed the right source first a little more often on this set, but that advantage disappears by Recall@3. That makes ranking quality worth tracking separately from whether a source appears at all. Some queries also have several closely related documentation pages, so a single gold section can be an imperfect measure of whether retrieval was useful.

Building the answer endpoint uncovered a separate issue: a small local model could produce a correct-looking answer but omit citation markers. I switched to structured JSON citations, validated the indices against retrieved excerpts, and collapsed duplicate citations pointing to different chunks of the same page. The model can still cite an irrelevant excerpt; this is not a substitute for faithfulness evaluation.

## Limitations and next steps

- The current benchmark tests **dense retrieval only**. Hybrid/BM25 retrieval and hosted-versus-local model comparisons are not implemented in this version.
- The corpus is only 30 selected documentation pages, and documentation can change upstream despite the versioned URL. The stored SHA identifies the evaluated local snapshot.
- A stronger test set would include more paraphrases, multi-source questions, and unanswerable questions; an independent reviewer should check the gold labels.
- The `/ask` path has deterministic unit tests for citation validation and error handling, but no measured answer-quality or faithfulness score. An LLM judge would need calibration against human-reviewed examples.
- Query latency should be benchmarked after warm-up, over repeated runs.

## Project layout

```text
app/            downloader, chunkers, Qdrant ingestion, retrieval API, Ollama answering
evaluation/     verified questions, retrieval metrics, benchmark runner
tests/          offline regression tests
results/        small committed benchmark summary; full output generated locally
docker-compose.yml
```

The documentation belongs to the PostgreSQL project and contributors. Original sources: <https://www.postgresql.org/docs/18/>.
