# Apply the final patch to the existing project

This patch is **not** a replacement repository. It updates only `app/answering.py`, `tests/test_answering.py`, and `README.md`, and adds `results/benchmark-summary.json`. Your `app/main.py`, corpus, Qdrant collections, evaluation code, environment, and Docker setup remain unchanged.

Before copying, commit or back up your current files.

From the existing project root:

```bash
cd ~/projects/rag-retrieval-eval
cp app/answering.py app/answering.py.before-final
cp README.md README.md.before-final
unzip -o ~/Downloads/rag-final-inplace-patch.zip
source venv/bin/activate
python -m pytest -q
```

The ZIP also contains this installation guide. Verify that the end of `app/main.py` still has:

```python
from app.answering import install_ask_route
install_ask_route(app, search)
```

Test the live API with Qdrant and Ollama running:

```bash
uvicorn app.main:app --reload
```

In another terminal:

```bash
curl -sS -X POST 'http://127.0.0.1:8000/ask' \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is a partial index?","strategy":"fixed","k":5}' \
  | python -m json.tool
```

Expected: a nonempty grounded answer and one `sources` entry for the Partial Indexes *page* even if Ollama selects two chunks from it. This is not an automatic proof of faithfulness.

Optional negative check: Ask about a topic absent from the selected documentation. The model should abstain, but small local models are not guaranteed to do so; log and discuss any failures rather than hiding them.

To publish to GitHub (using your existing repo):

```bash
git status
git add app/answering.py tests/test_answering.py README.md
git add -f results/benchmark-summary.json
git commit -m "Add citation regression tests and document retrieval benchmark"
git push
```

If no GitHub remote has been configured, create a repository and add its remote first. Do not commit `.env`, `venv/`, Qdrant data, or downloaded documents. `results/*.json` is currently ignored by the starter `.gitignore`, so `git add -f` is used **only** for the small benchmark summary. The complete `retrieval_results.json` remains local.

One maintenance note: your earlier logs showed `qdrant-client` 1.20.x against Qdrant server 1.15.5. Align client and server minor releases before declaring a reproducible install; avoid suppressing the version warning as a permanent fix.
