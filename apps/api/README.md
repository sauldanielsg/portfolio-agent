# Portfolio API

From this directory:

```bash
uv sync
AI_MODE=mock RAG_ENABLED=false uv run uvicorn api.main:app --app-dir src --reload
```

Run tests with `AI_MODE=mock RAG_ENABLED=false uv run pytest` so local `.env`
settings do not initialize the real RAG model during test collection.
See the root README for configuration and frontend setup.

For LangChain with local RAG, configure `.env` with `AI_MODE=gemini`,
`AGENT_BACKEND=langchain`, and `RAG_ENABLED=true`, plus your key and model.

```bash
uv sync --extra local-embeddings
uv run --extra local-embeddings python -m api.cli.ingest
uv run --extra local-embeddings uvicorn api.main:app --app-dir src --reload
```

Re-run ingestion after profile edits or when the index is missing or stale,
then restart the backend and refresh the browser. See
[the root README](../../README.md#langchain-and-local-rag) for full instructions.

With RAG enabled, each knowledge request searches once. Empty results return a
normal "not documented" answer; otherwise one generation call uses the evidence.
An initial classification/context call preserves follow-ups and resume requests.
The workflow uses at most two model calls per message, with quota reserved for each.
