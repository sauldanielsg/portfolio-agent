# AI Portfolio Agent

A reusable portfolio with a conversational AI assistant, built incrementally as a learning project.

Track remaining work in [TODO.md](TODO.md). The immediate next step is the
[end-to-end regression checklist](docs/regression-checklist.md), followed by visuals.

## Structure

```text
apps/
  api/
    src/api/
      main.py       FastAPI application and router registration
      routes/       HTTP handlers: chat, profile, health
      core/         Settings, shared filesystem paths, runtime initialization
      schemas/      Pydantic chat and profile models
      agents/       Tool-calling agent, tools, quota middleware
      providers/    Direct Gemini SDK integration
      rag/          Chunking, embeddings, retrieval, single-retrieval workflow
      usage/        Daily quota accounting
      cli/          Ingestion and local search commands
    tests/          Tests grouped by the same responsibilities
  web/
    src/
      main.tsx      React entry point
      app/          Application state and page composition
      features/
        resume/     Resume view
      types/        Chat API and portfolio TypeScript types
      styles/       Global and application styles
      assets/       Static assets
profiles/
  default/          Portfolio owner content, outside application code
artifacts/          Generated retrieval indexes (ignored by Git)
```

The Python package is named `api`; use imports such as
`from api.schemas.chat import ChatRequest`. HTTP routes consume initialized
services from `api.core.runtime`, while `api.main` only assembles the application.
Shared paths in `api.core.paths` keep profile, artifact, and `.env` locations
independent of the subfolder containing a particular module.

## Current milestone

Mock and Gemini chat, bounded conversation context, a data-driven resume,
local daily call limits, and an assistant resting screen are implemented.
The LangChain backend supports a bounded RAG workflow over a versioned profile index.

## Run locally

Backend, from `apps/api`:

```bash
uv sync
AI_MODE=mock RAG_ENABLED=false uv run uvicorn api.main:app --app-dir src --reload
```

Frontend, in another terminal from `apps/web`:

```bash
npm install
npm run dev
```

Open the URL printed by Vite. API documentation is at http://localhost:8000/docs.
Ask for a resume to display it and minimize the conversation.

To test the resting screen, restart the backend with:

```bash
AI_ENABLED=false RAG_ENABLED=false uv run uvicorn api.main:app --app-dir src --reload
```

Send a chat message to trigger the resting screen. Restart with `AI_ENABLED=true`
to restore chat, then return to the conversation. This switch does not disable
the profile endpoint. Gemini mode is available when explicitly configured; mock mode is the default.

Profile content lives in `profiles/default/profile.json`. Restart the backend and
refresh the browser after changing it: the backend loads it at startup and React
fetches it on mount.

Frontend checks, from `apps/web`: `npm run build` and `npm run lint`.

Backend tests, from `apps/api`:

```bash
AI_MODE=mock RAG_ENABLED=false uv run pytest
```

Tests cover API responses, invalid chat inputs, profile loading, resume actions,
unavailable mode, and model validation. They run locally without AWS or AI credentials.

Quota tests also cover exhausted and zero limits, concurrent reservations, and
reset at midnight in the configured timezone. The in-memory quota guards each Gemini request before the provider is called.
Mock replies do not consume quota. The local counter resets on restart and is
shared only within one backend process.

## Planned runtime requirements

- Python 3.12+
- Node.js and npm for the frontend

Broader retrieval evaluation and AWS deployment are upcoming milestones.

## Gemini configuration

Use `apps/api/.env.example` as a reference for `apps/api/.env`; keep real keys out
of source control. Settings load that file automatically. Keep `AI_MODE=mock`
until you have verified that the configured model and Google project support
free-tier API access. Then set `GEMINI_MODEL` explicitly and `AI_MODE=gemini`,
and restart FastAPI. No paid fallback is configured.

Every Gemini attempt reserves one daily call, including failed attempts. Exhaustion
returns HTTP 429 with `DAILY_LIMIT_REACHED`. Provider failures return HTTP 503 with
`AI_UNAVAILABLE`; Google rate limits get a separate safe message. Mock replies and
disabled chat do not reserve calls. The quota is local, resets on restart, and is
not suitable for multiple workers or production deployment. An application quota
does not guarantee zero billing on a billing-enabled Google project.

The API integration tests replace the provider with a fake; tests make no model calls.

The browser sends up to 10 recent messages with each question. History provides
context for follow-ups, while profile data supplies professional facts. History
is held in browser memory and clears on refresh; there is no persistent session yet.

## LangChain and local RAG

All commands below run from `apps/api`. In `apps/api/.env`, keep your existing
`GEMINI_API_KEY` and configure:

```dotenv
AI_ENABLED=true
AI_MODE=gemini
AGENT_BACKEND=langchain
RAG_ENABLED=true
GEMINI_MODEL=gemini-3.5-flash-lite
MAX_OUTPUT_TOKENS=500
DAILY_MODEL_CALL_LIMIT=10
QUOTA_TIMEZONE=America/Los_Angeles
```

Verify model availability and free-tier eligibility for your Google project
before making live calls. The model above worked during local development;
availability and pricing can change.

Install the optional local embedding dependencies and build the index:

```bash
uv sync --extra local-embeddings
uv run --extra local-embeddings python -m api.cli.ingest
```

Ingestion reads `profiles/default/profile.json`, generates a content-hash version,
creates knowledge chunks, and embeds them with
`sentence-transformers/all-MiniLM-L6-v2`. The first run downloads model files;
embedding runs locally and does not call Gemini or consume its quota.
The generated index is saved to `artifacts/default/index.json` and is ignored by Git.

Start the backend with the same optional dependencies:

```bash
uv run --extra local-embeddings uvicorn api.main:app --app-dir src --reload
```

At startup, the backend loads the profile and saved index and checks the profile
version and embedding model. RAG requires both Gemini mode and the LangChain
backend. Loading does not re-embed documents; searches embed the query locally.

RAG uses application-controlled retrieval rather than a model-driven search loop:

1. One structured model call identifies resume requests, conversation, or knowledge
   questions. It also resolves context-dependent follow-ups from recent history.
2. Knowledge requests perform exactly one local search. Self-contained questions
   retain their wording. An empty result returns a fixed "not documented" message
   without asking the model to generate an answer or search related technologies.
3. Nonempty results go to one structured answer-generation call with their source
   metadata. The model cannot invoke tools or perform another search.

Resume requests invoke `get_resume` directly after classification and return the
existing `show_resume` action. Greetings do not search. Only the resume branch can
set a UI action; generated answers cannot trigger resume rendering.

Each RAG request uses at most two model calls, each reserved against the daily
quota. Empty searches, greetings, and resume requests use one classification call.
No retries or agent graph are used in the RAG path. Genuine provider failures and
exhausted daily quota still use the existing resting-screen error responses.

To inspect retrieval without a Gemini call:

```bash
uv run --extra local-embeddings python -m api.cli.search
```

Try asking the web assistant about Python pipeline experience, then a follow-up
such as "At which company?". The terminal logs `RAG retrieval: count=... chunk_ids=...`
once for each knowledge request; it does not log the visitor's question or evidence
text. Kubernetes should return zero matches and a normal "not documented" chat
response. Supported RAG answers display expandable source passages underneath.

The model selects chunk IDs; the backend validates them against this request's
retrieved evidence and supplies source titles, text, and document versions itself.
Unknown references suppress the generated answer; missing citations return an
insufficient-evidence response. Citations verify provenance, not automatic factual
entailment. Empty retrieval, resume confirmations, and non-RAG modes do not show
retrieval citations.

Before visual changes, follow the [end-to-end regression checklist](docs/regression-checklist.md)
for sources, follow-ups, resume rendering, quota exhaustion, and failure recovery.

### Updating profile content or recovering from a missing/stale index

After editing the profile, stop the backend and rebuild:

```bash
uv run --extra local-embeddings python -m api.cli.ingest
uv run --extra local-embeddings uvicorn api.main:app --app-dir src --reload
```

Refresh the browser afterward. Startup deliberately rejects an index whose
profile content version or embedding model does not match. Rebuilding also
restores an index missing from a fresh checkout.

To use LangChain without retrieval, set `RAG_ENABLED=false` and restart.
To use the direct SDK, also set `AGENT_BACKEND=sdk`. Keep `RAG_ENABLED=false`
for mock mode. Without RAG, the LangChain backend retains its bounded tool-calling
agent; those requests may use several model turns. In both paths, daily quota is
reserved separately for each model call, rather than once per chat message.
