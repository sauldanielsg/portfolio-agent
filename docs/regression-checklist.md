# End-to-end regression checklist

Run this before visual changes and repeat it after major UI changes. Automated
checks use fake providers; the main browser flow uses your configured Gemini
project and consumes quota. No AWS deployment is needed.

## 1. Automated checks

From `apps/api`:

```bash
AI_MODE=mock RAG_ENABLED=false uv run pytest
```

From `apps/web`:

```bash
npm run build
npm run lint
```

These cover validation, quotas, provider errors, retrieval, persistence, and
citation provenance. They do not replace the browser interaction checks below.

## 2. Start the local application

Keep your API key in `apps/api/.env`. Use `AI_ENABLED=true`, `AI_MODE=gemini`,
`AGENT_BACKEND=langchain`, `RAG_ENABLED=true`, and your working `GEMINI_MODEL`.
Reuse the saved index; citation changes do not require ingestion. Rebuild only
if profile content, chunk generation, or the embedding model changed.

From `apps/api`:

```bash
DAILY_MODEL_CALL_LIMIT=20 uv run --extra local-embeddings uvicorn api.main:app --app-dir src --reload
```

From `apps/web`, in another terminal:

```bash
npm run dev
```

Open the URL printed by Vite and refresh for a fresh conversation. Optionally
open DevTools > Network and filter for `chat`. Responses expose `message`,
`action`, and `citations`. Keep the browser console open to spot uncaught errors.

## 3. Main browser flow

| Done | Action | Expected result |
| --- | --- | --- |
| [ ] | Open the page | Greeting, input, and Send button appear. Resume is initially hidden. |
| [ ] | Try an empty or whitespace-only message | Submission is blocked; no chat request is sent. |
| [ ] | Ask **What Python experience does Saul have?** | Answer mentions Intuit/QuickBooks pipeline work. One `RAG retrieval` log appears. |
| [ ] | Expand a source under the answer | Its supporting passage documents the claim and relevant employer. |
| [ ] | Inspect the chat response in Network | Each citation has `chunkId`, `sourceId`, `documentVersion`, `title`, and `text`. The version matches `document_version` in the saved index; the chunk ID appears in this request's retrieval log. |
| [ ] | Ask **At which company?** | Resolves the Python follow-up to Intuit and shows supporting sources. |
| [ ] | Ask **What React experience does Saul have?** | Answers from Intuit component-library or MarTech work, with relevant source passages. |
| [ ] | Ask **Does Saul have Kubernetes experience?** | Normal "not documented" response, one retrieval with zero results, no Docker search, no Sources block, HTTP 200. |
| [ ] | Ask **Show me the full resume** | Resume displays and chat minimizes. Response action is `show_resume`, with no invented citations. |
| [ ] | Click **Continue conversation** | Earlier messages and their expandable sources remain. |
| [ ] | Ask **Tell me more about that React work** | Resolves the earlier topic and shows relevant sources. |
| [ ] | Try submitting during loading | Input/Send stay disabled until the current request finishes, then re-enable. |
| [ ] | Refresh | Conversation resets intentionally; profile loads and chat remains usable. |

Citation validation checks provenance: selected IDs must have been retrieved
in the current request, and source text/version come from the index. It does
not automatically prove that every generated claim follows from a passage;
reading the expanded evidence is part of this check. SDK and non-RAG responses
currently have empty citation lists.

## 4. Daily quota exhaustion without model calls

Stop FastAPI, then from `apps/api` run:

```bash
DAILY_MODEL_CALL_LIMIT=0 uv run --extra local-embeddings uvicorn api.main:app --app-dir src --reload
```

- [ ] Refresh and send any message.
- [ ] Confirm HTTP 429 with `DAILY_LIMIT_REACHED` and the resting screen.
- [ ] Confirm `http://localhost:8000/api/v1/profile` still works.
- [ ] Stop this process and restart with the normal command from step 2.
- [ ] Click **Back to conversation**, send a new message, and confirm recovery.

Restarting resets the local in-memory counter, not Google's quota. Command-line
overrides do not change `.env`.

## 5. Unavailable response and recovery without model calls

Stop FastAPI, then run:

```bash
AI_ENABLED=false uv run --extra local-embeddings uvicorn api.main:app --app-dir src --reload
```

- [ ] Refresh and send a message.
- [ ] Confirm HTTP 503 with `AI_UNAVAILABLE` and the resting screen.
- [ ] Confirm the profile endpoint still works.
- [ ] Stop this process and restart with the normal command from step 2.
- [ ] Return to chat and confirm another question succeeds.

Automated tests separately simulate actual provider failures. This switch checks
the same public 503 response contract in the browser without deliberately failing
a real Google request.

## 6. Connection failure

- [ ] Leave Vite running and stop FastAPI.
- [ ] Send a message. A visible request error should appear in chat, rather than
      a blank page or unhandled exception.
- [ ] Confirm loading ends and submission becomes available again.
- [ ] Restart the regular backend and send a new message; it should succeed.

## 7. Finish

- [ ] Restore your usual backend startup command and daily limit.
- [ ] Record any failing step, exact question, HTTP status, and backend log.
- [ ] Commit the working functionality checkpoint before visual changes.

Exact answer wording can vary. Assess supported facts, evidence, API responses,
UI transitions, and recovery rather than matching sentences word for word.
