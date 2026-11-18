# AI Portfolio Agent

A reusable portfolio with a conversational AI assistant, built incrementally as a learning project.

## Structure

```text
apps/
  web/              React and TypeScript frontend
  api/              Python and FastAPI backend
profiles/
  default/          Portfolio owner configuration and content
```

## Current milestone

Mock and Gemini chat, bounded conversation context, a data-driven resume,
local daily call limits, and an assistant resting screen are implemented.

## Run locally

Backend, from `apps/api`:

```bash
uv sync
uv run uvicorn main:app --app-dir src --reload
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
AI_ENABLED=false uv run uvicorn main:app --app-dir src --reload
```

Send a chat message to trigger the resting screen. Restart with `AI_ENABLED=true`
to restore mock chat, then return to the conversation. This switch does not disable
the profile endpoint. Gemini mode is available when explicitly configured; mock mode is the default.

Profile content lives in `profiles/default/profile.json`. Restart the backend and
refresh the browser after changing it: the backend loads it at startup and React
fetches it on mount.

Frontend checks, from `apps/web`: `npm run build` and `npm run lint`.

Backend tests, from `apps/api`:

```bash
uv run pytest
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

LangChain tools, RAG, and AWS deployment are future milestones.

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
