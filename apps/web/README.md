# Portfolio web app

Run from this directory:

```bash
npm install
npm run dev
```

Start FastAPI separately using the commands in the [root README](../../README.md).
Vite proxies `/api` requests to `http://localhost:8000` during development.

## Source layout

- `src/main.tsx`: React entry point.
- `src/app/App.tsx`: chat state, API calls, and page composition.
- `src/features/resume/ResumeView.tsx`: data-driven resume rendering.
- `src/types/`: shared chat and profile TypeScript contracts.
- `src/styles/`: application and global CSS.
- `src/assets/`: static assets.

## Checks

```bash
npm run build
npm run lint
```

API keys belong in the backend's `.env`, never frontend code or `VITE_` variables.
