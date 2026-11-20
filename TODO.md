# Project TODO

Build incrementally. Keep wording refinements and retrieval tuning for a later
pass unless they block a core flow. AWS provisioning and paid services require
explicit approval.

## Current priority: finish the functionality checkpoint

- [ ] Run the [browser regression checklist](docs/regression-checklist.md).
- [ ] Verify Python and React answers display passages that support their claims.
- [ ] Verify follow-ups keep context and return appropriate citations.
- [ ] Verify unsupported technologies return a normal "not documented" answer.
- [ ] Verify resume rendering, chat minimization, and reopening preserve messages and sources.
- [ ] Verify quota exhaustion, unavailable mode, connection failure, and recovery.
- [ ] Commit the working functionality checkpoint before visual changes.

## Next: visual design

- [ ] Choose the visual direction: branding, colors, typography, and tone.
- [ ] Design the full-page chat landing state and suggested questions.
- [ ] Style user/assistant messages, input, loading, and error states.
- [ ] Design source citations and expandable evidence.
- [ ] Design the full resume and minimized conversation panel.
- [ ] Design the resting screen with basic professional information.
- [ ] Support mobile and desktop layouts.
- [ ] Check keyboard navigation, focus behavior, labels, contrast, and reduced motion.
- [ ] Repeat regression checks after the visual changes.

## Retrieval and answer quality

- [ ] Create a representative recruiter-question evaluation dataset with expected sources.
- [ ] Evaluate retrieval relevance, unsupported claims, follow-up resolution, latency, and usage.
- [ ] Compare RAG against the full-profile baseline before choosing production defaults.
- [ ] Review keyword/semantic weights and the unsupported-technology heuristic using evaluation results.
- [ ] Pin the embedding model revision and record chunking/index-generation versions.
- [ ] Make index publication atomic so interrupted rebuilds cannot replace a working artifact with a partial file.
- [ ] Add regression tests for automatic content-hash version generation.
- [ ] Add education and languages to knowledge chunks if they are needed for RAG answers.
- [ ] Replace remaining temporary search prints with deliberate structured diagnostics.
- [ ] Revisit wording and answer formatting in one focused pass.

## Portfolio content and additional features

- [ ] Review all public profile content and contact information before release.
- [ ] Add project descriptions and structured project data.
- [ ] Add project cards and experience views when useful to the conversation.
- [ ] Add a downloadable resume PDF if desired.
- [ ] Add PDF/document ingestion when structured profile data is no longer sufficient.
- [ ] Add the data-engineering CV variant after the full-stack version is stable.
- [ ] Decide whether persistent conversation sessions are needed; current history clears on refresh.
- [ ] Verify another portfolio owner can configure the application without editing application logic.

## Before public deployment

- [ ] Add per-visitor rate limiting and API throttling.
- [ ] Replace local quota accounting with shared atomic production storage.
- [ ] Add settings-validation tests and startup/configuration checks.
- [ ] Review input limits, prompt injection, citation handling, and safe errors.
- [ ] Configure CORS, production secrets, and private document access.
- [ ] Add CI checks for backend tests, frontend build, and lint.
- [ ] Add automated browser regression tests for the main flows.
- [ ] Choose an embedding deployment approach compatible with the hosting budget and startup latency.
- [ ] Create and review AWS CDK infrastructure before provisioning.
- [ ] Deploy regional resources in `us-east-2` with minimal IAM permissions.
- [ ] Add monitoring, useful structured logs, usage visibility, and budget alerts.
- [ ] Verify free-tier model access and cost controls; do not treat application quotas as a billing guarantee.
- [ ] Document deployment, rollback, source/index updates, and operational recovery.

## Already working

- [x] React frontend and FastAPI backend run locally.
- [x] Reusable full-stack profile data and resume rendering.
- [x] Direct Gemini SDK integration.
- [x] LangChain integration with portfolio tools.
- [x] Bounded conversation history.
- [x] Local embeddings, versioned retrieval artifacts, and hybrid search.
- [x] Single-retrieval RAG workflow without model-controlled search loops.
- [x] Daily local model-call quota and AI kill switch.
- [x] Resting screen and resume/chat transitions.
- [x] Backend-validated citation references and expandable source passages.
- [x] Backend/frontend folder organization and updated local commands.
- [x] Automated backend coverage and a documented manual regression checklist.
