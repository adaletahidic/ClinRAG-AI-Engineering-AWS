# Copilot instructions for ClinRAG-AI-Engineering-AWS

## Repository overview

This is a Python clinical AI decision-support prototype. The trained model is authoritative for the prediction; the LLM is only allowed to explain that prediction using retrieved evidence. Do not treat a model output as a medical diagnosis or add treatment recommendations.

The primary workflow is `app.graph.clinrag_graph.ClinRAGGraph`:

1. Call the prediction MCP client.
2. Retrieve supporting evidence with `KnowledgeAgent`.
3. Generate an explanation with `ExplanationAgent`.
4. Evaluate prediction validity, evidence grounding, safety, and relevance.
5. End successfully only when all checks pass; otherwise produce a safe-failure response.

The repository also contains `app.orchestrator.ClinRAGOrchestrator`, a synchronous orchestration path with similar stages. Preserve both APIs when changing shared domain behavior unless the task explicitly removes one.

## Build, test, and lint

There is no build system or configured lint command in the repository. Dependencies are pinned in `requirements.txt`.

PowerShell setup from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the configured test suite:

```powershell
python -m pytest
```

`pytest.ini` restricts collection to `tests/`, adds the repository root to `PYTHONPATH`, and uses strict asyncio mode. Run one file or one test with pytest selectors, for example:

```powershell
python -m pytest tests/test_retriever.py
python -m pytest tests/test_clinrag_graph.py::test_clinrag_graph_runs_end_to_end
```

The root-level `test_groq.py` and `test_mistral.py` are provider smoke scripts and are not part of the configured `tests/` suite. Provider-backed execution requires the corresponding API key in the local, ignored `.env`; unit tests should inject `tests.fake_llm.FakeLLM` instead of making network calls.

## Architecture and important boundaries

- `app/domain/schemas.py` defines the Pydantic contracts used at boundaries: patient features, authoritative prediction results, evidence chunks, agent responses, and evaluation results.
- `app/domain/predictor.py` loads `models/bilstm_model.keras`, `models/scaler.joblib`, and `models/metadata.json`. Metadata supplies the required feature columns, threshold, and fallback medians. Input features are reordered, coerced to numeric values, median-filled when configured, scaled, reshaped to one timestep, and passed to the BiLSTM.
- `app/mcp/prediction_server.py` lazily creates the predictor and exposes prediction and model-metadata tools over stdio. `app/mcp/prediction_client.py` starts that server with `python -m app.mcp.prediction_server`; keep the client free of model logic and preserve structured-content/JSON result extraction.
- `app/knowledge/document_loader.py` loads `.pdf`, `.txt`, and `.md` files, normalizes text, and creates overlapping character-window chunks. `app/knowledge/retriever.py` indexes chunks with local TF-IDF and cosine similarity, returning only positive-scoring `EvidenceChunk` results. The default knowledge source is `app/knowledge/documents/`.
- `app/agents/knowledge_agent.py` adapts retriever results to the Pydantic evidence contract. `app/agents/explanation_agent.py` formats the immutable prediction and retrieved evidence into an LLM prompt; its `grounded` flag is based on supplied evidence, not on an LLM claim.
- `app/llm/factory.py` selects Groq or Mistral from `LLM_PROVIDER` (default `groq`). Provider configuration is read in `app/config.py` from `.env`/environment variables: `GROQ_API_KEY`, `GROQ_MODEL`, `MISTRAL_API_KEY`, and `MISTRAL_MODEL`.
- `app/evaluation/evaluator.py` is the workflow gate. It validates prediction fields and prediction integrity, requires evidence when requested, checks positive retrieval relevance, rejects empty responses and known unsafe diagnostic phrases, and detects explicit unsupported evidence attributions. A failed check must not be turned into a successful response.
- `app/evaluation/test_cases.py` defines repeatable workflow scenarios, including normal, missing-feature, and retrieval-failure expectations. `app/evaluation/runner.py` executes those scenarios against an injected async workflow and aggregates results with `app/evaluation/metrics.py`. Supply valid model features to the runner for scenarios that are expected to reach prediction.
- `app/observability/events.py` emits structured JSON workflow events through the `clinrag.workflow` logger. `ClinRAGGraph.run()` assigns a request ID, records UTC timestamps and per-step latency in `trace`, and returns total `workflow_latency_ms`; keep these fields stable for later CloudWatch ingestion.
- `app/api.py` is the local HTTP boundary. Use `create_app(graph=...)` for dependency injection in tests; `/workflow` is canonical, while `/predict` and `/explain` expose the same workflow contract for client compatibility. Propagate `X-Request-ID` when supplied and keep controlled workflow safe failures distinct from unexpected HTTP 500 errors.
- Local container execution uses `Dockerfile` and `docker-compose.yml`: build with `docker compose build`, start with `docker compose up`, and reach the API at `http://localhost:8000`. The image copies `app/` and the model artifacts, and the MCP prediction client starts `app.mcp.prediction_server` inside the same container over stdio.
- `app/llm/bedrock_provider.py`, `app/knowledge/s3_source.py`, `app/knowledge/interfaces.py`, and `app/config_secrets.py` are AWS-ready adapters. They use lazy `boto3` imports; install `requirements-aws.txt` only for AWS-backed execution, not for the local Groq container.

## Codebase-specific conventions

- Keep prediction authority separate from explanation: do not let prompts, LLM output, or retrieval code create, reinterpret, or overwrite prediction, probability, risk group, model name, or model version.
- Use the existing Pydantic models at module boundaries and validate serialized state when it re-enters the graph. Keep evidence traceability (`source`, optional `page`, text, and relevance) intact.
- Prefer dependency injection used by `ClinRAGGraph` and `ExplanationAgent` so tests can provide fake prediction clients, knowledge agents, evaluators, and LLMs. Do not require live Groq/Mistral calls in unit tests.
- Preserve the LangGraph state keys and routing semantics in `ClinRAGGraph`: prediction errors route directly to `safe_failure`; evaluation failures or any failed safety/grounding/relevance condition also route to `safe_failure`.
- MCP is an inter-process boundary, not just a helper function. The prediction server must remain runnable as the `app.mcp.prediction_server` module and retain stdio transport compatibility.
- Resolve repository assets through the existing module-relative paths rather than relying on the caller’s current directory. Model artifacts and the bundled knowledge document are required runtime assets.
- Keep validation behavior explicit: empty retrieval queries, invalid chunk parameters, unsupported document extensions, missing model artifacts/features, and missing provider keys are expected to raise clear errors.
- Keep secrets in the ignored `.env`; never add API keys or other credentials to tracked files.
