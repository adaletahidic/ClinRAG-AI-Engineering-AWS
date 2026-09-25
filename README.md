# ClinRAG local API

## Start the local container

```powershell
docker compose build
docker compose up -d
docker compose logs -f
```

The API is available at `http://localhost:8000` and Swagger UI at
`http://localhost:8000/docs`.

## Health check

```powershell
Invoke-WebRequest http://localhost:8000/health
```

## Workflow request

Use `POST /workflow` in Swagger or with a client. The request requires a
question and the model feature dictionary:

```json
{
  "question": "Explain the model prediction using the available clinical evidence.",
  "features": {
    "radius_mean": 13.37,
    "texture_mean": 18.84
  }
}
```

The predictor validates the complete feature set and fills configured missing
values with model medians when a supplied value is non-numeric. For a real
inference request, send all feature columns from `models/metadata.json`.
The Docker smoke script builds a valid request from those configured medians.

The response includes `request_id`, `prediction`, `answer`, `evidence`,
evaluation flags, `trace`, and `workflow_latency_ms`.

## Request IDs and configuration

Send `X-Request-ID` to correlate API responses with workflow events. If it is
omitted, the API generates one.

The local `.env` supports:

```text
ALLOWED_ORIGINS=http://localhost:3000
LOG_LEVEL=INFO
WORKFLOW_TIMEOUT_SECONDS=120
MAX_REQUEST_BODY_BYTES=1048576
```

`/predict` and `/explain` are compatibility aliases for `/workflow`.

## Failure behavior

- Invalid request bodies return HTTP `422`.
- Oversized request bodies return HTTP `413`.
- Unexpected API execution failures and workflow timeouts return HTTP `500`
  with a request ID and controlled error message.
- Prediction, retrieval, explanation, or evaluation failures are returned as a
  controlled workflow safe failure with `safe_failure: true`; these are not
  converted into successful clinical responses.

## Docker smoke test

```powershell
docker compose up -d
Invoke-WebRequest http://localhost:8000/health
docker compose logs --tail 100 clinrag-api
docker compose down
```

Or run the Windows smoke script:

```powershell
.\scripts\docker_smoke.ps1
```
