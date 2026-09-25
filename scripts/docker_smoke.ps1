$ErrorActionPreference = "Stop"

docker compose up -d
try {
    $health = $null
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        try {
            $health = Invoke-WebRequest -UseBasicParsing `
                -Uri "http://localhost:8000/health" `
                -TimeoutSec 5
            break
        } catch {
            Start-Sleep -Seconds 2
        }
    }

    if ($null -eq $health) {
        throw "Health endpoint did not become ready within 60 seconds."
    }

    if ($health.StatusCode -ne 200) {
        throw "Health check returned HTTP $($health.StatusCode)."
    }

    $metadata = Get-Content ".\models\metadata.json" -Raw | ConvertFrom-Json
    $features = @{}
    foreach ($property in $metadata.feature_medians.psobject.Properties) {
        $features[$property.Name] = [double]$property.Value
    }

    $body = @{
        question = "Explain the model prediction using the available clinical evidence."
        features = $features
    } | ConvertTo-Json -Depth 4

    $workflow = Invoke-WebRequest -UseBasicParsing `
        -Method Post `
        -ContentType "application/json" `
        -Body $body `
        -Uri "http://localhost:8000/workflow"

    if ($workflow.StatusCode -ne 200) {
        throw "Workflow smoke test returned HTTP $($workflow.StatusCode)."
    }
    $workflowResult = $workflow.Content | ConvertFrom-Json
    if (-not $workflowResult.prediction) {
        throw "Workflow smoke test did not return a prediction."
    }
}
finally {
    docker compose logs --tail 100 clinrag-api
    docker compose down
}
