from __future__ import annotations

import os


def load_secret_environment(
    secret_name: str | None = None,
    client=None,
) -> dict[str, str]:
    """Load JSON key/value configuration from Secrets Manager when configured."""
    name = secret_name or os.getenv("AWS_SECRET_NAME")
    if not name:
        return {}

    if client is None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required to load AWS secrets."
            ) from exc
        client = boto3.client("secretsmanager")

    import json

    response = client.get_secret_value(SecretId=name)
    secret_string = response.get("SecretString")
    if not secret_string:
        raise ValueError("AWS secret does not contain SecretString.")
    values = json.loads(secret_string)
    if not isinstance(values, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in values.items()
    ):
        raise ValueError("AWS secret must contain a JSON string dictionary.")
    return values
