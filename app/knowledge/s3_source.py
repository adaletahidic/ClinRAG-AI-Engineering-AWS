from __future__ import annotations

import tempfile
from pathlib import Path

from app.knowledge.document_loader import DocumentChunk, DocumentLoader


class S3KnowledgeSource:
    """Loads supported knowledge files from S3 into the existing loader contract."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        client=None,
        loader: DocumentLoader | None = None,
    ):
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 is required for the S3 knowledge source."
                ) from exc
            client = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix
        self.client = client
        self.loader = loader or DocumentLoader()

    def load_documents(self) -> list[DocumentChunk]:
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.prefix,
        )
        chunks: list[DocumentChunk] = []
        for item in response.get("Contents", []):
            key = item["Key"]
            if Path(key).suffix.lower() not in {".pdf", ".txt", ".md"}:
                continue
            body = self.client.get_object(
                Bucket=self.bucket,
                Key=key,
            )["Body"].read()
            if Path(key).suffix.lower() == ".pdf":
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False,
                ) as temporary:
                    temporary.write(body)
                    temporary_path = Path(temporary.name)
                try:
                    chunks.extend(self.loader.load(temporary_path))
                finally:
                    temporary_path.unlink(missing_ok=True)
                continue
            text = body.decode("utf-8")
            chunks.extend(
                self.loader._load_text_from_content(
                    text,
                    source=f"s3://{self.bucket}/{key}",
                )
            )
        return chunks
