from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parents[2]

import json
def _extract_tool_result(result: Any) -> dict[str, Any]:
    """
    Extract a dictionary result from an MCP CallToolResult.

    Prefer structured_content when available, otherwise
    fall back to JSON returned through text content.
    """
    if result.structured_content:
        return result.structured_content

    if result.content:
        for item in result.content:
            if hasattr(item, "text") and item.text:
                try:
                    value = json.loads(item.text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "MCP tool returned non-JSON text content."
                    ) from exc

                if isinstance(value, dict):
                    return value

    raise RuntimeError(
        "MCP tool returned no usable structured or text content."
    )

class PredictionMCPClient:
    """
    MCP client for the ClinRAG prediction server.

    The client starts the prediction MCP server as a
    Python module and communicates with it through stdio.

    The client contains no prediction/model logic.
    """

    def __init__(
        self,
        project_root: str | Path = PROJECT_ROOT,
    ):
        self.project_root = Path(project_root)

    def _server_parameters(
        self,
    ) -> StdioServerParameters:

        return StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "app.mcp.prediction_server",
            ],
        )

    async def list_tools(
        self,
    ) -> list[str]:

        server = self._server_parameters()

        async with Client(server) as client:

            result = await client.list_tools()

            return [
                tool.name
                for tool in result.tools
            ]

    async def predict_patient(
            self,
            features: dict[str, float],
    ) -> dict[str, Any]:
        server = self._server_parameters()

        async with Client(server) as client:
            result = await client.call_tool(
                "predict_patient",
                {"features": features},
            )

            if result.is_error:
                raise RuntimeError(
                    "Prediction MCP tool returned an error."
                )

            return _extract_tool_result(result)

    async def get_model_metadata(
            self,
    ) -> dict[str, Any]:
        server = self._server_parameters()

        async with Client(server) as client:
            result = await client.call_tool(
                "get_prediction_model_metadata",
                {},
            )

            if result.is_error:
                raise RuntimeError(
                    "Metadata MCP tool returned an error."
                )

            return _extract_tool_result(result)


def predict_patient(
    features: dict[str, float],
) -> dict[str, Any]:

    return asyncio.run(
        PredictionMCPClient().predict_patient(
            features
        )
    )


def get_model_metadata() -> dict[str, Any]:

    return asyncio.run(
        PredictionMCPClient().get_model_metadata()
    )