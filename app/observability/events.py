from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger("clinrag.workflow")


def emit_workflow_event(event: dict[str, Any]) -> None:
    """Emit a JSON event suitable for local logs and CloudWatch ingestion."""
    logger.info(json.dumps(event, default=str, sort_keys=True))
