"""Structured logging.

Fixed at INFO in JSON. Debug logging is a code change, not a production toggle,
and prompts are user content: there is no good default for logging them other
than "never", so there is no switch to get it wrong.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    # urllib3 logs every connection at DEBUG and every retry at WARNING; the
    # latter is noise for a worker that disables retries on purpose.
    logging.getLogger("urllib3").setLevel(logging.ERROR)
