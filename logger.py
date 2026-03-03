"""
Centralised logging configuration for interview-levelup-agent.

Usage:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("something happened")

Log level is controlled by the LOG_LEVEL environment variable (default INFO).
All output goes to stdout so Docker / container runtimes pick it up naturally.

Design note: we use a dedicated "agent" root logger with propagate=False so
that uvicorn's logging reconfiguration never touches our handlers.
Third-party library loggers are quieted to WARNING.
"""

import logging
import logging.config
import os
import sys

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

_FMT = "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"

_ROOT = "agent"

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "openai",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langgraph",
)


def reconfigure() -> None:
    """Set up (or restore) the agent logger.

    Safe to call multiple times. Called once at import time and again inside
    the FastAPI startup event so that uvicorn's dictConfig
    (disable_existing_loggers=True) cannot permanently silence our loggers.
    """
    app_root = logging.getLogger(_ROOT)
    app_root.disabled = False
    app_root.setLevel(LOG_LEVEL)
    app_root.propagate = False

    app_root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    app_root.addHandler(handler)

    # Re-enable any child loggers that were disabled.
    for name, logger in logging.Logger.manager.loggerDict.items():
        if name.startswith(_ROOT + ".") and isinstance(logger, logging.Logger):
            logger.disabled = False

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# Run once at import time so logs are available during module load.
reconfigure()


def get_logger(name: str) -> logging.Logger:
    """Return an app logger under the 'agent' hierarchy.

    Pass __name__ from the calling module; the 'agent.' prefix is added
    automatically so all app loggers share the same handler/level.
    """
    leaf = name.rsplit(".", 1)[-1]
    return logging.getLogger(f"{_ROOT}.{leaf}")
