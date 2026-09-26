"""Logging and lightweight timing helpers for data annotation."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from functools import wraps
import logging
import os
from pathlib import Path
import threading
from time import perf_counter


class Logger:
    """Provide one configured logger to the annotation workers."""

    _logger: logging.Logger | None = None

    @classmethod
    def get_logger(cls, name: str | None = None) -> logging.Logger:
        if cls._logger is None:
            cls._logger = cls.init_logger(name)
        return cls._logger

    @classmethod
    def init_logger(cls, name: str | None = None) -> logging.Logger:
        log = logging.getLogger(name or "data_annotation")
        level_name = os.getenv("CONTINUO_LOG_LEVEL", "INFO").upper()
        log.setLevel(getattr(logging, level_name, logging.INFO))
        log.propagate = False
        if log.handlers:
            return log

        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        log.addHandler(console)

        directory = Path(os.getenv("CONTINUO_LOG_DIR", "logs"))
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{log.name}-{stamp}-{os.getpid()}.log"
        file_handler = logging.FileHandler(directory / filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
        return log


def time_logger(func):
    """Log a callable's elapsed time, including calls that raise."""

    @wraps(func)
    def measured(*args, **kwargs):
        started = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
            Logger.get_logger().debug("%s completed in %.1f ms", func.__name__, elapsed_ms)

    return measured


_timings = threading.local()


def drain_timed_spans() -> list[tuple[str, float]]:
    """Return and clear timing records accumulated by the current thread."""
    records = getattr(_timings, "records", [])
    _timings.records = []
    return records


@contextmanager
def time_span(label: str, *, logger: logging.Logger | None = None):
    """Record the duration of a block in milliseconds."""
    started = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - started) * 1000
        if not hasattr(_timings, "records"):
            _timings.records = []
        _timings.records.append((label, elapsed_ms))
        (logger or Logger.get_logger()).debug("[span] %s: %.1f ms", label, elapsed_ms)
