"""
logger.py — Lightweight structured logger for the stock streaming system.
Supports console output, file rotation, and JSON-formatted event records
without external dependencies.
"""

import os
import sys
import json
import time
import queue
import threading
import traceback
from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, Optional


class Level(IntEnum):
    DEBUG   = 10
    INFO    = 20
    WARNING = 30
    ERROR   = 40
    CRITICAL = 50

    @classmethod
    def from_str(cls, s: str) -> "Level":
        return cls[s.upper()]


LEVEL_LABELS = {
    Level.DEBUG:    "DEBUG",
    Level.INFO:     "INFO ",
    Level.WARNING:  "WARN ",
    Level.ERROR:    "ERROR",
    Level.CRITICAL: "CRIT ",
}


class LogRecord:
    """Immutable log event."""
    __slots__ = ("ts", "level", "logger_name", "message", "extra")

    def __init__(self, level: Level, logger_name: str,
                 message: str, extra: Optional[Dict] = None):
        self.ts          = time.time()
        self.level       = level
        self.logger_name = logger_name
        self.message     = message
        self.extra       = extra or {}

    def to_json(self) -> str:
        d = {
            "ts":      datetime.fromtimestamp(self.ts).isoformat(),
            "level":   LEVEL_LABELS[self.level].strip(),
            "logger":  self.logger_name,
            "message": self.message,
        }
        if self.extra:
            d["extra"] = self.extra
        return json.dumps(d)

    def to_text(self) -> str:
        ts_str = datetime.fromtimestamp(self.ts).strftime("%H:%M:%S.%f")[:-3]
        label  = LEVEL_LABELS[self.level]
        extra  = f" | {self.extra}" if self.extra else ""
        return f"[{ts_str}] {label} [{self.logger_name}] {self.message}{extra}"


class FileRotatingHandler:
    """
    Rotates the log file once it exceeds max_bytes.
    Keeps up to backup_count old files (suffixed .1, .2, …).
    """

    def __init__(self, filepath: str, max_bytes: int = 5 * 1024 * 1024,
                 backup_count: int = 3):
        self._path         = filepath
        self._max_bytes    = max_bytes
        self._backup_count = backup_count
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self._file = open(filepath, "a", encoding="utf-8")

    def emit(self, record: LogRecord) -> None:
        self._file.write(record.to_json() + "\n")
        self._file.flush()
        if os.path.getsize(self._path) >= self._max_bytes:
            self._rotate()

    def _rotate(self) -> None:
        self._file.close()
        for i in range(self._backup_count - 1, 0, -1):
            src = f"{self._path}.{i}"
            dst = f"{self._path}.{i + 1}"
            if os.path.exists(src):
                os.rename(src, dst)
        os.rename(self._path, f"{self._path}.1")
        self._file = open(self._path, "a", encoding="utf-8")

    def close(self) -> None:
        if self._file:
            self._file.close()


class AsyncLogDispatcher:
    """Background thread that drains a queue and writes records to handlers."""

    def __init__(self):
        self._queue   = queue.Queue(maxsize=10_000)
        self._handlers = []
        self._thread  = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def add_handler(self, handler) -> None:
        self._handlers.append(handler)

    def submit(self, record: LogRecord) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass  # Drop under extreme back-pressure

    def _drain(self) -> None:
        while True:
            try:
                record = self._queue.get(timeout=0.1)
                for h in self._handlers:
                    try:
                        h.emit(record)
                    except Exception:
                        pass
            except queue.Empty:
                pass

    def flush(self, timeout: float = 2.0) -> None:
        deadline = time.time() + timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.05)


# Shared async dispatcher (singleton)
_DISPATCHER = AsyncLogDispatcher()

# Console handler
class _ConsoleHandler:
    def __init__(self, min_level: Level = Level.INFO):
        self._min = min_level
    def emit(self, record: LogRecord) -> None:
        if record.level >= self._min:
            print(record.to_text(), file=sys.stdout)

_DISPATCHER.add_handler(_ConsoleHandler(min_level=Level.INFO))


class Logger:
    """
    Per-module logger.  Obtain via get_logger(__name__).
    """

    def __init__(self, name: str, min_level: Level = Level.DEBUG):
        self._name      = name
        self._min_level = min_level

    def _log(self, level: Level, msg: str, **extra) -> None:
        if level < self._min_level:
            return
        _DISPATCHER.submit(LogRecord(level, self._name, msg, extra or None))

    def debug(self, msg: str, **kw)    -> None: self._log(Level.DEBUG,    msg, **kw)
    def info(self, msg: str, **kw)     -> None: self._log(Level.INFO,     msg, **kw)
    def warning(self, msg: str, **kw)  -> None: self._log(Level.WARNING,  msg, **kw)
    def error(self, msg: str, **kw)    -> None: self._log(Level.ERROR,    msg, **kw)
    def critical(self, msg: str, **kw) -> None: self._log(Level.CRITICAL, msg, **kw)

    def exception(self, msg: str, exc: Exception) -> None:
        tb = traceback.format_exc()
        self._log(Level.ERROR, f"{msg} — {exc}\n{tb}")


def get_logger(name: str, min_level: Level = Level.DEBUG) -> Logger:
    """Factory — use this everywhere in the codebase."""
    return Logger(name, min_level)


def add_file_handler(filepath: str, max_bytes: int = 5_242_880) -> None:
    """Attach a rotating file handler to the global dispatcher."""
    _DISPATCHER.add_handler(FileRotatingHandler(filepath, max_bytes))


def flush() -> None:
    """Flush all pending log records (call before process exit)."""
    _DISPATCHER.flush()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log = get_logger("smoke_test")
    log.debug("Debug message — usually suppressed in console")
    log.info("Stream processor started", tick_rate_ms=100)
    log.warning("Back-pressure detected — queue near capacity", fill_pct=91.3)
    log.error("Inference timeout exceeded", latency_ms=67.2, sla_ms=50.0)
    flush()
    time.sleep(0.1)   # Let async dispatcher drain
    print("Logger smoke test complete.")
