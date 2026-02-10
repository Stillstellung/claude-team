"""
Logging configuration for Maniple.

Launchd captures stdout/stderr into files configured in the LaunchAgent plist.
To avoid unbounded growth, we log primarily to a rotating file under ~/.maniple/logs/
and keep stderr output at a higher severity by default.
"""

from __future__ import annotations

import gzip
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil


def configure_logging() -> Path:
    """
    Configure Maniple logging with on-disk rotation.

    Returns:
        Path to the primary log file.
    """
    from maniple.paths import resolve_data_dir

    log_dir = resolve_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "maniple.log"

    max_mb = _get_int_env("MANIPLE_LOG_MAX_SIZE_MB", default=10, min_value=1)
    backups = _get_int_env("MANIPLE_LOG_BACKUP_COUNT", default=5, min_value=1)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backups,
        encoding="utf-8",
    )

    def _namer(name: str) -> str:
        return f"{name}.gz"

    def _rotator(source: str, dest: str) -> None:
        with open(source, "rb") as src, gzip.open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
        try:
            os.remove(source)
        except FileNotFoundError:
            pass

    file_handler.namer = _namer
    file_handler.rotator = _rotator

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    # Allow per-logger overrides; default to a reasonable severity.
    level_name = os.getenv("MANIPLE_LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))

    # Replace existing handlers so we don't duplicate output.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.addHandler(file_handler)

    # Stderr handler is what launchd captures; keep it quieter by default.
    stderr_level_name = os.getenv("MANIPLE_STDERR_LOG_LEVEL", "WARNING").upper()
    stderr_level = getattr(logging, stderr_level_name, logging.WARNING)
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(stderr_level)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)

    return log_path


def _get_int_env(name: str, *, default: int, min_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    if parsed < min_value:
        return default
    return parsed

