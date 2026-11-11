"""Small shared helpers: logging, timing, run registry, image IO."""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import save_json
from .paths import artifact_dir


def setup_logging(name: str = "dm_ahr_vit", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[dict]:
    start = time.time()
    info: dict = {"label": label, "started_at": utc_now()}
    try:
        yield info
    finally:
        info["seconds"] = round(time.time() - start, 3)
        info["finished_at"] = utc_now()
        if logger is not None:
            logger.info("%s finished in %.1fs", label, info["seconds"])


def run_registry_path() -> Path:
    return artifact_dir("logs") / "run_registry.jsonl"


def record_run(
    job: str,
    *,
    gpu: str | None,
    seconds: float,
    status: str = "ok",
    **extra,
) -> None:
    """Append one JSON line per job. Never contains data or secrets."""
    entry = {
        "job": job,
        "gpu": gpu,
        "seconds": round(float(seconds), 2),
        "status": status,
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "timestamp": utc_now(),
        "pid": os.getpid(),
        **extra,
    }
    path = run_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def write_json(path: str | Path, obj) -> Path:
    return save_json(obj, path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())

