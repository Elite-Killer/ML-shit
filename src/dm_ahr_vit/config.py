"""Configuration loading and hashing.

Configs are JSON so the exact bytes can be hashed and committed. Every phase
records the hash of the config it used so results are traceable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT
from .paths import artifact_root

CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Expected C0 class counts from the HAM10000 data descriptor. These are only
# used to compare against recomputed values, never as data.
EXPECTED_C0_COUNTS = {
    "akiec": 327,
    "bcc": 514,
    "bkl": 1099,
    "df": 115,
    "mel": 1113,
    "nv": 6705,
    "vasc": 142,
}
EXPECTED_C0_TOTAL = 10015

DEFAULT_PROTOCOL = "configs/protocol.json"


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def config_hash_file(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(p)
    return p


def frozen_path(name: str) -> Path:
    return artifact_root() / "logs" / "frozen" / f"{name}.json"


def load_frozen(name: str, default_config: str) -> dict[str, Any]:
    """Prefer a hash-frozen config produced at runtime over the repo default."""
    p = frozen_path(name)
    if p.exists():
        return json.loads(p.read_text())
    return load_config(default_config)


def save_frozen(name: str, cfg: dict[str, Any]) -> Path:
    return save_json(cfg, frozen_path(name))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

