"""Tiny content-addressed disk cache. Used to memoize retrievals and agent
intermediate outputs so re-runs on the same input don't burn API credits."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from src.utils.config import settings


class DiskCache:
    def __init__(self, namespace: str, root: Path | None = None):
        self.dir = (root or settings.cache_dir) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(payload: Any) -> str:
        s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(s).hexdigest()[:24]

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.pkl"

    def get(self, payload: Any) -> Any | None:
        p = self._path(self._key(payload))
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def set(self, payload: Any, value: Any) -> None:
        p = self._path(self._key(payload))
        with open(p, "wb") as f:
            pickle.dump(value, f)

    def clear(self) -> None:
        for f in self.dir.glob("*.pkl"):
            f.unlink()
