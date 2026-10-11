"""Tiny content-addressed disk cache.

Used to memoize retrievals and agent intermediate outputs so re-runs on the same
input don't burn API credits.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from src.utils.config import settings


class DiskCache:
    """Pickle-file cache keyed by a hash of a JSON-serializable payload.

    Warning: values are stored with ``pickle``; only use a cache directory you trust.
    """

    def __init__(self, namespace: str, root: Path | None = None) -> None:
        """Create (if needed) ``<root>/<namespace>``; ``root`` defaults to ``settings.cache_dir``."""
        self.dir = (root or settings.cache_dir) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(payload: Any) -> str:
        """Return a stable 24-hex-char SHA-256 prefix of the payload's sorted-key JSON (``str`` for non-JSON types)."""
        s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(s).hexdigest()[:24]

    def _path(self, key: str) -> Path:
        """Return the ``.pkl`` file path for a key."""
        return self.dir / f"{key}.pkl"

    def get(self, payload: Any) -> Any | None:
        """Return the cached value for ``payload``, or None on a miss.

        Note: a stored value of None is indistinguishable from a miss.
        """
        p = self._path(self._key(payload))
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def set(self, payload: Any, value: Any) -> None:
        """Store ``value`` under ``payload``'s key, overwriting any existing entry."""
        p = self._path(self._key(payload))
        p.parent.mkdir(parents=True, exist_ok=True)  # survive the folder being deleted mid-session
        with open(p, "wb") as f:
            pickle.dump(value, f)

    def clear(self) -> None:
        """Delete every ``*.pkl`` entry in this namespace."""
        for f in self.dir.glob("*.pkl"):
            f.unlink()
