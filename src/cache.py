"""
    query-results cache (bonus #4, second half): memoize
    (query, k, mode) -> ranked (chunk_id, score) results on disk, so a
    repeated query skips retrieval entirely on a later process.

    for semantic/hybrid modes this is the real cold-start win: a cache
    hit lets the caller skip loading the embedding model altogether
    (~4s), not just skip the retrieval math (~ms). callers are
    expected to check the cache *before* loading the model - see
    retriever.search_dataset's cache_dir handling.

    invalidation: every row is tagged with a fingerprint of the
    persisted index's files (mtime + size). a reindex changes the
    fingerprint, so stale rows simply stop matching; put() also sweeps
    them so the table doesn't grow unbounded across reindexes.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

CACHE_FILENAME = "query_cache.sqlite3"
# files whose mtime+size fingerprint the cache. embeddings.npy is
# included even for lexical-only setups where it may not exist yet -
# _fingerprint() just skips missing files.
FINGERPRINTED_FILES = ("index.pkl", "embeddings.npy")


def _fingerprint(processed_dir: Path) -> str:
    """
        hash the mtime+size of the persisted index files.

        args:
            processed_dir: dir holding index.pkl (+ embeddings.npy)

        return:
            hex digest; changes whenever the index is rebuilt
    """
    parts = []
    for name in FINGERPRINTED_FILES:
        path = processed_dir / name
        if path.is_file():
            stat = path.stat()
            parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _key(query: str, k: int, mode: str) -> str:
    """
        build the cache row key from (query, k, mode).

        args:
            query
            k
            mode

        return:
            hex digest identifying this exact request
    """
    raw = f"{mode}\x1f{k}\x1f{query}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _connect(processed_dir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(processed_dir / CACHE_FILENAME)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache ("
        "fingerprint TEXT NOT NULL, "
        "key TEXT NOT NULL, "
        "value TEXT NOT NULL, "
        "PRIMARY KEY (fingerprint, key))"
    )
    return conn


def get(
    processed_dir: Path, query: str, k: int, mode: str
) -> list[tuple[int, float]] | None:
    """
        look up a cached (query, k, mode) result.

        a missing or corrupt cache file is treated as a miss, never
        raises - caching must never crash the CLI, only speed it up.

        args:
            processed_dir: dir holding the index + cache file
            query
            k
            mode

        return:
            cached (chunk_id, score) list, or None on miss
    """
    try:
        fingerprint = _fingerprint(processed_dir)
        with _connect(processed_dir) as conn:
            row = conn.execute(
                "SELECT value FROM cache WHERE fingerprint = ? AND key = ?",
                (fingerprint, _key(query, k, mode)),
            ).fetchone()
        if row is None:
            return None
        return [(int(cid), float(score)) for cid, score in json.loads(row[0])]
    except (sqlite3.DatabaseError, OSError, ValueError):
        return None


def put(
    processed_dir: Path,
    query: str,
    k: int,
    mode: str,
    ranked: list[tuple[int, float]],
) -> None:
    """
        persist a (query, k, mode) -> ranked result.

        also sweeps rows from a stale fingerprint (a prior index
        build), so the cache doesn't grow unbounded across reindexes.
        a write failure is swallowed - a cache is an optimization, not
        a correctness requirement, so it must never crash the CLI.

        args:
            processed_dir: dir holding the index + cache file
            query
            k
            mode
            ranked: (chunk_id, score) pairs to store
    """
    try:
        fingerprint = _fingerprint(processed_dir)
        with _connect(processed_dir) as conn:
            conn.execute(
                "DELETE FROM cache WHERE fingerprint != ?", (fingerprint,)
            )
            conn.execute(
                "INSERT OR REPLACE INTO cache (fingerprint, key, value) "
                "VALUES (?, ?, ?)",
                (fingerprint, _key(query, k, mode), json.dumps(ranked)),
            )
    except (sqlite3.DatabaseError, OSError):
        pass
