"""
ChromaDB cache layer for Citewatch.

Handles caching of LLM responses, knowledge base storage, and
query result deduplication using SHA256 content hashing.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import chromadb
import structlog

from citewatch.embeddings import EmbeddingManager, get_embedding_manager

logger = structlog.get_logger()


def make_cache_key(query: str, model: str, mode: str = "api", temperature: float = 0.0) -> str:
    """Generate a deterministic SHA256 cache key."""
    raw = f"{query}|{model}|{mode}|{temperature}"
    return hashlib.sha256(raw.encode()).hexdigest()


class CacheManager:
    """ChromaDB-backed cache for LLM responses and analysis results."""

    def __init__(self, db_path: str = "./.citewatch_cache", ttl_days: int = 30, embedding_mgr: EmbeddingManager | None = None):
        self.db_path = Path(db_path)
        self.ttl_days = ttl_days
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.db_path))
        self._emb = embedding_mgr or get_embedding_manager()
        self._collection = self._client.get_or_create_collection(
            name="geo_cache",
            embedding_function=self._emb.chroma_ef,
        )
        logger.info("cache_initialized", path=str(self.db_path), ttl_days=ttl_days)

    def get(self, cache_key: str) -> dict | None:
        """Retrieve a cached entry by key. Returns None on miss or expiry."""
        try:
            results = self._collection.get(ids=[cache_key], include=["documents", "metadatas"])
            if not results["documents"] or not results["documents"][0]:
                return None

            metadata = results["metadatas"][0] if results["metadatas"] else {}
            cached_at = metadata.get("cached_at", "")
            if cached_at:
                cached_dt = datetime.fromisoformat(cached_at)
                if datetime.utcnow() - cached_dt > timedelta(days=self.ttl_days):
                    logger.info("cache_expired", key=cache_key[:12])
                    self._collection.delete(ids=[cache_key])
                    return None

            doc = results["documents"][0]
            payload = metadata.get("payload_json", doc)
            logger.info("cache_hit", key=cache_key[:12])
            return json.loads(payload)
        except Exception:
            return None

    def put(self, cache_key: str, data: dict, query_text: str = "") -> None:
        """Store data in the cache."""
        payload = json.dumps(data, default=str)
        metadata = {
            "cached_at": datetime.utcnow().isoformat(),
            "query": query_text[:200],
            "payload_json": payload,
        }
        # Cache lookups are exact-id based, so the document text itself can stay short.
        # This avoids sending a large serialized LLM response through the embedding model.
        doc = query_text[:200] or cache_key[:64]
        try:
            self._collection.upsert(
                ids=[cache_key],
                documents=[doc],
                metadatas=[metadata],
            )
            logger.info("cache_stored", key=cache_key[:12])
        except Exception as e:
            logger.warning("cache_store_failed", key=cache_key[:12], error=str(e)[:160])

    def has(self, cache_key: str) -> bool:
        return self.get(cache_key) is not None

    def clear(self) -> None:
        """Clear all cached entries."""
        self._client.delete_collection("geo_cache")
        self._collection = self._client.get_or_create_collection(
            name="geo_cache", embedding_function=self._emb.chroma_ef,
        )
        logger.info("cache_cleared")

    @property
    def count(self) -> int:
        return self._collection.count()


class KnowledgeBase:
    """ChromaDB-backed knowledge base for the configured target site."""

    def __init__(self, db_path: str = "./.citewatch_kb", embedding_mgr: EmbeddingManager | None = None):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.db_path))
        self._emb = embedding_mgr or get_embedding_manager()
        self._collection = self._client.get_or_create_collection(
            name="site_kb",
            embedding_function=self._emb.chroma_ef,
        )
        logger.info("kb_initialized", path=str(self.db_path))

    def add_chunks(self, chunk_ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        """Add content chunks to the KB."""
        self._collection.upsert(ids=chunk_ids, documents=documents, metadatas=metadatas)
        logger.info("kb_chunks_added", count=len(chunk_ids))

    def query(self, query_text: str, n_results: int = 8, where: dict | None = None) -> dict:
        """Query the KB for relevant chunks."""
        kwargs: dict = {"query_texts": [query_text], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        logger.debug("kb_queried", n_results=len(results.get("documents", [[]])[0]))
        return results

    @property
    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection("site_kb")
        self._collection = self._client.get_or_create_collection(
            name="site_kb", embedding_function=self._emb.chroma_ef,
        )
