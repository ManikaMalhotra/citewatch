"""
Embedding wrapper for Citewatch.

Supports two backends:
1. Ollama (nomic-embed-text) — if Ollama is running locally
2. Chroma's DefaultEmbeddingFunction (all-MiniLM-L6-v2 via onnxruntime) — zero-setup fallback

The fallback is automatic: if Ollama isn't reachable, we use the built-in.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity — always returns native Python float."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _check_ollama_available(url: str) -> bool:
    """Quick check if Ollama server is reachable."""
    try:
        import httpx
        resp = httpx.get(f"{url}/api/tags", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


class EmbeddingManager:
    """
    Manages embeddings with automatic backend selection.

    Priority: Ollama (if available) → Chroma default (always works).
    """

    def __init__(self, ollama_url: str = DEFAULT_OLLAMA_URL, model_name: str = DEFAULT_MODEL):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self._ef = None
        self._backend = "unknown"

    @property
    def chroma_ef(self):
        """Get the Chroma-compatible embedding function (lazy init with fallback)."""
        if self._ef is None:
            self._ef, self._backend = self._init_embedding_function()
        return self._ef

    def _init_embedding_function(self):
        """Try Ollama first, fall back to Chroma's built-in."""
        # Try Ollama
        if _check_ollama_available(self.ollama_url):
            try:
                from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
                ef = OllamaEmbeddingFunction(url=self.ollama_url, model_name=self.model_name)
                # Quick test
                ef(["test"])
                logger.info("embedding_init", backend="ollama", model=self.model_name)
                return ef, "ollama"
            except Exception as e:
                logger.warning("ollama_failed_fallback", error=str(e)[:100])

        # Fallback: Chroma's default (all-MiniLM-L6-v2, onnxruntime — already installed)
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        ef = DefaultEmbeddingFunction()
        logger.info("embedding_init", backend="default (all-MiniLM-L6-v2)", msg="Ollama not available, using built-in embeddings")
        return ef, "default"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.chroma_ef(texts)

    def embed_single(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        return _cosine_sim(a, b)

    def health_check(self) -> bool:
        vec = self.embed_single("health check")
        if len(vec) > 0:
            logger.info("embedding_healthy", backend=self._backend, dim=len(vec))
            return True
        return False


_manager: EmbeddingManager | None = None


def get_embedding_manager(ollama_url: str = DEFAULT_OLLAMA_URL, model_name: str = DEFAULT_MODEL) -> EmbeddingManager:
    global _manager
    if _manager is None:
        _manager = EmbeddingManager(ollama_url=ollama_url, model_name=model_name)
    return _manager
