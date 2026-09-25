"""Text embedder for EvoCode."""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Protocol for converting text to fixed-dimensional vectors."""

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


class FakeEmbedder:
    """Deterministic fake embedder for testing.

    Produces a hash-based vector so different inputs get different outputs
    but the same input always produces the same output.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31)
        rng = np.random.default_rng(seed)
        vec = rng.random(self._dimension, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.array([self.embed(t) for t in texts])


class SentenceTransformerEmbedder:
    """Real embedder using sentence-transformers.

    Lazy-loads the model on first call to embed().
    Uses HF_HOME env var for cache location.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension  # type: ignore[return-value]

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        if self._model is None:
            self._load_model()
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            if self._dimension is None:
                self._load_model()
            return np.zeros((0, self._dimension), dtype=np.float32)
        if self._model is None:
            self._load_model()
        vecs = self._model.encode(texts, convert_to_numpy=True)
        return vecs.astype(np.float32)
