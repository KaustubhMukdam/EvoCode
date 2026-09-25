"""FAISS vector index for EvoCode."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorIndex(Protocol):
    """Protocol for vector similarity search."""

    @property
    def size(self) -> int: ...

    def add(self, vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, top_k: int) -> list[int]: ...


class FaissVectorIndex:
    """Real vector index backed by FAISS IndexFlatIP."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._index = None
        self._count = 0

    def _load(self) -> None:
        import faiss

        self._index = faiss.IndexFlatIP(self._dimension)

    @property
    def size(self) -> int:
        return self._count

    def add(self, vectors: np.ndarray) -> None:
        if self._index is None:
            self._load()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        normalized = vectors / norms
        self._index.add(normalized.astype(np.float32))
        self._count += len(vectors)

    def search(self, query: np.ndarray, top_k: int) -> list[int]:
        if self._index is None or self._count == 0:
            return []
        q_norm = np.linalg.norm(query)
        q = (query / q_norm if q_norm > 0 else query).reshape(1, -1).astype(np.float32)
        k = min(top_k, self._count)
        _, indices = self._index.search(q, k)
        return [int(i) for i in indices[0] if i >= 0]


class FakeVectorIndex:
    """Deterministic fake vector index using brute-force cosine similarity."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._vectors: list[np.ndarray] = []

    @property
    def size(self) -> int:
        return len(self._vectors)

    def add(self, vectors: np.ndarray) -> None:
        for v in vectors:
            norm = np.linalg.norm(v)
            self._vectors.append(v / norm if norm > 0 else v)

    def search(self, query: np.ndarray, top_k: int) -> list[int]:
        if not self._vectors:
            return []
        q_norm = np.linalg.norm(query)
        q = query / q_norm if q_norm > 0 else query
        sims = [float(np.dot(q, v)) for v in self._vectors]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        return ranked[:top_k]
