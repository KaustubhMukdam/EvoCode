"""Static retrieval baselines for comparison."""

from __future__ import annotations

import random

from evocode.retrieval.chunker import CodeChunk
from evocode.retrieval.retriever import Retriever


class FixedKRetriever:
    """Always returns the first k chunks (no ranking)."""

    def __init__(self, chunks: list[CodeChunk], k: int = 5) -> None:
        self._chunks = chunks
        self._k = k

    def retrieve(self, query: str, top_k: int | None = None) -> list[CodeChunk]:  # noqa: ARG002
        return self._chunks[: self._k]


class TopKByLength:
    """Returns the k longest chunks."""

    def __init__(self, chunks: list[CodeChunk], k: int = 5) -> None:
        self._chunks = sorted(chunks, key=lambda c: len(c.text), reverse=True)
        self._k = k

    def retrieve(self, query: str, top_k: int | None = None) -> list[CodeChunk]:  # noqa: ARG002
        return self._chunks[: self._k]


class RandomRetriever:
    """Returns k random chunks (deterministic with seed)."""

    def __init__(self, chunks: list[CodeChunk], k: int = 5, seed: int = 42) -> None:
        self._chunks = chunks
        self._k = k
        self._seed = seed

    def retrieve(self, query: str, top_k: int | None = None) -> list[CodeChunk]:  # noqa: ARG002
        rng = random.Random(self._seed)
        shuffled = list(self._chunks)
        rng.shuffle(shuffled)
        return shuffled[: self._k]
