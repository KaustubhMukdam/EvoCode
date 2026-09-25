"""Semantic retriever — ties Chunker + Embedder + VectorIndex into Retriever protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evocode.retrieval.chunker import Chunker, CodeChunk
from evocode.retrieval.embedder import Embedder
from evocode.retrieval.index import VectorIndex


class SemanticRetriever:
    """Concrete retriever using semantic embedding search."""

    def __init__(self, chunker: Chunker, embedder: Embedder, index: VectorIndex) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._index = index
        self._chunks: list[CodeChunk] = []

    def index_directory(self, directory: Path) -> None:
        chunks = self._chunker.chunk_directory(directory)
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = self._embedder.embed_batch(texts)
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[CodeChunk]:
        if not self._chunks:
            return []
        vec = self._embedder.embed(query)
        indices = self._index.search(vec, top_k=min(top_k, len(self._chunks)))
        return [self._chunks[i] for i in indices]
