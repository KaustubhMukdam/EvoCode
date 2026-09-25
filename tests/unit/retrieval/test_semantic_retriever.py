"""Tests for SemanticRetriever — ties Chunker + Embedder + Index into Retriever protocol."""

import numpy as np
from pathlib import Path

from evocode.retrieval.chunker import CodeChunk, FakeChunker
from evocode.retrieval.embedder import FakeEmbedder
from evocode.retrieval.index import FakeVectorIndex
from evocode.retrieval.retriever import Retriever
from evocode.retrieval.semantic_retriever import SemanticRetriever


def _make_chunks(n: int = 3) -> list[CodeChunk]:
    return [
        CodeChunk(
            text=f"chunk_{i}",
            file_path=f"file_{i}.py",
            start_line=i * 10,
            end_line=i * 10 + 5,
            language="python",
            metadata={"index": i},
        )
        for i in range(n)
    ]


class TestSemanticRetrieverProtocol:
    """Test SemanticRetriever satisfies Retriever protocol."""

    def test_satisfies_retriever_protocol(self) -> None:
        retriever = SemanticRetriever(
            chunker=FakeChunker(chunks=[]),
            embedder=FakeEmbedder(dimension=384),
            index=FakeVectorIndex(dimension=384),
        )
        assert isinstance(retriever, Retriever)


class TestSemanticRetrieverIndexing:
    """Test indexing a directory."""

    def test_index_directory_adds_chunks(self) -> None:
        chunks = _make_chunks(3)
        chunker = FakeChunker(chunks=chunks)
        embedder = FakeEmbedder(dimension=384)
        index = FakeVectorIndex(dimension=384)
        retriever = SemanticRetriever(chunker=chunker, embedder=embedder, index=index)
        retriever.index_directory(Path("."))
        assert index.size == 3

    def test_index_empty_directory(self) -> None:
        chunker = FakeChunker(chunks=[])
        embedder = FakeEmbedder(dimension=384)
        index = FakeVectorIndex(dimension=384)
        retriever = SemanticRetriever(chunker=chunker, embedder=embedder, index=index)
        retriever.index_directory(Path("."))
        assert index.size == 0


class TestSemanticRetrieverRetrieval:
    """Test retrieval returns ranked chunks."""

    def test_retrieve_returns_chunks(self) -> None:
        chunks = _make_chunks(5)
        chunker = FakeChunker(chunks=chunks)
        embedder = FakeEmbedder(dimension=384)
        index = FakeVectorIndex(dimension=384)
        retriever = SemanticRetriever(chunker=chunker, embedder=embedder, index=index)
        retriever.index_directory(Path("."))
        results = retriever.retrieve("test query", top_k=3)
        assert len(results) == 3
        assert all(isinstance(c, CodeChunk) for c in results)

    def test_retrieve_empty_index(self) -> None:
        chunker = FakeChunker(chunks=[])
        embedder = FakeEmbedder(dimension=384)
        index = FakeVectorIndex(dimension=384)
        retriever = SemanticRetriever(chunker=chunker, embedder=embedder, index=index)
        results = retriever.retrieve("query", top_k=3)
        assert len(results) == 0

    def test_retrieve_top_k_clamped(self) -> None:
        chunks = _make_chunks(2)
        chunker = FakeChunker(chunks=chunks)
        embedder = FakeEmbedder(dimension=384)
        index = FakeVectorIndex(dimension=384)
        retriever = SemanticRetriever(chunker=chunker, embedder=embedder, index=index)
        retriever.index_directory(Path("."))
        results = retriever.retrieve("query", top_k=10)
        assert len(results) == 2
