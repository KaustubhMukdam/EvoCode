"""Tests for static retrieval baselines."""

from evocode.retrieval.baselines import (
    FixedKRetriever,
    TopKByLength,
    RandomRetriever,
)
from evocode.retrieval.chunker import CodeChunk


def _chunks(n: int = 5) -> list[CodeChunk]:
    return [
        CodeChunk(
            text=f"chunk_{i} " * (i + 1),
            file_path=f"f{i}.py",
            start_line=i,
            end_line=i,
            language="python",
            metadata={},
        )
        for i in range(n)
    ]


class TestFixedKRetriever:
    def test_returns_first_k(self) -> None:
        cs = _chunks(5)
        r = FixedKRetriever(chunks=cs, k=3)
        results = r.retrieve("query")
        assert len(results) == 3
        assert results[0].file_path == "f0.py"

    def test_k_larger_than_chunks(self) -> None:
        r = FixedKRetriever(chunks=_chunks(2), k=10)
        assert len(r.retrieve("q")) == 2


class TestTopKByLength:
    def test_returns_longest_k(self) -> None:
        cs = _chunks(5)
        r = TopKByLength(chunks=cs, k=2)
        results = r.retrieve("query")
        assert len(results) == 2
        # longest chunks have most text
        assert results[0].text.count("chunk_") >= results[1].text.count("chunk_")


class TestRandomRetriever:
    def test_returns_k_random(self) -> None:
        r = RandomRetriever(chunks=_chunks(10), k=3, seed=42)
        results = r.retrieve("query")
        assert len(results) == 3

    def test_deterministic_with_seed(self) -> None:
        r1 = RandomRetriever(chunks=_chunks(10), k=3, seed=42)
        r2 = RandomRetriever(chunks=_chunks(10), k=3, seed=42)
        assert [c.file_path for c in r1.retrieve("q")] == [c.file_path for c in r2.retrieve("q")]
