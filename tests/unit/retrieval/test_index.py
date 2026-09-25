"""Tests for FAISS vector index."""

import numpy as np

from evocode.retrieval.index import FakeVectorIndex, VectorIndex


class TestVectorIndexProtocol:
    """Test VectorIndex protocol contract."""

    def test_protocol_has_add(self) -> None:
        assert hasattr(VectorIndex, "add")

    def test_protocol_has_search(self) -> None:
        assert hasattr(VectorIndex, "search")

    def test_protocol_has_size(self) -> None:
        assert hasattr(VectorIndex, "size")


class TestFakeVectorIndex:
    """Test FakeVectorIndex implementation."""

    def test_add_and_search(self) -> None:
        idx = FakeVectorIndex(dimension=4)
        vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        idx.add(vectors)
        results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)
        assert len(results) == 1
        assert results[0] == 0

    def test_search_top_k(self) -> None:
        idx = FakeVectorIndex(dimension=2)
        vecs = np.array([[1, 0], [0, 1], [1, 0.1]], dtype=np.float32)
        idx.add(vecs)
        results = idx.search(np.array([1, 0], dtype=np.float32), top_k=2)
        assert len(results) == 2
        assert results[0] == 0  # exact match first

    def test_search_empty_index(self) -> None:
        idx = FakeVectorIndex(dimension=4)
        results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=5)
        assert len(results) == 0

    def test_size_empty(self) -> None:
        idx = FakeVectorIndex(dimension=4)
        assert idx.size == 0

    def test_size_after_add(self) -> None:
        idx = FakeVectorIndex(dimension=4)
        idx.add(np.zeros((5, 4), dtype=np.float32))
        assert idx.size == 5

    def test_add_increments_size(self) -> None:
        idx = FakeVectorIndex(dimension=2)
        idx.add(np.zeros((3, 2), dtype=np.float32))
        idx.add(np.zeros((2, 2), dtype=np.float32))
        assert idx.size == 5
