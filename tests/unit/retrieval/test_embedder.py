"""Tests for text embedder."""

import numpy as np

from evocode.retrieval.embedder import Embedder, FakeEmbedder, SentenceTransformerEmbedder


class TestEmbedderProtocol:
    """Test Embedder protocol contract."""

    def test_protocol_has_embed(self) -> None:
        """Test protocol defines embed method."""
        assert hasattr(Embedder, "embed")

    def test_protocol_has_embed_batch(self) -> None:
        """Test protocol defines embed_batch method."""
        assert hasattr(Embedder, "embed_batch")

    def test_protocol_has_dimension(self) -> None:
        """Test protocol defines dimension property."""
        assert hasattr(Embedder, "dimension")


class TestFakeEmbedder:
    """Test FakeEmbedder implementation."""

    def test_embed_returns_vector(self) -> None:
        """Test embed returns a numpy array."""
        embedder = FakeEmbedder(dimension=384)
        vec = embedder.embed("hello world")
        assert isinstance(vec, np.ndarray)

    def test_embed_dimension(self) -> None:
        """Test embed returns correct dimension."""
        embedder = FakeEmbedder(dimension=384)
        vec = embedder.embed("test")
        assert vec.shape == (384,)

    def test_embed_dtype(self) -> None:
        """Test embed returns float32."""
        embedder = FakeEmbedder(dimension=384)
        vec = embedder.embed("test")
        assert vec.dtype == np.float32

    def test_embed_deterministic(self) -> None:
        """Test same input produces same output."""
        embedder = FakeEmbedder(dimension=384)
        v1 = embedder.embed("same input")
        v2 = embedder.embed("same input")
        np.testing.assert_array_equal(v1, v2)

    def test_embed_different_inputs(self) -> None:
        """Test different inputs produce different outputs."""
        embedder = FakeEmbedder(dimension=384)
        v1 = embedder.embed("hello")
        v2 = embedder.embed("world")
        assert not np.array_equal(v1, v2)

    def test_embed_ignores_python_hash_randomization(self, monkeypatch) -> None:
        """Test deterministic embed generation is not tied to Python hash randomness."""
        embedder = FakeEmbedder(dimension=384)
        monkeypatch.setattr("builtins.hash", lambda value: 999999999)
        v1 = embedder.embed("deterministic input")
        v2 = embedder.embed("deterministic input")
        np.testing.assert_array_equal(v1, v2)

    def test_embed_batch(self) -> None:
        """Test embed_batch returns multiple vectors."""
        embedder = FakeEmbedder(dimension=384)
        vecs = embedder.embed_batch(["a", "b", "c"])
        assert vecs.shape == (3, 384)

    def test_embed_batch_empty(self) -> None:
        """Test embed_batch with empty list."""
        embedder = FakeEmbedder(dimension=384)
        vecs = embedder.embed_batch([])
        assert vecs.shape == (0, 384)

    def test_dimension_property(self) -> None:
        """Test dimension property returns correct value."""
        embedder = FakeEmbedder(dimension=128)
        assert embedder.dimension == 128


class TestSentenceTransformerEmbedder:
    """Test SentenceTransformerEmbedder (protocol-only, no real model)."""

    def test_satisfies_protocol(self) -> None:
        """Test SentenceTransformerEmbedder class exists and has required methods."""
        assert hasattr(SentenceTransformerEmbedder, "embed")
        assert hasattr(SentenceTransformerEmbedder, "embed_batch")
        assert hasattr(SentenceTransformerEmbedder, "dimension")
