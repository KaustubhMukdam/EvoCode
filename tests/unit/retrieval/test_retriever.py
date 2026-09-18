"""Tests for retrieval interfaces."""

from evocode.retrieval.retriever import Chunk, FakeRetriever, Retriever


class TestChunk:
    """Test Chunk dataclass."""

    def test_chunk_creation(self) -> None:
        """Test creating a chunk with all fields."""
        chunk = Chunk(
            text="def greet(name):",
            file_path="main.py",
            start_line=10,
            end_line=12,
            metadata={"function": "greet"},
        )
        assert chunk.text == "def greet(name):"
        assert chunk.file_path == "main.py"
        assert chunk.start_line == 10
        assert chunk.end_line == 12
        assert chunk.metadata["function"] == "greet"

    def test_chunk_empty_metadata(self) -> None:
        """Test chunk defaults to empty metadata."""
        chunk = Chunk(
            text="x = 1",
            file_path="a.py",
            start_line=1,
            end_line=1,
            metadata={},
        )
        assert len(chunk.metadata) == 0

    def test_chunk_line_range(self) -> None:
        """Test start_line <= end_line convention."""
        chunk = Chunk(text="line", file_path="f.py", start_line=5, end_line=8, metadata={})
        assert chunk.start_line <= chunk.end_line


class TestRetrieverProtocol:
    """Test Retriever protocol contract."""

    def test_protocol_has_retrieve(self) -> None:
        """Test Retriever protocol defines retrieve method."""
        assert hasattr(Retriever, "retrieve")

    def test_fake_retriever_satisfies_protocol(self) -> None:
        """Test FakeRetriever satisfies Retriever protocol."""
        retriever = FakeRetriever(chunks=[])
        assert hasattr(retriever, "retrieve")


class TestFakeRetriever:
    """Test FakeRetriever implementation."""

    def test_retrieve_empty(self) -> None:
        """Test retrieve from empty retriever."""
        retriever = FakeRetriever(chunks=[])
        results = retriever.retrieve("query", top_k=5)
        assert results == []

    def test_retrieve_returns_chunks(self) -> None:
        """Test retrieve returns pre-seeded chunks."""
        chunks = [
            Chunk(text="chunk1", file_path="a.py", start_line=1, end_line=5, metadata={}),
            Chunk(text="chunk2", file_path="b.py", start_line=1, end_line=3, metadata={}),
        ]
        retriever = FakeRetriever(chunks=chunks)
        results = retriever.retrieve("query", top_k=5)
        assert len(results) == 2

    def test_retrieve_respects_top_k(self) -> None:
        """Test retrieve limits results to top_k."""
        chunks = [
            Chunk(text=f"chunk{i}", file_path=f"f{i}.py", start_line=1, end_line=1, metadata={})
            for i in range(10)
        ]
        retriever = FakeRetriever(chunks=chunks)
        results = retriever.retrieve("query", top_k=3)
        assert len(results) == 3

    def test_retrieve_top_k_larger_than_chunks(self) -> None:
        """Test retrieve returns all chunks when top_k > len(chunks)."""
        chunks = [
            Chunk(text="only", file_path="a.py", start_line=1, end_line=1, metadata={})
        ]
        retriever = FakeRetriever(chunks=chunks)
        results = retriever.retrieve("query", top_k=100)
        assert len(results) == 1

    def test_retrieve_preserves_order(self) -> None:
        """Test retrieve preserves insertion order."""
        chunks = [
            Chunk(text="first", file_path="a.py", start_line=1, end_line=1, metadata={}),
            Chunk(text="second", file_path="b.py", start_line=1, end_line=1, metadata={}),
            Chunk(text="third", file_path="c.py", start_line=1, end_line=1, metadata={}),
        ]
        retriever = FakeRetriever(chunks=chunks)
        results = retriever.retrieve("query", top_k=2)
        assert results[0].text == "first"
        assert results[1].text == "second"

    def test_retrieve_returns_chunk_type(self) -> None:
        """Test retrieve returns list of Chunk objects."""
        chunks = [
            Chunk(text="x", file_path="a.py", start_line=1, end_line=1, metadata={})
        ]
        retriever = FakeRetriever(chunks=chunks)
        results = retriever.retrieve("q", top_k=1)
        assert isinstance(results[0], Chunk)
