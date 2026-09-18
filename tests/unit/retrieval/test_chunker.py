"""Tests for code chunker."""

from pathlib import Path

from evocode.retrieval.chunker import Chunker, CodeChunk, FakeChunker


class TestCodeChunk:
    """Test CodeChunk dataclass."""

    def test_chunk_creation(self) -> None:
        """Test creating a chunk with all fields."""
        chunk = CodeChunk(
            text="def greet(name):",
            file_path="main.py",
            start_line=10,
            end_line=12,
            language="python",
            metadata={"function": "greet"},
        )
        assert chunk.text == "def greet(name):"
        assert chunk.file_path == "main.py"
        assert chunk.start_line == 10
        assert chunk.end_line == 12
        assert chunk.language == "python"
        assert chunk.metadata["function"] == "greet"

    def test_chunk_empty_metadata(self) -> None:
        """Test chunk defaults to empty metadata."""
        chunk = CodeChunk(
            text="x = 1",
            file_path="a.py",
            start_line=1,
            end_line=1,
            language="python",
            metadata={},
        )
        assert len(chunk.metadata) == 0

    def test_chunk_line_range(self) -> None:
        """Test start_line <= end_line."""
        chunk = CodeChunk(
            text="line", file_path="f.py", start_line=5, end_line=8, language="python", metadata={}
        )
        assert chunk.start_line <= chunk.end_line


class TestChunkerProtocol:
    """Test Chunker protocol contract."""

    def test_protocol_has_chunk(self) -> None:
        """Test Chunker protocol defines chunk_file method."""
        assert hasattr(Chunker, "chunk_file")

    def test_protocol_has_chunk_directory(self) -> None:
        """Test Chunker protocol defines chunk_directory method."""
        assert hasattr(Chunker, "chunk_directory")


class TestFakeChunker:
    """Test FakeChunker implementation."""

    def test_chunk_file_returns_chunks(self) -> None:
        """Test chunk_file returns a list of chunks."""
        fake = FakeChunker(chunks=[])
        result = fake.chunk_file(Path("a.py"), "x = 1")
        assert isinstance(result, list)

    def test_chunk_file_returns_pre_seeded(self) -> None:
        """Test chunk_file returns pre-seeded chunks."""
        seeded = [
            CodeChunk(
                text="seeded",
                file_path="a.py",
                start_line=1,
                end_line=1,
                language="python",
                metadata={},
            )
        ]
        fake = FakeChunker(chunks=seeded)
        result = fake.chunk_file(Path("a.py"), "content")
        assert len(result) == 1
        assert result[0].text == "seeded"

    def test_chunk_directory_returns_chunks(self) -> None:
        """Test chunk_directory returns a list of chunks."""
        fake = FakeChunker(chunks=[])
        result = fake.chunk_directory(Path("."))
        assert isinstance(result, list)

    def test_chunk_directory_returns_pre_seeded(self) -> None:
        """Test chunk_directory returns pre-seeded chunks."""
        seeded = [
            CodeChunk(
                text="dir chunk",
                file_path="b.py",
                start_line=1,
                end_line=5,
                language="python",
                metadata={},
            )
        ]
        fake = FakeChunker(chunks=seeded)
        result = fake.chunk_directory(Path("."))
        assert len(result) == 1
