"""Tests for real code chunker (line-splitting, not just protocol)."""

from pathlib import Path

from evocode.retrieval.line_chunker import LineChunker
from evocode.retrieval.chunker import CodeChunk


class TestLineChunker:
    """Test real line-based code chunker."""

    def test_single_file(self) -> None:
        content = "line1\nline2\nline3\n"
        chunks = LineChunker().chunk_file(Path("a.py"), content)
        assert len(chunks) == 1
        assert chunks[0].file_path == "a.py"
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 3
        assert "line1" in chunks[0].text

    def test_respects_max_lines(self) -> None:
        content = "\n".join([f"line{i}" for i in range(10)])
        chunker = LineChunker(max_lines=3)
        chunks = chunker.chunk_file(Path("a.py"), content)
        assert len(chunks) == 4  # 3+3+3+1

    def test_empty_file(self) -> None:
        chunks = LineChunker().chunk_file(Path("empty.py"), "")
        assert len(chunks) == 1
        assert chunks[0].text == ""

    def test_language_detection(self) -> None:
        chunker = LineChunker()
        assert chunker._detect_language(Path("main.py")) == "python"
        assert chunker._detect_language(Path("app.js")) == "javascript"
        assert chunker._detect_language(Path("lib.rs")) == "rust"
        assert chunker._detect_language(Path("file.txt")) == "text"

    def test_chunk_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
        (tmp_path / "b.py").write_text("z = 3\n")
        chunker = LineChunker()
        chunks = chunker.chunk_directory(tmp_path)
        assert len(chunks) == 2
        files = {c.file_path for c in chunks}
        assert "a.py" in files
        assert "b.py" in files

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("x")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("ok")
        chunks = LineChunker().chunk_directory(tmp_path)
        assert len(chunks) == 1

    def test_chunk_metadata(self) -> None:
        chunks = LineChunker(max_lines=2).chunk_file(Path("f.py"), "a\nb\nc\n")
        assert all("line_range" in c.metadata for c in chunks)
        assert chunks[0].metadata["line_range"] == (1, 2)
        assert chunks[1].metadata["line_range"] == (3, 3)
