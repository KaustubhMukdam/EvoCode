"""Line-based code chunker for EvoCode."""

from __future__ import annotations

from pathlib import Path

from evocode.retrieval.chunker import Chunker, CodeChunk

_HIDDEN_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache", "huggingface_cache"}

_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".sh": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".md": "markdown",
    ".sql": "sql",
}


class LineChunker:
    """Splits files into line-range chunks."""

    def __init__(self, max_lines: int = 50) -> None:
        self._max_lines = max_lines

    def _detect_language(self, file_path: Path) -> str:
        return _LANG_MAP.get(file_path.suffix.lower(), "text")

    def chunk_file(self, file_path: Path, content: str) -> list[CodeChunk]:
        lines = content.splitlines(keepends=True)
        if not lines:
            return [CodeChunk(text="", file_path=str(file_path), start_line=1, end_line=0, language=self._detect_language(file_path), metadata={"line_range": (1, 0)})]
        chunks: list[CodeChunk] = []
        lang = self._detect_language(file_path)
        for i in range(0, len(lines), self._max_lines):
            batch = lines[i : i + self._max_lines]
            start = i + 1
            end = i + len(batch)
            chunks.append(CodeChunk(
                text="".join(batch),
                file_path=str(file_path),
                start_line=start,
                end_line=end,
                language=lang,
                metadata={"line_range": (start, end)},
            ))
        return chunks

    def chunk_directory(self, directory: Path) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if any(part.startswith(".") and part not in (".",) for part in path.relative_to(directory).parts if part):
                continue
            if any(d in path.parts for d in _HIDDEN_DIRS):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            chunks.extend(self.chunk_file(path.relative_to(directory), text))
        return chunks
