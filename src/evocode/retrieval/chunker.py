"""Code chunker for EvoCode."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class CodeChunk:
    """A chunk of code from a source file."""

    text: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Chunker(Protocol):
    """Protocol for chunking code files into searchable pieces."""

    def chunk_file(self, file_path: Path, content: str) -> list[CodeChunk]: ...

    def chunk_directory(self, directory: Path) -> list[CodeChunk]: ...


class FakeChunker:
    """Deterministic fake chunker for testing."""

    def __init__(self, chunks: list[CodeChunk]) -> None:
        self._chunks = chunks

    def chunk_file(self, file_path: Path, content: str) -> list[CodeChunk]:  # noqa: ARG002
        return self._chunks

    def chunk_directory(self, directory: Path) -> list[CodeChunk]:  # noqa: ARG002
        return self._chunks
