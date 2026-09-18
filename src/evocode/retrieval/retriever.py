"""Retriever protocol and types for EvoCode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Chunk:
    """A chunk of code/text from a repository file."""

    text: str
    file_path: str
    start_line: int
    end_line: int
    metadata: dict


@runtime_checkable
class Retriever(Protocol):
    """Protocol for retrieving relevant code chunks."""

    def retrieve(self, query: str, top_k: int) -> list[Chunk]: ...


class FakeRetriever:
    """Deterministic fake retriever for testing."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, top_k: int) -> list[Chunk]:  # noqa: ARG002
        return self._chunks[:top_k]
