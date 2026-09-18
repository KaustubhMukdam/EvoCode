"""Tool base types for EvoCode."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ToolCategory(Enum):
    """Safety classification for tool execution."""

    READ_ONLY = "read_only"
    CONTROLLED = "controlled"
    WRITE = "write"
    DANGEROUS = "dangerous"


@dataclass
class ToolResult:
    """Structured result from tool execution."""

    success: bool
    output: str
    error: str | None
    latency_ms: float
    tool_name: str


class ToolError(Exception):
    """Exception raised when a tool execution fails."""

    def __init__(self, message: str, tool_name: str, category: ToolCategory) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.category = category


@runtime_checkable
class Tool(Protocol):
    """Protocol defining the contract for all tools."""

    name: str
    category: ToolCategory

    def execute(self, **kwargs: Any) -> ToolResult: ...
