"""Read file tool."""

from __future__ import annotations

import time
from pathlib import Path

from evocode.tools.base import Tool, ToolCategory, ToolResult
from evocode.tools.path_validator import PathValidator


class ReadFileTool:
    """Tool for reading files within sandbox."""

    name = "read_file"
    category = ToolCategory.READ_ONLY

    def __init__(self, path_validator: PathValidator | None, max_file_size_mb: int) -> None:
        self._validator = path_validator
        self._max_file_size_mb = max_file_size_mb

    def execute(self, path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
        """Read a file from the sandbox."""
        start_time = time.perf_counter()

        if self._validator is None:
            return ToolResult(
                success=False,
                output="",
                error="Path validator not configured",
                latency_ms=0.0,
                tool_name=self.name,
            )

        # Validate path
        result = self._validator.validate(path)
        if not result.is_valid:
            return ToolResult(
                success=False,
                output="",
                error=result.error or "Path validation failed",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                tool_name=self.name,
            )

        resolved_path = result.resolved_path
        assert resolved_path is not None

        # Try to read as text
        try:
            content = resolved_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                output="",
                error="Binary file: cannot read as text",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                tool_name=self.name,
            )
        except OSError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to read file: {e}",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                tool_name=self.name,
            )

        # Apply line range if specified
        if start_line is not None or end_line is not None:
            lines = content.splitlines(keepends=True)
            start = (start_line - 1) if start_line is not None else 0
            end = end_line if end_line is not None else len(lines)
            content = "".join(lines[start:end])

        latency_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=True,
            output=content,
            error=None,
            latency_ms=latency_ms,
            tool_name=self.name,
        )