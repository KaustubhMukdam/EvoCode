"""Grep tool."""

from __future__ import annotations

import re
import time
from pathlib import Path

from evocode.tools.base import Tool, ToolCategory, ToolResult
from evocode.tools.path_validator import PathValidator


class GrepTool:
    """Tool for searching files with regex patterns within sandbox."""

    name = "grep"
    category = ToolCategory.READ_ONLY

    def __init__(self, path_validator: PathValidator | None, max_results: int = 50) -> None:
        self._validator = path_validator
        self._max_results = max_results

    def execute(
        self,
        pattern: str,
        path: str | None = None,
        case_sensitive: bool = True,
        max_results: int | None = None,
    ) -> ToolResult:
        """Search for pattern in files."""
        start_time = time.perf_counter()

        if self._validator is None:
            return ToolResult(
                success=False,
                output="",
                error="Path validator not configured",
                latency_ms=0.0,
                tool_name=self.name,
            )

        # Compile regex
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid regex pattern: {e}",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                tool_name=self.name,
            )

        # Determine files to search
        sandbox_root = self._validator.sandbox_root
        search_root = sandbox_root
        if path is not None:
            # Validate the path kwarg
            result = self._validator.validate(path)
            if not result.is_valid:
                return ToolResult(
                    success=False,
                    output="",
                    error=result.error or "Path validation failed",
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    tool_name=self.name,
                )
            search_root = result.resolved_path
            assert search_root is not None

        # Collect files
        files: list[Path] = []
        if search_root.is_file():
            files = [search_root]
        else:
            for f in search_root.rglob("*"):
                if f.is_file():
                    files.append(f)

        # Search
        matches: list[str] = []
        limit = max_results if max_results is not None else self._max_results

        for f in files:
            if len(matches) >= limit:
                break
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except (UnicodeDecodeError, OSError):
                continue  # skip binary/unreadable

            for i, line in enumerate(content.splitlines(), 1):
                if len(matches) >= limit:
                    break
                if regex.search(line):
                    rel = f.relative_to(sandbox_root)
                    matches.append(f"{rel}:{i}:{line.rstrip()}")

        latency_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=True,
            output="\n".join(matches),
            error=None,
            latency_ms=latency_ms,
            tool_name=self.name,
        )