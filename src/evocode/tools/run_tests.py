"""Run tests tool."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from evocode.tools.base import Tool, ToolCategory, ToolResult


class RunTestsTool:
    """Tool for running pytest within sandbox."""

    name = "run_tests"
    category = ToolCategory.CONTROLLED

    def __init__(self, sandbox_root: Path, timeout_seconds: int) -> None:
        self._sandbox_root = sandbox_root
        self._timeout_seconds = timeout_seconds

    def execute(self, test_path: str | None = None) -> ToolResult:
        """Run pytest in the sandbox."""
        start_time = time.perf_counter()

        # Build pytest command
        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]
        if test_path:
            cmd.append(test_path)

        try:
            proc = subprocess.run(
                cmd,
                cwd=self._sandbox_root,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Test execution timed out after {self._timeout_seconds} seconds",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                tool_name=self.name,
            )
        except OSError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to run pytest: {e}",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                tool_name=self.name,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000
        output = proc.stdout + "\n" + proc.stderr

        return ToolResult(
            success=proc.returncode == 0,
            output=output,
            error=None if proc.returncode == 0 else f"Tests failed with exit code {proc.returncode}",
            latency_ms=latency_ms,
            tool_name=self.name,
        )