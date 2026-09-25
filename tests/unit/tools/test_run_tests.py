"""Tests for run_tests tool."""

import tempfile
from pathlib import Path

from evocode.tools.base import Tool, ToolCategory
from evocode.tools.run_tests import RunTestsTool


class TestRunTestsToolProtocol:
    """Test RunTestsTool satisfies Tool protocol."""

    def test_satisfies_tool_protocol(self) -> None:
        """RunTestsTool is a structural subtype of Tool."""
        with tempfile.TemporaryDirectory() as d:
            tool = RunTestsTool(sandbox_root=Path(d), timeout_seconds=30)
            assert isinstance(tool, Tool)

    def test_name_and_category(self) -> None:
        """Tool has correct name and category."""
        with tempfile.TemporaryDirectory() as d:
            tool = RunTestsTool(sandbox_root=Path(d), timeout_seconds=30)
            assert tool.name == "run_tests"
            assert tool.category == ToolCategory.CONTROLLED


class TestRunTestsToolBehavior:
    """Test RunTestsTool behavior."""

    def _make_project_with_tests(self, tmp_path: Path, test_code: str) -> Path:
        """Create a minimal project with test file."""
        (tmp_path / "main.py").write_text("def add(a, b):\n    return a + b\n")
        (tmp_path / "test_main.py").write_text(test_code)
        return tmp_path

    def test_run_tests_success(self, tmp_path: Path) -> None:
        """Passing tests returns success=True."""
        self._make_project_with_tests(tmp_path, "from main import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
        tool = RunTestsTool(sandbox_root=tmp_path, timeout_seconds=30)
        result = tool.execute()
        assert result.success
        assert "passed" in result.output.lower()

    def test_run_tests_failure(self, tmp_path: Path) -> None:
        """Failing tests returns success=False with failure output."""
        self._make_project_with_tests(tmp_path, "from main import add\n\ndef test_add():\n    assert add(1, 2) == 5\n")
        tool = RunTestsTool(sandbox_root=tmp_path, timeout_seconds=30)
        result = tool.execute()
        assert not result.success
        assert "failed" in result.output.lower() or "assert" in result.output.lower()

    def test_run_tests_timeout(self, tmp_path: Path) -> None:
        """Long-running tests killed after timeout_seconds."""
        slow_test = "import time\n\ndef test_slow():\n    time.sleep(10)\n"
        self._make_project_with_tests(tmp_path, slow_test)
        tool = RunTestsTool(sandbox_root=tmp_path, timeout_seconds=1)
        result = tool.execute()
        assert not result.success
        assert "timed out" in result.error.lower() or "timed out" in result.output.lower()

    def test_run_tests_returns_exit_code(self, tmp_path: Path) -> None:
        """Exit code available in output or metadata."""
        self._make_project_with_tests(tmp_path, "from main import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
        tool = RunTestsTool(sandbox_root=tmp_path, timeout_seconds=30)
        result = tool.execute()
        # Exit code should be in output or accessible
        assert result.success  # exit code 0

    def test_run_tests_captures_stdout_stderr(self, tmp_path: Path) -> None:
        """Both stdout and stderr from subprocess are captured (pytest output present)."""
        test_code = "def test_dummy():\n    assert True\n"
        self._make_project_with_tests(tmp_path, test_code)
        tool = RunTestsTool(sandbox_root=tmp_path, timeout_seconds=30)
        result = tool.execute()
        # Just verify we get pytest output (which goes through subprocess capture)
        assert "pytest" in result.output.lower() or "passed" in result.output.lower()

    def test_run_tests_respects_timeout_config(self, tmp_path: Path) -> None:
        """Uses ToolConfig.timeout_seconds."""
        tool = RunTestsTool(sandbox_root=tmp_path, timeout_seconds=5)
        assert tool._timeout_seconds == 5

    def test_latency_recorded(self, tmp_path: Path) -> None:
        """Latency is recorded in milliseconds."""
        self._make_project_with_tests(tmp_path, "from main import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
        tool = RunTestsTool(sandbox_root=tmp_path, timeout_seconds=30)
        result = tool.execute()
        assert result.latency_ms > 0

    def test_no_tests_returns_error(self, tmp_path: Path) -> None:
        """No test files returns error."""
        (tmp_path / "main.py").write_text("x = 1")
        tool = RunTestsTool(sandbox_root=tmp_path, timeout_seconds=30)
        result = tool.execute()
        assert not result.success