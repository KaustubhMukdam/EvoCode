"""Tests for tool dispatcher."""

from pathlib import Path

from evocode.config.settings import ToolConfig
from evocode.domain.actions import ActionType
from evocode.tools.base import ToolCategory
from evocode.tools.dispatcher import ToolDispatcher
from evocode.tools.read_file import ReadFileTool
from evocode.tools.grep_tool import GrepTool
from evocode.tools.run_tests import RunTestsTool
from evocode.tools.path_validator import PathValidator


class TestToolDispatcher:
    """Test ToolDispatcher routes actions to correct tools."""

    def _make_dispatcher(self, sandbox: Path, tool_config: ToolConfig | None = None) -> ToolDispatcher:
        validator = PathValidator(sandbox_root=sandbox, max_file_size_mb=5)
        tools = {
            "read_file": ReadFileTool(path_validator=validator, max_file_size_mb=5),
            "grep": GrepTool(path_validator=validator, max_results=50),
            "run_tests": RunTestsTool(sandbox_root=sandbox, timeout_seconds=30),
        }
        return ToolDispatcher(tools=tools, tool_config=tool_config or ToolConfig())

    def test_dispatch_read_file(self, tmp_path: Path) -> None:
        """READ_FILE action routes to ReadFileTool."""
        (tmp_path / "main.py").write_text("x = 1")
        dispatcher = self._make_dispatcher(tmp_path)
        result = dispatcher.dispatch(ActionType.READ_FILE, path="main.py")
        assert result.success
        assert "x = 1" in result.output

    def test_dispatch_grep(self, tmp_path: Path) -> None:
        """GREP action routes to GrepTool."""
        (tmp_path / "main.py").write_text("def foo(): pass")
        dispatcher = self._make_dispatcher(tmp_path)
        result = dispatcher.dispatch(ActionType.GREP, pattern="def ")
        assert result.success
        assert "foo" in result.output

    def test_dispatch_run_tests(self, tmp_path: Path) -> None:
        """RUN_TESTS action routes to RunTestsTool."""
        (tmp_path / "main.py").write_text("def add(a, b): return a + b")
        (tmp_path / "test_main.py").write_text("from main import add\ndef test_add():\n    assert add(1,2)==3")
        dispatcher = self._make_dispatcher(tmp_path)
        result = dispatcher.dispatch(ActionType.RUN_TESTS)
        assert result.success

    def test_dispatch_retrieval_actions_return_error(self, tmp_path: Path) -> None:
        """RETRIEVE_* actions are not tool actions."""
        dispatcher = self._make_dispatcher(tmp_path)
        result = dispatcher.dispatch(ActionType.RETRIEVE_SMALL)
        assert not result.success
        assert "not a tool" in result.error.lower() or "not implemented" in result.error.lower()

    def test_dispatch_control_actions(self, tmp_path: Path) -> None:
        """STOP/SUMMARIZE are not tool actions."""
        dispatcher = self._make_dispatcher(tmp_path)
        result = dispatcher.dispatch(ActionType.STOP)
        assert not result.success
        assert "not a tool" in result.error.lower()

    def test_dispatch_summarize(self, tmp_path: Path) -> None:
        """SUMMARIZE is not a tool action."""
        dispatcher = self._make_dispatcher(tmp_path)
        result = dispatcher.dispatch(ActionType.SUMMARIZE)
        assert not result.success
        assert "not a tool" in result.error.lower()

    def test_dispatch_unsupported_memory_action_is_rejected(self, tmp_path: Path) -> None:
        """Unsupported non-tool actions are rejected with a clear error."""
        dispatcher = self._make_dispatcher(tmp_path)
        result = dispatcher.dispatch(ActionType.RECALL_SKILL)
        assert not result.success
        assert "not a tool" in result.error.lower()

    def test_dispatch_respects_tool_config(self, tmp_path: Path) -> None:
        """ToolConfig blocks disallowed actions before execution."""
        config = ToolConfig(allow_read=False, allow_grep=True, allow_run_tests=True)
        dispatcher = self._make_dispatcher(tmp_path, tool_config=config)
        (tmp_path / "main.py").write_text("x = 1")

        result = dispatcher.dispatch(ActionType.READ_FILE, path="main.py")
        assert not result.success
        assert "not allowed" in result.error.lower()

    def test_dispatch_unknown_action(self, tmp_path: Path) -> None:
        """Unknown action returns error."""
        dispatcher = self._make_dispatcher(tmp_path)
        result = dispatcher.dispatch(ActionType.RECALL_SKILL)
        assert not result.success
        assert result.error is not None

    def test_dispatch_records_latency(self, tmp_path: Path) -> None:
        """All dispatches record latency."""
        (tmp_path / "main.py").write_text("x = 1")
        dispatcher = self._make_dispatcher(tmp_path)
        result = dispatcher.dispatch(ActionType.READ_FILE, path="main.py")
        assert result.latency_ms > 0

    def test_dispatch_unavailable_tool(self, tmp_path: Path) -> None:
        """Tool not in registry returns error."""
        dispatcher = ToolDispatcher(tools={})
        result = dispatcher.dispatch(ActionType.READ_FILE, path="x.py")
        assert not result.success
        assert "not available" in result.error.lower() or "not found" in result.error.lower()