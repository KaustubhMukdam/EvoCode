"""Tests for read_file tool."""

from pathlib import Path

from evocode.tools.base import Tool, ToolCategory
from evocode.tools.read_file import ReadFileTool


class TestReadFileToolProtocol:
    """Test ReadFileTool satisfies Tool protocol."""

    def test_satisfies_tool_protocol(self) -> None:
        """ReadFileTool is a structural subtype of Tool."""
        tool = ReadFileTool(path_validator=None, max_file_size_mb=5)
        assert isinstance(tool, Tool)

    def test_name_and_category(self) -> None:
        """Tool has correct name and category."""
        tool = ReadFileTool(path_validator=None, max_file_size_mb=5)
        assert tool.name == "read_file"
        assert tool.category == ToolCategory.READ_ONLY


class TestReadFileToolBehavior:
    """Test ReadFileTool behavior with real path validator."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        """Read existing file returns success with content."""
        (tmp_path / "main.py").write_text("def greet():\n    return 'hello'\n")
        tool = self._make_tool(tmp_path)
        result = tool.execute(path="main.py")
        assert result.success
        assert "def greet" in result.output
        assert result.error is None

    def test_read_missing_file(self, tmp_path: Path) -> None:
        """Missing file returns error result."""
        tool = self._make_tool(tmp_path)
        result = tool.execute(path="missing.py")
        assert not result.success
        assert result.error is not None
        assert result.output == ""

    def test_read_file_too_large(self, tmp_path: Path) -> None:
        """File exceeding max_file_size_mb returns error."""
        big = tmp_path / "big.py"
        big.write_text("x" * (6 * 1024 * 1024))
        tool = self._make_tool(tmp_path, max_file_size_mb=5)
        result = tool.execute(path="big.py")
        assert not result.success
        assert "size" in result.error.lower()

    def test_read_binary_file(self, tmp_path: Path) -> None:
        """Binary file returns error or degraded output."""
        bin_file = tmp_path / "data.bin"
        bin_file.write_bytes(b"\x00\x01\x02\xff\xfe")
        tool = self._make_tool(tmp_path)
        result = tool.execute(path="data.bin")
        # Either error or empty/safe output
        assert not result.success or result.output == ""

    def test_read_with_line_range(self, tmp_path: Path) -> None:
        """start_line/end_line kwargs slice output."""
        (tmp_path / "main.py").write_text("line1\nline2\nline3\nline4\nline5\n")
        tool = self._make_tool(tmp_path)
        result = tool.execute(path="main.py", start_line=2, end_line=4)
        assert result.success
        assert "line2" in result.output
        assert "line3" in result.output
        assert "line4" in result.output
        assert "line1" not in result.output
        assert "line5" not in result.output

    def test_latency_recorded(self, tmp_path: Path) -> None:
        """Latency is recorded in milliseconds."""
        (tmp_path / "main.py").write_text("x = 1")
        tool = self._make_tool(tmp_path)
        result = tool.execute(path="main.py")
        assert result.latency_ms > 0

    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        """Path traversal is rejected."""
        tool = self._make_tool(tmp_path)
        result = tool.execute(path="../../etc/passwd")
        assert not result.success
        assert "traversal" in result.error.lower() or "outside" in result.error.lower()

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns success with empty output."""
        (tmp_path / "empty.py").write_text("")
        tool = self._make_tool(tmp_path)
        result = tool.execute(path="empty.py")
        assert result.success
        assert result.output == ""

    def test_read_respects_sandbox(self, tmp_path: Path) -> None:
        """File outside sandbox returns error."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.py").write_text("x")
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        tool = ReadFileTool(
            path_validator=None, max_file_size_mb=5
        )
        # We test the validator rejects it
        from evocode.tools.path_validator import PathValidator
        validator = PathValidator(sandbox_root=sandbox, max_file_size_mb=5)
        tool = ReadFileTool(path_validator=validator, max_file_size_mb=5)
        result = tool.execute(path="../outside/secret.py")
        assert not result.success

    def _make_tool(self, sandbox: Path, max_file_size_mb: int = 5) -> ReadFileTool:
        from evocode.tools.path_validator import PathValidator
        validator = PathValidator(sandbox_root=sandbox, max_file_size_mb=max_file_size_mb)
        return ReadFileTool(path_validator=validator, max_file_size_mb=max_file_size_mb)