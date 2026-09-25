"""Tests for grep tool."""

from pathlib import Path

from evocode.tools.base import Tool, ToolCategory
from evocode.tools.grep_tool import GrepTool


class TestGrepToolProtocol:
    """Test GrepTool satisfies Tool protocol."""

    def test_satisfies_tool_protocol(self) -> None:
        """GrepTool is a structural subtype of Tool."""
        tool = GrepTool(path_validator=None, max_results=50)
        assert isinstance(tool, Tool)

    def test_name_and_category(self) -> None:
        """Tool has correct name and category."""
        tool = GrepTool(path_validator=None, max_results=50)
        assert tool.name == "grep"
        assert tool.category == ToolCategory.READ_ONLY


class TestGrepToolBehavior:
    """Test GrepTool behavior with real path validator."""

    def _make_tool(self, sandbox: Path) -> GrepTool:
        from evocode.tools.path_validator import PathValidator
        validator = PathValidator(sandbox_root=sandbox, max_file_size_mb=5)
        return GrepTool(path_validator=validator, max_results=50)

    def test_grep_matches_pattern(self, tmp_path: Path) -> None:
        """Regex pattern matches file contents."""
        (tmp_path / "main.py").write_text("def greet():\n    return 'hello'\n")
        (tmp_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")
        tool = self._make_tool(tmp_path)
        result = tool.execute(pattern="def ")
        assert result.success
        assert "def greet" in result.output
        assert "def add" in result.output

    def test_grep_no_matches(self, tmp_path: Path) -> None:
        """No matches returns empty output, success=True."""
        (tmp_path / "main.py").write_text("x = 1\ny = 2\n")
        tool = self._make_tool(tmp_path)
        result = tool.execute(pattern="def ")
        assert result.success
        assert result.output == ""

    def test_grep_case_insensitive(self, tmp_path: Path) -> None:
        """case_sensitive=False flag works."""
        (tmp_path / "main.py").write_text("Def greet():\n    return 'hello'\n")
        tool = self._make_tool(tmp_path)
        result = tool.execute(pattern="def ", case_sensitive=False)
        assert result.success
        assert "Def greet" in result.output

    def test_grep_max_results(self, tmp_path: Path) -> None:
        """max_results kwarg caps output."""
        lines = "\n".join([f"def func{i}():" for i in range(100)])
        (tmp_path / "big.py").write_text(lines)
        tool = GrepTool(path_validator=None, max_results=5)
        # Need validator for real test
        from evocode.tools.path_validator import PathValidator
        validator = PathValidator(sandbox_root=tmp_path, max_file_size_mb=5)
        tool = GrepTool(path_validator=validator, max_results=5)
        result = tool.execute(pattern="def func", max_results=3)
        assert result.success
        count = result.output.count("def func")
        assert count <= 3

    def test_grep_multiple_files(self, tmp_path: Path) -> None:
        """Searches across multiple files."""
        (tmp_path / "a.py").write_text("def foo(): pass")
        (tmp_path / "b.py").write_text("def bar(): pass")
        tool = self._make_tool(tmp_path)
        result = tool.execute(pattern="def ")
        assert result.success
        assert "foo" in result.output
        assert "bar" in result.output

    def test_grep_invalid_regex(self, tmp_path: Path) -> None:
        """Bad regex returns error."""
        tool = self._make_tool(tmp_path)
        result = tool.execute(pattern="[invalid")
        assert not result.success
        assert "regex" in result.error.lower() or "pattern" in result.error.lower()

    def test_grep_respects_sandbox(self, tmp_path: Path) -> None:
        """Cannot grep outside sandbox."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.py").write_text("def secret(): pass")
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        from evocode.tools.path_validator import PathValidator
        validator = PathValidator(sandbox_root=sandbox, max_file_size_mb=5)
        tool = GrepTool(path_validator=validator, max_results=50)
        result = tool.execute(pattern="def ")
        assert result.success
        assert "secret" not in result.output

    def test_grep_skips_binary(self, tmp_path: Path) -> None:
        """Binary files are skipped."""
        (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")
        (tmp_path / "main.py").write_text("def real(): pass")
        tool = self._make_tool(tmp_path)
        result = tool.execute(pattern="def ")
        assert result.success
        assert "real" in result.output

    def test_grep_line_numbers(self, tmp_path: Path) -> None:
        """Output includes file:line format."""
        (tmp_path / "main.py").write_text("x = 1\ndef greet():\n    pass\n")
        tool = self._make_tool(tmp_path)
        result = tool.execute(pattern="def ")
        assert result.success
        assert "main.py" in result.output
        assert "2" in result.output  # line number

    def test_latency_recorded(self, tmp_path: Path) -> None:
        """Latency is recorded in milliseconds."""
        (tmp_path / "main.py").write_text("x = 1")
        tool = self._make_tool(tmp_path)
        result = tool.execute(pattern="x")
        assert result.latency_ms > 0

    def test_grep_with_path_kwarg(self, tmp_path: Path) -> None:
        """Optional path kwarg limits search to specific file/dir."""
        (tmp_path / "main.py").write_text("def foo(): pass")
        (tmp_path / "utils.py").write_text("def bar(): pass")
        tool = self._make_tool(tmp_path)
        result = tool.execute(pattern="def ", path="main.py")
        assert result.success
        assert "foo" in result.output
        assert "bar" not in result.output