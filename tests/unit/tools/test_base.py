"""Tests for tool base and registry."""

from __future__ import annotations

from typing import Any

import pytest

from evocode.tools.base import Tool, ToolCategory, ToolError, ToolResult
from evocode.tools.registry import ToolRegistry


class TestToolCategory:
    """Test ToolCategory enum."""

    def test_all_categories_defined(self) -> None:
        """Test expected categories exist."""
        assert ToolCategory.READ_ONLY.value == "read_only"
        assert ToolCategory.CONTROLLED.value == "controlled"
        assert ToolCategory.WRITE.value == "write"
        assert ToolCategory.DANGEROUS.value == "dangerous"

    def test_four_categories(self) -> None:
        """Test exactly four safety categories."""
        assert len(ToolCategory) == 4


class TestToolResult:
    """Test ToolResult dataclass."""

    def test_tool_result_creation(self) -> None:
        """Test creating a successful tool result."""
        result = ToolResult(
            success=True,
            output="file contents here",
            error=None,
            latency_ms=12.5,
            tool_name="read_file",
        )
        assert result.success is True
        assert result.output == "file contents here"
        assert result.error is None
        assert result.latency_ms == 12.5
        assert result.tool_name == "read_file"

    def test_tool_result_failure(self) -> None:
        """Test creating a failed tool result."""
        result = ToolResult(
            success=False,
            output="",
            error="File not found: main.py",
            latency_ms=1.0,
            tool_name="read_file",
        )
        assert result.success is False
        assert result.error == "File not found: main.py"

    def test_tool_result_defaults(self) -> None:
        """Test ToolResult default values."""
        result = ToolResult(
            success=True,
            output="",
            error=None,
            latency_ms=0.0,
            tool_name="test",
        )
        assert result.latency_ms == 0.0


class TestToolError:
    """Test ToolError exception."""

    def test_tool_error_message(self) -> None:
        """Test ToolError stores message."""
        err = ToolError("permission denied", tool_name="grep", category=ToolCategory.READ_ONLY)
        assert str(err) == "permission denied"
        assert err.tool_name == "grep"
        assert err.category == ToolCategory.READ_ONLY

    def test_tool_error_is_exception(self) -> None:
        """Test ToolError can be raised and caught."""
        with pytest.raises(ToolError):
            raise ToolError("timeout", tool_name="run_tests", category=ToolCategory.CONTROLLED)

    def test_tool_error_in_result(self) -> None:
        """Test ToolError can be converted to ToolResult."""
        err = ToolError("missing file", tool_name="read_file", category=ToolCategory.READ_ONLY)
        result = ToolResult(
            success=False,
            output="",
            error=str(err),
            latency_ms=0.0,
            tool_name=err.tool_name,
        )
        assert result.success is False
        assert "missing file" in result.error


class TestToolProtocol:
    """Test Tool protocol contract."""

    def test_protocol_has_execute(self) -> None:
        """Test Tool protocol defines execute method."""
        assert hasattr(Tool, "execute")

    def test_protocol_has_name(self) -> None:
        """Test Tool protocol defines name attribute."""
        assert "name" in Tool.__annotations__

    def test_protocol_has_category(self) -> None:
        """Test Tool protocol defines category attribute."""
        assert "category" in Tool.__annotations__

    def test_fake_tool_satisfies_protocol(self) -> None:
        """Test FakeTool satisfies Tool protocol via structural subtyping."""

        class FakeTool:
            name: str = "fake_tool"
            category: ToolCategory = ToolCategory.READ_ONLY

            def execute(self, **kwargs: Any) -> ToolResult:
                return ToolResult(
                    success=True,
                    output="fake output",
                    error=None,
                    latency_ms=0.0,
                    tool_name=self.name,
                )

        tool = FakeTool()
        assert tool.name == "fake_tool"
        assert tool.category == ToolCategory.READ_ONLY
        result = tool.execute(query="test")
        assert result.success is True


class TestToolRegistry:
    """Test ToolRegistry."""

    def _make_tool(
        self,
        name: str = "read_file",
        category: ToolCategory = ToolCategory.READ_ONLY,
    ) -> Tool:
        """Create a minimal fake tool for testing."""

        class _FakeTool:
            def __init__(self, n: str, c: ToolCategory) -> None:
                self.name = n
                self.category = c

            def execute(self, **kwargs: Any) -> ToolResult:
                return ToolResult(
                    success=True,
                    output="",
                    error=None,
                    latency_ms=0.0,
                    tool_name=self.name,
                )

        return _FakeTool(name, category)  # type: ignore[return-value]

    def test_register_and_get(self) -> None:
        """Test registering a tool and retrieving it."""
        registry = ToolRegistry()
        tool = self._make_tool("read_file", ToolCategory.READ_ONLY)
        registry.register(tool)
        assert registry.get("read_file") is tool

    def test_get_unknown_tool_returns_none(self) -> None:
        """Test getting an unregistered tool returns None."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_tools(self) -> None:
        """Test listing all registered tools."""
        registry = ToolRegistry()
        t1 = self._make_tool("read_file", ToolCategory.READ_ONLY)
        t2 = self._make_tool("grep", ToolCategory.READ_ONLY)
        registry.register(t1)
        registry.register(t2)
        tools = registry.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"read_file", "grep"}

    def test_register_duplicate_raises(self) -> None:
        """Test registering a tool with duplicate name raises ValueError."""
        registry = ToolRegistry()
        t1 = self._make_tool("read_file")
        t2 = self._make_tool("read_file")
        registry.register(t1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(t2)

    def test_is_allowed_registered_read_only(self) -> None:
        """Test registered read-only tool is allowed."""
        registry = ToolRegistry()
        tool = self._make_tool("read_file", ToolCategory.READ_ONLY)
        registry.register(tool)
        assert registry.is_allowed(tool) is True

    def test_is_allowed_unregistered(self) -> None:
        """Test unregistered tool is not allowed."""
        registry = ToolRegistry()
        tool = self._make_tool("read_file", ToolCategory.READ_ONLY)
        assert registry.is_allowed(tool) is False

    def test_get_by_category(self) -> None:
        """Test filtering tools by category."""
        registry = ToolRegistry()
        t1 = self._make_tool("read_file", ToolCategory.READ_ONLY)
        t2 = self._make_tool("grep", ToolCategory.READ_ONLY)
        t3 = self._make_tool("run_tests", ToolCategory.CONTROLLED)
        registry.register(t1)
        registry.register(t2)
        registry.register(t3)
        read_only = registry.get_by_category(ToolCategory.READ_ONLY)
        assert len(read_only) == 2
        controlled = registry.get_by_category(ToolCategory.CONTROLLED)
        assert len(controlled) == 1
        write = registry.get_by_category(ToolCategory.WRITE)
        assert len(write) == 0

    def test_register_multiple_categories(self) -> None:
        """Test tools from all categories can be registered."""
        registry = ToolRegistry()
        for cat in ToolCategory:
            tool = self._make_tool(f"tool_{cat.value}", cat)
            registry.register(tool)
        assert len(registry.list_tools()) == 4

    def test_empty_registry(self) -> None:
        """Test empty registry returns empty lists."""
        registry = ToolRegistry()
        assert registry.list_tools() == []
        assert registry.get_by_category(ToolCategory.READ_ONLY) == []
