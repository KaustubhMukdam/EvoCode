"""Tool registry for allowlisted tool management."""

from __future__ import annotations

from evocode.tools.base import Tool, ToolCategory


class ToolRegistry:
    """Registry of allowlisted tools.

    Tools must be registered before execution. The registry provides
    lookup by name and filtering by safety category.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError if name already registered."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name, or None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def is_allowed(self, tool: Tool) -> bool:
        """Check if a tool is registered (and therefore allowed)."""
        return tool.name in self._tools

    def get_by_category(self, category: ToolCategory) -> list[Tool]:
        """Get all registered tools in a given category."""
        return [t for t in self._tools.values() if t.category == category]
