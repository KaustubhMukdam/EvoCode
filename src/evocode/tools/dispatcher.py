"""Tool dispatcher - routes ActionType to Tool."""

from __future__ import annotations

from evocode.config.settings import ToolConfig
from evocode.domain.actions import ActionType
from evocode.tools.base import Tool, ToolResult


class ToolDispatcher:
    """Dispatches actions to registered tools."""

    def __init__(self, tools: dict[str, Tool], tool_config: ToolConfig | None = None) -> None:
        self._tools = tools
        self._tool_config = tool_config or ToolConfig()

    def dispatch(self, action: ActionType, **kwargs) -> ToolResult:
        """Dispatch an action to the appropriate tool."""
        import time
        start_time = time.perf_counter()

        tool_name = self._action_to_tool(action)
        if tool_name is None:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=False,
                output="",
                error=f"Action {action.value} is not a tool action",
                latency_ms=latency_ms,
                tool_name="dispatcher",
            )

        if not self._is_tool_allowed(action):
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=False,
                output="",
                error=f"Action {action.value} is not allowed by ToolConfig",
                latency_ms=latency_ms,
                tool_name=tool_name,
            )

        tool = self._tools.get(tool_name)
        if tool is None:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=False,
                output="",
                error=f"Tool {tool_name} not available",
                latency_ms=latency_ms,
                tool_name=tool_name,
            )

        # Prepare kwargs for the tool
        tool_kwargs = self._prepare_tool_kwargs(action, **kwargs)

        try:
            result = tool.execute(**tool_kwargs)
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution failed: {e}",
                latency_ms=latency_ms,
                tool_name=tool_name,
            )

        return result

    def _is_tool_allowed(self, action: ActionType) -> bool:
        """Check whether a tool action is enabled by configuration."""
        if action == ActionType.READ_FILE:
            return self._tool_config.allow_read
        if action == ActionType.GREP:
            return self._tool_config.allow_grep
        if action == ActionType.RUN_TESTS:
            return self._tool_config.allow_run_tests
        return True

    def _action_to_tool(self, action: ActionType) -> str | None:
        """Map ActionType to tool name."""
        if action == ActionType.READ_FILE:
            return "read_file"
        if action == ActionType.GREP:
            return "grep"
        if action == ActionType.RUN_TESTS:
            return "run_tests"
        # Retrieval, memory, control actions are not tools
        return None

    def _prepare_tool_kwargs(self, action: ActionType, **kwargs) -> dict:
        """Prepare kwargs for specific tool based on action."""
        # For now, pass through all kwargs
        # In future, could add defaults per action
        return kwargs