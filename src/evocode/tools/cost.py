"""Tool cost model for efficiency reward."""

from __future__ import annotations

from evocode.domain.actions import ActionType


class ToolCostModel:
    """Assigns cost to tool actions for efficiency reward."""

    _BASE_COSTS = {
        ActionType.READ_FILE: 0.05,
        ActionType.GREP: 0.1,
        ActionType.RUN_TESTS: 1.0,
        ActionType.RETRIEVE_SMALL: 0.1,
        ActionType.RETRIEVE_MEDIUM: 0.2,
        ActionType.RETRIEVE_LARGE: 0.5,
        ActionType.RECALL_SKILL: 0.05,
        ActionType.SAVE_SKILL: 0.1,
        ActionType.SUMMARIZE: 0.05,
        ActionType.STOP: 0.0,
    }

    def get_cost(self, action: ActionType, **kwargs) -> float:
        """Get cost for an action."""
        return self._BASE_COSTS.get(action, 0.0)