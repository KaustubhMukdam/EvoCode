"""Tests for tool cost model."""

from evocode.domain.actions import ActionType
from evocode.tools.cost import ToolCostModel


class TestToolCostModel:
    """Test ToolCostModel assigns costs correctly."""

    def test_read_file_cost(self) -> None:
        """Read file has low cost."""
        model = ToolCostModel()
        cost = model.get_cost(ActionType.READ_FILE)
        assert 0 < cost <= 0.1

    def test_grep_cost(self) -> None:
        """Grep has medium cost."""
        model = ToolCostModel()
        cost = model.get_cost(ActionType.GREP)
        assert 0.05 < cost <= 0.2

    def test_run_tests_cost(self) -> None:
        """Run tests has high cost."""
        model = ToolCostModel()
        cost = model.get_cost(ActionType.RUN_TESTS)
        assert cost >= 0.5

    def test_retrieval_cost_scales_with_k(self) -> None:
        """Larger retrieval = higher cost."""
        model = ToolCostModel()
        cost_small = model.get_cost(ActionType.RETRIEVE_SMALL)
        cost_medium = model.get_cost(ActionType.RETRIEVE_MEDIUM)
        cost_large = model.get_cost(ActionType.RETRIEVE_LARGE)
        assert cost_small < cost_medium < cost_large

    def test_memory_action_cost(self) -> None:
        """Save/recall has cost."""
        model = ToolCostModel()
        cost_recall = model.get_cost(ActionType.RECALL_SKILL)
        cost_save = model.get_cost(ActionType.SAVE_SKILL)
        assert cost_recall > 0
        assert cost_save > 0

    def test_stop_cost_zero(self) -> None:
        """Stop has zero cost."""
        model = ToolCostModel()
        cost = model.get_cost(ActionType.STOP)
        assert cost == 0.0

    def test_total_cost_multiple_actions(self) -> None:
        """Costs sum correctly for multiple actions."""
        model = ToolCostModel()
        actions = [ActionType.READ_FILE, ActionType.GREP, ActionType.RUN_TESTS]
        total = sum(model.get_cost(a) for a in actions)
        assert total == model.get_cost(ActionType.READ_FILE) + model.get_cost(ActionType.GREP) + model.get_cost(ActionType.RUN_TESTS)