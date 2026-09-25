"""Tests for weighted reward calculator."""

from evocode.domain.rewards import RewardCalculator, RewardComponents, RewardConfig
from evocode.domain.reward_calculator import WeightedRewardCalculator


class TestWeightedRewardCalculatorProtocol:
    """Test WeightedRewardCalculator satisfies RewardCalculator protocol."""

    def test_satisfies_protocol(self) -> None:
        """WeightedRewardCalculator has the required calculate method."""
        calc = WeightedRewardCalculator()
        assert hasattr(calc, "calculate")
        assert callable(calc.calculate)


class TestWeightedRewardCalculatorBehavior:
    """Test WeightedRewardCalculator behavior."""

    def test_basic_weighted_sum(self) -> None:
        """Matches expected formula."""
        calc = WeightedRewardCalculator()
        components = RewardComponents(
            quality=1.0,
            success=1.0,
            efficiency=-2.0,
            latency=-1000.0,
            tool_cost=-0.5,
            safety_penalty=0.0,
        )
        config = RewardConfig(
            w_quality=1.0,
            w_success=1.0,
            w_efficiency=0.01,
            w_latency=0.001,
            w_tool_cost=0.05,
            w_safety=10.0,
        )
        result = calc.calculate(components, config)
        expected = 1.0 + 1.0 + 0.01 * (-2.0) + 0.001 * (-1000.0) + 0.05 * (-0.5)
        assert abs(result - expected) < 1e-6

    def test_safety_penalty_dominates(self) -> None:
        """Large safety penalty yields negative reward."""
        calc = WeightedRewardCalculator()
        components = RewardComponents(
            quality=1.0,
            success=1.0,
            efficiency=0.0,
            latency=0.0,
            tool_cost=0.0,
            safety_penalty=1.0,
        )
        config = RewardConfig(w_safety=100.0, w_quality=1.0, w_success=1.0)
        result = calc.calculate(components, config)
        assert result < 0

    def test_zero_weights(self) -> None:
        """All zero weights yields zero reward."""
        calc = WeightedRewardCalculator()
        components = RewardComponents(quality=5.0, success=5.0, efficiency=5.0)
        config = RewardConfig(
            w_quality=0.0, w_success=0.0, w_efficiency=0.0,
            w_latency=0.0, w_tool_cost=0.0, w_safety=0.0,
        )
        result = calc.calculate(components, config)
        assert result == 0.0

    def test_single_component_isolation(self) -> None:
        """Isolating one weight works."""
        calc = WeightedRewardCalculator()
        components = RewardComponents(quality=2.0)
        config = RewardConfig(w_quality=3.0, w_success=0.0, w_efficiency=0.0)
        result = calc.calculate(components, config)
        assert result == 6.0

    def test_efficiency_negative_contribution(self) -> None:
        """Efficiency is negative (fewer tools = better)."""
        calc = WeightedRewardCalculator()
        components = RewardComponents(efficiency=-10.0)  # used 10 tools
        config = RewardConfig(w_efficiency=0.01)
        result = calc.calculate(components, config)
        assert result == -0.1

    def test_latency_negative_contribution(self) -> None:
        """Latency is negative (faster = better)."""
        calc = WeightedRewardCalculator()
        components = RewardComponents(latency=-500.0)  # 500ms
        config = RewardConfig(w_latency=0.001)
        result = calc.calculate(components, config)
        assert result == -0.5