"""Tests for reward domain."""

import pytest

from evocode.domain.rewards import RewardCalculator, RewardComponents, RewardConfig


class TestRewardComponents:
    """Test RewardComponents dataclass."""

    def test_reward_components_creation(self) -> None:
        """Test creating reward components with all fields."""
        rc = RewardComponents(
            quality=1.0,
            success=1.0,
            efficiency=-0.5,
            latency=-0.1,
            tool_cost=-0.05,
            safety_penalty=0.0,
        )
        assert rc.quality == 1.0
        assert rc.success == 1.0
        assert rc.efficiency == -0.5
        assert rc.latency == -0.1
        assert rc.tool_cost == -0.05
        assert rc.safety_penalty == 0.0

    def test_reward_components_defaults(self) -> None:
        """Test reward components default to zero."""
        rc = RewardComponents()
        assert rc.quality == 0.0
        assert rc.success == 0.0
        assert rc.efficiency == 0.0
        assert rc.latency == 0.0
        assert rc.tool_cost == 0.0
        assert rc.safety_penalty == 0.0

    def test_reward_components_non_negative_quality(self) -> None:
        """Test quality can be negative (for failed tasks)."""
        rc = RewardComponents(quality=-1.0)
        assert rc.quality == -1.0


class TestRewardConfig:
    """Test RewardConfig dataclass."""

    def test_default_weights(self) -> None:
        """Test default reward weights match documented values."""
        config = RewardConfig()
        assert config.w_quality == 1.0
        assert config.w_success == 1.0
        assert config.w_efficiency == 0.01
        assert config.w_latency == 0.001
        assert config.w_tool_cost == 0.05
        assert config.w_safety == 10.0

    def test_custom_weights(self) -> None:
        """Test custom weights can be set."""
        config = RewardConfig(
            w_quality=2.0,
            w_success=0.5,
            w_efficiency=0.02,
            w_latency=0.005,
            w_tool_cost=0.1,
            w_safety=5.0,
        )
        assert config.w_quality == 2.0
        assert config.w_success == 0.5
        assert config.w_efficiency == 0.02
        assert config.w_latency == 0.005
        assert config.w_tool_cost == 0.1
        assert config.w_safety == 5.0

    def test_weights_non_negative(self) -> None:
        """Test all weights must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            RewardConfig(w_quality=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            RewardConfig(w_success=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            RewardConfig(w_efficiency=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            RewardConfig(w_latency=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            RewardConfig(w_tool_cost=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            RewardConfig(w_safety=-1.0)


class TestRewardCalculatorProtocol:
    """Test RewardCalculator protocol."""

    def test_protocol_exists(self) -> None:
        """Test RewardCalculator protocol is defined."""
        assert hasattr(RewardCalculator, "calculate")
        assert callable(RewardCalculator.calculate)

    def test_protocol_signature(self) -> None:
        """Test RewardCalculator.calculate has expected signature."""
        import inspect

        sig = inspect.signature(RewardCalculator.calculate)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "components" in params
        assert "config" in params

    def test_protocol_returns_float(self) -> None:
        """Test protocol specifies float return type."""
        import inspect

        sig = inspect.signature(RewardCalculator.calculate)
        assert sig.return_annotation is float


class TestFakeRewardCalculator:
    """Test FakeRewardCalculator implementation."""

    def test_calculate_basic(self) -> None:
        """Test basic reward calculation."""
        from evocode.domain.rewards import FakeRewardCalculator

        calculator = FakeRewardCalculator()
        components = RewardComponents(
            quality=1.0,
            success=1.0,
            efficiency=-1.0,
            latency=-100.0,
            tool_cost=-1.0,
            safety_penalty=0.0,
        )
        config = RewardConfig()
        total = calculator.calculate(components, config)
        expected = (
            1.0 * 1.0 + 1.0 * 1.0 + 0.01 * (-1.0) + 0.001 * (-100.0) + 0.05 * (-1.0) + 10.0 * 0.0
        )
        assert abs(total - expected) < 1e-6

    def test_calculate_with_safety_penalty(self) -> None:
        """Test reward calculation with safety penalty."""
        from evocode.domain.rewards import FakeRewardCalculator

        calculator = FakeRewardCalculator()
        components = RewardComponents(
            quality=1.0,
            success=1.0,
            efficiency=0.0,
            latency=0.0,
            tool_cost=0.0,
            safety_penalty=1.0,
        )
        config = RewardConfig()
        total = calculator.calculate(components, config)
        # 1.0 + 1.0 - 10.0 = -8.0
        assert total == -8.0

    def test_calculate_zero_weights(self) -> None:
        """Test reward calculation with zero weights."""
        from evocode.domain.rewards import FakeRewardCalculator

        calculator = FakeRewardCalculator()
        components = RewardComponents(
            quality=1.0,
            success=1.0,
            efficiency=-1.0,
            latency=-100.0,
            tool_cost=-1.0,
            safety_penalty=1.0,
        )
        config = RewardConfig(
            w_quality=0.0,
            w_success=0.0,
            w_efficiency=0.0,
            w_latency=0.0,
            w_tool_cost=0.0,
            w_safety=0.0,
        )
        total = calculator.calculate(components, config)
        assert total == 0.0

    def test_calculate_individual_contributions(self) -> None:
        """Test individual weight contributions."""
        from evocode.domain.rewards import FakeRewardCalculator

        calculator = FakeRewardCalculator()
        components = RewardComponents(
            quality=1.0,
            success=0.0,
            efficiency=0.0,
            latency=0.0,
            tool_cost=0.0,
            safety_penalty=0.0,
        )
        config = RewardConfig(
            w_quality=2.0,
            w_success=0.0,
            w_efficiency=0.0,
            w_latency=0.0,
            w_tool_cost=0.0,
            w_safety=0.0,
        )
        total = calculator.calculate(components, config)
        assert total == 2.0

    def test_calculate_immutable(self) -> None:
        """Test calculator doesn't modify inputs."""
        from evocode.domain.rewards import FakeRewardCalculator

        calculator = FakeRewardCalculator()
        components = RewardComponents(quality=1.0, success=1.0)
        config = RewardConfig()
        original_quality = components.quality
        calculator.calculate(components, config)
        assert components.quality == original_quality


class TestRewardComponentsArithmetic:
    """Test arithmetic operations on reward components."""

    def test_addition(self) -> None:
        """Test adding two reward components."""
        rc1 = RewardComponents(quality=1.0, efficiency=-0.5)
        rc2 = RewardComponents(success=1.0, latency=-0.1)
        total = RewardComponents(
            quality=rc1.quality + rc2.quality,
            success=rc1.success + rc2.success,
            efficiency=rc1.efficiency + rc2.efficiency,
            latency=rc1.latency + rc2.latency,
            tool_cost=rc1.tool_cost + rc2.tool_cost,
            safety_penalty=rc1.safety_penalty + rc2.safety_penalty,
        )
        assert total.quality == 1.0
        assert total.success == 1.0
        assert total.efficiency == -0.5
        assert total.latency == -0.1

    def test_scalar_multiplication(self) -> None:
        """Test multiplying reward components by scalar."""
        rc = RewardComponents(quality=2.0, efficiency=-1.0)
        scaled = RewardComponents(
            quality=rc.quality * 0.5,
            success=rc.success * 0.5,
            efficiency=rc.efficiency * 0.5,
            latency=rc.latency * 0.5,
            tool_cost=rc.tool_cost * 0.5,
            safety_penalty=rc.safety_penalty * 0.5,
        )
        assert scaled.quality == 1.0
        assert scaled.efficiency == -0.5


class TestRewardCalculatorConfig:
    """Test RewardCalculator with different configurations."""

    def test_ablation_study(self) -> None:
        """Test reward ablation by zeroing weights."""
        from evocode.domain.rewards import FakeRewardCalculator

        calculator = FakeRewardCalculator()
        components = RewardComponents(
            quality=1.0,
            success=1.0,
            efficiency=-5.0,
            latency=-1000.0,
            tool_cost=-10.0,
            safety_penalty=0.0,
        )

        # Full reward
        full_config = RewardConfig()
        full_reward = calculator.calculate(components, full_config)

        # Ablation: remove efficiency
        no_efficiency = RewardConfig(
            w_efficiency=0.0,
            w_quality=1.0,
            w_success=1.0,
            w_latency=0.001,
            w_tool_cost=0.05,
            w_safety=10.0,
        )
        reward_no_eff = calculator.calculate(components, no_efficiency)
        assert reward_no_eff > full_reward  # Less negative

        # Ablation: remove latency
        no_latency = RewardConfig(
            w_latency=0.0,
            w_quality=1.0,
            w_success=1.0,
            w_efficiency=0.01,
            w_tool_cost=0.05,
            w_safety=10.0,
        )
        reward_no_lat = calculator.calculate(components, no_latency)
        assert reward_no_lat > full_reward


class TestRewardComponentsRepr:
    """Test reward components string representation."""

    def test_repr(self) -> None:
        """Test RewardComponents has readable repr."""
        rc = RewardComponents(quality=1.0, success=1.0)
        repr_str = repr(rc)
        assert "RewardComponents" in repr_str
        assert "quality=1.0" in repr_str
        assert "success=1.0" in repr_str
