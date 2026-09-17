"""Reward domain types for EvoCode."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RewardComponents:
    """Decomposed reward components for transparent reward computation.

    Each component is computed independently and weighted by RewardConfig.
    """

    quality: float = 0.0  # task success indicator (0/1 or graded)
    success: float = 0.0  # episode completion bonus
    efficiency: float = 0.0  # -tool_calls - retrieval_calls
    latency: float = 0.0  # -wall_time_ms
    tool_cost: float = 0.0  # weighted by tool type
    safety_penalty: float = 0.0  # invalid actions, path violations


@dataclass
class RewardConfig:
    """Configuration for reward weights.

    Weights are configurable to enable ablation studies.
    All weights must be non-negative.
    """

    w_quality: float = 1.0
    w_success: float = 1.0
    w_efficiency: float = 0.01
    w_latency: float = 0.001
    w_tool_cost: float = 0.05
    w_safety: float = 10.0

    def __post_init__(self) -> None:
        for field_name, value in [
            ("w_quality", self.w_quality),
            ("w_success", self.w_success),
            ("w_efficiency", self.w_efficiency),
            ("w_latency", self.w_latency),
            ("w_tool_cost", self.w_tool_cost),
            ("w_safety", self.w_safety),
        ]:
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")


class RewardCalculator(Protocol):
    """Protocol for computing aggregate reward from components."""

    def calculate(self, components: RewardComponents, config: RewardConfig) -> float: ...


class FakeRewardCalculator:
    """Deterministic reward calculator for testing."""

    def calculate(self, components: RewardComponents, config: RewardConfig) -> float:
        """Compute weighted sum of reward components."""
        return (
            config.w_quality * components.quality
            + config.w_success * components.success
            + config.w_efficiency * components.efficiency
            + config.w_latency * components.latency
            + config.w_tool_cost * components.tool_cost
            - config.w_safety * components.safety_penalty
        )
