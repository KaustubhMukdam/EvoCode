"""Weighted reward calculator."""

from __future__ import annotations

from evocode.domain.rewards import RewardCalculator, RewardComponents, RewardConfig


class WeightedRewardCalculator:
    """Computes aggregate reward as weighted sum of components."""

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