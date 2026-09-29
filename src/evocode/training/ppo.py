"""PPO Configuration for EvoCode."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 10
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not (0 < self.gamma <= 1):
            raise ValueError("gamma must be in (0, 1]")
        if not (0 <= self.gae_lambda <= 1):
            raise ValueError("gae_lambda must be in [0, 1]")
        if not (0 < self.clip_range <= 1):
            raise ValueError("clip_range must be in (0, 1]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive")
        if self.entropy_coef < 0:
            raise ValueError("entropy_coef must be non-negative")
        if self.value_coef < 0:
            raise ValueError("value_coef must be non-negative")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be non-negative")