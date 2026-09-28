"""Policy protocol and configuration for EvoCode RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class PolicyConfig:
    """Configuration for RL policy network."""

    observation_dim: int = 448
    action_dim: int = 11
    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = "relu"
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.activation not in ("relu", "tanh", "gelu"):
            raise ValueError(f"activation must be one of: relu, tanh, gelu; got {self.activation}")


@runtime_checkable
class Policy(Protocol):
    """Protocol for RL policies."""

    config: PolicyConfig

    def predict(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict action logits and state values.
        
        Args:
            obs: Observations of shape (batch, observation_dim)
        
        Returns:
            Tuple of (logits, values) where:
            - logits: (batch, action_dim) action logits
            - values: (batch,) state values
        """
        ...

    def predict_values(self, obs: np.ndarray) -> np.ndarray:
        """Predict state values only.
        
        Args:
            obs: Observations of shape (batch, observation_dim)
        
        Returns:
            Values of shape (batch,)
        """
        ...

    def save(self, path: Path) -> None:
        """Save policy to checkpoint."""
        ...

    @classmethod
    def load(cls, path: Path) -> Policy:
        """Load policy from checkpoint."""
        ...