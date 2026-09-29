"""MLP Policy implementation for EvoCode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from evocode.policy.base import Policy, PolicyConfig


class MLPPolicy(nn.Module, Policy):
    """Multi-layer perceptron policy with separate policy and value heads."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config

        # Set seed for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        # Build shared trunk
        layers = []
        in_dim = config.observation_dim
        activation = self._get_activation(config.activation)

        for _ in range(config.num_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(activation())
            in_dim = config.hidden_dim

        self.trunk = nn.Sequential(*layers)

        # Policy head: outputs logits over actions
        self.policy_head = nn.Linear(config.hidden_dim, config.action_dim)

        # Value head: outputs scalar state value
        self.value_head = nn.Linear(config.hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str) -> type[nn.Module]:
        if name == "relu":
            return nn.ReLU
        elif name == "tanh":
            return nn.Tanh
        elif name == "gelu":
            return nn.GELU
        else:
            raise ValueError(f"Unknown activation: {name}")

    def _init_weights(self) -> None:
        """Initialize weights with Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and values."""
        features = self.trunk(x)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict action logits and state values from numpy observations."""
        self.eval()
        x = torch.from_numpy(obs).float()
        logits, values = self.forward(x)
        return logits.numpy(), values.numpy()

    @torch.no_grad()
    def predict_values(self, obs: np.ndarray) -> np.ndarray:
        """Predict state values only."""
        self.eval()
        x = torch.from_numpy(obs).float()
        _, values = self.forward(x)
        return values.numpy()

    def save(self, path: Path) -> None:
        """Save policy state dict and config to checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "MLPPolicy":
        """Load policy from checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        policy = cls(config)
        policy.load_state_dict(checkpoint["state_dict"])
        return policy

    def get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """Get action probabilities (softmax over logits)."""
        logits, _ = self.predict(obs)
        return torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

    def sample_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample action from policy distribution."""
        logits, _ = self.predict(obs)
        if deterministic:
            return logits.argmax(axis=-1)
        probs = torch.softmax(torch.from_numpy(logits), dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().numpy()