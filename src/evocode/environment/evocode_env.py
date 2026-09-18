"""Environment protocol for EvoCode."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EvoCodeEnv(ABC):
    """Abstract base class for the EvoCode RL environment.

    Follows the Gymnasium API contract:
    - reset() -> (observation, info)
    - step(action) -> (observation, reward, terminated, truncated, info)
    - close()
    """

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment and return initial observation + info."""
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step: return (obs, reward, terminated, truncated, info)."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        ...


class FakeEvoCodeEnv(EvoCodeEnv):
    """Deterministic fake environment for testing."""

    def __init__(self, max_steps: int = 10) -> None:
        self.max_steps = max_steps
        self._step_count = 0
        self._reset_called = False

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self._step_count = 0
        self._reset_called = True
        rng = np.random.default_rng(seed)
        obs = rng.random(448, dtype=np.float32)
        return obs, {"step_count": 0}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self._reset_called:
            raise RuntimeError("Must call reset() before step()")
        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        obs = np.zeros(448, dtype=np.float32)
        return obs, 0.0, terminated, False, {"step_count": self._step_count, "action": action}

    def close(self) -> None:
        pass
