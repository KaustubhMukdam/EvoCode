"""State domain types for EvoCode."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from evocode.domain.actions import ActionSpace, ActionType


@dataclass
class Observation:
    """Bounded numerical observation for RL policy.

    Fixed dimension: 448 dims (384 + 50 + 7 + 7).
    All values normalized to [0, 1] where applicable.
    """

    query_embedding: np.ndarray  # (384,) from all-MiniLM-L6-v2
    recent_action_history: np.ndarray  # (max_history, n_actions) one-hot
    step_count: float  # normalized [0, 1]
    retrieval_count: float  # normalized
    estimated_context_tokens: float  # normalized
    time_budget_remaining: float  # normalized [0, 1]
    token_budget_remaining: float  # normalized [0, 1]
    memory_candidate_score: float  # [0, 1] top skill similarity
    task_metadata: np.ndarray  # (n_categories,) one-hot

    def to_array(self) -> np.ndarray:
        """Flatten observation to 1D array for policy input."""
        return np.concatenate(
            [
                self.query_embedding.astype(np.float32),
                self.recent_action_history.flatten().astype(np.float32),
                np.array(
                    [
                        self.step_count,
                        self.retrieval_count,
                        self.estimated_context_tokens,
                        self.time_budget_remaining,
                        self.token_budget_remaining,
                        self.memory_candidate_score,
                    ],
                    dtype=np.float32,
                ),
                self.task_metadata.astype(np.float32),
            ]
        )


class StateBuilder(Protocol):
    """Protocol for building observations from environment state."""

    def build(
        self,
        query: str,
        action_history: list[ActionType],
        step_count: int,
        retrieval_count: int,
        context_tokens: int,
        time_budget: float,
        token_budget: int,
        memory_score: float,
        task_category: str,
    ) -> Observation: ...


# Configuration constants
DEFAULT_MAX_STEPS = 20
DEFAULT_MAX_HISTORY = 5
TASK_CATEGORIES = [
    "repo_understanding",
    "bug_localization",
    "debugging",
    "test_diagnosis",
    "refactoring",
    "documentation",
    "code_navigation",
]


class FakeStateBuilder:
    """Deterministic fake StateBuilder for testing.

    Uses a fixed embedding for reproducible tests.
    """

    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        self.max_steps = max_steps
        self.max_history = max_history
        self._fixed_embedding = np.arange(384, dtype=np.float32) / 384.0

    def build(
        self,
        query: str,  # noqa: ARG002
        action_history: list[ActionType],
        step_count: int,
        retrieval_count: int,
        context_tokens: int,
        time_budget: float,
        token_budget: int,
        memory_score: float,
        task_category: str,
    ) -> Observation:
        space = ActionSpace.default()
        n_actions = len(space.actions)
        history = np.zeros((self.max_history, n_actions), dtype=np.float32)

        # Fill from most recent (end of list)
        for i, action in enumerate(reversed(action_history[-self.max_history :])):
            idx = space.index(action)
            history[self.max_history - 1 - i, idx] = 1.0

        # Task category one-hot
        task_meta = np.zeros(len(TASK_CATEGORIES), dtype=np.float32)
        if task_category in TASK_CATEGORIES:
            task_meta[TASK_CATEGORIES.index(task_category)] = 1.0

        return Observation(
            query_embedding=self._fixed_embedding.copy(),
            recent_action_history=history,
            step_count=min(step_count / self.max_steps, 1.0),
            retrieval_count=min(retrieval_count / 10.0, 1.0),  # normalize by max 10 retrievals
            estimated_context_tokens=min(context_tokens / 8000.0, 1.0),
            time_budget_remaining=min(time_budget / 60.0, 1.0),  # normalize by 60s
            token_budget_remaining=min(token_budget / 8000.0, 1.0),
            memory_candidate_score=max(0.0, min(memory_score, 1.0)),
            task_metadata=task_meta,
        )
