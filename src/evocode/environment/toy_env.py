"""ToyEvoCodeEnv - concrete RL environment implementation."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from evocode.environment.evocode_env import EvoCodeEnv
from evocode.domain.actions import ActionType, ActionSpace
from evocode.domain.rewards import RewardComponents, RewardConfig
from evocode.config.settings import ToolConfig
from evocode.tools.dispatcher import ToolDispatcher
from evocode.tools.cost import ToolCostModel
from evocode.domain.reward_calculator import WeightedRewardCalculator
from evocode.retrieval.retriever import Retriever
from evocode.retrieval.chunker import CodeChunk
from evocode.tools.base import ToolResult


class ToyEvoCodeEnv(EvoCodeEnv):
    """Toy EvoCode environment for development and testing.

    Wires ActionType -> ToolDispatcher -> RewardCalculator.
    Uses FakeRetriever and FakeStateBuilder by default.
    """

    def __init__(
        self,
        task_id: str,
        query: str,
        tool_config: ToolConfig,
        reward_config: RewardConfig,
        action_space: ActionSpace,
        tool_dispatcher: ToolDispatcher,
        retriever: Retriever,
        state_builder: Callable[[], np.ndarray] | None = None,
        max_steps: int = 20,
    ) -> None:
        self._task_id = task_id
        self._query = query
        self._tool_config = tool_config
        self._reward_config = reward_config
        self._action_space = action_space
        self._dispatcher = tool_dispatcher
        self._retriever = retriever
        self._cost_model = ToolCostModel()
        self._reward_calculator = WeightedRewardCalculator()
        self._max_steps = max_steps

        self._step_count = 0
        self._reset_called = False
        self._terminated = False
        self._truncated = False

        # Default state builder
        if state_builder is None:
            self._state_builder = lambda: np.random.default_rng(42).random(448, dtype=np.float32)
        else:
            self._state_builder = state_builder

    @property
    def action_space(self) -> spaces.Discrete:
        """Return gymnasium Discrete action space."""
        return spaces.Discrete(len(self._action_space))

    @property
    def domain_action_space(self) -> ActionSpace:
        """Return the domain ActionSpace for custom logic."""
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(448,),
            dtype=np.float32,
        )

    @property
    def render_modes(self) -> list[str] | None:
        return None

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment and return initial observation + info."""
        if seed is not None:
            np.random.seed(seed)

        self._step_count = 0
        self._reset_called = True
        self._terminated = False
        self._truncated = False

        obs = self._state_builder()
        info = {
            "step_count": self._step_count,
            "task_id": self._task_id,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step: return (obs, reward, terminated, truncated, info)."""
        if not self._reset_called:
            raise RuntimeError("Must call reset() before step()")

        if self._terminated or self._truncated:
            raise RuntimeError("Episode already finished")

        start_time = time.perf_counter()

        # Convert action index to ActionType
        if action >= len(self._action_space) or action < 0:
            # Invalid action - safety penalty
            components = RewardComponents(safety_penalty=1.0)
            reward = self._reward_calculator.calculate(components, self._reward_config)
            self._terminated = True
            obs = self._state_builder()
            info = {
                "step_count": self._step_count,
                "safety_penalty": 1.0,
                "latency_ms": (time.perf_counter() - start_time) * 1000,
                "tool_cost": 0.0,
            }
            return obs, reward, True, False, info

        action_type = self._action_space.actions[action]
        self._step_count += 1

        # Dispatch action
        tool_result = self._dispatch_action(action_type)

        # Compute reward components
        components = self._compute_reward_components(action_type, tool_result)
        reward = self._reward_calculator.calculate(components, self._reward_config)

        # Check termination
        if action_type == ActionType.STOP:
            self._terminated = True
        elif self._step_count >= self._max_steps:
            self._truncated = True

        # Build observation
        obs = self._state_builder()

        # Build info
        latency_ms = (time.perf_counter() - start_time) * 1000
        info = {
            "step_count": self._step_count,
            "latency_ms": latency_ms,
            "tool_cost": components.tool_cost,
            "tool_result": tool_result.output if tool_result.success else tool_result.error,
        }

        return obs, reward, self._terminated, self._truncated, info

    def _dispatch_action(self, action_type: ActionType) -> ToolResult:
        """Dispatch action to tool or retriever."""
        if action_type.is_tool:
            if action_type == ActionType.READ_FILE and not self._tool_config.allow_read:
                return ToolResult(
                    success=False,
                    output="",
                    error="Action READ_FILE is not allowed by ToolConfig",
                    latency_ms=0.0,
                    tool_name="dispatcher",
                )
            if action_type == ActionType.GREP and not self._tool_config.allow_grep:
                return ToolResult(
                    success=False,
                    output="",
                    error="Action GREP is not allowed by ToolConfig",
                    latency_ms=0.0,
                    tool_name="dispatcher",
                )
            if action_type == ActionType.RUN_TESTS and not self._tool_config.allow_run_tests:
                return ToolResult(
                    success=False,
                    output="",
                    error="Action RUN_TESTS is not allowed by ToolConfig",
                    latency_ms=0.0,
                    tool_name="dispatcher",
                )
            return self._dispatcher.dispatch(action_type)
        elif action_type.is_retrieval:
            return self._dispatch_retrieval(action_type)
        elif action_type.is_control:
            return self._dispatch_control(action_type)
        elif action_type.is_memory:
            return self._dispatch_memory(action_type)
        else:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown action type: {action_type}",
                latency_ms=0.0,
                tool_name="dispatcher",
            )

    def _dispatch_retrieval(self, action_type: ActionType) -> ToolResult:
        """Dispatch retrieval action."""
        import time
        start = time.perf_counter()
        top_k = 3 if action_type == ActionType.RETRIEVE_SMALL else 5 if action_type == ActionType.RETRIEVE_MEDIUM else 10
        chunks = self._retriever.retrieve(self._query, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        return ToolResult(
            success=True,
            output=f"Retrieved {len(chunks)} chunks",
            error=None,
            latency_ms=latency_ms,
            tool_name="retriever",
        )

    def _dispatch_control(self, action_type: ActionType) -> ToolResult:
        """Dispatch control action (STOP, SUMMARIZE)."""
        import time
        start = time.perf_counter()
        if action_type == ActionType.STOP:
            return ToolResult(
                success=True,
                output="Episode terminated by STOP action",
                error=None,
                latency_ms=(time.perf_counter() - start) * 1000,
                tool_name="control",
            )
        elif action_type == ActionType.SUMMARIZE:
            return ToolResult(
                success=True,
                output=f"Summary for: {self._query}",
                error=None,
                latency_ms=(time.perf_counter() - start) * 1000,
                tool_name="control",
            )
        return ToolResult(
            success=False,
            output="",
            error=f"Unknown control action: {action_type}",
            latency_ms=0.0,
            tool_name="control",
        )

    def _dispatch_memory(self, action_type: ActionType) -> ToolResult:
        """Handle memory actions in the toy environment.

        This is intentionally a lightweight placeholder rather than full persistence.
        It records the action without executing external side effects.
        """
        import time
        start = time.perf_counter()
        if action_type == ActionType.RECALL_SKILL:
            return ToolResult(
                success=True,
                output="Memory recall requested; toy environment records the action without executing a skill.",
                error=None,
                latency_ms=(time.perf_counter() - start) * 1000,
                tool_name="memory",
            )
        if action_type == ActionType.SAVE_SKILL:
            return ToolResult(
                success=True,
                output="Memory save requested; toy environment records the action without persisting it.",
                error=None,
                latency_ms=(time.perf_counter() - start) * 1000,
                tool_name="memory",
            )
        return ToolResult(
            success=False,
            output="",
            error=f"Unknown memory action: {action_type}",
            latency_ms=(time.perf_counter() - start) * 1000,
            tool_name="memory",
        )

    def _compute_reward_components(
        self, action_type: ActionType, tool_result: ToolResult
    ) -> RewardComponents:
        """Compute reward components for a step."""
        # Quality: positive if tool succeeded
        quality = 1.0 if tool_result.success else 0.0

        # Success: only on STOP with successful outcome (simplified)
        success = 0.0

        # Efficiency: negative cost per tool call
        tool_cost = self._cost_model.get_cost(action_type)
        efficiency = -tool_cost

        # Latency: negative wall time
        latency = -tool_result.latency_ms

        # Tool cost: separate component
        tool_cost_component = -tool_cost

        # Safety penalty: for invalid actions or tool errors
        safety_penalty = 0.0 if tool_result.success else 0.5

        return RewardComponents(
            quality=quality,
            success=success,
            efficiency=efficiency,
            latency=latency,
            tool_cost=tool_cost_component,
            safety_penalty=safety_penalty,
        )

    def close(self) -> None:
        """Clean up environment resources."""
        pass