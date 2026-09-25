"""Tests for ToyEvoCodeEnv - the real RL environment."""

import numpy as np
from pathlib import Path

from evocode.environment.evocode_env import EvoCodeEnv
from evocode.environment.toy_env import ToyEvoCodeEnv
from evocode.domain.actions import ActionType, ActionSpace
from evocode.domain.rewards import RewardConfig
from evocode.config.settings import ToolConfig
from evocode.tools.dispatcher import ToolDispatcher
from evocode.tools.read_file import ReadFileTool
from evocode.tools.grep_tool import GrepTool
from evocode.tools.run_tests import RunTestsTool
from evocode.tools.path_validator import PathValidator
from evocode.retrieval.retriever import FakeRetriever
from evocode.retrieval.chunker import CodeChunk


class TestToyEvoCodeEnvInheritance:
    """Test ToyEvoCodeEnv inherits from EvoCodeEnv."""

    def test_inherits_evo_code_env(self) -> None:
        """ToyEvoCodeEnv is subclass of EvoCodeEnv."""
        assert issubclass(ToyEvoCodeEnv, EvoCodeEnv)


class TestToyEvoCodeEnvBehavior:
    """Test ToyEvoCodeEnv behavior."""

    def _make_env(self, tmp_path: Path, max_steps: int = 10) -> ToyEvoCodeEnv:
        """Create a ToyEvoCodeEnv with all dependencies."""
        # Setup tools
        validator = PathValidator(sandbox_root=tmp_path, max_file_size_mb=5)
        tools = {
            "read_file": ReadFileTool(path_validator=validator, max_file_size_mb=5),
            "grep": GrepTool(path_validator=validator, max_results=50),
            "run_tests": RunTestsTool(sandbox_root=tmp_path, timeout_seconds=30),
        }
        dispatcher = ToolDispatcher(tools=tools)

        # Create toy repo
        (tmp_path / "main.py").write_text("def add(a, b):\n    return a + b\n")
        (tmp_path / "test_main.py").write_text("from main import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")

        # Retriever
        retriever = FakeRetriever(chunks=[
            CodeChunk(text="def add(a, b):", file_path="main.py", start_line=1, end_line=2, language="python", metadata={})
        ])

        # Configs
        tool_config = ToolConfig()
        reward_config = RewardConfig()
        action_space = ActionSpace.default()

        return ToyEvoCodeEnv(
            task_id="test_task",
            query="add two numbers",
            tool_config=tool_config,
            reward_config=reward_config,
            action_space=action_space,
            tool_dispatcher=dispatcher,
            retriever=retriever,
            max_steps=max_steps,
        )

    def test_reset_returns_observation(self, tmp_path: Path) -> None:
        """Reset returns 448-dim float32 array."""
        env = self._make_env(tmp_path)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (448,)
        assert obs.dtype == np.float32

    def test_reset_info_contains_metadata(self, tmp_path: Path) -> None:
        """Reset info contains step_count and task_id."""
        env = self._make_env(tmp_path)
        _, info = env.reset()
        assert info["step_count"] == 0
        assert info["task_id"] == "test_task"

    def test_step_returns_five_tuple(self, tmp_path: Path) -> None:
        """Step returns (obs, reward, terminated, truncated, info)."""
        env = self._make_env(tmp_path)
        env.reset()
        result = env.step(0)  # READ_FILE
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_read_file_action(self, tmp_path: Path) -> None:
        """READ_FILE dispatches to ReadFileTool."""
        env = self._make_env(tmp_path)
        env.reset()
        action_idx = env.domain_action_space.index(ActionType.READ_FILE)
        obs, reward, terminated, truncated, info = env.step(action_idx)
        assert not terminated
        assert "tool_result" in info or "latency_ms" in info

    def test_step_grep_action(self, tmp_path: Path) -> None:
        """GREP dispatches to GrepTool."""
        env = self._make_env(tmp_path)
        env.reset()
        action_idx = env.domain_action_space.index(ActionType.GREP)
        obs, reward, terminated, truncated, info = env.step(action_idx)
        assert not terminated

    def test_step_run_tests_action(self, tmp_path: Path) -> None:
        """RUN_TESTS dispatches to RunTestsTool."""
        env = self._make_env(tmp_path)
        env.reset()
        action_idx = env.domain_action_space.index(ActionType.RUN_TESTS)
        obs, reward, terminated, truncated, info = env.step(action_idx)
        assert not terminated

    def test_step_retrieve_action(self, tmp_path: Path) -> None:
        """RETRIEVE_* uses retriever."""
        env = self._make_env(tmp_path)
        env.reset()
        action_idx = env.domain_action_space.index(ActionType.RETRIEVE_SMALL)
        obs, reward, terminated, truncated, info = env.step(action_idx)
        assert not terminated

    def test_step_stop_terminates(self, tmp_path: Path) -> None:
        """STOP sets terminated=True."""
        env = self._make_env(tmp_path)
        env.reset()
        action_idx = env.domain_action_space.index(ActionType.STOP)
        obs, reward, terminated, truncated, info = env.step(action_idx)
        assert terminated

    def test_step_memory_action_is_supported(self, tmp_path: Path) -> None:
        """Memory actions are handled explicitly rather than crashing."""
        env = self._make_env(tmp_path)
        env.reset()
        action_idx = env.domain_action_space.index(ActionType.RECALL_SKILL)
        obs, reward, terminated, truncated, info = env.step(action_idx)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(info, dict)
        assert "memory" in str(info).lower() or "tool_result" in info

    def test_step_max_steps_truncates(self, tmp_path: Path) -> None:
        """Reaching max_steps sets truncated=True."""
        env = self._make_env(tmp_path, max_steps=2)
        env.reset()
        # Take non-STOP actions
        action_idx = env.domain_action_space.index(ActionType.READ_FILE)
        env.step(action_idx)  # step 1
        obs, reward, terminated, truncated, info = env.step(action_idx)  # step 2 = max
        assert truncated
        assert not terminated

    def test_step_before_reset_raises(self, tmp_path: Path) -> None:
        """RuntimeError if step before reset."""
        env = self._make_env(tmp_path)
        try:
            env.step(0)
            assert False, "Should have raised"
        except RuntimeError:
            pass

    def test_reward_computed_each_step(self, tmp_path: Path) -> None:
        """Reward is float, computed by RewardCalculator."""
        env = self._make_env(tmp_path)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(reward, float)

    def test_tool_cost_recorded_in_info(self, tmp_path: Path) -> None:
        """tool_cost in info dict."""
        env = self._make_env(tmp_path)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "tool_cost" in info

    def test_latency_recorded_in_info(self, tmp_path: Path) -> None:
        """latency_ms in info dict."""
        env = self._make_env(tmp_path)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "latency_ms" in info

    def test_episode_length_bounded(self, tmp_path: Path) -> None:
        """Episode doesn't exceed max_steps."""
        env = self._make_env(tmp_path, max_steps=3)
        env.reset()
        action_idx = env.domain_action_space.index(ActionType.READ_FILE)
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(action_idx)
            if terminated or truncated:
                break
        assert terminated or truncated

    def test_close_cleans_up(self, tmp_path: Path) -> None:
        """Close doesn't crash."""
        env = self._make_env(tmp_path)
        env.reset()
        env.close()
        # Should not raise

    def test_action_space_property(self, tmp_path: Path) -> None:
        """Returns gymnasium Discrete action space."""
        env = self._make_env(tmp_path)
        assert hasattr(env, "action_space")
        from gymnasium import spaces
        assert isinstance(env.action_space, spaces.Discrete)

    def test_invalid_action_returns_safety_penalty(self, tmp_path: Path) -> None:
        """Invalid action index yields safety penalty."""
        env = self._make_env(tmp_path)
        env.reset()
        # Use invalid index
        invalid_idx = env.action_space.n + 10
        obs, reward, terminated, truncated, info = env.step(invalid_idx)
        assert reward < 0  # safety penalty
        assert "safety" in str(info).lower() or info.get("safety_penalty", 0) > 0