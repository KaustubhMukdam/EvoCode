"""Tests for Gymnasium API compatibility."""

import numpy as np
from pathlib import Path
from gymnasium import spaces

from evocode.environment.toy_env import ToyEvoCodeEnv
from evocode.domain.actions import ActionSpace
from evocode.domain.rewards import RewardConfig
from evocode.config.settings import ToolConfig
from evocode.tools.dispatcher import ToolDispatcher
from evocode.tools.read_file import ReadFileTool
from evocode.tools.grep_tool import GrepTool
from evocode.tools.run_tests import RunTestsTool
from evocode.tools.path_validator import PathValidator
from evocode.retrieval.retriever import FakeRetriever
from evocode.retrieval.chunker import CodeChunk


def _make_env(tmp_path: Path) -> ToyEvoCodeEnv:
    validator = PathValidator(sandbox_root=tmp_path, max_file_size_mb=5)
    tools = {
        "read_file": ReadFileTool(path_validator=validator, max_file_size_mb=5),
        "grep": GrepTool(path_validator=validator, max_results=50),
        "run_tests": RunTestsTool(sandbox_root=tmp_path, timeout_seconds=30),
    }
    dispatcher = ToolDispatcher(tools=tools)
    (tmp_path / "main.py").write_text("x = 1")
    retriever = FakeRetriever(chunks=[
        CodeChunk(text="x = 1", file_path="main.py", start_line=1, end_line=1, language="python", metadata={})
    ])
    return ToyEvoCodeEnv(
        task_id="test",
        query="test",
        tool_config=ToolConfig(),
        reward_config=RewardConfig(),
        action_space=ActionSpace.default(),
        tool_dispatcher=dispatcher,
        retriever=retriever,
    )


class TestGymnasiumCompatibility:
    """Test ToyEvoCodeEnv passes basic Gymnasium checks."""

    def test_observation_space_defined(self, tmp_path: Path) -> None:
        """env.observation_space returns a Box."""
        env = _make_env(tmp_path)
        assert hasattr(env, "observation_space")
        assert isinstance(env.observation_space, spaces.Box)

    def test_action_space_defined(self, tmp_path: Path) -> None:
        """env.action_space returns a Discrete."""
        env = _make_env(tmp_path)
        assert hasattr(env, "action_space")
        assert isinstance(env.action_space, spaces.Discrete)

    def test_observation_shape_matches(self, tmp_path: Path) -> None:
        """Reset obs shape matches observation_space.shape."""
        env = _make_env(tmp_path)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_step_obs_shape_matches(self, tmp_path: Path) -> None:
        """Step obs shape matches observation_space.shape."""
        env = _make_env(tmp_path)
        env.reset()
        obs, _, _, _, _ = env.step(0)
        assert obs.shape == env.observation_space.shape

    def test_action_in_range(self, tmp_path: Path) -> None:
        """All valid actions accepted."""
        env = _make_env(tmp_path)
        env.reset()
        for action in range(env.action_space.n):
            obs, _, _, _, _ = env.step(action)
            assert obs.shape == env.observation_space.shape

    def test_render_modes(self, tmp_path: Path) -> None:
        """render_modes property exists."""
        env = _make_env(tmp_path)
        assert hasattr(env, "render_modes")