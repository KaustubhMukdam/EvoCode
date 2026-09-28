"""Tests for PPO Trainer."""

import numpy as np
import pytest

from evocode.policy.mlp import MLPPolicy, PolicyConfig
from evocode.environment.toy_env import ToyEvoCodeEnv
from evocode.training import PPOTrainer, PPOConfig
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


def _make_env(tmp_path) -> ToyEvoCodeEnv:
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
        task_id="test_task",
        query="test query",
        tool_config=ToolConfig(),
        reward_config=RewardConfig(),
        action_space=ActionSpace.default(),
        tool_dispatcher=dispatcher,
        retriever=retriever,
        max_steps=10,
    )


class TestPPOConfig:
    """Test PPO configuration."""

    def test_default_config(self) -> None:
        """Default PPO config has sensible values."""
        cfg = PPOConfig()
        assert cfg.learning_rate == 3e-4
        assert cfg.gamma == 0.99
        assert cfg.gae_lambda == 0.95
        assert cfg.clip_range == 0.2
        assert cfg.batch_size == 64
        assert cfg.n_steps == 2048
        assert cfg.n_epochs == 10
        assert cfg.entropy_coef == 0.01
        assert cfg.value_coef == 0.5
        assert cfg.max_grad_norm == 0.5

    def test_custom_config(self) -> None:
        """Custom config values accepted."""
        cfg = PPOConfig(learning_rate=1e-3, gamma=0.95, batch_size=32)
        assert cfg.learning_rate == 1e-3
        assert cfg.gamma == 0.95
        assert cfg.batch_size == 32


class TestPPOTrainer:
    """Test PPOTrainer."""

    def test_creation(self, tmp_path) -> None:
        """Trainer can be created with policy and env."""
        policy = MLPPolicy(PolicyConfig())
        env = _make_env(tmp_path)
        trainer = PPOTrainer(policy=policy, env=env, config=PPOConfig())
        assert trainer.policy is policy
        assert trainer.env is env

    def test_collect_rollouts(self, tmp_path) -> None:
        """Can collect rollouts from environment."""
        policy = MLPPolicy(PolicyConfig())
        env = _make_env(tmp_path)
        trainer = PPOTrainer(policy=policy, env=env, config=PPOConfig(n_steps=100))
        rollouts = trainer.collect_rollouts()
        assert len(rollouts["observations"]) == 100
        assert len(rollouts["actions"]) == 100
        assert len(rollouts["rewards"]) == 100
        assert len(rollouts["values"]) == 100
        assert len(rollouts["log_probs"]) == 100
        assert len(rollouts["dones"]) == 100

    def test_compute_advantages(self, tmp_path) -> None:
        """GAE advantages computed correctly."""
        policy = MLPPolicy(PolicyConfig())
        env = _make_env(tmp_path)
        trainer = PPOTrainer(policy=policy, env=env, config=PPOConfig(n_steps=100))
        rollouts = trainer.collect_rollouts()
        advantages = trainer.compute_advantages(rollouts)
        assert advantages.shape == (100,)
        assert np.all(np.isfinite(advantages))

    def test_update_policy(self, tmp_path) -> None:
        """Policy update runs without error."""
        policy = MLPPolicy(PolicyConfig())
        env = _make_env(tmp_path)
        trainer = PPOTrainer(policy=policy, env=env, config=PPOConfig(n_steps=50, n_epochs=2, batch_size=16))
        rollouts = trainer.collect_rollouts()
        advantages = trainer.compute_advantages(rollouts)
        loss = trainer.update_policy(rollouts, advantages)
        assert "policy_loss" in loss
        assert "value_loss" in loss
        assert "entropy" in loss
        assert np.isfinite(loss["policy_loss"])
        assert np.isfinite(loss["value_loss"])

    def test_train_step(self, tmp_path) -> None:
        """Single train step (collect + update) works."""
        policy = MLPPolicy(PolicyConfig())
        env = _make_env(tmp_path)
        trainer = PPOTrainer(policy=policy, env=env, config=PPOConfig(n_steps=50, n_epochs=1, batch_size=16))
        metrics = trainer.train_step()
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "mean_reward" in metrics

    def test_save_checkpoint(self, tmp_path) -> None:
        """Checkpoint saves policy and optimizer state."""
        policy = MLPPolicy(PolicyConfig())
        env = _make_env(tmp_path)
        trainer = PPOTrainer(policy=policy, env=env, config=PPOConfig())
        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(path)
        assert path.exists()

    def test_load_checkpoint(self, tmp_path) -> None:
        """Checkpoint loads policy and optimizer state."""
        policy = MLPPolicy(PolicyConfig())
        env = _make_env(tmp_path)
        trainer = PPOTrainer(policy=policy, env=env, config=PPOConfig())
        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(path)

        # Create new trainer and load
        policy2 = MLPPolicy(PolicyConfig())
        env2 = _make_env(tmp_path)
        trainer2 = PPOTrainer(policy=policy2, env=env2, config=PPOConfig())
        trainer2.load_checkpoint(path)

        # Verify weights match
        obs = np.random.randn(1, 448).astype(np.float32)
        logits1, values1 = policy.predict(obs)
        logits2, values2 = policy2.predict(obs)
        np.testing.assert_allclose(logits1, logits2, rtol=1e-5)
        np.testing.assert_allclose(values1, values2, rtol=1e-5)


class TestPPOEvaluation:
    """Test evaluation against baselines."""

    def test_evaluate_policy(self, tmp_path) -> None:
        """Can evaluate policy and return metrics."""
        policy = MLPPolicy(PolicyConfig())
        env = _make_env(tmp_path)
        trainer = PPOTrainer(policy=policy, env=env, config=PPOConfig())
        metrics = trainer.evaluate(n_episodes=5)
        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        assert "success_rate" in metrics
        assert "mean_episode_length" in metrics

    def test_random_policy_baseline(self, tmp_path) -> None:
        """Random policy baseline for comparison."""
        from evocode.training import RandomPolicy
        env = _make_env(tmp_path)
        random_policy = RandomPolicy(ActionSpace.default())
        obs, _ = env.reset()
        for _ in range(10):
            action = random_policy.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break