"""Tests for environment protocol."""

import numpy as np

from evocode.environment.evocode_env import EvoCodeEnv, FakeEvoCodeEnv


class TestEvoCodeEnvProtocol:
    """Test EvoCodeEnv abstract base class contract."""

    def test_has_reset_method(self) -> None:
        """Test EvoCodeEnv defines reset method."""
        assert hasattr(EvoCodeEnv, "reset")

    def test_has_step_method(self) -> None:
        """Test EvoCodeEnv defines step method."""
        assert hasattr(EvoCodeEnv, "step")

    def test_has_close_method(self) -> None:
        """Test EvoCodeEnv defines close method."""
        assert hasattr(EvoCodeEnv, "close")


class TestFakeEvoCodeEnv:
    """Test FakeEvoCodeEnv implementation."""

    def test_reset_returns_observation(self) -> None:
        """Test reset returns a numpy observation array."""
        env = FakeEvoCodeEnv()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_observation_shape(self) -> None:
        """Test reset observation has expected shape."""
        env = FakeEvoCodeEnv()
        obs, _ = env.reset()
        assert obs.shape == (448,)

    def test_reset_observation_dtype(self) -> None:
        """Test reset observation is float32."""
        env = FakeEvoCodeEnv()
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_step_returns_tuple(self) -> None:
        """Test step returns (obs, reward, terminated, truncated, info)."""
        env = FakeEvoCodeEnv()
        env.reset()
        result = env.step(0)
        assert len(result) == 5

    def test_step_observation_shape(self) -> None:
        """Test step observation has correct shape."""
        env = FakeEvoCodeEnv()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (448,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_increments_count(self) -> None:
        """Test step count increases."""
        env = FakeEvoCodeEnv()
        env.reset()
        _, _, _, _, info1 = env.step(0)
        _, _, _, _, info2 = env.step(0)
        assert info2["step_count"] > info1["step_count"]

    def test_env_reset_deterministic(self) -> None:
        """Test reset produces same observation given same seed."""
        env1 = FakeEvoCodeEnv()
        env2 = FakeEvoCodeEnv()
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_close_does_not_raise(self) -> None:
        """Test close completes without error."""
        env = FakeEvoCodeEnv()
        env.close()

    def test_max_steps_termination(self) -> None:
        """Test environment terminates after max_steps."""
        env = FakeEvoCodeEnv(max_steps=3)
        env.reset()
        for _ in range(2):
            obs, reward, terminated, truncated, info = env.step(0)
            assert not terminated
        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated

    def test_step_before_reset_raises(self) -> None:
        """Test step before reset raises an error."""
        env = FakeEvoCodeEnv()
        try:
            env.step(0)
            assert False, "Should have raised"
        except RuntimeError:
            pass
