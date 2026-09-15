"""Tests for state domain."""

import numpy as np

from evocode.domain.actions import ActionSpace, ActionType
from evocode.domain.states import Observation, StateBuilder


class TestObservation:
    """Test Observation dataclass."""

    def test_observation_creation(self) -> None:
        """Test creating an observation with all fields."""
        obs = Observation(
            query_embedding=np.zeros(384),
            recent_action_history=np.zeros((5, 10)),
            step_count=0.5,
            retrieval_count=0.2,
            estimated_context_tokens=0.3,
            time_budget_remaining=1.0,
            token_budget_remaining=0.8,
            memory_candidate_score=0.9,
            task_metadata=np.zeros(5),
        )
        assert obs is not None
        assert obs.query_embedding.shape == (384,)
        assert obs.recent_action_history.shape == (5, 10)

    def test_observation_to_array(self) -> None:
        """Test converting observation to flat numpy array."""
        obs = Observation(
            query_embedding=np.ones(384) * 0.1,
            recent_action_history=np.zeros((5, 10)),
            step_count=0.5,
            retrieval_count=0.2,
            estimated_context_tokens=0.3,
            time_budget_remaining=1.0,
            token_budget_remaining=0.8,
            memory_candidate_score=0.9,
            task_metadata=np.ones(7) * 0.05,  # 7 categories
        )
        arr = obs.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1
        # 384 + (5*10) + 6 + 7 = 384 + 50 + 6 + 7 = 447
        assert arr.shape[0] == 447

    def test_observation_bounds(self) -> None:
        """Test observation values are within expected bounds."""
        obs = Observation(
            query_embedding=np.random.randn(384).astype(np.float32),
            recent_action_history=np.zeros((5, 10)),
            step_count=0.0,
            retrieval_count=0.0,
            estimated_context_tokens=0.0,
            time_budget_remaining=0.0,
            token_budget_remaining=0.0,
            memory_candidate_score=0.0,
            task_metadata=np.zeros(5),
        )
        arr = obs.to_array()
        # Normalized fields should be in [0, 1]
        assert np.all(arr[-7:] >= 0.0)
        assert np.all(arr[-7:] <= 1.0)

    def test_observation_dtype(self) -> None:
        """Test observation array dtype is float32."""
        obs = Observation(
            query_embedding=np.zeros(384, dtype=np.float32),
            recent_action_history=np.zeros((5, 10), dtype=np.float32),
            step_count=0.5,
            retrieval_count=0.2,
            estimated_context_tokens=0.3,
            time_budget_remaining=1.0,
            token_budget_remaining=0.8,
            memory_candidate_score=0.9,
            task_metadata=np.zeros(5, dtype=np.float32),
        )
        arr = obs.to_array()
        assert arr.dtype == np.float32


class TestStateBuilder:
    """Test StateBuilder protocol."""

    def test_state_builder_protocol_exists(self) -> None:
        """Test StateBuilder protocol is defined."""
        assert hasattr(StateBuilder, "build")
        assert callable(StateBuilder.build)

    def test_state_builder_build_signature(self) -> None:
        """Test StateBuilder.build has expected signature."""
        import inspect

        sig = inspect.signature(StateBuilder.build)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "query" in params
        assert "action_history" in params
        assert "step_count" in params
        assert "retrieval_count" in params
        assert "context_tokens" in params
        assert "time_budget" in params
        assert "token_budget" in params
        assert "memory_score" in params
        assert "task_category" in params

    def test_fake_state_builder(self) -> None:
        """Test a fake implementation of StateBuilder works."""
        from evocode.domain.states import FakeStateBuilder

        builder = FakeStateBuilder()
        obs = builder.build(
            query="test query",
            action_history=[ActionType.RETRIEVE_SMALL, ActionType.READ_FILE],
            step_count=2,
            retrieval_count=1,
            context_tokens=100,
            time_budget=30.0,
            token_budget=4000,
            memory_score=0.5,
            task_category="bug_localization",
        )
        assert isinstance(obs, Observation)
        assert obs.query_embedding.shape == (384,)
        assert obs.recent_action_history.shape == (5, 10)
        assert obs.step_count == 2 / 20  # normalized by max_steps=20


class TestStateBuilderFake:
    """Test FakeStateBuilder implementation details."""

    def test_fake_builder_history_padding(self) -> None:
        """Test action history is padded/truncated to max_history."""
        from evocode.domain.states import FakeStateBuilder

        builder = FakeStateBuilder()
        space = ActionSpace.default()

        # Short history - should pad
        obs = builder.build(
            query="test",
            action_history=[ActionType.RETRIEVE_SMALL],
            step_count=1,
            retrieval_count=0,
            context_tokens=0,
            time_budget=30.0,
            token_budget=4000,
            memory_score=0.0,
            task_category="repo_understanding",
        )
        assert obs.recent_action_history.shape == (5, 10)
        # Most recent action should be at the last row (index 4 for max_history=5)
        assert obs.recent_action_history[4, space.index(ActionType.RETRIEVE_SMALL)] == 1.0

        # Long history - should truncate to last max_history
        long_history = [ActionType.RETRIEVE_SMALL] * 10
        obs2 = builder.build(
            query="test",
            action_history=long_history,
            step_count=10,
            retrieval_count=0,
            context_tokens=0,
            time_budget=30.0,
            token_budget=4000,
            memory_score=0.0,
            task_category="repo_understanding",
        )
        assert obs2.recent_action_history.shape == (5, 10)

    def test_fake_builder_task_categories(self) -> None:
        """Test task category one-hot encoding."""
        from evocode.domain.states import FakeStateBuilder

        builder = FakeStateBuilder()
        categories = [
            "repo_understanding",
            "bug_localization",
            "debugging",
            "test_diagnosis",
            "refactoring",
            "documentation",
            "code_navigation",
        ]
        for i, cat in enumerate(categories):
            obs = builder.build(
                query="test",
                action_history=[],
                step_count=0,
                retrieval_count=0,
                context_tokens=0,
                time_budget=30.0,
                token_budget=4000,
                memory_score=0.0,
                task_category=cat,
            )
            assert obs.task_metadata.shape == (7,)
            assert obs.task_metadata[i] == 1.0

    def test_fake_builder_unknown_category(self) -> None:
        """Test unknown task category defaults to zeros."""
        from evocode.domain.states import FakeStateBuilder

        builder = FakeStateBuilder()
        obs = builder.build(
            query="test",
            action_history=[],
            step_count=0,
            retrieval_count=0,
            context_tokens=0,
            time_budget=30.0,
            token_budget=4000,
            memory_score=0.0,
            task_category="unknown_category",
        )
        assert obs.task_metadata.shape == (7,)
        assert np.all(obs.task_metadata == 0.0)


class TestStateBuilderConfig:
    """Test StateBuilder configuration."""

    def test_builder_respects_max_steps(self) -> None:
        """Test step_count normalization uses max_steps."""
        from evocode.domain.states import FakeStateBuilder

        builder = FakeStateBuilder(max_steps=10)
        obs = builder.build(
            query="test",
            action_history=[],
            step_count=5,
            retrieval_count=0,
            context_tokens=0,
            time_budget=30.0,
            token_budget=4000,
            memory_score=0.0,
            task_category="repo_understanding",
        )
        assert obs.step_count == 0.5  # 5/10

    def test_builder_respects_max_history(self) -> None:
        """Test action history uses max_history."""
        from evocode.domain.states import FakeStateBuilder

        builder = FakeStateBuilder(max_history=3)
        obs = builder.build(
            query="test",
            action_history=[ActionType.RETRIEVE_SMALL, ActionType.READ_FILE],
            step_count=1,
            retrieval_count=0,
            context_tokens=0,
            time_budget=30.0,
            token_budget=4000,
            memory_score=0.0,
            task_category="repo_understanding",
        )
        assert obs.recent_action_history.shape == (3, 10)
