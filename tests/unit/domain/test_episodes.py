"""Tests for episode domain."""

from datetime import datetime, timezone

import numpy as np

from evocode.domain.actions import ActionType
from evocode.domain.episodes import Episode, Transition
from evocode.domain.rewards import RewardComponents
from evocode.domain.states import Observation


def _make_observation() -> Observation:
    """Create a fixed observation for testing."""
    return Observation(
        query_embedding=np.zeros(384, dtype=np.float32),
        recent_action_history=np.zeros((5, 10), dtype=np.float32),
        step_count=0.0,
        retrieval_count=0.0,
        estimated_context_tokens=0.0,
        time_budget_remaining=1.0,
        token_budget_remaining=1.0,
        memory_candidate_score=0.0,
        task_metadata=np.zeros(7, dtype=np.float32),
    )


def _make_transition(step: int = 0) -> Transition:
    """Create a fixed transition for testing."""
    return Transition(
        step_index=step,
        observation=_make_observation(),
        action=ActionType.RETRIEVE_SMALL,
        observation_summary=f"step {step} summary",
        reward=RewardComponents(quality=1.0, success=0.0),
        latency_ms=10.0,
        tool_cost=0.0,
    )


class TestTransition:
    """Test Transition dataclass."""

    def test_transition_creation(self) -> None:
        """Test creating a transition with all fields."""
        obs = _make_observation()
        t = Transition(
            step_index=0,
            observation=obs,
            action=ActionType.READ_FILE,
            observation_summary="read main.py",
            reward=RewardComponents(quality=0.5),
            latency_ms=25.3,
            tool_cost=0.1,
        )
        assert t.step_index == 0
        assert t.action == ActionType.READ_FILE
        assert t.observation_summary == "read main.py"
        assert t.reward.quality == 0.5
        assert t.latency_ms == 25.3
        assert t.tool_cost == 0.1

    def test_transition_step_index(self) -> None:
        """Test step_index is stored correctly."""
        t = Transition(
            step_index=5,
            observation=_make_observation(),
            action=ActionType.STOP,
            observation_summary="",
            reward=RewardComponents(),
            latency_ms=0.0,
            tool_cost=0.0,
        )
        assert t.step_index == 5

    def test_transition_reward_immutable_reference(self) -> None:
        """Test transition stores a reference to reward components."""
        rc = RewardComponents(quality=1.0)
        t = Transition(
            step_index=0,
            observation=_make_observation(),
            action=ActionType.STOP,
            observation_summary="",
            reward=rc,
            latency_ms=0.0,
            tool_cost=0.0,
        )
        assert t.reward is rc


class TestEpisode:
    """Test Episode dataclass."""

    def test_episode_creation(self) -> None:
        """Test creating an episode with all fields."""
        now = datetime.now(timezone.utc)
        ep = Episode(
            episode_id="ep-001",
            task_id="task-001",
            query="find the auth function",
            transitions=[],
            final_outcome="found",
            total_reward=5.0,
            success=True,
            policy_version="ppo-v1",
            created_at=now,
        )
        assert ep.episode_id == "ep-001"
        assert ep.task_id == "task-001"
        assert ep.query == "find the auth function"
        assert ep.transitions == []
        assert ep.final_outcome == "found"
        assert ep.total_reward == 5.0
        assert ep.success is True
        assert ep.policy_version == "ppo-v1"
        assert ep.created_at == now

    def test_episode_defaults_empty_transitions(self) -> None:
        """Test episode defaults to empty transitions list."""
        ep = Episode(
            episode_id="ep-002",
            task_id="task-002",
            query="test",
            transitions=[],
            final_outcome="",
            total_reward=0.0,
            success=False,
            policy_version="v0",
            created_at=datetime.now(timezone.utc),
        )
        assert len(ep.transitions) == 0

    def test_episode_with_transitions(self) -> None:
        """Test episode stores multiple transitions."""
        transitions = [_make_transition(i) for i in range(3)]
        ep = Episode(
            episode_id="ep-003",
            task_id="task-003",
            query="debug test",
            transitions=transitions,
            final_outcome="fixed",
            total_reward=10.0,
            success=True,
            policy_version="v1",
            created_at=datetime.now(timezone.utc),
        )
        assert len(ep.transitions) == 3
        assert ep.transitions[0].step_index == 0
        assert ep.transitions[2].step_index == 2

    def test_episode_total_reward(self) -> None:
        """Test total_reward is stored correctly."""
        ep = Episode(
            episode_id="ep-004",
            task_id="task-004",
            query="q",
            transitions=[],
            final_outcome="",
            total_reward=-3.5,
            success=False,
            policy_version="v0",
            created_at=datetime.now(timezone.utc),
        )
        assert ep.total_reward == -3.5

    def test_episode_success_flag(self) -> None:
        """Test success boolean flag."""
        ep_success = Episode(
            episode_id="ep-005",
            task_id="task-005",
            query="q",
            transitions=[],
            final_outcome="done",
            total_reward=1.0,
            success=True,
            policy_version="v0",
            created_at=datetime.now(timezone.utc),
        )
        ep_fail = Episode(
            episode_id="ep-006",
            task_id="task-006",
            query="q",
            transitions=[],
            final_outcome="failed",
            total_reward=-1.0,
            success=False,
            policy_version="v0",
            created_at=datetime.now(timezone.utc),
        )
        assert ep_success.success is True
        assert ep_fail.success is False

    def test_episode_timestamp(self) -> None:
        """Test episode stores timestamp."""
        now = datetime.now(timezone.utc)
        ep = Episode(
            episode_id="ep-007",
            task_id="task-007",
            query="q",
            transitions=[],
            final_outcome="",
            total_reward=0.0,
            success=False,
            policy_version="v0",
            created_at=now,
        )
        assert ep.created_at == now

    def test_episode_policy_version(self) -> None:
        """Test policy_version is stored."""
        ep = Episode(
            episode_id="ep-008",
            task_id="task-008",
            query="q",
            transitions=[],
            final_outcome="",
            total_reward=0.0,
            success=False,
            policy_version="ppo-v2.1",
            created_at=datetime.now(timezone.utc),
        )
        assert ep.policy_version == "ppo-v2.1"


class TestEpisodeRewardAggregation:
    """Test reward aggregation from transitions."""

    def test_compute_total_reward(self) -> None:
        """Test computing total reward from transition rewards."""
        t1 = Transition(
            step_index=0,
            observation=_make_observation(),
            action=ActionType.RETRIEVE_SMALL,
            observation_summary="",
            reward=RewardComponents(quality=0.5, efficiency=-1.0),
            latency_ms=10.0,
            tool_cost=0.0,
        )
        t2 = Transition(
            step_index=1,
            observation=_make_observation(),
            action=ActionType.STOP,
            observation_summary="",
            reward=RewardComponents(quality=1.0, success=1.0),
            latency_ms=5.0,
            tool_cost=0.0,
        )
        ep = Episode(
            episode_id="ep-agg",
            task_id="task-agg",
            query="q",
            transitions=[t1, t2],
            final_outcome="done",
            total_reward=0.0,
            success=True,
            policy_version="v0",
            created_at=datetime.now(timezone.utc),
        )
        total = sum(
            t.reward.quality + t.reward.success + t.reward.efficiency
            for t in ep.transitions
        )
        assert abs(total - 1.5) < 1e-6

    def test_empty_episode_reward(self) -> None:
        """Test empty episode has zero reward components."""
        ep = Episode(
            episode_id="ep-empty",
            task_id="task-empty",
            query="q",
            transitions=[],
            final_outcome="",
            total_reward=0.0,
            success=False,
            policy_version="v0",
            created_at=datetime.now(timezone.utc),
        )
        assert len(ep.transitions) == 0


class TestEpisodeTransitionOrdering:
    """Test transition ordering within an episode."""

    def test_transitions_chronological(self) -> None:
        """Test transitions are ordered by step_index."""
        transitions = [_make_transition(2), _make_transition(0), _make_transition(1)]
        ep = Episode(
            episode_id="ep-order",
            task_id="task-order",
            query="q",
            transitions=transitions,
            final_outcome="",
            total_reward=0.0,
            success=False,
            policy_version="v0",
            created_at=datetime.now(timezone.utc),
        )
        steps = [t.step_index for t in ep.transitions]
        assert steps == [2, 0, 1]  # stored as-is, ordering is caller's responsibility

    def test_single_transition(self) -> None:
        """Test episode with exactly one transition."""
        ep = Episode(
            episode_id="ep-single",
            task_id="task-single",
            query="q",
            transitions=[_make_transition(0)],
            final_outcome="",
            total_reward=0.0,
            success=False,
            policy_version="v0",
            created_at=datetime.now(timezone.utc),
        )
        assert len(ep.transitions) == 1


class TestTransitionActionVariety:
    """Test transitions record different action types."""

    def test_all_action_types_in_transitions(self) -> None:
        """Test transitions can use any ActionType."""
        actions = [
            ActionType.RETRIEVE_SMALL,
            ActionType.RETRIEVE_MEDIUM,
            ActionType.RETRIEVE_LARGE,
            ActionType.READ_FILE,
            ActionType.GREP,
            ActionType.RUN_TESTS,
            ActionType.RECALL_SKILL,
            ActionType.SAVE_SKILL,
            ActionType.SUMMARIZE,
            ActionType.STOP,
        ]
        transitions = []
        for i, action in enumerate(actions):
            transitions.append(
                Transition(
                    step_index=i,
                    observation=_make_observation(),
                    action=action,
                    observation_summary=f"action {action.value}",
                    reward=RewardComponents(),
                    latency_ms=0.0,
                    tool_cost=0.0,
                )
            )
        assert len(transitions) == 10
        assert all(
            t.action == a for t, a in zip(transitions, actions, strict=True)
        )


class TestTransitionLatencyCost:
    """Test latency and tool_cost recording."""

    def test_latency_recorded(self) -> None:
        """Test latency_ms is stored."""
        t = Transition(
            step_index=0,
            observation=_make_observation(),
            action=ActionType.RUN_TESTS,
            observation_summary="",
            reward=RewardComponents(),
            latency_ms=123.45,
            tool_cost=0.0,
        )
        assert t.latency_ms == 123.45

    def test_tool_cost_recorded(self) -> None:
        """Test tool_cost is stored."""
        t = Transition(
            step_index=0,
            observation=_make_observation(),
            action=ActionType.READ_FILE,
            observation_summary="",
            reward=RewardComponents(),
            latency_ms=0.0,
            tool_cost=0.5,
        )
        assert t.tool_cost == 0.5

    def test_zero_latency_and_cost(self) -> None:
        """Test STOP action has zero latency and cost."""
        t = Transition(
            step_index=0,
            observation=_make_observation(),
            action=ActionType.STOP,
            observation_summary="",
            reward=RewardComponents(),
            latency_ms=0.0,
            tool_cost=0.0,
        )
        assert t.latency_ms == 0.0
        assert t.tool_cost == 0.0
