"""Episode domain types for EvoCode."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from evocode.domain.actions import ActionType
from evocode.domain.rewards import RewardComponents
from evocode.domain.states import Observation


@dataclass
class Transition:
    """A single step in an episode trajectory."""

    step_index: int
    observation: Observation
    action: ActionType
    observation_summary: str
    reward: RewardComponents
    latency_ms: float
    tool_cost: float


@dataclass
class Episode:
    """A complete trajectory from task start to termination."""

    episode_id: str
    task_id: str
    query: str
    transitions: list[Transition]
    final_outcome: str
    total_reward: float
    success: bool
    policy_version: str
    created_at: datetime
