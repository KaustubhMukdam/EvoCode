"""Skill domain types for EvoCode."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from evocode.domain.actions import ActionType


@dataclass
class Skill:
    """A reusable strategy derived from a successful episode trajectory."""

    skill_id: str
    name: str
    description: str
    action_sequence: list[ActionType]
    applicable_conditions: dict
    success_count: int
    reuse_count: int
    embedding: np.ndarray
    source_episode_id: str
    created_at: datetime
    updated_at: datetime
