"""Action domain types for EvoCode."""

from collections.abc import Iterator
from enum import Enum
from typing import Self


class ActionType(Enum):
    """Discrete action types for the RL policy.

    Coarse-grained actions with internal parameter resolution.
    This is an experimental variable - can be expanded later.
    """

    # Retrieval actions (different context sizes)
    RETRIEVE_SMALL = "retrieve_small"  # top-k=3
    RETRIEVE_MEDIUM = "retrieve_medium"  # top-k=5
    RETRIEVE_LARGE = "retrieve_large"  # top-k=10

    # Tool actions
    READ_FILE = "read_file"
    GREP = "grep"
    RUN_TESTS = "run_tests"

    # Memory actions
    RECALL_SKILL = "recall_skill"
    SAVE_SKILL = "save_skill"  # environment-internal

    # Control actions
    SUMMARIZE = "summarize"
    STOP = "stop"

    @property
    def is_retrieval(self) -> bool:
        """Check if action is a retrieval action."""
        return self in (
            ActionType.RETRIEVE_SMALL,
            ActionType.RETRIEVE_MEDIUM,
            ActionType.RETRIEVE_LARGE,
        )

    @property
    def is_tool(self) -> bool:
        """Check if action is a tool execution action."""
        return self in (
            ActionType.READ_FILE,
            ActionType.GREP,
            ActionType.RUN_TESTS,
        )

    @property
    def is_memory(self) -> bool:
        """Check if action is a memory action."""
        return self in (
            ActionType.RECALL_SKILL,
            ActionType.SAVE_SKILL,
        )

    @property
    def is_control(self) -> bool:
        """Check if action is a control flow action."""
        return self in (
            ActionType.SUMMARIZE,
            ActionType.STOP,
        )


class ActionSpace:
    """Configurable action space for the RL environment.

    Allows defining custom action subsets for ablation studies.
    """

    def __init__(self, actions: tuple[ActionType, ...]) -> None:
        if not actions:
            raise ValueError("Action space must have at least one action")
        self._actions = actions
        self._action_to_index = {action: i for i, action in enumerate(actions)}

    @classmethod
    def default(cls) -> Self:
        """Create default action space with all actions."""
        return cls(tuple(ActionType))

    @classmethod
    def from_config(cls, config: list[str]) -> Self:
        """Create action space from list of action value strings."""
        if not config:
            raise ValueError("Action space must have at least one action")

        actions = []
        for action_str in config:
            try:
                action = ActionType(action_str)
            except ValueError as e:
                raise ValueError(f"Invalid action: {action_str}") from e
            actions.append(action)

        return cls(tuple(actions))

    @property
    def actions(self) -> tuple[ActionType, ...]:
        """Get ordered tuple of actions."""
        return self._actions

    def __contains__(self, action: ActionType) -> bool:
        return action in self._action_to_index

    def index(self, action: ActionType) -> int:
        """Get index of action in space."""
        if action not in self._action_to_index:
            raise ValueError(f"Action {action} not in action space")
        return self._action_to_index[action]

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self) -> Iterator[ActionType]:
        return iter(self._actions)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActionSpace):
            return NotImplemented
        return self._actions == other._actions

    def __repr__(self) -> str:
        return f"ActionSpace(actions={len(self._actions)})"

    def to_config(self) -> list[str]:
        """Convert to list of action value strings for serialization."""
        return [action.value for action in self._actions]
