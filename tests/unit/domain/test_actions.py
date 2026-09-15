"""Tests for action domain."""

import pytest

from evocode.domain.actions import ActionSpace, ActionType


class TestActionType:
    """Test ActionType enum."""

    def test_all_actions_defined(self) -> None:
        """Test all expected actions are defined."""
        expected = {
            "RETRIEVE_SMALL",
            "RETRIEVE_MEDIUM",
            "RETRIEVE_LARGE",
            "READ_FILE",
            "GREP",
            "RUN_TESTS",
            "RECALL_SKILL",
            "SAVE_SKILL",
            "SUMMARIZE",
            "STOP",
        }
        actual = {a.name for a in ActionType}
        assert actual == expected

    def test_action_values_are_strings(self) -> None:
        """Test action values are strings."""
        for action in ActionType:
            assert isinstance(action.value, str)
            assert len(action.value) > 0

    def test_action_values_unique(self) -> None:
        """Test action values are unique."""
        values = [a.value for a in ActionType]
        assert len(values) == len(set(values))

    def test_retrieval_actions_exist(self) -> None:
        """Test retrieval actions exist."""
        assert ActionType.RETRIEVE_SMALL.value == "retrieve_small"
        assert ActionType.RETRIEVE_MEDIUM.value == "retrieve_medium"
        assert ActionType.RETRIEVE_LARGE.value == "retrieve_large"

    def test_tool_actions_exist(self) -> None:
        """Test tool actions exist."""
        assert ActionType.READ_FILE.value == "read_file"
        assert ActionType.GREP.value == "grep"
        assert ActionType.RUN_TESTS.value == "run_tests"

    def test_memory_actions_exist(self) -> None:
        """Test memory actions exist."""
        assert ActionType.RECALL_SKILL.value == "recall_skill"
        assert ActionType.SAVE_SKILL.value == "save_skill"

    def test_control_actions_exist(self) -> None:
        """Test control actions exist."""
        assert ActionType.SUMMARIZE.value == "summarize"
        assert ActionType.STOP.value == "stop"


class TestActionSpace:
    """Test ActionSpace class."""

    def test_default_action_space(self) -> None:
        """Test default action space includes all actions."""
        space = ActionSpace.default()
        assert len(space.actions) == 10
        assert set(space.actions) == set(ActionType)

    def test_action_space_from_config(self) -> None:
        """Test action space can be created from config list."""
        config_actions = ["retrieve_small", "read_file", "stop"]
        space = ActionSpace.from_config(config_actions)
        assert len(space.actions) == 3
        assert space.actions == (
            ActionType.RETRIEVE_SMALL,
            ActionType.READ_FILE,
            ActionType.STOP,
        )

    def test_action_space_from_config_invalid(self) -> None:
        """Test invalid action in config raises error."""
        with pytest.raises(ValueError, match="Invalid action"):
            ActionSpace.from_config(["retrieve_small", "invalid_action"])

    def test_action_space_empty_raises(self) -> None:
        """Test empty action space raises error."""
        with pytest.raises(ValueError, match="at least one action"):
            ActionSpace.from_config([])

    def test_action_space_contains(self) -> None:
        """Test action space membership check."""
        space = ActionSpace.default()
        assert ActionType.RETRIEVE_SMALL in space
        assert ActionType.STOP in space

    def test_action_space_index(self) -> None:
        """Test getting action index."""
        space = ActionSpace.default()
        idx = space.index(ActionType.RETRIEVE_SMALL)
        assert isinstance(idx, int)
        assert 0 <= idx < len(space.actions)

    def test_action_space_index_invalid(self) -> None:
        """Test index for invalid action raises error."""
        space = ActionSpace.from_config(["retrieve_small", "stop"])
        with pytest.raises(ValueError, match="not in action space"):
            space.index(ActionType.READ_FILE)

    def test_action_space_iteration(self) -> None:
        """Test action space is iterable."""
        space = ActionSpace.from_config(["retrieve_small", "stop"])
        actions = list(space)
        assert actions == [ActionType.RETRIEVE_SMALL, ActionType.STOP]

    def test_action_space_len(self) -> None:
        """Test action space length."""
        space = ActionSpace.from_config(["retrieve_small", "read_file", "stop"])
        assert len(space) == 3

    def test_action_space_repr(self) -> None:
        """Test action space string representation."""
        space = ActionSpace.default()
        repr_str = repr(space)
        assert "ActionSpace" in repr_str
        assert "10" in repr_str

    def test_action_space_equality(self) -> None:
        """Test action space equality."""
        space1 = ActionSpace.from_config(["retrieve_small", "stop"])
        space2 = ActionSpace.from_config(["retrieve_small", "stop"])
        space3 = ActionSpace.from_config(["stop", "retrieve_small"])
        assert space1 == space2
        assert space1 != space3  # order matters


class TestActionSpaceConfig:
    """Test ActionSpace configuration serialization."""

    def test_to_config(self) -> None:
        """Test converting action space to config list."""
        space = ActionSpace.from_config(["retrieve_small", "read_file", "stop"])
        config = space.to_config()
        assert config == ["retrieve_small", "read_file", "stop"]

    def test_roundtrip(self) -> None:
        """Test config -> space -> config roundtrip."""
        original = ["retrieve_medium", "grep", "run_tests", "stop"]
        space = ActionSpace.from_config(original)
        assert space.to_config() == original
