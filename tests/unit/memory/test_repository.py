"""Tests for memory repository interfaces."""

from datetime import datetime, timezone

from evocode.domain.episodes import Episode
from evocode.domain.rewards import RewardComponents
from evocode.domain.skills import Skill
from evocode.memory.repository import (
    EpisodeRepository,
    FakeEpisodeRepository,
    FakeSkillRepository,
    SkillRepository,
)

import numpy as np


def _make_episode(ep_id: str = "ep-001", task_id: str = "task-001") -> Episode:
    """Create a minimal episode for testing."""
    return Episode(
        episode_id=ep_id,
        task_id=task_id,
        query="test query",
        transitions=[],
        final_outcome="done",
        total_reward=1.0,
        success=True,
        policy_version="v0",
        created_at=datetime.now(timezone.utc),
    )


def _make_skill(sk_id: str = "sk-001", name: str = "test skill") -> Skill:
    """Create a minimal skill for testing."""
    return Skill(
        skill_id=sk_id,
        name=name,
        description="test",
        action_sequence=[],
        applicable_conditions={},
        success_count=0,
        reuse_count=0,
        embedding=np.zeros(384, dtype=np.float32),
        source_episode_id="ep-001",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


class TestEpisodeRepositoryProtocol:
    """Test EpisodeRepository protocol contract."""

    def test_protocol_has_save(self) -> None:
        """Test protocol defines save method."""
        assert hasattr(EpisodeRepository, "save")

    def test_protocol_has_get(self) -> None:
        """Test protocol defines get method."""
        assert hasattr(EpisodeRepository, "get")

    def test_protocol_has_list_all(self) -> None:
        """Test protocol defines list_all method."""
        assert hasattr(EpisodeRepository, "list_all")


class TestFakeEpisodeRepository:
    """Test FakeEpisodeRepository implementation."""

    def test_save_and_get(self) -> None:
        """Test saving and retrieving an episode."""
        repo = FakeEpisodeRepository()
        ep = _make_episode("ep-001")
        repo.save(ep)
        assert repo.get("ep-001") is ep

    def test_get_missing_returns_none(self) -> None:
        """Test get returns None for missing episode."""
        repo = FakeEpisodeRepository()
        assert repo.get("nonexistent") is None

    def test_list_all_empty(self) -> None:
        """Test list_all returns empty list for empty repo."""
        repo = FakeEpisodeRepository()
        assert repo.list_all() == []

    def test_list_all_multiple(self) -> None:
        """Test list_all returns all saved episodes."""
        repo = FakeEpisodeRepository()
        ep1 = _make_episode("ep-001", "task-001")
        ep2 = _make_episode("ep-002", "task-002")
        repo.save(ep1)
        repo.save(ep2)
        episodes = repo.list_all()
        assert len(episodes) == 2
        ids = {e.episode_id for e in episodes}
        assert ids == {"ep-001", "ep-002"}

    def test_save_overwrites_existing(self) -> None:
        """Test saving same ID overwrites previous episode."""
        repo = FakeEpisodeRepository()
        ep1 = _make_episode("ep-001")
        ep1.success = False
        repo.save(ep1)
        ep2 = _make_episode("ep-001")
        ep2.success = True
        repo.save(ep2)
        assert repo.get("ep-001").success is True


class TestSkillRepositoryProtocol:
    """Test SkillRepository protocol contract."""

    def test_protocol_has_save(self) -> None:
        """Test protocol defines save method."""
        assert hasattr(SkillRepository, "save")

    def test_protocol_has_get(self) -> None:
        """Test protocol defines get method."""
        assert hasattr(SkillRepository, "get")

    def test_protocol_has_list_all(self) -> None:
        """Test protocol defines list_all method."""
        assert hasattr(SkillRepository, "list_all")

    def test_protocol_has_delete(self) -> None:
        """Test protocol defines delete method."""
        assert hasattr(SkillRepository, "delete")


class TestFakeSkillRepository:
    """Test FakeSkillRepository implementation."""

    def test_save_and_get(self) -> None:
        """Test saving and retrieving a skill."""
        repo = FakeSkillRepository()
        skill = _make_skill("sk-001")
        repo.save(skill)
        assert repo.get("sk-001") is skill

    def test_get_missing_returns_none(self) -> None:
        """Test get returns None for missing skill."""
        repo = FakeSkillRepository()
        assert repo.get("nonexistent") is None

    def test_list_all_empty(self) -> None:
        """Test list_all returns empty list for empty repo."""
        repo = FakeSkillRepository()
        assert repo.list_all() == []

    def test_list_all_multiple(self) -> None:
        """Test list_all returns all saved skills."""
        repo = FakeSkillRepository()
        s1 = _make_skill("sk-001", "skill A")
        s2 = _make_skill("sk-002", "skill B")
        repo.save(s1)
        repo.save(s2)
        skills = repo.list_all()
        assert len(skills) == 2
        ids = {s.skill_id for s in skills}
        assert ids == {"sk-001", "sk-002"}

    def test_delete_existing(self) -> None:
        """Test deleting an existing skill."""
        repo = FakeSkillRepository()
        skill = _make_skill("sk-001")
        repo.save(skill)
        repo.delete("sk-001")
        assert repo.get("sk-001") is None

    def test_delete_missing_no_error(self) -> None:
        """Test deleting a missing skill does not raise."""
        repo = FakeSkillRepository()
        repo.delete("nonexistent")  # should not raise

    def test_save_overwrites_existing(self) -> None:
        """Test saving same ID overwrites previous skill."""
        repo = FakeSkillRepository()
        s1 = _make_skill("sk-001", "old name")
        repo.save(s1)
        s2 = _make_skill("sk-001", "new name")
        repo.save(s2)
        assert repo.get("sk-001").name == "new name"
