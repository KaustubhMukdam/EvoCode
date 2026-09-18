"""Tests for skill domain."""

from datetime import datetime, timezone

import numpy as np

from evocode.domain.actions import ActionType
from evocode.domain.skills import Skill


class TestSkill:
    """Test Skill dataclass."""

    def test_skill_creation(self) -> None:
        """Test creating a skill with all fields."""
        now = datetime.now(timezone.utc)
        embedding = np.random.default_rng(42).random(384, dtype=np.float32)
        skill = Skill(
            skill_id="sk-001",
            name="find auth function",
            description="Locate authentication function in codebase",
            action_sequence=[ActionType.RETRIEVE_SMALL, ActionType.GREP, ActionType.READ_FILE],
            applicable_conditions={"category": "code_navigation"},
            success_count=5,
            reuse_count=2,
            embedding=embedding,
            source_episode_id="ep-001",
            created_at=now,
            updated_at=now,
        )
        assert skill.skill_id == "sk-001"
        assert skill.name == "find auth function"
        assert skill.description == "Locate authentication function in codebase"
        assert len(skill.action_sequence) == 3
        assert skill.success_count == 5
        assert skill.reuse_count == 2
        assert skill.embedding.shape == (384,)
        assert skill.source_episode_id == "ep-001"
        assert skill.created_at == now
        assert skill.updated_at == now

    def test_skill_action_sequence(self) -> None:
        """Test action_sequence stores ordered actions."""
        seq = [ActionType.RETRIEVE_SMALL, ActionType.READ_FILE, ActionType.STOP]
        skill = Skill(
            skill_id="sk-002",
            name="read and stop",
            description="",
            action_sequence=seq,
            applicable_conditions={},
            success_count=0,
            reuse_count=0,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-002",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert skill.action_sequence == seq
        assert skill.action_sequence[0] == ActionType.RETRIEVE_SMALL
        assert skill.action_sequence[-1] == ActionType.STOP

    def test_skill_embedding_dimension(self) -> None:
        """Test embedding is 384-dim (all-MiniLM-L6-v2)."""
        embedding = np.ones(384, dtype=np.float32)
        skill = Skill(
            skill_id="sk-003",
            name="test",
            description="",
            action_sequence=[],
            applicable_conditions={},
            success_count=0,
            reuse_count=0,
            embedding=embedding,
            source_episode_id="ep-003",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert skill.embedding.shape == (384,)
        assert skill.embedding.dtype == np.float32

    def test_skill_applicable_conditions(self) -> None:
        """Test applicable_conditions stores metadata."""
        conditions = {
            "category": "bug_localization",
            "min_difficulty": "medium",
            "file_patterns": ["*.py"],
        }
        skill = Skill(
            skill_id="sk-004",
            name="debug helper",
            description="",
            action_sequence=[],
            applicable_conditions=conditions,
            success_count=0,
            reuse_count=0,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-004",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert skill.applicable_conditions["category"] == "bug_localization"
        assert skill.applicable_conditions["min_difficulty"] == "medium"
        assert "file_patterns" in skill.applicable_conditions

    def test_skill_success_count(self) -> None:
        """Test success_count tracks successful uses."""
        skill = Skill(
            skill_id="sk-005",
            name="test",
            description="",
            action_sequence=[],
            applicable_conditions={},
            success_count=10,
            reuse_count=3,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-005",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert skill.success_count == 10

    def test_skill_reuse_count(self) -> None:
        """Test reuse_count tracks recall invocations."""
        skill = Skill(
            skill_id="sk-006",
            name="test",
            description="",
            action_sequence=[],
            applicable_conditions={},
            success_count=0,
            reuse_count=7,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-006",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert skill.reuse_count == 7

    def test_skill_source_episode_provenance(self) -> None:
        """Test source_episode_id links skill to originating episode."""
        skill = Skill(
            skill_id="sk-007",
            name="test",
            description="",
            action_sequence=[],
            applicable_conditions={},
            success_count=0,
            reuse_count=0,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-orig-42",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert skill.source_episode_id == "ep-orig-42"

    def test_skill_timestamps(self) -> None:
        """Test created_at and updated_at are stored."""
        now = datetime.now(timezone.utc)
        skill = Skill(
            skill_id="sk-008",
            name="test",
            description="",
            action_sequence=[],
            applicable_conditions={},
            success_count=0,
            reuse_count=0,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-008",
            created_at=now,
            updated_at=now,
        )
        assert skill.created_at == now
        assert skill.updated_at == now


class TestSkillEmbeddingSimilarity:
    """Test skill embedding similarity computation."""

    def test_cosine_similarity_identical(self) -> None:
        """Test identical embeddings have similarity 1.0."""
        embedding = np.ones(384, dtype=np.float32)
        sim = np.dot(embedding, embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(embedding)
        )
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self) -> None:
        """Test orthogonal embeddings have similarity 0.0."""
        a = np.zeros(384, dtype=np.float32)
        b = np.zeros(384, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert abs(sim) < 1e-6

    def test_cosine_similarity_range(self) -> None:
        """Test similarity is in [-1, 1] for unit vectors."""
        rng = np.random.default_rng(42)
        a = rng.random(384, dtype=np.float32)
        b = rng.random(384, dtype=np.float32)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        sim = np.dot(a, b)
        assert -1.0 <= sim <= 1.0


class TestSkillActionSequence:
    """Test skill action sequence properties."""

    def test_empty_action_sequence(self) -> None:
        """Test skill can have empty action sequence."""
        skill = Skill(
            skill_id="sk-empty",
            name="empty",
            description="",
            action_sequence=[],
            applicable_conditions={},
            success_count=0,
            reuse_count=0,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-empty",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert len(skill.action_sequence) == 0

    def test_single_action_sequence(self) -> None:
        """Test skill with one action."""
        skill = Skill(
            skill_id="sk-single",
            name="single",
            description="",
            action_sequence=[ActionType.STOP],
            applicable_conditions={},
            success_count=0,
            reuse_count=0,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-single",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert len(skill.action_sequence) == 1
        assert skill.action_sequence[0] == ActionType.STOP

    def test_long_action_sequence(self) -> None:
        """Test skill with realistic action sequence length."""
        seq = [
            ActionType.RETRIEVE_MEDIUM,
            ActionType.READ_FILE,
            ActionType.GREP,
            ActionType.READ_FILE,
            ActionType.RUN_TESTS,
            ActionType.STOP,
        ]
        skill = Skill(
            skill_id="sk-long",
            name="long",
            description="",
            action_sequence=seq,
            applicable_conditions={},
            success_count=0,
            reuse_count=0,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-long",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert len(skill.action_sequence) == 6
        assert skill.action_sequence[0] == ActionType.RETRIEVE_MEDIUM
        assert skill.action_sequence[-1] == ActionType.STOP


class TestSkillApplicableConditions:
    """Test skill applicable conditions flexibility."""

    def test_empty_conditions(self) -> None:
        """Test skill can have empty conditions."""
        skill = Skill(
            skill_id="sk-nocond",
            name="no conditions",
            description="",
            action_sequence=[],
            applicable_conditions={},
            success_count=0,
            reuse_count=0,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-nocond",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert len(skill.applicable_conditions) == 0

    def test_conditions_with_lists(self) -> None:
        """Test conditions can contain list values."""
        conditions = {"file_patterns": ["*.py", "*.js"], "tags": ["auth", "security"]}
        skill = Skill(
            skill_id="sk-list",
            name="list conditions",
            description="",
            action_sequence=[],
            applicable_conditions=conditions,
            success_count=0,
            reuse_count=0,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-list",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert len(skill.applicable_conditions["file_patterns"]) == 2
        assert "auth" in skill.applicable_conditions["tags"]


class TestSkillCounters:
    """Test skill success and reuse counters."""

    def test_zero_counters(self) -> None:
        """Test new skill starts with zero counters."""
        skill = Skill(
            skill_id="sk-new",
            name="new",
            description="",
            action_sequence=[],
            applicable_conditions={},
            success_count=0,
            reuse_count=0,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-new",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert skill.success_count == 0
        assert skill.reuse_count == 0

    def test_increment_success_count(self) -> None:
        """Test success_count can be incremented."""
        skill = Skill(
            skill_id="sk-inc",
            name="inc",
            description="",
            action_sequence=[],
            applicable_conditions={},
            success_count=3,
            reuse_count=0,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-inc",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        skill.success_count += 1
        assert skill.success_count == 4

    def test_increment_reuse_count(self) -> None:
        """Test reuse_count can be incremented."""
        skill = Skill(
            skill_id="sk-reuse",
            name="reuse",
            description="",
            action_sequence=[],
            applicable_conditions={},
            success_count=0,
            reuse_count=5,
            embedding=np.zeros(384, dtype=np.float32),
            source_episode_id="ep-reuse",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        skill.reuse_count += 1
        assert skill.reuse_count == 6
