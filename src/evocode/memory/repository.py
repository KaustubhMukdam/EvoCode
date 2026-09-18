"""Memory repository protocols for EvoCode."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evocode.domain.episodes import Episode
from evocode.domain.skills import Skill


@runtime_checkable
class EpisodeRepository(Protocol):
    """Protocol for episode persistence."""

    def save(self, episode: Episode) -> None: ...

    def get(self, episode_id: str) -> Episode | None: ...

    def list_all(self) -> list[Episode]: ...


@runtime_checkable
class SkillRepository(Protocol):
    """Protocol for skill persistence."""

    def save(self, skill: Skill) -> None: ...

    def get(self, skill_id: str) -> Skill | None: ...

    def list_all(self) -> list[Skill]: ...

    def delete(self, skill_id: str) -> None: ...


class FakeEpisodeRepository:
    """In-memory fake episode repository for testing."""

    def __init__(self) -> None:
        self._episodes: dict[str, Episode] = {}

    def save(self, episode: Episode) -> None:
        self._episodes[episode.episode_id] = episode

    def get(self, episode_id: str) -> Episode | None:
        return self._episodes.get(episode_id)

    def list_all(self) -> list[Episode]:
        return list(self._episodes.values())


class FakeSkillRepository:
    """In-memory fake skill repository for testing."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def save(self, skill: Skill) -> None:
        self._skills[skill.skill_id] = skill

    def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def list_all(self) -> list[Skill]:
        return list(self._skills.values())

    def delete(self, skill_id: str) -> None:
        self._skills.pop(skill_id, None)
