"""Configuration settings for EvoCode."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalConfig(BaseModel):
    """Configuration for retrieval system."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k_default: int = 5
    index_type: Literal["flat", "ivf"] = "flat"

    @field_validator("chunk_size", "chunk_overlap", "top_k_default")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_less_than_chunk(cls, v: int, info) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class ToolConfig(BaseModel):
    """Configuration for tool execution."""

    max_file_size_mb: int = 5
    timeout_seconds: int = 30
    allow_read: bool = True
    allow_grep: bool = True
    allow_run_tests: bool = True
    allow_write: bool = False

    @field_validator("max_file_size_mb", "timeout_seconds")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v


class RewardConfig(BaseModel):
    """Configuration for reward weights.

    Weights are configurable to enable ablation studies.
    """

    w_quality: float = 1.0
    w_success: float = 1.0
    w_efficiency: float = 0.01
    w_latency: float = 0.001
    w_tool_cost: float = 0.05
    w_safety: float = 10.0

    @field_validator(
        "w_quality", "w_success", "w_efficiency", "w_latency", "w_tool_cost", "w_safety"
    )
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weight must be non-negative")
        return v


class PolicyConfig(BaseModel):
    """Configuration for RL policy."""

    algorithm: Literal["PPO", "DQN", "A2C"] = "PPO"
    hidden_dim: int = 128
    num_layers: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    batch_size: int = 64
    n_steps: int = 2048

    @field_validator("hidden_dim", "num_layers", "batch_size", "n_steps")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v

    @field_validator("learning_rate")
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("learning_rate must be positive")
        return v

    @field_validator("gamma")
    @classmethod
    def _gamma_range(cls, v: float) -> float:
        if not (0 < v <= 1):
            raise ValueError("gamma must be in (0, 1]")
        return v

    @field_validator("gae_lambda")
    @classmethod
    def _gae_lambda_range(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError("gae_lambda must be in [0, 1]")
        return v

    @field_validator("clip_range")
    @classmethod
    def _clip_range_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("clip_range must be positive")
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "text"] = "json"
    log_file: str | None = None


class Settings(BaseSettings):
    """Main settings for EvoCode.

    Priority order (highest to lowest):
    1. Environment variables (EVOCODE_*)
    2. YAML config file
    3. Defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="EVOCODE_",
        env_nested_delimiter="__",
        extra="forbid",
        validate_assignment=True,
    )

    environment: Literal["development", "test", "production"] = "development"
    seed: int = 42

    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tool: ToolConfig = Field(default_factory=ToolConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("seed")
    @classmethod
    def _non_negative_seed(cls, v: int) -> int:
        if v < 0:
            raise ValueError("seed must be non-negative")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from YAML file.

        Note: Values from YAML take precedence over environment variables
        when using this method. For env var precedence, load YAML manually
        and pass to Settings() constructor.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r") as f:
            data = yaml.safe_load(f) or {}

        # Validate and create instance - YAML values take precedence
        return cls.model_validate(data)
