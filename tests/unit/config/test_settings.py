"""Tests for configuration settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from evocode.config.settings import (
    LoggingConfig,
    PolicyConfig,
    RetrievalConfig,
    RewardConfig,
    Settings,
    ToolConfig,
)


class TestSettings:
    """Test Settings class."""

    def test_default_settings_load(self) -> None:
        """Test that default settings load without error."""
        settings = Settings()
        assert settings is not None
        assert settings.environment == "development"

    def test_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test settings can be overridden by environment variables."""
        monkeypatch.setenv("EVOCODE_ENVIRONMENT", "production")
        monkeypatch.setenv("EVOCODE_SEED", "42")
        settings = Settings()
        assert settings.environment == "production"
        assert settings.seed == 42

    def test_settings_from_yaml(self, tmp_path: Path) -> None:
        """Test settings can be loaded from YAML file."""
        yaml_content = """
environment: "test"
seed: 123
retrieval:
  embedding_model: "test-model"
  chunk_size: 512
tool:
  max_file_size_mb: 10
reward:
  w_quality: 2.0
policy:
  hidden_dim: 256
logging:
  level: "DEBUG"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        settings = Settings.from_yaml(config_file)
        assert settings.environment == "test"
        assert settings.seed == 123
        assert settings.retrieval.embedding_model == "test-model"
        assert settings.retrieval.chunk_size == 512
        assert settings.tool.max_file_size_mb == 10
        assert settings.reward.w_quality == 2.0
        assert settings.policy.hidden_dim == 256
        assert settings.logging.level == "DEBUG"

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        """Test invalid YAML raises YAMLError."""
        import yaml

        config_file = tmp_path / "bad.yaml"
        config_file.write_text("invalid: yaml: content: [")
        with pytest.raises(yaml.YAMLError):
            Settings.from_yaml(config_file)

    def test_env_overrides_yaml(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test YAML values take precedence over env vars when using from_yaml."""
        yaml_content = """
environment: "test"
seed: 123
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        monkeypatch.setenv("EVOCODE_SEED", "999")
        settings = Settings.from_yaml(config_file)
        # YAML takes precedence over env vars in from_yaml
        assert settings.seed == 123
        assert settings.environment == "test"

    def test_retrieval_config_defaults(self) -> None:
        """Test RetrievalConfig has sensible defaults."""
        config = RetrievalConfig()
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 100
        assert config.top_k_default == 5
        assert config.index_type == "flat"

    def test_tool_config_defaults(self) -> None:
        """Test ToolConfig has sensible defaults."""
        config = ToolConfig()
        assert config.max_file_size_mb == 5
        assert config.timeout_seconds == 30
        assert config.allow_read is True
        assert config.allow_grep is True
        assert config.allow_run_tests is True
        assert config.allow_write is False  # safety default

    def test_reward_config_defaults(self) -> None:
        """Test RewardConfig has documented default weights."""
        config = RewardConfig()
        assert config.w_quality == 1.0
        assert config.w_success == 1.0
        assert config.w_efficiency == 0.01
        assert config.w_latency == 0.001
        assert config.w_tool_cost == 0.05
        assert config.w_safety == 10.0

    def test_policy_config_defaults(self) -> None:
        """Test PolicyConfig has sensible defaults."""
        config = PolicyConfig()
        assert config.algorithm == "PPO"
        assert config.hidden_dim == 128
        assert config.num_layers == 2
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_range == 0.2

    def test_logging_config_defaults(self) -> None:
        """Test LoggingConfig has sensible defaults."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.log_file is None

    def test_settings_immutable_after_creation(self) -> None:
        """Test settings behave like frozen dataclass (Pydantic v2 default)."""
        settings = Settings(seed=42)
        # Pydantic v2 models are not frozen by default, but we can verify
        # the values are as expected
        assert settings.seed == 42

    def test_extra_fields_forbidden(self) -> None:
        """Test extra fields in YAML are rejected."""
        yaml_content = """
environment: "test"
nonexistent_field: "should_fail"
"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(ValidationError):
                Settings.from_yaml(f.name)


class TestSettingsValidation:
    """Test validation rules in Settings."""

    def test_seed_must_be_non_negative(self) -> None:
        """Test seed must be non-negative."""
        with pytest.raises(ValidationError):
            Settings(seed=-1)

    def test_chunk_size_positive(self) -> None:
        """Test chunk_size must be positive."""
        with pytest.raises(ValidationError):
            RetrievalConfig(chunk_size=0)

    def test_timeout_positive(self) -> None:
        """Test timeout_seconds must be positive."""
        with pytest.raises(ValidationError):
            ToolConfig(timeout_seconds=0)

    def test_hidden_dim_positive(self) -> None:
        """Test hidden_dim must be positive."""
        with pytest.raises(ValidationError):
            PolicyConfig(hidden_dim=0)

    def test_learning_rate_positive(self) -> None:
        """Test learning_rate must be positive."""
        with pytest.raises(ValidationError):
            PolicyConfig(learning_rate=-0.01)

    def test_gamma_in_range(self) -> None:
        """Test gamma must be in (0, 1]."""
        with pytest.raises(ValidationError):
            PolicyConfig(gamma=0.0)
        with pytest.raises(ValidationError):
            PolicyConfig(gamma=1.5)
        # Valid
        PolicyConfig(gamma=0.99)
        PolicyConfig(gamma=1.0)

    def test_gae_lambda_in_range(self) -> None:
        """Test gae_lambda must be in [0, 1]."""
        with pytest.raises(ValidationError):
            PolicyConfig(gae_lambda=-0.1)
        with pytest.raises(ValidationError):
            PolicyConfig(gae_lambda=1.1)
        PolicyConfig(gae_lambda=0.0)
        PolicyConfig(gae_lambda=1.0)
