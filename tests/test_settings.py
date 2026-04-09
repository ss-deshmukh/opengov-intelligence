from __future__ import annotations

import os
from pathlib import Path

import pytest

from oscat.config.settings import OscatSettings


_OSCAT_MODEL_KEYS = [
    "OSCAT_PLANNING_PROVIDER",
    "OSCAT_PLANNING_MODEL",
    "OSCAT_CODING_PROVIDER",
    "OSCAT_CODING_MODEL",
]


class TestOscatSettingsDefaults:
    def test_default_planning_provider(self, monkeypatch):
        for k in _OSCAT_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        s = OscatSettings(anthropic_api_key="test", _env_file=None)
        assert s.planning_provider == "anthropic"

    def test_default_planning_model(self, monkeypatch):
        for k in _OSCAT_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        s = OscatSettings(anthropic_api_key="test", _env_file=None)
        assert s.planning_model == "claude-sonnet-4-6"

    def test_default_coding_provider(self, monkeypatch):
        for k in _OSCAT_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        s = OscatSettings(anthropic_api_key="test", _env_file=None)
        assert s.coding_provider == "anthropic"

    def test_default_coding_model(self, monkeypatch):
        for k in _OSCAT_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        s = OscatSettings(anthropic_api_key="test", _env_file=None)
        assert s.coding_model == "claude-haiku-4-5-20251001"

    def test_default_memory_dir(self):
        s = OscatSettings(anthropic_api_key="test")
        assert s.memory_dir == ".oscat"

    def test_default_context_dir(self):
        s = OscatSettings(anthropic_api_key="test")
        assert s.context_dir == ".oscat/context"

    def test_default_api_key_is_none(self):
        s = OscatSettings(_env_file=None)
        assert s.anthropic_api_key is None


class TestOscatSettingsEnvOverride:
    def test_env_overrides_planning_model(self, monkeypatch):
        monkeypatch.setenv("OSCAT_PLANNING_MODEL", "custom-model")
        s = OscatSettings(_env_file=None)
        assert s.planning_model == "custom-model"

    def test_env_overrides_api_key(self, monkeypatch):
        monkeypatch.setenv("OSCAT_ANTHROPIC_API_KEY", "sk-test-key")
        s = OscatSettings(_env_file=None)
        assert s.anthropic_api_key == "sk-test-key"

class TestWorkspaceResolution:
    def test_resolve_workspace_defaults_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = OscatSettings(anthropic_api_key="test", _env_file=None)
        s.resolve_workspace()

        assert s.workspace_path == tmp_path
        assert Path(s.memory_dir) == tmp_path / ".oscat"
        assert Path(s.context_dir) == tmp_path / ".oscat" / "context"

    def test_resolve_workspace_with_explicit_folder(self, tmp_path):
        s = OscatSettings(anthropic_api_key="test", _env_file=None)
        s.resolve_workspace(str(tmp_path))

        assert s.workspace_path == tmp_path
        assert Path(s.memory_dir) == tmp_path / ".oscat"
        assert Path(s.context_dir) == tmp_path / ".oscat" / "context"

    def test_resolve_workspace_does_not_create_oscat_dir(self, tmp_path):
        """resolve_workspace only resolves paths — directory creation is deferred to initialize()."""
        s = OscatSettings(anthropic_api_key="test", _env_file=None)
        s.resolve_workspace(str(tmp_path))

        assert not (tmp_path / ".oscat").exists()

    def test_resolve_workspace_preserves_absolute_paths(self, tmp_path):
        s = OscatSettings(
            anthropic_api_key="test",
            memory_dir="/absolute/path",
            _env_file=None,
        )
        s.resolve_workspace(str(tmp_path))

        # Absolute path should not be changed
        assert s.memory_dir == "/absolute/path"

    def test_workspace_path_before_resolve(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = OscatSettings(anthropic_api_key="test", _env_file=None)
        # Before resolve, workspace_path falls back to cwd
        assert s.workspace_path == tmp_path
