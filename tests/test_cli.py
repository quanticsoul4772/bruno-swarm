# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for bruno_swarm.cli — helper functions and Click command smoke tests."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from bruno_swarm.cli import (
    _download_model,
    _get_or_create_agent,
    _parse_agents,
    _prewarm_model,
    _validate_ollama_url,
    _validate_output_path,
    cli,
)

# ---------------------------------------------------------------------------
# _parse_agents
# ---------------------------------------------------------------------------


class TestParseAgents:
    def test_none_returns_none(self):
        assert _parse_agents(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_agents("") is None

    def test_valid_single(self):
        assert _parse_agents("backend") == ["backend"]

    def test_valid_multiple(self):
        result = _parse_agents("frontend, backend, test")
        assert result == ["frontend", "backend", "test"]

    def test_invalid_exits(self, mock_console):
        with pytest.raises(SystemExit):
            _parse_agents("nonexistent")


# ---------------------------------------------------------------------------
# _validate_ollama_url
# ---------------------------------------------------------------------------


class TestValidateOllamaUrl:
    def test_http_passes(self):
        assert _validate_ollama_url("http://localhost:11434") == "http://localhost:11434"

    def test_https_passes(self):
        assert _validate_ollama_url("https://ollama.example.com") == "https://ollama.example.com"

    def test_ftp_exits(self, mock_console):
        with pytest.raises(SystemExit):
            _validate_ollama_url("ftp://localhost:11434")

    def test_file_exits(self, mock_console):
        with pytest.raises(SystemExit):
            _validate_ollama_url("file:///etc/passwd")

    def test_no_scheme_exits(self, mock_console):
        with pytest.raises(SystemExit):
            _validate_ollama_url("localhost:11434")


# ---------------------------------------------------------------------------
# _validate_output_path
# ---------------------------------------------------------------------------


class TestValidateOutputPath:
    def test_relative_path_inside_cwd(self, mock_console):
        result = _validate_output_path("result.txt")
        assert result == Path.cwd().resolve() / "result.txt"
        mock_console.print.assert_not_called()

    def test_absolute_path_outside_cwd_warns(self, mock_console, tmp_path):
        outside = str(tmp_path / "outside.txt")
        result = _validate_output_path(outside)
        assert result == Path(outside).resolve()
        # Should have printed a warning (unless tmp_path happens to be under cwd)
        if not str(Path(outside).resolve()).startswith(str(Path.cwd().resolve())):
            mock_console.print.assert_called_once()
            warning_text = mock_console.print.call_args[0][0]
            assert "Warning" in warning_text


# ---------------------------------------------------------------------------
# Click CLI smoke tests
# ---------------------------------------------------------------------------


class TestCliHelp:
    def test_main_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Bruno AI Developer Swarm CLI" in result.output

    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--task" in result.output

    def test_setup_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["setup", "--help"])
        assert result.exit_code == 0
        assert "--models" in result.output

    def test_status_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0
        assert "--ollama-url" in result.output


class TestAgentsCommand:
    def test_agents_lists_all(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["agents"])
        assert result.exit_code == 0
        assert "orchestrator" in result.output
        assert "frontend" in result.output
        assert "backend" in result.output
        assert "test" in result.output
        assert "security" in result.output
        assert "docs" in result.output
        assert "devops" in result.output


# ---------------------------------------------------------------------------
# OPT-1: _download_model
# ---------------------------------------------------------------------------


class TestDownloadModel:
    def test_success_returns_path(self, mock_console):
        import threading

        fake_hf = lambda repo_id, filename: "/tmp/model.gguf"  # noqa: E731
        lock = threading.Lock()
        name, path = _download_model("backend-3b-f16.gguf", "backend", fake_hf, mock_console, lock)
        assert name == "backend"
        assert path == "/tmp/model.gguf"

    def test_failure_returns_none(self, mock_console):
        import threading

        def _raise(**kwargs):
            raise RuntimeError("network error")

        lock = threading.Lock()
        name, path = _download_model("bad.gguf", "backend", _raise, mock_console, lock)
        assert name == "backend"
        assert path is None


# ---------------------------------------------------------------------------
# OPT-2: _prewarm_model
# ---------------------------------------------------------------------------


class TestPrewarmModel:
    def test_sends_generate_request(self, mock_console, monkeypatch):
        import json
        import urllib.request

        captured = {}

        class FakeResponse:
            def read(self):
                return b'{"response": ""}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode())
            captured["method"] = req.get_method()
            return FakeResponse()

        # Patch in the cli module's namespace — _prewarm_model imports urllib lazily
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        _prewarm_model("backend", "http://localhost:11434")

        assert captured["url"] == "http://localhost:11434/api/generate"
        assert captured["body"]["model"] == "backend"
        assert captured["method"] == "POST"

    def test_failure_is_silent(self, mock_console, monkeypatch):
        import urllib.request
        from urllib.error import URLError

        def fail_urlopen(req, timeout=None):
            raise URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

        # Should not raise
        _prewarm_model("backend", "http://localhost:11434")


# ---------------------------------------------------------------------------
# OPT-3: _get_or_create_agent / crew with cache
# ---------------------------------------------------------------------------


class TestGetOrCreateAgent:
    def test_caches_agent(self, mock_console, monkeypatch):
        import sys

        cli_mod = sys.modules["bruno_swarm.cli"]
        call_count = 0
        sentinel = object()

        def fake_create_agent(name, base_url):
            nonlocal call_count
            call_count += 1
            return sentinel

        monkeypatch.setattr(cli_mod, "create_agent", fake_create_agent)

        cache: dict = {}
        result1 = _get_or_create_agent("backend", "http://localhost:11434", cache)
        result2 = _get_or_create_agent("backend", "http://localhost:11434", cache)

        assert result1 is sentinel
        assert result2 is sentinel
        assert call_count == 1

    def test_different_names_create_separate(self, mock_console, monkeypatch):
        import sys

        cli_mod = sys.modules["bruno_swarm.cli"]
        created = []

        def fake_create_agent(name, base_url):
            created.append(name)
            return f"agent-{name}"

        monkeypatch.setattr(cli_mod, "create_agent", fake_create_agent)

        cache: dict = {}
        _get_or_create_agent("backend", "http://localhost:11434", cache)
        _get_or_create_agent("frontend", "http://localhost:11434", cache)

        assert created == ["backend", "frontend"]


class TestCrewWithAgentCache:
    def test_flat_crew_uses_cache(self, mock_console, monkeypatch):
        import sys

        cli_mod = sys.modules["bruno_swarm.cli"]

        # Mock crewai classes
        from unittest.mock import MagicMock

        fake_crew_cls = MagicMock()
        fake_task_cls = MagicMock()
        fake_process = MagicMock()
        fake_process.sequential = "sequential"

        monkeypatch.setattr(cli_mod, "make_step_callback", lambda c: None)

        # Pre-populate cache with two agents
        cache = {
            ("backend", "http://localhost:11434"): MagicMock(name="cached-backend"),
            ("frontend", "http://localhost:11434"): MagicMock(name="cached-frontend"),
        }

        create_agent_calls = []
        original_create = cli_mod.create_agent

        def spy_create(name, base_url):
            create_agent_calls.append(name)
            return original_create(name, base_url)

        monkeypatch.setattr(cli_mod, "create_agent", spy_create)

        # Patch crewai imports inside create_flat_crew
        import unittest.mock

        with unittest.mock.patch.dict(
            "sys.modules",
            {
                "crewai": MagicMock(Crew=fake_crew_cls, Task=fake_task_cls, Process=fake_process),
            },
        ):
            from bruno_swarm.cli import create_flat_crew

            create_flat_crew(
                "test task",
                "http://localhost:11434",
                agent_names=["backend", "frontend"],
                agent_cache=cache,
            )

        # create_agent should NOT have been called for cached agents
        assert create_agent_calls == []
