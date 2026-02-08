# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for bruno_swarm.cli — helper functions and Click command smoke tests."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from bruno_swarm.cli import (
    _parse_agents,
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
