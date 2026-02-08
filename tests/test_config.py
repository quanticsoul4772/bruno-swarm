# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for bruno_swarm.config — shared constants and helpers."""

from unittest.mock import MagicMock

from bruno_swarm.config import (
    AGENT_CONFIGS,
    DEFAULT_OLLAMA_URL,
    EXPECTED_OUTPUTS,
    HF_MODELS,
    SPECIALISTS,
    TASK_TEMPLATES,
    make_step_callback,
)

# ---------------------------------------------------------------------------
# AGENT_CONFIGS
# ---------------------------------------------------------------------------


class TestAgentConfigs:
    """Validate the structure and content of AGENT_CONFIGS."""

    REQUIRED_KEYS = {"model", "role", "goal", "backstory", "system_prompt", "allow_delegation"}

    def test_has_seven_agents(self):
        assert len(AGENT_CONFIGS) == 7

    def test_all_agents_have_required_keys(self):
        for name, cfg in AGENT_CONFIGS.items():
            missing = self.REQUIRED_KEYS - set(cfg.keys())
            assert not missing, f"Agent '{name}' missing keys: {missing}"

    def test_orchestrator_allows_delegation(self):
        assert AGENT_CONFIGS["orchestrator"]["allow_delegation"] is True

    def test_specialists_disallow_delegation(self):
        for name in SPECIALISTS:
            assert AGENT_CONFIGS[name]["allow_delegation"] is False, (
                f"Specialist '{name}' should not allow delegation"
            )

    def test_model_name_matches_agent_name(self):
        for name, cfg in AGENT_CONFIGS.items():
            assert cfg["model"] == name

    def test_system_prompt_is_nonempty_string(self):
        for name, cfg in AGENT_CONFIGS.items():
            assert isinstance(cfg["system_prompt"], str)
            assert len(cfg["system_prompt"]) > 20, f"Agent '{name}' system_prompt too short"


# ---------------------------------------------------------------------------
# SPECIALISTS
# ---------------------------------------------------------------------------


class TestSpecialists:
    def test_six_specialists(self):
        assert len(SPECIALISTS) == 6

    def test_orchestrator_not_in_specialists(self):
        assert "orchestrator" not in SPECIALISTS

    def test_all_specialists_in_agent_configs(self):
        for name in SPECIALISTS:
            assert name in AGENT_CONFIGS


# ---------------------------------------------------------------------------
# TASK_TEMPLATES / EXPECTED_OUTPUTS
# ---------------------------------------------------------------------------


class TestTaskTemplates:
    def test_covers_all_specialists(self):
        for name in SPECIALISTS:
            assert name in TASK_TEMPLATES, f"Missing template for '{name}'"

    def test_templates_use_task_placeholder(self):
        for name, tmpl in TASK_TEMPLATES.items():
            assert "{task}" in tmpl, f"Template for '{name}' missing {{task}} placeholder"

    def test_format_does_not_raise(self):
        for name, tmpl in TASK_TEMPLATES.items():
            result = tmpl.format(task="build a REST API")
            assert "build a REST API" in result


class TestExpectedOutputs:
    def test_covers_all_specialists(self):
        for name in SPECIALISTS:
            assert name in EXPECTED_OUTPUTS, f"Missing expected output for '{name}'"

    def test_outputs_are_nonempty_strings(self):
        for name, out in EXPECTED_OUTPUTS.items():
            assert isinstance(out, str) and len(out) > 5


# ---------------------------------------------------------------------------
# HF_MODELS / DEFAULT_OLLAMA_URL
# ---------------------------------------------------------------------------


class TestHfModels:
    def test_seven_models(self):
        assert len(HF_MODELS) == 7

    def test_values_match_agent_names(self):
        for ollama_name in HF_MODELS.values():
            assert ollama_name in AGENT_CONFIGS


class TestDefaultOllamaUrl:
    def test_is_http_localhost(self):
        assert DEFAULT_OLLAMA_URL.startswith("http://")
        assert "localhost" in DEFAULT_OLLAMA_URL


# ---------------------------------------------------------------------------
# make_step_callback
# ---------------------------------------------------------------------------


class TestMakeStepCallback:
    def test_returns_callable(self):
        console = MagicMock()
        cb = make_step_callback(console)
        assert callable(cb)

    def test_callback_calls_console_print(self):
        console = MagicMock()
        cb = make_step_callback(console)

        # CrewAI passes AgentAction/AgentFinish which don't carry agent identity
        step_output = MagicMock()
        cb(step_output)

        console.print.assert_called_once()
        printed = console.print.call_args[0][0]
        assert "Agent" in printed

    def test_callback_handles_missing_agent(self):
        console = MagicMock()
        cb = make_step_callback(console)

        step_output = MagicMock(spec=[])  # no agent attribute
        cb(step_output)

        console.print.assert_called_once()
        printed = console.print.call_args[0][0]
        assert "Agent" in printed
