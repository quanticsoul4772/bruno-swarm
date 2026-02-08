# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the Bruno Swarm TUI."""

import pytest

textual = pytest.importorskip("textual", reason="textual not installed")

from bruno_swarm.config import AGENT_CONFIGS  # noqa: E402
from bruno_swarm.tui import SwarmTUI  # noqa: E402
from bruno_swarm.widgets import AgentSidebar, AgentStatusLine  # noqa: E402


class TestSwarmTUIComposition:
    """Test that the TUI app composes correctly with all widgets."""

    @pytest.mark.asyncio
    async def test_app_starts_with_all_widgets(self):
        """App starts and contains all required widgets."""
        app = SwarmTUI()
        async with app.run_test():
            assert app.query_one("#chat-log") is not None
            assert app.query_one("#task-input") is not None
            assert app.query_one("#agent-sidebar") is not None
            assert app.query_one("#mode-indicator") is not None

    @pytest.mark.asyncio
    async def test_default_mode_is_hierarchical(self):
        """Default mode should be hierarchical."""
        app = SwarmTUI()
        async with app.run_test():
            assert app.mode == "hierarchical"
            assert app.selected_agents is None

    @pytest.mark.asyncio
    async def test_flat_mode_initialization(self):
        """App respects flat=True and agent_names on init."""
        app = SwarmTUI(flat=True, agent_names=["backend", "frontend"])
        async with app.run_test():
            assert app.mode == "flat"
            assert app.selected_agents == ["backend", "frontend"]


class TestCommandHandling:
    """Test slash command handling in the TUI."""

    @pytest.mark.asyncio
    async def test_help_command(self):
        """The /help command writes to the chat log without crashing."""
        app = SwarmTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#task-input")
            inp.value = "/help"
            await inp.action_submit()
            await pilot.pause()
            # If we get here without exception, /help works

    @pytest.mark.asyncio
    async def test_mode_switch_to_flat(self):
        """The /mode flat command switches the mode."""
        app = SwarmTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#task-input")
            inp.value = "/mode flat"
            await inp.action_submit()
            await pilot.pause()
            assert app.mode == "flat"

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        """Unknown commands post a warning without crashing."""
        app = SwarmTUI()
        async with app.run_test() as pilot:
            inp = app.query_one("#task-input")
            inp.value = "/notacommand"
            await inp.action_submit()
            await pilot.pause()
            # If we get here without exception, unknown command handling works


class TestThemeCycling:
    """Test theme cycling keybinding."""

    @pytest.mark.asyncio
    async def test_theme_cycles(self):
        """Cycling theme via action changes the theme."""
        app = SwarmTUI()
        async with app.run_test() as pilot:
            initial_theme = app.theme
            app.action_cycle_theme()
            await pilot.pause()
            assert app.theme != initial_theme


class TestAgentSidebar:
    """Test the agent sidebar widget."""

    @pytest.mark.asyncio
    async def test_all_agents_in_sidebar(self):
        """All 7 agents should appear in the sidebar."""
        app = SwarmTUI()
        async with app.run_test():
            for name in AGENT_CONFIGS:
                line = app.query_one(f"#agent-{name}", AgentStatusLine)
                assert line.agent_name == name

    @pytest.mark.asyncio
    async def test_status_update(self):
        """Updating agent status changes the reactive property."""
        app = SwarmTUI()
        async with app.run_test() as pilot:
            sidebar = app.query_one("#agent-sidebar", AgentSidebar)
            sidebar.update_status("backend", "working")
            await pilot.pause()
            line = app.query_one("#agent-backend", AgentStatusLine)
            assert line.status == "working"

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """reset_all() sets all agents back to idle."""
        app = SwarmTUI()
        async with app.run_test() as pilot:
            sidebar = app.query_one("#agent-sidebar", AgentSidebar)
            sidebar.update_status("backend", "done")
            sidebar.update_status("frontend", "error")
            sidebar.reset_all()
            await pilot.pause()
            for name in AGENT_CONFIGS:
                line = app.query_one(f"#agent-{name}", AgentStatusLine)
                assert line.status == "idle"
