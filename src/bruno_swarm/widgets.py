# SPDX-License-Identifier: AGPL-3.0-or-later
# Bruno Swarm TUI Widgets

"""Custom Textual widgets for the Bruno Swarm TUI."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from .config import AGENT_CONFIGS

# Status symbols and their CSS classes
STATUS_SYMBOLS = {
    "idle": "\u25cb",  # ○
    "working": "\u25cf",  # ●
    "done": "\u2713",  # ✓
    "error": "\u2717",  # ✗
}

STATUS_CLASSES = {
    "idle": "status-idle",
    "working": "status-working",
    "done": "status-done",
    "error": "status-error",
}


class AgentStatusLine(Static):
    """Single agent line showing status symbol + name."""

    status = reactive("idle")

    def __init__(self, agent_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent_name = agent_name

    def render(self) -> str:
        symbol = STATUS_SYMBOLS.get(self.status, STATUS_SYMBOLS["idle"])
        return f" {symbol} {self.agent_name}"

    def watch_status(self, new_status: str) -> None:
        for cls in STATUS_CLASSES.values():
            self.remove_class(cls)
        css_class = STATUS_CLASSES.get(new_status, STATUS_CLASSES["idle"])
        self.add_class(css_class)


class AgentSidebar(Widget):
    """Vertical panel showing all 7 agents with status indicators."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._agent_lines: dict[str, AgentStatusLine] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-list"):
            yield Static("[b]Agents[/b]", id="sidebar-title")
            for name in AGENT_CONFIGS:
                line = AgentStatusLine(name, id=f"agent-{name}")
                self._agent_lines[name] = line
                yield line

    def update_status(self, agent_name: str, status: str) -> None:
        """Update a single agent's status display."""
        if agent_name in self._agent_lines:
            self._agent_lines[agent_name].status = status

    def reset_all(self) -> None:
        """Reset all agents to idle status."""
        for line in self._agent_lines.values():
            line.status = "idle"


class ModeIndicator(Static):
    """Top bar showing current mode, agents, and Ollama URL."""

    def __init__(
        self,
        mode: str = "hierarchical",
        agents_str: str = "all specialists",
        ollama_url: str = "http://localhost:11434",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._mode = mode
        self._agents_str = agents_str
        self._ollama_url = ollama_url

    def render(self) -> str:
        return (
            f" Mode: [cyan]{self._mode}[/] "
            f"| Agents: [cyan]{self._agents_str}[/] "
            f"| Ollama: [cyan]{self._ollama_url}[/]"
        )

    def update_info(self, mode: str, agents_str: str) -> None:
        """Update the displayed mode and agents info."""
        self._mode = mode
        self._agents_str = agents_str
        self.refresh()
