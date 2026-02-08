# SPDX-License-Identifier: AGPL-3.0-or-later
# Bruno Swarm Chat TUI
#
# Full-screen terminal interface for running the agent swarm.
# Built on Textual (Rich-based TUI framework).

import os
import threading
from datetime import datetime
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog

from .config import AGENT_CONFIGS, DEFAULT_OLLAMA_URL, SPECIALISTS
from .widgets import AgentSidebar, ModeIndicator

# Reverse lookup: role -> agent name (e.g. "Backend Developer" -> "backend")
_ROLE_TO_NAME = {cfg["role"]: name for name, cfg in AGENT_CONFIGS.items()}


class SwarmTUI(App):
    """Interactive chat TUI for Bruno Swarm."""

    TITLE = "Bruno Swarm"
    CSS_PATH = "tui.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("t", "cycle_theme", "Theme", show=True, key_display="t"),
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    _THEMES = ["textual-dark", "textual-light", "textual-ansi"]

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        flat: bool = False,
        agent_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.ollama_url = ollama_url
        self.mode = "flat" if flat else "hierarchical"
        self.selected_agents: list[str] | None = agent_names
        self._task_running = False
        self._history: list[dict] = []
        self._last_result = None
        self._agent_cache: dict = {}
        self._theme_index = 0

    @property
    def _agents_str(self) -> str:
        return ", ".join(self.selected_agents) if self.selected_agents else "all specialists"

    def compose(self) -> ComposeResult:
        yield Header()
        yield ModeIndicator(
            mode=self.mode,
            agents_str=self._agents_str,
            ollama_url=self.ollama_url,
            id="mode-indicator",
        )
        with Horizontal():
            yield AgentSidebar(id="agent-sidebar")
            with Vertical(id="main-content"):
                yield RichLog(id="chat-log", wrap=True, highlight=True, markup=True)
        yield Input(placeholder="Type a task or /command...", id="task-input")
        yield Footer()

    def on_mount(self) -> None:
        """Focus the input on startup and show welcome message."""
        self.query_one("#task-input", Input).focus()
        log = self.query_one("#chat-log", RichLog)
        log.write("[bold cyan]Bruno AI Developer Swarm[/]")
        log.write("Type a task to run the swarm, or /help for commands.\n")

    def action_cycle_theme(self) -> None:
        """Cycle through available themes."""
        self._theme_index = (self._theme_index + 1) % len(self._THEMES)
        self.theme = self._THEMES[self._theme_index]

    def action_focus_input(self) -> None:
        """Focus the task input field."""
        self.query_one("#task-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle task input or commands."""
        text = event.value.strip()
        if not text:
            return
        event.input.clear()

        if text.startswith("/"):
            self._handle_command(text)
        elif self._task_running:
            self._post("[yellow]A task is already running. Please wait.[/]")
        else:
            self._post(f"[bold]You:[/] {text}")
            self._execute_task(text)

    def _post(self, message: str) -> None:
        """Post a message to the chat log."""
        self.query_one("#chat-log", RichLog).write(message)

    def _post_agent_step(self, agent_role: str) -> None:
        """Post an agent step notification to the chat log."""
        self._post(f"  [cyan]{agent_role}[/] completed a step")

    # --- Commands ---

    def _handle_command(self, text: str) -> None:
        """Dispatch slash commands."""
        parts = text.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            self.exit()
        elif cmd == "/help":
            self._cmd_help()
        elif cmd == "/agents":
            self._cmd_agents()
        elif cmd == "/status":
            self._cmd_status()
        elif cmd == "/mode":
            self._cmd_mode(arg)
        elif cmd == "/use":
            self._cmd_use(arg)
        elif cmd == "/history":
            self._cmd_history()
        elif cmd == "/save":
            self._cmd_save(arg)
        elif cmd == "/clear":
            self.query_one("#chat-log", RichLog).clear()
        else:
            self._post(f"[yellow]Unknown command: {cmd}[/] — type /help for available commands")

    def _cmd_help(self) -> None:
        self._post(
            "[bold]Commands:[/]\n"
            "  /agents        — list available agents\n"
            "  /status        — check Ollama connectivity\n"
            "  /mode flat|hier — switch execution mode\n"
            "  /use a1,a2|all — select specific agents\n"
            "  /history       — show task history\n"
            "  /save <file>   — save last result to file\n"
            "  /clear         — clear chat log\n"
            "  /quit          — exit\n\n"
            "[bold]Keys:[/]\n"
            "  t              — cycle theme\n"
            "  Ctrl+Q         — quit\n"
            "  Escape         — focus input"
        )

    def _cmd_agents(self) -> None:
        from rich.table import Table

        table = Table(title="Bruno Swarm Agents", expand=True)
        table.add_column("Name", style="cyan")
        table.add_column("Role", style="bold")
        table.add_column("Goal")
        for name, config in AGENT_CONFIGS.items():
            table.add_row(name, config["role"], config["goal"])
        self.query_one("#chat-log", RichLog).write(table)

    def _cmd_status(self) -> None:
        self._check_ollama_status()

    @work(thread=True)
    def _check_ollama_status(self) -> None:
        """Check Ollama connectivity in a background thread."""
        import json
        import urllib.error
        import urllib.request

        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            models = data.get("models", [])
            model_names = {m.get("name", "").split(":")[0] for m in models}
            available = sum(1 for c in AGENT_CONFIGS.values() if c["model"] in model_names)
            total = len(AGENT_CONFIGS)

            self.call_from_thread(
                self._post,
                f"[green]Ollama is running[/] at {self.ollama_url}\n"
                f"  Models: {available}/{total} agent models available",
            )
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            self.call_from_thread(self._post, f"[red]Cannot connect to Ollama: {reason}[/]")
        except Exception as e:
            self.call_from_thread(self._post, f"[red]Ollama error: {e}[/]")

    def _cmd_mode(self, arg: str) -> None:
        arg_lower = arg.lower().strip()
        if arg_lower == "flat":
            self.mode = "flat"
            self._post("[green]Switched to flat mode[/]")
        elif arg_lower in ("hierarchical", "hier"):
            self.mode = "hierarchical"
            self._post("[green]Switched to hierarchical mode[/]")
        else:
            self._post("[yellow]Usage: /mode flat  or  /mode hierarchical[/]")
            return
        indicator = self.query_one("#mode-indicator", ModeIndicator)
        indicator.update_info(self.mode, self._agents_str)

    def _cmd_use(self, arg: str) -> None:
        arg_stripped = arg.strip()
        if not arg_stripped:
            self._post("[yellow]Usage: /use agent1,agent2  or  /use all[/]")
            return
        if arg_stripped.lower() == "all":
            self.selected_agents = None
            self._post("[green]Using all specialists[/]")
        else:
            names = [a.strip() for a in arg_stripped.split(",")]
            invalid = [a for a in names if a not in AGENT_CONFIGS]
            if invalid:
                self._post(f"[red]Unknown agents: {invalid}[/]")
                self._post(f"Available: {', '.join(AGENT_CONFIGS.keys())}")
                return
            self.selected_agents = names
            self._post(f"[green]Using agents: {', '.join(names)}[/]")
        indicator = self.query_one("#mode-indicator", ModeIndicator)
        indicator.update_info(self.mode, self._agents_str)

    def _cmd_history(self) -> None:
        if not self._history:
            self._post("[dim]No tasks yet.[/]")
            return
        for i, entry in enumerate(self._history, 1):
            status = "[green]OK[/]" if entry["success"] else "[red]FAILED[/]"
            self._post(
                f"  {i}. [{entry['time']}] {status} [dim]({entry['mode']})[/] {entry['task']}"
            )

    def _cmd_save(self, arg: str) -> None:
        if not self._last_result:
            self._post("[yellow]No result to save.[/]")
            return
        if not arg.strip():
            self._post("[yellow]Usage: /save <filename>[/]")
            return
        raw_path = arg.strip()
        resolved = Path(raw_path).resolve()
        cwd = Path.cwd().resolve()
        if not str(resolved).startswith(str(cwd)):
            self._post(f"[yellow]Warning: writing outside current directory: {resolved}[/]")
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(str(self._last_result), encoding="utf-8")
            self._post(f"[green]Saved to {resolved}[/]")
        except Exception as e:
            self._post(f"[red]Save failed: {e}[/]")

    # --- Task execution ---

    @work(thread=True, exclusive=True)
    def _execute_task(self, task_description: str) -> None:
        """Run the swarm task in a background thread."""
        from .cli import (
            _prewarm_model,
            create_flat_crew,
            create_hierarchical_crew,
        )

        self._task_running = True
        sidebar = self.query_one("#agent-sidebar", AgentSidebar)
        self.call_from_thread(sidebar.reset_all)

        agents_str = self._agents_str
        self.call_from_thread(self._post, f"[dim]Running ({self.mode}, {agents_str})...[/]")

        # Disable CrewAI tracing
        os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

        def _make_agent_callback(agent_name, agent_role):
            """Create a per-agent step callback that knows which agent it belongs to."""

            def cb(step_output):
                try:
                    self.call_from_thread(sidebar.update_status, agent_name, "working")
                    self.call_from_thread(self._post_agent_step, agent_role)
                except Exception:
                    pass  # App may be shutting down

            return cb

        try:
            flat = self.mode == "flat"
            if flat:
                crew = create_flat_crew(
                    task_description,
                    self.ollama_url,
                    self.selected_agents,
                    agent_cache=self._agent_cache,
                )
                first_model = (self.selected_agents or SPECIALISTS)[0]
            else:
                crew = create_hierarchical_crew(
                    task_description,
                    self.ollama_url,
                    self.selected_agents,
                    agent_cache=self._agent_cache,
                )
                first_model = "orchestrator"

            # Assign per-agent step callbacks so each agent identifies itself
            for agent in crew.agents:
                role = agent.role
                name = _ROLE_TO_NAME.get(role, role)
                agent.step_callback = _make_agent_callback(name, role)

            # Pre-warm first model in background
            threading.Thread(
                target=_prewarm_model, args=(first_model, self.ollama_url), daemon=True
            ).start()

            result = crew.kickoff()
            self._last_result = result

            # Mark all active agents as done
            for name in self.selected_agents or SPECIALISTS:
                self.call_from_thread(sidebar.update_status, name, "done")
            if not flat:
                self.call_from_thread(sidebar.update_status, "orchestrator", "done")

            self.call_from_thread(
                self._post, f"\n[bold green]--- Result ---[/]\n{result}\n[bold green]--- End ---[/]"
            )

            self._history.append(
                {
                    "task": task_description,
                    "mode": self.mode,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "success": True,
                }
            )

        except Exception as e:
            self.call_from_thread(self._post, f"\n[red]Failed: {type(e).__name__}: {e}[/]")
            self._history.append(
                {
                    "task": task_description,
                    "mode": self.mode,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "success": False,
                }
            )

        finally:
            self._task_running = False


def run_tui(
    ollama_url: str = DEFAULT_OLLAMA_URL,
    flat: bool = False,
    agent_names: list[str] | None = None,
) -> None:
    """Launch the SwarmTUI application."""
    SwarmTUI(ollama_url=ollama_url, flat=flat, agent_names=agent_names).run()
