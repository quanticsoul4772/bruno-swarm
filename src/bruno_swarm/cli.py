# SPDX-License-Identifier: AGPL-3.0-or-later
# Bruno Swarm CLI
#
# Production CLI for running multi-agent development swarms
# powered by abliterated Bruno models via CrewAI + Ollama.

import os
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import (
    AGENT_CONFIGS,
    DEFAULT_OLLAMA_URL,
    EXPECTED_OUTPUTS,
    HF_MODELS,
    HF_REPO,
    SPECIALISTS,
    TASK_TEMPLATES,
    make_step_callback,
)
from .logging import get_logger

logger = get_logger(__name__)

console = Console()


def _check_crewai() -> bool:
    """Check that CrewAI is installed, exit with install instructions if not."""
    try:
        import crewai  # noqa: F401

        return True
    except ImportError:
        console.print("[red]Error: CrewAI is not installed.[/]")
        console.print()
        console.print("Install it with:")
        console.print("  [cyan]pip install bruno-swarm[/]")
        console.print()
        console.print("[dim]CrewAI is included in bruno-swarm's dependencies.[/]")
        sys.exit(1)


def _parse_agents(agents_str: str | None) -> list[str] | None:
    """Parse comma-separated agent names, validate, and return list or None."""
    if not agents_str:
        return None

    agent_names = [a.strip() for a in agents_str.split(",")]
    invalid = [a for a in agent_names if a not in AGENT_CONFIGS]
    if invalid:
        console.print(f"[red]Unknown agents: {invalid}[/]")
        console.print(f"Available: {', '.join(AGENT_CONFIGS.keys())}")
        sys.exit(1)

    return agent_names


def _validate_output_path(raw_path: str) -> Path:
    """Resolve an output path and warn if it escapes the current directory."""
    resolved = Path(raw_path).resolve()
    cwd = Path.cwd().resolve()
    if not str(resolved).startswith(str(cwd)):
        console.print(f"[yellow]Warning: writing outside current directory: {resolved}[/]")
    return resolved


def _validate_ollama_url(url: str) -> str:
    """Validate that the Ollama URL uses http or https scheme."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        console.print(
            f"[red]Error: --ollama-url must use http:// or https://, got: {parsed.scheme}://[/]"
        )
        sys.exit(1)
    return url


def _ollama_api_get(ollama_url: str, endpoint: str) -> dict:
    """GET an Ollama API endpoint and return parsed JSON, or exit on failure."""
    import json
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(f"{ollama_url}{endpoint}", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        console.print(f"[red]Cannot connect to Ollama: {reason}[/]")
        console.print("Make sure Ollama is running: [cyan]ollama serve[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error connecting to Ollama: {e}[/]")
        sys.exit(1)


def create_llm(model_name: str, base_url: str):
    """Create a CrewAI LLM instance for an Ollama model."""
    from crewai import LLM

    return LLM(
        model=f"ollama/{model_name}",
        base_url=base_url,
        timeout=1200,
        max_retries=3,
    )


def create_agent(name: str, base_url: str):
    """Create a CrewAI agent from config."""
    from crewai import Agent

    config = AGENT_CONFIGS[name]
    return Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=create_llm(config["model"], base_url),
        verbose=False,
        allow_delegation=config["allow_delegation"],
        max_iter=10,
        max_retry_limit=3,
    )


def _get_or_create_agent(name: str, base_url: str, cache: dict):
    """Return a cached Agent or create and cache a new one. Key: (name, base_url)."""
    key = (name, base_url)
    if key not in cache:
        cache[key] = create_agent(name, base_url)
    return cache[key]


def _prewarm_model(model_name: str, base_url: str) -> None:
    """Best-effort: send minimal /api/generate to preload model into Ollama memory."""
    import json
    import urllib.request

    try:
        payload = json.dumps({"model": model_name, "prompt": ".", "stream": False}).encode()
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120):
            pass
    except Exception:
        pass  # Pre-warming is best-effort


def create_hierarchical_crew(
    task_description: str,
    base_url: str,
    agent_names: list[str] | None = None,
    agent_cache: dict | None = None,
    step_callback=None,
):
    """Create a hierarchical crew using sequential process.

    CrewAI's Process.hierarchical is broken (manager doesn't properly
    route tasks). Instead, use sequential with the orchestrator as
    first and last agent: plan -> specialists -> review.
    """
    from crewai import Crew, Process, Task

    if agent_names is None:
        agent_names = SPECIALISTS

    # Create all agents including orchestrator
    _make = (
        (lambda n, u: _get_or_create_agent(n, u, agent_cache))
        if agent_cache is not None
        else create_agent
    )
    orchestrator = _make("orchestrator", base_url)
    agents = {"orchestrator": orchestrator}
    for name in agent_names:
        agents[name] = _make(name, base_url)

    tasks = []

    # Task 1: Orchestrator plans the architecture
    tasks.append(
        Task(
            description=(
                f"Plan the architecture for: {task_description}\n\n"
                "Break this into components and define the technical approach "
                "for each part. Output a clear plan with file structure, "
                "technology choices, and implementation order."
            ),
            agent=orchestrator,
            expected_output="Architecture plan with components and implementation order",
        )
    )

    # Tasks 2-N: Each specialist implements their part
    for name in agent_names:
        if name in TASK_TEMPLATES:
            tasks.append(
                Task(
                    description=TASK_TEMPLATES[name].format(task=task_description),
                    agent=agents[name],
                    expected_output=EXPECTED_OUTPUTS[name],
                )
            )

    # Final task: Orchestrator reviews and integrates
    tasks.append(
        Task(
            description=(
                f"Review all the work produced by the team for: {task_description}\n\n"
                "Integrate all components into a cohesive implementation. "
                "Identify any gaps, inconsistencies, or issues. "
                "Output the complete integrated implementation."
            ),
            agent=orchestrator,
            expected_output="Complete integrated implementation with all components",
        )
    )

    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
        step_callback=step_callback if step_callback is not None else make_step_callback(console),
    )


def create_flat_crew(
    task_description: str,
    base_url: str,
    agent_names: list[str] | None = None,
    agent_cache: dict | None = None,
    step_callback=None,
):
    """Create a flat sequential crew without orchestrator.

    Each specialist handles their portion of the task sequentially.
    """
    from crewai import Crew, Process, Task

    if agent_names is None:
        agent_names = SPECIALISTS

    _make = (
        (lambda n, u: _get_or_create_agent(n, u, agent_cache))
        if agent_cache is not None
        else create_agent
    )
    agents = {}
    for name in agent_names:
        agents[name] = _make(name, base_url)

    tasks = []
    for name in agent_names:
        if name in TASK_TEMPLATES:
            tasks.append(
                Task(
                    description=TASK_TEMPLATES[name].format(task=task_description),
                    agent=agents[name],
                    expected_output=EXPECTED_OUTPUTS[name],
                )
            )

    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
        step_callback=step_callback if step_callback is not None else make_step_callback(console),
    )


def _run_interactive(ollama_url: str, flat: bool, agent_names: list[str] | None) -> None:
    """Interactive TUI mode -- like Claude Code.

    Presents a prompt loop where the user types tasks and sees results.
    Supports task history, agent selection, and mode switching.
    """
    from datetime import datetime

    from rich.prompt import Prompt

    _check_crewai()

    # Disable CrewAI tracing
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

    mode = "flat" if flat else "hierarchical"
    agents_str = ", ".join(agent_names) if agent_names else "all specialists"
    history: list[dict] = []
    agent_cache: dict = {}  # (name, base_url) -> Agent

    # Welcome banner
    console.print()
    console.print(
        Panel(
            "[bold]Bruno AI Developer Swarm[/]\n\n"
            "Interactive mode -- type a task and the swarm will execute it.\n\n"
            "Commands:\n"
            "  [cyan]/agents[/]          -- list available agents\n"
            "  [cyan]/status[/]          -- check Ollama and model status\n"
            "  [cyan]/mode flat[/]       -- switch to flat mode\n"
            "  [cyan]/mode hierarchical[/] -- switch to hierarchical mode\n"
            "  [cyan]/use agent1,agent2[/] -- select specific agents\n"
            "  [cyan]/use all[/]         -- use all specialists\n"
            "  [cyan]/history[/]         -- show task history\n"
            "  [cyan]/save <file>[/]     -- save last result to file\n"
            "  [cyan]/quit[/]            -- exit\n\n"
            f"Mode: [cyan]{mode}[/]  |  Agents: [cyan]{agents_str}[/]  |  Ollama: [cyan]{ollama_url}[/]",
            title="Swarm",
            border_style="cyan",
        )
    )
    console.print()

    last_result = None

    while True:
        try:
            # Prompt
            mode_label = "[dim]flat[/]" if flat else "[dim]hier[/]"
            task_input = Prompt.ask(f"[bold cyan]swarm[/] {mode_label}")

            if not task_input.strip():
                continue

            task_input = task_input.strip()

            # Handle commands
            if task_input.startswith("/"):
                cmd_parts = task_input.split(None, 1)
                cmd = cmd_parts[0].lower()
                cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                if cmd in ("/quit", "/exit", "/q"):
                    console.print("[dim]Goodbye.[/]")
                    break

                elif cmd == "/agents":
                    list_agents.invoke(click.Context(list_agents))
                    continue

                elif cmd == "/status":
                    ctx = click.Context(check_status)
                    ctx.params = {"ollama_url": ollama_url}
                    check_status.invoke(ctx)
                    continue

                elif cmd == "/mode":
                    if cmd_arg.lower() == "flat":
                        flat = True
                        console.print("[green]Switched to flat mode[/]")
                    elif cmd_arg.lower() in ("hierarchical", "hier"):
                        flat = False
                        console.print("[green]Switched to hierarchical mode[/]")
                    else:
                        console.print("[yellow]Usage: /mode flat  or  /mode hierarchical[/]")
                    continue

                elif cmd == "/use":
                    if cmd_arg.lower() == "all":
                        agent_names = None
                        console.print("[green]Using all specialists[/]")
                    else:
                        parsed = _parse_agents(cmd_arg)
                        if parsed:
                            agent_names = parsed
                            console.print(f"[green]Using agents: {', '.join(agent_names)}[/]")
                    continue

                elif cmd == "/history":
                    if not history:
                        console.print("[dim]No tasks yet.[/]")
                    else:
                        for i, entry in enumerate(history, 1):
                            status = "[green]OK[/]" if entry["success"] else "[red]FAILED[/]"
                            console.print(
                                f"  {i}. [{entry['time']}] {status} "
                                f"[dim]({entry['mode']})[/] {entry['task']}"
                            )
                    continue

                elif cmd == "/save":
                    if not last_result:
                        console.print("[yellow]No result to save.[/]")
                    elif not cmd_arg:
                        console.print("[yellow]Usage: /save <filename>[/]")
                    else:
                        save_path = _validate_output_path(cmd_arg)
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        save_path.write_text(str(last_result), encoding="utf-8")
                        console.print(f"[green]Saved to {save_path}[/]")
                    continue

                else:
                    console.print(f"[yellow]Unknown command: {cmd}[/]")
                    continue

            # Execute task
            mode_str = "flat" if flat else "hierarchical"
            current_agents = ", ".join(agent_names) if agent_names else "all specialists"
            console.print()
            console.print(f"[dim]Running ({mode_str}, {current_agents})...[/]")
            console.print()

            try:
                if flat:
                    crew = create_flat_crew(
                        task_input, ollama_url, agent_names, agent_cache=agent_cache
                    )
                    first_model = (agent_names or SPECIALISTS)[0]
                else:
                    crew = create_hierarchical_crew(
                        task_input, ollama_url, agent_names, agent_cache=agent_cache
                    )
                    first_model = "orchestrator"

                # Pre-warm first model in background while crew starts
                threading.Thread(
                    target=_prewarm_model, args=(first_model, ollama_url), daemon=True
                ).start()

                result = crew.kickoff()
                last_result = result

                console.print()
                console.print(
                    Panel(
                        str(result),
                        title="[bold green]Result[/]",
                        border_style="green",
                    )
                )

                history.append(
                    {
                        "task": task_input,
                        "mode": mode_str,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "success": True,
                    }
                )

            except Exception as e:
                logger.error("Swarm execution failed", exc_info=True)
                console.print(f"\n[red]Failed: {type(e).__name__}: {e}[/]")
                history.append(
                    {
                        "task": task_input,
                        "mode": mode_str,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "success": False,
                    }
                )

            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Ctrl+C -- type /quit to exit[/]")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye.[/]")
            break


# CLI Commands
@click.group(invoke_without_command=True)
@click.option(
    "--ollama-url",
    "-u",
    default=DEFAULT_OLLAMA_URL,
    help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
)
@click.option("--flat", is_flag=True, help="Start interactive mode in flat mode")
@click.option(
    "--agents",
    "-a",
    default=None,
    help="Comma-separated agent names for interactive mode",
)
@click.pass_context
def cli(ctx, ollama_url: str, flat: bool, agents: str | None):
    """Bruno AI Developer Swarm CLI.

    Multi-agent development team powered by abliterated Bruno models.
    Uses CrewAI for orchestration and Ollama for local inference.

    Run without a subcommand to enter interactive mode.
    """
    ctx.ensure_object(dict)
    ctx.obj["ollama_url"] = ollama_url
    if ctx.invoked_subcommand is None:
        _validate_ollama_url(ollama_url)
        agent_names = _parse_agents(agents)
        _run_interactive(ollama_url, flat=flat, agent_names=agent_names)


@cli.command("run")
@click.option("--task", "-t", required=True, help="Development task to execute")
@click.option("--flat", is_flag=True, help="Use flat sequential mode (no orchestrator)")
@click.option(
    "--agents",
    "-a",
    default=None,
    help="Comma-separated agent names (default: all specialists)",
)
@click.option(
    "--ollama-url",
    "-u",
    default=DEFAULT_OLLAMA_URL,
    help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Save result to file",
)
def run_task(task: str, flat: bool, agents: str | None, ollama_url: str, output: str | None):
    """Execute a development task with the agent swarm."""
    _validate_ollama_url(ollama_url)
    _check_crewai()

    # Disable CrewAI tracing in non-interactive mode
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

    agent_names = _parse_agents(agents)
    mode = "flat" if flat else "hierarchical"
    agents_str = ", ".join(agent_names) if agent_names else "all specialists"

    console.print()
    console.print(
        Panel.fit(
            f"[bold]Bruno AI Developer Swarm[/]\n\n"
            f"Mode: [cyan]{mode}[/]\n"
            f"Agents: [cyan]{agents_str}[/]\n"
            f"Ollama: [cyan]{ollama_url}[/]\n"
            f"Task: [cyan]{task}[/]",
            title="Swarm",
            border_style="cyan",
        )
    )

    try:
        if flat:
            crew = create_flat_crew(task, ollama_url, agent_names)
            first_model = (agent_names or SPECIALISTS)[0]
        else:
            crew = create_hierarchical_crew(task, ollama_url, agent_names)
            first_model = "orchestrator"

        # Pre-warm first model in background while crew starts
        threading.Thread(target=_prewarm_model, args=(first_model, ollama_url), daemon=True).start()

        result = crew.kickoff()

        console.print()
        console.print(
            Panel(
                str(result),
                title="[bold green]Swarm Result[/]",
                border_style="green",
            )
        )

        if output:
            output_path = _validate_output_path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(str(result), encoding="utf-8")
            console.print(f"\nResult saved to [cyan]{output_path}[/]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")
        sys.exit(0)
    except Exception as e:
        logger.error("Swarm execution failed", exc_info=True)
        console.print(f"\n[red]Swarm failed: {type(e).__name__}: {e}[/]")
        sys.exit(1)


def _download_model(
    filename, ollama_name, hf_hub_download, console, console_lock
) -> tuple[str, str | None]:
    """Download a single GGUF from HuggingFace. Returns (ollama_name, gguf_path | None)."""
    with console_lock:
        console.print(f"  [cyan]{ollama_name}[/] -- downloading {filename}...")
    try:
        gguf_path = hf_hub_download(repo_id=HF_REPO, filename=filename)
        with console_lock:
            console.print(f"  [green]{ollama_name}[/] -- download complete")
        return (ollama_name, gguf_path)
    except Exception as e:
        with console_lock:
            console.print(f"  [red]{ollama_name}[/] -- download failed: {e}")
        return (ollama_name, None)


@cli.command("setup")
@click.option(
    "--ollama-url",
    "-u",
    default=DEFAULT_OLLAMA_URL,
    help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
)
@click.option(
    "--models",
    "-m",
    default=None,
    help="Comma-separated model names to download (default: all)",
)
def setup_models(ollama_url: str, models: str | None):
    """Download models from HuggingFace and import into Ollama.

    Downloads pre-built GGUF models from rawcell/bruno-swarm-models
    and creates Ollama models using the bundled Modelfiles.
    """
    import subprocess
    import tempfile

    _validate_ollama_url(ollama_url)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        console.print("[red]Error: huggingface-hub is not installed.[/]")
        console.print("Install it with: [cyan]pip install huggingface-hub[/]")
        sys.exit(1)

    # Check Ollama connectivity
    console.print(f"Checking Ollama at [cyan]{ollama_url}[/]...")
    data = _ollama_api_get(ollama_url, "/api/tags")

    existing_models = {m.get("name", "").split(":")[0] for m in data.get("models", [])}

    # Filter models if specified
    if models:
        requested = {m.strip() for m in models.split(",")}
        download_map = {k: v for k, v in HF_MODELS.items() if v in requested}
        if not download_map:
            console.print(f"[red]No matching models found for: {models}[/]")
            console.print(f"Available: {', '.join(HF_MODELS.values())}")
            sys.exit(1)
    else:
        download_map = HF_MODELS

    console.print()
    console.print(f"[bold]Downloading {len(download_map)} models from {HF_REPO}[/]")
    console.print()

    # Report skipped models
    for filename, ollama_name in download_map.items():
        if ollama_name in existing_models:
            console.print(f"  [dim]{ollama_name}[/] -- already in Ollama, skipping")

    to_download = {fn: name for fn, name in download_map.items() if name not in existing_models}

    # Phase 1: Parallel downloads from HuggingFace
    downloaded: dict[str, str] = {}  # ollama_name -> gguf_path
    if to_download:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        console_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    _download_model, fn, name, hf_hub_download, console, console_lock
                ): name
                for fn, name in to_download.items()
            }
            for future in as_completed(futures):
                ollama_name, gguf_path = future.result()
                if gguf_path:
                    downloaded[ollama_name] = gguf_path

    # Phase 2: Sequential Ollama imports (Ollama DB is single-writer)
    for ollama_name, gguf_path in downloaded.items():
        try:
            is_orchestrator = ollama_name == "orchestrator"
            system_prompt = AGENT_CONFIGS[ollama_name]["system_prompt"]
            num_predict = "4096" if is_orchestrator else "2048"

            modelfile_content = (
                f"FROM {gguf_path}\n\n"
                f"SYSTEM {system_prompt}\n\n"
                f"PARAMETER temperature 0.7\n"
                f"PARAMETER top_p 0.9\n"
                f"PARAMETER top_k 40\n"
                f"PARAMETER num_ctx 8192\n"
                f"PARAMETER num_predict {num_predict}\n"
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".Modelfile", delete=False, encoding="utf-8"
            ) as f:
                f.write(modelfile_content)
                temp_modelfile = f.name

            try:
                console.print(f"  [cyan]{ollama_name}[/] -- importing into Ollama...")
                result = subprocess.run(
                    ["ollama", "create", ollama_name, "-f", temp_modelfile],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode == 0:
                    console.print(f"  [green]{ollama_name}[/] -- imported successfully")
                else:
                    console.print(
                        f"  [red]{ollama_name}[/] -- import failed: {result.stderr.strip()}"
                    )
            finally:
                Path(temp_modelfile).unlink(missing_ok=True)

        except Exception as e:
            console.print(f"  [red]{ollama_name}[/] -- import failed: {e}")

    console.print()
    console.print("[green]Setup complete.[/] Run [cyan]bruno-swarm status[/] to verify.")


@cli.command("agents")
def list_agents():
    """List all available swarm agents and their roles."""
    table = Table(title="Bruno Swarm Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Role", style="bold")
    table.add_column("Ollama Model", style="dim")
    table.add_column("Goal")

    for name, config in AGENT_CONFIGS.items():
        table.add_row(
            name,
            config["role"],
            config["model"],
            config["goal"],
        )

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Specialists (flat mode):[/]", ", ".join(SPECIALISTS))
    console.print("[dim]Hierarchical mode uses orchestrator as manager + specialists[/]")


@cli.command("status")
@click.option(
    "--ollama-url",
    "-u",
    default=DEFAULT_OLLAMA_URL,
    help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
)
def check_status(ollama_url: str):
    """Check Ollama connectivity and loaded models."""
    import json
    import urllib.request

    _validate_ollama_url(ollama_url)

    console.print()
    console.print(f"Checking Ollama at [cyan]{ollama_url}[/]...")
    console.print()

    data = _ollama_api_get(ollama_url, "/api/tags")
    console.print("[green]Ollama is running[/]")
    console.print()

    # List available models and check which swarm agents are present
    models = data.get("models", [])
    model_names = {m.get("name", "").split(":")[0] for m in models}

    table = Table(title="Swarm Agent Models")
    table.add_column("Agent", style="cyan")
    table.add_column("Ollama Model")
    table.add_column("Status")
    table.add_column("Size", justify="right")

    for name, config in AGENT_CONFIGS.items():
        ollama_model = config["model"]
        if ollama_model in model_names:
            # Find the model entry for size info
            size_str = ""
            for m in models:
                if m.get("name", "").split(":")[0] == ollama_model:
                    size_bytes = m.get("size", 0)
                    if size_bytes > 0:
                        size_str = f"{size_bytes / (1024**3):.1f} GB"
                    break
            table.add_row(name, ollama_model, "[green]available[/]", size_str)
        else:
            table.add_row(name, ollama_model, "[red]missing[/]", "")

    console.print(table)

    # Summary
    available = sum(1 for c in AGENT_CONFIGS.values() if c["model"] in model_names)
    total = len(AGENT_CONFIGS)
    console.print()
    if available == total:
        console.print(f"[green]All {total} agent models available[/]")
    else:
        console.print(f"[yellow]{available}/{total} agent models available[/]")
        missing = [n for n, c in AGENT_CONFIGS.items() if c["model"] not in model_names]
        console.print(f"[dim]Missing: {', '.join(missing)}[/]")
        if available < total:
            console.print()
            console.print("Run [cyan]bruno-swarm setup[/] to download and install missing models.")

    # Show loaded models (currently in memory)
    try:
        req = urllib.request.Request(f"{ollama_url}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            ps_data = json.loads(resp.read().decode())
            running = ps_data.get("models", [])
            if running:
                console.print()
                console.print("[bold]Currently loaded in memory:[/]")
                for m in running:
                    name = m.get("name", "unknown")
                    size_bytes = m.get("size", 0)
                    size_str = f"{size_bytes / (1024**3):.1f} GB" if size_bytes else ""
                    console.print(f"  [green]{name}[/] {size_str}")
    except Exception:
        pass  # /api/ps may not be available in older Ollama versions


@cli.command("tui")
@click.option(
    "--ollama-url",
    "-u",
    default=DEFAULT_OLLAMA_URL,
    help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
)
@click.option("--flat", is_flag=True, help="Start in flat mode (no orchestrator)")
@click.option(
    "--agents",
    "-a",
    default=None,
    help="Comma-separated agent names",
)
def launch_tui(ollama_url: str, flat: bool, agents: str | None):
    """Launch the interactive chat TUI (requires textual).

    A full-screen terminal interface for running the agent swarm.
    Install textual with: pip install bruno-swarm[tui]
    """
    _validate_ollama_url(ollama_url)
    agent_names = _parse_agents(agents)
    try:
        from .tui import run_tui
    except ImportError:
        console.print("[red]Textual is not installed.[/]")
        console.print("Install with: [cyan]pip install bruno-swarm[tui][/]")
        sys.exit(1)
    run_tui(ollama_url=ollama_url, flat=flat, agent_names=agent_names)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
