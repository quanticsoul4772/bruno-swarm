# SPDX-License-Identifier: AGPL-3.0-or-later
# Bruno Swarm CLI
#
# Production CLI for running multi-agent development swarms
# powered by abliterated Bruno models via CrewAI + Ollama.

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .logging import get_logger

logger = get_logger(__name__)

console = Console()

# Default Ollama URL
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# HuggingFace repo for pre-built GGUF models
HF_REPO = "rawcell/bruno-swarm-models"

# Model filenames on HuggingFace -> Ollama model name mapping
HF_MODELS = {
    "orchestrator-14b-f16.gguf": "orchestrator",
    "frontend-3b-f16.gguf": "frontend",
    "backend-3b-f16.gguf": "backend",
    "test-3b-f16.gguf": "test",
    "security-3b-f16.gguf": "security",
    "docs-3b-f16.gguf": "docs",
    "devops-3b-f16.gguf": "devops",
}

# Agent configurations: name -> (ollama_model, role, goal, backstory)
AGENT_CONFIGS = {
    "orchestrator": {
        "model": "orchestrator",
        "role": "Senior Software Architect",
        "goal": "Plan development tasks, design system architecture, and coordinate the team",
        "backstory": (
            "Senior architect with 20 years of experience. Thinks step by step, "
            "breaks complex problems into clear tasks, and delegates to specialists. "
            "Reviews all work for quality and architectural consistency. "
            "CRITICAL: You must delegate ONE task at a time to ONE coworker. "
            "Never send multiple delegations at once."
        ),
        "allow_delegation": True,
    },
    "frontend": {
        "model": "frontend",
        "role": "Frontend Developer",
        "goal": "Build responsive, user-friendly React components with TypeScript",
        "backstory": (
            "Expert in React, TypeScript, Tailwind CSS. Writes clean, concise "
            "code without over-engineering. Focuses on accessibility and UX. "
            "IMPORTANT: Output your code and explanation directly. "
            "Never simulate a conversation or generate User/Response patterns."
        ),
        "allow_delegation": False,
    },
    "backend": {
        "model": "backend",
        "role": "Backend Developer",
        "goal": "Create scalable FastAPI endpoints and database schemas",
        "backstory": (
            "Expert in FastAPI, PostgreSQL, async patterns. Focuses on clean "
            "architecture without premature optimization. Writes clear API contracts. "
            "IMPORTANT: Output your code and explanation directly. "
            "Never simulate a conversation or generate User/Response patterns."
        ),
        "allow_delegation": False,
    },
    "test": {
        "model": "test",
        "role": "QA Engineer",
        "goal": "Write comprehensive test suites with high coverage",
        "backstory": (
            "Expert in pytest, coverage analysis, edge cases. Proactively writes "
            "tests for all code paths including error handling and boundary conditions. "
            "IMPORTANT: Output your tests and explanation directly. "
            "Never simulate a conversation or generate User/Response patterns."
        ),
        "allow_delegation": False,
    },
    "security": {
        "model": "security",
        "role": "Security Engineer",
        "goal": "Identify vulnerabilities and enforce secure coding practices",
        "backstory": (
            "Expert in OWASP Top 10, penetration testing, secure code review. "
            "Paranoid about security -- catches issues others miss. Reviews all "
            "code for injection, auth bypass, and data exposure risks. "
            "IMPORTANT: Output your analysis directly. "
            "Never simulate a conversation or generate User/Response patterns."
        ),
        "allow_delegation": False,
    },
    "docs": {
        "model": "docs",
        "role": "Technical Writer",
        "goal": "Write clear API docs, README files, and developer guides",
        "backstory": (
            "Expert in technical documentation, API references, and developer "
            "onboarding. Writes concise docs without unnecessary jargon. "
            "Focuses on examples and practical usage. "
            "IMPORTANT: Output your documentation directly. "
            "Never simulate a conversation or generate User/Response patterns."
        ),
        "allow_delegation": False,
    },
    "devops": {
        "model": "devops",
        "role": "DevOps Engineer",
        "goal": "Create Docker configs, CI/CD pipelines, and deployment scripts",
        "backstory": (
            "Expert in Docker, GitHub Actions, infrastructure as code. "
            "Writes practical deployment configurations without overengineering. "
            "Focuses on reproducibility and security. "
            "IMPORTANT: Output your configurations directly. "
            "Never simulate a conversation or generate User/Response patterns."
        ),
        "allow_delegation": False,
    },
}

# Specialist agent names (all except orchestrator)
SPECIALISTS = ["frontend", "backend", "test", "security", "docs", "devops"]


def _check_crewai():
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


def create_hierarchical_crew(
    task_description: str,
    base_url: str,
    agent_names: list[str] | None = None,
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
    orchestrator = create_agent("orchestrator", base_url)
    agents = {"orchestrator": orchestrator}
    for name in agent_names:
        agents[name] = create_agent(name, base_url)

    # Role-specific task descriptions
    specialist_tasks = {
        "backend": (
            f"Design and implement the backend for: {task_description}\n"
            "Include API endpoints, schemas, and database models."
        ),
        "frontend": (
            f"Build the frontend components for: {task_description}\n"
            "Use React with TypeScript and Tailwind CSS."
        ),
        "test": (
            f"Write comprehensive tests for: {task_description}\n"
            "Use pytest with fixtures. Cover happy paths, edge cases, and error handling."
        ),
        "security": (
            f"Perform a security review of the implementation for: {task_description}\n"
            "Check for OWASP Top 10 vulnerabilities, auth issues, injection risks."
        ),
        "docs": (
            f"Write documentation for: {task_description}\n"
            "Include API reference, setup guide, and usage examples."
        ),
        "devops": (
            f"Create deployment configuration for: {task_description}\n"
            "Include Dockerfile, docker-compose.yml, and CI/CD pipeline."
        ),
    }

    specialist_outputs = {
        "backend": "Complete backend code with API endpoints and schemas",
        "frontend": "Complete React components with TypeScript types",
        "test": "Complete pytest test suite with fixtures and assertions",
        "security": "Security audit report with vulnerability fixes",
        "docs": "Complete documentation in Markdown format",
        "devops": "Dockerfile, docker-compose.yml, and CI/CD config",
    }

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
        if name in specialist_tasks:
            tasks.append(
                Task(
                    description=specialist_tasks[name],
                    agent=agents[name],
                    expected_output=specialist_outputs[name],
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

    def _step_callback(step_output):
        agent = getattr(step_output, "agent", None)
        agent_name = getattr(agent, "role", "Agent") if agent else "Agent"
        console.print(f"  [cyan]{agent_name}[/] completed a step")

    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
        step_callback=_step_callback,
    )


def create_flat_crew(
    task_description: str,
    base_url: str,
    agent_names: list[str] | None = None,
):
    """Create a flat sequential crew without orchestrator.

    Each specialist handles their portion of the task sequentially.
    """
    from crewai import Crew, Process, Task

    if agent_names is None:
        agent_names = SPECIALISTS

    agents = {}
    for name in agent_names:
        agents[name] = create_agent(name, base_url)

    # Role-specific task templates
    task_templates = {
        "backend": (
            "Design and implement the backend for: {task}\n"
            "Include API endpoints, schemas, and database models."
        ),
        "frontend": (
            "Build the frontend components for: {task}\n"
            "Use React with TypeScript and Tailwind CSS."
        ),
        "test": (
            "Write comprehensive tests for: {task}\n"
            "Use pytest with fixtures. Cover happy paths, edge cases, and error handling."
        ),
        "security": (
            "Perform a security review of the implementation for: {task}\n"
            "Check for OWASP Top 10 vulnerabilities, auth issues, injection risks."
        ),
        "docs": (
            "Write documentation for: {task}\n"
            "Include API reference, setup guide, and usage examples."
        ),
        "devops": (
            "Create deployment configuration for: {task}\n"
            "Include Dockerfile, docker-compose.yml, and CI/CD pipeline."
        ),
    }

    expected_outputs = {
        "backend": "Complete backend code with API endpoints and schemas",
        "frontend": "Complete React components with TypeScript types",
        "test": "Complete pytest test suite with fixtures and assertions",
        "security": "Security audit report with vulnerability fixes",
        "docs": "Complete documentation in Markdown format",
        "devops": "Dockerfile, docker-compose.yml, and CI/CD config",
    }

    tasks = []
    for name in agent_names:
        if name in task_templates:
            tasks.append(
                Task(
                    description=task_templates[name].format(task=task_description),
                    agent=agents[name],
                    expected_output=expected_outputs[name],
                )
            )

    def _step_callback(step_output):
        agent = getattr(step_output, "agent", None)
        agent_name = getattr(agent, "role", "Agent") if agent else "Agent"
        console.print(f"  [cyan]{agent_name}[/] completed a step")

    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
        step_callback=_step_callback,
    )


def _run_interactive(ollama_url: str, flat: bool, agent_names: list[str] | None):
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
                        console.print(
                            "[yellow]Usage: /mode flat  or  /mode hierarchical[/]"
                        )
                    continue

                elif cmd == "/use":
                    if cmd_arg.lower() == "all":
                        agent_names = None
                        console.print("[green]Using all specialists[/]")
                    else:
                        parsed = _parse_agents(cmd_arg)
                        if parsed:
                            agent_names = parsed
                            console.print(
                                f"[green]Using agents: {', '.join(agent_names)}[/]"
                            )
                    continue

                elif cmd == "/history":
                    if not history:
                        console.print("[dim]No tasks yet.[/]")
                    else:
                        for i, entry in enumerate(history, 1):
                            status = (
                                "[green]OK[/]" if entry["success"] else "[red]FAILED[/]"
                            )
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
                        save_path = Path(cmd_arg)
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        save_path.write_text(str(last_result), encoding="utf-8")
                        console.print(f"[green]Saved to {save_path}[/]")
                    continue

                else:
                    console.print(f"[yellow]Unknown command: {cmd}[/]")
                    continue

            # Execute task
            mode_str = "flat" if flat else "hierarchical"
            current_agents = (
                ", ".join(agent_names) if agent_names else "all specialists"
            )
            console.print()
            console.print(f"[dim]Running ({mode_str}, {current_agents})...[/]")
            console.print()

            try:
                if flat:
                    crew = create_flat_crew(task_input, ollama_url, agent_names)
                else:
                    crew = create_hierarchical_crew(task_input, ollama_url, agent_names)

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
def run_task(
    task: str, flat: bool, agents: str | None, ollama_url: str, output: str | None
):
    """Execute a development task with the agent swarm."""
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
        else:
            crew = create_hierarchical_crew(task, ollama_url, agent_names)

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
            output_path = Path(output)
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
    import json
    import subprocess
    import tempfile
    import urllib.error
    import urllib.request

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        console.print("[red]Error: huggingface-hub is not installed.[/]")
        console.print("Install it with: [cyan]pip install huggingface-hub[/]")
        sys.exit(1)

    # Check Ollama connectivity
    console.print(f"Checking Ollama at [cyan]{ollama_url}[/]...")
    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, Exception) as e:
        console.print(f"[red]Cannot connect to Ollama: {e}[/]")
        console.print("Make sure Ollama is running: [cyan]ollama serve[/]")
        sys.exit(1)

    existing_models = {m.get("name", "").split(":")[0] for m in data.get("models", [])}

    # Filter models if specified
    if models:
        requested = {m.strip() for m in models.split(",")}
        download_map = {
            k: v for k, v in HF_MODELS.items() if v in requested
        }
        if not download_map:
            console.print(f"[red]No matching models found for: {models}[/]")
            console.print(f"Available: {', '.join(HF_MODELS.values())}")
            sys.exit(1)
    else:
        download_map = HF_MODELS

    # Find modelfiles directory (relative to package)
    package_dir = Path(__file__).parent.parent.parent
    modelfiles_dir = package_dir / "modelfiles"

    console.print()
    console.print(f"[bold]Downloading {len(download_map)} models from {HF_REPO}[/]")
    console.print()

    for filename, ollama_name in download_map.items():
        if ollama_name in existing_models:
            console.print(f"  [dim]{ollama_name}[/] -- already in Ollama, skipping")
            continue

        console.print(f"  [cyan]{ollama_name}[/] -- downloading {filename}...")

        try:
            # Download GGUF from HuggingFace
            gguf_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
            )

            # Determine system prompt and params based on agent type
            is_orchestrator = ollama_name == "orchestrator"
            system_prompts = {
                "orchestrator": "You are a Senior Software Architect and Project Manager. Plan development tasks, design system architecture, delegate work to specialists, and review code quality. Think step by step before delegating.",
                "frontend": "You are a Frontend Developer specializing in React, TypeScript, and Tailwind CSS. Write clean, concise code without over-engineering.",
                "backend": "You are a Backend Developer specializing in FastAPI, PostgreSQL, and async patterns. Focus on clean architecture without premature optimization.",
                "test": "You are a QA Engineer specializing in pytest, coverage analysis, and edge case testing. Proactively write comprehensive tests for all code.",
                "security": "You are a Security Engineer specializing in vulnerability assessment, OWASP Top 10, and secure coding patterns. Identify security issues aggressively and recommend hardened implementations.",
                "docs": "You are a Technical Writer specializing in API documentation, README files, and developer guides. Write clear, concise documentation without unnecessary jargon.",
                "devops": "You are a DevOps Engineer specializing in Docker, CI/CD pipelines, and infrastructure as code. Write practical deployment configurations without overengineering.",
            }

            num_predict = "4096" if is_orchestrator else "2048"

            # Create a temporary Modelfile pointing to the downloaded GGUF
            modelfile_content = (
                f"FROM {gguf_path}\n\n"
                f"SYSTEM {system_prompts.get(ollama_name, '')}\n\n"
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
                # Import into Ollama
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
            console.print(f"  [red]{ollama_name}[/] -- failed: {e}")

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
    console.print(
        "[dim]Hierarchical mode uses orchestrator as manager + specialists[/]"
    )


@cli.command("status")
@click.option(
    "--ollama-url",
    "-u",
    default=DEFAULT_OLLAMA_URL,
    help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
)
def check_status(ollama_url: str):
    """Check Ollama connectivity and loaded models."""
    import urllib.error
    import urllib.request

    console.print()
    console.print(f"Checking Ollama at [cyan]{ollama_url}[/]...")
    console.print()

    # Check Ollama connectivity
    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            import json

            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        console.print(f"[red]Cannot connect to Ollama: {e.reason}[/]")
        console.print()
        console.print("Make sure Ollama is running:")
        console.print("  [cyan]ollama serve[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error connecting to Ollama: {e}[/]")
        sys.exit(1)

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
            console.print(
                "Run [cyan]bruno-swarm setup[/] to download and install missing models."
            )

    # Show loaded models (currently in memory)
    try:
        req = urllib.request.Request(f"{ollama_url}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            import json

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


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
