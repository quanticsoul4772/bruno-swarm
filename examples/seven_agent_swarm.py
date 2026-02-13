#!/usr/bin/env python3
"""Seven-agent development swarm using abliterated Bruno models via CrewAI.

Production swarm team:
- Orchestrator (14B): Project planning, architecture, task delegation
- Frontend (3B): React/TypeScript/Tailwind specialist
- Backend (3B): FastAPI/PostgreSQL specialist
- Testing (3B): pytest/coverage specialist
- Security (3B): Vulnerability assessment specialist
- Docs (3B): Technical documentation specialist
- DevOps (3B): Docker/CI-CD specialist

Usage:
    # Hierarchical mode (with 14B orchestrator as manager)
    python seven_agent_swarm.py --task "Build user authentication system"

    # Flat mode (6 specialists, no orchestrator)
    python seven_agent_swarm.py --task "Build user authentication system" --flat

    # Specific agents only
    python seven_agent_swarm.py --task "Fix SQL injection" --agents security,backend

    # Custom Ollama URL
    python seven_agent_swarm.py --task "Build API" --ollama-url http://remote:11434

NOTE: This is a standalone example script. For production use, install
bruno-swarm and use the CLI: bruno-swarm run --task "..."
"""

import argparse
import os
import sys
import traceback

from crewai import LLM, Agent, Crew, Process, Task

# Import configs from the package (avoids duplicating 100+ lines of agent definitions).
# If running standalone without bruno-swarm installed, this will fail with ImportError.
from bruno_swarm.config import AGENT_CONFIGS, SPECIALISTS

# Disable CrewAI tracing prompt in non-interactive mode
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")


def create_llm(model_name: str, base_url: str) -> LLM:
    """Create a CrewAI LLM instance for an Ollama model."""
    return LLM(
        model=f"ollama/{model_name}",
        base_url=base_url,
        timeout=1200,
        max_retries=3,
    )


def create_agent(name: str, base_url: str) -> Agent:
    """Create a CrewAI agent from config."""
    config = AGENT_CONFIGS[name]
    return Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=create_llm(config["model"], base_url),
        verbose=True,
        allow_delegation=config["allow_delegation"],
        max_iter=10,
        max_retry_limit=3,
    )


def create_hierarchical_crew(
    task_description: str,
    base_url: str,
    agent_names: list[str] | None = None,
) -> Crew:
    """Create a hierarchical crew with 14B orchestrator as manager.

    The orchestrator plans and delegates to specialist agents.
    """
    if agent_names is None:
        agent_names = SPECIALISTS

    # Create orchestrator as manager
    manager = create_agent("orchestrator", base_url)

    # Create specialist agents
    agents = [create_agent(name, base_url) for name in agent_names]

    # Single high-level task -- orchestrator delegates to specialists
    task = Task(
        description=(
            f"{task_description}\n\n"
            "Break this into subtasks and delegate to the appropriate specialists. "
            "Each specialist should return ONLY code, no explanations. "
            "Review all outputs for quality and consistency before finalizing."
        ),
        expected_output="Complete implementation with all components integrated",
    )

    return Crew(
        agents=agents,
        tasks=[task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True,
    )


def create_flat_crew(
    task_description: str,
    base_url: str,
    agent_names: list[str] | None = None,
) -> Crew:
    """Create a flat sequential crew without orchestrator.

    Each specialist handles their portion of the task sequentially.
    """
    if agent_names is None:
        agent_names = SPECIALISTS

    agents = {}
    for name in agent_names:
        agents[name] = create_agent(name, base_url)

    # Create role-specific tasks
    task_templates = {
        "backend": (
            "Design and implement the backend for: {task}\n"
            "Include API endpoints, schemas, and database models. "
            "Return ONLY the code, no explanations."
        ),
        "frontend": (
            "Build the frontend components for: {task}\n"
            "Use React with TypeScript and Tailwind CSS. "
            "Return ONLY the code, no explanations."
        ),
        "test": (
            "Write comprehensive tests for: {task}\n"
            "Use pytest with fixtures. Cover happy paths, edge cases, and error handling. "
            "Return ONLY the code, no explanations."
        ),
        "security": (
            "Perform a security review of the implementation for: {task}\n"
            "Check for OWASP Top 10 vulnerabilities, auth issues, injection risks. "
            "Return findings and fixed code snippets."
        ),
        "docs": (
            "Write documentation for: {task}\n"
            "Include API reference, setup guide, and usage examples. "
            "Return ONLY the documentation in Markdown."
        ),
        "devops": (
            "Create deployment configuration for: {task}\n"
            "Include Dockerfile, docker-compose.yml, and CI/CD pipeline. "
            "Return ONLY the configuration files."
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

    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Bruno AI Developer Swarm -- 7-agent team powered by abliterated models"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Development task to execute",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Use flat sequential mode (no orchestrator)",
    )
    parser.add_argument(
        "--agents",
        default=None,
        help="Comma-separated agent names (default: all specialists)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    args = parser.parse_args()

    # Parse agent selection
    agent_names = None
    if args.agents:
        agent_names = [a.strip() for a in args.agents.split(",")]
        invalid = [a for a in agent_names if a not in AGENT_CONFIGS]
        if invalid:
            print(f"Unknown agents: {invalid}")
            print(f"Available: {list(AGENT_CONFIGS.keys())}")
            sys.exit(1)

    mode = "flat" if args.flat else "hierarchical"
    agents_str = ", ".join(agent_names) if agent_names else "all specialists"

    print("=" * 80)
    print("BRUNO AI DEVELOPER SWARM")
    print(f"Mode: {mode}")
    print(f"Agents: {agents_str}")
    print(f"Ollama: {args.ollama_url}")
    print(f"Task: {args.task}")
    print("=" * 80)

    if args.flat:
        crew = create_flat_crew(args.task, args.ollama_url, agent_names)
    else:
        crew = create_hierarchical_crew(args.task, args.ollama_url, agent_names)

    try:
        result = crew.kickoff()
        print("\n" + "=" * 80)
        print("SWARM RESULT")
        print("=" * 80)
        print(result)
        print("=" * 80)
        return result
    except Exception as e:
        print(f"\nSWARM FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        result = main()
        if result:
            print("\nSUCCESS -- All agents completed their tasks!")
        else:
            print("\nFAILED -- Check errors above")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
