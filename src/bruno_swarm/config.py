# SPDX-License-Identifier: AGPL-3.0-or-later
# Bruno Swarm Configuration
#
# Shared constants, agent configs, task templates, and callback factories.
# Single source of truth — imported by cli.py and tests.

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

# Agent configurations: name -> model, role, goal, backstory, system_prompt, delegation flag
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
        "system_prompt": (
            "You are a Senior Software Architect and Project Manager. Plan development "
            "tasks, design system architecture, delegate work to specialists, and review "
            "code quality. Think step by step before delegating."
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
        "system_prompt": (
            "You are a Frontend Developer specializing in React, TypeScript, and "
            "Tailwind CSS. Write clean, concise code without over-engineering."
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
        "system_prompt": (
            "You are a Backend Developer specializing in FastAPI, PostgreSQL, and "
            "async patterns. Focus on clean architecture without premature optimization."
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
        "system_prompt": (
            "You are a QA Engineer specializing in pytest, coverage analysis, and "
            "edge case testing. Proactively write comprehensive tests for all code."
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
        "system_prompt": (
            "You are a Security Engineer specializing in vulnerability assessment, "
            "OWASP Top 10, and secure coding patterns. Identify security issues "
            "aggressively and recommend hardened implementations."
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
        "system_prompt": (
            "You are a Technical Writer specializing in API documentation, README "
            "files, and developer guides. Write clear, concise documentation without "
            "unnecessary jargon."
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
        "system_prompt": (
            "You are a DevOps Engineer specializing in Docker, CI/CD pipelines, and "
            "infrastructure as code. Write practical deployment configurations without "
            "overengineering."
        ),
        "allow_delegation": False,
    },
}

# Specialist agent names (all except orchestrator)
SPECIALISTS = ["frontend", "backend", "test", "security", "docs", "devops"]

# Role-specific task description templates — use .format(task=task_description)
TASK_TEMPLATES = {
    "backend": (
        "Design and implement the backend for: {task}\n"
        "Include API endpoints, schemas, and database models."
    ),
    "frontend": (
        "Build the frontend components for: {task}\nUse React with TypeScript and Tailwind CSS."
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
        "Write documentation for: {task}\nInclude API reference, setup guide, and usage examples."
    ),
    "devops": (
        "Create deployment configuration for: {task}\n"
        "Include Dockerfile, docker-compose.yml, and CI/CD pipeline."
    ),
}

# Expected output descriptions per specialist role
EXPECTED_OUTPUTS = {
    "backend": "Complete backend code with API endpoints and schemas",
    "frontend": "Complete React components with TypeScript types",
    "test": "Complete pytest test suite with fixtures and assertions",
    "security": "Security audit report with vulnerability fixes",
    "docs": "Complete documentation in Markdown format",
    "devops": "Dockerfile, docker-compose.yml, and CI/CD config",
}


def make_step_callback(console):
    """Factory returning a CrewAI step_callback that prints progress to *console*."""

    def _step_callback(step_output):
        agent = getattr(step_output, "agent", None)
        agent_name = getattr(agent, "role", "Agent") if agent else "Agent"
        console.print(f"  [cyan]{agent_name}[/] completed a step")

    return _step_callback
