#!/usr/bin/env python3
"""Download GGUFs from HuggingFace and generate Ollama Modelfiles.

Stage 1 of the Docker build. Runs in python:3.12-slim with only
huggingface-hub installed — no crewai, click, rich, or bruno-swarm.

Model config data is intentionally duplicated here rather than imported.
Authoritative source: src/bruno_swarm/config.py
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Mirror of config.py — keep in sync manually
# ---------------------------------------------------------------------------

HF_REPO = "rawcell/bruno-swarm-models"

HF_MODELS: dict[str, str] = {
    "orchestrator-14b-f16.gguf": "orchestrator",
    "frontend-3b-f16.gguf": "frontend",
    "backend-3b-f16.gguf": "backend",
    "test-3b-f16.gguf": "test",
    "security-3b-f16.gguf": "security",
    "docs-3b-f16.gguf": "docs",
    "devops-3b-f16.gguf": "devops",
}

SYSTEM_PROMPTS: dict[str, str] = {
    "orchestrator": (
        "You are a Senior Software Architect and Project Manager. Plan development "
        "tasks, design system architecture, delegate work to specialists, and review "
        "code quality. Think step by step before delegating."
    ),
    "frontend": (
        "You are a Frontend Developer specializing in React, TypeScript, and "
        "Tailwind CSS. Write clean, concise code without over-engineering."
    ),
    "backend": (
        "You are a Backend Developer specializing in FastAPI, PostgreSQL, and "
        "async patterns. Focus on clean architecture without premature optimization."
    ),
    "test": (
        "You are a QA Engineer specializing in pytest, coverage analysis, and "
        "edge case testing. Proactively write comprehensive tests for all code."
    ),
    "security": (
        "You are a Security Engineer specializing in vulnerability assessment, "
        "OWASP Top 10, and secure coding patterns. Identify security issues "
        "aggressively and recommend hardened implementations."
    ),
    "docs": (
        "You are a Technical Writer specializing in API documentation, README "
        "files, and developer guides. Write clear, concise documentation without "
        "unnecessary jargon."
    ),
    "devops": (
        "You are a DevOps Engineer specializing in Docker, CI/CD pipelines, and "
        "infrastructure as code. Write practical deployment configurations without "
        "overengineering."
    ),
}

# ---------------------------------------------------------------------------

MODELS_DIR = Path("/tmp/models")
MODELFILES_DIR = Path("/tmp/modelfiles")


def generate_modelfile(name: str, gguf_path: Path) -> str:
    """Generate an Ollama Modelfile for the given agent."""
    num_predict = 4096 if name == "orchestrator" else 2048
    system_prompt = SYSTEM_PROMPTS[name]
    return (
        f"FROM {gguf_path}\n"
        f"SYSTEM \"{system_prompt}\"\n"
        f"PARAMETER temperature 0.7\n"
        f"PARAMETER top_p 0.9\n"
        f"PARAMETER top_k 40\n"
        f"PARAMETER num_ctx 8192\n"
        f"PARAMETER num_predict {num_predict}\n"
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODELFILES_DIR.mkdir(parents=True, exist_ok=True)

    for filename, name in HF_MODELS.items():
        print(f"[download] {filename} -> {name}")
        dest = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may place the file in a subdirectory; move to flat layout
        dest_path = Path(dest)
        target = MODELS_DIR / filename
        if dest_path != target:
            os.rename(dest_path, target)

        modelfile_content = generate_modelfile(name, target)
        modelfile_path = MODELFILES_DIR / f"Modelfile.{name}"
        modelfile_path.write_text(modelfile_content)
        print(f"[modelfile] {modelfile_path}")

    print(f"\n[done] {len(HF_MODELS)} models downloaded, {len(HF_MODELS)} Modelfiles generated")


if __name__ == "__main__":
    main()
