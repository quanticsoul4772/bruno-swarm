# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

bruno-swarm is a multi-agent AI developer CLI that orchestrates 7 specialized coding agents (1x 14B orchestrator + 6x 3B specialists) running locally via Ollama and coordinated through CrewAI. The agents use abliterated Qwen2.5-Coder models with role-specific behavioral modifications created with [Bruno](https://github.com/quanticsoul4772/bruno) (neural behavior engineering).

## Build & Development Commands

```bash
# Install in development mode
pip install -e .

# Install dependencies only
pip install .

# Run the CLI (entry point: bruno_swarm.cli:main)
bruno-swarm                                    # interactive mode
bruno-swarm run --task "Build X" --flat         # flat mode (no orchestrator)
bruno-swarm run --task "Build X"                # hierarchical mode
bruno-swarm agents                             # list agents
bruno-swarm status                             # check Ollama connectivity
bruno-swarm setup                              # download models from HuggingFace

# Lint
ruff check src/
ruff format src/
```

There are no tests yet. The project has no test suite.

## Architecture

### Single-package Python CLI (`src/bruno_swarm/`)

- **`cli.py`** (~960 lines) — The entire application logic lives here. Click CLI with 4 subcommands (`run`, `setup`, `agents`, `status`) plus interactive REPL mode. Contains:
  - `AGENT_CONFIGS` dict — All 7 agent definitions (model, role, goal, backstory, delegation flag)
  - `create_hierarchical_crew()` — Workaround for broken `Process.hierarchical`: uses sequential with orchestrator as first and last agent (plan → specialists → review)
  - `create_flat_crew()` — Sequential specialist-only execution, no orchestrator
  - `_run_interactive()` — REPL with `/commands` for mode switching, agent selection, history, save
  - `setup_models()` — Downloads GGUFs from HuggingFace, generates Modelfiles inline, imports into Ollama via `ollama create`
- **`logging.py`** — Thin wrapper around stdlib `logging.getLogger()`
- **`__init__.py`** — Exports `cli`, `main`, `get_logger`, `__version__`

### Key Design Decisions

- **CrewAI hierarchical process is intentionally avoided** in the production CLI. `create_hierarchical_crew()` uses `Process.sequential` with the orchestrator bookending the task list (plan first, review last) because CrewAI's `Process.hierarchical` doesn't properly route tasks.
- **CrewAI is imported lazily** inside functions, not at module level, so `agents` and `status` subcommands work without CrewAI installed.
- **`CREWAI_TRACING_ENABLED=false`** is set to suppress CrewAI's tracing prompt.
- **Ollama model names match agent names** (e.g., the "backend" agent uses the Ollama model named "backend").

### Models & Ollama

- 7 Ollama Modelfiles in `modelfiles/` — one per agent, pointing to GGUF files
- Orchestrator: `num_predict 4096`, specialists: `num_predict 2048` (caps repetitive 3B output)
- All models: `num_ctx 8192` (minimum for CrewAI system prompts), `timeout 1200s` (model loading latency)
- `OLLAMA_MAX_LOADED_MODELS=3` and `OLLAMA_KEEP_ALIVE=30m` recommended for multi-model use

### Docker Deployment (`deploy/`)

- **`Dockerfile.ollama`** — Three-stage build: Python slim downloader → Ollama builder → clean runtime
- **`download-and-prepare.py`** — Downloads GGUFs from HF + generates Modelfiles (mirrors AGENT_CONFIGS)
- **`docker-entrypoint.sh`** — GPU detection, model health check, graceful shutdown
- **`docker-compose.yml`** — GPU-enabled Compose with health checks

### Examples (`examples/`)

Standalone scripts that directly import CrewAI (not the CLI). These are reference implementations from the research/prototyping phase and use slightly different configurations (e.g., `verbose=True`, different model names, `Process.hierarchical`).

## Dependencies

- **click** — CLI framework
- **rich** — Terminal UI (Console, Panel, Table, Prompt)
- **crewai** — Multi-agent orchestration (agents, tasks, crews)
- **huggingface-hub** — Model downloads (`hf_hub_download`)
- **Build system**: Hatchling
- **Linter**: Ruff (line-length=100, import sorting enabled)
- **Python**: 3.10–3.12

## Known Quirks

- 3B models are repetitive — `num_predict 2048` in Modelfiles caps output length
- 14B orchestrator sometimes sends batch delegations — `max_iter=10` tolerates retries
- The `setup` command generates Modelfiles dynamically (doesn't use the static ones in `modelfiles/`)
- The example scripts use different model names than the CLI (e.g., `frontend-agent` vs `frontend`)
