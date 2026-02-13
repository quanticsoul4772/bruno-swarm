# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

bruno-swarm is a multi-agent AI developer CLI that orchestrates 7 specialized coding agents (1x 14B orchestrator + 6x 3B specialists) running locally via Ollama and coordinated through CrewAI. The agents use abliterated Qwen2.5-Coder models with role-specific behavioral modifications created with [Bruno](https://github.com/quanticsoul4772/bruno) (neural behavior engineering).

## Build & Development Commands

```bash
# Install in development mode (includes test deps)
pip install -e ".[dev]"

# Install with TUI support
pip install -e ".[tui]"

# Run the CLI
bruno-swarm                                    # interactive REPL
bruno-swarm run --task "Build X"               # hierarchical mode
bruno-swarm run --task "Build X" --flat         # flat mode (no orchestrator)
bruno-swarm run --task "Build X" --agents security,backend  # subset of agents
bruno-swarm tui                                # full-screen Textual chat TUI
bruno-swarm agents                             # list agents
bruno-swarm status                             # check Ollama connectivity
bruno-swarm setup                              # download models from HuggingFace

# Tests
pytest                                         # run all 77 tests
pytest tests/test_cli.py                       # CLI tests only
pytest tests/test_config.py                    # config tests only
pytest tests/test_tui.py                       # TUI tests (skips if textual not installed)
pytest -k "test_parse_agents"                  # single test by name

# Lint
ruff check src/
ruff format src/
```

## Architecture

### Single-package Python CLI (`src/bruno_swarm/`)

- **`config.py`** — Single source of truth for all shared constants: `AGENT_CONFIGS` (7 agent definitions), `TASK_TEMPLATES`, `EXPECTED_OUTPUTS`, `HF_MODELS`, `SPECIALISTS`, `DEFAULT_OLLAMA_URL`, `HF_REPO`, `ollama_api_get()` (shared Ollama HTTP utility), and `make_step_callback()` factory.
- **`tools.py`** — Custom CrewAI tools for the orchestrator: `ShellTool` (execute commands), `FileWriteTool`, `FileReadTool`, `DirectoryListTool`. Uses `BaseTool` + Pydantic schemas. Only stdlib dependencies.
- **`cli.py`** — Click CLI with 5 subcommands (`run`, `setup`, `agents`, `status`, `tui`) plus interactive REPL. Contains crew creation logic (`create_hierarchical_crew()`, `create_flat_crew()`), model downloading, and the `_run_interactive()` REPL.
- **`tui.py`** — Textual-based full-screen chat TUI (`SwarmTUI` app). Optional dependency — works without textual installed via lazy import. Supports slash commands (`/help`, `/mode`, `/agents`, `/history`, `/save`, `/quit`).
- **`widgets.py`** — Custom Textual widgets: `AgentSidebar` (agent status panel), `ModeIndicator` (top bar), `AgentStatusLine` (per-agent status with reactive state).
- **`tui.tcss`** — CSS stylesheet for TUI layout.
- **`logging.py`** — Thin wrapper around stdlib `logging.getLogger()`.
- **`__init__.py`** — Exports `cli`, `main`, `get_logger`, `__version__`.

### Key Design Decisions

- **CrewAI hierarchical process is intentionally avoided.** `create_hierarchical_crew()` uses `Process.sequential` with the orchestrator bookending the task list (plan first, review last) because CrewAI's `Process.hierarchical` doesn't properly route tasks.
- **CrewAI is imported lazily** inside functions, not at module level, so `agents` and `status` subcommands work without CrewAI installed.
- **`CREWAI_TRACING_ENABLED=false`** is set to suppress CrewAI's tracing prompt.
- **Ollama model names match agent names** (e.g., the "backend" agent uses the Ollama model named "backend").
- **The `setup` command generates Modelfiles dynamically** (doesn't use the static ones in `modelfiles/`).
- **Only the orchestrator gets tools.** 3B specialist models can't reliably do function/tool calling. The orchestrator (14B) reads files, writes code to disk, and runs shell commands. Specialists remain text-only reasoners.

### Orchestrator Tools

In hierarchical mode, the orchestrator has 4 tools (defined in `tools.py`):

- **`read_file`** — Read source files before making changes
- **`write_file`** — Write code to files (creates directories automatically)
- **`execute_shell`** — Run shell commands (tests, linters, builds, git)
- **`list_directory`** — Explore project structure

Disable with `--no-tools` flag. Tools are never available in flat mode (no orchestrator).

### Import Gotcha

`import bruno_swarm.cli` resolves to the Click group object, not the module, because `__init__.py` exports `from .cli import cli`. In tests, use `sys.modules["bruno_swarm.cli"]` to get the actual module. The `conftest.py` fixture demonstrates this pattern for patching `console`.

### Models & Ollama

- 7 Ollama Modelfiles in `modelfiles/` — one per agent, pointing to GGUF files
- Orchestrator: `num_predict 4096`, specialists: `num_predict 2048` (caps repetitive 3B output)
- All models: `num_ctx 8192` (minimum for CrewAI system prompts), `timeout 1200s` (model loading latency)
- `OLLAMA_MAX_LOADED_MODELS=3` and `OLLAMA_KEEP_ALIVE=30m` recommended for multi-model use

### Docker Deployment (`deploy/`)

- **`Dockerfile.ollama`** — Two-stage build: Ollama + Python downloader/builder → clean runtime
- **`download-and-prepare.py`** — Downloads GGUFs from HF + generates Modelfiles (mirrors AGENT_CONFIGS)
- **`docker-entrypoint.sh`** — GPU detection, model health check, graceful shutdown
- **`docker-compose.yml`** — GPU-enabled Compose with health checks
- **`.github/workflows/docker-build.yml`** — CI: builds and pushes image to Docker Hub on version tags or deploy/ changes

### Examples (`examples/`)

Standalone scripts that directly import CrewAI (not the CLI). Reference implementations from the prototyping phase with slightly different configurations (e.g., `verbose=True`, different model names, `Process.hierarchical`).

## Dependencies

- **click** — CLI framework
- **rich** — Terminal UI (Console, Panel, Table, Prompt)
- **crewai** (>=0.86,<1.0) — Multi-agent orchestration
- **huggingface-hub** — Model downloads
- **textual** (optional, `pip install bruno-swarm[tui]`) — Full-screen chat TUI
- **Build system**: Hatchling
- **Linter**: Ruff (line-length=100, import sorting enabled)
- **Python**: 3.10–3.12

## Known Quirks

- 3B models are repetitive — `num_predict 2048` in Modelfiles caps output length
- 14B orchestrator sometimes sends batch delegations — `max_iter=10` tolerates retries
- The example scripts use different model names than the CLI (e.g., `frontend-agent` vs `frontend`)
- `callable` is a builtin function, not a type — `callable | None` fails at runtime in Python 3.12; use `typing.Callable` or plain `=None`
- Textual `App.is_running` is a read-only property — don't shadow it
- Textual single-key bindings (e.g. `t`) are consumed by focused Input widgets; test via `app.action_*()` instead of `pilot.press()`
