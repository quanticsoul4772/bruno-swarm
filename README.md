# bruno-swarm

Multi-agent AI developer swarm powered by abliterated models. A team of 7 specialized coding agents that run locally via Ollama and coordinate through CrewAI.

## Agent Team

| Agent | Model | Role |
|-------|-------|------|
| **Orchestrator** | 14B | Senior Architect -- plans, delegates, reviews |
| **Frontend** | 3B | React / TypeScript / Tailwind specialist |
| **Backend** | 3B | FastAPI / PostgreSQL / async specialist |
| **Test** | 3B | pytest / coverage / edge case specialist |
| **Security** | 3B | OWASP / vulnerability / secure code specialist |
| **Docs** | 3B | Technical writing / API docs specialist |
| **DevOps** | 3B | Docker / CI-CD / infrastructure specialist |

All models are abliterated Qwen2.5-Coder variants with role-specific behavioral modifications, created with [Bruno](https://github.com/quanticsoul4772/bruno) (neural behavior engineering).

## Quick Start

```bash
# Install
pip install bruno-swarm

# Download models from HuggingFace and import into Ollama
bruno-swarm setup

# Check that Ollama has all 7 models
bruno-swarm status

# Run a task (hierarchical mode with 14B orchestrator)
bruno-swarm run --task "Build a user authentication system with JWT"

# Run in flat mode (specialists only, no orchestrator)
bruno-swarm run --task "Build a REST API for todos" --flat

# Use specific agents only
bruno-swarm run --task "Fix SQL injection" --agents security,backend

# Interactive mode (like Claude Code)
bruno-swarm
```

## Requirements

- **Python** 3.10-3.12
- **Ollama** running locally (or remote via `--ollama-url`)
- **GPU** with sufficient VRAM:
  - Hierarchical mode (all 7): ~47GB (A100 80GB recommended)
  - Flat mode (6 specialists): ~18GB (RTX 4090 24GB)
  - Subset (2-3 agents): ~6-9GB (RTX 4070 8GB)

## Modes

### Hierarchical (default)

The 14B orchestrator plans the architecture, delegates to 3B specialists, then reviews and integrates all work.

```bash
bruno-swarm run --task "Build user auth"
```

### Flat

Specialists execute sequentially without an orchestrator. Faster, uses less VRAM, good for focused tasks.

```bash
bruno-swarm run --task "Build user auth" --flat
```

### Interactive

REPL-style interface with commands for switching modes, selecting agents, and saving results.

```bash
bruno-swarm
```

## Setup Details

### Automatic (recommended)

```bash
bruno-swarm setup
```

Downloads pre-built GGUF models from [rawcell/bruno-swarm-models](https://huggingface.co/rawcell/bruno-swarm-models) on HuggingFace and imports them into Ollama.

### Manual

If you want to abliterate your own models:

1. Install [Bruno](https://github.com/quanticsoul4772/bruno): `pip install bruno-ai`
2. Abliterate models per role (see `docs/AI_AGENT_SWARM_RESEARCH.md`)
3. Convert to GGUF via llama.cpp
4. Import into Ollama using the Modelfiles in `modelfiles/`

## Ollama Configuration

For best results with multiple models loaded simultaneously:

```bash
# Set before starting Ollama
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_KEEP_ALIVE=30m
```

## Known Issues

- 3B models can be repetitive -- `num_predict 2048` in Modelfiles caps output length
- 14B orchestrator sometimes sends batch delegations -- `max_iter=10` tolerates retries
- CrewAI needs `num_ctx 8192` minimum for its system prompts
- Model loading can take 30-60s on first use -- `timeout=1200` handles this

## License

AGPL-3.0-or-later
