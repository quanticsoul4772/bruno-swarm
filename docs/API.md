# Bruno-Swarm API Reference

Complete API reference for the `bruno-swarm` Python package.

## Package: `bruno_swarm`

```python
import bruno_swarm

bruno_swarm.__version__  # "0.1.0"
bruno_swarm.cli          # Click group (entry point)
bruno_swarm.main()       # CLI entry point function
bruno_swarm.get_logger() # stdlib logging.Logger factory
```

**Exports:** `__version__`, `cli`, `main`, `get_logger`

---

## Module: `bruno_swarm.config`

Shared constants and utilities. Single source of truth imported by `cli.py`, `tui.py`, and `examples/`.

### Constants

#### `DEFAULT_OLLAMA_URL`
```python
DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
```

#### `HF_REPO`
```python
HF_REPO: str = "rawcell/bruno-swarm-models"
```
HuggingFace repository containing pre-built GGUF model files.

#### `HF_MODELS`
```python
HF_MODELS: dict[str, str]
```
Mapping of HuggingFace filenames to Ollama model names:
| HF Filename | Ollama Name |
|---|---|
| `orchestrator-14b-f16.gguf` | `orchestrator` |
| `frontend-3b-f16.gguf` | `frontend` |
| `backend-3b-f16.gguf` | `backend` |
| `test-3b-f16.gguf` | `test` |
| `security-3b-f16.gguf` | `security` |
| `docs-3b-f16.gguf` | `docs` |
| `devops-3b-f16.gguf` | `devops` |

#### `AGENT_CONFIGS`
```python
AGENT_CONFIGS: dict[str, dict]
```
Configuration for all 7 agents. Each entry has keys:
- `model` (str) -- Ollama model name (matches the agent name)
- `role` (str) -- Agent role title (e.g. "Senior Software Architect")
- `goal` (str) -- Agent goal description
- `backstory` (str) -- Agent backstory/personality prompt
- `system_prompt` (str) -- System prompt for Ollama Modelfile
- `allow_delegation` (bool) -- Whether agent can delegate (True only for orchestrator)

**Agents:**
| Name | Model Size | Role |
|---|---|---|
| `orchestrator` | 14B | Senior Software Architect |
| `frontend` | 3B | Frontend Developer |
| `backend` | 3B | Backend Developer |
| `test` | 3B | QA Engineer |
| `security` | 3B | Security Engineer |
| `docs` | 3B | Technical Writer |
| `devops` | 3B | DevOps Engineer |

#### `SPECIALISTS`
```python
SPECIALISTS: list[str] = ["frontend", "backend", "test", "security", "docs", "devops"]
```
All agent names except orchestrator, used as the default for flat mode.

#### `TASK_TEMPLATES`
```python
TASK_TEMPLATES: dict[str, str]
```
Role-specific task description templates. Each contains a `{task}` placeholder:
```python
TASK_TEMPLATES["backend"].format(task="Build user auth")
# "Design and implement the backend for: Build user auth\n..."
```

#### `EXPECTED_OUTPUTS`
```python
EXPECTED_OUTPUTS: dict[str, str]
```
Expected output descriptions per specialist, used as CrewAI Task `expected_output`.

### Functions

#### `ollama_api_get()`
```python
def ollama_api_get(ollama_url: str, endpoint: str) -> dict
```
GET an Ollama API endpoint and return parsed JSON.

**Parameters:**
- `ollama_url` -- Base URL (e.g. `"http://localhost:11434"`)
- `endpoint` -- API path (e.g. `"/api/tags"`)

**Returns:** Parsed JSON as `dict`.

**Raises:** `urllib.error.URLError` on connection failure; `Exception` for other errors.

**Usage:**
```python
from bruno_swarm.config import ollama_api_get

data = ollama_api_get("http://localhost:11434", "/api/tags")
models = data.get("models", [])
```

#### `make_step_callback()`
```python
def make_step_callback(console) -> Callable
```
Factory returning a CrewAI `step_callback` that prints generic progress messages.

**Parameters:**
- `console` -- Rich `Console` instance for output

**Returns:** Callback function `(step_output) -> None`

**Note:** CrewAI step callbacks don't carry agent identity. For per-agent identification, assign individual callbacks to each `Agent.step_callback`.

---

## Module: `bruno_swarm.tools`

Custom CrewAI tools for the orchestrator agent. Only the 14B orchestrator uses these -- 3B specialist models cannot reliably do function/tool calling.

### Tool Classes

All tools inherit from `crewai.tools.BaseTool` with Pydantic input schemas.

#### `ShellTool`
```python
class ShellTool(BaseTool):
    name = "execute_shell"
```
Execute a shell command and return stdout/stderr/exit code.

**Parameters:**
- `command` (str) -- Shell command to execute
- `working_dir` (str, default `"."`) -- Working directory

**Behavior:** Uses `subprocess.run(shell=True)` with 120-second timeout. Returns formatted string with stdout, stderr, and exit code sections.

#### `FileWriteTool`
```python
class FileWriteTool(BaseTool):
    name = "write_file"
```
Write content to a file, creating parent directories as needed.

**Parameters:**
- `filepath` (str) -- Path to the file
- `content` (str) -- Content to write (UTF-8)

#### `FileReadTool`
```python
class FileReadTool(BaseTool):
    name = "read_file"
```
Read the contents of a file as UTF-8 text.

**Parameters:**
- `filepath` (str) -- Path to the file

**Returns:** File contents, or error message if file not found.

#### `DirectoryListTool`
```python
class DirectoryListTool(BaseTool):
    name = "list_directory"
```
List files and directories at a given path.

**Parameters:**
- `dirpath` (str, default `"."`) -- Directory to list
- `recursive` (bool, default `False`) -- List recursively

**Filtered directories:** `.git`, `__pycache__`, `.pytest_cache`, `node_modules`, `.ruff_cache`

### Factory

#### `create_orchestrator_tools()`
```python
def create_orchestrator_tools() -> list[BaseTool]
```
Returns `[ShellTool(), FileWriteTool(), FileReadTool(), DirectoryListTool()]`.

---

## Module: `bruno_swarm.cli`

Click CLI application with 5 subcommands and interactive REPL mode.

### Agent/Crew Functions

#### `create_llm()`
```python
def create_llm(model_name: str, base_url: str) -> LLM
```
Create a CrewAI `LLM` instance for an Ollama model.

**Parameters:**
- `model_name` -- Ollama model name (e.g. `"backend"`)
- `base_url` -- Ollama server URL

**Returns:** `LLM(model=f"ollama/{model_name}", timeout=1200, max_retries=3)`

#### `create_agent()`
```python
def create_agent(name: str, base_url: str, tools: list | None = None) -> Agent
```
Create a CrewAI `Agent` from `AGENT_CONFIGS`.

**Parameters:**
- `name` -- Agent name (key in `AGENT_CONFIGS`)
- `base_url` -- Ollama server URL
- `tools` -- Optional list of `BaseTool` instances (default: empty)

**Returns:** Configured `Agent` with `verbose=False`, `max_iter=10`, `max_retry_limit=3`.

#### `create_hierarchical_crew()`
```python
def create_hierarchical_crew(
    task_description: str,
    base_url: str,
    agent_names: list[str] | None = None,
    agent_cache: dict | None = None,
    step_callback=None,
    enable_tools: bool = True,
) -> Crew
```
Create a hierarchical crew using sequential process.

**Structure:** Orchestrator plans -> Specialists work -> Orchestrator reviews.

**Parameters:**
- `task_description` -- Development task to execute
- `base_url` -- Ollama server URL
- `agent_names` -- Specialist names to include (default: all `SPECIALISTS`)
- `agent_cache` -- Optional dict for agent instance caching
- `step_callback` -- Optional CrewAI step callback
- `enable_tools` -- Give orchestrator filesystem/shell tools (default: `True`)

**Returns:** `Crew` with `Process.sequential`.

**Note:** Uses sequential process, not `Process.hierarchical`, because CrewAI's hierarchical mode doesn't properly route tasks.

#### `create_flat_crew()`
```python
def create_flat_crew(
    task_description: str,
    base_url: str,
    agent_names: list[str] | None = None,
    agent_cache: dict | None = None,
    step_callback=None,
) -> Crew
```
Create a flat sequential crew without orchestrator. Each specialist handles their portion sequentially.

**Parameters:** Same as `create_hierarchical_crew()` except no `enable_tools`.

**Returns:** `Crew` with `Process.sequential`, no tools.

### CLI Commands

#### `bruno-swarm` (no subcommand)
Launches the interactive REPL mode.

```
bruno-swarm [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--ollama-url` | `-u` | `http://localhost:11434` | Ollama server URL |
| `--flat` | | `False` | Start in flat mode |
| `--agents` | `-a` | all | Comma-separated agent names |
| `--no-tools` | | `False` | Disable orchestrator tools |

**REPL Commands:**
| Command | Description |
|---|---|
| `/agents` | List available agents |
| `/status` | Check Ollama connectivity |
| `/mode flat\|hier` | Switch execution mode |
| `/use agent1,agent2\|all` | Select specific agents |
| `/tools on\|off` | Toggle orchestrator tools |
| `/history` | Show task history |
| `/save <file>` | Save last result to file |
| `/quit` | Exit |

#### `bruno-swarm run`
Execute a development task with the agent swarm.

```
bruno-swarm run [OPTIONS]
```

| Option | Short | Required | Default | Description |
|---|---|---|---|---|
| `--task` | `-t` | Yes | -- | Development task to execute |
| `--flat` | | No | `False` | Use flat mode (no orchestrator) |
| `--agents` | `-a` | No | all | Comma-separated agent names |
| `--ollama-url` | `-u` | No | `http://localhost:11434` | Ollama server URL |
| `--output` | `-o` | No | -- | Save result to file |
| `--no-tools` | | No | `False` | Disable orchestrator tools |

#### `bruno-swarm setup`
Download models from HuggingFace and import into Ollama.

```
bruno-swarm setup [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--ollama-url` | `-u` | `http://localhost:11434` | Ollama server URL |
| `--models` | `-m` | all | Comma-separated model names |

Downloads GGUFs in parallel (3 workers), then imports into Ollama sequentially.

#### `bruno-swarm agents`
List all 7 available swarm agents and their roles.

#### `bruno-swarm status`
Check Ollama connectivity and model availability.

```
bruno-swarm status [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--ollama-url` | `-u` | `http://localhost:11434` | Ollama server URL |

Shows: connection status, agent model availability table, currently loaded models in memory.

#### `bruno-swarm tui`
Launch the interactive full-screen chat TUI. Requires `pip install bruno-swarm[tui]`.

```
bruno-swarm tui [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--ollama-url` | `-u` | `http://localhost:11434` | Ollama server URL |
| `--flat` | | `False` | Start in flat mode |
| `--agents` | `-a` | all | Comma-separated agent names |

---

## Module: `bruno_swarm.tui`

Textual-based full-screen chat TUI. Optional dependency (`pip install bruno-swarm[tui]`).

### `SwarmTUI`
```python
class SwarmTUI(App):
    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        flat: bool = False,
        agent_names: list[str] | None = None,
    ) -> None
```

**Key Bindings:**
| Key | Action |
|---|---|
| `Ctrl+Q` | Quit |
| `t` | Cycle theme |
| `Escape` | Focus input |

**Themes:** `textual-dark`, `textual-light`, `textual-ansi`

**Slash Commands:** `/help`, `/agents`, `/status`, `/mode`, `/use`, `/history`, `/save`, `/clear`, `/quit`

### `run_tui()`
```python
def run_tui(
    ollama_url: str = DEFAULT_OLLAMA_URL,
    flat: bool = False,
    agent_names: list[str] | None = None,
) -> None
```
Launch the `SwarmTUI` application.

---

## Module: `bruno_swarm.widgets`

Custom Textual widgets for the TUI.

### `AgentStatusLine`
```python
class AgentStatusLine(Static):
    status = reactive("idle")  # "idle" | "working" | "done" | "error"

    def __init__(self, agent_name: str, **kwargs) -> None
```
Single agent line with status symbol and name.

**Status Symbols:** `idle` = ○, `working` = ●, `done` = ✓, `error` = ✗

### `AgentSidebar`
```python
class AgentSidebar(Widget):
    def update_status(self, agent_name: str, status: str) -> None
    def reset_all(self) -> None
```
Vertical panel showing all 7 agents with live status indicators.

### `ModeIndicator`
```python
class ModeIndicator(Static):
    def __init__(
        self,
        mode: str = "hierarchical",
        agents_str: str = "all specialists",
        ollama_url: str = "http://localhost:11434",
        **kwargs,
    ) -> None

    def update_info(self, mode: str, agents_str: str) -> None
```
Top bar displaying current mode, selected agents, and Ollama URL.

---

## Module: `bruno_swarm.logging`

### `get_logger()`
```python
def get_logger(name: str = "bruno-swarm") -> logging.Logger
```
Returns a stdlib `logging.Logger` instance. Configure via `logging.basicConfig()`.

---

## Agent Caching

When using `agent_cache` with crew functions, agents are cached by a 3-tuple key:

```python
key = (name: str, base_url: str, tools_id: tuple[str, ...])
```

Where `tools_id` is a tuple of tool class names (e.g. `("ShellTool", "FileWriteTool", ...)`), or `()` for agents without tools. This means the same agent name with different tool sets creates separate cache entries.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CREWAI_TRACING_ENABLED` | `false` (set by CLI) | Suppress CrewAI tracing prompt |
| `OLLAMA_MAX_LOADED_MODELS` | -- | Recommended: `3` for multi-model use |
| `OLLAMA_KEEP_ALIVE` | -- | Recommended: `30m` to keep models loaded |
