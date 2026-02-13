# SPDX-License-Identifier: AGPL-3.0-or-later

"""Custom CrewAI tools for filesystem and shell operations.

Only the 14B orchestrator agent uses these tools -- 3B specialist models
cannot reliably do function/tool calling.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class ShellToolInput(BaseModel):
    """Input schema for ShellTool."""

    command: str = Field(..., description="Shell command to execute")
    working_dir: str = Field(default=".", description="Working directory for the command")


class FileWriteToolInput(BaseModel):
    """Input schema for FileWriteTool."""

    filepath: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write to the file")


class FileReadToolInput(BaseModel):
    """Input schema for FileReadTool."""

    filepath: str = Field(..., description="Path to the file to read")


class DirectoryListToolInput(BaseModel):
    """Input schema for DirectoryListTool."""

    dirpath: str = Field(default=".", description="Directory path to list")
    recursive: bool = Field(default=False, description="List contents recursively")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class ShellTool(BaseTool):
    """Execute a shell command and return stdout/stderr."""

    name: str = "execute_shell"
    description: str = (
        "Execute a shell command and return the output. "
        "Use this to run tests, linters, builds, git commands, or any shell operation."
    )
    args_schema: Type[BaseModel] = ShellToolInput

    def _run(self, command: str, working_dir: str = ".") -> str:
        work_dir = Path(working_dir).resolve()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(work_dir),
            )
            parts = []
            if result.stdout:
                parts.append(f"stdout:\n{result.stdout}")
            if result.stderr:
                parts.append(f"stderr:\n{result.stderr}")
            parts.append(f"exit_code: {result.returncode}")
            return "\n".join(parts)
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 120 seconds"
        except Exception as e:
            return f"Error: {e}"


class FileWriteTool(BaseTool):
    """Write content to a file, creating parent directories as needed."""

    name: str = "write_file"
    description: str = (
        "Write content to a file. Creates parent directories automatically. "
        "Use this to create or overwrite source files, configs, scripts, etc."
    )
    args_schema: Type[BaseModel] = FileWriteToolInput

    def _run(self, filepath: str, content: str) -> str:
        try:
            path = Path(filepath).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error writing {filepath}: {e}"


class FileReadTool(BaseTool):
    """Read the contents of a file."""

    name: str = "read_file"
    description: str = (
        "Read the contents of a file and return them as text. "
        "Use this to examine existing source code before making changes."
    )
    args_schema: Type[BaseModel] = FileReadToolInput

    def _run(self, filepath: str) -> str:
        try:
            path = Path(filepath).resolve()
            if not path.exists():
                return f"Error: file not found: {path}"
            if not path.is_file():
                return f"Error: not a file: {path}"
            return path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading {filepath}: {e}"


class DirectoryListTool(BaseTool):
    """List files and directories at a given path."""

    name: str = "list_directory"
    description: str = (
        "List files and directories in a path. "
        "Use this to explore project structure before making changes."
    )
    args_schema: Type[BaseModel] = DirectoryListToolInput

    def _run(self, dirpath: str = ".", recursive: bool = False) -> str:
        try:
            path = Path(dirpath).resolve()
            if not path.exists():
                return f"Error: directory not found: {path}"
            if not path.is_dir():
                return f"Error: not a directory: {path}"

            skip = {".git", "__pycache__", ".pytest_cache", "node_modules", ".ruff_cache"}
            if recursive:
                entries = sorted(
                    e
                    for e in path.rglob("*")
                    if not any(part in skip for part in e.relative_to(path).parts)
                )
            else:
                entries = sorted(path.iterdir())
            lines = []
            for entry in entries:
                rel = entry.relative_to(path)
                if rel.parts and rel.parts[0] in skip:
                    continue
                suffix = "/" if entry.is_dir() else ""
                lines.append(f"{rel}{suffix}")
            if not lines:
                return f"{path} is empty"
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing {dirpath}: {e}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_orchestrator_tools() -> list[BaseTool]:
    """Return the set of tools for the orchestrator agent."""
    return [ShellTool(), FileWriteTool(), FileReadTool(), DirectoryListTool()]
