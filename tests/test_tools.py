# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for custom CrewAI tools (shell, file I/O, directory listing)."""

from __future__ import annotations

import platform

from bruno_swarm.tools import (
    DirectoryListTool,
    FileReadTool,
    FileWriteTool,
    ShellTool,
    create_orchestrator_tools,
)

# ---------------------------------------------------------------------------
# ShellTool
# ---------------------------------------------------------------------------


class TestShellTool:
    def setup_method(self):
        self.tool = ShellTool()

    def test_simple_command(self):
        cmd = "echo hello" if platform.system() != "Windows" else "echo hello"
        result = self.tool._run(command=cmd)
        assert "hello" in result
        assert "exit_code: 0" in result

    def test_captures_stderr(self):
        if platform.system() == "Windows":
            cmd = "echo err 1>&2"
        else:
            cmd = "echo err >&2"
        result = self.tool._run(command=cmd)
        assert "stderr:" in result
        assert "err" in result

    def test_nonzero_exit_code(self):
        if platform.system() == "Windows":
            cmd = "cmd /c exit 42"
        else:
            cmd = "exit 42"
        result = self.tool._run(command=cmd)
        assert "exit_code: 42" in result

    def test_working_directory(self, tmp_path):
        result = self.tool._run(command="echo ok", working_dir=str(tmp_path))
        assert "exit_code: 0" in result

    def test_timeout_returns_error(self, monkeypatch):
        import subprocess as sp

        def fake_run(*args, **kwargs):
            raise sp.TimeoutExpired(cmd="sleep", timeout=120)

        monkeypatch.setattr(sp, "run", fake_run)
        result = self.tool._run(command="sleep 999")
        assert "timed out" in result


# ---------------------------------------------------------------------------
# FileWriteTool
# ---------------------------------------------------------------------------


class TestFileWriteTool:
    def setup_method(self):
        self.tool = FileWriteTool()

    def test_write_simple_file(self, tmp_path):
        filepath = tmp_path / "hello.txt"
        result = self.tool._run(filepath=str(filepath), content="hello world")
        assert "Successfully wrote" in result
        assert filepath.read_text(encoding="utf-8") == "hello world"

    def test_creates_parent_directories(self, tmp_path):
        filepath = tmp_path / "a" / "b" / "c" / "deep.txt"
        result = self.tool._run(filepath=str(filepath), content="deep")
        assert "Successfully wrote" in result
        assert filepath.exists()
        assert filepath.read_text(encoding="utf-8") == "deep"

    def test_overwrites_existing_file(self, tmp_path):
        filepath = tmp_path / "overwrite.txt"
        filepath.write_text("old", encoding="utf-8")
        self.tool._run(filepath=str(filepath), content="new")
        assert filepath.read_text(encoding="utf-8") == "new"


# ---------------------------------------------------------------------------
# FileReadTool
# ---------------------------------------------------------------------------


class TestFileReadTool:
    def setup_method(self):
        self.tool = FileReadTool()

    def test_read_existing_file(self, tmp_path):
        filepath = tmp_path / "readme.txt"
        filepath.write_text("contents here", encoding="utf-8")
        result = self.tool._run(filepath=str(filepath))
        assert result == "contents here"

    def test_file_not_found(self):
        result = self.tool._run(filepath="/nonexistent/path/file.txt")
        assert "Error: file not found" in result

    def test_not_a_file(self, tmp_path):
        result = self.tool._run(filepath=str(tmp_path))
        assert "Error: not a file" in result


# ---------------------------------------------------------------------------
# DirectoryListTool
# ---------------------------------------------------------------------------


class TestDirectoryListTool:
    def setup_method(self):
        self.tool = DirectoryListTool()

    def test_list_directory(self, tmp_path):
        (tmp_path / "file1.py").touch()
        (tmp_path / "file2.py").touch()
        (tmp_path / "subdir").mkdir()
        result = self.tool._run(dirpath=str(tmp_path))
        assert "file1.py" in result
        assert "file2.py" in result
        assert "subdir/" in result

    def test_recursive_listing(self, tmp_path):
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "mod.py").touch()
        result = self.tool._run(dirpath=str(tmp_path), recursive=True)
        assert "mod.py" in result

    def test_directory_not_found(self):
        result = self.tool._run(dirpath="/nonexistent/dir")
        assert "Error: directory not found" in result

    def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = self.tool._run(dirpath=str(empty))
        assert "empty" in result.lower()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateOrchestratorTools:
    def test_returns_four_tools(self):
        tools = create_orchestrator_tools()
        assert len(tools) == 4

    def test_tool_types(self):
        tools = create_orchestrator_tools()
        types = {type(t) for t in tools}
        assert types == {ShellTool, FileWriteTool, FileReadTool, DirectoryListTool}

    def test_tools_have_names(self):
        tools = create_orchestrator_tools()
        names = {t.name for t in tools}
        assert names == {"execute_shell", "write_file", "read_file", "list_directory"}
