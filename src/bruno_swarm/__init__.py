# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bruno Swarm -- Multi-agent AI developer team powered by abliterated models."""

__version__ = "0.1.0"

from .cli import cli, main
from .logging import get_logger

__all__ = ["__version__", "cli", "get_logger", "main"]
