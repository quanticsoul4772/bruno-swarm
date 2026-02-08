# SPDX-License-Identifier: AGPL-3.0-or-later

"""Minimal logging setup for bruno-swarm."""

import logging


def get_logger(name: str = "bruno-swarm") -> logging.Logger:
    """Get a logger instance for bruno-swarm.

    Uses stdlib logging. Configure with standard logging.basicConfig()
    or logging.config if needed.
    """
    return logging.getLogger(name)
