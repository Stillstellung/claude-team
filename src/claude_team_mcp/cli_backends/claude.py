"""
Claude Code CLI backend.

Implements the AgentCLI protocol for Claude Code CLI.
This preserves the existing behavior from iterm_utils.py.
"""

import os
from typing import Literal

from .base import AgentCLI


class ClaudeCLI(AgentCLI):
    """
    Claude Code CLI implementation.

    Supports:
    - --dangerously-skip-permissions flag
    - --settings flag for Stop hook injection
    - Ready detection via TUI patterns (robot banner, '>' prompt, 'tokens' status)
    - Idle detection via Stop hook markers in JSONL
    """

    @property
    def engine_id(self) -> str:
        """Return 'claude' as the engine identifier."""
        return "claude"

    def command(self) -> str:
        """
        Return the Claude CLI command.

        Respects CLAUDE_TEAM_COMMAND environment variable for overrides
        (e.g., "happy" wrapper).
        """
        return os.environ.get("CLAUDE_TEAM_COMMAND", "claude")

    def build_args(
        self,
        *,
        dangerously_skip_permissions: bool = False,
        settings_file: str | None = None,
    ) -> list[str]:
        """
        Build Claude CLI arguments.

        Args:
            dangerously_skip_permissions: Add --dangerously-skip-permissions
            settings_file: Path to settings JSON for Stop hook injection

        Returns:
            List of CLI arguments
        """
        args: list[str] = []

        if dangerously_skip_permissions:
            args.append("--dangerously-skip-permissions")

        # Only add --settings for the default 'claude' command.
        # Custom commands like 'happy' have their own session tracking mechanisms.
        # See HAPPY_INTEGRATION_RESEARCH.md for full analysis.
        if settings_file and self._is_default_command():
            args.append("--settings")
            args.append(settings_file)

        return args

    def ready_patterns(self) -> list[str]:
        """
        Return patterns indicating Claude TUI is ready.

        These patterns appear in Claude's startup:
        - '>' prompt indicates input ready
        - 'tokens' in status bar
        - Parts of the robot ASCII art banner
        """
        return [
            ">",  # Input prompt
            "tokens",  # Status bar
            "Claude Code v",  # Version line in banner
            "▐▛███▜▌",  # Top of robot head
            "▝▜█████▛▘",  # Middle of robot
        ]

    def idle_detection_method(self) -> Literal["stop_hook", "jsonl_stream", "none"]:
        """
        Claude uses Stop hook for idle detection.

        A Stop hook writes a marker to the JSONL when Claude finishes responding.
        """
        return "stop_hook"

    def supports_settings_file(self) -> bool:
        """
        Claude supports --settings for hook injection.

        Only returns True for the default 'claude' command.
        Custom wrappers may have their own settings mechanisms.
        """
        return self._is_default_command()

    def _is_default_command(self) -> bool:
        """Check if using the default 'claude' command (not a custom wrapper)."""
        cmd = os.environ.get("CLAUDE_TEAM_COMMAND", "claude")
        return cmd == "claude"


# Singleton instance for convenience
claude_cli = ClaudeCLI()
