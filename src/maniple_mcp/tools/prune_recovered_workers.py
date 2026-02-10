"""
Prune recovered workers tool.

Provides prune_recovered_workers for marking stale recovered sessions closed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

if TYPE_CHECKING:
    from ..server import AppContext


logger = logging.getLogger("maniple")


def register_tools(mcp: FastMCP) -> None:
    """Register prune_recovered_workers tool on the MCP server."""

    @mcp.tool()
    async def prune_recovered_workers(
        ctx: Context[ServerSession, "AppContext"],
    ) -> dict:
        """
        Prune stale recovered worker sessions.

        This is a safety valve for "ghost" recovered sessions (source=event_log)
        that show as active even though their underlying terminal pane no longer
        exists (common after crashes or manual tmux cleanup).

        On tmux backend, this:
        - checks whether recovered tmux panes still exist
        - emits worker_closed(reason=stale_recovered) for stale recovered sessions
        - updates the in-memory recovered state to event_state=closed

        Returns:
            Dict with:
                - pruned: number of sessions pruned
                - emitted_closed: number of worker_closed events emitted
                - session_ids: list of pruned session IDs
                - errors: list of non-fatal errors encountered
        """
        app_ctx = ctx.request_context.lifespan_context
        registry = app_ctx.registry
        backend = app_ctx.terminal_backend

        report = await registry.prune_stale_recovered_sessions(backend)
        if report.errors:
            logger.warning("prune_recovered_workers encountered errors: %s", report.errors)

        return {
            "pruned": report.pruned,
            "emitted_closed": report.emitted_closed,
            "session_ids": list(report.session_ids),
            "errors": list(report.errors),
        }

