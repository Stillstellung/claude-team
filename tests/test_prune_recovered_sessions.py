"""Tests for pruning stale recovered sessions."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from maniple_mcp.registry import RecoveredSession, SessionRegistry, SessionStatus, TerminalId
from maniple_mcp.terminal_backends.base import TerminalSession


class _FakeTmuxBackend:
    backend_id = "tmux"

    def __init__(self, panes: list[str] | None = None, *, raise_on_list: bool = False) -> None:
        self._panes = panes or []
        self._raise_on_list = raise_on_list

    async def list_sessions(self) -> list[TerminalSession]:
        if self._raise_on_list:
            raise RuntimeError("tmux unavailable")
        return [
            TerminalSession(backend_id="tmux", native_id=pane_id, handle=pane_id)
            for pane_id in self._panes
        ]


class _FakeItermBackend:
    backend_id = "iterm"

    def __init__(self, sessions: list[str] | None = None, *, raise_on_list: bool = False) -> None:
        self._sessions = sessions or []
        self._raise_on_list = raise_on_list

    async def list_sessions(self) -> list[TerminalSession]:
        if self._raise_on_list:
            raise RuntimeError("iterm unavailable")
        return [
            TerminalSession(backend_id="iterm", native_id=session_id, handle=session_id)
            for session_id in self._sessions
        ]


def _recovered(
    *,
    session_id: str = "w1",
    pane_id: str | None = "%1",
    event_state: str = "active",
    worktree_path: str | None = None,
) -> RecoveredSession:
    now = datetime(2026, 2, 10, tzinfo=timezone.utc)
    terminal_id = TerminalId("tmux", pane_id) if pane_id is not None else None
    status = SessionStatus.READY if event_state == "idle" else SessionStatus.BUSY
    return RecoveredSession(
        session_id=session_id,
        name="Ghost",
        project_path="/tmp/project",
        terminal_id=terminal_id,
        agent_type="claude",
        status=status,
        last_activity=now,
        created_at=now,
        event_state=event_state,  # type: ignore[arg-type]
        recovered_at=now,
        last_event_ts=now,
        worktree_path=worktree_path,
    )


def _recovered_iterm(
    *,
    session_id: str = "w1",
    uuid: str | None = "ABC-123",
    event_state: str = "active",
    worktree_path: str | None = None,
) -> RecoveredSession:
    now = datetime(2026, 2, 10, tzinfo=timezone.utc)
    terminal_id = TerminalId("iterm", uuid) if uuid is not None else None
    status = SessionStatus.READY if event_state == "idle" else SessionStatus.BUSY
    return RecoveredSession(
        session_id=session_id,
        name="Ghost",
        project_path="/tmp/project",
        terminal_id=terminal_id,
        agent_type="claude",
        status=status,
        last_activity=now,
        created_at=now,
        event_state=event_state,  # type: ignore[arg-type]
        recovered_at=now,
        last_event_ts=now,
        worktree_path=worktree_path,
    )


@pytest.mark.asyncio
async def test_prune_marks_closed_when_tmux_pane_missing() -> None:
    registry = SessionRegistry()
    registry._recovered_sessions["w1"] = _recovered(pane_id="%1", event_state="active")

    backend = _FakeTmuxBackend(panes=[])  # pane does not exist

    with patch("maniple.events.append_events") as mock_append:
        report = await registry.prune_stale_recovered_sessions(backend)

    assert report.pruned == 1
    assert report.emitted_closed == 1
    assert registry._recovered_sessions["w1"].event_state == "closed"
    mock_append.assert_called_once()
    emitted = mock_append.call_args[0][0]
    assert emitted[0].type == "worker_closed"
    assert emitted[0].data["reason"] == "stale_recovered"


@pytest.mark.asyncio
async def test_prune_noop_when_tmux_pane_exists() -> None:
    registry = SessionRegistry()
    registry._recovered_sessions["w1"] = _recovered(pane_id="%1", event_state="active")

    backend = _FakeTmuxBackend(panes=["%1"])  # pane exists

    with patch("maniple.events.append_events") as mock_append:
        report = await registry.prune_stale_recovered_sessions(backend)

    assert report.pruned == 0
    assert report.emitted_closed == 0
    assert registry._recovered_sessions["w1"].event_state == "active"
    mock_append.assert_not_called()


@pytest.mark.asyncio
async def test_prune_uses_missing_worktree_when_terminal_check_unavailable(tmp_path) -> None:
    missing = str(tmp_path / "does-not-exist")
    registry = SessionRegistry()
    registry._recovered_sessions["w1"] = _recovered(
        pane_id="%1",
        event_state="active",
        worktree_path=missing,
    )

    backend = _FakeTmuxBackend(raise_on_list=True)

    with patch("maniple.events.append_events") as mock_append:
        report = await registry.prune_stale_recovered_sessions(backend)

    assert report.pruned == 1
    assert report.emitted_closed == 1
    mock_append.assert_called_once()


@pytest.mark.asyncio
async def test_prune_does_not_prune_when_terminal_check_unavailable_and_no_worktree() -> None:
    registry = SessionRegistry()
    registry._recovered_sessions["w1"] = _recovered(
        pane_id="%1",
        event_state="active",
        worktree_path=None,
    )

    backend = _FakeTmuxBackend(raise_on_list=True)

    with patch("maniple.events.append_events") as mock_append:
        report = await registry.prune_stale_recovered_sessions(backend)

    assert report.pruned == 0
    assert report.emitted_closed == 0
    mock_append.assert_not_called()


@pytest.mark.asyncio
async def test_prune_marks_closed_when_iterm_session_missing() -> None:
    registry = SessionRegistry()
    registry._recovered_sessions["w1"] = _recovered_iterm(uuid="ABC-123", event_state="active")

    backend = _FakeItermBackend(sessions=[])  # UUID does not exist

    with patch("maniple.events.append_events") as mock_append:
        report = await registry.prune_stale_recovered_sessions(backend)

    assert report.pruned == 1
    assert report.emitted_closed == 1
    assert registry._recovered_sessions["w1"].event_state == "closed"
    mock_append.assert_called_once()


@pytest.mark.asyncio
async def test_prune_noop_when_iterm_session_exists() -> None:
    registry = SessionRegistry()
    registry._recovered_sessions["w1"] = _recovered_iterm(uuid="ABC-123", event_state="active")

    backend = _FakeItermBackend(sessions=["ABC-123"])  # UUID exists

    with patch("maniple.events.append_events") as mock_append:
        report = await registry.prune_stale_recovered_sessions(backend)

    assert report.pruned == 0
    assert report.emitted_closed == 0
    assert registry._recovered_sessions["w1"].event_state == "active"
    mock_append.assert_not_called()
