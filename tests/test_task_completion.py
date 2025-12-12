"""
Tests for task completion detection.
"""

import pytest
from datetime import datetime, timedelta

from claude_team_mcp.task_completion import (
    TaskStatus,
    TaskCompletionInfo,
    TaskContext,
    detect_markers_in_message,
    detect_from_conversation,
    detect_from_screen,
    COMPLETION_MARKERS,
    FAILURE_MARKERS,
)
from claude_team_mcp.session_state import Message, SessionState
from pathlib import Path


class TestMarkerDetection:
    """Test convention-based marker detection."""

    def test_explicit_completion_marker(self):
        """Test detection of explicit TASK_COMPLETE marker."""
        content = "I have finished the work.\n\nTASK_COMPLETE"
        status, confidence, marker = detect_markers_in_message(content)
        assert status == TaskStatus.COMPLETED
        assert confidence >= 0.9
        assert marker == "TASK_COMPLETE"

    def test_explicit_failure_marker(self):
        """Test detection of explicit TASK_FAILED marker."""
        content = "I could not complete the task due to errors.\n\nTASK_FAILED"
        status, confidence, marker = detect_markers_in_message(content)
        assert status == TaskStatus.FAILED
        assert confidence >= 0.9
        assert marker == "TASK_FAILED"

    def test_natural_language_completion(self):
        """Test detection of natural language completion patterns."""
        content = "I've completed the task successfully."
        status, confidence, marker = detect_markers_in_message(content)
        assert status == TaskStatus.COMPLETED
        assert confidence >= 0.7

    def test_natural_language_failure(self):
        """Test detection of natural language failure patterns."""
        content = "I cannot complete this task due to missing dependencies."
        status, confidence, marker = detect_markers_in_message(content)
        assert status == TaskStatus.FAILED
        assert confidence >= 0.7

    def test_no_markers_found(self):
        """Test when no completion markers are found."""
        content = "Let me analyze the code and make some changes."
        status, confidence, marker = detect_markers_in_message(content)
        assert status == TaskStatus.UNKNOWN
        assert confidence == 0.0

    def test_all_completion_markers(self):
        """Test all explicit completion markers are detected."""
        for marker in COMPLETION_MARKERS:
            content = f"Work done.\n{marker}"
            status, conf, detected = detect_markers_in_message(content)
            assert status == TaskStatus.COMPLETED, f"Failed for marker: {marker}"

    def test_all_failure_markers(self):
        """Test all explicit failure markers are detected."""
        for marker in FAILURE_MARKERS:
            content = f"Work failed.\n{marker}"
            status, conf, detected = detect_markers_in_message(content)
            assert status == TaskStatus.FAILED, f"Failed for marker: {marker}"


class TestConversationDetection:
    """Test detection from conversation state."""

    def _make_message(self, role: str, content: str, uuid: str = "msg-1") -> Message:
        """Helper to create a test message."""
        return Message(
            uuid=uuid,
            parent_uuid=None,
            role=role,
            content=content,
            timestamp=datetime.now(),
            tool_uses=[],
        )

    def _make_state(self, messages: list[Message]) -> SessionState:
        """Helper to create a test session state."""
        return SessionState(
            session_id="test-session",
            project_path="/test/path",
            jsonl_path=Path("/test/path/session.jsonl"),
            messages=messages,
        )

    def test_detect_completion_in_last_message(self):
        """Test completion detection in most recent assistant message."""
        messages = [
            self._make_message("user", "Please implement the feature", "u-1"),
            self._make_message("assistant", "Done! TASK_COMPLETE", "a-1"),
        ]
        state = self._make_state(messages)

        result = detect_from_conversation(state)
        assert result is not None
        assert result.status == TaskStatus.COMPLETED
        assert result.confidence >= 0.9

    def test_detect_failure_in_conversation(self):
        """Test failure detection in conversation."""
        messages = [
            self._make_message("user", "Please fix the bug", "u-1"),
            self._make_message(
                "assistant",
                "I encountered errors. TASK_FAILED",
                "a-1"
            ),
        ]
        state = self._make_state(messages)

        result = detect_from_conversation(state)
        assert result is not None
        assert result.status == TaskStatus.FAILED

    def test_no_detection_in_neutral_conversation(self):
        """Test no detection when conversation has no markers."""
        messages = [
            self._make_message("user", "What files are here?", "u-1"),
            self._make_message("assistant", "I found several files.", "a-1"),
        ]
        state = self._make_state(messages)

        result = detect_from_conversation(state)
        assert result is None

    def test_baseline_filtering(self):
        """Test that messages before baseline are ignored."""
        messages = [
            self._make_message("assistant", "TASK_COMPLETE", "old-msg"),
            self._make_message("user", "Now do this other task", "u-1"),
            self._make_message("assistant", "Working on it...", "a-1"),
        ]
        state = self._make_state(messages)

        # With baseline set to old-msg, only messages after should be checked
        result = detect_from_conversation(state, since_message_uuid="old-msg")
        # The "Working on it..." message doesn't have markers
        assert result is None


class TestTaskContext:
    """Test TaskContext creation and usage."""

    def test_task_context_creation(self):
        """Test TaskContext can be created with required fields."""
        ctx = TaskContext(
            session_id="worker-1",
            project_path="/path/to/project",
            started_at=datetime.now(),
            baseline_message_uuid="msg-123",
            beads_issue_id="a60",
            task_description="Implement the feature",
        )

        assert ctx.session_id == "worker-1"
        assert ctx.project_path == "/path/to/project"
        assert ctx.beads_issue_id == "a60"


class TestTaskCompletionInfo:
    """Test TaskCompletionInfo serialization."""

    def test_to_dict(self):
        """Test TaskCompletionInfo serializes correctly."""
        info = TaskCompletionInfo(
            status=TaskStatus.COMPLETED,
            confidence=0.95,
            detection_method="convention_markers",
            details={"marker": "TASK_COMPLETE"},
        )

        result = info.to_dict()
        assert result["status"] == "completed"
        assert result["confidence"] == 0.95
        assert result["detection_method"] == "convention_markers"
        assert "detected_at" in result


class TestScreenDetection:
    """Test screen-based completion detection."""

    def test_screen_convention_marker_completion(self):
        """Test that TASK_COMPLETE on screen is detected with high confidence."""
        import asyncio

        async def mock_read_screen(_session):
            return "Some output\nMore output\nTASK_COMPLETE\n> "

        async def run_test():
            return await detect_from_screen(None, mock_read_screen)

        result = asyncio.run(run_test())
        assert result is not None
        assert result.status == TaskStatus.COMPLETED
        assert result.confidence >= 0.9
        assert result.detection_method == "screen_convention_marker"
        assert result.details["matched_marker"] == "TASK_COMPLETE"

    def test_screen_convention_marker_failure(self):
        """Test that TASK_FAILED on screen is detected with high confidence."""
        import asyncio

        async def mock_read_screen(_session):
            return "Error occurred\nTASK_FAILED\n> "

        async def run_test():
            return await detect_from_screen(None, mock_read_screen)

        result = asyncio.run(run_test())
        assert result is not None
        assert result.status == TaskStatus.FAILED
        assert result.confidence >= 0.9
        assert result.detection_method == "screen_convention_marker"
        assert result.details["matched_marker"] == "TASK_FAILED"

    def test_screen_all_completion_markers(self):
        """Test all completion markers are detected on screen with high confidence."""
        import asyncio

        for marker in COMPLETION_MARKERS:
            async def mock_read_screen(_session, m=marker):
                return f"Output\n{m}\n> "

            async def run_test():
                return await detect_from_screen(None, mock_read_screen)

            result = asyncio.run(run_test())
            assert result is not None, f"Failed for marker: {marker}"
            assert result.status == TaskStatus.COMPLETED, f"Failed for marker: {marker}"
            assert result.confidence >= 0.9, f"Low confidence for marker: {marker}"
            assert result.detection_method == "screen_convention_marker"

    def test_screen_all_failure_markers(self):
        """Test all failure markers are detected on screen with high confidence."""
        import asyncio

        for marker in FAILURE_MARKERS:
            async def mock_read_screen(_session, m=marker):
                return f"Output\n{m}\n> "

            async def run_test():
                return await detect_from_screen(None, mock_read_screen)

            result = asyncio.run(run_test())
            assert result is not None, f"Failed for marker: {marker}"
            assert result.status == TaskStatus.FAILED, f"Failed for marker: {marker}"
            assert result.confidence >= 0.9, f"Low confidence for marker: {marker}"
            assert result.detection_method == "screen_convention_marker"

    def test_screen_generic_patterns_lower_confidence(self):
        """Test that generic patterns like 'done' have lower confidence."""
        import asyncio

        async def mock_read_screen(_session):
            return "Build succeeded\ndone\n> "

        async def run_test():
            return await detect_from_screen(None, mock_read_screen)

        result = asyncio.run(run_test())
        assert result is not None
        assert result.status == TaskStatus.COMPLETED
        # Generic patterns should have lower confidence than convention markers
        assert result.confidence < 0.9
        assert result.detection_method == "screen_parsing"

    def test_screen_no_markers(self):
        """Test no detection when screen has no markers."""
        import asyncio

        async def mock_read_screen(_session):
            return "Working on it...\nMaking changes...\n> "

        async def run_test():
            return await detect_from_screen(None, mock_read_screen)

        result = asyncio.run(run_test())
        assert result is None

    def test_screen_failure_marker_priority_over_completion(self):
        """Test that failure markers are checked before completion markers."""
        import asyncio

        async def mock_read_screen(_session):
            # TASK_FAILED should be detected even if success patterns present
            return "Build passed\nTASK_FAILED\n> "

        async def run_test():
            return await detect_from_screen(None, mock_read_screen)

        result = asyncio.run(run_test())
        assert result is not None
        assert result.status == TaskStatus.FAILED
        assert result.confidence >= 0.9
