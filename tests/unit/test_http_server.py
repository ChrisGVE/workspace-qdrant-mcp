"""
Unit tests for HTTP server.

Tests all endpoints, session management, error handling, and daemon integration.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from common.grpc.daemon_client import (
    DaemonTimeoutError,
    DaemonUnavailableError,
)
from common.grpc.generated import workspace_daemon_pb2 as pb2
from fastapi.testclient import TestClient
from workspace_qdrant_mcp.http_server import (
    HealthResponse,
    SessionInfo,
    SessionManager,
    SuccessResponse,
    app,
)


@pytest.fixture
def mock_daemon_client():
    """Create a mock DaemonClient."""
    client = AsyncMock()

    # Health check response
    health_response = pb2.HealthCheckResponse(
        status=pb2.SERVICE_STATUS_HEALTHY
    )
    client.health_check = AsyncMock(return_value=health_response)

    # Status response
    status_response = pb2.SystemStatusResponse(
        status=pb2.SERVICE_STATUS_HEALTHY
    )
    client.get_status = AsyncMock(return_value=status_response)

    # Notify server status returns Empty
    client.notify_server_status = AsyncMock()

    return client


@pytest.fixture
def session_manager(mock_daemon_client):
    """Create a SessionManager with mocked daemon client."""
    return SessionManager(daemon_client=mock_daemon_client)


class TestSessionManager:
    """Test SessionManager functionality."""

    @pytest.mark.asyncio
    async def test_start_session_tracks_session(self, session_manager):
        """Test that start_session adds session to active sessions."""
        result = await session_manager.start_session(
            session_id="test-123",
            project_dir="/path/to/project",
            source="startup"
        )

        assert result.success is True
        assert result.session_id == "test-123"
        assert "test-123" in session_manager.active_sessions
        assert session_manager.get_active_session_count() == 1

        session_info = session_manager.active_sessions["test-123"]
        assert session_info.session_id == "test-123"
        assert session_info.project_dir == "/path/to/project"
        assert session_info.source == "startup"

    @pytest.mark.asyncio
    async def test_start_session_notifies_daemon(self, session_manager, mock_daemon_client):
        """Test that start_session notifies daemon of SERVER_STATE_UP."""
        await session_manager.start_session(
            session_id="test-123",
            project_dir="/path/to/project",
            source="startup"
        )

        mock_daemon_client.notify_server_status.assert_called_once()
        call_args = mock_daemon_client.notify_server_status.call_args
        assert call_args.kwargs["state"] == pb2.SERVER_STATE_UP
        assert call_args.kwargs["project_root"] == "/path/to/project"

    @pytest.mark.asyncio
    async def test_start_session_handles_daemon_failure(self, session_manager, mock_daemon_client):
        """Test that start_session continues if daemon notification fails."""
        mock_daemon_client.notify_server_status.side_effect = DaemonUnavailableError("Daemon down")

        result = await session_manager.start_session(
            session_id="test-123",
            project_dir="/path/to/project",
            source="startup"
        )

        # Should still succeed
        assert result.success is True
        assert "test-123" in session_manager.active_sessions

    @pytest.mark.asyncio
    async def test_end_session_removes_session(self, session_manager):
        """Test that end_session removes session from active sessions."""
        # Start session first
        await session_manager.start_session(
            session_id="test-123",
            project_dir="/path/to/project",
            source="startup"
        )

        # End session
        result = await session_manager.end_session(
            session_id="test-123",
            reason="clear"
        )

        assert result.success is True
        assert "test-123" not in session_manager.active_sessions
        assert session_manager.get_active_session_count() == 0

    @pytest.mark.asyncio
    async def test_end_session_notifies_daemon_for_other_reason(self, session_manager, mock_daemon_client):
        """Test that end_session notifies daemon for 'other' reason."""
        # Start session
        await session_manager.start_session(
            session_id="test-123",
            project_dir="/path/to/project",
            source="startup"
        )

        # Reset mock to check only end notification
        mock_daemon_client.notify_server_status.reset_mock()

        # End session with "other" reason
        await session_manager.end_session(
            session_id="test-123",
            reason="other"
        )

        mock_daemon_client.notify_server_status.assert_called_once()
        call_args = mock_daemon_client.notify_server_status.call_args
        assert call_args.kwargs["state"] == pb2.SERVER_STATE_DOWN

    @pytest.mark.asyncio
    async def test_end_session_notifies_daemon_for_prompt_input_exit(self, session_manager, mock_daemon_client):
        """Test that end_session notifies daemon for 'prompt_input_exit' reason."""
        # Start session
        await session_manager.start_session(
            session_id="test-123",
            project_dir="/path/to/project",
            source="startup"
        )

        # Reset mock
        mock_daemon_client.notify_server_status.reset_mock()

        # End session with "prompt_input_exit" reason
        await session_manager.end_session(
            session_id="test-123",
            reason="prompt_input_exit"
        )

        mock_daemon_client.notify_server_status.assert_called_once()
        call_args = mock_daemon_client.notify_server_status.call_args
        assert call_args.kwargs["state"] == pb2.SERVER_STATE_DOWN

    @pytest.mark.asyncio
    async def test_end_session_no_daemon_notification_for_clear(self, session_manager, mock_daemon_client):
        """Test that end_session doesn't notify daemon for 'clear' reason."""
        # Start session
        await session_manager.start_session(
            session_id="test-123",
            project_dir="/path/to/project",
            source="startup"
        )

        # Reset mock
        mock_daemon_client.notify_server_status.reset_mock()

        # End session with "clear" reason
        await session_manager.end_session(
            session_id="test-123",
            reason="clear"
        )

        # Should not notify daemon
        mock_daemon_client.notify_server_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_end_unknown_session_returns_success(self, session_manager):
        """Test that ending unknown session returns success (idempotent)."""
        result = await session_manager.end_session(
            session_id="unknown-session",
            reason="clear"
        )

        assert result.success is True
        assert "not found" in result.message

    def test_get_uptime_seconds(self, session_manager):
        """Test that uptime is tracked."""
        uptime = session_manager.get_uptime_seconds()
        assert uptime >= 0
        assert uptime < 1.0  # Should be very recent


class TestHTTPEndpoints:
    """Test HTTP endpoint functionality."""

    @pytest.fixture
    def client(self):
        """Create TestClient for FastAPI app."""
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def setup_session_manager(self, mock_daemon_client):
        """Set up global session_manager for tests."""
        from workspace_qdrant_mcp import http_server
        http_server.session_manager = SessionManager(daemon_client=mock_daemon_client)
        http_server.daemon_client = mock_daemon_client
        yield
        http_server.session_manager = None
        http_server.daemon_client = None

    def test_session_start_endpoint(self, client):
        """Test POST /api/v1/hooks/session-start."""
        response = client.post(
            "/api/v1/hooks/session-start",
            json={
                "session_id": "test-123",
                "project_dir": "/path/to/project",
                "source": "startup"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["session_id"] == "test-123"

    def test_session_start_invalid_source(self, client):
        """Test session start with invalid source value."""
        response = client.post(
            "/api/v1/hooks/session-start",
            json={
                "session_id": "test-123",
                "project_dir": "/path/to/project",
                "source": "invalid"
            }
        )

        assert response.status_code == 422  # Validation error

    def test_session_start_missing_field(self, client):
        """Test session start with missing required field."""
        response = client.post(
            "/api/v1/hooks/session-start",
            json={
                "session_id": "test-123",
                "source": "startup"
                # Missing project_dir
            }
        )

        assert response.status_code == 422

    def test_session_end_endpoint(self, client):
        """Test POST /api/v1/hooks/session-end."""
        # Start session first
        client.post(
            "/api/v1/hooks/session-start",
            json={
                "session_id": "test-123",
                "project_dir": "/path/to/project",
                "source": "startup"
            }
        )

        # End session
        response = client.post(
            "/api/v1/hooks/session-end",
            json={
                "session_id": "test-123",
                "reason": "clear"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["session_id"] == "test-123"

    def test_session_end_invalid_reason(self, client):
        """Test session end with invalid reason value."""
        response = client.post(
            "/api/v1/hooks/session-end",
            json={
                "session_id": "test-123",
                "reason": "invalid"
            }
        )

        assert response.status_code == 422

    def test_health_endpoint_healthy(self, client, mock_daemon_client):
        """Test GET /api/v1/health with healthy daemon."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["daemon_connected"] is True
        assert data["qdrant_connected"] is True
        assert data["version"] == "0.2.1"
        assert data["active_sessions"] >= 0

    def test_health_endpoint_daemon_unavailable(self, client, mock_daemon_client):
        """Test health endpoint when daemon is unavailable."""
        mock_daemon_client.health_check.side_effect = DaemonUnavailableError("Daemon down")

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["daemon_connected"] is False
        assert data["qdrant_connected"] is False

    def test_health_endpoint_daemon_timeout(self, client, mock_daemon_client):
        """Test health endpoint when daemon times out."""
        mock_daemon_client.health_check.side_effect = DaemonTimeoutError("Timeout")

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["daemon_connected"] is False

    def test_pre_tool_use_placeholder(self, client):
        """Test POST /api/v1/hooks/pre-tool-use placeholder."""
        response = client.post(
            "/api/v1/hooks/pre-tool-use",
            json={
                "session_id": "test-123",
                "project_dir": "/path/to/project",
                "tool_name": "Edit"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "placeholder" in data["message"]

    def test_post_tool_use_placeholder(self, client):
        """Test POST /api/v1/hooks/post-tool-use placeholder."""
        response = client.post(
            "/api/v1/hooks/post-tool-use",
            json={
                "session_id": "test-123",
                "project_dir": "/path/to/project",
                "tool_name": "Write"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_user_prompt_submit_placeholder(self, client):
        """Test POST /api/v1/hooks/user-prompt-submit placeholder."""
        response = client.post(
            "/api/v1/hooks/user-prompt-submit",
            json={
                "session_id": "test-123",
                "project_dir": "/path/to/project"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_notification_placeholder(self, client):
        """Test POST /api/v1/hooks/notification placeholder."""
        response = client.post(
            "/api/v1/hooks/notification",
            json={
                "session_id": "test-123",
                "project_dir": "/path/to/project"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_stop_hook_placeholder(self, client):
        """Test POST /api/v1/hooks/stop placeholder."""
        response = client.post(
            "/api/v1/hooks/stop",
            json={
                "session_id": "test-123",
                "project_dir": "/path/to/project"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_subagent_stop_placeholder(self, client):
        """Test POST /api/v1/hooks/subagent-stop placeholder."""
        response = client.post(
            "/api/v1/hooks/subagent-stop",
            json={
                "session_id": "test-123",
                "project_dir": "/path/to/project"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_pre_compact_placeholder(self, client):
        """Test POST /api/v1/hooks/pre-compact placeholder."""
        response = client.post(
            "/api/v1/hooks/pre-compact",
            json={
                "session_id": "test-123",
                "project_dir": "/path/to/project"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestErrorHandling:
    """Test error handling."""

    @pytest.fixture
    def client(self):
        """Create TestClient for FastAPI app."""
        return TestClient(app)

    def test_invalid_json(self, client):
        """Test request with invalid JSON."""
        response = client.post(
            "/api/v1/hooks/session-start",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_session_manager_not_initialized(self, client):
        """Test endpoint when session_manager is None."""
        from workspace_qdrant_mcp import http_server
        http_server.session_manager = None

        response = client.post(
            "/api/v1/hooks/session-start",
            json={
                "session_id": "test-123",
                "project_dir": "/path/to/project",
                "source": "startup"
            }
        )

        assert response.status_code == 503
        assert "not initialized" in response.json()["error"]


class TestConcurrency:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, session_manager):
        """Test handling multiple concurrent sessions."""
        # Start multiple sessions
        await session_manager.start_session("session-1", "/project/1", "startup")
        await session_manager.start_session("session-2", "/project/2", "clear")
        await session_manager.start_session("session-3", "/project/3", "compact")

        assert session_manager.get_active_session_count() == 3

        # End sessions in different order
        await session_manager.end_session("session-2", "clear")
        assert session_manager.get_active_session_count() == 2

        await session_manager.end_session("session-1", "logout")
        assert session_manager.get_active_session_count() == 1

        await session_manager.end_session("session-3", "other")
        assert session_manager.get_active_session_count() == 0
