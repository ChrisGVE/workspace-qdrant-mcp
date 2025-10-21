"""
HTTP server for Claude Code hooks integration.

Provides HTTP endpoints for Claude Code to integrate with workspace-qdrant-mcp
during development sessions. Handles session lifecycle, file change notifications,
and provides health status for monitoring.

This server complements the MCP stdio server by enabling Claude Code hooks
to trigger memory ingestion, track active sessions, and manage daemon state.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Literal, Optional, Any
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from common.grpc.daemon_client import (
    DaemonClient,
    DaemonUnavailableError,
    DaemonTimeoutError,
)
from common.grpc.generated import workspace_daemon_pb2 as pb2

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class SessionStartRequest(BaseModel):
    """Request to start a new Claude Code session."""
    session_id: str = Field(..., description="Unique session identifier")
    project_dir: str = Field(..., description="Absolute path to project directory")
    source: Literal["startup", "clear", "compact"] = Field(
        ..., description="Event that triggered session start"
    )


class SessionEndRequest(BaseModel):
    """Request to end an active Claude Code session."""
    session_id: str = Field(..., description="Session identifier to end")
    reason: Literal["clear", "logout", "other", "prompt_input_exit"] = Field(
        ..., description="Reason for session termination"
    )


class HookRequest(BaseModel):
    """Generic request for hook endpoints."""
    session_id: str = Field(..., description="Active session identifier")
    project_dir: str = Field(..., description="Project directory path")
    tool_name: Optional[str] = Field(None, description="Name of tool being used")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional hook data")


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    daemon_connected: bool
    qdrant_connected: bool
    version: str
    uptime_seconds: float
    active_sessions: int


class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool
    message: str
    session_id: Optional[str] = None


# =============================================================================
# Session Management
# =============================================================================

@dataclass
class SessionInfo:
    """Information about an active session."""
    session_id: str
    project_dir: str
    started_at: datetime
    source: str


class SessionManager:
    """
    Manages active Claude Code sessions.

    Tracks which projects have active sessions, notifies daemon of session
    lifecycle events, and triggers memory ingestion when sessions start.
    """

    def __init__(self, daemon_client: Optional[DaemonClient] = None):
        """
        Initialize session manager.

        Args:
            daemon_client: Optional DaemonClient instance (created if None)
        """
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.daemon_client = daemon_client
        self._start_time = datetime.now()

    async def start_session(
        self,
        session_id: str,
        project_dir: str,
        source: str
    ) -> SuccessResponse:
        """
        Start a new session and trigger memory ingestion.

        Args:
            session_id: Unique session identifier
            project_dir: Absolute path to project directory
            source: Event that triggered session start

        Returns:
            SuccessResponse indicating result

        Raises:
            HTTPException: If daemon communication fails
        """
        # Track session
        session_info = SessionInfo(
            session_id=session_id,
            project_dir=project_dir,
            started_at=datetime.now(),
            source=source
        )
        self.active_sessions[session_id] = session_info

        logger.info(
            f"Session started: {session_id}, project={project_dir}, "
            f"source={source}, total_sessions={len(self.active_sessions)}"
        )

        # Notify daemon of server status
        if self.daemon_client:
            try:
                await self.daemon_client.notify_server_status(
                    state=pb2.SERVER_STATE_UP,
                    project_root=project_dir
                )
                logger.info(f"Notified daemon of session start: {session_id}")
            except (DaemonUnavailableError, DaemonTimeoutError) as e:
                logger.warning(f"Failed to notify daemon: {e}")
                # Don't fail the request if daemon notification fails
            except Exception as e:
                logger.error(f"Unexpected error notifying daemon: {e}")

        return SuccessResponse(
            success=True,
            message=f"Session {session_id} started successfully",
            session_id=session_id
        )

    async def end_session(
        self,
        session_id: str,
        reason: str
    ) -> SuccessResponse:
        """
        End an active session and notify daemon if needed.

        Args:
            session_id: Session identifier to end
            reason: Reason for session termination

        Returns:
            SuccessResponse indicating result
        """
        # Get session info before removing
        session_info = self.active_sessions.get(session_id)

        if not session_info:
            logger.warning(f"Attempted to end unknown session: {session_id}")
            return SuccessResponse(
                success=True,
                message=f"Session {session_id} not found (already ended)",
                session_id=session_id
            )

        # Remove from active sessions
        del self.active_sessions[session_id]

        logger.info(
            f"Session ended: {session_id}, reason={reason}, "
            f"remaining_sessions={len(self.active_sessions)}"
        )

        # Notify daemon of server status for certain reasons
        if reason in ["other", "prompt_input_exit"] and self.daemon_client:
            try:
                await self.daemon_client.notify_server_status(
                    state=pb2.SERVER_STATE_DOWN,
                    project_root=session_info.project_dir
                )
                logger.info(f"Notified daemon of session end: {session_id}")
            except (DaemonUnavailableError, DaemonTimeoutError) as e:
                logger.warning(f"Failed to notify daemon: {e}")
            except Exception as e:
                logger.error(f"Unexpected error notifying daemon: {e}")

        return SuccessResponse(
            success=True,
            message=f"Session {session_id} ended ({reason})",
            session_id=session_id
        )

    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self.active_sessions)

    def get_uptime_seconds(self) -> float:
        """Get server uptime in seconds."""
        return (datetime.now() - self._start_time).total_seconds()


# =============================================================================
# FastAPI Application
# =============================================================================

# Global session manager (initialized in lifespan)
session_manager: Optional[SessionManager] = None
daemon_client: Optional[DaemonClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Initializes daemon client and session manager on startup,
    cleans up on shutdown.
    """
    global session_manager, daemon_client

    # Startup
    logger.info("Starting HTTP server for Claude Code hooks")

    # Initialize daemon client
    daemon_client = DaemonClient()
    try:
        await daemon_client.start()
        logger.info("Daemon client connected successfully")
    except Exception as e:
        logger.warning(f"Failed to connect to daemon: {e}")
        # Continue without daemon - endpoints will handle gracefully

    # Initialize session manager
    session_manager = SessionManager(daemon_client=daemon_client)
    logger.info("Session manager initialized")

    yield

    # Shutdown
    logger.info("Shutting down HTTP server")
    if daemon_client:
        await daemon_client.stop()
    logger.info("HTTP server shutdown complete")


app = FastAPI(
    title="Workspace Qdrant MCP - Claude Code Hooks",
    description="HTTP endpoints for Claude Code hook integrations",
    version="0.3.0",
    lifespan=lifespan
)


# =============================================================================
# Critical Endpoints (Phase 1)
# =============================================================================

@app.post("/api/v1/hooks/session-start", response_model=SuccessResponse)
async def session_start(request: SessionStartRequest) -> SuccessResponse:
    """
    Handle Claude Code session start event.

    Tracks the session, notifies daemon of server status, and triggers
    memory collection ingestion for the project.

    Args:
        request: Session start request with session_id, project_dir, source

    Returns:
        SuccessResponse with session details

    Raises:
        HTTPException: 503 if session manager not initialized
    """
    if not session_manager:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized"
        )

    try:
        return await session_manager.start_session(
            session_id=request.session_id,
            project_dir=request.project_dir,
            source=request.source
        )
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start session: {str(e)}"
        )


@app.post("/api/v1/hooks/session-end", response_model=SuccessResponse)
async def session_end(request: SessionEndRequest) -> SuccessResponse:
    """
    Handle Claude Code session end event.

    Cleans up session tracking and notifies daemon of server status
    change for relevant termination reasons.

    Args:
        request: Session end request with session_id and reason

    Returns:
        SuccessResponse with session details

    Raises:
        HTTPException: 503 if session manager not initialized
    """
    if not session_manager:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized"
        )

    try:
        return await session_manager.end_session(
            session_id=request.session_id,
            reason=request.reason
        )
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to end session: {str(e)}"
        )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring.

    Returns daemon connection status, Qdrant connection status,
    version information, and active session count.

    Returns:
        HealthResponse with system health details
    """
    daemon_connected = False
    qdrant_connected = False
    status = "unhealthy"

    # Check daemon connection
    if daemon_client:
        try:
            health_response = await daemon_client.health_check(timeout=2.0)
            daemon_connected = health_response.status == pb2.SERVICE_STATUS_HEALTHY

            # Get Qdrant status from daemon
            if daemon_connected:
                system_status = await daemon_client.get_status(timeout=2.0)
                qdrant_connected = system_status.status == pb2.SERVICE_STATUS_HEALTHY
        except (DaemonUnavailableError, DaemonTimeoutError):
            daemon_connected = False
        except Exception as e:
            logger.warning(f"Health check error: {e}")

    # Determine overall status
    if daemon_connected and qdrant_connected:
        status = "healthy"
    elif daemon_connected or qdrant_connected:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        daemon_connected=daemon_connected,
        qdrant_connected=qdrant_connected,
        version="0.3.0",  # TODO: Get from package metadata
        uptime_seconds=session_manager.get_uptime_seconds() if session_manager else 0.0,
        active_sessions=session_manager.get_active_session_count() if session_manager else 0
    )


# =============================================================================
# Placeholder Endpoints (Phase 2)
# =============================================================================

@app.post("/api/v1/hooks/pre-tool-use", response_model=SuccessResponse)
async def pre_tool_use(request: HookRequest) -> SuccessResponse:
    """
    Placeholder for pre-tool-use hook.

    Future: Pause file watchers before tool execution.
    """
    logger.debug(f"Pre-tool-use hook called: session={request.session_id}, tool={request.tool_name}")
    return SuccessResponse(
        success=True,
        message="Pre-tool-use hook (placeholder)",
        session_id=request.session_id
    )


@app.post("/api/v1/hooks/post-tool-use", response_model=SuccessResponse)
async def post_tool_use(request: HookRequest) -> SuccessResponse:
    """
    Placeholder for post-tool-use hook.

    Future: Resume file watchers and queue changed files.
    """
    logger.debug(f"Post-tool-use hook called: session={request.session_id}, tool={request.tool_name}")
    return SuccessResponse(
        success=True,
        message="Post-tool-use hook (placeholder)",
        session_id=request.session_id
    )


@app.post("/api/v1/hooks/user-prompt-submit", response_model=SuccessResponse)
async def user_prompt_submit(request: HookRequest) -> SuccessResponse:
    """
    Placeholder for user-prompt-submit hook.

    Future: Trigger memory refresh before prompt processing.
    """
    logger.debug(f"User-prompt-submit hook called: session={request.session_id}")
    return SuccessResponse(
        success=True,
        message="User-prompt-submit hook (placeholder)",
        session_id=request.session_id
    )


@app.post("/api/v1/hooks/notification", response_model=SuccessResponse)
async def notification(request: HookRequest) -> SuccessResponse:
    """
    Placeholder for notification hook.

    Future: Log or forward notifications to monitoring.
    """
    logger.debug(f"Notification hook called: session={request.session_id}")
    return SuccessResponse(
        success=True,
        message="Notification hook (placeholder)",
        session_id=request.session_id
    )


@app.post("/api/v1/hooks/stop", response_model=SuccessResponse)
async def stop_hook(request: HookRequest) -> SuccessResponse:
    """
    Placeholder for stop hook.

    Future: Handle graceful shutdown requests.
    """
    logger.debug(f"Stop hook called: session={request.session_id}")
    return SuccessResponse(
        success=True,
        message="Stop hook (placeholder)",
        session_id=request.session_id
    )


@app.post("/api/v1/hooks/subagent-stop", response_model=SuccessResponse)
async def subagent_stop(request: HookRequest) -> SuccessResponse:
    """
    Placeholder for subagent-stop hook.

    Future: Handle subagent completion events.
    """
    logger.debug(f"Subagent-stop hook called: session={request.session_id}")
    return SuccessResponse(
        success=True,
        message="Subagent-stop hook (placeholder)",
        session_id=request.session_id
    )


@app.post("/api/v1/hooks/pre-compact", response_model=SuccessResponse)
async def pre_compact(request: HookRequest) -> SuccessResponse:
    """
    Placeholder for pre-compact hook.

    Future: Prepare for conversation compaction.
    """
    logger.debug(f"Pre-compact hook called: session={request.session_id}")
    return SuccessResponse(
        success=True,
        message="Pre-compact hook (placeholder)",
        session_id=request.session_id
    )


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with JSON responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred"
        }
    )
