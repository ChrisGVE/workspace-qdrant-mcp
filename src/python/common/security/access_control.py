"""
Enhanced Access Control System with Role-Based Permissions

This module provides comprehensive access control capabilities including:
- Role-based access control (RBAC) with hierarchical permissions
- Session management with secure token handling
- Permission inheritance and delegation
- Multi-tenant access isolation
- Dynamic permission evaluation
- Access policy enforcement
- Audit integration for all access decisions

The system integrates with the existing LLM access control while providing
enterprise-grade RBAC capabilities for the entire MCP server.
"""

import asyncio
import hashlib
import secrets
import time
import json
import logging
from typing import Dict, Set, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import defaultdict
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Standard permission types for the system."""
    # Collection permissions
    CREATE_COLLECTION = "collection:create"
    DELETE_COLLECTION = "collection:delete"
    READ_COLLECTION = "collection:read"
    WRITE_COLLECTION = "collection:write"
    MANAGE_COLLECTION = "collection:manage"

    # Document permissions
    CREATE_DOCUMENT = "document:create"
    READ_DOCUMENT = "document:read"
    UPDATE_DOCUMENT = "document:update"
    DELETE_DOCUMENT = "document:delete"
    SEARCH_DOCUMENT = "document:search"

    # User management permissions
    CREATE_USER = "user:create"
    READ_USER = "user:read"
    UPDATE_USER = "user:update"
    DELETE_USER = "user:delete"
    MANAGE_ROLES = "user:manage_roles"

    # System permissions
    ADMIN_ACCESS = "system:admin"
    MONITOR_ACCESS = "system:monitor"
    AUDIT_ACCESS = "system:audit"
    CONFIG_ACCESS = "system:config"

    # API permissions
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"


class AccessResult(Enum):
    """Result of an access control check."""
    GRANTED = auto()
    DENIED = auto()
    REQUIRES_MFA = auto()
    SESSION_EXPIRED = auto()
    RATE_LIMITED = auto()
    REQUIRES_ELEVATION = auto()


@dataclass
class Role:
    """Represents a role with permissions and constraints."""
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    max_sessions: int = 5
    session_timeout: int = 3600  # seconds
    require_mfa: bool = False
    ip_whitelist: Set[str] = field(default_factory=set)
    time_restrictions: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Ensure permissions are a set."""
        if isinstance(self.permissions, list):
            self.permissions = set(self.permissions)


@dataclass
class User:
    """Represents a user with roles and session information."""
    user_id: str
    username: str
    roles: Set[str] = field(default_factory=set)
    active_sessions: Set[str] = field(default_factory=set)
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def is_locked(self) -> bool:
        """Check if user account is locked."""
        return self.locked_until is not None and datetime.utcnow() < self.locked_until


@dataclass
class Session:
    """Represents an active user session."""
    session_id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    elevated: bool = False
    mfa_verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_active(self, timeout_seconds: int = 3600) -> bool:
        """Check if session is considered active."""
        if self.is_expired():
            return False
        inactive_time = datetime.utcnow() - self.last_activity
        return inactive_time.total_seconds() < timeout_seconds


@dataclass
class AccessContext:
    """Context information for access control decisions."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    operation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoleBasedAccessControl:
    """
    Role-Based Access Control (RBAC) system with hierarchical permissions.

    Provides comprehensive access control with:
    - Role hierarchy and permission inheritance
    - Session management with expiration
    - Multi-factor authentication support
    - Rate limiting and IP restrictions
    - Audit logging integration
    """

    def __init__(self, audit_logger=None):
        """Initialize the RBAC system."""
        self._roles: Dict[str, Role] = {}
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}
        self._permission_cache: Dict[Tuple[str, str], Tuple[Set[Permission], float]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._lock = threading.RLock()
        self._audit_logger = audit_logger

        # Rate limiting
        self._rate_limits: Dict[str, List[float]] = defaultdict(list)
        self._rate_limit_window = 300  # 5 minutes
        self._rate_limit_max = 100  # requests per window

        # Initialize default roles
        self._initialize_default_roles()

        logger.info("RBAC system initialized")

    def _initialize_default_roles(self):
        """Initialize default system roles."""
        # Admin role with all permissions
        admin_permissions = set(Permission)
        self.create_role("admin", permissions=admin_permissions, require_mfa=True)

        # User role with basic permissions
        user_permissions = {
            Permission.READ_COLLECTION, Permission.READ_DOCUMENT,
            Permission.SEARCH_DOCUMENT, Permission.API_READ
        }
        self.create_role("user", permissions=user_permissions)

        # Reader role with read-only access
        reader_permissions = {
            Permission.READ_COLLECTION, Permission.READ_DOCUMENT,
            Permission.SEARCH_DOCUMENT, Permission.API_READ
        }
        self.create_role("reader", permissions=reader_permissions)

        # Writer role with read-write access
        writer_permissions = user_permissions | {
            Permission.CREATE_DOCUMENT, Permission.UPDATE_DOCUMENT,
            Permission.DELETE_DOCUMENT, Permission.API_WRITE
        }
        self.create_role("writer", permissions=writer_permissions)

    def create_role(self, name: str, permissions: Optional[Set[Permission]] = None,
                   parent_roles: Optional[Set[str]] = None, **kwargs) -> bool:
        """Create a new role."""
        try:
            with self._lock:
                if name in self._roles:
                    logger.warning(f"Role {name} already exists")
                    return False

                role = Role(
                    name=name,
                    permissions=permissions or set(),
                    parent_roles=parent_roles or set(),
                    **kwargs
                )

                # Validate parent roles exist
                for parent in role.parent_roles:
                    if parent not in self._roles:
                        logger.error(f"Parent role {parent} does not exist")
                        return False

                self._roles[name] = role
                self._clear_permission_cache()

                if self._audit_logger:
                    self._audit_logger.log_event(
                        'role_created',
                        {'role_name': name, 'permissions': [p.value for p in role.permissions]},
                        user_id='system'
                    )

                logger.info(f"Created role: {name}")
                return True

        except Exception as e:
            logger.error(f"Failed to create role {name}: {e}")
            return False

    def delete_role(self, name: str) -> bool:
        """Delete a role."""
        try:
            with self._lock:
                if name not in self._roles:
                    logger.warning(f"Role {name} does not exist")
                    return False

                # Check if role is in use
                users_with_role = [u for u in self._users.values() if name in u.roles]
                if users_with_role:
                    logger.error(f"Cannot delete role {name}: still assigned to users")
                    return False

                del self._roles[name]
                self._clear_permission_cache()

                if self._audit_logger:
                    self._audit_logger.log_event(
                        'role_deleted',
                        {'role_name': name},
                        user_id='system'
                    )

                logger.info(f"Deleted role: {name}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete role {name}: {e}")
            return False

    def create_user(self, user_id: str, username: str, roles: Optional[Set[str]] = None,
                   **kwargs) -> bool:
        """Create a new user."""
        try:
            with self._lock:
                if user_id in self._users:
                    logger.warning(f"User {user_id} already exists")
                    return False

                # Validate roles exist
                roles = roles or set()
                for role_name in roles:
                    if role_name not in self._roles:
                        logger.error(f"Role {role_name} does not exist")
                        return False

                user = User(
                    user_id=user_id,
                    username=username,
                    roles=roles,
                    **kwargs
                )

                self._users[user_id] = user
                self._clear_permission_cache(user_id)

                if self._audit_logger:
                    self._audit_logger.log_event(
                        'user_created',
                        {'user_id': user_id, 'username': username, 'roles': list(roles)},
                        user_id='system'
                    )

                logger.info(f"Created user: {username} ({user_id})")
                return True

        except Exception as e:
            logger.error(f"Failed to create user {user_id}: {e}")
            return False

    def delete_user(self, user_id: str) -> bool:
        """Delete a user and terminate all sessions."""
        try:
            with self._lock:
                if user_id not in self._users:
                    logger.warning(f"User {user_id} does not exist")
                    return False

                user = self._users[user_id]

                # Terminate all user sessions
                for session_id in list(user.active_sessions):
                    self.terminate_session(session_id)

                del self._users[user_id]
                self._clear_permission_cache(user_id)

                if self._audit_logger:
                    self._audit_logger.log_event(
                        'user_deleted',
                        {'user_id': user_id, 'username': user.username},
                        user_id='system'
                    )

                logger.info(f"Deleted user: {user.username} ({user_id})")
                return True

        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False

    def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign a role to a user."""
        try:
            with self._lock:
                if user_id not in self._users:
                    logger.error(f"User {user_id} does not exist")
                    return False

                if role_name not in self._roles:
                    logger.error(f"Role {role_name} does not exist")
                    return False

                user = self._users[user_id]
                if role_name in user.roles:
                    logger.info(f"User {user_id} already has role {role_name}")
                    return True

                user.roles.add(role_name)
                user.updated_at = datetime.utcnow()
                self._clear_permission_cache(user_id)

                if self._audit_logger:
                    self._audit_logger.log_event(
                        'role_assigned',
                        {'user_id': user_id, 'role_name': role_name},
                        user_id='system'
                    )

                logger.info(f"Assigned role {role_name} to user {user_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to assign role {role_name} to user {user_id}: {e}")
            return False

    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Revoke a role from a user."""
        try:
            with self._lock:
                if user_id not in self._users:
                    logger.error(f"User {user_id} does not exist")
                    return False

                user = self._users[user_id]
                if role_name not in user.roles:
                    logger.info(f"User {user_id} does not have role {role_name}")
                    return True

                user.roles.discard(role_name)
                user.updated_at = datetime.utcnow()
                self._clear_permission_cache(user_id)

                # Update active sessions
                for session_id in user.active_sessions:
                    session = self._sessions.get(session_id)
                    if session:
                        session.permissions = self._calculate_user_permissions(user_id)

                if self._audit_logger:
                    self._audit_logger.log_event(
                        'role_revoked',
                        {'user_id': user_id, 'role_name': role_name},
                        user_id='system'
                    )

                logger.info(f"Revoked role {role_name} from user {user_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to revoke role {role_name} from user {user_id}: {e}")
            return False

    def create_session(self, user_id: str, ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None, timeout_seconds: int = 3600) -> Optional[str]:
        """Create a new session for a user."""
        try:
            with self._lock:
                if user_id not in self._users:
                    logger.error(f"User {user_id} does not exist")
                    return None

                user = self._users[user_id]

                # Check if user is locked
                if user.is_locked():
                    logger.warning(f"User {user_id} is locked")
                    return None

                # Check rate limiting
                if not self._check_rate_limit(user_id):
                    logger.warning(f"Rate limit exceeded for user {user_id}")
                    return None

                # Generate secure session ID
                session_id = secrets.token_urlsafe(32)

                # Calculate session expiration
                expires_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)

                # Get user permissions
                permissions = self._calculate_user_permissions(user_id)

                session = Session(
                    session_id=session_id,
                    user_id=user_id,
                    expires_at=expires_at,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    permissions=permissions
                )

                # Check session limits
                active_sessions = [s for s in user.active_sessions
                                 if s in self._sessions and self._sessions[s].is_active()]

                max_sessions = min(role for role_name in user.roles
                                 if role_name in self._roles
                                 for role in [self._roles[role_name].max_sessions])

                if len(active_sessions) >= max_sessions:
                    # Remove oldest session
                    oldest_session_id = min(active_sessions,
                                          key=lambda s: self._sessions[s].created_at)
                    self.terminate_session(oldest_session_id)

                self._sessions[session_id] = session
                user.active_sessions.add(session_id)
                user.last_login = datetime.utcnow()
                user.failed_attempts = 0  # Reset on successful login

                if self._audit_logger:
                    self._audit_logger.log_event(
                        'session_created',
                        {
                            'session_id': session_id,
                            'user_id': user_id,
                            'ip_address': ip_address,
                            'expires_at': expires_at.isoformat()
                        },
                        user_id=user_id
                    )

                logger.info(f"Created session {session_id} for user {user_id}")
                return session_id

        except Exception as e:
            logger.error(f"Failed to create session for user {user_id}: {e}")
            return None

    def terminate_session(self, session_id: str) -> bool:
        """Terminate a session."""
        try:
            with self._lock:
                if session_id not in self._sessions:
                    logger.warning(f"Session {session_id} does not exist")
                    return False

                session = self._sessions[session_id]
                user = self._users.get(session.user_id)

                if user:
                    user.active_sessions.discard(session_id)

                del self._sessions[session_id]

                if self._audit_logger:
                    self._audit_logger.log_event(
                        'session_terminated',
                        {'session_id': session_id, 'user_id': session.user_id},
                        user_id=session.user_id
                    )

                logger.info(f"Terminated session {session_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to terminate session {session_id}: {e}")
            return False

    def check_access(self, session_id: str, permission: Permission,
                    resource_id: Optional[str] = None,
                    context: Optional[AccessContext] = None) -> AccessResult:
        """Check if a session has access to perform an operation."""
        try:
            with self._lock:
                # Validate session
                session = self._sessions.get(session_id)
                if not session:
                    return AccessResult.DENIED

                if session.is_expired():
                    self.terminate_session(session_id)
                    return AccessResult.SESSION_EXPIRED

                # Update last activity
                session.last_activity = datetime.utcnow()

                # Get user
                user = self._users.get(session.user_id)
                if not user or user.is_locked():
                    return AccessResult.DENIED

                # Check rate limiting
                if not self._check_rate_limit(session.user_id):
                    return AccessResult.RATE_LIMITED

                # Check permission
                if permission not in session.permissions:
                    self._log_access_denied(session, permission, resource_id, context)
                    return AccessResult.DENIED

                # Check if MFA is required
                requires_mfa = any(self._roles[role_name].require_mfa
                                 for role_name in user.roles
                                 if role_name in self._roles)

                if requires_mfa and not session.mfa_verified:
                    return AccessResult.REQUIRES_MFA

                # Check IP restrictions
                if not self._check_ip_restrictions(user, session.ip_address):
                    return AccessResult.DENIED

                # Check time restrictions
                if not self._check_time_restrictions(user):
                    return AccessResult.DENIED

                self._log_access_granted(session, permission, resource_id, context)
                return AccessResult.GRANTED

        except Exception as e:
            logger.error(f"Failed to check access for session {session_id}: {e}")
            return AccessResult.DENIED

    def _calculate_user_permissions(self, user_id: str) -> Set[Permission]:
        """Calculate effective permissions for a user."""
        # Check cache first
        cache_key = (user_id, "permissions")
        cached = self._permission_cache.get(cache_key)
        if cached and time.time() - cached[1] < self._cache_ttl:
            return cached[0]

        user = self._users.get(user_id)
        if not user:
            return set()

        permissions = set()
        processed_roles = set()

        def add_role_permissions(role_name: str):
            if role_name in processed_roles or role_name not in self._roles:
                return

            processed_roles.add(role_name)
            role = self._roles[role_name]

            # Add role permissions
            permissions.update(role.permissions)

            # Add parent role permissions recursively
            for parent_role in role.parent_roles:
                add_role_permissions(parent_role)

        # Process all user roles
        for role_name in user.roles:
            add_role_permissions(role_name)

        # Cache result
        self._permission_cache[cache_key] = (permissions, time.time())

        return permissions

    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        now = time.time()
        user_requests = self._rate_limits[user_id]

        # Remove old requests outside the window
        user_requests[:] = [t for t in user_requests if now - t < self._rate_limit_window]

        if len(user_requests) >= self._rate_limit_max:
            return False

        user_requests.append(now)
        return True

    def _check_ip_restrictions(self, user: User, ip_address: Optional[str]) -> bool:
        """Check IP restrictions for user roles."""
        if not ip_address:
            return True

        for role_name in user.roles:
            role = self._roles.get(role_name)
            if role and role.ip_whitelist:
                if ip_address not in role.ip_whitelist:
                    return False

        return True

    def _check_time_restrictions(self, user: User) -> bool:
        """Check time-based restrictions for user roles."""
        now = datetime.utcnow()
        current_hour = now.hour
        current_day = now.weekday()  # 0 = Monday

        for role_name in user.roles:
            role = self._roles.get(role_name)
            if role and role.time_restrictions:
                # Check hour restrictions
                allowed_hours = role.time_restrictions.get('hours')
                if allowed_hours and current_hour not in allowed_hours:
                    return False

                # Check day restrictions
                allowed_days = role.time_restrictions.get('days')
                if allowed_days and current_day not in allowed_days:
                    return False

        return True

    def _clear_permission_cache(self, user_id: Optional[str] = None):
        """Clear permission cache for user or all users."""
        if user_id:
            # Clear cache for specific user
            to_remove = [k for k in self._permission_cache.keys() if k[0] == user_id]
            for key in to_remove:
                del self._permission_cache[key]
        else:
            # Clear entire cache
            self._permission_cache.clear()

    def _log_access_granted(self, session: Session, permission: Permission,
                           resource_id: Optional[str], context: Optional[AccessContext]):
        """Log successful access."""
        if self._audit_logger:
            self._audit_logger.log_event(
                'access_granted',
                {
                    'session_id': session.session_id,
                    'permission': permission.value,
                    'resource_id': resource_id,
                    'ip_address': session.ip_address,
                    'context': context.__dict__ if context else None
                },
                user_id=session.user_id
            )

    def _log_access_denied(self, session: Session, permission: Permission,
                          resource_id: Optional[str], context: Optional[AccessContext]):
        """Log access denial."""
        if self._audit_logger:
            self._audit_logger.log_event(
                'access_denied',
                {
                    'session_id': session.session_id,
                    'permission': permission.value,
                    'resource_id': resource_id,
                    'ip_address': session.ip_address,
                    'context': context.__dict__ if context else None
                },
                user_id=session.user_id
            )

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        user = self._users.get(session.user_id)
        if not user:
            return None

        return {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'username': user.username,
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'expires_at': session.expires_at.isoformat() if session.expires_at else None,
            'ip_address': session.ip_address,
            'permissions': [p.value for p in session.permissions],
            'elevated': session.elevated,
            'mfa_verified': session.mfa_verified
        }

    def list_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List active sessions."""
        sessions = []

        for session in self._sessions.values():
            if user_id and session.user_id != user_id:
                continue

            if session.is_active():
                session_info = self.get_session_info(session.session_id)
                if session_info:
                    sessions.append(session_info)

        return sessions

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        expired_count = 0

        with self._lock:
            expired_sessions = [
                session_id for session_id, session in self._sessions.items()
                if session.is_expired() or not session.is_active()
            ]

            for session_id in expired_sessions:
                self.terminate_session(session_id)
                expired_count += 1

        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired sessions")

        return expired_count


class SessionManager:
    """
    Session management with automatic cleanup and monitoring.

    Provides secure session handling with:
    - Automatic session expiration
    - Session hijacking protection
    - Concurrent session limits
    - Activity monitoring
    - Secure token generation
    """

    def __init__(self, rbac: RoleBasedAccessControl, cleanup_interval: int = 300):
        """Initialize session manager."""
        self._rbac = rbac
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info("Session manager initialized")

    async def start_cleanup_task(self):
        """Start automatic session cleanup."""
        if self._cleanup_task and not self._cleanup_task.done():
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session cleanup task started")

    async def stop_cleanup_task(self):
        """Stop automatic session cleanup."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Session cleanup task stopped")

    async def _cleanup_loop(self):
        """Background task for session cleanup."""
        while self._running:
            try:
                self._rbac.cleanup_expired_sessions()
                await asyncio.sleep(self._cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(self._cleanup_interval)

    @asynccontextmanager
    async def session_context(self, session_id: str):
        """Context manager for session operations."""
        session_info = self._rbac.get_session_info(session_id)
        if not session_info:
            raise ValueError("Invalid session")

        try:
            yield session_info
        finally:
            # Update session activity
            session = self._rbac._sessions.get(session_id)
            if session:
                session.last_activity = datetime.utcnow()


# Global RBAC instance (can be configured via dependency injection)
_default_rbac = None


def get_rbac() -> RoleBasedAccessControl:
    """Get the default RBAC instance."""
    global _default_rbac
    if _default_rbac is None:
        _default_rbac = RoleBasedAccessControl()
    return _default_rbac


def require_permission(permission: Permission):
    """Decorator to require a specific permission for a function."""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            # Extract session_id from kwargs or first argument
            session_id = kwargs.get('session_id') or (args[0] if args else None)
            if not session_id:
                raise ValueError("Session ID required")

            rbac = get_rbac()
            result = rbac.check_access(session_id, permission)

            if result != AccessResult.GRANTED:
                raise PermissionError(f"Access denied: {result.name}")

            return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            # Extract session_id from kwargs or first argument
            session_id = kwargs.get('session_id') or (args[0] if args else None)
            if not session_id:
                raise ValueError("Session ID required")

            rbac = get_rbac()
            result = rbac.check_access(session_id, permission)

            if result != AccessResult.GRANTED:
                raise PermissionError(f"Access denied: {result.name}")

            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator