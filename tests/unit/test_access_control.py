"""
Unit tests for the enhanced access control system.

This test suite provides comprehensive coverage of:
- Role-based access control (RBAC) functionality
- User and role management
- Session management with security features
- Permission inheritance and evaluation
- Rate limiting and security restrictions
- Session cleanup and expiration handling
- Error conditions and edge cases
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Set

from src.python.common.security.access_control import (
    RoleBasedAccessControl, SessionManager, Permission, AccessResult,
    Role, User, Session, AccessContext, get_rbac, require_permission
)


class TestRole:
    """Test Role data class functionality."""

    def test_role_creation(self):
        """Test basic role creation."""
        permissions = {Permission.READ_COLLECTION, Permission.WRITE_COLLECTION}
        role = Role(
            name="test_role",
            permissions=permissions,
            parent_roles={"parent_role"},
            max_sessions=10,
            require_mfa=True
        )

        assert role.name == "test_role"
        assert role.permissions == permissions
        assert role.parent_roles == {"parent_role"}
        assert role.max_sessions == 10
        assert role.require_mfa is True
        assert isinstance(role.created_at, datetime)

    def test_role_with_list_permissions(self):
        """Test role creation with permissions as list."""
        permissions = [Permission.READ_COLLECTION, Permission.WRITE_COLLECTION]
        role = Role(name="test_role", permissions=permissions)

        assert isinstance(role.permissions, set)
        assert role.permissions == set(permissions)

    def test_role_defaults(self):
        """Test role with default values."""
        role = Role(name="minimal_role")

        assert role.permissions == set()
        assert role.parent_roles == set()
        assert role.max_sessions == 5
        assert role.session_timeout == 3600
        assert role.require_mfa is False
        assert role.ip_whitelist == set()
        assert role.time_restrictions == {}
        assert role.resource_limits == {}


class TestUser:
    """Test User data class functionality."""

    def test_user_creation(self):
        """Test basic user creation."""
        user = User(
            user_id="user123",
            username="testuser",
            roles={"admin", "user"},
            mfa_enabled=True
        )

        assert user.user_id == "user123"
        assert user.username == "testuser"
        assert user.roles == {"admin", "user"}
        assert user.mfa_enabled is True
        assert user.failed_attempts == 0
        assert user.locked_until is None
        assert isinstance(user.created_at, datetime)

    def test_user_is_locked_false(self):
        """Test user is not locked."""
        user = User(user_id="user123", username="testuser")
        assert not user.is_locked()

    def test_user_is_locked_true(self):
        """Test user is locked."""
        future_time = datetime.utcnow() + timedelta(hours=1)
        user = User(
            user_id="user123",
            username="testuser",
            locked_until=future_time
        )
        assert user.is_locked()

    def test_user_lock_expired(self):
        """Test user lock has expired."""
        past_time = datetime.utcnow() - timedelta(hours=1)
        user = User(
            user_id="user123",
            username="testuser",
            locked_until=past_time
        )
        assert not user.is_locked()


class TestSession:
    """Test Session data class functionality."""

    def test_session_creation(self):
        """Test basic session creation."""
        expires_at = datetime.utcnow() + timedelta(hours=1)
        permissions = {Permission.READ_COLLECTION}
        session = Session(
            session_id="sess123",
            user_id="user123",
            expires_at=expires_at,
            ip_address="192.168.1.1",
            permissions=permissions,
            elevated=True
        )

        assert session.session_id == "sess123"
        assert session.user_id == "user123"
        assert session.expires_at == expires_at
        assert session.ip_address == "192.168.1.1"
        assert session.permissions == permissions
        assert session.elevated is True
        assert isinstance(session.created_at, datetime)

    def test_session_is_expired_false(self):
        """Test session is not expired."""
        future_time = datetime.utcnow() + timedelta(hours=1)
        session = Session(
            session_id="sess123",
            user_id="user123",
            expires_at=future_time
        )
        assert not session.is_expired()

    def test_session_is_expired_true(self):
        """Test session is expired."""
        past_time = datetime.utcnow() - timedelta(hours=1)
        session = Session(
            session_id="sess123",
            user_id="user123",
            expires_at=past_time
        )
        assert session.is_expired()

    def test_session_no_expiration(self):
        """Test session with no expiration time."""
        session = Session(session_id="sess123", user_id="user123")
        assert not session.is_expired()

    def test_session_is_active_true(self):
        """Test session is active."""
        session = Session(session_id="sess123", user_id="user123")
        session.last_activity = datetime.utcnow() - timedelta(minutes=30)
        assert session.is_active(timeout_seconds=3600)

    def test_session_is_active_false_timeout(self):
        """Test session is inactive due to timeout."""
        session = Session(session_id="sess123", user_id="user123")
        session.last_activity = datetime.utcnow() - timedelta(hours=2)
        assert not session.is_active(timeout_seconds=3600)

    def test_session_is_active_false_expired(self):
        """Test session is inactive due to expiration."""
        past_time = datetime.utcnow() - timedelta(hours=1)
        session = Session(
            session_id="sess123",
            user_id="user123",
            expires_at=past_time
        )
        assert not session.is_active()


class TestAccessContext:
    """Test AccessContext data class functionality."""

    def test_context_creation(self):
        """Test access context creation."""
        context = AccessContext(
            user_id="user123",
            session_id="sess123",
            ip_address="192.168.1.1",
            resource_type="collection",
            resource_id="coll123",
            operation="read"
        )

        assert context.user_id == "user123"
        assert context.session_id == "sess123"
        assert context.ip_address == "192.168.1.1"
        assert context.resource_type == "collection"
        assert context.resource_id == "coll123"
        assert context.operation == "read"
        assert isinstance(context.timestamp, datetime)

    def test_context_defaults(self):
        """Test access context with default values."""
        context = AccessContext()

        assert context.user_id is None
        assert context.session_id is None
        assert context.ip_address is None
        assert context.resource_type is None
        assert context.resource_id is None
        assert context.operation is None
        assert context.metadata == {}


class TestRoleBasedAccessControl:
    """Test RBAC system functionality."""

    @pytest.fixture
    def rbac(self):
        """Create RBAC instance for testing."""
        audit_logger = Mock()
        audit_logger.log_event = Mock()
        return RoleBasedAccessControl(audit_logger=audit_logger)

    @pytest.fixture
    def rbac_no_audit(self):
        """Create RBAC instance without audit logging."""
        return RoleBasedAccessControl()

    def test_rbac_initialization(self, rbac):
        """Test RBAC system initialization."""
        # Default roles should be created
        assert "admin" in rbac._roles
        assert "user" in rbac._roles
        assert "reader" in rbac._roles
        assert "writer" in rbac._roles

        # Admin should have all permissions
        admin_role = rbac._roles["admin"]
        assert len(admin_role.permissions) == len(Permission)
        assert admin_role.require_mfa is True

        # User role should have basic permissions
        user_role = rbac._roles["user"]
        expected_permissions = {
            Permission.READ_COLLECTION, Permission.READ_DOCUMENT,
            Permission.SEARCH_DOCUMENT, Permission.API_READ
        }
        assert user_role.permissions == expected_permissions

    def test_create_role_success(self, rbac):
        """Test successful role creation."""
        permissions = {Permission.CREATE_COLLECTION, Permission.DELETE_COLLECTION}
        result = rbac.create_role(
            "collection_manager",
            permissions=permissions,
            require_mfa=True
        )

        assert result is True
        assert "collection_manager" in rbac._roles
        role = rbac._roles["collection_manager"]
        assert role.permissions == permissions
        assert role.require_mfa is True

        # Verify audit logging
        rbac._audit_logger.log_event.assert_called_with(
            'role_created',
            {
                'role_name': 'collection_manager',
                'permissions': [p.value for p in permissions]
            },
            user_id='system'
        )

    def test_create_role_duplicate(self, rbac):
        """Test creating duplicate role."""
        result = rbac.create_role("admin")  # admin already exists
        assert result is False

    def test_create_role_invalid_parent(self, rbac):
        """Test creating role with invalid parent."""
        result = rbac.create_role(
            "invalid_role",
            parent_roles={"nonexistent_parent"}
        )
        assert result is False

    def test_create_role_with_exception(self, rbac):
        """Test role creation with exception."""
        with patch.object(rbac, '_clear_permission_cache', side_effect=Exception("Test error")):
            result = rbac.create_role("error_role")
            assert result is False

    def test_delete_role_success(self, rbac):
        """Test successful role deletion."""
        # Create role first
        rbac.create_role("temp_role")
        assert "temp_role" in rbac._roles

        result = rbac.delete_role("temp_role")
        assert result is True
        assert "temp_role" not in rbac._roles

    def test_delete_role_nonexistent(self, rbac):
        """Test deleting nonexistent role."""
        result = rbac.delete_role("nonexistent")
        assert result is False

    def test_delete_role_in_use(self, rbac):
        """Test deleting role that's assigned to users."""
        # Create role and user
        rbac.create_role("test_role")
        rbac.create_user("user123", "testuser", roles={"test_role"})

        result = rbac.delete_role("test_role")
        assert result is False
        assert "test_role" in rbac._roles

    def test_delete_role_with_exception(self, rbac):
        """Test role deletion with exception."""
        rbac.create_role("error_role")
        with patch.object(rbac, '_clear_permission_cache', side_effect=Exception("Test error")):
            result = rbac.delete_role("error_role")
            assert result is False

    def test_create_user_success(self, rbac):
        """Test successful user creation."""
        roles = {"admin", "user"}
        result = rbac.create_user(
            "user123",
            "testuser",
            roles=roles,
            mfa_enabled=True
        )

        assert result is True
        assert "user123" in rbac._users
        user = rbac._users["user123"]
        assert user.username == "testuser"
        assert user.roles == roles
        assert user.mfa_enabled is True

    def test_create_user_duplicate(self, rbac):
        """Test creating duplicate user."""
        rbac.create_user("user123", "testuser")
        result = rbac.create_user("user123", "duplicate")
        assert result is False

    def test_create_user_invalid_role(self, rbac):
        """Test creating user with invalid role."""
        result = rbac.create_user(
            "user123",
            "testuser",
            roles={"nonexistent_role"}
        )
        assert result is False

    def test_create_user_with_exception(self, rbac):
        """Test user creation with exception."""
        with patch.object(rbac, '_clear_permission_cache', side_effect=Exception("Test error")):
            result = rbac.create_user("error_user", "testuser")
            assert result is False

    def test_delete_user_success(self, rbac):
        """Test successful user deletion."""
        # Create user and session
        rbac.create_user("user123", "testuser")
        session_id = rbac.create_session("user123")
        assert session_id is not None

        result = rbac.delete_user("user123")
        assert result is True
        assert "user123" not in rbac._users
        assert session_id not in rbac._sessions

    def test_delete_user_nonexistent(self, rbac):
        """Test deleting nonexistent user."""
        result = rbac.delete_user("nonexistent")
        assert result is False

    def test_delete_user_with_exception(self, rbac):
        """Test user deletion with exception."""
        rbac.create_user("error_user", "testuser")
        with patch.object(rbac, '_clear_permission_cache', side_effect=Exception("Test error")):
            result = rbac.delete_user("error_user")
            assert result is False

    def test_assign_role_success(self, rbac):
        """Test successful role assignment."""
        rbac.create_user("user123", "testuser")
        result = rbac.assign_role("user123", "admin")

        assert result is True
        user = rbac._users["user123"]
        assert "admin" in user.roles

    def test_assign_role_already_assigned(self, rbac):
        """Test assigning role that's already assigned."""
        rbac.create_user("user123", "testuser", roles={"admin"})
        result = rbac.assign_role("user123", "admin")
        assert result is True  # Should succeed but no change

    def test_assign_role_invalid_user(self, rbac):
        """Test assigning role to invalid user."""
        result = rbac.assign_role("nonexistent", "admin")
        assert result is False

    def test_assign_role_invalid_role(self, rbac):
        """Test assigning invalid role."""
        rbac.create_user("user123", "testuser")
        result = rbac.assign_role("user123", "nonexistent_role")
        assert result is False

    def test_assign_role_with_exception(self, rbac):
        """Test role assignment with exception."""
        rbac.create_user("user123", "testuser")
        with patch.object(rbac, '_clear_permission_cache', side_effect=Exception("Test error")):
            result = rbac.assign_role("user123", "admin")
            assert result is False

    def test_revoke_role_success(self, rbac):
        """Test successful role revocation."""
        rbac.create_user("user123", "testuser", roles={"admin", "user"})
        session_id = rbac.create_session("user123")

        result = rbac.revoke_role("user123", "admin")
        assert result is True

        user = rbac._users["user123"]
        assert "admin" not in user.roles
        assert "user" in user.roles

        # Session permissions should be updated
        session = rbac._sessions[session_id]
        admin_permissions = rbac._roles["admin"].permissions
        assert not admin_permissions.issubset(session.permissions)

    def test_revoke_role_not_assigned(self, rbac):
        """Test revoking role that's not assigned."""
        rbac.create_user("user123", "testuser", roles={"user"})
        result = rbac.revoke_role("user123", "admin")
        assert result is True  # Should succeed but no change

    def test_revoke_role_invalid_user(self, rbac):
        """Test revoking role from invalid user."""
        result = rbac.revoke_role("nonexistent", "admin")
        assert result is False

    def test_revoke_role_with_exception(self, rbac):
        """Test role revocation with exception."""
        rbac.create_user("user123", "testuser", roles={"admin"})
        with patch.object(rbac, '_clear_permission_cache', side_effect=Exception("Test error")):
            result = rbac.revoke_role("user123", "admin")
            assert result is False

    def test_create_session_success(self, rbac):
        """Test successful session creation."""
        rbac.create_user("user123", "testuser", roles={"admin"})
        session_id = rbac.create_session(
            "user123",
            ip_address="192.168.1.1",
            user_agent="test-agent",
            timeout_seconds=1800
        )

        assert session_id is not None
        assert session_id in rbac._sessions

        session = rbac._sessions[session_id]
        assert session.user_id == "user123"
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "test-agent"
        assert session.expires_at is not None

        user = rbac._users["user123"]
        assert session_id in user.active_sessions
        assert user.last_login is not None
        assert user.failed_attempts == 0

    def test_create_session_invalid_user(self, rbac):
        """Test creating session for invalid user."""
        session_id = rbac.create_session("nonexistent")
        assert session_id is None

    def test_create_session_locked_user(self, rbac):
        """Test creating session for locked user."""
        rbac.create_user("user123", "testuser")
        user = rbac._users["user123"]
        user.locked_until = datetime.utcnow() + timedelta(hours=1)

        session_id = rbac.create_session("user123")
        assert session_id is None

    def test_create_session_rate_limited(self, rbac):
        """Test session creation with rate limiting."""
        rbac.create_user("user123", "testuser")

        # Mock rate limit check to return False
        with patch.object(rbac, '_check_rate_limit', return_value=False):
            session_id = rbac.create_session("user123")
            assert session_id is None

    def test_create_session_max_sessions(self, rbac):
        """Test session creation with session limit."""
        # Create custom role with low session limit
        rbac.create_role("limited", max_sessions=2)
        rbac.create_user("user123", "testuser", roles={"limited"})

        # Create maximum sessions
        session1 = rbac.create_session("user123")
        session2 = rbac.create_session("user123")
        assert session1 is not None
        assert session2 is not None

        # Third session should remove oldest
        session3 = rbac.create_session("user123")
        assert session3 is not None
        assert session1 not in rbac._sessions  # Oldest removed
        assert session2 in rbac._sessions
        assert session3 in rbac._sessions

    def test_create_session_with_exception(self, rbac):
        """Test session creation with exception."""
        rbac.create_user("user123", "testuser")
        with patch('secrets.token_urlsafe', side_effect=Exception("Token error")):
            session_id = rbac.create_session("user123")
            assert session_id is None

    def test_terminate_session_success(self, rbac):
        """Test successful session termination."""
        rbac.create_user("user123", "testuser")
        session_id = rbac.create_session("user123")

        result = rbac.terminate_session(session_id)
        assert result is True
        assert session_id not in rbac._sessions

        user = rbac._users["user123"]
        assert session_id not in user.active_sessions

    def test_terminate_session_nonexistent(self, rbac):
        """Test terminating nonexistent session."""
        result = rbac.terminate_session("nonexistent")
        assert result is False

    def test_terminate_session_with_exception(self, rbac):
        """Test session termination with exception."""
        rbac.create_user("user123", "testuser")
        session_id = rbac.create_session("user123")

        with patch.object(rbac._audit_logger, 'log_event', side_effect=Exception("Audit error")):
            result = rbac.terminate_session(session_id)
            assert result is False

    def test_check_access_granted(self, rbac):
        """Test access check with granted result."""
        rbac.create_user("user123", "testuser", roles={"admin"})
        session_id = rbac.create_session("user123")

        result = rbac.check_access(session_id, Permission.CREATE_COLLECTION)
        assert result == AccessResult.GRANTED

        # Verify last activity was updated
        session = rbac._sessions[session_id]
        assert session.last_activity > session.created_at

    def test_check_access_invalid_session(self, rbac):
        """Test access check with invalid session."""
        result = rbac.check_access("nonexistent", Permission.READ_COLLECTION)
        assert result == AccessResult.DENIED

    def test_check_access_expired_session(self, rbac):
        """Test access check with expired session."""
        rbac.create_user("user123", "testuser", roles={"user"})
        session_id = rbac.create_session("user123")

        # Make session expired
        session = rbac._sessions[session_id]
        session.expires_at = datetime.utcnow() - timedelta(hours=1)

        result = rbac.check_access(session_id, Permission.READ_COLLECTION)
        assert result == AccessResult.SESSION_EXPIRED
        assert session_id not in rbac._sessions  # Should be terminated

    def test_check_access_locked_user(self, rbac):
        """Test access check with locked user."""
        rbac.create_user("user123", "testuser", roles={"user"})
        session_id = rbac.create_session("user123")

        # Lock user
        user = rbac._users["user123"]
        user.locked_until = datetime.utcnow() + timedelta(hours=1)

        result = rbac.check_access(session_id, Permission.READ_COLLECTION)
        assert result == AccessResult.DENIED

    def test_check_access_rate_limited(self, rbac):
        """Test access check with rate limiting."""
        rbac.create_user("user123", "testuser", roles={"user"})
        session_id = rbac.create_session("user123")

        with patch.object(rbac, '_check_rate_limit', return_value=False):
            result = rbac.check_access(session_id, Permission.READ_COLLECTION)
            assert result == AccessResult.RATE_LIMITED

    def test_check_access_permission_denied(self, rbac):
        """Test access check with insufficient permissions."""
        rbac.create_user("user123", "testuser", roles={"reader"})
        session_id = rbac.create_session("user123")

        result = rbac.check_access(session_id, Permission.CREATE_COLLECTION)
        assert result == AccessResult.DENIED

        # Verify access denial was logged
        rbac._audit_logger.log_event.assert_called()
        calls = rbac._audit_logger.log_event.call_args_list
        access_denied_calls = [call for call in calls if call[0][0] == 'access_denied']
        assert len(access_denied_calls) > 0

    def test_check_access_requires_mfa(self, rbac):
        """Test access check requiring MFA."""
        # Create role requiring MFA
        rbac.create_role("secure_role", permissions={Permission.ADMIN_ACCESS}, require_mfa=True)
        rbac.create_user("user123", "testuser", roles={"secure_role"})
        session_id = rbac.create_session("user123")

        result = rbac.check_access(session_id, Permission.ADMIN_ACCESS)
        assert result == AccessResult.REQUIRES_MFA

    def test_check_access_ip_restrictions(self, rbac):
        """Test access check with IP restrictions."""
        # Create role with IP whitelist
        rbac.create_role(
            "restricted_role",
            permissions={Permission.READ_COLLECTION},
            ip_whitelist={"192.168.1.100"}
        )
        rbac.create_user("user123", "testuser", roles={"restricted_role"})
        session_id = rbac.create_session("user123", ip_address="192.168.1.1")

        result = rbac.check_access(session_id, Permission.READ_COLLECTION)
        assert result == AccessResult.DENIED

    def test_check_access_time_restrictions(self, rbac):
        """Test access check with time restrictions."""
        # Create role with time restrictions (only allow hour 0)
        rbac.create_role(
            "time_restricted",
            permissions={Permission.READ_COLLECTION},
            time_restrictions={'hours': [0]}
        )
        rbac.create_user("user123", "testuser", roles={"time_restricted"})
        session_id = rbac.create_session("user123")

        # Mock current hour to be outside allowed range
        with patch('src.python.common.security.access_control.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 12, 0, 0)  # Hour 12
            result = rbac.check_access(session_id, Permission.READ_COLLECTION)
            assert result == AccessResult.DENIED

    def test_check_access_with_exception(self, rbac):
        """Test access check with exception."""
        rbac.create_user("user123", "testuser", roles={"user"})
        session_id = rbac.create_session("user123")

        with patch.object(rbac, '_check_rate_limit', side_effect=Exception("Rate limit error")):
            result = rbac.check_access(session_id, Permission.READ_COLLECTION)
            assert result == AccessResult.DENIED

    def test_calculate_user_permissions_basic(self, rbac):
        """Test basic permission calculation."""
        rbac.create_user("user123", "testuser", roles={"user"})
        permissions = rbac._calculate_user_permissions("user123")

        expected = {
            Permission.READ_COLLECTION, Permission.READ_DOCUMENT,
            Permission.SEARCH_DOCUMENT, Permission.API_READ
        }
        assert permissions == expected

    def test_calculate_user_permissions_multiple_roles(self, rbac):
        """Test permission calculation with multiple roles."""
        rbac.create_user("user123", "testuser", roles={"reader", "writer"})
        permissions = rbac._calculate_user_permissions("user123")

        reader_permissions = rbac._roles["reader"].permissions
        writer_permissions = rbac._roles["writer"].permissions
        expected = reader_permissions | writer_permissions

        assert permissions == expected

    def test_calculate_user_permissions_hierarchy(self, rbac):
        """Test permission calculation with role hierarchy."""
        # Create parent and child roles
        rbac.create_role("parent", permissions={Permission.READ_COLLECTION})
        rbac.create_role("child", permissions={Permission.WRITE_COLLECTION}, parent_roles={"parent"})
        rbac.create_user("user123", "testuser", roles={"child"})

        permissions = rbac._calculate_user_permissions("user123")
        expected = {Permission.READ_COLLECTION, Permission.WRITE_COLLECTION}

        assert permissions == expected

    def test_calculate_user_permissions_complex_hierarchy(self, rbac):
        """Test permission calculation with complex hierarchy."""
        # Create multi-level hierarchy
        rbac.create_role("grandparent", permissions={Permission.READ_DOCUMENT})
        rbac.create_role("parent", permissions={Permission.WRITE_DOCUMENT}, parent_roles={"grandparent"})
        rbac.create_role("child", permissions={Permission.DELETE_DOCUMENT}, parent_roles={"parent"})
        rbac.create_user("user123", "testuser", roles={"child"})

        permissions = rbac._calculate_user_permissions("user123")
        expected = {Permission.READ_DOCUMENT, Permission.WRITE_DOCUMENT, Permission.DELETE_DOCUMENT}

        assert permissions == expected

    def test_calculate_user_permissions_circular_prevention(self, rbac):
        """Test permission calculation prevents circular references."""
        # Create potentially circular roles (though this shouldn't be allowed in real use)
        rbac.create_role("role_a", permissions={Permission.READ_COLLECTION})
        rbac.create_role("role_b", permissions={Permission.WRITE_COLLECTION})

        # Manually add circular reference for testing
        rbac._roles["role_a"].parent_roles.add("role_b")
        rbac._roles["role_b"].parent_roles.add("role_a")

        rbac.create_user("user123", "testuser", roles={"role_a"})

        permissions = rbac._calculate_user_permissions("user123")
        expected = {Permission.READ_COLLECTION, Permission.WRITE_COLLECTION}

        assert permissions == expected

    def test_calculate_user_permissions_cache(self, rbac):
        """Test permission calculation caching."""
        rbac.create_user("user123", "testuser", roles={"user"})

        # First call should populate cache
        permissions1 = rbac._calculate_user_permissions("user123")

        # Mock time to ensure cache is valid
        with patch('time.time', return_value=100):
            rbac._permission_cache[("user123", "permissions")] = (permissions1, 99)
            permissions2 = rbac._calculate_user_permissions("user123")

        assert permissions1 == permissions2

    def test_calculate_user_permissions_nonexistent_user(self, rbac):
        """Test permission calculation for nonexistent user."""
        permissions = rbac._calculate_user_permissions("nonexistent")
        assert permissions == set()

    def test_check_rate_limit_allowed(self, rbac):
        """Test rate limiting allows requests."""
        result = rbac._check_rate_limit("user123")
        assert result is True

        # User should have one request recorded
        assert len(rbac._rate_limits["user123"]) == 1

    def test_check_rate_limit_exceeded(self, rbac):
        """Test rate limiting blocks excess requests."""
        user_id = "user123"
        now = time.time()

        # Fill up rate limit
        rbac._rate_limits[user_id] = [now - i for i in range(rbac._rate_limit_max)]

        result = rbac._check_rate_limit(user_id)
        assert result is False

    def test_check_rate_limit_window_cleanup(self, rbac):
        """Test rate limiting cleans up old requests."""
        user_id = "user123"
        now = time.time()
        old_time = now - rbac._rate_limit_window - 100

        # Add old requests
        rbac._rate_limits[user_id] = [old_time, old_time - 10, old_time - 20]

        result = rbac._check_rate_limit(user_id)
        assert result is True

        # Old requests should be cleaned up
        assert len(rbac._rate_limits[user_id]) == 1  # Only the new request

    def test_check_ip_restrictions_no_restrictions(self, rbac):
        """Test IP restrictions with no restrictions."""
        rbac.create_user("user123", "testuser", roles={"user"})
        user = rbac._users["user123"]

        result = rbac._check_ip_restrictions(user, "192.168.1.1")
        assert result is True

    def test_check_ip_restrictions_allowed(self, rbac):
        """Test IP restrictions with allowed IP."""
        rbac.create_role("restricted", ip_whitelist={"192.168.1.1", "10.0.0.1"})
        rbac.create_user("user123", "testuser", roles={"restricted"})
        user = rbac._users["user123"]

        result = rbac._check_ip_restrictions(user, "192.168.1.1")
        assert result is True

    def test_check_ip_restrictions_blocked(self, rbac):
        """Test IP restrictions with blocked IP."""
        rbac.create_role("restricted", ip_whitelist={"192.168.1.1"})
        rbac.create_user("user123", "testuser", roles={"restricted"})
        user = rbac._users["user123"]

        result = rbac._check_ip_restrictions(user, "192.168.1.100")
        assert result is False

    def test_check_ip_restrictions_no_ip(self, rbac):
        """Test IP restrictions with no IP address."""
        rbac.create_role("restricted", ip_whitelist={"192.168.1.1"})
        rbac.create_user("user123", "testuser", roles={"restricted"})
        user = rbac._users["user123"]

        result = rbac._check_ip_restrictions(user, None)
        assert result is True  # No IP to check

    def test_check_time_restrictions_no_restrictions(self, rbac):
        """Test time restrictions with no restrictions."""
        rbac.create_user("user123", "testuser", roles={"user"})
        user = rbac._users["user123"]

        result = rbac._check_time_restrictions(user)
        assert result is True

    def test_check_time_restrictions_hour_allowed(self, rbac):
        """Test time restrictions with allowed hour."""
        rbac.create_role("time_restricted", time_restrictions={'hours': [9, 10, 11]})
        rbac.create_user("user123", "testuser", roles={"time_restricted"})
        user = rbac._users["user123"]

        with patch('src.python.common.security.access_control.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 10, 0, 0)  # Hour 10
            result = rbac._check_time_restrictions(user)
            assert result is True

    def test_check_time_restrictions_hour_blocked(self, rbac):
        """Test time restrictions with blocked hour."""
        rbac.create_role("time_restricted", time_restrictions={'hours': [9, 10, 11]})
        rbac.create_user("user123", "testuser", roles={"time_restricted"})
        user = rbac._users["user123"]

        with patch('src.python.common.security.access_control.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 15, 0, 0)  # Hour 15
            result = rbac._check_time_restrictions(user)
            assert result is False

    def test_check_time_restrictions_day_allowed(self, rbac):
        """Test time restrictions with allowed day."""
        rbac.create_role("time_restricted", time_restrictions={'days': [0, 1, 2, 3, 4]})  # Weekdays
        rbac.create_user("user123", "testuser", roles={"time_restricted"})
        user = rbac._users["user123"]

        with patch('src.python.common.security.access_control.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 2, 10, 0, 0)  # Monday
            result = rbac._check_time_restrictions(user)
            assert result is True

    def test_check_time_restrictions_day_blocked(self, rbac):
        """Test time restrictions with blocked day."""
        rbac.create_role("time_restricted", time_restrictions={'days': [0, 1, 2, 3, 4]})  # Weekdays
        rbac.create_user("user123", "testuser", roles={"time_restricted"})
        user = rbac._users["user123"]

        with patch('src.python.common.security.access_control.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 7, 10, 0, 0)  # Saturday
            result = rbac._check_time_restrictions(user)
            assert result is False

    def test_clear_permission_cache_user(self, rbac):
        """Test clearing permission cache for specific user."""
        rbac.create_user("user123", "testuser")
        rbac.create_user("user456", "testuser2")

        # Populate cache
        rbac._calculate_user_permissions("user123")
        rbac._calculate_user_permissions("user456")

        assert len(rbac._permission_cache) == 2

        # Clear cache for specific user
        rbac._clear_permission_cache("user123")

        # Only user456 cache should remain
        remaining_keys = [key for key in rbac._permission_cache.keys() if key[0] == "user456"]
        assert len(remaining_keys) == 1

    def test_clear_permission_cache_all(self, rbac):
        """Test clearing entire permission cache."""
        rbac.create_user("user123", "testuser")
        rbac.create_user("user456", "testuser2")

        # Populate cache
        rbac._calculate_user_permissions("user123")
        rbac._calculate_user_permissions("user456")

        assert len(rbac._permission_cache) == 2

        # Clear entire cache
        rbac._clear_permission_cache()

        assert len(rbac._permission_cache) == 0

    def test_get_session_info_valid(self, rbac):
        """Test getting session information for valid session."""
        rbac.create_user("user123", "testuser", roles={"admin"})
        session_id = rbac.create_session("user123", ip_address="192.168.1.1")

        info = rbac.get_session_info(session_id)
        assert info is not None
        assert info['session_id'] == session_id
        assert info['user_id'] == "user123"
        assert info['username'] == "testuser"
        assert info['ip_address'] == "192.168.1.1"
        assert 'permissions' in info
        assert 'created_at' in info
        assert 'expires_at' in info

    def test_get_session_info_invalid(self, rbac):
        """Test getting session information for invalid session."""
        info = rbac.get_session_info("nonexistent")
        assert info is None

    def test_get_session_info_invalid_user(self, rbac):
        """Test getting session info when user is deleted."""
        rbac.create_user("user123", "testuser")
        session_id = rbac.create_session("user123")

        # Delete user but keep session
        del rbac._users["user123"]

        info = rbac.get_session_info(session_id)
        assert info is None

    def test_list_active_sessions_all(self, rbac):
        """Test listing all active sessions."""
        rbac.create_user("user123", "testuser")
        rbac.create_user("user456", "testuser2")

        session1 = rbac.create_session("user123")
        session2 = rbac.create_session("user456")

        sessions = rbac.list_active_sessions()
        assert len(sessions) == 2

        session_ids = [s['session_id'] for s in sessions]
        assert session1 in session_ids
        assert session2 in session_ids

    def test_list_active_sessions_user_specific(self, rbac):
        """Test listing active sessions for specific user."""
        rbac.create_user("user123", "testuser")
        rbac.create_user("user456", "testuser2")

        session1 = rbac.create_session("user123")
        session2 = rbac.create_session("user456")

        sessions = rbac.list_active_sessions("user123")
        assert len(sessions) == 1
        assert sessions[0]['session_id'] == session1
        assert sessions[0]['user_id'] == "user123"

    def test_list_active_sessions_inactive_filtered(self, rbac):
        """Test listing active sessions filters inactive sessions."""
        rbac.create_user("user123", "testuser")
        session_id = rbac.create_session("user123")

        # Make session inactive
        session = rbac._sessions[session_id]
        session.last_activity = datetime.utcnow() - timedelta(hours=2)

        sessions = rbac.list_active_sessions()
        assert len(sessions) == 0

    def test_cleanup_expired_sessions(self, rbac):
        """Test cleaning up expired sessions."""
        rbac.create_user("user123", "testuser")
        rbac.create_user("user456", "testuser2")

        session1 = rbac.create_session("user123")
        session2 = rbac.create_session("user456")

        # Make one session expired
        rbac._sessions[session1].expires_at = datetime.utcnow() - timedelta(hours=1)

        count = rbac.cleanup_expired_sessions()
        assert count == 1
        assert session1 not in rbac._sessions
        assert session2 in rbac._sessions

    def test_cleanup_expired_sessions_inactive(self, rbac):
        """Test cleaning up inactive sessions."""
        rbac.create_user("user123", "testuser")
        session_id = rbac.create_session("user123")

        # Make session inactive
        session = rbac._sessions[session_id]
        session.last_activity = datetime.utcnow() - timedelta(hours=2)

        count = rbac.cleanup_expired_sessions()
        assert count == 1
        assert session_id not in rbac._sessions


class TestSessionManager:
    """Test SessionManager functionality."""

    @pytest.fixture
    def rbac(self):
        """Create RBAC instance for testing."""
        return RoleBasedAccessControl()

    @pytest.fixture
    def session_manager(self, rbac):
        """Create SessionManager for testing."""
        return SessionManager(rbac, cleanup_interval=1)

    @pytest.mark.asyncio
    async def test_session_manager_initialization(self, session_manager):
        """Test session manager initialization."""
        assert session_manager._cleanup_interval == 1
        assert session_manager._running is False
        assert session_manager._cleanup_task is None

    @pytest.mark.asyncio
    async def test_start_cleanup_task(self, session_manager):
        """Test starting cleanup task."""
        await session_manager.start_cleanup_task()

        assert session_manager._running is True
        assert session_manager._cleanup_task is not None
        assert not session_manager._cleanup_task.done()

        # Stop task
        await session_manager.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_start_cleanup_task_already_running(self, session_manager):
        """Test starting cleanup task when already running."""
        await session_manager.start_cleanup_task()
        first_task = session_manager._cleanup_task

        await session_manager.start_cleanup_task()
        second_task = session_manager._cleanup_task

        assert first_task == second_task

        # Stop task
        await session_manager.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_stop_cleanup_task(self, session_manager):
        """Test stopping cleanup task."""
        await session_manager.start_cleanup_task()
        assert session_manager._running is True

        await session_manager.stop_cleanup_task()
        assert session_manager._running is False
        assert session_manager._cleanup_task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_cleanup_task_not_running(self, session_manager):
        """Test stopping cleanup task when not running."""
        # Should not raise exception
        await session_manager.stop_cleanup_task()
        assert session_manager._running is False

    @pytest.mark.asyncio
    async def test_cleanup_loop_calls_cleanup(self, session_manager):
        """Test cleanup loop calls RBAC cleanup."""
        # Mock cleanup method
        cleanup_called = False

        def mock_cleanup():
            nonlocal cleanup_called
            cleanup_called = True
            session_manager._running = False  # Stop after one iteration

        session_manager._rbac.cleanup_expired_sessions = mock_cleanup

        # Start and wait briefly
        await session_manager.start_cleanup_task()
        await asyncio.sleep(0.1)  # Give it time to run once

        await session_manager.stop_cleanup_task()
        assert cleanup_called

    @pytest.mark.asyncio
    async def test_cleanup_loop_handles_exception(self, session_manager):
        """Test cleanup loop handles exceptions."""
        # Mock cleanup method to raise exception
        def mock_cleanup():
            session_manager._running = False  # Stop after one iteration
            raise Exception("Test exception")

        session_manager._rbac.cleanup_expired_sessions = mock_cleanup

        # Start and wait briefly
        await session_manager.start_cleanup_task()
        await asyncio.sleep(0.1)

        # Should not crash
        await session_manager.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_session_context_valid(self, rbac, session_manager):
        """Test session context with valid session."""
        rbac.create_user("user123", "testuser")
        session_id = rbac.create_session("user123")

        async with session_manager.session_context(session_id) as session_info:
            assert session_info is not None
            assert session_info['session_id'] == session_id
            assert session_info['user_id'] == "user123"

        # Verify last activity was updated
        session = rbac._sessions[session_id]
        assert session.last_activity > session.created_at

    @pytest.mark.asyncio
    async def test_session_context_invalid(self, session_manager):
        """Test session context with invalid session."""
        with pytest.raises(ValueError, match="Invalid session"):
            async with session_manager.session_context("nonexistent"):
                pass

    @pytest.mark.asyncio
    async def test_session_context_exception_handling(self, rbac, session_manager):
        """Test session context handles exceptions properly."""
        rbac.create_user("user123", "testuser")
        session_id = rbac.create_session("user123")

        try:
            async with session_manager.session_context(session_id):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Session should still exist and be updated
        session = rbac._sessions[session_id]
        assert session.last_activity > session.created_at


class TestGlobalFunctions:
    """Test global functions and utilities."""

    def test_get_rbac_singleton(self):
        """Test get_rbac returns singleton instance."""
        rbac1 = get_rbac()
        rbac2 = get_rbac()

        assert rbac1 is rbac2
        assert isinstance(rbac1, RoleBasedAccessControl)

    @pytest.mark.asyncio
    async def test_require_permission_decorator_async_granted(self):
        """Test require_permission decorator with async function - access granted."""
        rbac = get_rbac()
        rbac.create_user("user123", "testuser", roles={"admin"})
        session_id = rbac.create_session("user123")

        @require_permission(Permission.CREATE_COLLECTION)
        async def protected_function(session_id):
            return "success"

        result = await protected_function(session_id)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_require_permission_decorator_async_denied(self):
        """Test require_permission decorator with async function - access denied."""
        rbac = get_rbac()
        rbac.create_user("user123", "testuser", roles={"reader"})
        session_id = rbac.create_session("user123")

        @require_permission(Permission.CREATE_COLLECTION)
        async def protected_function(session_id):
            return "success"

        with pytest.raises(PermissionError, match="Access denied"):
            await protected_function(session_id)

    def test_require_permission_decorator_sync_granted(self):
        """Test require_permission decorator with sync function - access granted."""
        rbac = get_rbac()
        rbac.create_user("user123", "testuser", roles={"admin"})
        session_id = rbac.create_session("user123")

        @require_permission(Permission.CREATE_COLLECTION)
        def protected_function(session_id):
            return "success"

        result = protected_function(session_id)
        assert result == "success"

    def test_require_permission_decorator_sync_denied(self):
        """Test require_permission decorator with sync function - access denied."""
        rbac = get_rbac()
        rbac.create_user("user123", "testuser", roles={"reader"})
        session_id = rbac.create_session("user123")

        @require_permission(Permission.CREATE_COLLECTION)
        def protected_function(session_id):
            return "success"

        with pytest.raises(PermissionError, match="Access denied"):
            protected_function(session_id)

    @pytest.mark.asyncio
    async def test_require_permission_decorator_no_session(self):
        """Test require_permission decorator with missing session."""
        @require_permission(Permission.CREATE_COLLECTION)
        async def protected_function():
            return "success"

        with pytest.raises(ValueError, match="Session ID required"):
            await protected_function()

    def test_require_permission_decorator_kwargs_session(self):
        """Test require_permission decorator with session in kwargs."""
        rbac = get_rbac()
        rbac.create_user("user123", "testuser", roles={"admin"})
        session_id = rbac.create_session("user123")

        @require_permission(Permission.CREATE_COLLECTION)
        def protected_function(other_param, session_id=None):
            return "success"

        result = protected_function("test", session_id=session_id)
        assert result == "success"


class TestEdgeCasesAndErrorConditions:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def rbac(self):
        """Create RBAC instance for testing."""
        return RoleBasedAccessControl()

    def test_role_with_empty_permissions(self, rbac):
        """Test role with empty permissions set."""
        rbac.create_role("empty_role", permissions=set())
        rbac.create_user("user123", "testuser", roles={"empty_role"})

        permissions = rbac._calculate_user_permissions("user123")
        assert permissions == set()

    def test_user_with_no_roles(self, rbac):
        """Test user with no roles assigned."""
        rbac.create_user("user123", "testuser", roles=set())
        permissions = rbac._calculate_user_permissions("user123")
        assert permissions == set()

    def test_session_with_malformed_data(self, rbac):
        """Test session with malformed data."""
        rbac.create_user("user123", "testuser", roles={"user"})
        session_id = rbac.create_session("user123")

        # Corrupt session data
        session = rbac._sessions[session_id]
        session.user_id = None

        result = rbac.check_access(session_id, Permission.READ_COLLECTION)
        assert result == AccessResult.DENIED

    def test_concurrent_session_creation(self, rbac):
        """Test concurrent session creation."""
        import threading

        rbac.create_user("user123", "testuser", roles={"user"})
        results = []

        def create_session():
            session_id = rbac.create_session("user123")
            results.append(session_id)

        threads = [threading.Thread(target=create_session) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All sessions should be created successfully
        assert len([r for r in results if r is not None]) == 10

    def test_permission_cache_expiration(self, rbac):
        """Test permission cache expiration."""
        rbac.create_user("user123", "testuser", roles={"user"})

        # Get permissions (populate cache)
        permissions1 = rbac._calculate_user_permissions("user123")

        # Mock time to make cache expire
        with patch('time.time', return_value=time.time() + rbac._cache_ttl + 1):
            # Add role to user
            rbac.assign_role("user123", "admin")

            # Get permissions again (cache should be expired)
            permissions2 = rbac._calculate_user_permissions("user123")

        # Permissions should be different due to new role
        assert len(permissions2) > len(permissions1)

    def test_rate_limit_edge_cases(self, rbac):
        """Test rate limiting edge cases."""
        user_id = "user123"

        # Test with exactly at limit
        now = time.time()
        rbac._rate_limits[user_id] = [now - i for i in range(rbac._rate_limit_max - 1)]

        result = rbac._check_rate_limit(user_id)
        assert result is True

        # Now should be at limit
        result = rbac._check_rate_limit(user_id)
        assert result is False

    def test_session_expiration_edge_cases(self, rbac):
        """Test session expiration edge cases."""
        rbac.create_user("user123", "testuser", roles={"user"})
        session_id = rbac.create_session("user123")

        session = rbac._sessions[session_id]

        # Set expiration to exactly now
        session.expires_at = datetime.utcnow()
        time.sleep(0.001)  # Small delay to ensure expiration

        result = rbac.check_access(session_id, Permission.READ_COLLECTION)
        assert result == AccessResult.SESSION_EXPIRED

    def test_invalid_permission_enum(self, rbac):
        """Test with invalid permission enum."""
        rbac.create_user("user123", "testuser", roles={"admin"})
        session_id = rbac.create_session("user123")

        # This should work - just testing the type system
        result = rbac.check_access(session_id, Permission.ADMIN_ACCESS)
        assert result == AccessResult.GRANTED

    def test_memory_cleanup_on_user_deletion(self, rbac):
        """Test memory is properly cleaned up on user deletion."""
        rbac.create_user("user123", "testuser", roles={"user"})

        # Create some sessions and cache entries
        session1 = rbac.create_session("user123")
        session2 = rbac.create_session("user123")
        rbac._calculate_user_permissions("user123")

        # Verify data exists
        assert len(rbac._users["user123"].active_sessions) == 2
        assert any(key[0] == "user123" for key in rbac._permission_cache.keys())

        # Delete user
        rbac.delete_user("user123")

        # Verify cleanup
        assert "user123" not in rbac._users
        assert session1 not in rbac._sessions
        assert session2 not in rbac._sessions
        assert not any(key[0] == "user123" for key in rbac._permission_cache.keys())

    def test_large_role_hierarchy(self, rbac):
        """Test with large role hierarchy."""
        # Create a deep hierarchy
        for i in range(10):
            parent_roles = {f"role_{i-1}"} if i > 0 else set()
            rbac.create_role(f"role_{i}", permissions={list(Permission)[i % len(Permission)]}, parent_roles=parent_roles)

        rbac.create_user("user123", "testuser", roles={"role_9"})
        permissions = rbac._calculate_user_permissions("user123")

        # Should have permissions from all levels
        assert len(permissions) == 10

    def test_unicode_and_special_characters(self, rbac):
        """Test with unicode and special characters in names."""
        # Test with unicode characters
        rbac.create_role("роль_测试", permissions={Permission.READ_COLLECTION})
        rbac.create_user("用户123", "тестユーザー", roles={"роль_测试"})

        result = rbac.assign_role("用户123", "admin")
        assert result is True

        session_id = rbac.create_session("用户123")
        assert session_id is not None

        result = rbac.check_access(session_id, Permission.ADMIN_ACCESS)
        assert result == AccessResult.GRANTED

    def test_time_zone_handling(self, rbac):
        """Test time zone handling in time restrictions."""
        # Create role with specific hour restriction
        rbac.create_role("time_role", permissions={Permission.READ_COLLECTION}, time_restrictions={'hours': [12]})
        rbac.create_user("user123", "testuser", roles={"time_role"})

        # Mock different time zones (though we use UTC internally)
        with patch('src.python.common.security.access_control.datetime') as mock_datetime:
            # Test UTC hour 12
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 12, 0, 0)
            result = rbac._check_time_restrictions(rbac._users["user123"])
            assert result is True

            # Test UTC hour 13
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 13, 0, 0)
            result = rbac._check_time_restrictions(rbac._users["user123"])
            assert result is False

    def test_session_metadata_preservation(self, rbac):
        """Test session metadata is preserved correctly."""
        rbac.create_user("user123", "testuser", roles={"user"})
        session_id = rbac.create_session("user123", user_agent="TestAgent/1.0")

        session = rbac._sessions[session_id]
        session.metadata = {"custom_field": "custom_value", "number": 42}

        info = rbac.get_session_info(session_id)
        assert info is not None
        assert session.user_agent == "TestAgent/1.0"
        assert session.metadata["custom_field"] == "custom_value"
        assert session.metadata["number"] == 42

    def test_audit_logging_with_none_logger(self, rbac_no_audit):
        """Test audit logging when logger is None."""
        # Should not raise exceptions
        rbac_no_audit.create_role("test_role")
        rbac_no_audit.create_user("user123", "testuser")
        session_id = rbac_no_audit.create_session("user123")

        result = rbac_no_audit.check_access(session_id, Permission.READ_COLLECTION)
        assert result == AccessResult.GRANTED