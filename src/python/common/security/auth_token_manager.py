"""Authentication token management with rotation and security features.

This module provides comprehensive token management including:
- Token rotation and refresh
- Token validation and expiration checking
- Secure token storage and retrieval
- Token revocation and blacklisting
- Multi-tenant token management
- Token encryption and decryption
"""

import asyncio
import time
import secrets
import hashlib
import hmac
from typing import Dict, Optional, Set, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
import json
import base64

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger


class TokenValidationError(Exception):
    """Raised when token validation fails."""
    pass


class TokenRotationError(Exception):
    """Raised when token rotation fails."""
    pass


class AuthToken:
    """Authentication token with metadata."""

    def __init__(
        self,
        token: str,
        expires_at: Optional[datetime] = None,
        issued_at: Optional[datetime] = None,
        issuer: str = "workspace-qdrant-mcp",
        subject: Optional[str] = None,
        scopes: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize authentication token.

        Args:
            token: The actual token string
            expires_at: Token expiration time
            issued_at: Token issuance time
            issuer: Token issuer
            subject: Token subject (user/client identifier)
            scopes: Token scopes/permissions
            metadata: Additional token metadata
        """
        self.token = token
        self.expires_at = expires_at
        self.issued_at = issued_at or datetime.utcnow()
        self.issuer = issuer
        self.subject = subject
        self.scopes = scopes or set()
        self.metadata = metadata or {}
        self.token_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate secure hash of token."""
        return hashlib.sha256(self.token.encode()).hexdigest()

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def expires_in(self) -> Optional[timedelta]:
        """Get time until expiration."""
        if self.expires_at is None:
            return None
        return self.expires_at - datetime.utcnow()

    def is_valid(self) -> bool:
        """Check if token is valid (not expired and properly formatted)."""
        return not self.is_expired() and len(self.token) > 0

    def has_scope(self, scope: str) -> bool:
        """Check if token has specific scope."""
        return scope in self.scopes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'token_hash': self.token_hash,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'issued_at': self.issued_at.isoformat(),
            'issuer': self.issuer,
            'subject': self.subject,
            'scopes': list(self.scopes),
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], token: str) -> 'AuthToken':
        """Create token from dictionary representation."""
        expires_at = None
        if data.get('expires_at'):
            expires_at = datetime.fromisoformat(data['expires_at'])

        issued_at = datetime.fromisoformat(data['issued_at'])

        return cls(
            token=token,
            expires_at=expires_at,
            issued_at=issued_at,
            issuer=data.get('issuer', 'workspace-qdrant-mcp'),
            subject=data.get('subject'),
            scopes=set(data.get('scopes', [])),
            metadata=data.get('metadata', {}),
        )


class SecureTokenStorage:
    """Secure encrypted token storage."""

    def __init__(self, storage_path: Optional[Path] = None, encryption_key: Optional[bytes] = None):
        """Initialize secure token storage.

        Args:
            storage_path: Path to token storage file
            encryption_key: Encryption key for token storage
        """
        self.storage_path = storage_path or Path.home() / ".workspace-qdrant" / "tokens.enc"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize encryption
        if encryption_key:
            self.fernet = Fernet(encryption_key)
        else:
            self.fernet = self._derive_key()

        self._lock = RLock()

    def _derive_key(self) -> Fernet:
        """Derive encryption key from system information."""
        # Use a combination of system info and a stored salt
        salt_file = self.storage_path.parent / "salt"

        if salt_file.exists():
            with open(salt_file, 'rb') as f:
                salt = f.read()
        else:
            salt = secrets.token_bytes(32)
            with open(salt_file, 'wb') as f:
                f.write(salt)
            # Restrict permissions
            salt_file.chmod(0o600)

        # Derive key from system hostname and salt
        import socket
        system_id = socket.gethostname().encode()

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(system_id))
        return Fernet(key)

    def store_tokens(self, tokens: Dict[str, AuthToken]) -> None:
        """Store tokens securely."""
        with self._lock:
            # Convert tokens to serializable format
            token_data = {}
            for key, token in tokens.items():
                token_data[key] = {
                    'token': token.token,
                    'metadata': token.to_dict(),
                }

            # Encrypt and store
            serialized = json.dumps(token_data).encode()
            encrypted = self.fernet.encrypt(serialized)

            with open(self.storage_path, 'wb') as f:
                f.write(encrypted)

            # Restrict file permissions
            self.storage_path.chmod(0o600)

    def load_tokens(self) -> Dict[str, AuthToken]:
        """Load tokens from storage."""
        with self._lock:
            if not self.storage_path.exists():
                return {}

            try:
                with open(self.storage_path, 'rb') as f:
                    encrypted = f.read()

                decrypted = self.fernet.decrypt(encrypted)
                token_data = json.loads(decrypted.decode())

                # Reconstruct tokens
                tokens = {}
                for key, data in token_data.items():
                    tokens[key] = AuthToken.from_dict(
                        data['metadata'],
                        data['token']
                    )

                return tokens

            except Exception as e:
                logger.error(f"Failed to load tokens from storage: {e}")
                return {}

    def clear_storage(self) -> None:
        """Clear token storage."""
        with self._lock:
            if self.storage_path.exists():
                self.storage_path.unlink()


class AuthTokenManager:
    """Comprehensive authentication token manager with rotation."""

    def __init__(
        self,
        default_token_lifetime: timedelta = timedelta(hours=24),
        rotation_threshold: timedelta = timedelta(hours=6),
        max_tokens_per_subject: int = 5,
        token_storage: Optional[SecureTokenStorage] = None,
        token_refresh_callback: Optional[Callable[[str], Tuple[str, Optional[datetime]]]] = None,
    ):
        """Initialize token manager.

        Args:
            default_token_lifetime: Default token lifetime
            rotation_threshold: Time before expiry to trigger rotation
            max_tokens_per_subject: Maximum tokens per subject
            token_storage: Secure token storage instance
            token_refresh_callback: Callback to refresh tokens
        """
        self.default_token_lifetime = default_token_lifetime
        self.rotation_threshold = rotation_threshold
        self.max_tokens_per_subject = max_tokens_per_subject
        self.token_storage = token_storage or SecureTokenStorage()
        self.token_refresh_callback = token_refresh_callback

        # Runtime token cache
        self._tokens: Dict[str, AuthToken] = {}
        self._revoked_tokens: Set[str] = set()
        self._token_usage: Dict[str, int] = {}
        self._lock = RLock()

        # Load tokens from storage
        self._load_tokens()

        # Start background rotation task
        self._rotation_task = None
        self._start_rotation_task()

    def _load_tokens(self) -> None:
        """Load tokens from persistent storage."""
        stored_tokens = self.token_storage.load_tokens()
        with self._lock:
            for key, token in stored_tokens.items():
                if token.is_valid() and token.token_hash not in self._revoked_tokens:
                    self._tokens[key] = token
                    self._token_usage[key] = 0

    def _save_tokens(self) -> None:
        """Save tokens to persistent storage."""
        with self._lock:
            # Only save valid, non-revoked tokens
            tokens_to_save = {
                key: token for key, token in self._tokens.items()
                if token.is_valid() and token.token_hash not in self._revoked_tokens
            }
            self.token_storage.store_tokens(tokens_to_save)

    def create_token(
        self,
        subject: str,
        scopes: Optional[Set[str]] = None,
        lifetime: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuthToken:
        """Create a new authentication token.

        Args:
            subject: Token subject (user/client identifier)
            scopes: Token scopes/permissions
            lifetime: Token lifetime (defaults to default_token_lifetime)
            metadata: Additional token metadata

        Returns:
            New authentication token
        """
        with self._lock:
            # Check token limits per subject
            subject_tokens = [
                token for token in self._tokens.values()
                if token.subject == subject and token.is_valid()
            ]

            if len(subject_tokens) >= self.max_tokens_per_subject:
                # Revoke oldest token
                oldest_token = min(subject_tokens, key=lambda t: t.issued_at)
                self.revoke_token_by_hash(oldest_token.token_hash)

            # Generate secure token
            token_bytes = secrets.token_urlsafe(32)
            token_string = f"wqm_{token_bytes}"

            # Create token with expiration
            lifetime = lifetime or self.default_token_lifetime
            expires_at = datetime.utcnow() + lifetime

            token = AuthToken(
                token=token_string,
                expires_at=expires_at,
                subject=subject,
                scopes=scopes or set(),
                metadata=metadata or {},
            )

            # Store token
            token_key = f"{subject}:{token.token_hash[:16]}"
            self._tokens[token_key] = token
            self._token_usage[token_key] = 0

            # Save to persistent storage
            self._save_tokens()

            logger.info(f"Created new token for subject {subject}, expires at {expires_at}")
            return token

    def get_token(self, subject: str) -> Optional[AuthToken]:
        """Get valid token for subject.

        Args:
            subject: Token subject

        Returns:
            Valid token or None
        """
        with self._lock:
            # Find valid tokens for subject
            subject_tokens = [
                (key, token) for key, token in self._tokens.items()
                if token.subject == subject and token.is_valid()
                and token.token_hash not in self._revoked_tokens
            ]

            if not subject_tokens:
                return None

            # Return the newest valid token
            key, token = max(subject_tokens, key=lambda x: x[1].issued_at)

            # Track usage
            self._token_usage[key] += 1

            # Check if token needs rotation
            if self._needs_rotation(token):
                try:
                    rotated_token = self._rotate_token(key, token)
                    if rotated_token:
                        return rotated_token
                except Exception as e:
                    logger.warning(f"Token rotation failed for {subject}: {e}")

            return token

    def validate_token(self, token_string: str) -> Optional[AuthToken]:
        """Validate a token string.

        Args:
            token_string: Token to validate

        Returns:
            Valid token or None
        """
        token_hash = hashlib.sha256(token_string.encode()).hexdigest()

        with self._lock:
            # Check if token is revoked
            if token_hash in self._revoked_tokens:
                return None

            # Find token in storage
            for key, token in self._tokens.items():
                if token.token == token_string:
                    if token.is_valid():
                        self._token_usage[key] += 1
                        return token
                    else:
                        # Remove expired token
                        del self._tokens[key]
                        if key in self._token_usage:
                            del self._token_usage[key]
                        break

            return None

    def revoke_token(self, token_string: str) -> bool:
        """Revoke a token.

        Args:
            token_string: Token to revoke

        Returns:
            True if token was revoked
        """
        token_hash = hashlib.sha256(token_string.encode()).hexdigest()
        return self.revoke_token_by_hash(token_hash)

    def revoke_token_by_hash(self, token_hash: str) -> bool:
        """Revoke a token by hash.

        Args:
            token_hash: Hash of token to revoke

        Returns:
            True if token was revoked
        """
        with self._lock:
            self._revoked_tokens.add(token_hash)

            # Remove from active tokens
            keys_to_remove = []
            for key, token in self._tokens.items():
                if token.token_hash == token_hash:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._tokens[key]
                if key in self._token_usage:
                    del self._token_usage[key]

            # Update storage
            self._save_tokens()

            logger.info(f"Revoked token with hash {token_hash[:16]}...")
            return len(keys_to_remove) > 0

    def revoke_all_tokens(self, subject: str) -> int:
        """Revoke all tokens for a subject.

        Args:
            subject: Subject whose tokens to revoke

        Returns:
            Number of tokens revoked
        """
        with self._lock:
            tokens_to_revoke = [
                token for token in self._tokens.values()
                if token.subject == subject
            ]

            count = 0
            for token in tokens_to_revoke:
                if self.revoke_token_by_hash(token.token_hash):
                    count += 1

            return count

    def _needs_rotation(self, token: AuthToken) -> bool:
        """Check if token needs rotation."""
        if token.expires_at is None:
            return False

        time_to_expiry = token.expires_at - datetime.utcnow()
        return time_to_expiry <= self.rotation_threshold

    def _rotate_token(self, old_key: str, old_token: AuthToken) -> Optional[AuthToken]:
        """Rotate an expiring token."""
        if not self.token_refresh_callback:
            logger.debug("No token refresh callback configured")
            return None

        try:
            # Call refresh callback
            new_token_string, new_expires_at = self.token_refresh_callback(old_token.token)

            # Create new token
            new_token = AuthToken(
                token=new_token_string,
                expires_at=new_expires_at,
                subject=old_token.subject,
                scopes=old_token.scopes,
                metadata=old_token.metadata,
            )

            with self._lock:
                # Replace old token
                del self._tokens[old_key]
                if old_key in self._token_usage:
                    usage_count = self._token_usage.pop(old_key)
                else:
                    usage_count = 0

                # Store new token
                new_key = f"{new_token.subject}:{new_token.token_hash[:16]}"
                self._tokens[new_key] = new_token
                self._token_usage[new_key] = usage_count

                # Revoke old token
                self._revoked_tokens.add(old_token.token_hash)

                # Save to storage
                self._save_tokens()

            logger.info(f"Rotated token for subject {old_token.subject}")
            return new_token

        except Exception as e:
            raise TokenRotationError(f"Token rotation failed: {e}")

    def _start_rotation_task(self) -> None:
        """Start background token rotation task."""
        if self._rotation_task is not None:
            return

        async def rotation_task():
            while True:
                try:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    self._cleanup_expired_tokens()
                    self._rotate_expiring_tokens()
                except Exception as e:
                    logger.error(f"Token rotation task error: {e}")

        self._rotation_task = asyncio.create_task(rotation_task())

    def _cleanup_expired_tokens(self) -> None:
        """Clean up expired tokens."""
        with self._lock:
            expired_keys = []
            for key, token in self._tokens.items():
                if not token.is_valid():
                    expired_keys.append(key)

            for key in expired_keys:
                token = self._tokens[key]
                logger.debug(f"Cleaning up expired token for {token.subject}")
                del self._tokens[key]
                if key in self._token_usage:
                    del self._token_usage[key]

            if expired_keys:
                self._save_tokens()

    def _rotate_expiring_tokens(self) -> None:
        """Rotate tokens that are close to expiring."""
        if not self.token_refresh_callback:
            return

        with self._lock:
            tokens_to_rotate = []
            for key, token in self._tokens.items():
                if self._needs_rotation(token):
                    tokens_to_rotate.append((key, token))

            for key, token in tokens_to_rotate:
                try:
                    self._rotate_token(key, token)
                except Exception as e:
                    logger.warning(f"Failed to rotate token for {token.subject}: {e}")

    def get_token_stats(self) -> Dict[str, Any]:
        """Get token manager statistics."""
        with self._lock:
            valid_tokens = sum(1 for token in self._tokens.values() if token.is_valid())
            subjects = len(set(token.subject for token in self._tokens.values() if token.is_valid()))

            return {
                'total_tokens': len(self._tokens),
                'valid_tokens': valid_tokens,
                'revoked_tokens': len(self._revoked_tokens),
                'unique_subjects': subjects,
                'token_usage': dict(self._token_usage),
            }

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._rotation_task:
            self._rotation_task.cancel()
            self._rotation_task = None

        self._save_tokens()