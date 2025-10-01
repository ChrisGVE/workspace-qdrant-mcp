"""Version tracking and hash comparison for language support configuration.

This module provides functionality to track changes to language_support.yaml using
SHA256 hash comparison. It stores version information in the language_support_version
table and detects when the configuration file has been modified.

Classes:
    LanguageSupportVersionTracker: Manages version tracking and hash comparison

Example:
    >>> from pathlib import Path
    >>> tracker = LanguageSupportVersionTracker(db_path)
    >>> await tracker.initialize()
    >>> needs_reload = await tracker.needs_update(Path("language_support.yaml"))
    >>> if needs_reload:
    ...     await tracker.update_version("1.0.0", hash_value, language_count=500)
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger


class LanguageSupportVersionTracker:
    """Tracks version and content changes for language support configuration.

    This class manages the language_support_version table, calculating file hashes,
    comparing them with stored values, and updating version information when the
    configuration changes.

    Attributes:
        db_path: Path to the SQLite database file
        connection: SQLite database connection

    Methods:
        initialize: Initialize database connection
        get_current_version: Retrieve stored version string
        get_content_hash: Retrieve stored content hash
        update_version: Update version and hash in database
        calculate_file_hash: Calculate SHA256 hash of file
        needs_update: Check if file has changed since last load
        close: Close database connection
    """

    def __init__(self, db_path: Path | str):
        """Initialize version tracker with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.connection: Optional[sqlite3.Connection] = None

    async def initialize(self) -> None:
        """Initialize database connection.

        Opens a connection to the SQLite database. The database and tables
        should already exist (created by SQLiteStateManager).

        Raises:
            sqlite3.Error: If database connection fails
        """
        if self.connection is None:
            self.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            # Enable foreign keys and WAL mode for consistency with state manager
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            logger.debug(f"Initialized version tracker with database: {self.db_path}")

    async def get_current_version(self) -> Optional[str]:
        """Retrieve the current stored version string.

        Queries the language_support_version table for the most recent version.
        The version string is stored in the yaml_hash field as we're using hash-based
        versioning rather than explicit version strings.

        Returns:
            Version hash string if found, None otherwise

        Raises:
            RuntimeError: If database connection is not initialized
            sqlite3.Error: If database query fails

        Example:
            >>> tracker = LanguageSupportVersionTracker(db_path)
            >>> await tracker.initialize()
            >>> version = await tracker.get_current_version()
            >>> print(version)  # e.g., "a3f5c8..."
        """
        if self.connection is None:
            raise RuntimeError("Database connection not initialized. Call initialize() first.")

        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT yaml_hash
                FROM language_support_version
                ORDER BY loaded_at DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except sqlite3.Error as e:
            logger.error(f"Failed to get current version: {e}")
            raise
        finally:
            cursor.close()

    async def get_content_hash(self) -> Optional[str]:
        """Retrieve the stored content hash.

        Queries the language_support_version table for the most recent content hash.
        This is the SHA256 hash of the language_support.yaml file content.

        Returns:
            SHA256 hash string (hex digest) if found, None otherwise

        Raises:
            RuntimeError: If database connection is not initialized
            sqlite3.Error: If database query fails

        Example:
            >>> tracker = LanguageSupportVersionTracker(db_path)
            >>> await tracker.initialize()
            >>> hash_value = await tracker.get_content_hash()
            >>> print(hash_value)  # e.g., "a3f5c8b2..."
        """
        # Since we're using hash-based versioning, yaml_hash IS the content hash
        return await self.get_current_version()

    async def update_version(
        self,
        version: str,
        content_hash: str,
        language_count: int = 0,
    ) -> None:
        """Update or insert version and hash in database.

        Uses INSERT OR REPLACE to update the version information. Since yaml_hash
        has a UNIQUE constraint, this will replace an existing record with the same
        hash or insert a new one.

        Args:
            version: Version string (for compatibility, but we use hash-based versioning)
            content_hash: SHA256 hash of file content (hex digest)
            language_count: Number of languages in configuration (default: 0)

        Raises:
            RuntimeError: If database connection is not initialized
            sqlite3.Error: If database update fails

        Example:
            >>> tracker = LanguageSupportVersionTracker(db_path)
            >>> await tracker.initialize()
            >>> hash_value = tracker.calculate_file_hash(Path("language_support.yaml"))
            >>> await tracker.update_version("1.0.0", hash_value, language_count=500)
        """
        if self.connection is None:
            raise RuntimeError("Database connection not initialized. Call initialize() first.")

        cursor = self.connection.cursor()
        try:
            # Get current timestamp
            now = datetime.now(timezone.utc).isoformat()

            # Use INSERT OR REPLACE to update or create record
            cursor.execute(
                """
                INSERT OR REPLACE INTO language_support_version
                    (yaml_hash, loaded_at, language_count, last_checked_at)
                VALUES (?, ?, ?, ?)
                """,
                (content_hash, now, language_count, now),
            )
            self.connection.commit()
            logger.info(
                f"Updated language support version: hash={content_hash[:8]}..., "
                f"languages={language_count}"
            )
        except sqlite3.Error as e:
            self.connection.rollback()
            logger.error(f"Failed to update version: {e}")
            raise
        finally:
            cursor.close()

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content.

        Reads the file in binary mode and computes its SHA256 hash.
        This is used to detect changes in the language_support.yaml file.

        Args:
            file_path: Path to file to hash

        Returns:
            SHA256 hash as hex digest string

        Raises:
            FileNotFoundError: If file doesn't exist
            OSError: If file cannot be read

        Example:
            >>> tracker = LanguageSupportVersionTracker(db_path)
            >>> hash_value = tracker.calculate_file_hash(Path("language_support.yaml"))
            >>> print(hash_value)  # e.g., "a3f5c8b2e4d6..."
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except OSError as e:
            logger.error(f"Failed to read file for hashing: {file_path}: {e}")
            raise

    async def needs_update(self, file_path: Path) -> bool:
        """Check if file has changed since last load.

        Calculates the current file hash and compares it with the stored hash.
        Returns True if the hashes differ or if no hash is stored.

        Args:
            file_path: Path to file to check

        Returns:
            True if file has changed or no hash exists, False if unchanged

        Raises:
            RuntimeError: If database connection is not initialized
            FileNotFoundError: If file doesn't exist
            OSError: If file cannot be read
            sqlite3.Error: If database query fails

        Example:
            >>> tracker = LanguageSupportVersionTracker(db_path)
            >>> await tracker.initialize()
            >>> if await tracker.needs_update(Path("language_support.yaml")):
            ...     print("Configuration has changed, reload required")
        """
        if self.connection is None:
            raise RuntimeError("Database connection not initialized. Call initialize() first.")

        # Calculate current file hash
        current_hash = self.calculate_file_hash(file_path)

        # Get stored hash
        stored_hash = await self.get_content_hash()

        # Update last_checked_at timestamp
        await self._update_last_checked()

        # Compare hashes
        if stored_hash is None:
            logger.info(
                f"No stored hash found for {file_path.name}, update needed"
            )
            return True

        needs_reload = current_hash != stored_hash
        if needs_reload:
            logger.info(
                f"File {file_path.name} has changed: "
                f"stored={stored_hash[:8]}..., current={current_hash[:8]}..."
            )
        else:
            logger.debug(f"File {file_path.name} unchanged: hash={current_hash[:8]}...")

        return needs_reload

    async def _update_last_checked(self) -> None:
        """Update the last_checked_at timestamp for the most recent version.

        This is an internal method called by needs_update to track when we last
        checked for configuration changes.

        Raises:
            sqlite3.Error: If database update fails
        """
        if self.connection is None:
            return

        cursor = self.connection.cursor()
        try:
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                """
                UPDATE language_support_version
                SET last_checked_at = ?
                WHERE id = (
                    SELECT id FROM language_support_version
                    ORDER BY loaded_at DESC
                    LIMIT 1
                )
                """,
                (now,),
            )
            self.connection.commit()
        except sqlite3.Error as e:
            self.connection.rollback()
            logger.warning(f"Failed to update last_checked_at: {e}")
        finally:
            cursor.close()

    async def close(self) -> None:
        """Close database connection.

        Closes the SQLite database connection if it's open. Should be called
        when the version tracker is no longer needed.

        Example:
            >>> tracker = LanguageSupportVersionTracker(db_path)
            >>> await tracker.initialize()
            >>> # ... use tracker ...
            >>> await tracker.close()
        """
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            logger.debug("Closed version tracker database connection")

    def __del__(self):
        """Cleanup database connection on object deletion."""
        if self.connection is not None:
            self.connection.close()
