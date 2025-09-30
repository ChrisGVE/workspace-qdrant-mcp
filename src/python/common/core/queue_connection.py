"""
SQLite Connection Management for Queue System

Provides optimized SQLite connection configuration with WAL mode,
connection pooling, and concurrent access support for the priority queue system.

Features:
    - Write-Ahead Logging (WAL) mode for concurrent reads
    - Connection pooling with configurable limits
    - Automatic busy timeout and retry handling
    - Transaction isolation level management
    - Performance optimizations for queue operations
    - Compatible with both sync and async code

Example:
    ```python
    from workspace_qdrant_mcp.core.queue_connection import QueueConnectionPool

    # Initialize connection pool
    pool = QueueConnectionPool(
        db_path="workspace_state.db",
        max_connections=10,
        busy_timeout=30.0
    )
    await pool.initialize()

    # Get connection from pool
    async with pool.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM ingestion_queue")
    ```
"""

import asyncio
import sqlite3
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

from loguru import logger


@dataclass
class ConnectionConfig:
    """Configuration for SQLite connections."""

    # Connection parameters
    busy_timeout: float = 30.0  # seconds to wait when database is locked

    # WAL mode settings
    wal_autocheckpoint: int = 1000  # pages before auto-checkpoint
    wal_checkpoint_interval: int = 300  # seconds between forced checkpoints

    # Performance tuning
    cache_size: int = 10000  # pages (~40MB with 4KB pages)
    mmap_size: int = 268435456  # 256MB memory-mapped I/O
    synchronous: str = "NORMAL"  # NORMAL balances safety and performance
    temp_store: str = "MEMORY"  # Use memory for temp tables

    # Connection pooling
    max_connections: int = 10
    min_connections: int = 2
    connection_timeout: float = 30.0  # seconds to wait for available connection

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 0.1  # seconds between retries


class QueueConnection:
    """
    Wrapper for SQLite connection with queue-optimized settings.

    Configures connection with WAL mode, optimized pragmas, and proper
    error handling for concurrent access patterns.
    """

    def __init__(self, db_path: str, config: ConnectionConfig):
        """
        Initialize queue connection.

        Args:
            db_path: Path to SQLite database
            config: Connection configuration
        """
        self.db_path = Path(db_path)
        self.config = config
        self.conn: Optional[sqlite3.Connection] = None
        self._is_connected = False

    def connect(self) -> sqlite3.Connection:
        """
        Establish connection with optimized settings.

        Returns:
            Configured SQLite connection
        """
        if self._is_connected and self.conn:
            return self.conn

        logger.debug(f"Opening SQLite connection: {self.db_path}")

        # Create connection with timeout
        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.config.busy_timeout,
            check_same_thread=False,  # Allow use across threads (with proper locking)
            isolation_level=None,  # Autocommit mode for explicit transaction control
        )

        # Enable row factory for dict-like access
        self.conn.row_factory = sqlite3.Row

        # Configure for optimal performance and concurrency
        self._apply_pragmas()

        self._is_connected = True
        logger.debug("SQLite connection configured successfully")

        return self.conn

    def _apply_pragmas(self):
        """Apply performance and concurrency PRAGMA settings."""
        pragmas = [
            # WAL mode for concurrent access
            ("journal_mode", "WAL"),

            # Synchronous mode (NORMAL is good balance for WAL)
            ("synchronous", self.config.synchronous),

            # Cache size (negative value = KB, positive = pages)
            ("cache_size", self.config.cache_size),

            # Memory-mapped I/O size
            ("mmap_size", self.config.mmap_size),

            # Temp storage in memory
            ("temp_store", self.config.temp_store),

            # WAL auto-checkpoint threshold
            ("wal_autocheckpoint", self.config.wal_autocheckpoint),

            # Enable foreign keys
            ("foreign_keys", "ON"),

            # Query optimizer
            ("optimize", None),  # Analyze query patterns
        ]

        for pragma, value in pragmas:
            if value is None:
                self.conn.execute(f"PRAGMA {pragma}")
            else:
                self.conn.execute(f"PRAGMA {pragma}={value}")

        # Log WAL mode confirmation
        cursor = self.conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        if mode.upper() != "WAL":
            logger.warning(f"Failed to enable WAL mode, using: {mode}")
        else:
            logger.debug("WAL mode enabled successfully")

    def checkpoint(self, mode: str = "PASSIVE") -> tuple:
        """
        Perform WAL checkpoint.

        Args:
            mode: Checkpoint mode (PASSIVE, FULL, RESTART, TRUNCATE)

        Returns:
            Tuple of (busy, log_size, checkpointed_frames)
        """
        cursor = self.conn.execute(f"PRAGMA wal_checkpoint({mode})")
        return cursor.fetchone()

    def close(self):
        """Close connection."""
        if self.conn and self._is_connected:
            logger.debug("Closing SQLite connection")
            try:
                # Perform final checkpoint
                self.checkpoint(mode="FULL")
            except Exception as e:
                logger.warning(f"Failed to perform final checkpoint: {e}")
            finally:
                self.conn.close()
                self.conn = None
                self._is_connected = False

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False  # Don't suppress exceptions


class QueueConnectionPool:
    """
    Connection pool for SQLite queue database.

    Manages a pool of connections for efficient concurrent access from
    multiple components (daemon, MCP server, CLI).

    Thread-safe connection pooling with automatic connection recycling.
    """

    def __init__(
        self,
        db_path: str,
        config: Optional[ConnectionConfig] = None
    ):
        """
        Initialize connection pool.

        Args:
            db_path: Path to SQLite database
            config: Connection configuration (uses defaults if None)
        """
        self.db_path = Path(db_path)
        self.config = config or ConnectionConfig()

        # Connection pool
        self._pool: Queue[QueueConnection] = Queue(maxsize=self.config.max_connections)
        self._lock = threading.RLock()
        self._initialized = False

        # Checkpoint management
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize connection pool."""
        if self._initialized:
            return

        logger.info(f"Initializing connection pool: {self.db_path}")

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create minimum connections
        with self._lock:
            for _ in range(self.config.min_connections):
                conn = QueueConnection(str(self.db_path), self.config)
                conn.connect()
                self._pool.put(conn)

        # Start periodic checkpoint task
        self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())

        self._initialized = True
        logger.info(f"Connection pool initialized with {self.config.min_connections} connections")

    async def close(self):
        """Close all connections and shut down pool."""
        if not self._initialized:
            return

        logger.info("Closing connection pool")

        # Signal shutdown
        self._shutdown_event.set()

        # Stop checkpoint task
        if self._checkpoint_task and not self._checkpoint_task.done():
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        with self._lock:
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except Empty:
                    break

        self._initialized = False
        logger.info("Connection pool closed")

    def _get_connection(self) -> QueueConnection:
        """
        Get connection from pool (synchronous).

        Returns:
            Connection from pool

        Raises:
            TimeoutError: If no connection available within timeout
        """
        try:
            # Try to get existing connection
            conn = self._pool.get(timeout=self.config.connection_timeout)

            # Verify connection is valid
            if not conn._is_connected:
                conn.connect()

            return conn

        except Empty:
            # Pool exhausted, create new connection if under max
            with self._lock:
                if self._pool.qsize() < self.config.max_connections:
                    logger.debug("Creating additional connection")
                    conn = QueueConnection(str(self.db_path), self.config)
                    conn.connect()
                    return conn
                else:
                    raise TimeoutError("Connection pool exhausted and at maximum capacity")

    def _return_connection(self, conn: QueueConnection):
        """Return connection to pool."""
        try:
            self._pool.put_nowait(conn)
        except:
            # Pool is full, close the connection
            logger.warning("Pool full, closing excess connection")
            conn.close()

    @contextmanager
    def get_connection(self):
        """
        Get connection from pool (context manager for sync code).

        Usage:
            with pool.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM table")
        """
        conn_wrapper = self._get_connection()
        try:
            yield conn_wrapper.conn
        finally:
            self._return_connection(conn_wrapper)

    @asynccontextmanager
    async def get_connection_async(self):
        """
        Get connection from pool (async context manager).

        Usage:
            async with pool.get_connection_async() as conn:
                cursor = conn.execute("SELECT * FROM table")
        """
        # Get connection in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        conn_wrapper = await loop.run_in_executor(None, self._get_connection)

        try:
            yield conn_wrapper.conn
        finally:
            await loop.run_in_executor(None, self._return_connection, conn_wrapper)

    async def _checkpoint_loop(self):
        """Background task for periodic WAL checkpoints."""
        logger.info("Starting WAL checkpoint loop")

        while not self._shutdown_event.is_set():
            try:
                # Wait for checkpoint interval
                await asyncio.sleep(self.config.wal_checkpoint_interval)

                # Perform checkpoint
                logger.debug("Performing periodic WAL checkpoint")

                with self.get_connection() as conn:
                    cursor = conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    result = cursor.fetchone()

                    if result:
                        busy, log_size, checkpointed = result
                        logger.debug(
                            f"Checkpoint complete: busy={busy}, "
                            f"log_size={log_size}, checkpointed={checkpointed}"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in checkpoint loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

        logger.info("WAL checkpoint loop stopped")

    def execute_with_retry(self, query: str, params: tuple = (), max_retries: Optional[int] = None):
        """
        Execute query with automatic retry on SQLITE_BUSY.

        Args:
            query: SQL query to execute
            params: Query parameters
            max_retries: Maximum retry attempts (uses config default if None)

        Returns:
            Cursor with query results

        Raises:
            sqlite3.OperationalError: If max retries exceeded
        """
        max_retries = max_retries or self.config.max_retries
        last_error = None

        for attempt in range(max_retries):
            try:
                with self.get_connection() as conn:
                    return conn.execute(query, params)

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower():
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.debug(f"Database locked, retrying ({attempt + 1}/{max_retries})...")
                        import time
                        time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    raise

        # Max retries exceeded
        raise sqlite3.OperationalError(f"Max retries exceeded: {last_error}")


def get_default_pool(db_path: Optional[str] = None) -> QueueConnectionPool:
    """
    Get default connection pool with standard configuration.

    Args:
        db_path: Optional custom database path (uses OS standard if None)

    Returns:
        Configured connection pool
    """
    if db_path is None:
        from ..utils.os_directories import OSDirectories
        os_dirs = OSDirectories()
        os_dirs.ensure_directories()
        db_path = str(os_dirs.get_state_file("workspace_state.db"))

    return QueueConnectionPool(db_path=db_path)
