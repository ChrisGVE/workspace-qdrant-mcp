"""
Connection optimization for web crawling with keep-alive and pooling.

This module provides advanced connection management features including
connection pooling, keep-alive optimization, and request pipelining.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import aiohttp
from loguru import logger


@dataclass
class ConnectionStats:
    """Statistics for connection management."""
    total_connections_created: int = 0
    total_connections_reused: int = 0
    total_requests: int = 0
    active_connections: int = 0
    connection_errors: int = 0
    avg_connection_time: float = 0.0
    total_connection_time: float = 0.0
    connection_pool_size: int = 0
    keep_alive_reused: int = 0


@dataclass
class ConnectionConfig:
    """Configuration for connection optimization."""
    # Connection pool settings
    pool_connections: int = 100  # Number of connection pools to maintain
    pool_maxsize: int = 10  # Max connections per pool
    max_retries: int = 3

    # Keep-alive settings
    keepalive_expiry: float = 30.0  # Keep-alive timeout in seconds
    keepalive_idle: float = 2.0  # Time before sending keep-alive probes

    # Connection timeouts
    total_timeout: float = 30.0
    connect_timeout: float = 10.0
    read_timeout: float = 30.0

    # TLS/SSL settings
    verify_ssl: bool = True
    ssl_context: Optional[Any] = None

    # HTTP/2 settings
    enable_http2: bool = False  # Enable HTTP/2 support

    # Connection limits per host
    limit_per_host: int = 30
    total_connections: int = 1000

    # DNS settings
    ttl_dns_cache: float = 300.0  # DNS cache TTL in seconds
    use_dns_cache: bool = True


class ConnectionOptimizer:
    """Optimizes HTTP connections for web crawling."""

    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.config = config or ConnectionConfig()
        self.stats = ConnectionStats()
        self.host_connection_pools: Dict[str, aiohttp.TCPConnector] = {}
        self.session_cache: Dict[str, aiohttp.ClientSession] = {}
        self.connection_start_times: Dict[str, float] = {}
        self.last_cleanup_time = time.time()

    async def create_optimized_connector(self, host: Optional[str] = None) -> aiohttp.TCPConnector:
        """Create an optimized TCP connector."""
        try:
            # Create SSL context if needed
            ssl_context = None
            if self.config.verify_ssl and self.config.ssl_context:
                ssl_context = self.config.ssl_context

            connector = aiohttp.TCPConnector(
                limit=self.config.total_connections,
                limit_per_host=self.config.limit_per_host,
                ttl_dns_cache=self.config.ttl_dns_cache if self.config.use_dns_cache else 0,
                use_dns_cache=self.config.use_dns_cache,
                keepalive_timeout=self.config.keepalive_expiry,
                enable_cleanup_closed=True,
                ssl=ssl_context if ssl_context else self.config.verify_ssl
            )

            self.stats.total_connections_created += 1

            # Store connector for reuse
            if host:
                self.host_connection_pools[host] = connector

            logger.debug(f"Created optimized connector for host: {host}")
            return connector

        except Exception as e:
            logger.error(f"Failed to create optimized connector: {e}")
            self.stats.connection_errors += 1
            raise

    async def create_optimized_session(self,
                                     host: Optional[str] = None,
                                     headers: Optional[Dict[str, str]] = None) -> aiohttp.ClientSession:
        """Create an optimized HTTP session."""
        try:
            start_time = time.time()

            # Check if we have a cached session for this host
            cache_key = f"{host or 'default'}"
            if cache_key in self.session_cache:
                session = self.session_cache[cache_key]
                if not session.closed:
                    self.stats.total_connections_reused += 1
                    self.stats.keep_alive_reused += 1
                    return session
                else:
                    # Remove closed session
                    del self.session_cache[cache_key]

            # Create timeout configuration
            timeout = aiohttp.ClientTimeout(
                total=self.config.total_timeout,
                connect=self.config.connect_timeout,
                sock_read=self.config.read_timeout
            )

            # Use existing connector or create new one
            connector = None
            if host and host in self.host_connection_pools:
                connector = self.host_connection_pools[host]
                self.stats.total_connections_reused += 1
            else:
                connector = await self.create_optimized_connector(host)

            # Default headers for optimization
            default_headers = {
                'Connection': 'keep-alive',
                'Keep-Alive': f'timeout={int(self.config.keepalive_expiry)}, max=1000',
                'Accept-Encoding': 'gzip, deflate, br',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }

            if headers:
                default_headers.update(headers)

            # Create session
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=default_headers,
                connector_owner=False  # Don't close connector with session
            )

            # Cache the session
            self.session_cache[cache_key] = session

            # Update statistics
            connection_time = time.time() - start_time
            self.stats.total_connection_time += connection_time
            self.stats.total_connections_created += 1

            logger.debug(f"Created optimized session for {host} in {connection_time:.3f}s")
            return session

        except Exception as e:
            logger.error(f"Failed to create optimized session: {e}")
            self.stats.connection_errors += 1
            raise

    async def make_request(self,
                          url: str,
                          method: str = 'GET',
                          headers: Optional[Dict[str, str]] = None,
                          **kwargs) -> aiohttp.ClientResponse:
        """Make an optimized HTTP request."""
        parsed_url = urlparse(url)
        host = parsed_url.netloc

        try:
            session = await self.create_optimized_session(host, headers)
            self.stats.total_requests += 1

            # Make the request
            response = await session.request(method, url, **kwargs)

            # Update keep-alive stats if connection was reused
            if hasattr(response, 'connection') and response.connection:
                if hasattr(response.connection, 'is_connected') and response.connection.is_connected():
                    self.stats.keep_alive_reused += 1

            return response

        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            self.stats.connection_errors += 1
            raise

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics."""
        # Calculate averages
        if self.stats.total_connections_created > 0:
            self.stats.avg_connection_time = (
                self.stats.total_connection_time / self.stats.total_connections_created
            )

        self.stats.active_connections = len([
            session for session in self.session_cache.values()
            if not session.closed
        ])

        self.stats.connection_pool_size = len(self.host_connection_pools)

        return {
            'total_connections_created': self.stats.total_connections_created,
            'total_connections_reused': self.stats.total_connections_reused,
            'total_requests': self.stats.total_requests,
            'active_connections': self.stats.active_connections,
            'connection_errors': self.stats.connection_errors,
            'avg_connection_time': self.stats.avg_connection_time,
            'connection_pool_size': self.stats.connection_pool_size,
            'keep_alive_reused': self.stats.keep_alive_reused,
            'reuse_rate': (
                self.stats.total_connections_reused /
                max(1, self.stats.total_connections_created + self.stats.total_connections_reused)
            ),
            'error_rate': self.stats.connection_errors / max(1, self.stats.total_requests)
        }

    async def cleanup_idle_connections(self) -> None:
        """Clean up idle and expired connections."""
        current_time = time.time()

        # Clean up closed sessions
        closed_sessions = [
            key for key, session in self.session_cache.items()
            if session.closed
        ]

        for key in closed_sessions:
            del self.session_cache[key]

        # Close idle connectors (this is a simplified approach)
        idle_connectors = []
        for host, connector in self.host_connection_pools.items():
            if connector.closed:
                idle_connectors.append(host)

        for host in idle_connectors:
            del self.host_connection_pools[host]

        self.last_cleanup_time = current_time

        if closed_sessions or idle_connectors:
            logger.debug(f"Cleaned up {len(closed_sessions)} sessions and {len(idle_connectors)} connectors")

    async def close_all_connections(self) -> None:
        """Close all connections and clean up resources."""
        try:
            # Close all sessions
            for session in self.session_cache.values():
                if not session.closed:
                    await session.close()

            # Close all connectors
            for connector in self.host_connection_pools.values():
                if not connector.closed:
                    await connector.close()

            # Clear caches
            self.session_cache.clear()
            self.host_connection_pools.clear()

            logger.info("All connections closed successfully")

        except Exception as e:
            logger.error(f"Error closing connections: {e}")

    def should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        return time.time() - self.last_cleanup_time > 300  # 5 minutes


class UserAgentRotator:
    """Rotates user agents to avoid detection."""

    def __init__(self, user_agents: Optional[List[str]] = None):
        """Initialize with user agents list."""
        if user_agents:
            self.user_agents = user_agents
        else:
            self.user_agents = self._get_default_user_agents()

        self.current_index = 0
        self.usage_stats = {ua: 0 for ua in self.user_agents}

    def _get_default_user_agents(self) -> List[str]:
        """Get default user agent strings."""
        return [
            # Chrome on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Chrome on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Chrome on Linux
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Firefox on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            # Firefox on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            # Safari on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            # Edge on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            # Custom crawler identification
            "WorkspaceQdrant-WebCrawler/1.0 (+https://github.com/workspace-qdrant-mcp; contact@example.com)"
        ]

    def get_next_user_agent(self) -> str:
        """Get next user agent in rotation."""
        user_agent = self.user_agents[self.current_index]
        self.usage_stats[user_agent] += 1

        self.current_index = (self.current_index + 1) % len(self.user_agents)
        return user_agent

    def get_random_user_agent(self) -> str:
        """Get a random user agent."""
        import random
        user_agent = random.choice(self.user_agents)
        self.usage_stats[user_agent] += 1
        return user_agent

    def get_least_used_user_agent(self) -> str:
        """Get the least used user agent."""
        least_used = min(self.usage_stats.items(), key=lambda x: x[1])
        user_agent = least_used[0]
        self.usage_stats[user_agent] += 1
        return user_agent

    def add_user_agent(self, user_agent: str) -> None:
        """Add a new user agent to rotation."""
        if user_agent not in self.user_agents:
            self.user_agents.append(user_agent)
            self.usage_stats[user_agent] = 0

    def remove_user_agent(self, user_agent: str) -> bool:
        """Remove a user agent from rotation."""
        if user_agent in self.user_agents and len(self.user_agents) > 1:
            self.user_agents.remove(user_agent)
            del self.usage_stats[user_agent]

            # Reset index if it's out of bounds
            if self.current_index >= len(self.user_agents):
                self.current_index = 0

            return True
        return False

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get user agent usage statistics."""
        total_uses = sum(self.usage_stats.values())

        return {
            'total_user_agents': len(self.user_agents),
            'total_uses': total_uses,
            'current_index': self.current_index,
            'usage_distribution': {
                ua: {
                    'count': count,
                    'percentage': (count / max(1, total_uses)) * 100
                }
                for ua, count in self.usage_stats.items()
            },
            'most_used': max(self.usage_stats.items(), key=lambda x: x[1])[0] if self.usage_stats else None,
            'least_used': min(self.usage_stats.items(), key=lambda x: x[1])[0] if self.usage_stats else None
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.usage_stats = {ua: 0 for ua in self.user_agents}
        self.current_index = 0


# Request pipelining for batch operations
class RequestPipeline:
    """Handles request pipelining for efficient batch processing."""

    def __init__(self,
                 connection_optimizer: ConnectionOptimizer,
                 max_concurrent: int = 50,
                 rate_limit: float = 2.0):
        self.optimizer = connection_optimizer
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit  # Requests per second
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_request_time = 0.0

    async def process_url_batch(self,
                               urls: List[str],
                               request_kwargs: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Any]]:
        """Process a batch of URLs concurrently with rate limiting."""
        if not request_kwargs:
            request_kwargs = {}

        async def process_single_url(url: str) -> Tuple[str, Any]:
            async with self.semaphore:
                # Rate limiting
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < (1.0 / self.rate_limit):
                    wait_time = (1.0 / self.rate_limit) - time_since_last
                    await asyncio.sleep(wait_time)

                self.last_request_time = time.time()

                try:
                    response = await self.optimizer.make_request(url, **request_kwargs)
                    return (url, response)
                except Exception as e:
                    return (url, e)

        # Process all URLs concurrently
        tasks = [process_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def process_with_callback(self,
                                  urls: List[str],
                                  callback: callable,
                                  request_kwargs: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Process URLs with callback function for each result."""
        results = await self.process_url_batch(urls, request_kwargs)

        processed_results = []
        for url, result in results:
            try:
                if isinstance(result, Exception):
                    processed_result = await callback(url, None, result)
                else:
                    processed_result = await callback(url, result, None)
                processed_results.append(processed_result)
            except Exception as e:
                logger.error(f"Callback failed for {url}: {e}")
                processed_results.append(None)

        return processed_results