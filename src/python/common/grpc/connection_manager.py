"""
gRPC connection configuration for DaemonClient.

This module provides connection configuration for the gRPC client.
"""


class ConnectionConfig:
    """Configuration for gRPC connection management."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 50051,
        max_message_length: int = 100 * 1024 * 1024,  # 100MB
        max_retries: int = 3,
        retry_backoff_multiplier: float = 1.5,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        connection_timeout: float = 10.0,
        health_check_interval: float = 30.0,
        idle_timeout: float = 300.0,  # 5 minutes
        keepalive_time: int = 30,
        keepalive_timeout: int = 5,
        keepalive_without_calls: bool = True,
        max_concurrent_streams: int = 100,
        # Enhanced security options
        enable_tls: bool = False,
        tls_cert_path: str | None = None,
        tls_key_path: str | None = None,
        tls_ca_path: str | None = None,
        api_key: str | None = None,
        # Connection pooling options
        enable_connection_pooling: bool = True,
        pool_size: int = 5,
        max_pool_size: int = 10,
        pool_timeout: float = 30.0,
        # Circuit breaker options
        enable_circuit_breaker: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
    ):
        # Basic connection settings
        self.host = host
        self.port = port
        self.max_message_length = max_message_length
        self.max_retries = max_retries
        self.retry_backoff_multiplier = retry_backoff_multiplier
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.connection_timeout = connection_timeout
        self.health_check_interval = health_check_interval
        self.idle_timeout = idle_timeout
        self.keepalive_time = keepalive_time
        self.keepalive_timeout = keepalive_timeout
        self.keepalive_without_calls = keepalive_without_calls
        self.max_concurrent_streams = max_concurrent_streams

        # Enhanced security settings
        self.enable_tls = enable_tls
        self.tls_cert_path = tls_cert_path
        self.tls_key_path = tls_key_path
        self.tls_ca_path = tls_ca_path
        self.api_key = api_key

        # Connection pooling settings
        self.enable_connection_pooling = enable_connection_pooling
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self.pool_timeout = pool_timeout

        # Circuit breaker settings
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_failure_threshold = circuit_breaker_failure_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout

    @property
    def address(self) -> str:
        """Get the full gRPC address."""
        return f"{self.host}:{self.port}"

    def get_channel_options(self) -> list:
        """Get gRPC channel options based on configuration."""
        options = [
            ("grpc.max_send_message_length", self.max_message_length),
            ("grpc.max_receive_message_length", self.max_message_length),
            ("grpc.keepalive_time_ms", self.keepalive_time * 1000),
            ("grpc.keepalive_timeout_ms", self.keepalive_timeout * 1000),
            ("grpc.keepalive_permit_without_calls", self.keepalive_without_calls),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 300000),
            ("grpc.max_concurrent_streams", self.max_concurrent_streams),
            # Performance optimizations
            ("grpc.so_reuseport", 1),
            ("grpc.tcp_user_timeout", self.connection_timeout * 1000),
            ("grpc.enable_http_proxy", 0),
        ]

        return options
