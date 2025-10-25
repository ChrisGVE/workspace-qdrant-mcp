"""SSL/TLS configuration and warning management.

This module provides centralized SSL/TLS configuration management with context-aware
warning suppression. It replaces global SSL warning suppression with targeted,
environment-specific handling that preserves security warnings for remote connections
while suppressing them appropriately for local development environments.

Key Features:
    - Context-aware SSL warning suppression (localhost vs remote)
    - SSL certificate validation configuration
    - Authentication support for secure Qdrant deployments
    - Development vs production environment handling
    - Proper TLS configuration for HTTP transport

Example:
    ```python
    from workspace_qdrant_mcp.core.ssl_config import SSLContextManager

    ssl_manager = SSLContextManager()

    # Configure for localhost connection
    with suppress_qdrant_ssl_warnings():
        client = QdrantClient("http://localhost:6333")

    # Configure for remote connection with SSL
    ssl_config = ssl_manager.create_ssl_config(
        url="https://remote-qdrant.example.com",
        verify_certificates=True,
        auth_token="your_token_here"
    )
    ```
"""

import contextlib
import ssl
import warnings
from contextlib import AbstractContextManager
from typing import Any
from urllib.parse import urlparse

import urllib3
from loguru import logger

# Import enhanced certificate validation
try:
    from ..security.certificate_validator import (
        CertificateSecurityPolicy,
        CertificateValidationError,
        EnhancedCertificateValidator,
        create_secure_ssl_context,
    )
    _ENHANCED_VALIDATION_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced certificate validation not available")
    _ENHANCED_VALIDATION_AVAILABLE = False


class SSLConfiguration:
    """SSL configuration for Qdrant connections.

    This class encapsulates SSL/TLS settings and provides validation
    for secure connections to Qdrant instances.
    """

    def __init__(
        self,
        verify_certificates: bool = True,
        ca_cert_path: str | None = None,
        client_cert_path: str | None = None,
        client_key_path: str | None = None,
        auth_token: str | None = None,
        api_key: str | None = None,
        environment: str = "production",
        enhanced_validation: bool = True,
        certificate_pinning: dict[str, list[str]] | None = None,
        security_policy: Any | None = None,
    ):
        """Initialize SSL configuration.

        Args:
            verify_certificates: Whether to verify SSL certificates
            ca_cert_path: Path to CA certificate file
            client_cert_path: Path to client certificate file
            client_key_path: Path to client private key file
            auth_token: Authentication token for Qdrant
            api_key: API key for Qdrant
            environment: Environment type (development/production)
        """
        self.verify_certificates = verify_certificates
        self.ca_cert_path = ca_cert_path
        self.client_cert_path = client_cert_path
        self.client_key_path = client_key_path
        self.auth_token = auth_token
        self.api_key = api_key
        self.environment = environment
        self.enhanced_validation = enhanced_validation and _ENHANCED_VALIDATION_AVAILABLE
        self.certificate_pinning = certificate_pinning or {}
        self.security_policy = security_policy

        # Initialize enhanced validator if available
        self._certificate_validator = None
        if self.enhanced_validation:
            try:
                from ..security.certificate_validator import CertificateSecurityPolicy
                policy = security_policy or CertificateSecurityPolicy(
                    allow_self_signed=(environment == "development"),
                    require_san=(environment == "production"),
                )
                self._certificate_validator = EnhancedCertificateValidator(
                    security_policy=policy,
                    pinned_certificates=self.certificate_pinning,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced certificate validator: {e}")
                self.enhanced_validation = False

    def to_qdrant_config(self) -> dict[str, Any]:
        """Convert to Qdrant client configuration.

        Returns:
            Dict containing Qdrant-compatible SSL configuration
        """
        config = {}

        # Add authentication if provided
        if self.api_key:
            config["api_key"] = self.api_key
        elif self.auth_token:
            # For custom auth implementations
            config["metadata"] = {"authorization": f"Bearer {self.auth_token}"}

        # Add SSL settings
        if not self.verify_certificates:
            config["verify"] = False

        if self.ca_cert_path:
            config["ca_certs"] = self.ca_cert_path

        if self.client_cert_path and self.client_key_path:
            config["cert"] = (self.client_cert_path, self.client_key_path)

        return config

    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with configured settings.

        Returns:
            Configured SSL context
        """
        if self.environment == "development" and not self.verify_certificates:
            # Allow insecure connections in development
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            logger.warning("Using insecure SSL context for development environment")
        elif self.enhanced_validation:
            # Use enhanced secure SSL context
            try:
                context = create_secure_ssl_context(self._certificate_validator)
                logger.info("Using enhanced SSL context with certificate validation")
            except Exception as e:
                logger.warning(f"Failed to create enhanced SSL context: {e}")
                context = self._create_standard_ssl_context()
        else:
            # Standard secure configuration
            context = self._create_standard_ssl_context()

        return context

    def _create_standard_ssl_context(self) -> ssl.SSLContext:
        """Create standard SSL context."""
        context = ssl.create_default_context()

        # Set minimum TLS version for security
        context.minimum_version = ssl.TLSVersion.TLSv1_2

        if self.ca_cert_path:
            context.load_verify_locations(self.ca_cert_path)

        if self.client_cert_path and self.client_key_path:
            context.load_cert_chain(self.client_cert_path, self.client_key_path)

        return context

    def validate_certificate_for_host(self, hostname: str, certificate_chain: list[bytes]) -> bool:
        """Validate certificate chain for specific hostname.

        Args:
            hostname: Target hostname
            certificate_chain: List of certificate bytes

        Returns:
            True if validation passes

        Raises:
            CertificateValidationError: If validation fails
        """
        if not self.enhanced_validation or not self._certificate_validator:
            logger.debug("Enhanced certificate validation not available, skipping")
            return True

        try:
            return self._certificate_validator.validate_certificate_chain(
                hostname=hostname,
                certificate_chain=certificate_chain,
                verify_hostname=self.verify_certificates,
            )
        except Exception as e:
            logger.error(f"Certificate validation failed for {hostname}: {e}")
            raise


class SSLContextManager:
    """Context manager for SSL/TLS configuration and warning suppression.

    This class provides context-aware SSL warning management that suppresses
    warnings only when appropriate (e.g., localhost connections) while
    preserving important security warnings for remote connections.
    """

    def __init__(self):
        """Initialize SSL context manager."""
        self._original_warning_filters = []
        self._suppression_active = False

    @contextlib.contextmanager
    def for_localhost(self) -> AbstractContextManager[None]:
        """Context manager for localhost SSL warning suppression.

        This context manager temporarily suppresses SSL warnings that are
        commonly generated when connecting to localhost Qdrant instances
        during development. Warnings are restored when exiting the context.

        Example:
            ```python
            ssl_manager = SSLContextManager()
            with suppress_qdrant_ssl_warnings():
                # SSL warnings suppressed for localhost connections
                client = QdrantClient("http://localhost:6333")
                collections = client.get_collections()
            # SSL warnings restored
            ```
        """
        logger.debug("Temporarily suppressing SSL warnings for localhost connection")

        # Store current warning filters
        self._original_warning_filters = warnings.filters.copy()
        self._suppression_active = True

        try:
            # Suppress specific SSL-related warnings for localhost
            warnings.filterwarnings(
                "ignore",
                message=".*insecure connection.*",
                category=urllib3.exceptions.InsecureRequestWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*unverified HTTPS request.*",
                category=urllib3.exceptions.InsecureRequestWarning,
            )
            warnings.filterwarnings(
                "ignore", message=".*SSL.*localhost.*", category=UserWarning
            )

            # Temporarily disable urllib3 warnings
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            yield

        finally:
            # Restore original warning filters
            warnings.filters[:] = self._original_warning_filters
            self._suppression_active = False
            logger.debug("Restored SSL warning filters")

    def is_localhost_url(self, url: str) -> bool:
        """Check if URL points to localhost.

        Args:
            url: URL to check

        Returns:
            True if URL points to localhost, False otherwise
        """
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            return hostname in ("localhost", "127.0.0.1", "::1") or (
                hostname and hostname.startswith("127.")
            )
        except Exception:
            return False

    def create_ssl_config(
        self,
        url: str,
        verify_certificates: bool = True,
        auth_token: str | None = None,
        api_key: str | None = None,
        environment: str = "production",
    ) -> SSLConfiguration:
        """Create SSL configuration based on URL and environment.

        Args:
            url: Target URL for the connection
            verify_certificates: Whether to verify SSL certificates
            auth_token: Authentication token
            api_key: API key for authentication
            environment: Environment type (development/production)

        Returns:
            Configured SSL configuration
        """
        is_localhost = self.is_localhost_url(url)

        # Adjust SSL verification based on environment and URL
        if is_localhost and environment == "development":
            # Allow insecure connections to localhost in development
            verify_certificates = False
            logger.debug(
                "SSL verification disabled for localhost in development environment"
            )
        elif not is_localhost:
            # Always verify certificates for remote connections
            verify_certificates = True
            logger.debug("SSL verification enabled for remote connection")

        return SSLConfiguration(
            verify_certificates=verify_certificates,
            auth_token=auth_token,
            api_key=api_key,
            environment=environment,
        )

    def get_qdrant_client_config(
        self, base_config: dict[str, Any], ssl_config: SSLConfiguration
    ) -> dict[str, Any]:
        """Merge SSL configuration with Qdrant client config.

        Args:
            base_config: Base Qdrant client configuration
            ssl_config: SSL configuration to merge

        Returns:
            Merged configuration dictionary
        """
        config = base_config.copy()
        ssl_settings = ssl_config.to_qdrant_config()

        # Merge SSL settings into base config
        config.update(ssl_settings)

        logger.debug(
            "Created Qdrant client config",
            verify_ssl=ssl_config.verify_certificates,
            has_auth=bool(ssl_config.api_key or ssl_config.auth_token),
            environment=ssl_config.environment,
        )

        return config


# Global SSL context manager instance
_ssl_manager = SSLContextManager()


def get_ssl_manager() -> SSLContextManager:
    """Get the global SSL context manager instance.

    Returns:
        Global SSLContextManager instance
    """
    return _ssl_manager


def create_secure_qdrant_config(
    base_config: dict[str, Any],
    url: str,
    environment: str = "production",
    auth_token: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Create secure Qdrant client configuration.

    This is a convenience function that creates appropriate SSL configuration
    based on the target URL and environment.

    Args:
        base_config: Base Qdrant client configuration
        url: Target Qdrant URL
        environment: Environment type (development/production)
        auth_token: Optional authentication token
        api_key: Optional API key

    Returns:
        Secure configuration dictionary for Qdrant client
    """
    ssl_manager = get_ssl_manager()
    ssl_config = ssl_manager.create_ssl_config(
        url=url, environment=environment, auth_token=auth_token, api_key=api_key
    )

    return ssl_manager.get_qdrant_client_config(base_config, ssl_config)


@contextlib.contextmanager
def suppress_qdrant_ssl_warnings():
    """Context manager to suppress SSL warnings during QdrantClient creation.

    This suppresses the specific warning: "Api key is used with an insecure connection"
    and other SSL-related warnings that appear during QdrantClient instantiation.
    """
    # Store original warning filters
    original_filters = warnings.filters.copy()

    try:
        # Suppress specific SSL warnings
        warnings.filterwarnings("ignore", message=".*Api key is used with an insecure connection.*")
        warnings.filterwarnings("ignore", message=".*insecure connection.*")
        warnings.filterwarnings("ignore", message=".*unverified HTTPS request.*")
        warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

        # Temporarily disable urllib3 warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        yield

    finally:
        # Restore original warning filters
        warnings.filters[:] = original_filters
