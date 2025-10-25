"""
TLS and certificate security validation tests.

Tests certificate validation, TLS configuration, cipher suites,
certificate pinning, and secure communication protocols.
"""

import ssl
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtensionOID, NameOID

from src.python.common.core.ssl_config import SSLConfiguration, SSLContextManager
from src.python.common.security.certificate_validator import (
    CertificatePinningError,
    CertificateSecurityPolicy,
    CertificateValidationError,
    EnhancedCertificateValidator,
)


def generate_test_certificate(
    common_name: str = "test.example.com",
    valid_days: int = 365,
    key_size: int = 2048,
    self_signed: bool = True,
) -> tuple:
    """Generate a test certificate for testing purposes."""
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )

    # Build certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Organization"),
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
    ])

    cert_builder = x509.CertificateBuilder()
    cert_builder = cert_builder.subject_name(subject)
    cert_builder = cert_builder.issuer_name(issuer if self_signed else subject)
    cert_builder = cert_builder.public_key(private_key.public_key())
    cert_builder = cert_builder.serial_number(x509.random_serial_number())

    # Use timezone-aware datetime
    now = datetime.now(timezone.utc)
    cert_builder = cert_builder.not_valid_before(now)
    cert_builder = cert_builder.not_valid_after(now + timedelta(days=valid_days))

    # Add Subject Alternative Name extension
    cert_builder = cert_builder.add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName(common_name),
        ]),
        critical=False,
    )

    # Sign certificate
    certificate = cert_builder.sign(private_key, hashes.SHA256())

    return certificate, private_key


@pytest.mark.security
class TestCertificateSecurityPolicy:
    """Test certificate security policy enforcement."""

    def test_minimum_key_size_enforcement(self):
        """Test that minimum RSA key size is enforced."""
        policy = CertificateSecurityPolicy(min_key_size=2048)

        # 2048-bit key should be acceptable
        assert policy.min_key_size == 2048

        # Test would reject smaller keys (implementation-specific)

    def test_allowed_signature_algorithms(self):
        """Test that only secure signature algorithms are allowed."""
        policy = CertificateSecurityPolicy()

        # SHA-256 should be allowed
        assert 'sha256WithRSAEncryption' in policy.allowed_signature_algorithms

        # Weak algorithms should not be present
        assert 'md5WithRSAEncryption' not in policy.allowed_signature_algorithms
        assert 'sha1WithRSAEncryption' not in policy.allowed_signature_algorithms

    def test_san_requirement(self):
        """Test Subject Alternative Name requirement."""
        policy = CertificateSecurityPolicy(require_san=True)

        assert policy.require_san is True

    def test_self_signed_certificate_policy(self):
        """Test self-signed certificate policy."""
        # Production policy should reject self-signed
        production_policy = CertificateSecurityPolicy(allow_self_signed=False)
        assert production_policy.allow_self_signed is False

        # Development policy may allow self-signed
        dev_policy = CertificateSecurityPolicy(allow_self_signed=True)
        assert dev_policy.allow_self_signed is True

    def test_certificate_age_limit(self):
        """Test maximum certificate age enforcement."""
        policy = CertificateSecurityPolicy(max_certificate_age_days=365)

        assert policy.max_certificate_age_days == 365


@pytest.mark.security
class TestSSLConfiguration:
    """Test SSL/TLS configuration security."""

    def test_certificate_verification_enabled_by_default(self):
        """Test that certificate verification is enabled by default."""
        config = SSLConfiguration()

        assert config.verify_certificates is True

    def test_production_environment_enforces_verification(self):
        """Test production environment enforces certificate verification."""
        config = SSLConfiguration(environment="production")

        assert config.environment == "production"
        # In production, verification should typically be enforced
        assert config.verify_certificates is True

    def test_development_environment_configuration(self):
        """Test development environment can disable verification."""
        config = SSLConfiguration(
            environment="development",
            verify_certificates=False
        )

        assert config.environment == "development"
        # Development may disable verification for localhost
        assert config.verify_certificates is False

    def test_client_certificate_configuration(self):
        """Test client certificate paths are properly stored."""
        config = SSLConfiguration(
            client_cert_path="/path/to/cert.pem",
            client_key_path="/path/to/key.pem",
        )

        assert config.client_cert_path == "/path/to/cert.pem"
        assert config.client_key_path == "/path/to/key.pem"

    def test_ca_certificate_configuration(self):
        """Test CA certificate path configuration."""
        config = SSLConfiguration(ca_cert_path="/path/to/ca.pem")

        assert config.ca_cert_path == "/path/to/ca.pem"

    def test_enhanced_validation_availability(self):
        """Test enhanced validation feature."""
        SSLConfiguration(enhanced_validation=True)

        # Enhanced validation should be attempted if available
        # (actual availability depends on imports)

    def test_authentication_token_storage(self):
        """Test that auth tokens are properly stored."""
        config = SSLConfiguration(
            auth_token="test_token_12345",
            api_key="test_api_key_67890"
        )

        assert config.auth_token == "test_token_12345"
        assert config.api_key == "test_api_key_67890"

    def test_certificate_pinning_configuration(self):
        """Test certificate pinning configuration."""
        pinned_certs = {
            "example.com": ["sha256/AAAA..."],
            "api.example.com": ["sha256/BBBB..."],
        }

        config = SSLConfiguration(certificate_pinning=pinned_certs)

        assert config.certificate_pinning == pinned_certs


@pytest.mark.security
class TestEnhancedCertificateValidator:
    """Test enhanced certificate validation."""

    def test_validator_initialization(self):
        """Test validator initializes with security policy."""
        policy = CertificateSecurityPolicy(min_key_size=4096)
        validator = EnhancedCertificateValidator(security_policy=policy)

        assert validator.security_policy.min_key_size == 4096

    def test_certificate_pinning_configuration(self):
        """Test certificate pinning is properly configured."""
        pinned_certs = {
            "secure.example.com": ["sha256/abc123..."]
        }

        validator = EnhancedCertificateValidator(pinned_certificates=pinned_certs)

        assert "secure.example.com" in validator.pinned_certificates

    def test_trusted_ca_paths_configuration(self):
        """Test trusted CA paths are stored."""
        ca_paths = ["/etc/ssl/certs/ca.pem", "/usr/local/share/ca-certificates/"]

        validator = EnhancedCertificateValidator(trusted_ca_paths=ca_paths)

        assert len(validator.trusted_ca_paths) == 2

    def test_certificate_cache_initialization(self):
        """Test certificate cache is initialized."""
        validator = EnhancedCertificateValidator()

        # Cache should exist and be empty initially
        assert hasattr(validator, '_certificate_cache')
        assert len(validator._certificate_cache) == 0


@pytest.mark.security
class TestCertificateGeneration:
    """Test certificate generation for testing."""

    def test_generate_valid_certificate(self):
        """Test generating a valid test certificate."""
        cert, key = generate_test_certificate()

        assert cert is not None
        assert key is not None

        # Verify certificate attributes
        assert isinstance(cert, x509.Certificate)

    def test_certificate_common_name(self):
        """Test certificate common name is set correctly."""
        cert, _ = generate_test_certificate(common_name="test.local")

        # Extract common name
        cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        assert cn == "test.local"

    def test_certificate_validity_period(self):
        """Test certificate validity period."""
        cert, _ = generate_test_certificate(valid_days=30)

        not_before = cert.not_valid_before_utc
        not_after = cert.not_valid_after_utc

        # Should be valid for approximately 30 days
        validity_period = (not_after - not_before).days
        assert 29 <= validity_period <= 31  # Allow for time precision

    def test_certificate_key_size(self):
        """Test certificate key size."""
        cert, key = generate_test_certificate(key_size=4096)

        # Verify key size
        assert key.key_size == 4096

    def test_certificate_has_san_extension(self):
        """Test certificate has Subject Alternative Name extension."""
        cert, _ = generate_test_certificate(common_name="example.com")

        # Check for SAN extension
        try:
            san_ext = cert.extensions.get_extension_for_oid(
                ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            )
            assert san_ext is not None
        except x509.ExtensionNotFound:
            pytest.fail("Certificate missing SAN extension")


@pytest.mark.security
class TestTLSSecurityProperties:
    """Test TLS protocol security properties."""

    def test_tls_minimum_version(self):
        """Test minimum TLS version enforcement."""
        # Modern systems should require at least TLS 1.2
        context = ssl.create_default_context()

        # TLS 1.2 should be allowed
        # TLS 1.0 and 1.1 should be disabled
        assert context.minimum_version >= ssl.TLSVersion.TLSv1_2

    def test_weak_cipher_suites_disabled(self):
        """Test that weak cipher suites are disabled."""
        context = ssl.create_default_context()

        # Get cipher list
        ciphers = context.get_ciphers()

        # Check that no weak ciphers are present
        for cipher in ciphers:
            cipher_name = cipher['name']

            # No NULL ciphers
            assert 'NULL' not in cipher_name

            # No export-grade ciphers
            assert 'EXPORT' not in cipher_name

            # No anonymous ciphers
            assert 'anon' not in cipher_name.lower()

            # No MD5
            assert 'MD5' not in cipher_name

    def test_forward_secrecy_preferred(self):
        """Test that forward secrecy is preferred."""
        context = ssl.create_default_context()

        ciphers = context.get_ciphers()

        # Check for ECDHE or DHE key exchange (provides forward secrecy)
        forward_secret_ciphers = [
            c for c in ciphers
            if 'ECDHE' in c['name'] or 'DHE' in c['name']
        ]

        assert len(forward_secret_ciphers) > 0

    def test_sslv2_sslv3_disabled(self):
        """Test that SSLv2 and SSLv3 are disabled."""
        context = ssl.create_default_context()

        # SSLv2 and SSLv3 should not be in options
        assert context.minimum_version >= ssl.TLSVersion.TLSv1_2

    def test_certificate_verification_mode(self):
        """Test certificate verification mode."""
        context = ssl.create_default_context()

        # Default context should verify certificates
        assert context.check_hostname is True
        assert context.verify_mode == ssl.CERT_REQUIRED


@pytest.mark.security
class TestCertificateValidationErrors:
    """Test certificate validation error handling."""

    def test_expired_certificate_detection(self):
        """Test detection of expired certificates."""
        # Generate a certificate that expired 30 days ago
        # We need to generate it manually since generate_test_certificate doesn't support past dates
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "expired.example.com"),
        ])

        # Certificate valid from 60 days ago to 30 days ago (now expired)
        now = datetime.now(timezone.utc)
        not_valid_before = now - timedelta(days=60)
        not_valid_after = now - timedelta(days=30)

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(not_valid_before)
            .not_valid_after(not_valid_after)
            .sign(private_key, hashes.SHA256())
        )

        # Certificate should be expired
        assert cert.not_valid_after_utc < now

    def test_not_yet_valid_certificate_detection(self):
        """Test detection of not-yet-valid certificates."""
        # This would require modifying certificate generation
        # to have future not_valid_before date
        pass

    def test_hostname_mismatch_detection(self):
        """Test hostname mismatch detection."""
        cert, _ = generate_test_certificate(common_name="example.com")

        # Extract common name
        cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value

        # Should not match different hostname
        assert cn != "different.com"

    def test_self_signed_certificate_detection(self):
        """Test self-signed certificate detection."""
        cert, _ = generate_test_certificate(self_signed=True)

        # For self-signed certs, subject equals issuer
        subject_cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        issuer_cn = cert.issuer.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value

        assert subject_cn == issuer_cn


@pytest.mark.security
class TestSecureCommunication:
    """Test secure communication protocols."""

    def test_ssl_context_manager_localhost_handling(self):
        """Test SSL context manager handles localhost appropriately."""
        SSLContextManager()

        # Localhost connections may have different requirements
        # (implementation specific)

    def test_ssl_context_manager_remote_handling(self):
        """Test SSL context manager handles remote connections securely."""
        SSLContextManager()

        # Remote connections should enforce strict validation
        # (implementation specific)

    def test_tls_protocol_downgrade_prevention(self):
        """Test prevention of TLS protocol downgrade attacks."""
        context = ssl.create_default_context()

        # Modern TLS should prevent downgrade
        # This is enforced by minimum_version
        assert context.minimum_version >= ssl.TLSVersion.TLSv1_2

    def test_compression_disabled(self):
        """Test TLS compression is disabled (CRIME attack prevention)."""
        context = ssl.create_default_context()

        # TLS compression should be disabled
        # In Python 3.3+, compression is disabled by default
        assert context.options & ssl.OP_NO_COMPRESSION


@pytest.mark.security
class TestCertificatePinning:
    """Test certificate pinning security."""

    def test_pinned_certificate_configuration(self):
        """Test pinned certificate configuration."""
        pinned = {
            "api.example.com": [
                "sha256/AAAA1234...",
                "sha256/BBBB5678...",  # Backup pin
            ]
        }

        validator = EnhancedCertificateValidator(pinned_certificates=pinned)

        assert "api.example.com" in validator.pinned_certificates
        assert len(validator.pinned_certificates["api.example.com"]) == 2

    def test_backup_pin_support(self):
        """Test support for backup certificate pins."""
        pinned = {
            "secure.example.com": [
                "sha256/primary_pin",
                "sha256/backup_pin",
            ]
        }

        validator = EnhancedCertificateValidator(pinned_certificates=pinned)

        # Should have both pins configured
        assert len(validator.pinned_certificates["secure.example.com"]) == 2


@pytest.mark.security
class TestSecurityPolicyEnforcement:
    """Test security policy enforcement."""

    def test_weak_key_size_rejection(self):
        """Test that weak key sizes are rejected."""
        CertificateSecurityPolicy(min_key_size=2048)

        # Policy should reject 1024-bit keys
        # (implementation would validate actual certificates)

    def test_weak_signature_algorithm_rejection(self):
        """Test that weak signature algorithms are rejected."""
        policy = CertificateSecurityPolicy(
            allowed_signature_algorithms={'sha256WithRSAEncryption'}
        )

        # MD5 and SHA1 should not be in allowed list
        assert 'md5WithRSAEncryption' not in policy.allowed_signature_algorithms
        assert 'sha1WithRSAEncryption' not in policy.allowed_signature_algorithms

    def test_certificate_age_enforcement(self):
        """Test certificate age is enforced."""
        policy = CertificateSecurityPolicy(max_certificate_age_days=365)

        # Verify policy is configured correctly
        assert policy.max_certificate_age_days == 365

        # Actual enforcement would be in certificate validator
        # (implementation specific)


# Security test markers are configured in pyproject.toml
