"""Unit tests for binary security validation utilities."""

import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.python.wqm_cli.cli.binary_security import (
    BinarySecurityError,
    BinaryValidator,
)


class TestBinaryValidator:
    """Test binary validation functionality."""

    @pytest.fixture
    def validator(self):
        """Create BinaryValidator instance."""
        return BinaryValidator()

    @pytest.fixture
    def test_binary(self, tmp_path):
        """Create a test binary file."""
        binary_path = tmp_path / "test_binary"
        binary_path.write_bytes(b"test binary content")
        # Mock file as executable with owner-only permissions
        return binary_path

    def test_compute_checksum_success(self, validator, test_binary):
        """Test successful checksum computation."""
        checksum = validator.compute_checksum(test_binary)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex digest length
        assert checksum.isalnum()

    def test_compute_checksum_nonexistent_file(self, validator, tmp_path):
        """Test checksum computation fails for nonexistent file."""
        nonexistent = tmp_path / "doesnotexist"
        with pytest.raises(BinarySecurityError, match="Binary not found"):
            validator.compute_checksum(nonexistent)

    def test_compute_checksum_directory(self, validator, tmp_path):
        """Test checksum computation fails for directory."""
        with pytest.raises(BinarySecurityError, match="Not a file"):
            validator.compute_checksum(tmp_path)

    def test_compute_checksum_too_large(self, validator, tmp_path):
        """Test checksum computation fails for files exceeding size limit."""
        large_file = tmp_path / "large_file"
        # Create file larger than MAX_BINARY_SIZE
        with open(large_file, "wb") as f:
            f.write(b"x" * (validator.MAX_BINARY_SIZE + 1))

        with pytest.raises(BinarySecurityError, match="Binary too large"):
            validator.compute_checksum(large_file)

    def test_store_and_load_checksum(self, validator, test_binary):
        """Test checksum storage and retrieval."""
        # Compute and store checksum
        checksum = validator.compute_checksum(test_binary)
        checksum_path = validator.store_checksum(test_binary, checksum)

        assert checksum_path.exists()
        assert checksum_path.name == f"{test_binary.name}.sha256"

        # Load stored checksum
        loaded_checksum = validator.load_checksum(test_binary)
        assert loaded_checksum == checksum

    def test_store_checksum_auto_compute(self, validator, test_binary):
        """Test checksum storage with automatic computation."""
        checksum_path = validator.store_checksum(test_binary)
        assert checksum_path.exists()

        # Verify stored checksum matches computed checksum
        loaded = validator.load_checksum(test_binary)
        computed = validator.compute_checksum(test_binary)
        assert loaded == computed

    def test_load_checksum_missing_file(self, validator, test_binary):
        """Test loading checksum when file doesn't exist."""
        checksum = validator.load_checksum(test_binary)
        assert checksum is None

    def test_verify_checksum_success(self, validator, test_binary):
        """Test successful checksum verification."""
        # Store checksum
        validator.store_checksum(test_binary)

        # Verify should pass
        assert validator.verify_checksum(test_binary) is True

    def test_verify_checksum_tampered_binary(self, validator, test_binary):
        """Test checksum verification fails for tampered binary."""
        # Store original checksum
        validator.store_checksum(test_binary)

        # Tamper with binary
        test_binary.write_bytes(b"tampered content")

        # Verification should fail
        assert validator.verify_checksum(test_binary) is False

    def test_verify_checksum_no_stored_checksum(self, validator, test_binary):
        """Test checksum verification without stored checksum."""
        # No checksum file exists
        assert validator.verify_checksum(test_binary) is False

    def test_validate_ownership_success(self, validator, test_binary):
        """Test successful ownership validation."""
        # Mock file stat to simulate proper ownership (owner-only read/write/execute)
        with patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = stat.S_IFREG | stat.S_IRWXU  # 0o100700
            valid, message = validator.validate_ownership(test_binary)
            assert valid is True
            assert "valid" in message.lower()

    def test_validate_ownership_world_writable(self, validator, test_binary):
        """Test ownership validation fails for world-writable binary."""
        # Mock file stat to simulate world-writable permissions
        with patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = stat.S_IFREG | 0o777
            valid, message = validator.validate_ownership(test_binary)
            assert valid is False
            assert "world-writable" in message.lower()

    def test_validate_ownership_group_writable(self, validator, test_binary):
        """Test ownership validation fails for group-writable binary."""
        # Mock file stat to simulate group-writable permissions
        with patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = stat.S_IFREG | 0o770
            valid, message = validator.validate_ownership(test_binary)
            assert valid is False
            assert "group-writable" in message.lower()

    def test_validate_ownership_not_executable(self, validator, test_binary):
        """Test ownership validation fails for non-executable binary."""
        # Mock file stat to simulate non-executable permissions
        with patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = stat.S_IFREG | 0o600
            valid, message = validator.validate_ownership(test_binary)
            assert valid is False
            assert "not executable" in message.lower()

    def test_validate_binary_comprehensive_success(self, validator, test_binary):
        """Test comprehensive binary validation succeeds."""
        # Store checksum
        validator.store_checksum(test_binary)

        # Mock file stat to simulate proper ownership
        with patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = stat.S_IFREG | stat.S_IRWXU  # 0o100700
            # Validate
            result = validator.validate_binary(test_binary)

            assert result["valid"] is True
            assert result["checks"]["existence"] is True
            assert result["checks"]["is_file"] is True
            assert result["checks"]["ownership"] is True
            assert result["checks"]["checksum"] is True
            assert result["checks"]["executable"] is True
            assert len(result["errors"]) == 0

    def test_validate_binary_missing_file(self, validator, tmp_path):
        """Test binary validation fails for missing file."""
        nonexistent = tmp_path / "missing"

        result = validator.validate_binary(nonexistent)

        assert result["valid"] is False
        assert result["checks"]["existence"] is False
        assert len(result["errors"]) > 0
        assert "not found" in result["errors"][0].lower()

    def test_validate_binary_no_checksum(self, validator, test_binary):
        """Test binary validation with missing checksum."""
        result = validator.validate_binary(test_binary, verify_checksum=True)

        assert result["valid"] is False
        assert result["checks"]["checksum"] is False
        assert any("checksum" in error.lower() for error in result["errors"])

    def test_validate_binary_skip_checksum(self, validator, test_binary):
        """Test binary validation skipping checksum verification."""
        result = validator.validate_binary(test_binary, verify_checksum=False)

        assert result["valid"] is True
        assert "checksum" not in result["checks"]

    def test_validate_binary_skip_ownership(self, validator, test_binary):
        """Test binary validation skipping ownership checks."""
        # Mock file stat to simulate world-writable permissions (would fail strict check)
        with patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = stat.S_IFREG | 0o777

            result = validator.validate_binary(
                test_binary,
                strict_ownership=False,
                verify_checksum=False  # Also skip checksum for this test
            )

            assert result["valid"] is True  # Should pass without ownership/checksum checks
            assert "ownership" not in result["checks"]
            assert "checksum" not in result["checks"]

    def test_checksum_deterministic(self, validator, test_binary):
        """Test checksum computation is deterministic."""
        checksum1 = validator.compute_checksum(test_binary)
        checksum2 = validator.compute_checksum(test_binary)

        assert checksum1 == checksum2

    def test_checksum_different_for_different_content(self, validator, tmp_path):
        """Test different content produces different checksums."""
        file1 = tmp_path / "file1"
        file2 = tmp_path / "file2"

        file1.write_bytes(b"content1")
        file2.write_bytes(b"content2")

        checksum1 = validator.compute_checksum(file1)
        checksum2 = validator.compute_checksum(file2)

        assert checksum1 != checksum2

    def test_checksum_file_permissions(self, validator, test_binary):
        """Test checksum file has restricted permissions."""
        checksum_path = validator.store_checksum(test_binary)

        # Check permissions are owner-only (0o600)
        mode = checksum_path.stat().st_mode
        assert not (mode & stat.S_IRGRP)  # No group read
        assert not (mode & stat.S_IWGRP)  # No group write
        assert not (mode & stat.S_IROTH)  # No other read
        assert not (mode & stat.S_IWOTH)  # No other write

    def test_binary_validator_error_inheritance(self):
        """Test BinarySecurityError is proper Exception subclass."""
        assert issubclass(BinarySecurityError, Exception)

        # Can be raised and caught
        with pytest.raises(BinarySecurityError):
            raise BinarySecurityError("test error")


class TestRealWorldScenarios:
    """Test real-world binary validation scenarios."""

    @pytest.fixture
    def validator(self):
        return BinaryValidator()

    def test_binary_installation_workflow(self, validator, tmp_path):
        """Test complete binary installation and validation workflow."""
        # Simulate building a binary
        binary_path = tmp_path / "memexd"
        binary_path.write_bytes(b"compiled daemon binary")

        # Mock file stat to simulate proper ownership
        with patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = stat.S_IFREG | 0o700

            # Step 1: Compute and store checksum after build
            checksum = validator.compute_checksum(binary_path)
            validator.store_checksum(binary_path, checksum)

            # Step 2: Validate binary before first use
            result = validator.validate_binary(binary_path)
            assert result["valid"] is True

        # Step 3: Detect tampering
        binary_path.write_bytes(b"malicious code injected")

        # Mock file stat again for tampered binary validation
        with patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = stat.S_IFREG | 0o700
            # Validation should fail
            result = validator.validate_binary(binary_path)
            assert result["valid"] is False
            assert result["checks"]["checksum"] is False

    def test_detect_malicious_replacement(self, validator, tmp_path):
        """Test detection of malicious binary replacement attack."""
        # Original legitimate binary
        legitimate_binary = tmp_path / "memexd"
        legitimate_binary.write_bytes(b"legitimate daemon")

        # Mock file stat to simulate proper ownership
        with patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = stat.S_IFREG | 0o700
            validator.store_checksum(legitimate_binary)

        # Attacker replaces binary (same path, different content)
        legitimate_binary.write_bytes(b"malicious replacement")

        # Mock file stat again for validation
        with patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = stat.S_IFREG | 0o700
            # Validation should fail due to checksum mismatch
            result = validator.validate_binary(legitimate_binary)
            assert result["valid"] is False
            assert not result["checks"]["checksum"]

    def test_detect_privilege_escalation_attempt(self, validator, tmp_path):
        """Test detection of privilege escalation via world-writable binary."""
        binary_path = tmp_path / "memexd"
        binary_path.write_bytes(b"daemon binary")

        # Mock file stat to simulate world-writable permissions (attacker scenario)
        with patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = stat.S_IFREG | 0o777

            # Validation should fail due to insecure permissions
            result = validator.validate_binary(binary_path)
            assert result["valid"] is False
            assert not result["checks"]["ownership"]

    def test_upgrade_workflow(self, validator, tmp_path):
        """Test binary upgrade workflow with checksum update."""
        binary_path = tmp_path / "memexd"

        # Install v1
        binary_path.write_bytes(b"daemon v1.0")
        checksum_v1 = validator.compute_checksum(binary_path)
        validator.store_checksum(binary_path, checksum_v1)

        # Validate v1
        assert validator.verify_checksum(binary_path) is True

        # Upgrade to v2 (different content)
        binary_path.write_bytes(b"daemon v2.0")
        checksum_v2 = validator.compute_checksum(binary_path)
        validator.store_checksum(binary_path, checksum_v2)

        # New checksum should be different
        assert checksum_v1 != checksum_v2

        # Validation should pass with new checksum
        assert validator.verify_checksum(binary_path) is True
