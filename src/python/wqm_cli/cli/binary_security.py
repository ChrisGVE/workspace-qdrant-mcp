"""Binary security utilities for service installation.

This module provides cryptographic verification and permission validation for the memexd
daemon binary to prevent malicious binary substitution attacks.

Security Note: Addresses SECURITY_AUDIT.md finding about service installation vulnerabilities.
"""

import hashlib
import json
import os
import stat
from pathlib import Path

from loguru import logger


class BinarySecurityError(Exception):
    """Raised when binary security validation fails."""
    pass


class BinaryValidator:
    """Validates binary integrity, ownership, and permissions."""

    CHECKSUM_ALGORITHM = "sha256"
    CHECKSUM_FILE_SUFFIX = ".sha256"
    MAX_BINARY_SIZE = 500 * 1024 * 1024  # 500MB safety limit

    def __init__(self):
        self.current_uid = os.getuid() if hasattr(os, 'getuid') else None

    def compute_checksum(self, binary_path: Path) -> str:
        """Compute SHA-256 checksum of a binary file.

        Args:
            binary_path: Path to binary file

        Returns:
            Hexadecimal SHA-256 checksum

        Raises:
            BinarySecurityError: If file cannot be read or is too large
        """
        if not binary_path.exists():
            raise BinarySecurityError(f"Binary not found: {binary_path}")

        if not binary_path.is_file():
            raise BinarySecurityError(f"Not a file: {binary_path}")

        # Safety check: reject unreasonably large files
        file_size = binary_path.stat().st_size
        if file_size > self.MAX_BINARY_SIZE:
            raise BinarySecurityError(
                f"Binary too large ({file_size} bytes). Maximum: {self.MAX_BINARY_SIZE}"
            )

        logger.debug("Computing checksum", binary=str(binary_path), algorithm=self.CHECKSUM_ALGORITHM)

        hash_obj = hashlib.sha256()
        try:
            with open(binary_path, "rb") as f:
                # Read in chunks for memory efficiency
                while chunk := f.read(8192):
                    hash_obj.update(chunk)
        except OSError as e:
            raise BinarySecurityError(f"Failed to read binary for checksum: {e}") from e

        checksum = hash_obj.hexdigest()
        logger.debug("Checksum computed", checksum=checksum, size=file_size)
        return checksum

    def store_checksum(self, binary_path: Path, checksum: str | None = None) -> Path:
        """Store binary checksum in sidecar file.

        Args:
            binary_path: Path to binary file
            checksum: Pre-computed checksum (will compute if not provided)

        Returns:
            Path to checksum file

        Raises:
            BinarySecurityError: If checksum cannot be stored
        """
        if checksum is None:
            checksum = self.compute_checksum(binary_path)

        checksum_path = Path(str(binary_path) + self.CHECKSUM_FILE_SUFFIX)

        try:
            # Write checksum with metadata
            checksum_data = {
                "algorithm": self.CHECKSUM_ALGORITHM,
                "checksum": checksum,
                "binary_path": str(binary_path.resolve()),
                "binary_size": binary_path.stat().st_size
            }

            with open(checksum_path, "w") as f:
                json.dump(checksum_data, f, indent=2)

            # Restrict checksum file permissions (owner read/write only)
            if hasattr(os, 'chmod'):
                os.chmod(checksum_path, 0o600)

            logger.info("Stored binary checksum", checksum_file=str(checksum_path))
            return checksum_path

        except OSError as e:
            raise BinarySecurityError(f"Failed to store checksum: {e}") from e

    def load_checksum(self, binary_path: Path) -> str | None:
        """Load stored checksum for a binary.

        Args:
            binary_path: Path to binary file

        Returns:
            Stored checksum or None if not found
        """
        checksum_path = Path(str(binary_path) + self.CHECKSUM_FILE_SUFFIX)

        if not checksum_path.exists():
            logger.debug("No checksum file found", checksum_file=str(checksum_path))
            return None

        try:
            with open(checksum_path) as f:
                checksum_data = json.load(f)

            if checksum_data.get("algorithm") != self.CHECKSUM_ALGORITHM:
                logger.warning(
                    "Checksum algorithm mismatch",
                    expected=self.CHECKSUM_ALGORITHM,
                    found=checksum_data.get("algorithm")
                )
                return None

            return checksum_data.get("checksum")

        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load checksum", error=str(e))
            return None

    def verify_checksum(self, binary_path: Path, expected_checksum: str | None = None) -> bool:
        """Verify binary checksum matches expected value.

        Args:
            binary_path: Path to binary file
            expected_checksum: Expected checksum (will load from file if not provided)

        Returns:
            True if checksum matches, False otherwise

        Raises:
            BinarySecurityError: If checksum cannot be computed
        """
        if expected_checksum is None:
            expected_checksum = self.load_checksum(binary_path)

        if expected_checksum is None:
            logger.warning("No checksum available for verification", binary=str(binary_path))
            return False

        actual_checksum = self.compute_checksum(binary_path)

        if actual_checksum != expected_checksum:
            logger.error(
                "Checksum mismatch - binary may be compromised",
                binary=str(binary_path),
                expected=expected_checksum,
                actual=actual_checksum
            )
            return False

        logger.debug("Checksum verification passed", binary=str(binary_path))
        return True

    def validate_ownership(self, binary_path: Path) -> tuple[bool, str]:
        """Validate binary is owned by current user and not world-writable.

        Args:
            binary_path: Path to binary file

        Returns:
            Tuple of (is_valid, reason)
        """
        if not binary_path.exists():
            return False, f"Binary not found: {binary_path}"

        try:
            st = binary_path.stat()

            # Check ownership (Unix only)
            if self.current_uid is not None:
                if st.st_uid != self.current_uid:
                    return False, (
                        f"Binary owned by UID {st.st_uid}, expected {self.current_uid}. "
                        "Binary may have been replaced by another user."
                    )

            # Check permissions are not world-writable
            mode = st.st_mode
            if mode & stat.S_IWOTH:
                return False, (
                    f"Binary is world-writable (permissions: {oct(mode)}). "
                    "This allows any user to replace the binary."
                )

            # Check permissions are not group-writable (stricter security)
            if mode & stat.S_IWGRP:
                return False, (
                    f"Binary is group-writable (permissions: {oct(mode)}). "
                    "This allows group members to replace the binary."
                )

            # Verify executable permission
            if not (mode & stat.S_IXUSR):
                return False, f"Binary is not executable by owner (permissions: {oct(mode)})"

            logger.debug(
                "Ownership validation passed",
                binary=str(binary_path),
                uid=st.st_uid if hasattr(st, 'st_uid') else None,
                mode=oct(mode)
            )
            return True, "Ownership and permissions valid"

        except (OSError, AttributeError) as e:
            return False, f"Failed to validate ownership: {e}"

    def validate_binary(
        self,
        binary_path: Path,
        verify_checksum: bool = True,
        strict_ownership: bool = True
    ) -> dict[str, any]:
        """Comprehensive binary security validation.

        Args:
            binary_path: Path to binary file
            verify_checksum: Whether to verify checksum (default: True)
            strict_ownership: Whether to enforce ownership checks (default: True)

        Returns:
            Dict with validation results:
                - valid: bool - Overall validation result
                - checks: Dict[str, bool] - Individual check results
                - errors: List[str] - Validation errors
                - warnings: List[str] - Validation warnings
        """
        result = {
            "valid": True,
            "checks": {},
            "errors": [],
            "warnings": []
        }

        # Check 1: File exists and is a regular file
        if not binary_path.exists():
            result["valid"] = False
            result["errors"].append(f"Binary not found: {binary_path}")
            result["checks"]["existence"] = False
            return result

        if not binary_path.is_file():
            result["valid"] = False
            result["errors"].append(f"Not a regular file: {binary_path}")
            result["checks"]["is_file"] = False
            return result

        result["checks"]["existence"] = True
        result["checks"]["is_file"] = True

        # Check 2: Ownership and permissions
        if strict_ownership:
            ownership_valid, ownership_msg = self.validate_ownership(binary_path)
            result["checks"]["ownership"] = ownership_valid

            if not ownership_valid:
                result["valid"] = False
                result["errors"].append(ownership_msg)

        # Check 3: Checksum verification
        if verify_checksum:
            try:
                checksum_valid = self.verify_checksum(binary_path)
                result["checks"]["checksum"] = checksum_valid

                if not checksum_valid:
                    result["valid"] = False
                    result["errors"].append(
                        "Checksum verification failed. Binary may have been tampered with or "
                        "checksum file is missing. Run 'wqm service install --build' to rebuild."
                    )
            except BinarySecurityError as e:
                result["valid"] = False
                result["errors"].append(f"Checksum verification error: {e}")
                result["checks"]["checksum"] = False

        # Check 4: Executable permission
        if not os.access(binary_path, os.X_OK):
            result["valid"] = False
            result["errors"].append(f"Binary is not executable: {binary_path}")
            result["checks"]["executable"] = False
        else:
            result["checks"]["executable"] = True

        # Log validation result
        if result["valid"]:
            logger.info("Binary security validation passed", binary=str(binary_path))
        else:
            logger.error(
                "Binary security validation failed",
                binary=str(binary_path),
                errors=result["errors"]
            )

        return result


# Export main validator class
__all__ = ["BinaryValidator", "BinarySecurityError"]
