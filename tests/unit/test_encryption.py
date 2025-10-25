"""
Unit tests for the encryption and secure communication system.

This test suite provides comprehensive coverage of:
- Key management and rotation functionality
- Multiple encryption algorithms (AES-GCM, AES-CBC, ChaCha20-Poly1305, RSA)
- Key derivation functions (PBKDF2, Scrypt, HKDF)
- Message authentication and integrity verification
- Secure communication channels and certificate management
- Error conditions, edge cases, and security validations
"""

import base64
import json
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip tests if cryptography is not available
try:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

from src.python.common.security.encryption import (
    CryptoError,
    DecryptionError,
    EncryptedData,
    EncryptionAlgorithm,
    EncryptionEngine,
    EncryptionError,
    EncryptionKey,
    HashAlgorithm,
    KeyDerivationFunction,
    KeyManagementError,
    KeyManager,
    SecureChannel,
    SecureCommunication,
    constant_time_compare,
    decrypt_json,
    encrypt_json,
    generate_secure_token,
    get_encryption_engine,
    get_key_manager,
    get_secure_communication,
    secure_temp_key,
)


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography library not available")
class TestEncryptionKey:
    """Test EncryptionKey data class functionality."""

    def test_key_creation(self):
        """Test basic key creation."""
        key_data = os.urandom(32)
        key = EncryptionKey(
            key_id="test_key",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_data=key_data,
            max_usage=1000
        )

        assert key.key_id == "test_key"
        assert key.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert key.key_data == key_data
        assert key.max_usage == 1000
        assert key.usage_count == 0
        assert isinstance(key.created_at, datetime)

    def test_key_expiration_false(self):
        """Test key is not expired."""
        key = EncryptionKey(
            key_id="test_key",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_data=os.urandom(32)
        )
        assert not key.is_expired()

    def test_key_expiration_true(self):
        """Test key is expired."""
        past_time = datetime.utcnow() - timedelta(hours=1)
        key = EncryptionKey(
            key_id="test_key",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_data=os.urandom(32),
            expires_at=past_time
        )
        assert key.is_expired()

    def test_key_usage_not_exceeded(self):
        """Test key usage is not exceeded."""
        key = EncryptionKey(
            key_id="test_key",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_data=os.urandom(32),
            max_usage=10,
            usage_count=5
        )
        assert not key.is_usage_exceeded()

    def test_key_usage_exceeded(self):
        """Test key usage is exceeded."""
        key = EncryptionKey(
            key_id="test_key",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_data=os.urandom(32),
            max_usage=10,
            usage_count=10
        )
        assert key.is_usage_exceeded()

    def test_key_can_use_true(self):
        """Test key can be used."""
        key = EncryptionKey(
            key_id="test_key",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_data=os.urandom(32),
            max_usage=10,
            usage_count=5
        )
        assert key.can_use()

    def test_key_can_use_false_expired(self):
        """Test key cannot be used due to expiration."""
        past_time = datetime.utcnow() - timedelta(hours=1)
        key = EncryptionKey(
            key_id="test_key",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_data=os.urandom(32),
            expires_at=past_time
        )
        assert not key.can_use()

    def test_key_can_use_false_usage_exceeded(self):
        """Test key cannot be used due to usage limit."""
        key = EncryptionKey(
            key_id="test_key",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_data=os.urandom(32),
            max_usage=5,
            usage_count=5
        )
        assert not key.can_use()


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography library not available")
class TestEncryptedData:
    """Test EncryptedData data class functionality."""

    def test_encrypted_data_creation(self):
        """Test basic encrypted data creation."""
        data = b"encrypted content"
        iv = os.urandom(12)
        tag = os.urandom(16)

        encrypted = EncryptedData(
            data=data,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_id="test_key",
            iv=iv,
            tag=tag
        )

        assert encrypted.data == data
        assert encrypted.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert encrypted.key_id == "test_key"
        assert encrypted.iv == iv
        assert encrypted.tag == tag
        assert isinstance(encrypted.created_at, datetime)

    def test_encrypted_data_to_dict(self):
        """Test converting encrypted data to dictionary."""
        data = b"encrypted content"
        iv = os.urandom(12)
        tag = os.urandom(16)
        mac = os.urandom(32)
        metadata = {"test": "value"}

        encrypted = EncryptedData(
            data=data,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_id="test_key",
            iv=iv,
            tag=tag,
            mac=mac,
            metadata=metadata
        )

        result = encrypted.to_dict()

        assert result['data'] == base64.b64encode(data).decode('utf-8')
        assert result['algorithm'] == EncryptionAlgorithm.AES_256_GCM.value
        assert result['key_id'] == "test_key"
        assert result['iv'] == base64.b64encode(iv).decode('utf-8')
        assert result['tag'] == base64.b64encode(tag).decode('utf-8')
        assert result['mac'] == base64.b64encode(mac).decode('utf-8')
        assert result['metadata'] == metadata
        assert 'created_at' in result

    def test_encrypted_data_from_dict(self):
        """Test creating encrypted data from dictionary."""
        data = b"encrypted content"
        iv = os.urandom(12)
        tag = os.urandom(16)
        created_at = datetime.utcnow()

        data_dict = {
            'data': base64.b64encode(data).decode('utf-8'),
            'algorithm': EncryptionAlgorithm.AES_256_GCM.value,
            'key_id': 'test_key',
            'iv': base64.b64encode(iv).decode('utf-8'),
            'tag': base64.b64encode(tag).decode('utf-8'),
            'mac': None,
            'metadata': {'test': 'value'},
            'created_at': created_at.isoformat()
        }

        encrypted = EncryptedData.from_dict(data_dict)

        assert encrypted.data == data
        assert encrypted.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert encrypted.key_id == 'test_key'
        assert encrypted.iv == iv
        assert encrypted.tag == tag
        assert encrypted.mac is None
        assert encrypted.metadata == {'test': 'value'}
        assert encrypted.created_at == created_at

    def test_encrypted_data_to_from_dict_roundtrip(self):
        """Test roundtrip conversion to/from dictionary."""
        original = EncryptedData(
            data=b"test data",
            algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
            key_id="roundtrip_key",
            iv=os.urandom(12),
            tag=os.urandom(16),
            metadata={"round": "trip"}
        )

        dict_data = original.to_dict()
        restored = EncryptedData.from_dict(dict_data)

        assert restored.data == original.data
        assert restored.algorithm == original.algorithm
        assert restored.key_id == original.key_id
        assert restored.iv == original.iv
        assert restored.tag == original.tag
        assert restored.metadata == original.metadata


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography library not available")
class TestKeyManager:
    """Test KeyManager functionality."""

    @pytest.fixture
    def key_manager(self):
        """Create KeyManager for testing."""
        return KeyManager()

    def test_key_manager_initialization(self, key_manager):
        """Test key manager initialization."""
        assert key_manager._master_key is not None
        assert len(key_manager._master_key) == 32
        assert key_manager._kdf_iterations == 100000
        assert len(key_manager._kdf_salt) == 32

    def test_generate_aes_key(self, key_manager):
        """Test AES key generation."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

        assert key_id is not None
        key = key_manager.get_key(key_id)
        assert key is not None
        assert key.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert len(key.key_data) == 32
        assert key.public_key is None

    def test_generate_chacha20_key(self, key_manager):
        """Test ChaCha20 key generation."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.CHACHA20_POLY1305)

        key = key_manager.get_key(key_id)
        assert key.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305
        assert len(key.key_data) == 32

    def test_generate_rsa_key(self, key_manager):
        """Test RSA key generation."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.RSA_2048)

        key = key_manager.get_key(key_id)
        assert key.algorithm == EncryptionAlgorithm.RSA_2048
        assert key.public_key is not None
        assert b'-----BEGIN PRIVATE KEY-----' in key.key_data
        assert b'-----BEGIN PUBLIC KEY-----' in key.public_key

    def test_generate_key_with_expiration(self, key_manager):
        """Test key generation with expiration."""
        key_id = key_manager.generate_key(
            EncryptionAlgorithm.AES_256_GCM,
            expires_in_days=1
        )

        key = key_manager.get_key(key_id)
        assert key.expires_at is not None
        expected_time = datetime.utcnow() + timedelta(days=1)
        assert abs((key.expires_at - expected_time).total_seconds()) < 60

    def test_generate_key_with_usage_limit(self, key_manager):
        """Test key generation with usage limit."""
        key_id = key_manager.generate_key(
            EncryptionAlgorithm.AES_256_GCM,
            max_usage=1000
        )

        key = key_manager.get_key(key_id)
        assert key.max_usage == 1000

    def test_generate_key_custom_id(self, key_manager):
        """Test key generation with custom ID."""
        custom_id = "my_custom_key"
        key_id = key_manager.generate_key(
            EncryptionAlgorithm.AES_256_GCM,
            key_id=custom_id
        )

        assert key_id == custom_id
        key = key_manager.get_key(custom_id)
        assert key.key_id == custom_id

    def test_generate_key_duplicate_id(self, key_manager):
        """Test generating key with duplicate ID."""
        key_id = "duplicate_key"
        key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM, key_id=key_id)

        with pytest.raises(KeyManagementError, match="already exists"):
            key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM, key_id=key_id)

    def test_generate_key_unsupported_algorithm(self, key_manager):
        """Test key generation with unsupported algorithm."""
        # Mock an unsupported algorithm by patching
        with patch('src.python.common.security.encryption.EncryptionAlgorithm') as mock_enum:
            mock_enum.INVALID_ALG = "invalid-algorithm"
            with pytest.raises(KeyManagementError, match="Unsupported algorithm"):
                key_manager.generate_key(mock_enum.INVALID_ALG)

    def test_get_key_valid(self, key_manager):
        """Test getting valid key."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
        key = key_manager.get_key(key_id)
        assert key is not None
        assert key.key_id == key_id

    def test_get_key_nonexistent(self, key_manager):
        """Test getting nonexistent key."""
        key = key_manager.get_key("nonexistent")
        assert key is None

    def test_get_key_expired(self, key_manager):
        """Test getting expired key."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

        # Make key expired
        key = key_manager._keys[key_id]
        key.expires_at = datetime.utcnow() - timedelta(hours=1)

        result = key_manager.get_key(key_id)
        assert result is None

    def test_get_key_usage_exceeded(self, key_manager):
        """Test getting key with usage exceeded."""
        key_id = key_manager.generate_key(
            EncryptionAlgorithm.AES_256_GCM,
            max_usage=1
        )

        # Exceed usage
        key = key_manager._keys[key_id]
        key.usage_count = 1

        result = key_manager.get_key(key_id)
        assert result is None

    def test_delete_key_success(self, key_manager):
        """Test successful key deletion."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

        result = key_manager.delete_key(key_id)
        assert result is True
        assert key_manager.get_key(key_id) is None

    def test_delete_key_nonexistent(self, key_manager):
        """Test deleting nonexistent key."""
        result = key_manager.delete_key("nonexistent")
        assert result is False

    def test_rotate_key_success(self, key_manager):
        """Test successful key rotation."""
        old_key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

        # Mock callback
        callback = Mock()
        key_manager.add_rotation_callback(callback)

        new_key_id = key_manager.rotate_key(old_key_id)

        assert new_key_id is not None
        assert new_key_id != old_key_id

        old_key = key_manager.get_key(old_key_id)
        new_key = key_manager.get_key(new_key_id)
        assert old_key is not None
        assert new_key is not None
        assert new_key.algorithm == old_key.algorithm

        callback.assert_called_once_with(old_key_id, new_key_id)

    def test_rotate_key_nonexistent(self, key_manager):
        """Test rotating nonexistent key."""
        with pytest.raises(KeyManagementError, match="not found"):
            key_manager.rotate_key("nonexistent")

    def test_rotate_key_with_callback_exception(self, key_manager):
        """Test key rotation with callback exception."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

        # Add callback that raises exception
        def failing_callback(old_id, new_id):
            raise Exception("Callback error")

        key_manager.add_rotation_callback(failing_callback)

        # Should not raise exception
        new_key_id = key_manager.rotate_key(key_id)
        assert new_key_id is not None

    def test_derive_key_pbkdf2(self, key_manager):
        """Test key derivation using PBKDF2."""
        password = "test_password"
        key_id = key_manager.derive_key(
            password,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            kdf=KeyDerivationFunction.PBKDF2
        )

        key = key_manager.get_key(key_id)
        assert key is not None
        assert key.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert len(key.key_data) == 32
        assert key.metadata['kdf'] == KeyDerivationFunction.PBKDF2.value

    def test_derive_key_scrypt(self, key_manager):
        """Test key derivation using Scrypt."""
        password = "test_password"
        key_id = key_manager.derive_key(
            password,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            kdf=KeyDerivationFunction.SCRYPT
        )

        key = key_manager.get_key(key_id)
        assert key is not None
        assert key.metadata['kdf'] == KeyDerivationFunction.SCRYPT.value

    def test_derive_key_hkdf(self, key_manager):
        """Test key derivation using HKDF."""
        password = "test_password"
        key_id = key_manager.derive_key(
            password,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            kdf=KeyDerivationFunction.HKDF
        )

        key = key_manager.get_key(key_id)
        assert key is not None
        assert key.metadata['kdf'] == KeyDerivationFunction.HKDF.value

    def test_derive_key_with_salt(self, key_manager):
        """Test key derivation with custom salt."""
        password = "test_password"
        salt = os.urandom(32)

        key_id = key_manager.derive_key(password, salt=salt)

        key = key_manager.get_key(key_id)
        assert key is not None
        assert base64.b64decode(key.metadata['salt']) == salt

    def test_derive_key_deterministic(self, key_manager):
        """Test key derivation is deterministic with same inputs."""
        password = "test_password"
        salt = os.urandom(32)

        key_id1 = key_manager.derive_key(password, salt=salt)
        key_id2 = key_manager.derive_key(password, salt=salt)

        key1 = key_manager.get_key(key_id1)
        key2 = key_manager.get_key(key_id2)

        # Keys should have same derived data
        assert key1.key_data == key2.key_data

    def test_export_public_key_rsa(self, key_manager):
        """Test exporting public key for RSA."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.RSA_2048)
        public_key = key_manager.export_public_key(key_id)

        assert public_key is not None
        assert b'-----BEGIN PUBLIC KEY-----' in public_key

    def test_export_public_key_symmetric(self, key_manager):
        """Test exporting public key for symmetric algorithm."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

        with pytest.raises(KeyManagementError, match="Cannot export public key"):
            key_manager.export_public_key(key_id)

    def test_export_public_key_nonexistent(self, key_manager):
        """Test exporting public key for nonexistent key."""
        result = key_manager.export_public_key("nonexistent")
        assert result is None

    def test_cleanup_expired_keys(self, key_manager):
        """Test cleaning up expired keys."""
        # Generate keys with different expiration
        valid_key_id = key_manager.generate_key(
            EncryptionAlgorithm.AES_256_GCM,
            expires_in_days=1
        )
        expired_key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

        # Make one key expired
        expired_key = key_manager._keys[expired_key_id]
        expired_key.expires_at = datetime.utcnow() - timedelta(hours=1)

        count = key_manager.cleanup_expired_keys()

        assert count == 1
        assert key_manager.get_key(valid_key_id) is not None
        assert expired_key_id not in key_manager._keys

    def test_list_keys(self, key_manager):
        """Test listing keys."""
        key_id1 = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
        key_id2 = key_manager.generate_key(EncryptionAlgorithm.RSA_2048)

        keys = key_manager.list_keys()

        assert len(keys) == 2
        key_ids = [key['key_id'] for key in keys]
        assert key_id1 in key_ids
        assert key_id2 in key_ids

        # Check key structure
        key_info = keys[0]
        required_fields = ['key_id', 'algorithm', 'created_at', 'can_use']
        for field in required_fields:
            assert field in key_info


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography library not available")
class TestEncryptionEngine:
    """Test EncryptionEngine functionality."""

    @pytest.fixture
    def key_manager(self):
        """Create KeyManager for testing."""
        return KeyManager()

    @pytest.fixture
    def encryption_engine(self, key_manager):
        """Create EncryptionEngine for testing."""
        return EncryptionEngine(key_manager)

    def test_encryption_engine_initialization(self, encryption_engine):
        """Test encryption engine initialization."""
        assert encryption_engine._key_manager is not None
        assert encryption_engine._lock is not None

    def test_encrypt_decrypt_aes_gcm(self, key_manager, encryption_engine):
        """Test AES-GCM encryption and decryption."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
        plaintext = b"This is a test message for AES-GCM encryption"
        additional_data = b"additional authenticated data"

        # Encrypt
        encrypted_data = encryption_engine.encrypt(plaintext, key_id, additional_data)

        assert encrypted_data.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert encrypted_data.key_id == key_id
        assert encrypted_data.iv is not None
        assert len(encrypted_data.iv) == 12  # GCM IV
        assert encrypted_data.tag is not None
        assert len(encrypted_data.tag) == 16  # GCM tag
        assert encrypted_data.data != plaintext

        # Decrypt
        decrypted_data = encryption_engine.decrypt(encrypted_data, additional_data)
        assert decrypted_data == plaintext

    def test_encrypt_decrypt_aes_cbc(self, key_manager, encryption_engine):
        """Test AES-CBC encryption and decryption."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_CBC)
        plaintext = b"This is a test message for AES-CBC encryption"

        # Encrypt
        encrypted_data = encryption_engine.encrypt(plaintext, key_id)

        assert encrypted_data.algorithm == EncryptionAlgorithm.AES_256_CBC
        assert encrypted_data.iv is not None
        assert len(encrypted_data.iv) == 16  # CBC IV
        assert encrypted_data.mac is not None  # HMAC for integrity
        assert encrypted_data.data != plaintext

        # Decrypt
        decrypted_data = encryption_engine.decrypt(encrypted_data)
        assert decrypted_data == plaintext

    def test_encrypt_decrypt_chacha20_poly1305(self, key_manager, encryption_engine):
        """Test ChaCha20-Poly1305 encryption and decryption."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.CHACHA20_POLY1305)
        plaintext = b"This is a test message for ChaCha20-Poly1305 encryption"
        additional_data = b"additional data for ChaCha20"

        # Encrypt
        encrypted_data = encryption_engine.encrypt(plaintext, key_id, additional_data)

        assert encrypted_data.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305
        assert encrypted_data.iv is not None
        assert len(encrypted_data.iv) == 12  # ChaCha20 nonce
        assert encrypted_data.tag is not None
        assert encrypted_data.data != plaintext

        # Decrypt
        decrypted_data = encryption_engine.decrypt(encrypted_data, additional_data)
        assert decrypted_data == plaintext

    def test_encrypt_decrypt_rsa(self, key_manager, encryption_engine):
        """Test RSA encryption and decryption."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.RSA_2048)
        plaintext = b"Short message for RSA"  # RSA has size limits

        # Encrypt
        encrypted_data = encryption_engine.encrypt(plaintext, key_id)

        assert encrypted_data.algorithm == EncryptionAlgorithm.RSA_2048
        assert encrypted_data.data != plaintext
        assert encrypted_data.iv is None  # RSA doesn't use IV
        assert encrypted_data.tag is None

        # Decrypt
        decrypted_data = encryption_engine.decrypt(encrypted_data)
        assert decrypted_data == plaintext

    def test_encrypt_rsa_data_too_large(self, key_manager, encryption_engine):
        """Test RSA encryption with data too large."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.RSA_2048)
        large_data = b"x" * 1000  # Too large for RSA-2048

        with pytest.raises(EncryptionError, match="Data too large"):
            encryption_engine.encrypt(large_data, key_id)

    def test_encrypt_nonexistent_key(self, encryption_engine):
        """Test encryption with nonexistent key."""
        with pytest.raises(EncryptionError, match="Key .* not found"):
            encryption_engine.encrypt(b"test", "nonexistent_key")

    def test_encrypt_unusable_key(self, key_manager, encryption_engine):
        """Test encryption with unusable key."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM, max_usage=1)

        # Use key once
        key = key_manager._keys[key_id]
        key.usage_count = 1

        with pytest.raises(EncryptionError, match="Key .* not found or cannot be used"):
            encryption_engine.encrypt(b"test", key_id)

    def test_decrypt_nonexistent_key(self, encryption_engine):
        """Test decryption with nonexistent key."""
        fake_encrypted = EncryptedData(
            data=b"fake",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_id="nonexistent",
            iv=os.urandom(12),
            tag=os.urandom(16)
        )

        with pytest.raises(DecryptionError, match="Key .* not found"):
            encryption_engine.decrypt(fake_encrypted)

    def test_decrypt_corrupted_data(self, key_manager, encryption_engine):
        """Test decryption with corrupted data."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
        plaintext = b"test message"

        # Encrypt normally
        encrypted_data = encryption_engine.encrypt(plaintext, key_id)

        # Corrupt the data
        corrupted_data = EncryptedData(
            data=b"corrupted" + encrypted_data.data[9:],
            algorithm=encrypted_data.algorithm,
            key_id=encrypted_data.key_id,
            iv=encrypted_data.iv,
            tag=encrypted_data.tag
        )

        with pytest.raises(DecryptionError):
            encryption_engine.decrypt(corrupted_data)

    def test_decrypt_wrong_additional_data(self, key_manager, encryption_engine):
        """Test decryption with wrong additional data."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
        plaintext = b"test message"
        additional_data = b"correct additional data"

        encrypted_data = encryption_engine.encrypt(plaintext, key_id, additional_data)

        # Try to decrypt with wrong additional data
        with pytest.raises(DecryptionError):
            encryption_engine.decrypt(encrypted_data, b"wrong additional data")

    def test_key_usage_increment(self, key_manager, encryption_engine):
        """Test that encryption increments key usage count."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
        plaintext = b"test message"

        # Initial usage count
        key = key_manager.get_key(key_id)
        initial_count = key.usage_count

        # Encrypt
        encryption_engine.encrypt(plaintext, key_id)

        # Check usage increment
        assert key.usage_count == initial_count + 1

    def test_generate_mac(self, key_manager, encryption_engine):
        """Test MAC generation."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
        data = b"data to authenticate"

        mac = encryption_engine.generate_mac(data, key_id, HashAlgorithm.SHA256)

        assert mac is not None
        assert len(mac) == 32  # SHA256 output length

    def test_verify_mac_success(self, key_manager, encryption_engine):
        """Test successful MAC verification."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
        data = b"data to authenticate"

        mac = encryption_engine.generate_mac(data, key_id, HashAlgorithm.SHA256)
        result = encryption_engine.verify_mac(data, mac, key_id, HashAlgorithm.SHA256)

        assert result is True

    def test_verify_mac_failure(self, key_manager, encryption_engine):
        """Test MAC verification failure."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
        data = b"data to authenticate"
        wrong_data = b"wrong data"

        mac = encryption_engine.generate_mac(data, key_id, HashAlgorithm.SHA256)
        result = encryption_engine.verify_mac(wrong_data, mac, key_id, HashAlgorithm.SHA256)

        assert result is False

    def test_generate_mac_different_algorithms(self, key_manager, encryption_engine):
        """Test MAC generation with different hash algorithms."""
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
        data = b"data to authenticate"

        mac_sha256 = encryption_engine.generate_mac(data, key_id, HashAlgorithm.SHA256)
        mac_sha384 = encryption_engine.generate_mac(data, key_id, HashAlgorithm.SHA384)
        mac_sha512 = encryption_engine.generate_mac(data, key_id, HashAlgorithm.SHA512)

        assert len(mac_sha256) == 32
        assert len(mac_sha384) == 48
        assert len(mac_sha512) == 64

        # MACs should be different
        assert mac_sha256 != mac_sha384
        assert mac_sha384 != mac_sha512

    def test_generate_mac_nonexistent_key(self, encryption_engine):
        """Test MAC generation with nonexistent key."""
        with pytest.raises(EncryptionError, match="Key .* not found"):
            encryption_engine.generate_mac(b"data", "nonexistent", HashAlgorithm.SHA256)


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography library not available")
class TestSecureCommunication:
    """Test SecureCommunication functionality."""

    @pytest.fixture
    def key_manager(self):
        """Create KeyManager for testing."""
        return KeyManager()

    @pytest.fixture
    def encryption_engine(self, key_manager):
        """Create EncryptionEngine for testing."""
        return EncryptionEngine(key_manager)

    @pytest.fixture
    def secure_comm(self, key_manager, encryption_engine):
        """Create SecureCommunication for testing."""
        return SecureCommunication(key_manager, encryption_engine)

    def test_secure_communication_initialization(self, secure_comm):
        """Test secure communication initialization."""
        assert secure_comm._key_manager is not None
        assert secure_comm._encryption_engine is not None
        assert secure_comm._certificates == {}
        assert secure_comm._lock is not None

    def test_generate_self_signed_certificate(self, secure_comm):
        """Test self-signed certificate generation."""
        subject_name = "test.example.com"
        cert_pem, key_pem = secure_comm.generate_self_signed_certificate(subject_name)

        assert cert_pem is not None
        assert key_pem is not None
        assert "-----BEGIN CERTIFICATE-----" in cert_pem
        assert "-----BEGIN PRIVATE KEY-----" in key_pem
        assert subject_name in cert_pem

    def test_generate_certificate_with_validity(self, secure_comm):
        """Test certificate generation with custom validity period."""
        subject_name = "test.example.com"
        validity_days = 30

        cert_pem, key_pem = secure_comm.generate_self_signed_certificate(
            subject_name,
            validity_days=validity_days
        )

        # Parse certificate to check validity
        from cryptography import x509
        cert = x509.load_pem_x509_certificate(cert_pem.encode('utf-8'), default_backend())

        validity_period = cert.not_valid_after - cert.not_valid_before
        assert abs(validity_period.days - validity_days) <= 1  # Allow 1 day tolerance

    def test_validate_certificate_self_signed_valid(self, secure_comm):
        """Test validation of valid self-signed certificate."""
        subject_name = "test.example.com"
        cert_pem, _ = secure_comm.generate_self_signed_certificate(subject_name)

        result = secure_comm.validate_certificate(cert_pem)
        assert result is True

    def test_validate_certificate_expired(self, secure_comm):
        """Test validation of expired certificate."""
        # Generate certificate with past validity
        subject_name = "test.example.com"

        # Mock certificate generation to create expired cert
        with patch('src.python.common.security.encryption.datetime') as mock_datetime:
            past_time = datetime.utcnow() - timedelta(days=10)
            mock_datetime.utcnow.return_value = past_time

            cert_pem, _ = secure_comm.generate_self_signed_certificate(subject_name, validity_days=1)

        result = secure_comm.validate_certificate(cert_pem)
        assert result is False

    def test_validate_certificate_invalid_format(self, secure_comm):
        """Test validation with invalid certificate format."""
        invalid_cert = "not a certificate"

        result = secure_comm.validate_certificate(invalid_cert)
        assert result is False

    def test_create_secure_channel(self, secure_comm):
        """Test secure channel creation."""
        # Generate certificate for remote peer
        remote_cert_pem, _ = secure_comm.generate_self_signed_certificate("remote.example.com")

        # Generate key for local peer
        local_key_id = secure_comm._key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

        channel = secure_comm.create_secure_channel(remote_cert_pem, local_key_id)

        assert isinstance(channel, SecureChannel)
        assert channel._remote_cert_pem == remote_cert_pem
        assert channel._local_key_id == local_key_id


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography library not available")
class TestSecureChannel:
    """Test SecureChannel functionality."""

    @pytest.fixture
    def secure_setup(self):
        """Create complete secure communication setup."""
        key_manager = KeyManager()
        encryption_engine = EncryptionEngine(key_manager)
        secure_comm = SecureCommunication(key_manager, encryption_engine)

        # Generate certificates and keys
        remote_cert_pem, _ = secure_comm.generate_self_signed_certificate("remote.example.com")
        local_key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

        channel = secure_comm.create_secure_channel(remote_cert_pem, local_key_id)

        return {
            'key_manager': key_manager,
            'encryption_engine': encryption_engine,
            'secure_comm': secure_comm,
            'channel': channel
        }

    def test_secure_channel_initialization(self, secure_setup):
        """Test secure channel initialization."""
        channel = secure_setup['channel']

        assert channel._session_key_id is not None
        assert channel._message_counter == 0
        assert channel._lock is not None

    def test_send_receive_message(self, secure_setup):
        """Test sending and receiving messages."""
        channel = secure_setup['channel']
        message = b"Hello, secure world!"
        additional_data = b"metadata"

        # Send message
        encrypted_message = channel.send_message(message, additional_data)

        assert encrypted_message is not None
        assert encrypted_message.data != message

        # Receive message
        decrypted_message = channel.receive_message(encrypted_message, additional_data)

        # Should match original message (without counter prefix)
        assert decrypted_message == message

    def test_message_counter_increment(self, secure_setup):
        """Test message counter increments."""
        channel = secure_setup['channel']
        message = b"test message"

        initial_counter = channel._message_counter

        channel.send_message(message)

        assert channel._message_counter == initial_counter + 1

    def test_multiple_messages(self, secure_setup):
        """Test sending multiple messages."""
        channel = secure_setup['channel']
        messages = [b"message 1", b"message 2", b"message 3"]

        encrypted_messages = []
        for msg in messages:
            encrypted_messages.append(channel.send_message(msg))

        decrypted_messages = []
        for encrypted in encrypted_messages:
            decrypted_messages.append(channel.receive_message(encrypted))

        assert decrypted_messages == messages

    def test_rotate_session_key(self, secure_setup):
        """Test session key rotation."""
        channel = secure_setup['channel']
        key_manager = secure_setup['key_manager']

        old_session_key_id = channel._session_key_id

        channel.rotate_session_key()

        new_session_key_id = channel._session_key_id

        assert new_session_key_id != old_session_key_id
        assert key_manager.get_key(old_session_key_id) is None  # Old key deleted
        assert key_manager.get_key(new_session_key_id) is not None

    def test_close_channel(self, secure_setup):
        """Test closing secure channel."""
        channel = secure_setup['channel']
        key_manager = secure_setup['key_manager']

        session_key_id = channel._session_key_id

        channel.close()

        assert channel._session_key_id is None
        assert key_manager.get_key(session_key_id) is None

    def test_send_without_session_key(self, secure_setup):
        """Test sending message without session key."""
        channel = secure_setup['channel']

        # Remove session key
        channel._session_key_id = None

        with pytest.raises(CryptoError, match="No session key established"):
            channel.send_message(b"test")

    def test_receive_without_session_key(self, secure_setup):
        """Test receiving message without session key."""
        channel = secure_setup['channel']

        # Send message first
        encrypted_message = channel.send_message(b"test")

        # Remove session key
        channel._session_key_id = None

        with pytest.raises(CryptoError, match="No session key established"):
            channel.receive_message(encrypted_message)


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography library not available")
class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.fixture
    def crypto_setup(self):
        """Create crypto setup for testing."""
        key_manager = KeyManager()
        encryption_engine = EncryptionEngine(key_manager)
        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

        return {
            'key_manager': key_manager,
            'encryption_engine': encryption_engine,
            'key_id': key_id
        }

    def test_encrypt_decrypt_json(self, crypto_setup):
        """Test JSON encryption and decryption."""
        data = {"message": "hello", "number": 42, "list": [1, 2, 3]}

        encrypted_b64 = encrypt_json(
            data,
            crypto_setup['key_id'],
            crypto_setup['key_manager'],
            crypto_setup['encryption_engine']
        )

        assert encrypted_b64 is not None
        assert isinstance(encrypted_b64, str)

        decrypted_data = decrypt_json(encrypted_b64, crypto_setup['encryption_engine'])

        assert decrypted_data == data

    def test_secure_temp_key_context(self):
        """Test secure temporary key context manager."""
        key_manager = KeyManager()

        with secure_temp_key(key_manager, EncryptionAlgorithm.AES_256_GCM) as key_id:
            assert key_id is not None
            assert key_manager.get_key(key_id) is not None

        # Key should be deleted after context exit
        assert key_manager.get_key(key_id) is None

    def test_secure_temp_key_exception_handling(self):
        """Test secure temporary key context with exception."""
        key_manager = KeyManager()
        temp_key_id = None

        try:
            with secure_temp_key(key_manager) as key_id:
                temp_key_id = key_id
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Key should still be deleted even with exception
        assert key_manager.get_key(temp_key_id) is None

    def test_generate_secure_token(self):
        """Test secure token generation."""
        token1 = generate_secure_token()
        token2 = generate_secure_token()

        assert token1 != token2
        assert isinstance(token1, str)
        assert isinstance(token2, str)
        assert len(token1) > 0
        assert len(token2) > 0

    def test_generate_secure_token_length(self):
        """Test secure token generation with specific length."""
        length = 16
        token = generate_secure_token(length)

        # URL-safe base64 encoding increases length
        assert len(token) >= length

    def test_constant_time_compare_equal(self):
        """Test constant-time comparison with equal values."""
        data1 = b"secret_value"
        data2 = b"secret_value"

        result = constant_time_compare(data1, data2)
        assert result is True

    def test_constant_time_compare_not_equal(self):
        """Test constant-time comparison with different values."""
        data1 = b"secret_value"
        data2 = b"different_value"

        result = constant_time_compare(data1, data2)
        assert result is False

    def test_constant_time_compare_different_lengths(self):
        """Test constant-time comparison with different lengths."""
        data1 = b"short"
        data2 = b"much_longer_data"

        result = constant_time_compare(data1, data2)
        assert result is False


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography library not available")
class TestGlobalInstances:
    """Test global instance functions."""

    def test_get_key_manager_singleton(self):
        """Test get_key_manager returns singleton."""
        km1 = get_key_manager()
        km2 = get_key_manager()

        assert km1 is km2
        assert isinstance(km1, KeyManager)

    def test_get_encryption_engine_singleton(self):
        """Test get_encryption_engine returns singleton."""
        ee1 = get_encryption_engine()
        ee2 = get_encryption_engine()

        assert ee1 is ee2
        assert isinstance(ee1, EncryptionEngine)

    def test_get_secure_communication_singleton(self):
        """Test get_secure_communication returns singleton."""
        sc1 = get_secure_communication()
        sc2 = get_secure_communication()

        assert sc1 is sc2
        assert isinstance(sc1, SecureCommunication)


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography library not available")
class TestErrorConditionsAndEdgeCases:
    """Test error conditions and edge cases."""

    def test_key_manager_without_cryptography(self):
        """Test key manager initialization without cryptography library."""
        with patch('src.python.common.security.encryption.CRYPTOGRAPHY_AVAILABLE', False):
            with pytest.raises(KeyManagementError, match="cryptography library not available"):
                KeyManager()

    def test_encryption_engine_without_cryptography(self):
        """Test encryption engine initialization without cryptography library."""
        with patch('src.python.common.security.encryption.CRYPTOGRAPHY_AVAILABLE', False):
            with pytest.raises(EncryptionError, match="cryptography library not available"):
                EncryptionEngine(Mock())

    def test_secure_communication_without_cryptography(self):
        """Test secure communication initialization without cryptography library."""
        with patch('src.python.common.security.encryption.CRYPTOGRAPHY_AVAILABLE', False):
            with pytest.raises(CryptoError, match="cryptography library not available"):
                SecureCommunication(Mock(), Mock())

    def test_key_generation_with_os_random_failure(self):
        """Test key generation when os.urandom fails."""
        key_manager = KeyManager()

        with patch('os.urandom', side_effect=OSError("Random generation failed")):
            with pytest.raises(KeyManagementError, match="Key generation failed"):
                key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

    def test_rsa_key_generation_failure(self):
        """Test RSA key generation failure."""
        key_manager = KeyManager()

        with patch('cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key',
                  side_effect=Exception("RSA generation failed")):
            with pytest.raises(KeyManagementError, match="Key generation failed"):
                key_manager.generate_key(EncryptionAlgorithm.RSA_2048)

    def test_key_derivation_with_invalid_kdf(self):
        """Test key derivation with invalid KDF."""
        key_manager = KeyManager()

        # Mock an invalid KDF
        with patch('src.python.common.security.encryption.KeyDerivationFunction') as mock_kdf:
            mock_kdf.INVALID = "invalid_kdf"
            with pytest.raises(KeyManagementError, match="Unsupported KDF"):
                key_manager.derive_key("password", kdf=mock_kdf.INVALID)

    def test_encryption_with_corrupted_key(self):
        """Test encryption with corrupted key data."""
        key_manager = KeyManager()
        encryption_engine = EncryptionEngine(key_manager)

        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

        # Corrupt key data
        key = key_manager._keys[key_id]
        key.key_data = b"corrupted_key_data"

        with pytest.raises(EncryptionError):
            encryption_engine.encrypt(b"test", key_id)

    def test_decryption_with_wrong_algorithm(self):
        """Test decryption with mismatched algorithm."""
        key_manager = KeyManager()
        encryption_engine = EncryptionEngine(key_manager)

        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
        plaintext = b"test message"

        encrypted_data = encryption_engine.encrypt(plaintext, key_id)

        # Change algorithm in encrypted data
        encrypted_data.algorithm = EncryptionAlgorithm.AES_256_CBC

        with pytest.raises(DecryptionError):
            encryption_engine.decrypt(encrypted_data)

    def test_certificate_generation_failure(self):
        """Test certificate generation failure."""
        key_manager = KeyManager()
        encryption_engine = EncryptionEngine(key_manager)
        secure_comm = SecureCommunication(key_manager, encryption_engine)

        with patch('cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key',
                  side_effect=Exception("Key generation failed")):
            with pytest.raises(CryptoError, match="Certificate generation failed"):
                secure_comm.generate_self_signed_certificate("test.example.com")

    def test_large_data_encryption_performance(self):
        """Test encryption with large data."""
        key_manager = KeyManager()
        encryption_engine = EncryptionEngine(key_manager)

        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

        # Large data (1MB)
        large_data = os.urandom(1024 * 1024)

        # Should handle large data without issues
        encrypted_data = encryption_engine.encrypt(large_data, key_id)
        decrypted_data = encryption_engine.decrypt(encrypted_data)

        assert decrypted_data == large_data

    def test_concurrent_key_operations(self):
        """Test concurrent key operations."""
        import threading

        key_manager = KeyManager()
        results = []
        errors = []

        def generate_keys():
            try:
                for _i in range(10):
                    key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
                    results.append(key_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=generate_keys) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should generate all keys successfully
        assert len(results) == 50
        assert len(errors) == 0
        assert len(set(results)) == 50  # All keys should be unique

    def test_memory_cleanup_on_exception(self):
        """Test memory cleanup on exception during operations."""
        key_manager = KeyManager()
        encryption_engine = EncryptionEngine(key_manager)

        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)

        # Force an exception during encryption
        with patch.object(encryption_engine, '_encrypt_aes_gcm',
                         side_effect=Exception("Encryption failed")):
            try:
                encryption_engine.encrypt(b"test", key_id)
            except EncryptionError:
                pass

        # Key should still be accessible and usage count not incremented
        key = key_manager.get_key(key_id)
        assert key is not None
        assert key.usage_count == 0

    def test_unicode_handling_in_key_derivation(self):
        """Test key derivation with unicode passwords."""
        key_manager = KeyManager()

        unicode_password = "пароль_测试_🔐"

        key_id = key_manager.derive_key(unicode_password)
        key = key_manager.get_key(key_id)

        assert key is not None
        assert len(key.key_data) == 32

    def test_zero_length_data_encryption(self):
        """Test encryption of zero-length data."""
        key_manager = KeyManager()
        encryption_engine = EncryptionEngine(key_manager)

        key_id = key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
        empty_data = b""

        encrypted_data = encryption_engine.encrypt(empty_data, key_id)
        decrypted_data = encryption_engine.decrypt(encrypted_data)

        assert decrypted_data == empty_data
