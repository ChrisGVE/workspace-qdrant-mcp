"""Unit tests for analytics privacy manager."""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add docs framework to path for testing (if present)
docs_framework_path = os.path.join(os.path.dirname(__file__), '../../docs/framework')
sys.path.insert(0, docs_framework_path)

try:
    from analytics.privacy import (
        ConsentLevel,
        DataRetentionPolicy,
        PrivacyManager,
        PrivacySettings,
    )
except ModuleNotFoundError:
    pytest.skip("analytics framework not available", allow_module_level=True)


class TestConsentLevel:
    """Test ConsentLevel enum."""

    def test_consent_levels(self):
        """Test all consent level values."""
        assert ConsentLevel.NONE.value == "none"
        assert ConsentLevel.ESSENTIAL.value == "essential"
        assert ConsentLevel.FUNCTIONAL.value == "functional"
        assert ConsentLevel.ANALYTICS.value == "analytics"
        assert ConsentLevel.ALL.value == "all"


class TestDataRetentionPolicy:
    """Test DataRetentionPolicy enum."""

    def test_retention_policies(self):
        """Test all retention policy values."""
        assert DataRetentionPolicy.MINIMAL.value == 7
        assert DataRetentionPolicy.SHORT.value == 30
        assert DataRetentionPolicy.STANDARD.value == 90
        assert DataRetentionPolicy.EXTENDED.value == 365


class TestPrivacySettings:
    """Test PrivacySettings data class."""

    def test_default_settings(self):
        """Test default privacy settings."""
        settings = PrivacySettings()

        assert settings.consent_level == ConsentLevel.ESSENTIAL
        assert settings.retention_days == DataRetentionPolicy.STANDARD.value
        assert settings.anonymize_ip is True
        assert settings.respect_dnt is True
        assert settings.allow_performance_tracking is True
        assert settings.allow_error_tracking is True
        assert settings.allow_search_tracking is True
        assert settings.excluded_pages == []
        assert settings.consent_timestamp is not None
        assert settings.ip_hash_salt is None

    def test_custom_settings(self):
        """Test custom privacy settings."""
        custom_timestamp = datetime(2023, 1, 1, 12, 0, 0)
        settings = PrivacySettings(
            consent_level=ConsentLevel.ALL,
            retention_days=180,
            anonymize_ip=False,
            excluded_pages=['/admin', '/private'],
            consent_timestamp=custom_timestamp
        )

        assert settings.consent_level == ConsentLevel.ALL
        assert settings.retention_days == 180
        assert settings.anonymize_ip is False
        assert settings.excluded_pages == ['/admin', '/private']
        assert settings.consent_timestamp == custom_timestamp

    def test_excluded_pages_initialization(self):
        """Test excluded_pages initialization."""
        settings = PrivacySettings(excluded_pages=None)
        assert settings.excluded_pages == []

        settings = PrivacySettings(excluded_pages=['/test'])
        assert settings.excluded_pages == ['/test']


class TestPrivacyManager:
    """Test privacy manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.settings_path = Path(self.temp_dir) / "privacy_settings.json"
        self.manager = PrivacyManager(self.settings_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_no_settings_file(self):
        """Test initialization when no settings file exists."""
        assert not self.settings_path.exists()
        assert self.manager.settings.consent_level == ConsentLevel.ESSENTIAL
        assert self.manager._consent_cache == {}

    def test_initialization_with_settings_file(self):
        """Test initialization with existing settings file."""
        # Create settings file
        settings_data = {
            "consent_level": "all",
            "retention_days": 180,
            "anonymize_ip": False,
            "consent_timestamp": "2023-01-01T12:00:00"
        }

        with open(self.settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings_data, f)

        manager = PrivacyManager(self.settings_path)

        assert manager.settings.consent_level == ConsentLevel.ALL
        assert manager.settings.retention_days == 180
        assert manager.settings.anonymize_ip is False

    def test_initialization_corrupted_settings_file(self):
        """Test initialization with corrupted settings file."""
        # Create corrupted JSON file
        with open(self.settings_path, 'w', encoding='utf-8') as f:
            f.write("invalid json {")

        manager = PrivacyManager(self.settings_path)

        # Should fall back to defaults
        assert manager.settings.consent_level == ConsentLevel.ESSENTIAL

    def test_set_consent_level_success(self):
        """Test successful consent level setting."""
        result = self.manager.set_consent_level(ConsentLevel.ALL)

        assert result is True
        assert self.manager.settings.consent_level == ConsentLevel.ALL
        assert self.settings_path.exists()

        # Verify settings were saved
        with open(self.settings_path, encoding='utf-8') as f:
            saved_data = json.load(f)
        assert saved_data["consent_level"] == "all"

    def test_set_consent_level_file_error(self):
        """Test consent level setting with file write error."""
        with patch.object(self.manager, '_save_settings') as mock_save:
            mock_save.side_effect = Exception("Write error")

            result = self.manager.set_consent_level(ConsentLevel.ALL)
            assert result is False

    def test_get_consent_level(self):
        """Test getting consent level."""
        assert self.manager.get_consent_level() == ConsentLevel.ESSENTIAL

        self.manager.set_consent_level(ConsentLevel.ANALYTICS)
        assert self.manager.get_consent_level() == ConsentLevel.ANALYTICS

    def test_can_track_event_type_none_consent(self):
        """Test event tracking with NONE consent level."""
        self.manager.set_consent_level(ConsentLevel.NONE)

        assert self.manager.can_track_event_type("page_view") is False
        assert self.manager.can_track_event_type("search") is False
        assert self.manager.can_track_event_type("error") is False
        assert self.manager.can_track_event_type("performance") is False

    def test_can_track_event_type_essential_consent(self):
        """Test event tracking with ESSENTIAL consent level."""
        self.manager.set_consent_level(ConsentLevel.ESSENTIAL)

        assert self.manager.can_track_event_type("session_start") is True
        assert self.manager.can_track_event_type("session_end") is True
        assert self.manager.can_track_event_type("error") is True
        assert self.manager.can_track_event_type("page_view") is False
        assert self.manager.can_track_event_type("search") is False

    def test_can_track_event_type_functional_consent(self):
        """Test event tracking with FUNCTIONAL consent level."""
        self.manager.set_consent_level(ConsentLevel.FUNCTIONAL)

        assert self.manager.can_track_event_type("session_start") is True
        assert self.manager.can_track_event_type("page_view") is True
        assert self.manager.can_track_event_type("interaction") is True
        assert self.manager.can_track_event_type("search") is False
        assert self.manager.can_track_event_type("performance") is False

    def test_can_track_event_type_analytics_consent(self):
        """Test event tracking with ANALYTICS consent level."""
        self.manager.set_consent_level(ConsentLevel.ANALYTICS)

        assert self.manager.can_track_event_type("page_view") is True
        assert self.manager.can_track_event_type("search") is True
        assert self.manager.can_track_event_type("performance") is True
        assert self.manager.can_track_event_type("error") is True

    def test_can_track_event_type_all_consent(self):
        """Test event tracking with ALL consent level."""
        self.manager.set_consent_level(ConsentLevel.ALL)

        assert self.manager.can_track_event_type("page_view") is True
        assert self.manager.can_track_event_type("search") is True
        assert self.manager.can_track_event_type("performance") is True
        assert self.manager.can_track_event_type("error") is True
        assert self.manager.can_track_event_type("interaction") is True

    def test_can_track_event_type_with_disabled_settings(self):
        """Test event tracking with specific tracking disabled."""
        self.manager.set_consent_level(ConsentLevel.ALL)
        self.manager.settings.allow_search_tracking = False
        self.manager.settings.allow_performance_tracking = False
        self.manager.settings.allow_error_tracking = False

        assert self.manager.can_track_event_type("search") is False
        assert self.manager.can_track_event_type("performance") is False
        assert self.manager.can_track_event_type("error") is False
        assert self.manager.can_track_event_type("page_view") is True

    def test_can_track_event_type_with_caching(self):
        """Test event tracking with session caching."""
        session_id = "test-session"

        # First call should cache result
        result1 = self.manager.can_track_event_type("page_view", session_id)
        assert result1 is False  # ESSENTIAL doesn't allow page_view

        # Change consent level
        self.manager.set_consent_level(ConsentLevel.ALL)

        # Should still use cached result
        result2 = self.manager.can_track_event_type("page_view", session_id)
        assert result2 is False  # Still using cached ESSENTIAL level

        # Clear cache
        self.manager.clear_session_cache()

        # Now should use new consent level
        result3 = self.manager.can_track_event_type("page_view", session_id)
        assert result3 is True

    def test_should_respect_dnt_enabled(self):
        """Test DNT header respect when enabled."""
        assert self.manager.settings.respect_dnt is True

        headers_with_dnt = {"DNT": "1"}
        assert self.manager.should_respect_dnt(headers_with_dnt) is True

        headers_without_dnt = {}
        assert self.manager.should_respect_dnt(headers_without_dnt) is False

        headers_with_lowercase = {"dnt": "1"}
        assert self.manager.should_respect_dnt(headers_with_lowercase) is True

    def test_should_respect_dnt_disabled(self):
        """Test DNT header respect when disabled."""
        self.manager.settings.respect_dnt = False

        headers_with_dnt = {"DNT": "1"}
        assert self.manager.should_respect_dnt(headers_with_dnt) is False

    def test_anonymize_ip_address_enabled(self):
        """Test IP address anonymization when enabled."""
        assert self.manager.settings.anonymize_ip is True

        ip = "192.168.1.100"
        result = self.manager.anonymize_ip_address(ip)

        assert result != ip
        assert len(result) == 16
        assert result.isalnum()

        # Same IP should produce same hash
        result2 = self.manager.anonymize_ip_address(ip)
        assert result == result2

        # Different IP should produce different hash
        result3 = self.manager.anonymize_ip_address("10.0.0.1")
        assert result != result3

    def test_anonymize_ip_address_disabled(self):
        """Test IP address anonymization when disabled."""
        self.manager.settings.anonymize_ip = False

        ip = "192.168.1.100"
        result = self.manager.anonymize_ip_address(ip)

        assert result == ""

    def test_anonymize_ip_address_empty(self):
        """Test IP address anonymization with empty input."""
        result = self.manager.anonymize_ip_address("")
        assert result == ""

        result = self.manager.anonymize_ip_address(None)
        assert result == ""

    def test_anonymize_ip_address_generates_salt(self):
        """Test that IP anonymization generates and saves salt."""
        assert self.manager.settings.ip_hash_salt is None

        self.manager.anonymize_ip_address("192.168.1.1")

        # Salt should be generated and saved
        assert self.manager.settings.ip_hash_salt is not None
        assert len(self.manager.settings.ip_hash_salt) == 16

    def test_is_page_excluded_default(self):
        """Test page exclusion with default (empty) list."""
        assert self.manager.is_page_excluded("/test-page") is False
        assert self.manager.is_page_excluded("/any-page") is False

    def test_is_page_excluded_with_patterns(self):
        """Test page exclusion with defined patterns."""
        self.manager.settings.excluded_pages = ["/admin", "/private", "/test"]

        assert self.manager.is_page_excluded("/admin") is True
        assert self.manager.is_page_excluded("/admin/settings") is True
        assert self.manager.is_page_excluded("/private/data") is True
        assert self.manager.is_page_excluded("/test") is True
        assert self.manager.is_page_excluded("/public") is False

    def test_add_excluded_page_success(self):
        """Test adding excluded page pattern."""
        result = self.manager.add_excluded_page("/admin")
        assert result is True
        assert "/admin" in self.manager.settings.excluded_pages

        # Adding same pattern again should still work
        result = self.manager.add_excluded_page("/admin")
        assert result is True
        assert self.manager.settings.excluded_pages.count("/admin") == 1

    def test_add_excluded_page_error(self):
        """Test adding excluded page with save error."""
        with patch.object(self.manager, '_save_settings') as mock_save:
            mock_save.side_effect = Exception("Save error")

            result = self.manager.add_excluded_page("/admin")
            assert result is False

    def test_remove_excluded_page_success(self):
        """Test removing excluded page pattern."""
        self.manager.settings.excluded_pages = ["/admin", "/private"]

        result = self.manager.remove_excluded_page("/admin")
        assert result is True
        assert "/admin" not in self.manager.settings.excluded_pages
        assert "/private" in self.manager.settings.excluded_pages

    def test_remove_excluded_page_not_found(self):
        """Test removing non-existent excluded page."""
        result = self.manager.remove_excluded_page("/nonexistent")
        assert result is True  # Should succeed even if not found

    def test_remove_excluded_page_error(self):
        """Test removing excluded page with save error."""
        self.manager.settings.excluded_pages = ["/admin"]

        with patch.object(self.manager, '_save_settings') as mock_save:
            mock_save.side_effect = Exception("Save error")

            result = self.manager.remove_excluded_page("/admin")
            assert result is False

    def test_set_retention_policy(self):
        """Test setting retention policy."""
        result = self.manager.set_retention_policy(DataRetentionPolicy.EXTENDED)
        assert result is True
        assert self.manager.settings.retention_days == 365

    def test_set_retention_policy_error(self):
        """Test setting retention policy with save error."""
        with patch.object(self.manager, '_save_settings') as mock_save:
            mock_save.side_effect = Exception("Save error")

            result = self.manager.set_retention_policy(DataRetentionPolicy.SHORT)
            assert result is False

    def test_get_retention_days(self):
        """Test getting retention days."""
        assert self.manager.get_retention_days() == DataRetentionPolicy.STANDARD.value

        self.manager.set_retention_policy(DataRetentionPolicy.MINIMAL)
        assert self.manager.get_retention_days() == DataRetentionPolicy.MINIMAL.value

    def test_is_consent_expired_no_timestamp(self):
        """Test consent expiry check with no timestamp."""
        self.manager.settings.consent_timestamp = None
        assert self.manager.is_consent_expired() is True

    def test_is_consent_expired_recent(self):
        """Test consent expiry check with recent timestamp."""
        self.manager.settings.consent_timestamp = datetime.now() - timedelta(days=10)
        assert self.manager.is_consent_expired() is False

    def test_is_consent_expired_old(self):
        """Test consent expiry check with old timestamp."""
        self.manager.settings.consent_timestamp = datetime.now() - timedelta(days=400)
        assert self.manager.is_consent_expired() is True

    def test_is_consent_expired_custom_age(self):
        """Test consent expiry check with custom max age."""
        self.manager.settings.consent_timestamp = datetime.now() - timedelta(days=10)

        assert self.manager.is_consent_expired(max_age_days=5) is True
        assert self.manager.is_consent_expired(max_age_days=15) is False

    def test_get_privacy_summary(self):
        """Test privacy summary generation."""
        self.manager.set_consent_level(ConsentLevel.ANALYTICS)
        self.manager.settings.excluded_pages = ["/admin", "/private"]

        summary = self.manager.get_privacy_summary()

        assert summary["consent_level"] == "analytics"
        assert "consent_timestamp" in summary
        assert "consent_expired" in summary
        assert summary["retention_days"] == 90
        assert summary["anonymize_ip"] is True
        assert summary["respect_dnt"] is True
        assert summary["excluded_pages_count"] == 2
        assert "tracking_permissions" in summary
        assert summary["tracking_permissions"]["performance"] is True

    def test_export_user_data_success(self):
        """Test successful user data export."""
        self.manager.set_consent_level(ConsentLevel.ALL)
        self.manager.settings.excluded_pages = ["/admin"]

        export_path = Path(self.temp_dir) / "privacy_export.json"
        result = self.manager.export_user_data(export_path)

        assert result is True
        assert export_path.exists()

        # Verify exported data
        with open(export_path, encoding='utf-8') as f:
            export_data = json.load(f)

        assert "export_timestamp" in export_data
        assert "privacy_settings" in export_data
        assert "full_settings" in export_data
        assert export_data["full_settings"]["consent_level"] == "all"

    def test_export_user_data_file_error(self):
        """Test user data export with file error."""
        export_path = Path("/non/existent/directory/export.json")
        result = self.manager.export_user_data(export_path)

        assert result is False

    def test_reset_consent_success(self):
        """Test successful consent reset."""
        # Set some non-default values
        self.manager.set_consent_level(ConsentLevel.ALL)
        self.manager.settings.excluded_pages = ["/admin"]
        self.manager._consent_cache["session"] = ConsentLevel.ALL

        result = self.manager.reset_consent()

        assert result is True
        assert self.manager.settings.consent_level == ConsentLevel.ESSENTIAL
        assert self.manager.settings.excluded_pages == []
        assert self.manager._consent_cache == {}

    def test_reset_consent_error(self):
        """Test consent reset with save error."""
        with patch.object(self.manager, '_save_settings') as mock_save:
            mock_save.side_effect = Exception("Save error")

            result = self.manager.reset_consent()
            assert result is False

    def test_clear_session_cache(self):
        """Test clearing session cache."""
        self.manager._consent_cache = {
            "session1": ConsentLevel.ALL,
            "session2": ConsentLevel.ESSENTIAL
        }

        self.manager.clear_session_cache()

        assert self.manager._consent_cache == {}

    def test_validate_settings_valid(self):
        """Test settings validation with valid settings."""
        issues = self.manager.validate_settings()
        assert issues == []

    def test_validate_settings_invalid_retention(self):
        """Test settings validation with invalid retention days."""
        self.manager.settings.retention_days = 0
        issues = self.manager.validate_settings()
        assert len(issues) > 0
        assert any("Retention days must be at least 1" in issue for issue in issues)

        self.manager.settings.retention_days = 5000
        issues = self.manager.validate_settings()
        assert len(issues) > 0
        assert any("cannot exceed 10 years" in issue for issue in issues)

    def test_validate_settings_inconsistent_consent(self):
        """Test settings validation with inconsistent consent."""
        self.manager.settings.consent_level = ConsentLevel.NONE
        self.manager.settings.allow_performance_tracking = True

        issues = self.manager.validate_settings()
        assert len(issues) > 0
        assert any("inconsistent with NONE consent level" in issue for issue in issues)

    def test_save_settings_creates_directory(self):
        """Test that save_settings creates parent directory."""
        nested_path = Path(self.temp_dir) / "nested" / "deep" / "settings.json"
        manager = PrivacyManager(nested_path)

        result = manager.set_consent_level(ConsentLevel.ALL)

        assert result is True
        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_load_settings_with_missing_fields(self):
        """Test loading settings with missing fields."""
        # Create partial settings file
        partial_settings = {
            "consent_level": "all",
            "retention_days": 180
            # Missing other fields
        }

        with open(self.settings_path, 'w', encoding='utf-8') as f:
            json.dump(partial_settings, f)

        manager = PrivacyManager(self.settings_path)

        # Should use defaults for missing fields
        assert manager.settings.consent_level == ConsentLevel.ALL
        assert manager.settings.retention_days == 180
        assert manager.settings.anonymize_ip is True  # Default
        assert manager.settings.respect_dnt is True  # Default

    def test_consent_level_edge_cases(self):
        """Test consent level handling edge cases."""
        # Test unknown event type
        assert self.manager.can_track_event_type("unknown_event") is False

        # Test empty event type
        assert self.manager.can_track_event_type("") is False

        # Test with None session ID
        result = self.manager.can_track_event_type("page_view", None)
        assert isinstance(result, bool)

    def test_ip_anonymization_consistency(self):
        """Test IP anonymization produces consistent results."""
        ip1 = "192.168.1.1"
        ip2 = "10.0.0.1"

        # Multiple calls with same IP should produce same result
        hash1a = self.manager.anonymize_ip_address(ip1)
        hash1b = self.manager.anonymize_ip_address(ip1)
        assert hash1a == hash1b

        # Different IPs should produce different results
        hash2 = self.manager.anonymize_ip_address(ip2)
        assert hash1a != hash2

        # After restart (new manager instance), should still be consistent
        manager2 = PrivacyManager(self.settings_path)
        hash1c = manager2.anonymize_ip_address(ip1)
        assert hash1a == hash1c
