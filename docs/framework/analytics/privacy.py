"""Privacy controls and data protection for analytics system."""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class ConsentLevel(Enum):
    """Levels of user consent for data collection."""
    NONE = "none"
    ESSENTIAL = "essential"  # Only essential functionality analytics
    FUNCTIONAL = "functional"  # Include feature usage analytics
    ANALYTICS = "analytics"  # Include performance and error analytics
    ALL = "all"  # All analytics including detailed usage patterns


class DataRetentionPolicy(Enum):
    """Data retention policies."""
    MINIMAL = 7  # 7 days
    SHORT = 30  # 30 days
    STANDARD = 90  # 90 days
    EXTENDED = 365  # 1 year


@dataclass
class PrivacySettings:
    """User privacy settings for analytics."""
    consent_level: ConsentLevel = ConsentLevel.ESSENTIAL
    retention_days: int = DataRetentionPolicy.STANDARD.value
    anonymize_ip: bool = True
    respect_dnt: bool = True  # Respect Do Not Track header
    allow_performance_tracking: bool = True
    allow_error_tracking: bool = True
    allow_search_tracking: bool = True
    excluded_pages: List[str] = None
    consent_timestamp: Optional[datetime] = None
    ip_hash_salt: Optional[str] = None

    def __post_init__(self):
        if self.excluded_pages is None:
            self.excluded_pages = []
        if self.consent_timestamp is None:
            self.consent_timestamp = datetime.now()


class PrivacyManager:
    """Manages privacy controls and data protection for analytics."""

    def __init__(self, settings_path: Optional[Path] = None):
        """Initialize privacy manager.

        Args:
            settings_path: Path to store privacy settings
        """
        self.settings_path = settings_path or Path("analytics_privacy.json")
        self.settings = self._load_settings()
        self._consent_cache: Dict[str, ConsentLevel] = {}

    def _load_settings(self) -> PrivacySettings:
        """Load privacy settings from file.

        Returns:
            Privacy settings
        """
        if not self.settings_path.exists():
            return PrivacySettings()

        try:
            with open(self.settings_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert consent level from string
            if 'consent_level' in data:
                data['consent_level'] = ConsentLevel(data['consent_level'])

            # Convert timestamp
            if 'consent_timestamp' in data and data['consent_timestamp']:
                data['consent_timestamp'] = datetime.fromisoformat(data['consent_timestamp'])

            return PrivacySettings(**data)

        except Exception as e:
            logger.error(f"Failed to load privacy settings: {e}")
            return PrivacySettings()

    def _save_settings(self):
        """Save privacy settings to file."""
        try:
            data = {
                'consent_level': self.settings.consent_level.value,
                'retention_days': self.settings.retention_days,
                'anonymize_ip': self.settings.anonymize_ip,
                'respect_dnt': self.settings.respect_dnt,
                'allow_performance_tracking': self.settings.allow_performance_tracking,
                'allow_error_tracking': self.settings.allow_error_tracking,
                'allow_search_tracking': self.settings.allow_search_tracking,
                'excluded_pages': self.settings.excluded_pages,
                'consent_timestamp': self.settings.consent_timestamp.isoformat() if self.settings.consent_timestamp else None,
                'ip_hash_salt': self.settings.ip_hash_salt
            }

            # Ensure parent directory exists
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save privacy settings: {e}")

    def set_consent_level(self, consent_level: ConsentLevel) -> bool:
        """Set user consent level for data collection.

        Args:
            consent_level: Level of consent granted

        Returns:
            True if consent level was set successfully
        """
        try:
            self.settings.consent_level = consent_level
            self.settings.consent_timestamp = datetime.now()
            self._save_settings()

            logger.info(f"User consent level set to: {consent_level.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to set consent level: {e}")
            return False

    def get_consent_level(self) -> ConsentLevel:
        """Get current user consent level.

        Returns:
            Current consent level
        """
        return self.settings.consent_level

    def can_track_event_type(self, event_type: str, session_id: str = None) -> bool:
        """Check if a specific event type can be tracked based on consent.

        Args:
            event_type: Type of event to check
            session_id: Optional session ID for caching

        Returns:
            True if event can be tracked, False otherwise
        """
        # Check cached consent for session
        if session_id and session_id in self._consent_cache:
            cached_consent = self._consent_cache[session_id]
            return self._check_event_consent(event_type, cached_consent)

        current_consent = self.settings.consent_level

        # Cache consent for session
        if session_id:
            self._consent_cache[session_id] = current_consent

        return self._check_event_consent(event_type, current_consent)

    def _check_event_consent(self, event_type: str, consent_level: ConsentLevel) -> bool:
        """Check if event type is allowed for consent level.

        Args:
            event_type: Event type to check
            consent_level: User consent level

        Returns:
            True if allowed, False otherwise
        """
        if consent_level == ConsentLevel.NONE:
            return False

        if consent_level == ConsentLevel.ESSENTIAL:
            # Only allow essential functionality events
            essential_events = {"session_start", "session_end", "error"}
            return event_type in essential_events

        if consent_level == ConsentLevel.FUNCTIONAL:
            # Allow functional events
            functional_events = {
                "session_start", "session_end", "error", "page_view", "interaction"
            }
            return event_type in functional_events

        if consent_level == ConsentLevel.ANALYTICS:
            # Allow most analytics events
            if event_type == "search" and not self.settings.allow_search_tracking:
                return False
            if event_type == "performance" and not self.settings.allow_performance_tracking:
                return False
            if event_type == "error" and not self.settings.allow_error_tracking:
                return False
            return True

        # ConsentLevel.ALL allows everything (subject to other settings)
        return True

    def should_respect_dnt(self, request_headers: Dict[str, str]) -> bool:
        """Check if Do Not Track header should be respected.

        Args:
            request_headers: HTTP request headers

        Returns:
            True if tracking should be disabled due to DNT
        """
        if not self.settings.respect_dnt:
            return False

        dnt_header = request_headers.get('DNT') or request_headers.get('dnt')
        return dnt_header == '1'

    def anonymize_ip_address(self, ip_address: str) -> str:
        """Anonymize IP address for privacy protection.

        Args:
            ip_address: Raw IP address

        Returns:
            Anonymized IP hash or empty string if anonymization disabled
        """
        if not self.settings.anonymize_ip or not ip_address:
            return ""

        # Generate salt if not exists
        if not self.settings.ip_hash_salt:
            self.settings.ip_hash_salt = hashlib.sha256(
                f"{datetime.now().isoformat()}{ip_address}".encode('utf-8')
            ).hexdigest()[:16]
            self._save_settings()

        # Hash IP with salt
        combined = f"{ip_address}{self.settings.ip_hash_salt}"
        hash_obj = hashlib.sha256(combined.encode('utf-8'))
        return hash_obj.hexdigest()[:16]

    def is_page_excluded(self, page_path: str) -> bool:
        """Check if a page is excluded from tracking.

        Args:
            page_path: Page path to check

        Returns:
            True if page should be excluded from tracking
        """
        for excluded_pattern in self.settings.excluded_pages:
            if page_path.startswith(excluded_pattern):
                return True
        return False

    def add_excluded_page(self, page_pattern: str) -> bool:
        """Add a page pattern to be excluded from tracking.

        Args:
            page_pattern: Page pattern to exclude

        Returns:
            True if added successfully
        """
        try:
            if page_pattern not in self.settings.excluded_pages:
                self.settings.excluded_pages.append(page_pattern)
                self._save_settings()
            return True
        except Exception as e:
            logger.error(f"Failed to add excluded page: {e}")
            return False

    def remove_excluded_page(self, page_pattern: str) -> bool:
        """Remove a page pattern from exclusion list.

        Args:
            page_pattern: Page pattern to remove from exclusions

        Returns:
            True if removed successfully
        """
        try:
            if page_pattern in self.settings.excluded_pages:
                self.settings.excluded_pages.remove(page_pattern)
                self._save_settings()
            return True
        except Exception as e:
            logger.error(f"Failed to remove excluded page: {e}")
            return False

    def set_retention_policy(self, retention_policy: DataRetentionPolicy) -> bool:
        """Set data retention policy.

        Args:
            retention_policy: Retention policy to apply

        Returns:
            True if set successfully
        """
        try:
            self.settings.retention_days = retention_policy.value
            self._save_settings()
            return True
        except Exception as e:
            logger.error(f"Failed to set retention policy: {e}")
            return False

    def get_retention_days(self) -> int:
        """Get current data retention period in days.

        Returns:
            Number of days data should be retained
        """
        return self.settings.retention_days

    def is_consent_expired(self, max_age_days: int = 365) -> bool:
        """Check if consent has expired and needs to be renewed.

        Args:
            max_age_days: Maximum age of consent in days

        Returns:
            True if consent has expired
        """
        if not self.settings.consent_timestamp:
            return True

        expiry_date = self.settings.consent_timestamp + timedelta(days=max_age_days)
        return datetime.now() > expiry_date

    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get a summary of current privacy settings.

        Returns:
            Dictionary containing privacy settings summary
        """
        return {
            'consent_level': self.settings.consent_level.value,
            'consent_timestamp': self.settings.consent_timestamp.isoformat() if self.settings.consent_timestamp else None,
            'consent_expired': self.is_consent_expired(),
            'retention_days': self.settings.retention_days,
            'anonymize_ip': self.settings.anonymize_ip,
            'respect_dnt': self.settings.respect_dnt,
            'tracking_permissions': {
                'performance': self.settings.allow_performance_tracking,
                'errors': self.settings.allow_error_tracking,
                'search': self.settings.allow_search_tracking
            },
            'excluded_pages_count': len(self.settings.excluded_pages),
            'settings_path': str(self.settings_path)
        }

    def export_user_data(self, output_path: Path) -> bool:
        """Export user's privacy settings and consent record.

        Args:
            output_path: Path to export data to

        Returns:
            True if export was successful
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'privacy_settings': self.get_privacy_summary(),
                'full_settings': {
                    'consent_level': self.settings.consent_level.value,
                    'retention_days': self.settings.retention_days,
                    'anonymize_ip': self.settings.anonymize_ip,
                    'respect_dnt': self.settings.respect_dnt,
                    'allow_performance_tracking': self.settings.allow_performance_tracking,
                    'allow_error_tracking': self.settings.allow_error_tracking,
                    'allow_search_tracking': self.settings.allow_search_tracking,
                    'excluded_pages': self.settings.excluded_pages,
                    'consent_timestamp': self.settings.consent_timestamp.isoformat() if self.settings.consent_timestamp else None
                }
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Privacy data exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export privacy data: {e}")
            return False

    def reset_consent(self) -> bool:
        """Reset all consent settings to defaults.

        Returns:
            True if reset was successful
        """
        try:
            self.settings = PrivacySettings()
            self._save_settings()
            self._consent_cache.clear()

            logger.info("Privacy consent reset to defaults")
            return True

        except Exception as e:
            logger.error(f"Failed to reset consent: {e}")
            return False

    def clear_session_cache(self):
        """Clear cached consent decisions for sessions."""
        self._consent_cache.clear()

    def validate_settings(self) -> List[str]:
        """Validate privacy settings for consistency.

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if self.settings.retention_days < 1:
            issues.append("Retention days must be at least 1")

        if self.settings.retention_days > 3650:  # 10 years max
            issues.append("Retention days cannot exceed 10 years")

        if self.settings.consent_level == ConsentLevel.NONE and (
            self.settings.allow_performance_tracking or
            self.settings.allow_error_tracking or
            self.settings.allow_search_tracking
        ):
            issues.append("Tracking permissions inconsistent with NONE consent level")

        return issues