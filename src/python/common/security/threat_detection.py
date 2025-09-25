"""
Advanced Threat Detection System for workspace-qdrant-mcp.

This module provides comprehensive threat detection with behavioral analysis,
anomaly detection, and attack pattern recognition. It monitors system activity
and identifies potential security threats through multiple detection layers.

Features:
- Real-time behavioral analysis
- Machine learning-based anomaly detection
- Known attack pattern matching
- Rate-based threat detection
- Multi-layer threat scoring
- Automated response triggers
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import statistics
import re
from concurrent.futures import ThreadPoolExecutor

from loguru import logger


class ThreatType(Enum):
    """Types of security threats that can be detected."""

    BRUTE_FORCE_ATTACK = "brute_force_attack"
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    RATE_LIMIT_ABUSE = "rate_limit_abuse"
    DATA_EXFILTRATION = "data_exfiltration"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    SUSPICIOUS_QUERY = "suspicious_query"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PATTERN_EVASION = "pattern_evasion"
    DOS_ATTACK = "dos_attack"
    UNKNOWN_THREAT = "unknown_threat"


class ThreatLevel(Enum):
    """Threat severity levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SecurityEvent:
    """Represents a security-related event in the system."""

    event_id: str
    timestamp: datetime
    event_type: str
    source_ip: str
    user_id: Optional[str]
    collection: Optional[str]
    query: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize event data."""
        if not self.event_id:
            self.event_id = self._generate_event_id()

        # Sanitize sensitive data
        if self.query and len(self.query) > 10000:
            self.query = self.query[:10000] + "... [truncated]"

    def _generate_event_id(self) -> str:
        """Generate unique event ID based on content."""
        content = f"{self.timestamp}{self.event_type}{self.source_ip}{self.user_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ThreatDetection:
    """Represents a detected security threat."""

    threat_id: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    confidence: float  # 0.0 - 1.0
    description: str
    source_events: List[SecurityEvent]
    mitigation_suggestions: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'threat_id': self.threat_id,
            'threat_type': self.threat_type.value,
            'threat_level': self.threat_level.value,
            'confidence': self.confidence,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'source_events': len(self.source_events),
            'mitigation_suggestions': self.mitigation_suggestions
        }


class BehavioralAnalyzer:
    """Analyzes user behavior patterns to detect anomalies."""

    def __init__(self, learning_window: int = 1000, anomaly_threshold: float = 2.0):
        """
        Initialize behavioral analyzer.

        Args:
            learning_window: Number of events to consider for baseline
            anomaly_threshold: Standard deviations from mean to flag as anomaly
        """
        self.learning_window = learning_window
        self.anomaly_threshold = anomaly_threshold
        self.user_profiles: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'query_patterns': defaultdict(int),
            'request_intervals': deque(maxlen=100),
            'collection_access': defaultdict(int),
            'query_sizes': deque(maxlen=100),
            'error_rates': deque(maxlen=50),
            'activity_hours': defaultdict(int),
            'ip_addresses': defaultdict(int)
        })
        self._lock = asyncio.Lock()

    async def analyze_event(self, event: SecurityEvent) -> Optional[ThreatDetection]:
        """
        Analyze an event for behavioral anomalies.

        Args:
            event: Security event to analyze

        Returns:
            ThreatDetection if anomaly detected, None otherwise
        """
        if not event.user_id:
            return None

        async with self._lock:
            return await self._analyze_user_behavior(event)

    async def _analyze_user_behavior(self, event: SecurityEvent) -> Optional[ThreatDetection]:
        """Internal behavior analysis."""
        user_id = event.user_id
        profile = self.user_profiles[user_id]

        # Update profile with new event
        self._update_user_profile(profile, event)

        # Detect anomalies
        anomalies = []

        # Check for unusual query patterns
        if self._detect_query_anomaly(profile, event):
            anomalies.append("Unusual query pattern detected")

        # Check for abnormal request frequency
        if self._detect_frequency_anomaly(profile, event):
            anomalies.append("Abnormal request frequency")

        # Check for unusual collection access
        if self._detect_access_anomaly(profile, event):
            anomalies.append("Unusual collection access pattern")

        # Check for time-based anomalies
        if self._detect_temporal_anomaly(profile, event):
            anomalies.append("Activity during unusual hours")

        if anomalies:
            return ThreatDetection(
                threat_id=f"behavioral_{event.event_id}",
                threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                threat_level=ThreatLevel.MEDIUM,
                confidence=min(0.8, len(anomalies) * 0.3),
                description=f"Behavioral anomalies detected for user {user_id}: {', '.join(anomalies)}",
                source_events=[event],
                mitigation_suggestions=[
                    "Review user access patterns",
                    "Consider implementing additional authentication",
                    "Monitor user activity more closely"
                ]
            )

        return None

    def _update_user_profile(self, profile: Dict[str, Any], event: SecurityEvent) -> None:
        """Update user profile with new event data."""
        current_time = time.time()

        # Update query patterns
        if event.query:
            query_hash = hashlib.md5(event.query.encode()).hexdigest()
            profile['query_patterns'][query_hash] += 1
            profile['query_sizes'].append(len(event.query))

        # Update request intervals
        if hasattr(profile, 'last_request_time'):
            interval = current_time - profile.get('last_request_time', current_time)
            profile['request_intervals'].append(interval)
        profile['last_request_time'] = current_time

        # Update collection access
        if event.collection:
            profile['collection_access'][event.collection] += 1

        # Update activity hours
        hour = datetime.fromtimestamp(current_time).hour
        profile['activity_hours'][hour] += 1

        # Update IP addresses
        profile['ip_addresses'][event.source_ip] += 1

    def _detect_query_anomaly(self, profile: Dict[str, Any], event: SecurityEvent) -> bool:
        """Detect unusual query patterns."""
        if not event.query or len(profile['query_sizes']) < 10:
            return False

        # Check query size anomaly
        sizes = list(profile['query_sizes'])
        if len(sizes) > 5:
            mean_size = statistics.mean(sizes[:-1])  # Exclude current query
            std_size = statistics.stdev(sizes[:-1]) if len(sizes) > 2 else mean_size

            if std_size > 0:
                z_score = abs((len(event.query) - mean_size) / std_size)
                return z_score > self.anomaly_threshold

        return False

    def _detect_frequency_anomaly(self, profile: Dict[str, Any], event: SecurityEvent) -> bool:
        """Detect abnormal request frequency."""
        intervals = list(profile['request_intervals'])
        if len(intervals) < 10:
            return False

        # Check if requests are too frequent
        recent_intervals = intervals[-5:]  # Last 5 intervals
        avg_interval = statistics.mean(recent_intervals)

        # Flag if average interval is less than 0.1 seconds (10 requests per second)
        return avg_interval < 0.1

    def _detect_access_anomaly(self, profile: Dict[str, Any], event: SecurityEvent) -> bool:
        """Detect unusual collection access patterns."""
        if not event.collection:
            return False

        access_counts = profile['collection_access']
        total_accesses = sum(access_counts.values())

        if total_accesses < 20:  # Need sufficient data
            return False

        # Check if accessing a collection that's rarely accessed
        current_collection_count = access_counts[event.collection]
        access_ratio = current_collection_count / total_accesses

        # Flag if accessing a collection used in less than 5% of requests
        return access_ratio < 0.05 and current_collection_count == 1

    def _detect_temporal_anomaly(self, profile: Dict[str, Any], event: SecurityEvent) -> bool:
        """Detect activity during unusual hours."""
        activity_hours = profile['activity_hours']
        total_activity = sum(activity_hours.values())

        if total_activity < 20:
            return False

        current_hour = event.timestamp.hour
        hour_ratio = activity_hours[current_hour] / total_activity

        # Flag if activity during an hour that represents less than 2% of total activity
        return hour_ratio < 0.02 and activity_hours[current_hour] == 1


class AnomalyDetector:
    """Machine learning-based anomaly detection system."""

    def __init__(self, window_size: int = 1000):
        """Initialize anomaly detector with sliding window."""
        self.window_size = window_size
        self.event_history: deque = deque(maxlen=window_size)
        self.feature_baselines: Dict[str, Dict[str, float]] = {}
        self._lock = asyncio.Lock()

    async def detect_anomaly(self, event: SecurityEvent) -> Optional[ThreatDetection]:
        """
        Detect anomalies using statistical analysis.

        Args:
            event: Event to analyze

        Returns:
            ThreatDetection if anomaly detected
        """
        async with self._lock:
            # Extract features from event
            features = self._extract_features(event)

            # Check against baselines
            anomaly_score = await self._calculate_anomaly_score(features)

            # Update history and baselines
            self.event_history.append((event, features))
            await self._update_baselines()

            # Determine if anomaly
            if anomaly_score > 0.7:  # Threshold for anomaly detection
                return ThreatDetection(
                    threat_id=f"anomaly_{event.event_id}",
                    threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                    threat_level=self._get_threat_level(anomaly_score),
                    confidence=anomaly_score,
                    description=f"Statistical anomaly detected (score: {anomaly_score:.3f})",
                    source_events=[event],
                    mitigation_suggestions=[
                        "Investigate unusual activity patterns",
                        "Review system logs for correlated events",
                        "Consider blocking suspicious IP if pattern continues"
                    ]
                )

        return None

    def _extract_features(self, event: SecurityEvent) -> Dict[str, float]:
        """Extract numerical features from event for analysis."""
        features = {}

        # Basic features
        features['hour'] = event.timestamp.hour
        features['day_of_week'] = event.timestamp.weekday()
        features['query_length'] = len(event.query) if event.query else 0

        # IP-based features
        ip_parts = event.source_ip.split('.')
        if len(ip_parts) == 4:
            try:
                features['ip_subnet'] = int(ip_parts[0]) * 256 + int(ip_parts[1])
            except (ValueError, IndexError):
                features['ip_subnet'] = 0

        # Query complexity (rough estimate)
        if event.query:
            features['query_complexity'] = self._estimate_query_complexity(event.query)

        # Metadata features
        features['metadata_count'] = len(event.metadata)

        return features

    def _estimate_query_complexity(self, query: str) -> float:
        """Estimate query complexity based on structure."""
        complexity = 0.0

        # Count operators and functions
        operators = ['AND', 'OR', 'NOT', '=', '>', '<', 'LIKE', 'IN']
        for op in operators:
            complexity += query.upper().count(op) * 0.5

        # Count nested structures
        complexity += query.count('(') * 0.3
        complexity += query.count('{') * 0.4

        # Count special characters
        special_chars = set(query) & {'*', '%', '?', '$', '^'}
        complexity += len(special_chars) * 0.2

        return min(complexity, 10.0)  # Cap at 10

    async def _calculate_anomaly_score(self, features: Dict[str, float]) -> float:
        """Calculate anomaly score based on deviation from baselines."""
        if not self.feature_baselines:
            return 0.0

        scores = []

        for feature_name, value in features.items():
            if feature_name in self.feature_baselines:
                baseline = self.feature_baselines[feature_name]
                mean = baseline.get('mean', 0)
                std = baseline.get('std', 1)

                if std > 0:
                    z_score = abs((value - mean) / std)
                    # Convert z-score to probability-like score
                    normalized_score = min(z_score / 3.0, 1.0)  # 3-sigma rule
                    scores.append(normalized_score)

        # Return average anomaly score
        return statistics.mean(scores) if scores else 0.0

    async def _update_baselines(self) -> None:
        """Update feature baselines from event history."""
        if len(self.event_history) < 10:
            return

        # Collect features from all events
        feature_data = defaultdict(list)

        for _, features in self.event_history:
            for feature_name, value in features.items():
                feature_data[feature_name].append(value)

        # Calculate baselines
        for feature_name, values in feature_data.items():
            if len(values) > 1:
                self.feature_baselines[feature_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 1.0,
                    'min': min(values),
                    'max': max(values)
                }

    def _get_threat_level(self, anomaly_score: float) -> ThreatLevel:
        """Convert anomaly score to threat level."""
        if anomaly_score >= 0.9:
            return ThreatLevel.CRITICAL
        elif anomaly_score >= 0.8:
            return ThreatLevel.HIGH
        elif anomaly_score >= 0.7:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW


class ThreatAnalyzer:
    """Analyzes patterns to detect known attack signatures."""

    def __init__(self):
        """Initialize threat analyzer with attack patterns."""
        self.attack_patterns = {
            ThreatType.SQL_INJECTION: [
                r"(?i)(union\s+select|1=1|'.*or.*')",
                r"(?i)(drop\s+table|delete\s+from|update\s+.*set)",
                r"(?i)(script.*>|<.*script)",
                r"[\'\";].*(\bunion\b|\bselect\b|\bdrop\b)"
            ],
            ThreatType.COMMAND_INJECTION: [
                r"[;&|].*(\bcat\b|\bls\b|\bwget\b|\bcurl\b)",
                r"(?i)(eval\s*\(|exec\s*\(|system\s*\()",
                r"`.*`|\$\(.*\)",
                r"[\r\n].*(\bchmod\b|\bchown\b|\brm\s+-rf)"
            ],
            ThreatType.SUSPICIOUS_QUERY: [
                r"(?i)(password|passwd|secret|token|key).*=.*['\"]",
                r"\b\d{4}-\d{2}-\d{2}.*\b\d{4}-\d{2}-\d{2}\b",  # Date ranges
                r"limit\s+\d{4,}|offset\s+\d{4,}",  # Large limits
                r"(?i)(\*.*\*|\%.*\%)"  # Wildcard patterns
            ]
        }

        self.compiled_patterns = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for better performance."""
        for threat_type, patterns in self.attack_patterns.items():
            self.compiled_patterns[threat_type] = [
                re.compile(pattern) for pattern in patterns
            ]

    async def analyze_for_threats(self, event: SecurityEvent) -> List[ThreatDetection]:
        """
        Analyze event for known threat patterns.

        Args:
            event: Event to analyze

        Returns:
            List of detected threats
        """
        threats = []

        if not event.query:
            return threats

        # Check against each threat type
        for threat_type, patterns in self.compiled_patterns.items():
            matches = []

            for pattern in patterns:
                if pattern.search(event.query):
                    matches.append(pattern.pattern)

            if matches:
                threat = ThreatDetection(
                    threat_id=f"pattern_{threat_type.value}_{event.event_id}",
                    threat_type=threat_type,
                    threat_level=self._get_pattern_threat_level(threat_type, len(matches)),
                    confidence=min(0.9, len(matches) * 0.3 + 0.6),
                    description=f"Detected {threat_type.value} patterns in query",
                    source_events=[event],
                    mitigation_suggestions=self._get_mitigation_suggestions(threat_type)
                )
                threats.append(threat)

        return threats

    def _get_pattern_threat_level(self, threat_type: ThreatType, match_count: int) -> ThreatLevel:
        """Determine threat level based on pattern type and match count."""
        base_levels = {
            ThreatType.SQL_INJECTION: ThreatLevel.HIGH,
            ThreatType.COMMAND_INJECTION: ThreatLevel.CRITICAL,
            ThreatType.SUSPICIOUS_QUERY: ThreatLevel.MEDIUM
        }

        base_level = base_levels.get(threat_type, ThreatLevel.MEDIUM)

        # Escalate based on match count
        if match_count > 2:
            return ThreatLevel.CRITICAL
        elif match_count > 1 and base_level.value < ThreatLevel.HIGH.value:
            return ThreatLevel.HIGH

        return base_level

    def _get_mitigation_suggestions(self, threat_type: ThreatType) -> List[str]:
        """Get mitigation suggestions for specific threat types."""
        suggestions = {
            ThreatType.SQL_INJECTION: [
                "Implement input sanitization",
                "Use parameterized queries",
                "Apply the principle of least privilege",
                "Consider blocking the source IP"
            ],
            ThreatType.COMMAND_INJECTION: [
                "Sanitize all user inputs",
                "Disable command execution features",
                "Implement strict input validation",
                "Block the source IP immediately"
            ],
            ThreatType.SUSPICIOUS_QUERY: [
                "Review query patterns and intent",
                "Implement query complexity limits",
                "Monitor for data exfiltration patterns",
                "Consider implementing query approval workflow"
            ]
        }

        return suggestions.get(threat_type, [
            "Investigate the source of this activity",
            "Review security policies",
            "Consider implementing additional monitoring"
        ])


class ThreatDetectionEngine:
    """Main threat detection engine coordinating all detection methods."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize threat detection engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize detection components
        self.behavioral_analyzer = BehavioralAnalyzer(
            learning_window=self.config.get('learning_window', 1000),
            anomaly_threshold=self.config.get('anomaly_threshold', 2.0)
        )

        self.anomaly_detector = AnomalyDetector(
            window_size=self.config.get('anomaly_window', 1000)
        )

        self.threat_analyzer = ThreatAnalyzer()

        # Detection state
        self.active_threats: Dict[str, ThreatDetection] = {}
        self.event_buffer: deque = deque(maxlen=10000)
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Thread pool for concurrent analysis
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Callbacks for threat notifications
        self.threat_callbacks: List[Callable[[ThreatDetection], None]] = []

    def register_threat_callback(self, callback: Callable[[ThreatDetection], None]) -> None:
        """Register a callback to be called when threats are detected."""
        self.threat_callbacks.append(callback)

    async def process_event(self, event: SecurityEvent) -> List[ThreatDetection]:
        """
        Process a security event through all detection layers.

        Args:
            event: Security event to analyze

        Returns:
            List of detected threats
        """
        detected_threats = []

        try:
            # Add to event buffer
            self.event_buffer.append(event)

            # Rate-based threat detection
            rate_threat = await self._detect_rate_abuse(event)
            if rate_threat:
                detected_threats.append(rate_threat)

            # Run all detection methods concurrently
            analysis_tasks = [
                self.behavioral_analyzer.analyze_event(event),
                self.anomaly_detector.detect_anomaly(event),
                self.threat_analyzer.analyze_for_threats(event)
            ]

            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Detection error: {result}")
                    continue

                if isinstance(result, ThreatDetection):
                    detected_threats.append(result)
                elif isinstance(result, list):
                    detected_threats.extend(result)

            # Store active threats and notify
            for threat in detected_threats:
                self.active_threats[threat.threat_id] = threat

                # Notify callbacks
                for callback in self.threat_callbacks:
                    try:
                        callback(threat)
                    except Exception as e:
                        logger.error(f"Threat callback error: {e}")

            logger.info(f"Processed event {event.event_id}: {len(detected_threats)} threats detected")

        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")

        return detected_threats

    async def _detect_rate_abuse(self, event: SecurityEvent) -> Optional[ThreatDetection]:
        """Detect rate-based abuse patterns."""
        source_key = f"{event.source_ip}:{event.user_id or 'anonymous'}"
        current_time = time.time()

        # Add current request
        self.rate_limits[source_key].append(current_time)

        # Check various time windows
        windows = [
            (10, 50, ThreatLevel.MEDIUM),    # 50 requests in 10 seconds
            (60, 200, ThreatLevel.MEDIUM),   # 200 requests in 1 minute
            (300, 1000, ThreatLevel.LOW),    # 1000 requests in 5 minutes
        ]

        for window_seconds, max_requests, threat_level in windows:
            recent_requests = [
                req_time for req_time in self.rate_limits[source_key]
                if current_time - req_time <= window_seconds
            ]

            if len(recent_requests) > max_requests:
                return ThreatDetection(
                    threat_id=f"rate_abuse_{source_key}_{int(current_time)}",
                    threat_type=ThreatType.RATE_LIMIT_ABUSE,
                    threat_level=threat_level,
                    confidence=0.9,
                    description=f"Rate limit abuse: {len(recent_requests)} requests in {window_seconds}s from {event.source_ip}",
                    source_events=[event],
                    mitigation_suggestions=[
                        "Implement rate limiting",
                        "Consider blocking the source IP",
                        "Implement CAPTCHA verification",
                        "Review API usage patterns"
                    ]
                )

        return None

    def get_active_threats(self,
                          threat_level: Optional[ThreatLevel] = None,
                          threat_type: Optional[ThreatType] = None) -> List[ThreatDetection]:
        """
        Get currently active threats with optional filtering.

        Args:
            threat_level: Filter by threat level
            threat_type: Filter by threat type

        Returns:
            List of active threats
        """
        threats = list(self.active_threats.values())

        if threat_level:
            threats = [t for t in threats if t.threat_level == threat_level]

        if threat_type:
            threats = [t for t in threats if t.threat_type == threat_type]

        return threats

    def clear_old_threats(self, max_age_hours: int = 24) -> int:
        """
        Clear threats older than specified age.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of threats cleared
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

        old_threat_ids = [
            threat_id for threat_id, threat in self.active_threats.items()
            if threat.timestamp < cutoff_time
        ]

        for threat_id in old_threat_ids:
            del self.active_threats[threat_id]

        logger.info(f"Cleared {len(old_threat_ids)} old threats")
        return len(old_threat_ids)

    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary statistics of threat detection."""
        threats = list(self.active_threats.values())

        summary = {
            'total_active_threats': len(threats),
            'threats_by_level': {
                level.name: len([t for t in threats if t.threat_level == level])
                for level in ThreatLevel
            },
            'threats_by_type': {
                threat_type.name: len([t for t in threats if t.threat_type == threat_type])
                for threat_type in ThreatType
            },
            'events_processed': len(self.event_buffer),
            'detection_components': {
                'behavioral_analyzer': 'active',
                'anomaly_detector': 'active',
                'threat_analyzer': 'active'
            }
        }

        return summary


# Export public interface
__all__ = [
    'ThreatType',
    'ThreatLevel',
    'SecurityEvent',
    'ThreatDetection',
    'BehavioralAnalyzer',
    'AnomalyDetector',
    'ThreatAnalyzer',
    'ThreatDetectionEngine'
]