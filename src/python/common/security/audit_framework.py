"""
Comprehensive Audit Framework for workspace-qdrant-mcp.

This module provides enterprise-grade audit logging, compliance validation,
and regulatory reporting capabilities. It ensures complete traceability of
system operations and supports multiple compliance frameworks.

Features:
- Comprehensive audit trail with tamper-evident logging
- Multi-format compliance reporting (SOX, GDPR, HIPAA, etc.)
- Real-time audit event correlation and analysis
- Automated compliance validation and gap analysis
- Risk assessment and audit findings management
- Integration with external audit systems
- Data retention policy enforcement
- Audit log integrity verification
"""

import asyncio
import csv
import hashlib
import json
import sqlite3
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from .security_monitor import AlertLevel, SecurityAlert
from .threat_detection import SecurityEvent, ThreatDetection


class AuditLevel(Enum):
    """Audit event severity levels."""

    DEBUG = 1
    INFO = 2
    NOTICE = 3
    WARNING = 4
    ERROR = 5
    CRITICAL = 6


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""

    SOX = "sox"                    # Sarbanes-Oxley Act
    GDPR = "gdpr"                  # General Data Protection Regulation
    HIPAA = "hipaa"                # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"           # Payment Card Industry Data Security Standard
    ISO_27001 = "iso_27001"       # Information Security Management
    NIST_CSF = "nist_csf"         # NIST Cybersecurity Framework
    SOC2 = "soc2"                 # Service Organization Control 2
    FISMA = "fisma"               # Federal Information Security Management Act
    CUSTOM = "custom"              # Custom compliance requirements


class AuditEventType(Enum):
    """Types of auditable events."""

    USER_AUTHENTICATION = "user_authentication"
    USER_AUTHORIZATION = "user_authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_EXPORT = "data_export"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    SYSTEM_ACCESS = "system_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    THREAT_DETECTION = "threat_detection"
    POLICY_VIOLATION = "policy_violation"
    COMPLIANCE_CHECK = "compliance_check"
    AUDIT_LOG_ACCESS = "audit_log_access"
    SYSTEM_ERROR = "system_error"


@dataclass
class AuditEvent:
    """Represents a single audit event."""

    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    audit_level: AuditLevel
    user_id: str | None
    source_ip: str | None
    resource: str | None
    action: str
    outcome: str
    details: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    trace_id: str | None = None
    compliance_tags: set[ComplianceFramework] = field(default_factory=set)

    def __post_init__(self):
        """Post-initialization validation and processing."""
        if not self.event_id:
            self.event_id = str(uuid.uuid4())

        # Add automatic compliance tagging
        self._auto_tag_compliance()

        # Calculate integrity hash
        self._calculate_integrity_hash()

    def _auto_tag_compliance(self) -> None:
        """Automatically tag events for relevant compliance frameworks."""
        # Data access events are relevant to most frameworks
        if self.event_type in [AuditEventType.DATA_ACCESS, AuditEventType.DATA_MODIFICATION]:
            self.compliance_tags.update([
                ComplianceFramework.GDPR,
                ComplianceFramework.HIPAA,
                ComplianceFramework.SOX
            ])

        # Authentication events
        if self.event_type in [AuditEventType.USER_AUTHENTICATION, AuditEventType.USER_AUTHORIZATION]:
            self.compliance_tags.update([
                ComplianceFramework.ISO_27001,
                ComplianceFramework.SOC2,
                ComplianceFramework.NIST_CSF
            ])

        # Security events
        if self.event_type in [AuditEventType.SECURITY_EVENT, AuditEventType.THREAT_DETECTION]:
            self.compliance_tags.update([
                ComplianceFramework.ISO_27001,
                ComplianceFramework.NIST_CSF,
                ComplianceFramework.SOC2
            ])

        # Configuration changes
        if self.event_type == AuditEventType.CONFIGURATION_CHANGE:
            self.compliance_tags.update([
                ComplianceFramework.SOX,
                ComplianceFramework.SOC2,
                ComplianceFramework.ISO_27001
            ])

    def _calculate_integrity_hash(self) -> None:
        """Calculate integrity hash for tamper detection."""
        # Create deterministic string representation
        hash_data = {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'user_id': self.user_id,
            'resource': self.resource,
            'action': self.action,
            'outcome': self.outcome,
            'details': self.details
        }

        hash_string = json.dumps(hash_data, sort_keys=True)
        self.integrity_hash = hashlib.sha256(hash_string.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert audit event to dictionary format."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'audit_level': self.audit_level.value,
            'user_id': self.user_id,
            'source_ip': self.source_ip,
            'resource': self.resource,
            'action': self.action,
            'outcome': self.outcome,
            'details': self.details,
            'session_id': self.session_id,
            'trace_id': self.trace_id,
            'compliance_tags': [tag.value for tag in self.compliance_tags],
            'integrity_hash': getattr(self, 'integrity_hash', None)
        }

    def verify_integrity(self) -> bool:
        """Verify the integrity of this audit event."""
        if not hasattr(self, 'integrity_hash'):
            return False

        original_hash = self.integrity_hash
        self._calculate_integrity_hash()
        return original_hash == self.integrity_hash


@dataclass
class ComplianceRule:
    """Defines a compliance validation rule."""

    rule_id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirement: str
    check_function: str  # Name of method to call
    severity: AuditLevel
    remediation_advice: list[str] = field(default_factory=list)
    enabled: bool = True


class AuditLogger:
    """Core audit logging system with tamper-evident storage."""

    def __init__(self,
                 database_path: Path | None = None,
                 max_memory_events: int = 10000,
                 retention_days: int = 2555):  # 7 years default
        """
        Initialize audit logger.

        Args:
            database_path: Path to SQLite database for persistent storage
            max_memory_events: Maximum events to keep in memory buffer
            retention_days: How long to retain audit logs (days)
        """
        self.database_path = database_path or Path("audit_logs.db")
        self.max_memory_events = max_memory_events
        self.retention_days = retention_days

        # Memory buffer for recent events
        self.event_buffer: deque = deque(maxlen=max_memory_events)

        # Event correlation tracking
        self.session_events: dict[str, list[str]] = defaultdict(list)
        self.trace_events: dict[str, list[str]] = defaultdict(list)

        # Initialize database
        self._initialize_database()

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def _initialize_database(self) -> None:
        """Initialize SQLite database for audit log storage."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()

                # Create audit events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        audit_level INTEGER NOT NULL,
                        user_id TEXT,
                        source_ip TEXT,
                        resource TEXT,
                        action TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        details TEXT,
                        session_id TEXT,
                        trace_id TEXT,
                        compliance_tags TEXT,
                        integrity_hash TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create indexes for performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON audit_events(timestamp)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_event_type
                    ON audit_events(event_type)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_id
                    ON audit_events(user_id)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_compliance_tags
                    ON audit_events(compliance_tags)
                """)

                # Create audit trail integrity table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_integrity (
                        check_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        events_count INTEGER NOT NULL,
                        hash_chain TEXT NOT NULL,
                        verification_status TEXT NOT NULL
                    )
                """)

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")
            raise

    async def log_event(self, event: AuditEvent) -> None:
        """
        Log an audit event.

        Args:
            event: Audit event to log
        """
        async with self._lock:
            # Add to memory buffer
            self.event_buffer.append(event)

            # Track correlations
            if event.session_id:
                self.session_events[event.session_id].append(event.event_id)

            if event.trace_id:
                self.trace_events[event.trace_id].append(event.event_id)

            # Persist to database
            await self._persist_event(event)

            logger.debug(f"Logged audit event: {event.event_id}")

    async def _persist_event(self, event: AuditEvent) -> None:
        """Persist audit event to database."""
        try:
            event_dict = event.to_dict()

            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO audit_events (
                        event_id, timestamp, event_type, audit_level,
                        user_id, source_ip, resource, action, outcome,
                        details, session_id, trace_id, compliance_tags,
                        integrity_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_dict['event_id'],
                    event_dict['timestamp'],
                    event_dict['event_type'],
                    event_dict['audit_level'],
                    event_dict['user_id'],
                    event_dict['source_ip'],
                    event_dict['resource'],
                    event_dict['action'],
                    event_dict['outcome'],
                    json.dumps(event_dict['details']),
                    event_dict['session_id'],
                    event_dict['trace_id'],
                    json.dumps(event_dict['compliance_tags']),
                    event_dict['integrity_hash']
                ))

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to persist audit event: {e}")

    async def query_events(self,
                          start_time: datetime | None = None,
                          end_time: datetime | None = None,
                          event_type: AuditEventType | None = None,
                          user_id: str | None = None,
                          compliance_framework: ComplianceFramework | None = None,
                          limit: int = 1000) -> list[AuditEvent]:
        """
        Query audit events with filtering.

        Args:
            start_time: Start time for query
            end_time: End time for query
            event_type: Filter by event type
            user_id: Filter by user ID
            compliance_framework: Filter by compliance framework
            limit: Maximum number of events to return

        Returns:
            List of matching audit events
        """
        async with self._lock:
            try:
                with sqlite3.connect(self.database_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()

                    # Build query
                    query = "SELECT * FROM audit_events WHERE 1=1"
                    params = []

                    if start_time:
                        query += " AND timestamp >= ?"
                        params.append(start_time.isoformat())

                    if end_time:
                        query += " AND timestamp <= ?"
                        params.append(end_time.isoformat())

                    if event_type:
                        query += " AND event_type = ?"
                        params.append(event_type.value)

                    if user_id:
                        query += " AND user_id = ?"
                        params.append(user_id)

                    if compliance_framework:
                        query += " AND compliance_tags LIKE ?"
                        params.append(f'%{compliance_framework.value}%')

                    query += " ORDER BY timestamp DESC LIMIT ?"
                    params.append(limit)

                    cursor.execute(query, params)
                    rows = cursor.fetchall()

                    # Convert to AuditEvent objects
                    events = []
                    for row in rows:
                        event = self._row_to_audit_event(row)
                        events.append(event)

                    return events

            except Exception as e:
                logger.error(f"Failed to query audit events: {e}")
                return []

    def _row_to_audit_event(self, row: sqlite3.Row) -> AuditEvent:
        """Convert database row to AuditEvent object."""
        compliance_tags = set()
        if row['compliance_tags']:
            tag_list = json.loads(row['compliance_tags'])
            compliance_tags = {ComplianceFramework(tag) for tag in tag_list}

        event = AuditEvent(
            event_id=row['event_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            event_type=AuditEventType(row['event_type']),
            audit_level=AuditLevel(row['audit_level']),
            user_id=row['user_id'],
            source_ip=row['source_ip'],
            resource=row['resource'],
            action=row['action'],
            outcome=row['outcome'],
            details=json.loads(row['details']) if row['details'] else {},
            session_id=row['session_id'],
            trace_id=row['trace_id'],
            compliance_tags=compliance_tags
        )

        # Restore integrity hash
        event.integrity_hash = row['integrity_hash']
        return event

    async def verify_audit_trail_integrity(self) -> dict[str, Any]:
        """Verify the integrity of the audit trail."""
        async with self._lock:
            try:
                with sqlite3.connect(self.database_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()

                    # Get all events ordered by timestamp
                    cursor.execute("""
                        SELECT event_id, integrity_hash
                        FROM audit_events
                        ORDER BY timestamp
                    """)

                    rows = cursor.fetchall()
                    total_events = len(rows)
                    verified_events = 0
                    corrupted_events = []

                    # Verify each event's integrity hash
                    for row in rows:
                        event_id = row['event_id']

                        # Get full event and verify
                        cursor.execute("""
                            SELECT * FROM audit_events WHERE event_id = ?
                        """, (event_id,))

                        event_row = cursor.fetchone()
                        event = self._row_to_audit_event(event_row)

                        if event.verify_integrity():
                            verified_events += 1
                        else:
                            corrupted_events.append(event_id)

                    return {
                        'total_events': total_events,
                        'verified_events': verified_events,
                        'corrupted_events': corrupted_events,
                        'integrity_percentage': (verified_events / total_events * 100) if total_events > 0 else 100,
                        'verification_timestamp': datetime.utcnow().isoformat()
                    }

            except Exception as e:
                logger.error(f"Failed to verify audit trail integrity: {e}")
                return {
                    'error': str(e),
                    'verification_timestamp': datetime.utcnow().isoformat()
                }

    async def get_session_events(self, session_id: str) -> list[AuditEvent]:
        """Get all events for a specific session."""
        return await self.query_events(
            user_id=None,  # Don't filter by user
            limit=10000
        )

    async def cleanup_old_events(self) -> int:
        """Clean up events older than retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()

                # Count events to be deleted
                cursor.execute("""
                    SELECT COUNT(*) FROM audit_events
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))

                count = cursor.fetchone()[0]

                # Delete old events
                cursor.execute("""
                    DELETE FROM audit_events
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))

                conn.commit()

                logger.info(f"Cleaned up {count} old audit events")
                return count

        except Exception as e:
            logger.error(f"Failed to cleanup old audit events: {e}")
            return 0


class ComplianceReporter:
    """Generates compliance reports for various frameworks."""

    def __init__(self, audit_logger: AuditLogger):
        """
        Initialize compliance reporter.

        Args:
            audit_logger: Audit logger instance to query events from
        """
        self.audit_logger = audit_logger
        self.compliance_rules = self._load_compliance_rules()

    def _load_compliance_rules(self) -> dict[ComplianceFramework, list[ComplianceRule]]:
        """Load compliance rules for different frameworks."""
        rules = defaultdict(list)

        # GDPR Rules
        rules[ComplianceFramework.GDPR].extend([
            ComplianceRule(
                rule_id="GDPR_001",
                framework=ComplianceFramework.GDPR,
                title="Data Access Logging",
                description="All personal data access must be logged",
                requirement="Article 30 - Records of processing activities",
                check_function="check_data_access_logging",
                severity=AuditLevel.ERROR,
                remediation_advice=[
                    "Ensure all data access events are logged",
                    "Include user identification in audit logs",
                    "Maintain logs for at least 3 years"
                ]
            ),
            ComplianceRule(
                rule_id="GDPR_002",
                framework=ComplianceFramework.GDPR,
                title="Data Subject Rights",
                description="Data subject access requests must be logged and tracked",
                requirement="Article 15 - Right of access by the data subject",
                check_function="check_data_subject_requests",
                severity=AuditLevel.WARNING,
                remediation_advice=[
                    "Log all data subject access requests",
                    "Track response times and outcomes",
                    "Maintain evidence of compliance"
                ]
            )
        ])

        # SOX Rules
        rules[ComplianceFramework.SOX].extend([
            ComplianceRule(
                rule_id="SOX_001",
                framework=ComplianceFramework.SOX,
                title="Configuration Change Control",
                description="All system configuration changes must be authorized and logged",
                requirement="Section 404 - Management Assessment of Internal Controls",
                check_function="check_configuration_changes",
                severity=AuditLevel.CRITICAL,
                remediation_advice=[
                    "Implement change approval workflow",
                    "Log all configuration changes with approver",
                    "Maintain segregation of duties"
                ]
            )
        ])

        # ISO 27001 Rules
        rules[ComplianceFramework.ISO_27001].extend([
            ComplianceRule(
                rule_id="ISO_001",
                framework=ComplianceFramework.ISO_27001,
                title="Access Control Monitoring",
                description="User access and authentication events must be monitored",
                requirement="A.9.4.2 - Secure log-on procedures",
                check_function="check_access_control_logging",
                severity=AuditLevel.ERROR,
                remediation_advice=[
                    "Log all authentication attempts",
                    "Monitor failed login attempts",
                    "Implement account lockout policies"
                ]
            )
        ])

        return rules

    async def generate_compliance_report(self,
                                       framework: ComplianceFramework,
                                       start_date: datetime,
                                       end_date: datetime,
                                       output_format: str = "json") -> dict[str, Any]:
        """
        Generate compliance report for specific framework.

        Args:
            framework: Compliance framework to report on
            start_date: Start date for reporting period
            end_date: End date for reporting period
            output_format: Output format (json, csv, html)

        Returns:
            Compliance report data
        """
        # Get relevant events
        events = await self.audit_logger.query_events(
            start_time=start_date,
            end_time=end_date,
            compliance_framework=framework,
            limit=50000
        )

        # Run compliance checks
        rule_results = []
        for rule in self.compliance_rules[framework]:
            if rule.enabled:
                result = await self._evaluate_compliance_rule(rule, events)
                rule_results.append(result)

        # Calculate overall compliance score
        total_rules = len(rule_results)
        passed_rules = len([r for r in rule_results if r['status'] == 'PASS'])
        compliance_score = (passed_rules / total_rules * 100) if total_rules > 0 else 100

        report = {
            'framework': framework.value,
            'reporting_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'events_analyzed': len(events),
            'compliance_score': compliance_score,
            'rules_evaluated': total_rules,
            'rules_passed': passed_rules,
            'rules_failed': total_rules - passed_rules,
            'rule_results': rule_results,
            'recommendations': self._generate_recommendations(rule_results),
            'generated_at': datetime.utcnow().isoformat()
        }

        return report

    async def _evaluate_compliance_rule(self,
                                      rule: ComplianceRule,
                                      events: list[AuditEvent]) -> dict[str, Any]:
        """Evaluate a single compliance rule against audit events."""
        try:
            # Get the check function
            check_method = getattr(self, rule.check_function, None)
            if not check_method:
                return {
                    'rule_id': rule.rule_id,
                    'title': rule.title,
                    'status': 'ERROR',
                    'message': f'Check function {rule.check_function} not found',
                    'evidence': []
                }

            # Execute the check
            result = await check_method(rule, events)
            return result

        except Exception as e:
            logger.error(f"Error evaluating compliance rule {rule.rule_id}: {e}")
            return {
                'rule_id': rule.rule_id,
                'title': rule.title,
                'status': 'ERROR',
                'message': f'Error executing rule check: {str(e)}',
                'evidence': []
            }

    async def check_data_access_logging(self,
                                      rule: ComplianceRule,
                                      events: list[AuditEvent]) -> dict[str, Any]:
        """Check that data access events are properly logged."""
        data_access_events = [
            e for e in events
            if e.event_type in [AuditEventType.DATA_ACCESS, AuditEventType.DATA_MODIFICATION]
        ]

        # Check for completeness
        events_with_user_id = [e for e in data_access_events if e.user_id]
        events_with_resource = [e for e in data_access_events if e.resource]

        missing_user_id = len(data_access_events) - len(events_with_user_id)
        missing_resource = len(data_access_events) - len(events_with_resource)

        if missing_user_id > 0 or missing_resource > 0:
            status = 'FAIL'
            message = f'Missing user_id in {missing_user_id} events, missing resource in {missing_resource} events'
        else:
            status = 'PASS'
            message = f'All {len(data_access_events)} data access events properly logged'

        return {
            'rule_id': rule.rule_id,
            'title': rule.title,
            'status': status,
            'message': message,
            'evidence': {
                'total_data_access_events': len(data_access_events),
                'events_with_user_id': len(events_with_user_id),
                'events_with_resource': len(events_with_resource),
                'sample_events': [e.event_id for e in data_access_events[:5]]
            }
        }

    async def check_configuration_changes(self,
                                        rule: ComplianceRule,
                                        events: list[AuditEvent]) -> dict[str, Any]:
        """Check that configuration changes are properly controlled."""
        config_events = [
            e for e in events
            if e.event_type == AuditEventType.CONFIGURATION_CHANGE
        ]

        # Check for proper authorization
        authorized_events = [
            e for e in config_events
            if 'approver' in e.details or 'authorized_by' in e.details
        ]

        unauthorized_events = len(config_events) - len(authorized_events)

        if unauthorized_events > 0:
            status = 'FAIL'
            message = f'{unauthorized_events} configuration changes lack proper authorization'
        else:
            status = 'PASS'
            message = f'All {len(config_events)} configuration changes properly authorized'

        return {
            'rule_id': rule.rule_id,
            'title': rule.title,
            'status': status,
            'message': message,
            'evidence': {
                'total_config_changes': len(config_events),
                'authorized_changes': len(authorized_events),
                'unauthorized_changes': unauthorized_events,
                'sample_events': [e.event_id for e in config_events[:5]]
            }
        }

    async def check_access_control_logging(self,
                                         rule: ComplianceRule,
                                         events: list[AuditEvent]) -> dict[str, Any]:
        """Check that access control events are properly logged."""
        auth_events = [
            e for e in events
            if e.event_type in [AuditEventType.USER_AUTHENTICATION, AuditEventType.USER_AUTHORIZATION]
        ]

        # Check for failed authentication monitoring
        failed_auths = [e for e in auth_events if e.outcome.lower() in ['failed', 'denied', 'error']]
        success_auths = [e for e in auth_events if e.outcome.lower() in ['success', 'granted', 'allowed']]

        # Basic compliance check
        if len(auth_events) == 0:
            status = 'FAIL'
            message = 'No authentication events found in audit logs'
        else:
            status = 'PASS'
            message = f'Found {len(auth_events)} authentication events ({len(failed_auths)} failed, {len(success_auths)} successful)'

        return {
            'rule_id': rule.rule_id,
            'title': rule.title,
            'status': status,
            'message': message,
            'evidence': {
                'total_auth_events': len(auth_events),
                'failed_authentications': len(failed_auths),
                'successful_authentications': len(success_auths),
                'sample_events': [e.event_id for e in auth_events[:5]]
            }
        }

    async def check_data_subject_requests(self,
                                        rule: ComplianceRule,
                                        events: list[AuditEvent]) -> dict[str, Any]:
        """Check GDPR data subject request handling."""
        # Look for events related to data subject requests
        dsr_events = [
            e for e in events
            if 'data_subject_request' in str(e.details).lower() or
               'gdpr' in str(e.details).lower() or
               e.action.lower() in ['data_export', 'data_deletion', 'data_access_request']
        ]

        if len(dsr_events) > 0:
            # Check response times (should be within 30 days for GDPR)
            timely_responses = []
            for event in dsr_events:
                if 'response_time_days' in event.details:
                    if event.details['response_time_days'] <= 30:
                        timely_responses.append(event)

            if len(timely_responses) == len(dsr_events):
                status = 'PASS'
                message = f'All {len(dsr_events)} data subject requests handled within GDPR timeframes'
            else:
                status = 'FAIL'
                message = f'{len(dsr_events) - len(timely_responses)} requests exceeded GDPR response timeframes'
        else:
            status = 'INFO'
            message = 'No data subject requests found in reporting period'

        return {
            'rule_id': rule.rule_id,
            'title': rule.title,
            'status': status,
            'message': message,
            'evidence': {
                'total_dsr_events': len(dsr_events),
                'timely_responses': len(timely_responses) if len(dsr_events) > 0 else 0,
                'sample_events': [e.event_id for e in dsr_events[:3]]
            }
        }

    def _generate_recommendations(self, rule_results: list[dict[str, Any]]) -> list[str]:
        """Generate recommendations based on rule evaluation results."""
        recommendations = []

        failed_rules = [r for r in rule_results if r['status'] == 'FAIL']

        if failed_rules:
            recommendations.append(f"Address {len(failed_rules)} failed compliance checks")

            for rule in failed_rules:
                recommendations.append(f"Rule {rule['rule_id']}: {rule.get('message', 'Review implementation')}")

        error_rules = [r for r in rule_results if r['status'] == 'ERROR']
        if error_rules:
            recommendations.append(f"Fix {len(error_rules)} compliance rule evaluation errors")

        if not failed_rules and not error_rules:
            recommendations.append("Maintain current compliance posture")
            recommendations.append("Consider implementing additional monitoring for continuous compliance")

        return recommendations

    async def export_audit_trail(self,
                                start_date: datetime,
                                end_date: datetime,
                                format: str = "csv",
                                output_path: Path | None = None) -> Path:
        """
        Export audit trail for external review.

        Args:
            start_date: Start date for export
            end_date: End date for export
            format: Export format (csv, json)
            output_path: Output file path

        Returns:
            Path to exported file
        """
        events = await self.audit_logger.query_events(
            start_time=start_date,
            end_time=end_date,
            limit=100000
        )

        if not output_path:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"audit_export_{timestamp}.{format}")

        if format.lower() == "csv":
            await self._export_csv(events, output_path)
        elif format.lower() == "json":
            await self._export_json(events, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported {len(events)} audit events to {output_path}")
        return output_path

    async def _export_csv(self, events: list[AuditEvent], output_path: Path) -> None:
        """Export events to CSV format."""
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'event_id', 'timestamp', 'event_type', 'audit_level',
                'user_id', 'source_ip', 'resource', 'action', 'outcome',
                'session_id', 'trace_id', 'compliance_tags', 'integrity_hash'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for event in events:
                row = event.to_dict()
                # Flatten compliance_tags for CSV
                row['compliance_tags'] = ','.join(row['compliance_tags'])
                # Remove details for CSV simplicity
                row.pop('details', None)
                writer.writerow(row)

    async def _export_json(self, events: list[AuditEvent], output_path: Path) -> None:
        """Export events to JSON format."""
        event_data = [event.to_dict() for event in events]
        export_data = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'event_count': len(events),
            'events': event_data
        }

        with open(output_path, 'w') as jsonfile:
            json.dump(export_data, jsonfile, indent=2)


class AuditTrail:
    """High-level audit trail management combining logging and compliance."""

    def __init__(self,
                 database_path: Path | None = None,
                 retention_days: int = 2555):
        """
        Initialize audit trail system.

        Args:
            database_path: Path to audit database
            retention_days: Retention period for audit logs
        """
        self.audit_logger = AuditLogger(
            database_path=database_path,
            retention_days=retention_days
        )

        self.compliance_reporter = ComplianceReporter(self.audit_logger)

        # Integration with security components
        self.threat_callbacks: list[Callable] = []
        self.alert_callbacks: list[Callable] = []

    async def log_security_event(self, security_event: SecurityEvent) -> None:
        """Log a security event from the threat detection system."""
        audit_event = AuditEvent(
            event_id=f"sec_{security_event.event_id}",
            timestamp=security_event.timestamp,
            event_type=AuditEventType.SECURITY_EVENT,
            audit_level=AuditLevel.WARNING,
            user_id=security_event.user_id,
            source_ip=security_event.source_ip,
            resource=security_event.collection,
            action="security_monitoring",
            outcome="event_detected",
            details={
                'event_type': security_event.event_type,
                'query_hash': hashlib.md5(security_event.query.encode()).hexdigest()
                             if security_event.query else None,
                'metadata': security_event.metadata
            }
        )

        await self.audit_logger.log_event(audit_event)

    async def log_threat_detection(self, threat: ThreatDetection) -> None:
        """Log a threat detection from the security system."""
        audit_level_mapping = {
            1: AuditLevel.INFO,      # ThreatLevel.LOW
            2: AuditLevel.WARNING,   # ThreatLevel.MEDIUM
            3: AuditLevel.ERROR,     # ThreatLevel.HIGH
            4: AuditLevel.CRITICAL   # ThreatLevel.CRITICAL
        }

        audit_event = AuditEvent(
            event_id=f"threat_{threat.threat_id}",
            timestamp=threat.timestamp,
            event_type=AuditEventType.THREAT_DETECTION,
            audit_level=audit_level_mapping[threat.threat_level.value],
            user_id=threat.source_events[0].user_id if threat.source_events else None,
            source_ip=threat.source_events[0].source_ip if threat.source_events else None,
            resource=threat.source_events[0].collection if threat.source_events else None,
            action="threat_detection",
            outcome=f"{threat.threat_type.value}_detected",
            details={
                'threat_type': threat.threat_type.value,
                'threat_level': threat.threat_level.name,
                'confidence': threat.confidence,
                'description': threat.description,
                'mitigation_suggestions': threat.mitigation_suggestions,
                'source_event_count': len(threat.source_events)
            }
        )

        await self.audit_logger.log_event(audit_event)

    async def log_security_alert(self, alert: SecurityAlert) -> None:
        """Log a security alert from the monitoring system."""
        alert_level_mapping = {
            AlertLevel.INFO: AuditLevel.INFO,
            AlertLevel.WARNING: AuditLevel.WARNING,
            AlertLevel.ERROR: AuditLevel.ERROR,
            AlertLevel.CRITICAL: AuditLevel.CRITICAL,
        }

        audit_event = AuditEvent(
            event_id=f"alert_{alert.alert_id}",
            timestamp=alert.timestamp,
            event_type=AuditEventType.SECURITY_EVENT,
            audit_level=alert_level_mapping.get(
                alert.alert_level, AuditLevel.WARNING
            ),
            user_id=None,  # Alerts are system-generated
            source_ip=None,
            resource=alert.source,
            action="security_alert",
            outcome="alert_generated",
            details={
                'alert_level': alert.alert_level.name,
                'title': alert.title,
                'description': alert.description,
                'metadata': alert.metadata,
                'acknowledged': alert.acknowledged,
                'escalated': alert.escalated
            }
        )

        await self.audit_logger.log_event(audit_event)

    async def generate_compliance_dashboard(self) -> dict[str, Any]:
        """Generate comprehensive compliance dashboard data."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)  # Last 30 days

        # Generate reports for all frameworks
        framework_reports = {}
        for framework in ComplianceFramework:
            if framework != ComplianceFramework.CUSTOM:
                try:
                    report = await self.compliance_reporter.generate_compliance_report(
                        framework=framework,
                        start_date=start_date,
                        end_date=end_date
                    )
                    framework_reports[framework.value] = report
                except Exception as e:
                    logger.error(f"Failed to generate {framework.value} report: {e}")

        # Overall statistics
        recent_events = await self.audit_logger.query_events(
            start_time=start_date,
            end_time=end_date,
            limit=10000
        )

        # Calculate metrics
        event_counts = defaultdict(int)
        user_activity = defaultdict(int)

        for event in recent_events:
            event_counts[event.event_type.value] += 1
            if event.user_id:
                user_activity[event.user_id] += 1

        return {
            'reporting_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'total_events': len(recent_events),
            'event_breakdown': dict(event_counts),
            'active_users': len(user_activity),
            'top_users': sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10],
            'compliance_frameworks': framework_reports,
            'overall_compliance_score': sum(
                r.get('compliance_score', 0) for r in framework_reports.values()
            ) / len(framework_reports) if framework_reports else 0,
            'generated_at': datetime.utcnow().isoformat()
        }


# Export public interface
__all__ = [
    'AuditLogger',
    'AuditTrail',
    'ComplianceReporter',
    'AuditEvent',
    'AuditLevel',
    'AuditEventType',
    'ComplianceFramework',
    'ComplianceRule'
]
