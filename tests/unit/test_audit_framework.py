"""
Comprehensive unit tests for audit framework.

Tests cover all components with edge cases and error conditions:
- AuditLogger with database persistence and integrity verification
- ComplianceReporter with multiple framework validations
- AuditTrail integration with security components
- Database operations and concurrency handling
- Export functionality and data integrity
- Compliance rule evaluation and reporting
"""

import asyncio
import json
import pytest
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from collections import defaultdict

from src.python.common.security.audit_framework import (
    AuditLogger,
    ComplianceReporter,
    AuditTrail,
    AuditEvent,
    AuditLevel,
    AuditEventType,
    ComplianceFramework,
    ComplianceRule
)

from src.python.common.security.threat_detection import (
    ThreatDetection,
    ThreatLevel,
    ThreatType,
    SecurityEvent
)

from src.python.common.security.security_monitor import SecurityAlert, AlertLevel


class TestAuditEvent:
    """Test AuditEvent creation and validation."""

    def test_audit_event_creation(self):
        """Test basic audit event creation."""
        event = AuditEvent(
            event_id="test_001",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.DATA_ACCESS,
            audit_level=AuditLevel.INFO,
            user_id="user123",
            source_ip="192.168.1.100",
            resource="/api/documents",
            action="read",
            outcome="success"
        )

        assert event.event_id == "test_001"
        assert event.user_id == "user123"
        assert event.action == "read"
        assert event.outcome == "success"
        assert hasattr(event, 'integrity_hash')
        assert len(event.compliance_tags) > 0  # Should have auto-tagged

    def test_audit_event_auto_id_generation(self):
        """Test automatic event ID generation."""
        event = AuditEvent(
            event_id="",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.SYSTEM_ACCESS,
            audit_level=AuditLevel.INFO,
            user_id=None,
            source_ip=None,
            resource=None,
            action="login",
            outcome="success"
        )

        assert event.event_id  # Should be auto-generated
        assert len(event.event_id) == 36  # UUID format

    def test_compliance_auto_tagging(self):
        """Test automatic compliance framework tagging."""
        # Data access event should get GDPR, HIPAA, SOX tags
        data_event = AuditEvent(
            event_id="data_001",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.DATA_ACCESS,
            audit_level=AuditLevel.INFO,
            user_id="user123",
            source_ip="192.168.1.100",
            resource="/data/users",
            action="select",
            outcome="success"
        )

        expected_tags = {ComplianceFramework.GDPR, ComplianceFramework.HIPAA, ComplianceFramework.SOX}
        assert expected_tags.issubset(data_event.compliance_tags)

        # Authentication event should get ISO, SOC2, NIST tags
        auth_event = AuditEvent(
            event_id="auth_001",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.USER_AUTHENTICATION,
            audit_level=AuditLevel.INFO,
            user_id="user123",
            source_ip="192.168.1.100",
            resource="/login",
            action="authenticate",
            outcome="success"
        )

        expected_auth_tags = {ComplianceFramework.ISO_27001, ComplianceFramework.SOC2, ComplianceFramework.NIST_CSF}
        assert expected_auth_tags.issubset(auth_event.compliance_tags)

    def test_integrity_hash_calculation(self):
        """Test integrity hash calculation and verification."""
        event = AuditEvent(
            event_id="integrity_001",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.DATA_MODIFICATION,
            audit_level=AuditLevel.WARNING,
            user_id="user123",
            source_ip="192.168.1.100",
            resource="/data/documents/1",
            action="update",
            outcome="success",
            details={'field_changed': 'title', 'old_value': 'Old Title', 'new_value': 'New Title'}
        )

        # Should have integrity hash
        assert hasattr(event, 'integrity_hash')
        assert event.integrity_hash
        assert len(event.integrity_hash) == 64  # SHA256

        # Verification should pass
        assert event.verify_integrity()

        # Tampering should be detected
        original_hash = event.integrity_hash
        event.action = "tampered_action"
        event._calculate_integrity_hash()
        assert event.integrity_hash != original_hash

    def test_audit_event_to_dict(self):
        """Test conversion to dictionary format."""
        event = AuditEvent(
            event_id="dict_001",
            timestamp=datetime(2023, 6, 15, 10, 30, 0),
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            audit_level=AuditLevel.ERROR,
            user_id="admin",
            source_ip="10.0.0.1",
            resource="/config/database",
            action="update",
            outcome="success",
            details={'changed_setting': 'max_connections'},
            session_id="session_123",
            trace_id="trace_456"
        )

        event_dict = event.to_dict()

        assert event_dict['event_id'] == "dict_001"
        assert event_dict['event_type'] == 'configuration_change'
        assert event_dict['audit_level'] == 5  # ERROR = 5
        assert event_dict['user_id'] == "admin"
        assert event_dict['session_id'] == "session_123"
        assert event_dict['trace_id'] == "trace_456"
        assert isinstance(event_dict['compliance_tags'], list)
        assert 'integrity_hash' in event_dict


class TestAuditLogger:
    """Test audit logger functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.fixture
    def audit_logger(self, temp_db):
        """Create audit logger with temporary database."""
        return AuditLogger(database_path=temp_db, max_memory_events=100, retention_days=30)

    @pytest.fixture
    def sample_event(self):
        """Create sample audit event."""
        return AuditEvent(
            event_id="sample_001",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.DATA_ACCESS,
            audit_level=AuditLevel.INFO,
            user_id="test_user",
            source_ip="192.168.1.100",
            resource="/api/documents",
            action="read",
            outcome="success",
            details={'document_id': 123}
        )

    @pytest.mark.asyncio
    async def test_log_event(self, audit_logger, sample_event):
        """Test basic event logging."""
        await audit_logger.log_event(sample_event)

        # Event should be in memory buffer
        assert len(audit_logger.event_buffer) == 1
        assert audit_logger.event_buffer[0].event_id == "sample_001"

        # Event should be persisted to database
        with sqlite3.connect(audit_logger.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM audit_events")
            count = cursor.fetchone()[0]
            assert count == 1

            cursor.execute("SELECT event_id, user_id FROM audit_events WHERE event_id = ?",
                          ("sample_001",))
            row = cursor.fetchone()
            assert row[0] == "sample_001"
            assert row[1] == "test_user"

    @pytest.mark.asyncio
    async def test_query_events_basic(self, audit_logger, sample_event):
        """Test basic event querying."""
        await audit_logger.log_event(sample_event)

        events = await audit_logger.query_events(limit=10)
        assert len(events) == 1
        assert events[0].event_id == "sample_001"
        assert events[0].user_id == "test_user"

    @pytest.mark.asyncio
    async def test_query_events_with_filters(self, audit_logger):
        """Test event querying with various filters."""
        # Create multiple test events
        events = [
            AuditEvent(
                event_id="filter_001",
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.DATA_ACCESS,
                audit_level=AuditLevel.INFO,
                user_id="user1",
                source_ip="192.168.1.100",
                resource="/api/documents",
                action="read",
                outcome="success"
            ),
            AuditEvent(
                event_id="filter_002",
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.USER_AUTHENTICATION,
                audit_level=AuditLevel.WARNING,
                user_id="user2",
                source_ip="192.168.1.101",
                resource="/login",
                action="authenticate",
                outcome="failed"
            ),
            AuditEvent(
                event_id="filter_003",
                timestamp=datetime.utcnow() - timedelta(hours=2),
                event_type=AuditEventType.DATA_ACCESS,
                audit_level=AuditLevel.INFO,
                user_id="user1",
                source_ip="192.168.1.100",
                resource="/api/reports",
                action="read",
                outcome="success"
            )
        ]

        for event in events:
            await audit_logger.log_event(event)

        # Filter by event type
        data_events = await audit_logger.query_events(
            event_type=AuditEventType.DATA_ACCESS,
            limit=10
        )
        assert len(data_events) == 2

        # Filter by user
        user1_events = await audit_logger.query_events(
            user_id="user1",
            limit=10
        )
        assert len(user1_events) == 2

        # Filter by time range
        recent_events = await audit_logger.query_events(
            start_time=datetime.utcnow() - timedelta(hours=1),
            limit=10
        )
        assert len(recent_events) == 2  # Should exclude the 2-hour old event

        # Filter by compliance framework
        gdpr_events = await audit_logger.query_events(
            compliance_framework=ComplianceFramework.GDPR,
            limit=10
        )
        # DATA_ACCESS events should be tagged with GDPR
        assert len(gdpr_events) >= 2

    @pytest.mark.asyncio
    async def test_session_and_trace_correlation(self, audit_logger):
        """Test session and trace ID correlation tracking."""
        session_id = "session_123"
        trace_id = "trace_456"

        # Create events with same session/trace
        for i in range(3):
            event = AuditEvent(
                event_id=f"corr_{i}",
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.DATA_ACCESS,
                audit_level=AuditLevel.INFO,
                user_id="test_user",
                source_ip="192.168.1.100",
                resource=f"/api/resource_{i}",
                action="read",
                outcome="success",
                session_id=session_id,
                trace_id=trace_id
            )
            await audit_logger.log_event(event)

        # Check correlation tracking
        assert session_id in audit_logger.session_events
        assert len(audit_logger.session_events[session_id]) == 3

        assert trace_id in audit_logger.trace_events
        assert len(audit_logger.trace_events[trace_id]) == 3

    @pytest.mark.asyncio
    async def test_integrity_verification(self, audit_logger, sample_event):
        """Test audit trail integrity verification."""
        # Log multiple events
        events = []
        for i in range(5):
            event = AuditEvent(
                event_id=f"integrity_{i}",
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.DATA_ACCESS,
                audit_level=AuditLevel.INFO,
                user_id=f"user_{i}",
                source_ip="192.168.1.100",
                resource=f"/api/resource_{i}",
                action="read",
                outcome="success"
            )
            await audit_logger.log_event(event)
            events.append(event)

        # Verify integrity
        integrity_result = await audit_logger.verify_audit_trail_integrity()

        assert integrity_result['total_events'] == 5
        assert integrity_result['verified_events'] == 5
        assert len(integrity_result['corrupted_events']) == 0
        assert integrity_result['integrity_percentage'] == 100.0

    @pytest.mark.asyncio
    async def test_integrity_verification_with_corruption(self, audit_logger, temp_db):
        """Test integrity verification detects corruption."""
        # Log an event
        event = AuditEvent(
            event_id="corrupt_test",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.DATA_ACCESS,
            audit_level=AuditLevel.INFO,
            user_id="test_user",
            source_ip="192.168.1.100",
            resource="/api/documents",
            action="read",
            outcome="success"
        )
        await audit_logger.log_event(event)

        # Manually corrupt the database record
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE audit_events
                SET action = 'corrupted_action'
                WHERE event_id = 'corrupt_test'
            """)
            conn.commit()

        # Integrity check should detect corruption
        integrity_result = await audit_logger.verify_audit_trail_integrity()

        assert integrity_result['total_events'] == 1
        assert integrity_result['verified_events'] == 0
        assert len(integrity_result['corrupted_events']) == 1
        assert integrity_result['integrity_percentage'] == 0.0

    @pytest.mark.asyncio
    async def test_cleanup_old_events(self, audit_logger):
        """Test cleanup of old audit events."""
        # Set short retention for testing
        audit_logger.retention_days = 1

        # Create old and recent events
        old_event = AuditEvent(
            event_id="old_event",
            timestamp=datetime.utcnow() - timedelta(days=2),
            event_type=AuditEventType.DATA_ACCESS,
            audit_level=AuditLevel.INFO,
            user_id="test_user",
            source_ip="192.168.1.100",
            resource="/api/old",
            action="read",
            outcome="success"
        )

        recent_event = AuditEvent(
            event_id="recent_event",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.DATA_ACCESS,
            audit_level=AuditLevel.INFO,
            user_id="test_user",
            source_ip="192.168.1.100",
            resource="/api/recent",
            action="read",
            outcome="success"
        )

        await audit_logger.log_event(old_event)
        await audit_logger.log_event(recent_event)

        # Cleanup should remove old event
        cleaned_count = await audit_logger.cleanup_old_events()
        assert cleaned_count == 1

        # Verify only recent event remains
        remaining_events = await audit_logger.query_events(limit=10)
        assert len(remaining_events) == 1
        assert remaining_events[0].event_id == "recent_event"

    @pytest.mark.asyncio
    async def test_concurrent_logging(self, audit_logger):
        """Test concurrent event logging."""
        # Create multiple concurrent logging tasks
        tasks = []
        for i in range(10):
            event = AuditEvent(
                event_id=f"concurrent_{i}",
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.DATA_ACCESS,
                audit_level=AuditLevel.INFO,
                user_id=f"user_{i}",
                source_ip="192.168.1.100",
                resource=f"/api/resource_{i}",
                action="read",
                outcome="success"
            )
            tasks.append(audit_logger.log_event(event))

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)

        # All events should be logged
        events = await audit_logger.query_events(limit=20)
        assert len(events) == 10

        # Check database consistency
        with sqlite3.connect(audit_logger.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM audit_events")
            count = cursor.fetchone()[0]
            assert count == 10

    def test_database_initialization(self, temp_db):
        """Test database initialization and schema creation."""
        # Create logger to initialize database
        logger = AuditLogger(database_path=temp_db)

        # Check that tables were created
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()

            # Check audit_events table
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='audit_events'
            """)
            assert cursor.fetchone() is not None

            # Check indexes
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND tbl_name='audit_events'
            """)
            indexes = cursor.fetchall()
            assert len(indexes) >= 4  # Should have multiple indexes

    @pytest.mark.asyncio
    async def test_database_persistence_error_handling(self, audit_logger):
        """Test handling of database persistence errors."""
        # Create event
        event = AuditEvent(
            event_id="error_test",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.DATA_ACCESS,
            audit_level=AuditLevel.INFO,
            user_id="test_user",
            source_ip="192.168.1.100",
            resource="/api/test",
            action="read",
            outcome="success"
        )

        # Mock database error
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")):
            # Should not crash on database error
            await audit_logger.log_event(event)

            # Event should still be in memory buffer
            assert len(audit_logger.event_buffer) == 1


class TestComplianceReporter:
    """Test compliance reporting functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.fixture
    def audit_logger(self, temp_db):
        """Create audit logger with temporary database."""
        return AuditLogger(database_path=temp_db)

    @pytest.fixture
    def compliance_reporter(self, audit_logger):
        """Create compliance reporter."""
        return ComplianceReporter(audit_logger)

    @pytest.fixture
    async def sample_audit_data(self, audit_logger):
        """Create sample audit data for compliance testing."""
        events = [
            # Data access events for GDPR compliance
            AuditEvent(
                event_id="gdpr_001",
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.DATA_ACCESS,
                audit_level=AuditLevel.INFO,
                user_id="user123",
                source_ip="192.168.1.100",
                resource="/api/users/personal_data",
                action="read",
                outcome="success",
                details={'data_type': 'personal'}
            ),

            # Configuration change for SOX compliance
            AuditEvent(
                event_id="sox_001",
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.CONFIGURATION_CHANGE,
                audit_level=AuditLevel.WARNING,
                user_id="admin",
                source_ip="192.168.1.200",
                resource="/config/database",
                action="update",
                outcome="success",
                details={'approver': 'manager123', 'change_ticket': 'CHG-001'}
            ),

            # Authentication events for ISO 27001
            AuditEvent(
                event_id="iso_001",
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.USER_AUTHENTICATION,
                audit_level=AuditLevel.INFO,
                user_id="user456",
                source_ip="192.168.1.101",
                resource="/login",
                action="authenticate",
                outcome="success"
            ),

            AuditEvent(
                event_id="iso_002",
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.USER_AUTHENTICATION,
                audit_level=AuditLevel.WARNING,
                user_id="user789",
                source_ip="192.168.1.102",
                resource="/login",
                action="authenticate",
                outcome="failed",
                details={'failure_reason': 'invalid_password'}
            )
        ]

        for event in events:
            await audit_logger.log_event(event)

        return events

    @pytest.mark.asyncio
    async def test_gdpr_compliance_report(self, compliance_reporter, sample_audit_data):
        """Test GDPR compliance report generation."""
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow() + timedelta(hours=1)

        report = await compliance_reporter.generate_compliance_report(
            framework=ComplianceFramework.GDPR,
            start_date=start_date,
            end_date=end_date
        )

        assert report['framework'] == 'gdpr'
        assert report['events_analyzed'] >= 1
        assert 'compliance_score' in report
        assert 'rule_results' in report
        assert len(report['rule_results']) > 0

        # Check for GDPR-specific rules
        rule_ids = {rule['rule_id'] for rule in report['rule_results']}
        assert 'GDPR_001' in rule_ids  # Data access logging rule

    @pytest.mark.asyncio
    async def test_sox_compliance_report(self, compliance_reporter, sample_audit_data):
        """Test SOX compliance report generation."""
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow() + timedelta(hours=1)

        report = await compliance_reporter.generate_compliance_report(
            framework=ComplianceFramework.SOX,
            start_date=start_date,
            end_date=end_date
        )

        assert report['framework'] == 'sox'
        assert 'compliance_score' in report
        assert len(report['rule_results']) > 0

        # Check for SOX-specific rules
        rule_ids = {rule['rule_id'] for rule in report['rule_results']}
        assert 'SOX_001' in rule_ids  # Configuration change control

    @pytest.mark.asyncio
    async def test_data_access_logging_check(self, compliance_reporter, sample_audit_data):
        """Test GDPR data access logging compliance check."""
        events = await compliance_reporter.audit_logger.query_events(limit=100)

        rule = ComplianceRule(
            rule_id="TEST_GDPR_001",
            framework=ComplianceFramework.GDPR,
            title="Test Data Access Logging",
            description="Test rule",
            requirement="Test requirement",
            check_function="check_data_access_logging",
            severity=AuditLevel.ERROR
        )

        result = await compliance_reporter.check_data_access_logging(rule, events)

        assert result['rule_id'] == "TEST_GDPR_001"
        assert result['status'] in ['PASS', 'FAIL']
        assert 'evidence' in result
        assert result['evidence']['total_data_access_events'] >= 0

    @pytest.mark.asyncio
    async def test_configuration_change_check(self, compliance_reporter, sample_audit_data):
        """Test SOX configuration change compliance check."""
        events = await compliance_reporter.audit_logger.query_events(limit=100)

        rule = ComplianceRule(
            rule_id="TEST_SOX_001",
            framework=ComplianceFramework.SOX,
            title="Test Configuration Change Control",
            description="Test rule",
            requirement="Test requirement",
            check_function="check_configuration_changes",
            severity=AuditLevel.CRITICAL
        )

        result = await compliance_reporter.check_configuration_changes(rule, events)

        assert result['rule_id'] == "TEST_SOX_001"
        assert result['status'] in ['PASS', 'FAIL']
        assert 'evidence' in result

        # Should pass because sample data has authorized changes
        if result['evidence']['total_config_changes'] > 0:
            assert result['evidence']['authorized_changes'] > 0

    @pytest.mark.asyncio
    async def test_access_control_logging_check(self, compliance_reporter, sample_audit_data):
        """Test ISO 27001 access control logging check."""
        events = await compliance_reporter.audit_logger.query_events(limit=100)

        rule = ComplianceRule(
            rule_id="TEST_ISO_001",
            framework=ComplianceFramework.ISO_27001,
            title="Test Access Control Monitoring",
            description="Test rule",
            requirement="Test requirement",
            check_function="check_access_control_logging",
            severity=AuditLevel.ERROR
        )

        result = await compliance_reporter.check_access_control_logging(rule, events)

        assert result['rule_id'] == "TEST_ISO_001"
        assert result['status'] in ['PASS', 'FAIL', 'INFO']
        assert 'evidence' in result

    @pytest.mark.asyncio
    async def test_export_audit_trail_csv(self, compliance_reporter, sample_audit_data, temp_db):
        """Test CSV export functionality."""
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow() + timedelta(hours=1)

        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_path = Path(temp_file.name)

        try:
            output_path = await compliance_reporter.export_audit_trail(
                start_date=start_date,
                end_date=end_date,
                format="csv",
                output_path=temp_path
            )

            assert output_path.exists()
            assert output_path.suffix == '.csv'

            # Verify CSV content
            content = temp_path.read_text()
            assert 'event_id' in content  # Header
            assert 'gdpr_001' in content or 'sox_001' in content  # Data

        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_export_audit_trail_json(self, compliance_reporter, sample_audit_data):
        """Test JSON export functionality."""
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow() + timedelta(hours=1)

        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_path = Path(temp_file.name)

        try:
            output_path = await compliance_reporter.export_audit_trail(
                start_date=start_date,
                end_date=end_date,
                format="json",
                output_path=temp_path
            )

            assert output_path.exists()
            assert output_path.suffix == '.json'

            # Verify JSON content
            with open(temp_path) as f:
                data = json.load(f)

            assert 'export_timestamp' in data
            assert 'event_count' in data
            assert 'events' in data
            assert isinstance(data['events'], list)

        finally:
            temp_path.unlink(missing_ok=True)

    def test_compliance_rule_validation(self):
        """Test compliance rule validation."""
        # Valid rule
        valid_rule = ComplianceRule(
            rule_id="VALID_001",
            framework=ComplianceFramework.GDPR,
            title="Valid Rule",
            description="A valid compliance rule",
            requirement="Test requirement",
            check_function="test_check",
            severity=AuditLevel.ERROR
        )

        assert valid_rule.rule_id == "VALID_001"
        assert valid_rule.enabled == True
        assert len(valid_rule.remediation_advice) == 0

        # Rule with remediation advice
        rule_with_advice = ComplianceRule(
            rule_id="ADVICE_001",
            framework=ComplianceFramework.SOX,
            title="Rule with Advice",
            description="Rule with remediation advice",
            requirement="Test requirement",
            check_function="test_check",
            severity=AuditLevel.WARNING,
            remediation_advice=["Step 1", "Step 2"]
        )

        assert len(rule_with_advice.remediation_advice) == 2

    @pytest.mark.asyncio
    async def test_rule_evaluation_error_handling(self, compliance_reporter):
        """Test error handling in rule evaluation."""
        # Rule with non-existent check function
        invalid_rule = ComplianceRule(
            rule_id="INVALID_001",
            framework=ComplianceFramework.GDPR,
            title="Invalid Rule",
            description="Rule with missing check function",
            requirement="Test requirement",
            check_function="non_existent_function",
            severity=AuditLevel.ERROR
        )

        result = await compliance_reporter._evaluate_compliance_rule(invalid_rule, [])

        assert result['rule_id'] == "INVALID_001"
        assert result['status'] == 'ERROR'
        assert 'not found' in result['message']

    def test_recommendations_generation(self, compliance_reporter):
        """Test compliance recommendations generation."""
        # Test with failed rules
        failed_results = [
            {'rule_id': 'FAIL_001', 'status': 'FAIL', 'message': 'Rule failed'},
            {'rule_id': 'PASS_001', 'status': 'PASS', 'message': 'Rule passed'},
            {'rule_id': 'ERROR_001', 'status': 'ERROR', 'message': 'Rule error'}
        ]

        recommendations = compliance_reporter._generate_recommendations(failed_results)

        assert len(recommendations) > 0
        assert any('failed compliance checks' in rec for rec in recommendations)
        assert any('rule evaluation errors' in rec for rec in recommendations)

        # Test with all passed rules
        passed_results = [
            {'rule_id': 'PASS_001', 'status': 'PASS', 'message': 'Rule passed'},
            {'rule_id': 'PASS_002', 'status': 'PASS', 'message': 'Rule passed'}
        ]

        good_recommendations = compliance_reporter._generate_recommendations(passed_results)
        assert any('maintain current' in rec.lower() for rec in good_recommendations)


class TestAuditTrail:
    """Test high-level audit trail integration."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.fixture
    def audit_trail(self, temp_db):
        """Create audit trail system."""
        return AuditTrail(database_path=temp_db, retention_days=30)

    @pytest.fixture
    def sample_security_event(self):
        """Create sample security event."""
        return SecurityEvent(
            event_id="sec_001",
            timestamp=datetime.utcnow(),
            event_type="query",
            source_ip="192.168.1.100",
            user_id="test_user",
            collection="documents",
            query="SELECT * FROM sensitive_data WHERE user = 'admin'"
        )

    @pytest.fixture
    def sample_threat(self):
        """Create sample threat detection."""
        return ThreatDetection(
            threat_id="threat_001",
            threat_type=ThreatType.SQL_INJECTION,
            threat_level=ThreatLevel.HIGH,
            confidence=0.85,
            description="SQL injection attempt detected",
            source_events=[],
            mitigation_suggestions=["Sanitize inputs", "Use parameterized queries"]
        )

    @pytest.fixture
    def sample_alert(self):
        """Create sample security alert."""
        return SecurityAlert(
            alert_id="alert_001",
            alert_level=AlertLevel.ERROR,
            title="High Threat Count",
            description="Multiple threats detected in short time",
            source="threat_detection",
            metadata={'threat_count': 5}
        )

    @pytest.mark.asyncio
    async def test_log_security_event(self, audit_trail, sample_security_event):
        """Test logging security events."""
        await audit_trail.log_security_event(sample_security_event)

        # Should create audit event
        events = await audit_trail.audit_logger.query_events(limit=10)
        assert len(events) == 1

        logged_event = events[0]
        assert logged_event.event_type == AuditEventType.SECURITY_EVENT
        assert logged_event.user_id == "test_user"
        assert logged_event.source_ip == "192.168.1.100"
        assert logged_event.resource == "documents"

    @pytest.mark.asyncio
    async def test_log_threat_detection(self, audit_trail, sample_threat):
        """Test logging threat detections."""
        await audit_trail.log_threat_detection(sample_threat)

        # Should create audit event
        events = await audit_trail.audit_logger.query_events(limit=10)
        assert len(events) == 1

        logged_event = events[0]
        assert logged_event.event_type == AuditEventType.THREAT_DETECTION
        assert logged_event.audit_level == AuditLevel.ERROR  # HIGH threat -> ERROR audit
        assert logged_event.action == "threat_detection"
        assert logged_event.outcome == "sql_injection_detected"

    @pytest.mark.asyncio
    async def test_log_security_alert(self, audit_trail, sample_alert):
        """Test logging security alerts."""
        await audit_trail.log_security_alert(sample_alert)

        # Should create audit event
        events = await audit_trail.audit_logger.query_events(limit=10)
        assert len(events) == 1

        logged_event = events[0]
        assert logged_event.event_type == AuditEventType.SECURITY_EVENT
        assert logged_event.audit_level == AuditLevel.ERROR  # ERROR alert -> ERROR audit
        assert logged_event.action == "security_alert"
        assert logged_event.outcome == "alert_generated"

    @pytest.mark.asyncio
    async def test_generate_compliance_dashboard(self, audit_trail, sample_security_event, sample_threat, sample_alert):
        """Test compliance dashboard generation."""
        # Log various events
        await audit_trail.log_security_event(sample_security_event)
        await audit_trail.log_threat_detection(sample_threat)
        await audit_trail.log_security_alert(sample_alert)

        dashboard = await audit_trail.generate_compliance_dashboard()

        assert 'reporting_period' in dashboard
        assert 'total_events' in dashboard
        assert 'event_breakdown' in dashboard
        assert 'active_users' in dashboard
        assert 'compliance_frameworks' in dashboard
        assert 'overall_compliance_score' in dashboard

        # Should have events
        assert dashboard['total_events'] >= 3
        assert dashboard['active_users'] >= 1

        # Should have compliance framework reports
        assert len(dashboard['compliance_frameworks']) > 0

    @pytest.mark.asyncio
    async def test_threat_level_to_audit_level_mapping(self, audit_trail):
        """Test proper mapping of threat levels to audit levels."""
        # Test all threat levels
        threat_levels = [
            (ThreatLevel.LOW, AuditLevel.INFO),
            (ThreatLevel.MEDIUM, AuditLevel.WARNING),
            (ThreatLevel.HIGH, AuditLevel.ERROR),
            (ThreatLevel.CRITICAL, AuditLevel.CRITICAL)
        ]

        for threat_level, expected_audit_level in threat_levels:
            threat = ThreatDetection(
                threat_id=f"threat_level_{threat_level.value}",
                threat_type=ThreatType.SUSPICIOUS_QUERY,
                threat_level=threat_level,
                confidence=0.8,
                description=f"Test threat at {threat_level.name} level",
                source_events=[],
                mitigation_suggestions=[]
            )

            await audit_trail.log_threat_detection(threat)

            # Get the logged event
            events = await audit_trail.audit_logger.query_events(
                limit=1
            )

            assert len(events) >= 1
            logged_event = events[0]
            assert logged_event.audit_level == expected_audit_level

    @pytest.mark.asyncio
    async def test_integration_with_compliance_frameworks(self, audit_trail):
        """Test integration with compliance frameworks."""
        # Create events that should trigger different compliance tags
        events_to_log = [
            # GDPR relevant
            SecurityEvent(
                event_id="gdpr_event",
                timestamp=datetime.utcnow(),
                event_type="data_access",
                source_ip="192.168.1.100",
                user_id="user123",
                collection="personal_data",
                query="SELECT * FROM users WHERE email = 'test@example.com'"
            ),

            # SOX relevant (via configuration change audit event)
        ]

        for event in events_to_log:
            await audit_trail.log_security_event(event)

        # Generate dashboard to check compliance framework integration
        dashboard = await audit_trail.generate_compliance_dashboard()

        # Should have processed events for compliance
        assert dashboard['total_events'] >= len(events_to_log)

        # Should have generated compliance reports
        frameworks = dashboard['compliance_frameworks']
        assert len(frameworks) > 0

        # Check for specific frameworks
        if 'gdpr' in frameworks:
            gdpr_report = frameworks['gdpr']
            assert 'compliance_score' in gdpr_report
            assert gdpr_report['events_analyzed'] >= 0


class TestAuditFrameworkEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_database_connection_failure(self):
        """Test handling of database connection failures."""
        # Use invalid database path
        invalid_path = Path("/invalid/path/audit.db")

        with pytest.raises(Exception):
            # Should raise exception on invalid database path
            AuditLogger(database_path=invalid_path)

    @pytest.mark.asyncio
    async def test_memory_buffer_overflow(self):
        """Test memory buffer overflow handling."""
        # Create logger with small buffer
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()

        try:
            audit_logger = AuditLogger(
                database_path=temp_path,
                max_memory_events=5  # Small buffer
            )

            # Add more events than buffer size
            for i in range(10):
                event = AuditEvent(
                    event_id=f"overflow_{i}",
                    timestamp=datetime.utcnow(),
                    event_type=AuditEventType.DATA_ACCESS,
                    audit_level=AuditLevel.INFO,
                    user_id=f"user_{i}",
                    source_ip="192.168.1.100",
                    resource=f"/api/resource_{i}",
                    action="read",
                    outcome="success"
                )
                await audit_logger.log_event(event)

            # Buffer should be bounded
            assert len(audit_logger.event_buffer) == 5

            # All events should still be in database
            events = await audit_logger.query_events(limit=20)
            assert len(events) == 10

        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_invalid_export_format(self):
        """Test handling of invalid export formats."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()

        try:
            audit_logger = AuditLogger(database_path=temp_path)
            reporter = ComplianceReporter(audit_logger)

            with pytest.raises(ValueError, match="Unsupported export format"):
                await reporter.export_audit_trail(
                    start_date=datetime.utcnow() - timedelta(hours=1),
                    end_date=datetime.utcnow(),
                    format="invalid_format"
                )

        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_empty_query_results(self):
        """Test handling of empty query results."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()

        try:
            audit_logger = AuditLogger(database_path=temp_path)
            reporter = ComplianceReporter(audit_logger)

            # Generate report with no data
            report = await reporter.generate_compliance_report(
                framework=ComplianceFramework.GDPR,
                start_date=datetime.utcnow() - timedelta(hours=1),
                end_date=datetime.utcnow()
            )

            assert report['events_analyzed'] == 0
            assert report['compliance_score'] >= 0  # Should handle empty data gracefully

        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_malformed_audit_event_data(self):
        """Test handling of malformed audit event data."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()

        try:
            audit_logger = AuditLogger(database_path=temp_path)

            # Create event with extreme values
            extreme_event = AuditEvent(
                event_id="x" * 1000,  # Very long ID
                timestamp=datetime.max - timedelta(days=1),  # Edge case timestamp
                event_type=AuditEventType.DATA_ACCESS,
                audit_level=AuditLevel.INFO,
                user_id=None,
                source_ip="999.999.999.999",  # Invalid IP
                resource="/" + "x" * 10000,  # Very long resource path
                action="x" * 1000,  # Very long action
                outcome="x" * 1000,  # Very long outcome
                details={"key": "x" * 50000}  # Very large details
            )

            # Should handle extreme values without crashing
            await audit_logger.log_event(extreme_event)

            # Should be queryable
            events = await audit_logger.query_events(limit=1)
            assert len(events) == 1

        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])