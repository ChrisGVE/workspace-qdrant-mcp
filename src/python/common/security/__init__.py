"""Security modules for workspace-qdrant-mcp.

This package provides comprehensive security features including:
- Enhanced TLS certificate validation and secure communication
- Advanced threat detection with behavioral analysis
- Real-time security monitoring and alerting
- Enterprise audit framework with compliance reporting
- Role-based access control (RBAC) with session management
- End-to-end encryption and secure key management
- Authentication token management
- Input sanitization and validation
- Rate limiting and DoS protection
- Security policy enforcement
"""

from .access_control import (
    AccessContext,
    AccessResult,
    Permission,
    Role,
    RoleBasedAccessControl,
    Session,
    SessionManager,
    User,
    get_rbac,
    require_permission,
)
from .audit_framework import (
    AuditEvent,
    AuditEventType,
    AuditLevel,
    AuditLogger,
    AuditTrail,
    ComplianceFramework,
    ComplianceReporter,
    ComplianceRule,
)
from .certificate_validator import (
    CertificatePinningError,
    CertificateSecurityPolicy,
    CertificateValidationError,
    EnhancedCertificateValidator,
    create_secure_ssl_context,
)
from .encryption import (
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
from .security_monitor import (
    AlertingSystem,
    AlertLevel,
    MetricType,
    SecurityAlert,
    SecurityEventLogger,
    SecurityMetric,
    SecurityMetrics,
    SecurityMonitor,
)
from .threat_detection import (
    AnomalyDetector,
    BehavioralAnalyzer,
    SecurityEvent,
    ThreatAnalyzer,
    ThreatDetection,
    ThreatDetectionEngine,
    ThreatLevel,
    ThreatType,
)

__all__ = [
    # Certificate validation
    'EnhancedCertificateValidator',
    'CertificateSecurityPolicy',
    'CertificatePinningError',
    'CertificateValidationError',
    'create_secure_ssl_context',

    # Threat detection
    'ThreatDetectionEngine',
    'BehavioralAnalyzer',
    'AnomalyDetector',
    'ThreatAnalyzer',
    'ThreatLevel',
    'ThreatType',
    'SecurityEvent',
    'ThreatDetection',

    # Security monitoring
    'SecurityMonitor',
    'SecurityMetrics',
    'AlertingSystem',
    'SecurityEventLogger',
    'SecurityAlert',
    'SecurityMetric',
    'AlertLevel',
    'MetricType',

    # Audit framework
    'AuditLogger',
    'ComplianceReporter',
    'AuditTrail',
    'ComplianceFramework',
    'AuditEvent',
    'ComplianceRule',
    'AuditLevel',
    'AuditEventType',

    # Access control
    'RoleBasedAccessControl',
    'SessionManager',
    'Permission',
    'AccessResult',
    'Role',
    'User',
    'Session',
    'AccessContext',
    'get_rbac',
    'require_permission',

    # Encryption
    'KeyManager',
    'EncryptionEngine',
    'SecureCommunication',
    'SecureChannel',
    'EncryptionKey',
    'EncryptedData',
    'EncryptionAlgorithm',
    'HashAlgorithm',
    'KeyDerivationFunction',
    'CryptoError',
    'KeyManagementError',
    'EncryptionError',
    'DecryptionError',
    'encrypt_json',
    'decrypt_json',
    'secure_temp_key',
    'generate_secure_token',
    'constant_time_compare',
    'get_key_manager',
    'get_encryption_engine',
    'get_secure_communication',
]
