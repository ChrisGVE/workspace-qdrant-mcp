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

from .certificate_validator import (
    EnhancedCertificateValidator,
    CertificateSecurityPolicy,
    CertificatePinningError,
    CertificateValidationError,
    create_secure_ssl_context,
)

from .threat_detection import (
    ThreatDetectionEngine,
    BehavioralAnalyzer,
    AnomalyDetector,
    ThreatAnalyzer,
    ThreatLevel,
    ThreatType,
    ThreatEvent,
    BehavioralPattern,
    AnomalyType,
)

from .security_monitor import (
    SecurityMonitor,
    SecurityMetrics,
    AlertingSystem,
    SecurityEventLogger,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
)

from .audit_framework import (
    AuditLogger,
    ComplianceReporter,
    AuditTrail,
    ComplianceFramework,
    AuditEvent,
    ComplianceRule,
    ComplianceStatus,
)

from .access_control import (
    RoleBasedAccessControl,
    SessionManager,
    Permission,
    AccessResult,
    Role,
    User,
    Session,
    AccessContext,
    get_rbac,
    require_permission,
)

from .encryption import (
    KeyManager,
    EncryptionEngine,
    SecureCommunication,
    SecureChannel,
    EncryptionKey,
    EncryptedData,
    EncryptionAlgorithm,
    HashAlgorithm,
    KeyDerivationFunction,
    CryptoError,
    KeyManagementError,
    EncryptionError,
    DecryptionError,
    encrypt_json,
    decrypt_json,
    secure_temp_key,
    generate_secure_token,
    constant_time_compare,
    get_key_manager,
    get_encryption_engine,
    get_secure_communication,
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
    'ThreatEvent',
    'BehavioralPattern',
    'AnomalyType',

    # Security monitoring
    'SecurityMonitor',
    'SecurityMetrics',
    'AlertingSystem',
    'SecurityEventLogger',
    'AlertRule',
    'AlertSeverity',
    'AlertStatus',
    'NotificationChannel',

    # Audit framework
    'AuditLogger',
    'ComplianceReporter',
    'AuditTrail',
    'ComplianceFramework',
    'AuditEvent',
    'ComplianceRule',
    'ComplianceStatus',

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