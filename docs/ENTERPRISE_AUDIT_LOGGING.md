# Enterprise Audit Logging System

**Version**: 1.0  
**Target Audience**: Compliance Officers, Security Teams, System Administrators  
**Prerequisites**: workspace-qdrant-mcp v1.0+, Enterprise Authentication & RBAC

## Overview

This document provides comprehensive guidance for implementing enterprise-grade audit logging in workspace-qdrant-mcp. It covers security event logging, compliance reporting, audit trail management, and regulatory compliance patterns for enterprise deployments.

## Table of Contents

- [Audit Architecture](#audit-architecture)
- [Event Classification](#event-classification)
- [Logging Framework](#logging-framework)
- [Compliance Standards](#compliance-standards)
- [Audit Trail Management](#audit-trail-management)
- [Real-time Monitoring](#real-time-monitoring)
- [Retention and Archival](#retention-and-archival)
- [Reporting and Analytics](#reporting-and-analytics)
- [Security and Integrity](#security-and-integrity)
- [Implementation Examples](#implementation-examples)

## Audit Architecture

### High-Level Audit System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Application Layer                      │
├─────────────────────────────────────────────────────────┤
│                  Audit Interceptor                     │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │ Authentication│  │ Authorization │  │   Data       │ │
│  │   Events      │  │    Events     │  │  Operations  │ │
│  └───────────────┘  └───────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────┤
│                  Event Processing Layer                │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │Event Enricher │  │Event Validator│  │Event Filter  │ │
│  │& Correlator   │  │& Sanitizer    │  │& Router      │ │
│  └───────────────┘  └───────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────┤
│                    Storage Layer                       │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │  Primary      │  │   Compliance  │  │   Archive    │ │
│  │  Audit Log    │  │   Database    │  │   Storage    │ │
│  │ (PostgreSQL)  │  │  (Immutable)  │  │   (S3/GCS)   │ │
│  └───────────────┘  └───────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────┤
│                  Analytics & Reporting                 │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │  Real-time    │  │  Compliance   │  │   Security   │ │
│  │  Dashboard    │  │   Reports     │  │   Alerts     │ │
│  │ (Grafana)     │  │  (Custom)     │  │(Prometheus)  │ │
│  └───────────────┘  └───────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Core Components

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import hashlib
import uuid

class EventSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class EventCategory(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ADMINISTRATION = "system_administration"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"
    COMPLIANCE = "compliance"

class AuditOutcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"

@dataclass
class AuditEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: str = ""
    category: EventCategory = EventCategory.DATA_ACCESS
    severity: EventSeverity = EventSeverity.INFO
    outcome: AuditOutcome = AuditOutcome.SUCCESS
    
    # Actor information
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    
    # Action details
    action: Optional[str] = None
    description: str = ""
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    # Context information
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    source_system: str = "workspace-qdrant-mcp"
    
    # Integrity fields
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Generate event checksum for integrity verification"""
        if not self.checksum:
            self.checksum = self._generate_checksum()
    
    def _generate_checksum(self) -> str:
        """Generate SHA-256 checksum of event data"""
        # Create deterministic string representation
        event_data = {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'user_id': self.user_id,
            'action': self.action,
            'resource_id': self.resource_id,
            'outcome': self.outcome.value
        }
        
        event_string = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for storage"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'category': self.category.value,
            'severity': self.severity.value,
            'outcome': self.outcome.value,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'resource_name': self.resource_name,
            'action': self.action,
            'description': self.description,
            'additional_data': self.additional_data,
            'request_id': self.request_id,
            'correlation_id': self.correlation_id,
            'source_system': self.source_system,
            'checksum': self.checksum
        }
```

## Event Classification

### Authentication Events

```python
class AuthenticationEventBuilder:
    @staticmethod
    def login_attempt(user_id: str, ip_address: str, success: bool,
                     auth_method: str = "password") -> AuditEvent:
        """Create login attempt audit event"""
        return AuditEvent(
            event_type="user_login_attempt",
            category=EventCategory.AUTHENTICATION,
            severity=EventSeverity.INFO if success else EventSeverity.WARNING,
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            user_id=user_id,
            ip_address=ip_address,
            action="login",
            description=f"User login attempt via {auth_method}",
            additional_data={
                "authentication_method": auth_method,
                "login_successful": success
            }
        )
    
    @staticmethod
    def logout(user_id: str, session_id: str, session_duration: int) -> AuditEvent:
        """Create logout audit event"""
        return AuditEvent(
            event_type="user_logout",
            category=EventCategory.AUTHENTICATION,
            severity=EventSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            user_id=user_id,
            session_id=session_id,
            action="logout",
            description="User logged out",
            additional_data={
                "session_duration_seconds": session_duration
            }
        )
    
    @staticmethod
    def mfa_challenge(user_id: str, mfa_method: str, success: bool) -> AuditEvent:
        """Create MFA challenge audit event"""
        return AuditEvent(
            event_type="mfa_challenge",
            category=EventCategory.AUTHENTICATION,
            severity=EventSeverity.INFO if success else EventSeverity.WARNING,
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            user_id=user_id,
            action="mfa_verification",
            description=f"MFA challenge using {mfa_method}",
            additional_data={
                "mfa_method": mfa_method,
                "verification_successful": success
            }
        )
    
    @staticmethod
    def password_change(user_id: str, changed_by: str, forced: bool = False) -> AuditEvent:
        """Create password change audit event"""
        return AuditEvent(
            event_type="password_change",
            category=EventCategory.AUTHENTICATION,
            severity=EventSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            user_id=user_id,
            action="password_update",
            description="User password changed",
            additional_data={
                "changed_by": changed_by,
                "forced_change": forced
            }
        )

class AuthorizationEventBuilder:
    @staticmethod
    def access_granted(user_id: str, resource_type: str, resource_id: str,
                      permission: str, tenant_id: str = None) -> AuditEvent:
        """Create access granted audit event"""
        return AuditEvent(
            event_type="access_granted",
            category=EventCategory.AUTHORIZATION,
            severity=EventSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=f"access_{permission}",
            description=f"Access granted to {resource_type}",
            additional_data={
                "permission_granted": permission
            }
        )
    
    @staticmethod
    def access_denied(user_id: str, resource_type: str, resource_id: str,
                     attempted_action: str, reason: str,
                     tenant_id: str = None) -> AuditEvent:
        """Create access denied audit event"""
        return AuditEvent(
            event_type="access_denied",
            category=EventCategory.AUTHORIZATION,
            severity=EventSeverity.WARNING,
            outcome=AuditOutcome.FAILURE,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=f"attempted_{attempted_action}",
            description=f"Access denied to {resource_type}: {reason}",
            additional_data={
                "attempted_action": attempted_action,
                "denial_reason": reason
            }
        )
    
    @staticmethod
    def privilege_escalation(user_id: str, from_role: str, to_role: str,
                           authorized_by: str, tenant_id: str = None) -> AuditEvent:
        """Create privilege escalation audit event"""
        return AuditEvent(
            event_type="privilege_escalation",
            category=EventCategory.AUTHORIZATION,
            severity=EventSeverity.WARNING,
            outcome=AuditOutcome.SUCCESS,
            user_id=user_id,
            tenant_id=tenant_id,
            action="role_change",
            description=f"User role changed from {from_role} to {to_role}",
            additional_data={
                "previous_role": from_role,
                "new_role": to_role,
                "authorized_by": authorized_by
            }
        )

class DataEventBuilder:
    @staticmethod
    def document_access(user_id: str, collection_name: str, document_id: str,
                       access_type: str, tenant_id: str = None) -> AuditEvent:
        """Create document access audit event"""
        return AuditEvent(
            event_type="document_access",
            category=EventCategory.DATA_ACCESS,
            severity=EventSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type="document",
            resource_id=document_id,
            resource_name=collection_name,
            action=access_type,
            description=f"Document {access_type} in collection {collection_name}",
            additional_data={
                "collection_name": collection_name,
                "access_method": access_type
            }
        )
    
    @staticmethod
    def search_query(user_id: str, query: str, collections: List[str],
                    result_count: int, tenant_id: str = None) -> AuditEvent:
        """Create search query audit event"""
        return AuditEvent(
            event_type="search_query",
            category=EventCategory.DATA_ACCESS,
            severity=EventSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type="collection",
            action="search",
            description=f"Search performed across {len(collections)} collections",
            additional_data={
                "search_query": query[:200],  # Truncate for security
                "collections_searched": collections,
                "result_count": result_count
            }
        )
    
    @staticmethod
    def data_export(user_id: str, collection_name: str, export_format: str,
                   record_count: int, tenant_id: str = None) -> AuditEvent:
        """Create data export audit event"""
        return AuditEvent(
            event_type="data_export",
            category=EventCategory.DATA_ACCESS,
            severity=EventSeverity.WARNING,  # Data export is sensitive
            outcome=AuditOutcome.SUCCESS,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type="collection",
            resource_name=collection_name,
            action="export",
            description=f"Data exported from {collection_name} in {export_format} format",
            additional_data={
                "export_format": export_format,
                "record_count": record_count,
                "export_timestamp": datetime.utcnow().isoformat()
            }
        )
```

## Logging Framework

### Core Audit Logger

```python
import asyncio
import logging
from typing import List, Optional
from abc import ABC, abstractmethod
import asyncpg
import json

class AuditStorage(ABC):
    """Abstract base class for audit storage backends"""
    
    @abstractmethod
    async def store_event(self, event: AuditEvent) -> bool:
        """Store single audit event"""
        pass
    
    @abstractmethod
    async def store_events_batch(self, events: List[AuditEvent]) -> bool:
        """Store batch of audit events"""
        pass
    
    @abstractmethod
    async def query_events(self, filters: Dict[str, Any], 
                          limit: int = 100) -> List[Dict]:
        """Query audit events with filters"""
        pass

class PostgreSQLAuditStorage(AuditStorage):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection_pool = None
    
    async def initialize(self):
        """Initialize connection pool and create tables"""
        self.connection_pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=5,
            max_size=20
        )
        await self.create_audit_tables()
    
    async def create_audit_tables(self):
        """Create audit log tables if they don't exist"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id UUID PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            event_type VARCHAR(100) NOT NULL,
            category VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            outcome VARCHAR(20) NOT NULL,
            user_id VARCHAR(64),
            tenant_id VARCHAR(64),
            session_id VARCHAR(128),
            ip_address INET,
            user_agent TEXT,
            resource_type VARCHAR(100),
            resource_id VARCHAR(255),
            resource_name VARCHAR(255),
            action VARCHAR(100),
            description TEXT,
            additional_data JSONB,
            request_id VARCHAR(128),
            correlation_id VARCHAR(128),
            source_system VARCHAR(100),
            checksum VARCHAR(64) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_events_user_id ON audit_events(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_events_tenant_id ON audit_events(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_audit_events_category ON audit_events(category);
        CREATE INDEX IF NOT EXISTS idx_audit_events_resource_type ON audit_events(resource_type);
        CREATE INDEX IF NOT EXISTS idx_audit_events_checksum ON audit_events(checksum);
        
        -- Compliance table for immutable audit records
        CREATE TABLE IF NOT EXISTS compliance_audit_log (
            id SERIAL PRIMARY KEY,
            event_hash VARCHAR(64) NOT NULL,
            event_data JSONB NOT NULL,
            stored_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            integrity_verified BOOLEAN DEFAULT FALSE
        );
        
        CREATE INDEX IF NOT EXISTS idx_compliance_audit_hash ON compliance_audit_log(event_hash);
        """
        
        async with self.connection_pool.acquire() as conn:
            await conn.execute(create_table_sql)
    
    async def store_event(self, event: AuditEvent) -> bool:
        """Store single audit event in PostgreSQL"""
        try:
            event_dict = event.to_dict()
            
            insert_sql = """
            INSERT INTO audit_events (
                event_id, timestamp, event_type, category, severity, outcome,
                user_id, tenant_id, session_id, ip_address, user_agent,
                resource_type, resource_id, resource_name, action, description,
                additional_data, request_id, correlation_id, source_system, checksum
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                $12, $13, $14, $15, $16, $17, $18, $19, $20, $21
            )
            """
            
            async with self.connection_pool.acquire() as conn:
                await conn.execute(
                    insert_sql,
                    event.event_id, event.timestamp, event.event_type,
                    event.category.value, event.severity.value, event.outcome.value,
                    event.user_id, event.tenant_id, event.session_id,
                    event.ip_address, event.user_agent,
                    event.resource_type, event.resource_id, event.resource_name,
                    event.action, event.description,
                    json.dumps(event.additional_data),
                    event.request_id, event.correlation_id, event.source_system,
                    event.checksum
                )
                
                # Also store in compliance table
                await self.store_compliance_record(conn, event)
                
            return True
            
        except Exception as e:
            logging.error(f"Failed to store audit event: {e}")
            return False
    
    async def store_compliance_record(self, conn, event: AuditEvent):
        """Store immutable compliance record"""
        compliance_sql = """
        INSERT INTO compliance_audit_log (event_hash, event_data)
        VALUES ($1, $2)
        """
        
        event_data = event.to_dict()
        await conn.execute(
            compliance_sql,
            event.checksum,
            json.dumps(event_data)
        )

class EnterpriseAuditLogger:
    def __init__(self, storage_backends: List[AuditStorage],
                 buffer_size: int = 1000,
                 flush_interval: int = 30):
        self.storage_backends = storage_backends
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.event_buffer = []
        self.buffer_lock = asyncio.Lock()
        
        # Start background flush task
        self.flush_task = asyncio.create_task(self.periodic_flush())
    
    async def log_event(self, event: AuditEvent):
        """Log audit event with buffering"""
        async with self.buffer_lock:
            self.event_buffer.append(event)
            
            # Flush if buffer is full
            if len(self.event_buffer) >= self.buffer_size:
                await self.flush_buffer()
    
    async def flush_buffer(self):
        """Flush event buffer to all storage backends"""
        if not self.event_buffer:
            return
        
        events_to_flush = self.event_buffer.copy()
        self.event_buffer.clear()
        
        # Store in all backends concurrently
        tasks = []
        for backend in self.storage_backends:
            task = asyncio.create_task(
                backend.store_events_batch(events_to_flush)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any storage failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(
                    f"Audit storage backend {i} failed: {result}"
                )
    
    async def periodic_flush(self):
        """Periodic buffer flush task"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                async with self.buffer_lock:
                    await self.flush_buffer()
            except asyncio.CancelledError:
                # Final flush on shutdown
                async with self.buffer_lock:
                    await self.flush_buffer()
                break
            except Exception as e:
                logging.error(f"Periodic flush failed: {e}")
    
    async def log_authentication_event(self, event_type: str, user_id: str,
                                     success: bool, **kwargs):
        """Convenience method for authentication events"""
        builder_map = {
            'login': AuthenticationEventBuilder.login_attempt,
            'logout': AuthenticationEventBuilder.logout,
            'mfa': AuthenticationEventBuilder.mfa_challenge,
            'password_change': AuthenticationEventBuilder.password_change
        }
        
        if event_type in builder_map:
            event = builder_map[event_type](user_id, success, **kwargs)
            await self.log_event(event)
    
    async def log_data_access_event(self, user_id: str, action: str,
                                  resource_type: str, resource_id: str,
                                  **kwargs):
        """Convenience method for data access events"""
        event = DataEventBuilder.document_access(
            user_id, resource_id, resource_type, action, **kwargs
        )
        await self.log_event(event)
    
    async def shutdown(self):
        """Graceful shutdown with buffer flush"""
        self.flush_task.cancel()
        await self.flush_task
```

### Middleware Integration

```python
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

class AuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, audit_logger: EnterpriseAuditLogger):
        super().__init__(app)
        self.audit_logger = audit_logger
        self.sensitive_endpoints = {
            '/api/search', '/api/store', '/api/delete',
            '/api/admin', '/api/collections'
        }
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Extract user context
        user_id = getattr(request.state, 'user_id', 'anonymous')
        tenant_id = getattr(request.state, 'tenant_id', None)
        session_id = getattr(request.state, 'session_id', None)
        
        # Generate correlation ID for request tracking
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Process request
        response = await call_next(request)
        
        # Log audit event for sensitive endpoints
        if request.url.path in self.sensitive_endpoints:
            duration = time.time() - start_time
            
            # Determine outcome based on status code
            outcome = AuditOutcome.SUCCESS if response.status_code < 400 else AuditOutcome.FAILURE
            severity = EventSeverity.INFO if response.status_code < 400 else EventSeverity.WARNING
            
            # Extract resource information from request
            resource_type, resource_id = self.extract_resource_info(request)
            
            event = AuditEvent(
                event_type="api_request",
                category=EventCategory.DATA_ACCESS,
                severity=severity,
                outcome=outcome,
                user_id=user_id,
                tenant_id=tenant_id,
                session_id=session_id,
                ip_address=request.client.host,
                user_agent=request.headers.get('user-agent'),
                resource_type=resource_type,
                resource_id=resource_id,
                action=request.method.lower(),
                description=f"{request.method} {request.url.path}",
                correlation_id=correlation_id,
                additional_data={
                    'http_method': request.method,
                    'endpoint': request.url.path,
                    'status_code': response.status_code,
                    'response_time_ms': round(duration * 1000, 2),
                    'query_params': dict(request.query_params),
                    'content_length': response.headers.get('content-length')
                }
            )
            
            await self.audit_logger.log_event(event)
        
        return response
    
    def extract_resource_info(self, request: Request) -> tuple:
        """Extract resource type and ID from request"""
        path_parts = request.url.path.strip('/').split('/')
        
        if 'collections' in path_parts:
            idx = path_parts.index('collections')
            if idx + 1 < len(path_parts):
                return 'collection', path_parts[idx + 1]
        
        if 'tenants' in path_parts:
            idx = path_parts.index('tenants')
            if idx + 1 < len(path_parts):
                return 'tenant', path_parts[idx + 1]
        
        return 'system', request.url.path
```

## Compliance Standards

### GDPR Compliance Logging

```python
class GDPRComplianceLogger:
    def __init__(self, audit_logger: EnterpriseAuditLogger):
        self.audit_logger = audit_logger
    
    async def log_data_processing_event(self, user_id: str, data_subject_id: str,
                                       processing_purpose: str, legal_basis: str,
                                       data_categories: List[str]):
        """Log GDPR data processing event"""
        event = AuditEvent(
            event_type="gdpr_data_processing",
            category=EventCategory.COMPLIANCE,
            severity=EventSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            user_id=user_id,
            resource_type="personal_data",
            resource_id=data_subject_id,
            action="process_personal_data",
            description=f"Processing personal data for: {processing_purpose}",
            additional_data={
                "processing_purpose": processing_purpose,
                "legal_basis": legal_basis,
                "data_categories": data_categories,
                "gdpr_compliance": True
            }
        )
        
        await self.audit_logger.log_event(event)
    
    async def log_consent_event(self, data_subject_id: str, consent_type: str,
                               granted: bool, purpose: str):
        """Log GDPR consent events"""
        event = AuditEvent(
            event_type="gdpr_consent_update",
            category=EventCategory.COMPLIANCE,
            severity=EventSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            resource_type="consent_record",
            resource_id=data_subject_id,
            action="consent_granted" if granted else "consent_withdrawn",
            description=f"Consent {('granted' if granted else 'withdrawn')} for {purpose}",
            additional_data={
                "consent_type": consent_type,
                "consent_granted": granted,
                "processing_purpose": purpose,
                "consent_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        await self.audit_logger.log_event(event)
    
    async def log_right_to_erasure(self, data_subject_id: str, requested_by: str,
                                  collections_affected: List[str]):
        """Log GDPR right to erasure (right to be forgotten)"""
        event = AuditEvent(
            event_type="gdpr_right_to_erasure",
            category=EventCategory.COMPLIANCE,
            severity=EventSeverity.WARNING,  # Data deletion is sensitive
            outcome=AuditOutcome.SUCCESS,
            user_id=requested_by,
            resource_type="personal_data",
            resource_id=data_subject_id,
            action="erase_personal_data",
            description=f"Personal data erasure requested for data subject {data_subject_id}",
            additional_data={
                "erasure_request_by": requested_by,
                "collections_affected": collections_affected,
                "erasure_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        await self.audit_logger.log_event(event)

class SOXComplianceLogger:
    def __init__(self, audit_logger: EnterpriseAuditLogger):
        self.audit_logger = audit_logger
    
    async def log_financial_data_access(self, user_id: str, document_id: str,
                                       access_type: str, business_purpose: str):
        """Log SOX-compliant financial data access"""
        event = AuditEvent(
            event_type="sox_financial_data_access",
            category=EventCategory.COMPLIANCE,
            severity=EventSeverity.WARNING,  # Financial data access is sensitive
            outcome=AuditOutcome.SUCCESS,
            user_id=user_id,
            resource_type="financial_document",
            resource_id=document_id,
            action=access_type,
            description=f"Financial data accessed for: {business_purpose}",
            additional_data={
                "business_purpose": business_purpose,
                "sox_compliance": True,
                "access_authorized": True,
                "audit_trail_required": True
            }
        )
        
        await self.audit_logger.log_event(event)
    
    async def log_system_change(self, user_id: str, system_component: str,
                               change_type: str, change_description: str,
                               approved_by: str):
        """Log SOX-compliant system changes"""
        event = AuditEvent(
            event_type="sox_system_change",
            category=EventCategory.SYSTEM_ADMINISTRATION,
            severity=EventSeverity.WARNING,
            outcome=AuditOutcome.SUCCESS,
            user_id=user_id,
            resource_type="system_component",
            resource_id=system_component,
            action=change_type,
            description=f"System change: {change_description}",
            additional_data={
                "change_type": change_type,
                "change_description": change_description,
                "approved_by": approved_by,
                "sox_compliance": True,
                "change_control_process": True
            }
        )
        
        await self.audit_logger.log_event(event)

class HIPAAComplianceLogger:
    def __init__(self, audit_logger: EnterpriseAuditLogger):
        self.audit_logger = audit_logger
    
    async def log_phi_access(self, user_id: str, patient_id: str,
                           access_type: str, medical_purpose: str):
        """Log HIPAA-compliant PHI access"""
        event = AuditEvent(
            event_type="hipaa_phi_access",
            category=EventCategory.COMPLIANCE,
            severity=EventSeverity.WARNING,  # PHI access is highly sensitive
            outcome=AuditOutcome.SUCCESS,
            user_id=user_id,
            resource_type="protected_health_information",
            resource_id=patient_id,
            action=access_type,
            description=f"PHI accessed for medical purpose: {medical_purpose}",
            additional_data={
                "medical_purpose": medical_purpose,
                "hipaa_compliance": True,
                "phi_access_authorized": True,
                "minimum_necessary_standard": True
            }
        )
        
        await self.audit_logger.log_event(event)
```

This comprehensive audit logging system provides enterprise-grade compliance, security monitoring, and regulatory reporting capabilities for workspace-qdrant-mcp deployments. The implementation includes event classification, multiple storage backends, middleware integration, and compliance-specific logging patterns.