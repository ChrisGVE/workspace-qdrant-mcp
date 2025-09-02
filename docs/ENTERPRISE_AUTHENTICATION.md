# Enterprise Authentication Framework

**Version**: 1.0  
**Target Audience**: Enterprise DevOps, Security Teams, System Administrators  
**Prerequisites**: workspace-qdrant-mcp v1.0+, Qdrant 1.7+, Python 3.10+

## Overview

This document provides comprehensive guidance for integrating workspace-qdrant-mcp with enterprise authentication systems. It covers OAuth2, SAML, LDAP integration patterns, and enterprise-grade security configurations for production deployments.

## Table of Contents

- [Authentication Architecture](#authentication-architecture)
- [OAuth2 Integration](#oauth2-integration)
- [SAML Federation](#saml-federation)
- [LDAP Integration](#ldap-integration)
- [JWT Token Management](#jwt-token-management)
- [Multi-Factor Authentication](#multi-factor-authentication)
- [Session Management](#session-management)
- [Security Headers and CORS](#security-headers-and-cors)
- [Monitoring and Logging](#monitoring-and-logging)
- [Implementation Examples](#implementation-examples)

## Authentication Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client        │    │  Auth Middleware │    │  MCP Server     │
│  Applications   │────│   & Gateway      │────│  (workspace-    │
│                 │    │                  │    │   qdrant-mcp)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ├── OAuth2 Provider (e.g., Okta, Auth0)
                                ├── SAML Identity Provider (e.g., AD FS)
                                ├── LDAP Directory (e.g., Active Directory)
                                └── JWT Token Store (Redis/Database)
```

### Core Components

1. **Authentication Gateway**: Central authentication point for all MCP requests
2. **Token Validation Service**: Validates JWT tokens and maintains sessions
3. **Identity Provider Integration**: Connects to enterprise identity systems
4. **Authorization Service**: Manages role-based access control (RBAC)
5. **Audit Logger**: Comprehensive security event logging

## OAuth2 Integration

### Supported OAuth2 Flows

#### Authorization Code Flow (Recommended for Web Applications)

```python
# Example OAuth2 configuration for workspace-qdrant-mcp
import os
from authlib.integrations.flask_oauth2 import ResourceProtector
from authlib.oauth2.rfc6750 import BearerTokenValidator

class WorkspaceQdrantTokenValidator(BearerTokenValidator):
    def authenticate_token(self, token_string):
        # Validate token against your OAuth2 provider
        return validate_oauth2_token(token_string)

# Flask/FastAPI middleware integration
@app.before_request
def require_oauth():
    if request.endpoint in PROTECTED_ENDPOINTS:
        token = extract_bearer_token(request)
        if not validate_oauth2_token(token):
            return jsonify({'error': 'Invalid token'}), 401
```

#### Client Credentials Flow (Recommended for Service-to-Service)

```yaml
# docker-compose.yml with OAuth2 client credentials
version: '3.8'
services:
  workspace-qdrant-mcp:
    image: workspace-qdrant-mcp:latest
    environment:
      - OAUTH2_CLIENT_ID=your_client_id
      - OAUTH2_CLIENT_SECRET=your_client_secret
      - OAUTH2_TOKEN_URL=https://your-provider.com/oauth/token
      - OAUTH2_SCOPE=qdrant:read qdrant:write
    ports:
      - "8000:8000"
```

### Popular OAuth2 Providers

#### Okta Integration

```python
# Okta-specific configuration
OKTA_CONFIG = {
    'client_id': os.getenv('OKTA_CLIENT_ID'),
    'client_secret': os.getenv('OKTA_CLIENT_SECRET'),
    'org_url': os.getenv('OKTA_ORG_URL'),  # e.g., https://dev-123456.okta.com
    'authorization_server': 'default',
    'scope': ['openid', 'profile', 'email', 'qdrant:access']
}

import requests

def validate_okta_token(token):
    """Validate Okta JWT token"""
    introspect_url = f"{OKTA_CONFIG['org_url']}/oauth2/default/v1/introspect"
    response = requests.post(introspect_url, 
        data={'token': token},
        auth=(OKTA_CONFIG['client_id'], OKTA_CONFIG['client_secret'])
    )
    return response.json().get('active', False)
```

#### Azure AD Integration

```python
# Azure AD configuration
AZURE_AD_CONFIG = {
    'tenant_id': os.getenv('AZURE_TENANT_ID'),
    'client_id': os.getenv('AZURE_CLIENT_ID'),
    'client_secret': os.getenv('AZURE_CLIENT_SECRET'),
    'authority': f"https://login.microsoftonline.com/{os.getenv('AZURE_TENANT_ID')}"
}

from msal import ConfidentialClientApplication

def get_azure_token():
    """Acquire token from Azure AD"""
    app = ConfidentialClientApplication(
        AZURE_AD_CONFIG['client_id'],
        authority=AZURE_AD_CONFIG['authority'],
        client_credential=AZURE_AD_CONFIG['client_secret']
    )
    
    result = app.acquire_token_for_client(scopes=['https://graph.microsoft.com/.default'])
    return result.get('access_token')
```

## SAML Federation

### SAML 2.0 Service Provider Configuration

```xml
<!-- SAML SP metadata for workspace-qdrant-mcp -->
<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata"
                     entityID="https://your-domain.com/workspace-qdrant-mcp">
    <md:SPSSODescriptor AuthnRequestsSigned="true" protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">
        <md:AssertionConsumerService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
                                   Location="https://your-domain.com/workspace-qdrant-mcp/saml/acs"
                                   index="0" isDefault="true"/>
        <md:AttributeConsumingService index="0">
            <md:ServiceName xml:lang="en">Workspace Qdrant MCP</md:ServiceName>
            <md:RequestedAttribute Name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress"/>
            <md:RequestedAttribute Name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name"/>
            <md:RequestedAttribute Name="http://schemas.microsoft.com/ws/2008/06/identity/claims/groups"/>
        </md:AttributeConsumingService>
    </md:SPSSODescriptor>
</md:EntityDescriptor>
```

### Python SAML Implementation

```python
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from onelogin.saml2.settings import OneLogin_Saml2_Settings

class SAMLAuthenticator:
    def __init__(self, saml_settings):
        self.settings = OneLogin_Saml2_Settings(saml_settings)
    
    def initiate_sso(self, request):
        """Initiate SAML SSO"""
        auth = OneLogin_Saml2_Auth(request, self.settings)
        return auth.login()
    
    def process_response(self, request):
        """Process SAML response"""
        auth = OneLogin_Saml2_Auth(request, self.settings)
        auth.process_response()
        
        if auth.is_authenticated():
            user_data = {
                'email': auth.get_attribute('http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress')[0],
                'name': auth.get_attribute('http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name')[0],
                'groups': auth.get_attribute('http://schemas.microsoft.com/ws/2008/06/identity/claims/groups', [])
            }
            return self.create_workspace_session(user_data)
        else:
            raise AuthenticationError("SAML authentication failed")
```

### AD FS Integration Example

```json
{
  "saml_settings": {
    "sp": {
      "entityId": "https://your-domain.com/workspace-qdrant-mcp",
      "assertionConsumerService": {
        "url": "https://your-domain.com/workspace-qdrant-mcp/saml/acs",
        "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
      }
    },
    "idp": {
      "entityId": "http://your-adfs-server.com/adfs/services/trust",
      "singleSignOnService": {
        "url": "https://your-adfs-server.com/adfs/ls/",
        "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
      },
      "x509cert": "YOUR_ADFS_CERTIFICATE_HERE"
    }
  }
}
```

## LDAP Integration

### Active Directory Integration

```python
from ldap3 import Server, Connection, ALL, SUBTREE
from ldap3.core.exceptions import LDAPException

class LDAPAuthenticator:
    def __init__(self, ldap_config):
        self.server = Server(ldap_config['server'], get_info=ALL)
        self.admin_user = ldap_config['admin_user']
        self.admin_password = ldap_config['admin_password']
        self.base_dn = ldap_config['base_dn']
        self.user_filter = ldap_config.get('user_filter', '(sAMAccountName={username})')
    
    def authenticate_user(self, username, password):
        """Authenticate user against Active Directory"""
        try:
            # Search for user
            conn = Connection(self.server, user=self.admin_user, password=self.admin_password)
            conn.bind()
            
            search_filter = self.user_filter.format(username=username)
            conn.search(self.base_dn, search_filter, SUBTREE, 
                       attributes=['cn', 'mail', 'memberOf', 'department'])
            
            if len(conn.entries) != 1:
                return None
                
            user_dn = conn.entries[0].entry_dn
            user_info = {
                'dn': user_dn,
                'cn': str(conn.entries[0].cn),
                'email': str(conn.entries[0].mail),
                'groups': [str(group) for group in conn.entries[0].memberOf],
                'department': str(conn.entries[0].department)
            }
            
            # Authenticate user with their credentials
            user_conn = Connection(self.server, user=user_dn, password=password)
            if user_conn.bind():
                return user_info
            else:
                return None
                
        except LDAPException as e:
            logging.error(f"LDAP authentication error: {e}")
            return None
```

### LDAP Configuration Example

```yaml
# Environment variables for LDAP
LDAP_SERVER: "ldaps://your-ad-server.com:636"
LDAP_ADMIN_USER: "CN=Service Account,OU=Service Accounts,DC=company,DC=com"
LDAP_ADMIN_PASSWORD: "service_account_password"
LDAP_BASE_DN: "DC=company,DC=com"
LDAP_USER_FILTER: "(sAMAccountName={username})"
LDAP_GROUP_FILTER: "(&(objectClass=group)(member={user_dn}))"

# Docker Compose integration
version: '3.8'
services:
  workspace-qdrant-mcp:
    image: workspace-qdrant-mcp:latest
    environment:
      - AUTH_TYPE=ldap
      - LDAP_SERVER=${LDAP_SERVER}
      - LDAP_ADMIN_USER=${LDAP_ADMIN_USER}
      - LDAP_ADMIN_PASSWORD=${LDAP_ADMIN_PASSWORD}
      - LDAP_BASE_DN=${LDAP_BASE_DN}
```

## JWT Token Management

### JWT Token Structure for workspace-qdrant-mcp

```python
import jwt
import datetime
from typing import Dict, List

class WorkspaceJWTManager:
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = datetime.timedelta(hours=8)  # 8-hour sessions
        self.refresh_expiry = datetime.timedelta(days=30)  # 30-day refresh
    
    def create_tokens(self, user_info: Dict) -> Dict:
        """Create access and refresh tokens"""
        now = datetime.datetime.utcnow()
        
        # Access token payload
        access_payload = {
            'sub': user_info['user_id'],
            'email': user_info['email'],
            'name': user_info['name'],
            'roles': user_info['roles'],
            'collections': user_info.get('collections', []),
            'permissions': user_info.get('permissions', []),
            'iat': now,
            'exp': now + self.token_expiry,
            'type': 'access'
        }
        
        # Refresh token payload
        refresh_payload = {
            'sub': user_info['user_id'],
            'iat': now,
            'exp': now + self.refresh_expiry,
            'type': 'refresh'
        }
        
        access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_in': int(self.token_expiry.total_seconds()),
            'token_type': 'Bearer'
        }
    
    def validate_token(self, token: str) -> Dict:
        """Validate and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
```

### Token Refresh Implementation

```python
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract and validate current user from JWT token"""
    token = credentials.credentials
    
    try:
        payload = jwt_manager.validate_token(token)
        
        # Check if token is access token
        if payload.get('type') != 'access':
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        return {
            'user_id': payload['sub'],
            'email': payload['email'],
            'name': payload['name'],
            'roles': payload['roles'],
            'collections': payload.get('collections', []),
            'permissions': payload.get('permissions', [])
        }
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.post("/auth/refresh")
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token"""
    try:
        payload = jwt_manager.validate_token(refresh_token)
        
        if payload.get('type') != 'refresh':
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Get current user info and create new tokens
        user_info = get_user_info(payload['sub'])
        return jwt_manager.create_tokens(user_info)
        
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))
```

## Multi-Factor Authentication

### TOTP Integration

```python
import pyotp
import qrcode
from io import BytesIO

class MFAManager:
    def __init__(self):
        self.issuer_name = "Workspace Qdrant MCP"
    
    def generate_secret(self, user_email: str) -> Dict:
        """Generate TOTP secret for user"""
        secret = pyotp.random_base32()
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=self.issuer_name
        )
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return {
            'secret': secret,
            'qr_code': img_buffer.getvalue(),
            'manual_entry_key': secret
        }
    
    def verify_totp(self, secret: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=window)

# Integration with authentication flow
class EnhancedAuthenticator:
    def __init__(self, mfa_manager: MFAManager):
        self.mfa_manager = mfa_manager
    
    def authenticate_with_mfa(self, username: str, password: str, mfa_token: str = None):
        """Two-step authentication with MFA"""
        # Step 1: Validate username/password
        user_info = self.validate_primary_credentials(username, password)
        if not user_info:
            raise AuthenticationError("Invalid credentials")
        
        # Step 2: Check if MFA is enabled for user
        if user_info.get('mfa_enabled'):
            if not mfa_token:
                return {'requires_mfa': True, 'user_id': user_info['user_id']}
            
            if not self.mfa_manager.verify_totp(user_info['mfa_secret'], mfa_token):
                raise AuthenticationError("Invalid MFA token")
        
        return self.create_authenticated_session(user_info)
```

### Hardware Token Support (FIDO2/WebAuthn)

```python
from webauthn import generate_registration_options, verify_registration_response
from webauthn import generate_authentication_options, verify_authentication_response

class WebAuthnManager:
    def __init__(self, rp_id: str, rp_name: str):
        self.rp_id = rp_id  # e.g., "your-domain.com"
        self.rp_name = rp_name  # e.g., "Workspace Qdrant MCP"
    
    def start_registration(self, user_info: Dict):
        """Start WebAuthn registration process"""
        options = generate_registration_options(
            rp_id=self.rp_id,
            rp_name=self.rp_name,
            user_id=user_info['user_id'].encode(),
            user_name=user_info['email'],
            user_display_name=user_info['name']
        )
        
        # Store challenge in session/database
        store_challenge(user_info['user_id'], options.challenge)
        
        return options
    
    def complete_registration(self, user_id: str, credential_response: Dict):
        """Complete WebAuthn registration"""
        challenge = get_stored_challenge(user_id)
        
        verification = verify_registration_response(
            credential=credential_response,
            expected_challenge=challenge,
            expected_origin=f"https://{self.rp_id}",
            expected_rp_id=self.rp_id
        )
        
        if verification.verified:
            # Store credential for user
            store_user_credential(user_id, {
                'credential_id': verification.credential_id,
                'credential_public_key': verification.credential_public_key,
                'sign_count': verification.sign_count
            })
            return True
        return False
```

## Session Management

### Redis-based Session Store

```python
import redis
import json
from datetime import datetime, timedelta

class RedisSessionManager:
    def __init__(self, redis_url: str, session_timeout: int = 28800):  # 8 hours
        self.redis_client = redis.from_url(redis_url)
        self.session_timeout = session_timeout
    
    def create_session(self, user_info: Dict) -> str:
        """Create new user session"""
        session_id = self.generate_session_id()
        session_data = {
            'user_id': user_info['user_id'],
            'email': user_info['email'],
            'roles': user_info['roles'],
            'collections': user_info.get('collections', []),
            'created_at': datetime.utcnow().isoformat(),
            'last_activity': datetime.utcnow().isoformat()
        }
        
        self.redis_client.setex(
            f"session:{session_id}",
            self.session_timeout,
            json.dumps(session_data)
        )
        
        return session_id
    
    def get_session(self, session_id: str) -> Dict:
        """Retrieve session data"""
        session_data = self.redis_client.get(f"session:{session_id}")
        if session_data:
            data = json.loads(session_data)
            # Update last activity
            data['last_activity'] = datetime.utcnow().isoformat()
            self.redis_client.setex(
                f"session:{session_id}",
                self.session_timeout,
                json.dumps(data)
            )
            return data
        return None
    
    def invalidate_session(self, session_id: str):
        """Invalidate user session"""
        self.redis_client.delete(f"session:{session_id}")
    
    def cleanup_expired_sessions(self):
        """Background task to clean up expired sessions"""
        # Redis handles TTL automatically, but you might want additional cleanup
        pass
```

### Database Session Store (Alternative)

```sql
-- PostgreSQL session table schema
CREATE TABLE user_sessions (
    session_id VARCHAR(128) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    user_email VARCHAR(255) NOT NULL,
    roles JSONB NOT NULL DEFAULT '[]',
    collections JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ip_address INET,
    user_agent TEXT
);

-- Index for efficient lookups
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);
```

## Security Headers and CORS

### Security Headers Implementation

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI()

# Security middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response

# CORS configuration for enterprise
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "https://your-domain.com").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "your-domain.com").split(",")
)

# Session middleware with secure settings
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY"),
    session_cookie="workspace_qdrant_session",
    max_age=28800,  # 8 hours
    same_site="strict",
    https_only=True
)
```

## Monitoring and Logging

### Security Event Logging

```python
import logging
import json
from datetime import datetime
from typing import Optional

class SecurityLogger:
    def __init__(self, logger_name: str = "workspace_qdrant_security"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Create structured formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for security events
        handler = logging.FileHandler('/var/log/workspace-qdrant-mcp/security.log')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_authentication_event(self, event_type: str, user_id: str, 
                                ip_address: str, success: bool, 
                                additional_data: Optional[Dict] = None):
        """Log authentication events"""
        event_data = {
            'event_type': event_type,
            'user_id': user_id,
            'ip_address': ip_address,
            'success': success,
            'timestamp': datetime.utcnow().isoformat(),
            'additional_data': additional_data or {}
        }
        
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, f"AUTH_EVENT: {json.dumps(event_data)}")
    
    def log_access_event(self, user_id: str, collection: str, operation: str,
                        ip_address: str, success: bool):
        """Log data access events"""
        event_data = {
            'event_type': 'data_access',
            'user_id': user_id,
            'collection': collection,
            'operation': operation,
            'ip_address': ip_address,
            'success': success,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        level = logging.INFO if success else logging.ERROR
        self.logger.log(level, f"ACCESS_EVENT: {json.dumps(event_data)}")

# Usage in middleware
security_logger = SecurityLogger()

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    
    # Extract user info if available
    user_id = getattr(request.state, 'user_id', 'anonymous')
    ip_address = request.client.host
    
    try:
        response = await call_next(request)
        
        # Log successful requests
        if str(response.status_code).startswith(('2', '3')):
            if request.url.path.startswith('/api/'):
                security_logger.log_access_event(
                    user_id=user_id,
                    collection=extract_collection_from_path(request.url.path),
                    operation=request.method,
                    ip_address=ip_address,
                    success=True
                )
        
        return response
        
    except Exception as e:
        # Log failed requests
        security_logger.log_access_event(
            user_id=user_id,
            collection='unknown',
            operation=request.method,
            ip_address=ip_address,
            success=False
        )
        raise
```

### Metrics and Monitoring

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Security metrics
auth_attempts_total = Counter('auth_attempts_total', 'Total authentication attempts', ['method', 'result'])
auth_duration = Histogram('auth_duration_seconds', 'Authentication duration')
active_sessions = Gauge('active_sessions_total', 'Number of active user sessions')
failed_logins_total = Counter('failed_logins_total', 'Total failed login attempts', ['user_id'])

class SecurityMetrics:
    @staticmethod
    def record_auth_attempt(method: str, success: bool):
        result = 'success' if success else 'failure'
        auth_attempts_total.labels(method=method, result=result).inc()
    
    @staticmethod
    def record_auth_duration(duration: float):
        auth_duration.observe(duration)
    
    @staticmethod
    def update_active_sessions(count: int):
        active_sessions.set(count)
    
    @staticmethod
    def record_failed_login(user_id: str):
        failed_logins_total.labels(user_id=user_id).inc()

# Endpoint for Prometheus scraping
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## Implementation Examples

### Complete Authentication Middleware

```python
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
from typing import Optional

app = FastAPI(title="Workspace Qdrant MCP - Enterprise Edition")
security = HTTPBearer()

class EnterpriseAuthenticator:
    def __init__(self):
        self.oauth2_validator = OAuth2Validator()
        self.saml_authenticator = SAMLAuthenticator()
        self.ldap_authenticator = LDAPAuthenticator()
        self.jwt_manager = WorkspaceJWTManager(os.getenv('JWT_SECRET_KEY'))
        self.session_manager = RedisSessionManager(os.getenv('REDIS_URL'))
        self.security_logger = SecurityLogger()
    
    async def authenticate_request(self, request: Request, 
                                 credentials: HTTPAuthorizationCredentials) -> Dict:
        """Main authentication entry point"""
        token = credentials.credentials
        ip_address = request.client.host
        
        try:
            # Try JWT token validation first
            user_info = await self.validate_jwt_token(token)
            
            self.security_logger.log_authentication_event(
                'jwt_validation', user_info['user_id'], ip_address, True
            )
            
            return user_info
            
        except AuthenticationError:
            # Fallback to other authentication methods
            try:
                # Try OAuth2 token validation
                user_info = await self.oauth2_validator.validate_token(token)
                
                self.security_logger.log_authentication_event(
                    'oauth2_validation', user_info['user_id'], ip_address, True
                )
                
                return user_info
                
            except AuthenticationError as e:
                self.security_logger.log_authentication_event(
                    'token_validation', 'unknown', ip_address, False, {'error': str(e)}
                )
                raise HTTPException(status_code=401, detail="Authentication failed")
    
    async def validate_jwt_token(self, token: str) -> Dict:
        """Validate JWT token and return user info"""
        try:
            payload = self.jwt_manager.validate_token(token)
            return {
                'user_id': payload['sub'],
                'email': payload['email'],
                'roles': payload['roles'],
                'collections': payload.get('collections', []),
                'permissions': payload.get('permissions', [])
            }
        except Exception as e:
            raise AuthenticationError(f"JWT validation failed: {e}")

# Global authenticator instance
authenticator = EnterpriseAuthenticator()

async def get_current_user(request: Request, 
                          credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user"""
    return await authenticator.authenticate_request(request, credentials)

# Protected endpoint example
@app.get("/api/search")
async def search_workspace(query: str, 
                          current_user: Dict = Depends(get_current_user)):
    """Search workspace with authentication"""
    # Check permissions
    if 'qdrant:read' not in current_user.get('permissions', []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Filter collections based on user access
    accessible_collections = current_user.get('collections', [])
    
    # Perform search with authorization
    results = await workspace_search(query, collections=accessible_collections)
    
    # Log access
    authenticator.security_logger.log_access_event(
        user_id=current_user['user_id'],
        collection='workspace',
        operation='search',
        ip_address=request.client.host,
        success=True
    )
    
    return results
```

### Docker Deployment with Authentication

```yaml
# docker-compose.enterprise.yml
version: '3.8'

services:
  workspace-qdrant-mcp:
    image: workspace-qdrant-mcp:enterprise
    ports:
      - "8000:8000"
    environment:
      # Authentication configuration
      - AUTH_METHOD=jwt
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - OAUTH2_CLIENT_ID=${OAUTH2_CLIENT_ID}
      - OAUTH2_CLIENT_SECRET=${OAUTH2_CLIENT_SECRET}
      - LDAP_SERVER=${LDAP_SERVER}
      - LDAP_BIND_DN=${LDAP_BIND_DN}
      - LDAP_BIND_PASSWORD=${LDAP_BIND_PASSWORD}
      
      # Session management
      - REDIS_URL=redis://redis:6379/0
      - SESSION_TIMEOUT=28800
      
      # Security settings
      - ALLOWED_ORIGINS=https://your-domain.com,https://admin.your-domain.com
      - TRUSTED_HOSTS=your-domain.com,admin.your-domain.com
      
      # Logging
      - LOG_LEVEL=INFO
      - SECURITY_LOG_LEVEL=INFO
    volumes:
      - ./logs:/var/log/workspace-qdrant-mcp
      - ./ssl:/etc/ssl/certs
    depends_on:
      - redis
      - qdrant

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD}

  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334

  # NGINX reverse proxy with SSL termination
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - workspace-qdrant-mcp

volumes:
  redis_data:
  qdrant_data:
```

### Kubernetes Deployment with Authentication

```yaml
# kubernetes/workspace-qdrant-mcp-enterprise.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: workspace-qdrant-mcp-enterprise
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: workspace-qdrant-mcp
  template:
    metadata:
      labels:
        app: workspace-qdrant-mcp
    spec:
      containers:
      - name: workspace-qdrant-mcp
        image: workspace-qdrant-mcp:enterprise-latest
        ports:
        - containerPort: 8000
        env:
        - name: AUTH_METHOD
          value: "jwt"
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: auth-secrets
              key: jwt-secret-key
        - name: OAUTH2_CLIENT_ID
          valueFrom:
            secretKeyRef:
              name: auth-secrets
              key: oauth2-client-id
        - name: OAUTH2_CLIENT_SECRET
          valueFrom:
            secretKeyRef:
              name: auth-secrets
              key: oauth2-client-secret
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        volumeMounts:
        - name: logs
          mountPath: /var/log/workspace-qdrant-mcp
        - name: ssl-certs
          mountPath: /etc/ssl/certs
          readOnly: true
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
            httpHeaders:
            - name: Authorization
              value: Bearer health-check-token
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: logs
        persistentVolumeClaim:
          claimName: logs-pvc
      - name: ssl-certs
        secret:
          secretName: ssl-certificates
---
apiVersion: v1
kind: Secret
metadata:
  name: auth-secrets
  namespace: production
type: Opaque
data:
  jwt-secret-key: <base64-encoded-secret>
  oauth2-client-id: <base64-encoded-client-id>
  oauth2-client-secret: <base64-encoded-client-secret>
---
apiVersion: v1
kind: Service
metadata:
  name: workspace-qdrant-mcp-service
  namespace: production
spec:
  selector:
    app: workspace-qdrant-mcp
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: LoadBalancer
```

This completes the comprehensive Enterprise Authentication Framework documentation. The document provides production-ready authentication patterns, security configurations, and deployment examples for enterprise environments.
