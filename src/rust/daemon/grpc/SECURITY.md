# gRPC Security Configuration

## Overview

This document describes the security features and configuration options for the workspace-qdrant-mcp gRPC server.

**WARNING**: The gRPC server defaults to insecure mode for development convenience. For production deployments, you MUST enable TLS and authentication.

## Security Features

### 1. TLS Encryption

TLS (Transport Layer Security) encrypts all communication between clients and the server, preventing eavesdropping and man-in-the-middle attacks.

**Configuration Options:**
- `cert_path`: Path to server certificate (PEM format)
- `key_path`: Path to server private key (PEM format)
- `ca_cert_path`: Optional CA certificate for client verification (mutual TLS)
- `require_client_cert`: When true, requires clients to present valid certificates

**Example:**
```rust
use workspace_qdrant_grpc::{ServerConfig, TlsConfig};
use std::net::SocketAddr;

let addr: SocketAddr = "127.0.0.1:50051".parse().unwrap();
let config = ServerConfig::new_secure(
    addr,
    "/path/to/server-cert.pem".to_string(),
    "/path/to/server-key.pem".to_string(),
    "your-secure-api-key".to_string(),
);
```

### 2. API Key Authentication

Bearer token authentication requires clients to include a valid API key in the `Authorization` header.

**Client Request Example:**
```
Authorization: Bearer your-secure-api-key-here
```

**Configuration:**
```rust
use workspace_qdrant_grpc::AuthConfig;

let auth = AuthConfig {
    enabled: true,
    api_key: Some("your-secure-api-key-min-32-chars".to_string()),
    jwt_secret: None,
    allowed_origins: vec![],
};
```

### 3. Mutual TLS (mTLS)

Mutual TLS provides the highest level of security by requiring both server and client to present valid certificates.

**Example:**
```rust
let config = ServerConfig::new_mutual_tls(
    addr,
    "/path/to/server-cert.pem".to_string(),
    "/path/to/server-key.pem".to_string(),
    "/path/to/ca-cert.pem".to_string(),
);
```

### 4. Origin Validation

CORS (Cross-Origin Resource Sharing) protection prevents unauthorized web applications from accessing the gRPC server.

**Configuration:**
```rust
let auth = AuthConfig {
    enabled: true,
    api_key: Some("key".to_string()),
    jwt_secret: None,
    allowed_origins: vec![
        "https://app.example.com".to_string(),
        "https://admin.example.com".to_string(),
    ],
};
```

## Security Configuration Methods

### Method 1: Secure Configuration (Recommended)

Use `ServerConfig::new_secure()` for TLS + API key authentication:

```rust
let config = ServerConfig::new_secure(
    bind_addr,
    cert_path,
    key_path,
    api_key,
);
```

### Method 2: Mutual TLS Configuration (Maximum Security)

Use `ServerConfig::new_mutual_tls()` for zero-trust environments:

```rust
let config = ServerConfig::new_mutual_tls(
    bind_addr,
    cert_path,
    key_path,
    ca_cert_path,
);
```

### Method 3: Manual Configuration

Build configuration manually for custom requirements:

```rust
let config = ServerConfig::new(bind_addr)
    .with_tls(tls_config)
    .with_auth(auth_config)
    .with_timeouts(timeout_config);
```

## Security Warnings

The server automatically logs security warnings when starting with insecure configuration:

```
WARN  ===== SECURITY WARNINGS =====
WARN    - TLS is not enabled - all communication will be unencrypted
WARN    - Authentication is disabled - anyone can access the gRPC server
WARN  ============================
ERROR gRPC server is running in INSECURE mode - not suitable for production
```

## Generating TLS Certificates

### Development (Self-Signed)

For development and testing only:

```bash
# 1. Generate CA key and certificate
openssl req -x509 -newkey rsa:4096 -days 365 -nodes \
  -keyout ca-key.pem -out ca-cert.pem \
  -subj "/CN=Workspace Qdrant CA"

# 2. Generate server key and CSR
openssl req -newkey rsa:4096 -nodes \
  -keyout server-key.pem -out server-req.pem \
  -subj "/CN=localhost"

# 3. Sign server certificate
openssl x509 -req -in server-req.pem -days 365 \
  -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial \
  -out server-cert.pem

# 4. For client certificates (mutual TLS)
openssl req -newkey rsa:4096 -nodes \
  -keyout client-key.pem -out client-req.pem \
  -subj "/CN=client"

openssl x509 -req -in client-req.pem -days 365 \
  -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial \
  -out client-cert.pem
```

### Production (Let's Encrypt)

For production, use certificates from a trusted Certificate Authority:

```bash
# Using certbot for Let's Encrypt
certbot certonly --standalone -d your-domain.com

# Certificates will be in:
# /etc/letsencrypt/live/your-domain.com/fullchain.pem  (cert_path)
# /etc/letsencrypt/live/your-domain.com/privkey.pem    (key_path)
```

## Security Best Practices

### 1. Network Configuration

- **Bind to localhost by default**: Use `127.0.0.1:50051` unless external access is required
- **Use firewall rules**: Restrict access to port 50051 using iptables or cloud security groups
- **Enable TLS for any non-localhost binding**: Never expose unencrypted gRPC to the network

### 2. Authentication

- **Use strong API keys**: Minimum 32 characters, randomly generated
- **Rotate keys regularly**: Implement key rotation every 90 days
- **Never commit keys**: Use environment variables or secret management systems
- **Consider mutual TLS**: For high-security environments, require client certificates

### 3. Certificate Management

- **Use trusted CAs for production**: Avoid self-signed certificates in production
- **Set appropriate expiration**: Certificates should expire in 90-365 days
- **Automate renewal**: Use certbot or similar tools for automatic renewal
- **Protect private keys**: Store with permissions 0600, never share or commit

### 4. Monitoring and Auditing

- **Monitor authentication failures**: Track failed auth attempts via metrics
- **Enable comprehensive logging**: Log all connection attempts and security events
- **Set up alerts**: Alert on unusual patterns or security violations
- **Regular security audits**: Review configuration and access patterns

### 5. Configuration Security

- **Use environment variables**: For sensitive configuration like API keys
- **Restrict config file permissions**: `chmod 600` for configuration files
- **Separate dev and prod configs**: Never use development keys in production
- **Regular security reviews**: Review and update security configuration quarterly

## Python Client Configuration

When TLS is enabled on the server, Python clients must use secure channels:

```python
from grpc import ssl_channel_credentials, composite_channel_credentials, access_token_call_credentials

# TLS with API key
ssl_creds = ssl_channel_credentials(
    root_certificates=open('/path/to/ca-cert.pem', 'rb').read()
)
token_creds = access_token_call_credentials('your-api-key')
composite_creds = composite_channel_credentials(ssl_creds, token_creds)

channel = grpc.secure_channel('localhost:50051', composite_creds)
```

## Compliance Considerations

### GDPR/Privacy

- Enable TLS to protect personal data in transit
- Log access for audit trails
- Implement proper access controls

### SOC 2

- Require authentication for all access
- Monitor and log security events
- Implement incident response procedures

### HIPAA

- Use mutual TLS for healthcare data
- Encrypt all communications
- Maintain comprehensive audit logs

## Troubleshooting

### Common Issues

**Problem**: Certificate verification failed
**Solution**: Ensure CA certificate is correctly configured and client has access

**Problem**: Connection refused
**Solution**: Check firewall rules and server binding address

**Problem**: Authentication failed
**Solution**: Verify API key is correct and in proper format (`Bearer <key>`)

### Security Validation

Check if your configuration is secure:

```rust
let warnings = config.get_security_warnings();
if !warnings.is_empty() {
    for warning in warnings {
        eprintln!("WARNING: {}", warning);
    }
}

if !config.is_secure() {
    eprintln!("ERROR: Configuration is not secure!");
}
```

## References

- [gRPC Security Guide](https://grpc.io/docs/guides/auth/)
- [TLS Best Practices](https://ssl-config.mozilla.org/)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)

## Support

For security-related issues or questions, please:

1. Check the SECURITY.md file in the repository root
2. Review this documentation thoroughly
3. Consult the gRPC security documentation
4. Contact the security team for assistance

**Never disclose security vulnerabilities publicly**. Use the responsible disclosure process outlined in SECURITY.md.
