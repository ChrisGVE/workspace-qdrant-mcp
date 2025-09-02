# Web Ingestion Security Guide

## Overview

The workspace-qdrant-mcp web ingestion system provides secure web content crawling and processing with comprehensive security hardening. This guide covers the security features, configuration options, and best practices for safe web content ingestion.

## Security Architecture

### Multi-Layer Security

1. **URL Validation**
   - Scheme restrictions (only HTTP/HTTPS allowed)
   - Domain allowlist/blocklist enforcement
   - URL length limits and format validation
   - Malicious URL pattern detection

2. **Content Security Scanning**
   - JavaScript and script tag detection
   - Suspicious pattern recognition
   - Content size limits and type validation
   - Binary content detection in text streams

3. **Network Security**
   - Rate limiting with respectful crawling delays
   - Robots.txt compliance
   - SSL certificate verification
   - Timeout protections

4. **Content Isolation**
   - Suspicious content quarantine system
   - Sandboxed content processing
   - Temporary file cleanup

## Command Line Usage

### Basic Web Ingestion

```bash
# Ingest single web page
workspace-qdrant-ingest ingest-web https://example.com/docs --collection docs

# Crawl multiple pages with depth limit
workspace-qdrant-ingest ingest-web https://example.com/docs \
    --collection docs --max-pages 10 --max-depth 2

# Dry run to preview content
workspace-qdrant-ingest ingest-web https://example.com/docs \
    --collection docs --dry-run
```

### Security Configuration

```bash
# Restrict to specific domains
workspace-qdrant-ingest ingest-web https://example.com/docs \
    --collection docs --allowed-domains example.com,docs.example.com

# Configure rate limiting
workspace-qdrant-ingest ingest-web https://example.com/docs \
    --collection docs --request-delay 2.0

# Disable security scanning (NOT RECOMMENDED)
workspace-qdrant-ingest ingest-web https://example.com/docs \
    --collection docs --disable-security --allow-all-domains
```

## Python API Usage

### Basic Usage

```python
import asyncio
from workspace_qdrant_mcp.cli.parsers import (
    WebIngestionInterface, 
    SecurityConfig,
    create_secure_web_parser
)

async def ingest_web_content():
    # Create secure web parser with defaults
    parser = create_secure_web_parser(
        allowed_domains=['example.com', 'docs.example.com']
    )
    
    # Parse single page
    doc = await parser.parse('https://example.com/docs')
    print(f"Extracted {len(doc.content)} characters")
    
    # Check for security warnings
    if 'security_warnings' in doc.additional_metadata:
        warnings = doc.additional_metadata['security_warnings']
        if warnings:
            print(f"Security warnings: {len(warnings)}")

asyncio.run(ingest_web_content())
```

### Advanced Configuration

```python
from workspace_qdrant_mcp.cli.parsers import (
    WebIngestionInterface,
    SecurityConfig
)

# Create custom security configuration
config = SecurityConfig()
config.domain_allowlist = {'trusted-site.com', 'docs.trusted-site.com'}
config.max_content_size = 10 * 1024 * 1024  # 10MB limit
config.request_delay = 2.0  # 2 second delay between requests
config.max_total_pages = 20  # Maximum pages to crawl
config.enable_content_scanning = True
config.quarantine_suspicious = True

# Use with web ingestion interface
interface = WebIngestionInterface(config)

async def crawl_site():
    # Multi-page crawling with security
    doc = await interface.ingest_site(
        'https://trusted-site.com/docs',
        max_pages=15,
        max_depth=3
    )
    
    # Process results
    pages_crawled = doc.additional_metadata.get('pages_crawled', 0)
    print(f"Successfully crawled {pages_crawled} pages")
```

## Security Features in Detail

### URL Validation and Filtering

```python
# Domain allowlist (recommended)
config.domain_allowlist = {'example.com', 'subdomain.example.com'}

# Domain blocklist (additional protection)
config.domain_blocklist = {'malicious.com', 'localhost', '127.0.0.1'}

# Scheme restrictions
config.allowed_schemes = {'http', 'https'}  # Default
config.blocked_schemes = {'file', 'ftp', 'javascript'}  # Blocked

# URL length limits
config.max_url_length = 2048  # Maximum URL length
```

### Content Security Scanning

The system automatically scans for:

- JavaScript code and script tags
- Event handlers (onclick, onload, etc.)
- Dangerous functions (eval, document.write)
- Suspicious URL patterns
- Binary content in text streams
- Excessive script tag counts

```python
# Configure content scanning
config.enable_content_scanning = True  # Enable scanning
config.quarantine_suspicious = True    # Quarantine threats

# Custom content size limits
config.max_content_size = 50 * 1024 * 1024  # 50MB per page
```

### Rate Limiting and Respectful Crawling

```python
# Configure crawling behavior
config.request_delay = 1.0              # Seconds between requests
config.max_concurrent_requests = 5      # Concurrent connection limit
config.timeout_seconds = 30             # Request timeout
config.respect_robots_txt = True        # Honor robots.txt
config.user_agent = 'YourBot/1.0'       # Custom user agent
```

### Content Type Restrictions

```python
# Allowed content types
config.allowed_content_types = {
    'text/html',
    'text/plain', 
    'application/xhtml+xml',
    'text/xml'
}
```

## Security Best Practices

### 1. Always Use Domain Allowlists

```python
# GOOD: Restrict to trusted domains
config.domain_allowlist = {'your-trusted-site.com'}

# BAD: Allow all domains (security risk)
# Don't leave domain_allowlist empty without good reason
```

### 2. Enable Security Scanning

```python
# GOOD: Full security enabled
config.enable_content_scanning = True
config.quarantine_suspicious = True

# BAD: Disable security (high risk)
# config.enable_content_scanning = False
```

### 3. Implement Rate Limiting

```python
# GOOD: Respectful crawling
config.request_delay = 1.0  # At least 1 second
config.max_pages_per_domain = 100  # Reasonable limits

# BAD: Aggressive crawling
# config.request_delay = 0.1  # Too fast
```

### 4. Content Size Limits

```python
# GOOD: Reasonable size limits
config.max_content_size = 10 * 1024 * 1024  # 10MB
config.max_total_pages = 50  # Reasonable page limit

# BAD: No limits (resource exhaustion risk)
# config.max_content_size = float('inf')
```

## Error Handling and Monitoring

### Security Warnings

The system provides detailed security warnings:

```python
result = await parser.parse('https://example.com/page')

# Check for security issues
if 'security_warnings' in result.additional_metadata:
    warnings = result.additional_metadata['security_warnings']
    for warning in warnings:
        print(f"Security warning: {warning}")
```

### Quarantine System

Suspicious content is automatically quarantined:

```python
# Quarantine directory: /tmp/qdrant_quarantine/
# Files named: quarantine_<timestamp>_<hash>.html
# Contains metadata and original content for analysis
```

### Logging and Monitoring

```python
import logging

# Enable debug logging
logging.getLogger('workspace_qdrant_mcp.cli.parsers.web_crawler').setLevel(logging.DEBUG)

# Monitor crawling activity
logger = logging.getLogger(__name__)
logger.info("Starting secure web crawl")
```

## Common Use Cases

### 1. Documentation Site Ingestion

```bash
# Ingest technical documentation
workspace-qdrant-ingest ingest-web https://docs.example.com \
    --collection tech-docs \
    --max-pages 50 \
    --max-depth 3 \
    --allowed-domains docs.example.com \
    --request-delay 1.5
```

### 2. Blog Content Ingestion

```bash
# Ingest blog posts with conservative limits
workspace-qdrant-ingest ingest-web https://blog.example.com \
    --collection blog-posts \
    --max-pages 20 \
    --max-depth 2 \
    --allowed-domains blog.example.com,www.example.com
```

### 3. Research Paper Collection

```python
# Academic content with strict security
config = SecurityConfig()
config.domain_allowlist = {'arxiv.org', 'scholar.google.com'}
config.max_content_size = 5 * 1024 * 1024  # 5MB limit
config.enable_content_scanning = True
config.request_delay = 2.0  # Respectful to servers

interface = WebIngestionInterface(config)
```

## Troubleshooting

### Common Issues

1. **Domain Blocked Error**
   ```
   Error: Domain not in allowlist: suspicious.com
   ```
   Solution: Add domain to allowlist or verify it's trusted

2. **Content Security Scan Failed**
   ```
   Error: Content failed security scan: Script tag detected
   ```
   Solution: Review content or disable scanning if safe

3. **Rate Limit Timeouts**
   ```
   Warning: Rate limiting: waiting 1.5s for example.com
   ```
   Solution: Normal behavior, increase delay if needed

4. **Robots.txt Blocked**
   ```
   Error: Blocked by robots.txt
   ```
   Solution: Respect robots.txt or disable if appropriate

### Debug Mode

```bash
# Enable verbose logging
workspace-qdrant-ingest ingest-web https://example.com/docs \
    --collection docs --verbose --debug
```

## Security Considerations

### Threat Model

The web ingestion system protects against:

- **Malicious JavaScript**: Script injection and XSS attacks
- **Resource Exhaustion**: Large files and infinite crawling
- **Server Overload**: Rate limiting prevents server abuse
- **Data Exfiltration**: Domain restrictions limit access
- **Content Injection**: Security scanning detects threats

### Limitations

- Cannot detect all sophisticated attacks
- Static analysis only (no JavaScript execution)
- Relies on content-type headers (can be spoofed)
- May have false positives with legitimate content

### Recommendations

1. **Always use domain allowlists** for production systems
2. **Enable all security features** unless specifically needed
3. **Monitor quarantine directory** for suspicious content
4. **Implement additional scanning** for critical applications
5. **Regular security updates** of dependencies
6. **Audit crawled content** periodically

## Performance Considerations

### Optimization Tips

```python
# Optimize for speed vs security trade-offs
config.max_concurrent_requests = 10      # Increase concurrency
config.request_delay = 0.5              # Reduce delay (carefully)
config.enable_content_scanning = False   # Disable scanning (risky)

# Optimize for memory usage
config.max_content_size = 1024 * 1024   # 1MB limit
config.max_total_pages = 20             # Fewer pages
```

### Monitoring Resource Usage

```python
import psutil
import time

start_time = time.time()
start_memory = psutil.Process().memory_info().rss

# Perform web ingestion
result = await parser.parse(url)

end_time = time.time()
end_memory = psutil.Process().memory_info().rss

print(f"Processing time: {end_time - start_time:.2f}s")
print(f"Memory used: {(end_memory - start_memory) / 1024 / 1024:.2f}MB")
```

This comprehensive security system ensures safe web content ingestion while maintaining flexibility for various use cases.
