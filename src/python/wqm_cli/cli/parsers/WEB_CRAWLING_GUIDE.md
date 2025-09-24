# Web Crawling System Guide

## Overview

The workspace-qdrant-mcp web crawling system provides comprehensive, respectful web content extraction with advanced reliability features, content quality filtering, and seamless integration with the document processing pipeline.

## Architecture

### Core Components

1. **SecureWebCrawler** - Base secure crawler with security hardening
2. **EnhancedWebCrawler** - Advanced crawler with integrated features
3. **EnhancedContentExtractor** - Quality content extraction and filtering
4. **AdvancedRetryHandler** - Exponential backoff and circuit breaker
5. **WebCache** - Request/response caching with TTL and deduplication
6. **ConnectionOptimizer** - Connection pooling and keep-alive optimization
7. **WebPipelineIntegration** - Document processing pipeline integration

### Key Features

#### Respectful Crawling
- **Rate Limiting**: Configurable delay between requests (default 2 requests/second)
- **robots.txt Compliance**: Automatically checks and respects robots.txt
- **User-Agent Rotation**: Prevents detection with realistic browser user agents
- **Domain Restrictions**: Same-domain crawling with configurable exceptions
- **Circuit Breaker**: Automatically stops crawling failing domains

#### Content Quality
- **Boilerplate Removal**: Removes navigation, ads, sidebars, footers
- **Content Quality Scoring**: Evaluates content based on multiple metrics
- **Structured Data Extraction**: JSON-LD, microdata, Open Graph, Twitter Cards
- **Media Link Extraction**: Images, videos, audio, documents
- **Duplicate Detection**: Content fingerprinting for deduplication

#### Reliability & Performance
- **Exponential Backoff**: Smart retry logic with jitter
- **Request Caching**: Avoid duplicate fetches with TTL and ETag support
- **Connection Pooling**: Efficient connection reuse and keep-alive
- **Progress Tracking**: Comprehensive session management and reporting
- **Error Handling**: Graceful degradation and detailed error reporting

## Usage Examples

### Basic Single Page Crawl

```python
from src.python.wqm_cli.cli.parsers.enhanced_web_crawler import EnhancedWebCrawler

async def crawl_single_page():
    async with EnhancedWebCrawler() as crawler:
        result = await crawler.crawl_url("https://example.com")

        if result.success:
            print(f"Content quality: {result.content_quality_score}")
            print(f"Word count: {result.extracted_content['word_count']}")
            print(f"Media links: {len(result.media_links.get('images', []))}")
```

### Recursive Site Crawling

```python
async def crawl_website():
    async with EnhancedWebCrawler() as crawler:
        results = await crawler.crawl_recursive(
            "https://example.com",
            max_depth=2,
            max_pages=50,
            same_domain_only=True
        )

        for result in results:
            if result.success:
                print(f"Crawled: {result.url} (quality: {result.content_quality_score})")
```

### Pipeline Integration

```python
from src.python.wqm_cli.cli.parsers.web_pipeline_integration import WebCrawlPipeline

async def crawl_with_processing():
    pipeline = WebCrawlPipeline(
        collection_name="web_content"
    )

    session = await pipeline.crawl_and_process([
        "https://example.com/page1",
        "https://example.com/page2"
    ])

    print(f"Processed {session.urls_successful} pages successfully")
    await pipeline.export_session_results(session.session_id, Path("results.json"))
```

### Configuration Examples

#### Security Configuration

```python
from src.python.wqm_cli.cli.parsers.web_crawler import SecurityConfig

security_config = SecurityConfig()
security_config.request_delay = 3.0  # 3 seconds between requests
security_config.max_depth = 5
security_config.max_total_pages = 1000
security_config.domain_allowlist = {"example.com", "subdomain.example.com"}
security_config.enable_content_scanning = True
security_config.quarantine_suspicious = True
```

#### Caching Configuration

```python
from src.python.wqm_cli.cli.parsers.web_cache import CacheConfig

cache_config = CacheConfig()
cache_config.max_entries = 50000
cache_config.max_memory_mb = 1000
cache_config.default_ttl = 7200.0  # 2 hours
cache_config.enable_disk_cache = True
cache_config.enable_content_fingerprinting = True
```

#### Retry Configuration

```python
from src.python.wqm_cli.cli.parsers.advanced_retry import RetryConfig

retry_config = RetryConfig()
retry_config.max_retries = 5
retry_config.base_delay = 2.0
retry_config.max_delay = 120.0
retry_config.circuit_failure_threshold = 10
retry_config.circuit_recovery_timeout = 60.0
```

## Content Quality Filtering

### Quality Metrics

The system evaluates content quality using multiple factors:

- **Text Length**: Longer content generally scores higher
- **Paragraph Count**: Well-structured content with multiple paragraphs
- **Link Density**: Lower link density indicates more original content
- **Boilerplate Detection**: Removes navigation, ads, and template content
- **Readability**: Optimal sentence length and structure

### Structured Data Extraction

Automatically extracts:

- **JSON-LD**: Schema.org structured data
- **Microdata**: HTML microdata attributes
- **Open Graph**: Facebook/social media metadata
- **Twitter Cards**: Twitter-specific metadata
- **Standard Meta Tags**: Description, keywords, author

### Media Cataloging

Extracts and catalogs:

- **Images**: URLs, alt text, dimensions
- **Videos**: Sources, controls, autoplay settings
- **Audio**: Sources and playback controls
- **Documents**: PDF, DOC, XLS, PPT links

## Integration with Document Pipeline

### Automatic Processing

```python
# Crawled content is automatically:
# 1. Quality filtered and extracted
# 2. Chunked for vector database storage
# 3. Embedded using FastEmbed
# 4. Stored in Qdrant collections
# 5. Indexed for search and retrieval
```

### Session Management

The pipeline provides comprehensive session tracking:

- **Progress Monitoring**: Real-time crawl progress
- **Statistics**: Success rates, timing, cache performance
- **Error Tracking**: Detailed error analysis and patterns
- **Export Capabilities**: JSON reports with full metadata

## Best Practices

### Respectful Crawling Guidelines

1. **Set Appropriate Delays**: Minimum 1-2 seconds between requests
2. **Respect robots.txt**: Always enable robots.txt checking
3. **Use Reasonable Limits**: Set max pages and depth limits
4. **Monitor Resource Usage**: Track memory and disk usage
5. **Implement Proper Error Handling**: Handle failures gracefully

### Performance Optimization

1. **Enable Caching**: Use disk cache for large crawl sessions
2. **Configure Connection Pooling**: Optimize for your target sites
3. **Use Content Fingerprinting**: Avoid processing duplicate content
4. **Monitor Quality Scores**: Filter low-quality content early
5. **Batch Process**: Use pipeline integration for efficiency

### Security Considerations

1. **Content Scanning**: Enable malware and security scanning
2. **Domain Restrictions**: Use allowlists for trusted domains
3. **Quarantine Suspicious Content**: Isolate potentially harmful content
4. **Regular Updates**: Keep security patterns updated
5. **Audit Logs**: Maintain detailed crawling logs

## Troubleshooting

### Common Issues

#### High Memory Usage
- Reduce `max_entries` in cache configuration
- Enable disk cache instead of memory-only
- Process content in smaller batches

#### Rate Limiting
- Increase `request_delay` in security configuration
- Implement exponential backoff for retry logic
- Monitor server response codes

#### Content Quality Issues
- Adjust quality scoring thresholds
- Review boilerplate detection patterns
- Validate structured data extraction

#### Connection Problems
- Configure connection pooling limits
- Enable connection optimization
- Monitor circuit breaker status

### Debugging Tools

```python
# Get comprehensive statistics
stats = crawler.get_session_stats()
print(f"Cache hit rate: {stats['cache_stats']['hit_rate']}")
print(f"Retry attempts: {stats['retry_stats']['total_attempts']}")

# Export detailed session report
await pipeline.export_session_results(session_id, "debug_report.json")

# Monitor connection optimization
conn_stats = connection_optimizer.get_connection_stats()
print(f"Connection reuse rate: {conn_stats['reuse_rate']}")
```

## Edge Cases Handled

### Network Issues
- Connection timeouts and failures
- DNS resolution problems
- SSL/TLS certificate issues
- HTTP redirect loops

### Content Challenges
- Malformed HTML and XML
- Large files exceeding limits
- Binary content in text fields
- Special characters and encodings

### Server Responses
- 5xx server errors with retry logic
- 429 rate limiting with backoff
- 404 and client errors (no retry)
- 304 Not Modified cache validation

## Testing

The system includes comprehensive test coverage:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Full pipeline testing
- **Edge Case Tests**: Error condition handling
- **Performance Tests**: Load and stress testing

Run tests with:
```bash
pytest tests/unit/test_enhanced_web_crawling.py -v
```

## Future Enhancements

### Planned Features
- JavaScript rendering support (via Playwright)
- Advanced content classification
- Multi-language content detection
- Site structure analysis
- Automated sitemap discovery

### Performance Improvements
- HTTP/2 support implementation
- Parallel domain crawling
- Smart crawl scheduling
- Content change detection

## Support

For issues or questions about the web crawling system:

1. Check the comprehensive test suite for examples
2. Review the pipeline integration documentation
3. Monitor session statistics and error logs
4. Use the debugging tools and export features

The system is designed to be production-ready with enterprise-grade reliability, security, and performance features.