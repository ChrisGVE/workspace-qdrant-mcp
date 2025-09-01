# Basic Search Operations

## Objectives
- Perform your first semantic searches
- Understand hybrid search (semantic + keyword)
- Learn effective search strategies
- Test search across multiple collections
- Verify search results and quality

## Prerequisites
- [Installation and Setup](01-installation-setup.md) completed
- [First Steps with Collections](02-first-collections.md) completed
- At least one document stored in collections

## Overview
workspace-qdrant-mcp provides powerful hybrid search that combines semantic understanding with precise keyword matching. This tutorial teaches you effective search strategies through hands-on examples.

**Estimated time**: 20-25 minutes

## Step 1: Populate Collections with Test Data

First, let's create diverse content to search through. Use Claude to store various types of information:

### Store Technical Documentation
```
Claude: "Store this in my project collection: 

Authentication System Documentation

Our authentication system uses JWT tokens with the following components:
- Login endpoint: POST /api/auth/login
- Logout endpoint: POST /api/auth/logout  
- Token refresh: POST /api/auth/refresh
- User profile: GET /api/user/profile

The JWT tokens expire after 1 hour and include user ID, roles, and permissions.
Refresh tokens are valid for 30 days and stored securely in httpOnly cookies.

Security considerations:
- All endpoints use HTTPS in production
- Tokens are signed with RS256 algorithm
- Rate limiting applied: 10 requests per minute per IP
- CSRF protection enabled for state-changing operations
"
```

### Store Development Notes
```
Claude: "Store this in my scratchbook:

Development Log - Week 1

Today I worked on the authentication system. Key decisions:
- Chose JWT over session-based auth for stateless API design
- Implemented refresh token rotation for enhanced security  
- Added rate limiting to prevent brute force attacks

Issues encountered:
- CORS problems with localhost development (fixed with proper headers)
- TypeScript types needed for JWT payload structure
- Database connection pooling causing timeout issues

Next steps:
- Add comprehensive unit tests for auth middleware
- Implement role-based access control (RBAC)
- Set up integration tests with test database
- Documentation review with security team
"
```

### Store Meeting Notes
```
Claude: "Store this in my scratchbook:

Team Meeting Notes - Architecture Review

Attendees: Sarah (Lead), Mike (Backend), Lisa (Frontend), Tom (DevOps)

Decisions made:
1. Microservices architecture approved for user management
2. PostgreSQL chosen over MongoDB for relational data integrity
3. Redis for caching and session storage
4. Docker containers with Kubernetes orchestration

Action items:
- Mike: Set up API gateway by Friday
- Lisa: Create design system components
- Tom: Configure CI/CD pipeline with automated testing
- Sarah: Document service boundaries and communication patterns

Concerns raised:
- Network latency between services
- Distributed transaction complexity
- Monitoring and observability needs
"
```

### Store Research Information
```  
Claude: "Store this in my references collection:

OAuth 2.0 vs JWT Comparison Research

OAuth 2.0:
- Authorization framework, not authentication
- Supports multiple grant types (authorization code, client credentials, etc.)
- Excellent for third-party integrations
- Requires authorization server infrastructure
- Better for complex permission scenarios

JWT (JSON Web Tokens):
- Stateless authentication tokens
- Self-contained with claims and signatures
- Simpler implementation for first-party applications  
- No server-side session storage required
- Better performance for distributed systems

Recommendation: Use JWT for internal API authentication, OAuth 2.0 for third-party integrations.

Sources:
- RFC 6749 (OAuth 2.0)
- RFC 7519 (JWT)
- Auth0 documentation
- OWASP authentication guidelines
"
```

## Step 2: Basic Semantic Search

Now let's test semantic search - finding content based on meaning, not just exact words:

### Search for Authentication Concepts
```
Claude: "Search my project for information about user authentication"
```

**Expected results**: Should find content from both technical documentation and development notes, even though the search term doesn't exactly match all the text.

### Search for Security Information
```
Claude: "Search for security considerations and vulnerabilities"
```

**Expected results**: Should find security-related content from multiple collections, including CSRF protection, rate limiting, and security guidelines.

### Search for Development Process
```
Claude: "Search for anything about team decisions and project planning"
```

**Expected results**: Should find meeting notes, architecture decisions, and development planning information.

## Step 3: Keyword Search Precision

Test precise keyword matching for specific technical terms:

### Search for Specific Endpoints
```
Claude: "Search for '/api/auth/login' in my project"
```

**Expected results**: Should precisely find the authentication documentation containing the exact endpoint.

### Search for Technology Names
```
Claude: "Search for 'PostgreSQL' in my collections"
```

**Expected results**: Should find meeting notes mentioning PostgreSQL specifically.

### Search for Code Patterns
```
Claude: "Search for 'JWT tokens' across my project"
```

**Expected results**: Should find multiple documents mentioning JWT tokens with exact phrase matching.

## Step 4: Understanding Hybrid Search

workspace-qdrant-mcp combines semantic and keyword search using Reciprocal Rank Fusion (RRF):

### Semantic + Keyword Example
```
Claude: "Search for database connection problems and timeouts"
```

**Analysis of results**:
- **Semantic matching**: Finds related concepts like "connection pooling", "timeout issues"
- **Keyword matching**: Finds exact mentions of "database", "connection", "timeout"  
- **RRF fusion**: Combines and ranks results from both approaches

### Observe Result Quality
```
Claude: "Search for development workflow and testing strategies"
```

**What to notice**:
- Results ranked by relevance (semantic similarity + keyword importance)
- Multiple collection sources included automatically
- Context preserved in result snippets

## Step 5: Cross-Collection Search Power

Test how search automatically spans all your collections:

### Search Across All Collection Types
```
Claude: "Search for anything related to authentication or login functionality"
```

**Expected results should include**:
- Technical docs from project collection
- Development notes from scratchbook  
- Research information from references
- Meeting decisions from scratchbook

### Verify Collection Coverage
```
Claude: "What collections did you search through for authentication information?"
```

**Expected response**: Claude should mention it searched across all available collections: project, scratchbook, and references collections.

## Step 6: Search Result Analysis

### Understanding Result Structure
When Claude returns search results, notice the structure:

```
Found 3 relevant results:

1. From my-project-project collection:
   "Authentication System Documentation - Our authentication system uses JWT tokens..."
   
2. From my-project-scratchbook collection:  
   "Development Log - Week 1 - Today I worked on the authentication system..."
   
3. From references collection:
   "OAuth 2.0 vs JWT Comparison Research - JWT (JSON Web Tokens): Stateless authentication..."
```

### Relevance Scoring
Higher-ranked results indicate:
- **Higher semantic similarity** to your search query
- **More keyword matches** in the content
- **Better combination score** from RRF algorithm

## Step 7: Advanced Search Strategies

### Use Natural Language Queries
```
Claude: "What did I decide about database choices and why?"
```

### Ask Specific Questions
```  
Claude: "What security measures are implemented in our authentication system?"
```

### Search for Patterns or Examples
```
Claude: "Find any code examples or technical implementations I've documented"
```

### Time-based Queries
```
Claude: "What were the main issues I encountered during development?"
```

## Verification Steps

### Test Search Quality
1. **Precision Test**: Search for specific technical terms
   - Should find exact matches quickly
   - No irrelevant results

2. **Recall Test**: Search for broad concepts  
   - Should find all related content
   - Include synonyms and related terms

3. **Cross-Collection Test**: Search for themes spanning collections
   - Should aggregate information from all sources
   - Maintain context and attribution

### Performance Check
```bash
# Run performance benchmarks
workspace-qdrant-test --benchmark
```

**Expected output**:
```
🔍 Search Performance Benchmarks:

Query Processing:
- Embedding generation: 45ms avg
- Semantic search: 23ms avg  
- Keyword search: 18ms avg
- RRF fusion: 12ms avg
- Total query time: 98ms avg

Quality Metrics:
- Precision@5: 0.92
- Recall@10: 0.89  
- MRR (Mean Reciprocal Rank): 0.85

✅ Performance: Excellent
```

## Troubleshooting

### No Search Results
**Issue**: Searches return empty results

**Solutions**:
```bash
# Check if documents exist
wqutil list-collections

# Verify search is working
workspace-qdrant-test --component search

# Check collection status
wqutil collection-info my-project-scratchbook
```

### Poor Search Quality  
**Issue**: Irrelevant results or missing relevant content

**Solutions**:
```bash
# Check embedding model
workspace-qdrant-test --component embedding

# Consider upgrading embedding model
export FASTEMBED_MODEL="BAAI/bge-base-en-v1.5"

# Rebuild collections with better embeddings
```

### Slow Search Performance
**Issue**: Searches take too long

**Solutions**:
```bash
# Run performance analysis
workspace-qdrant-health --analyze

# Check system resources
workspace-qdrant-health --watch

# Consider Qdrant optimization
curl -X PUT http://localhost:6333/collections/my-project-scratchbook/index \
  -H "Content-Type: application/json" \
  -d '{"index_params": {"m": 16, "ef_construct": 200}}'
```

## Search Best Practices

### Effective Query Strategies

1. **Specific Technical Terms**: Use exact names, APIs, technologies
   ```
   "Search for FastAPI middleware configuration"
   ```

2. **Conceptual Queries**: Ask about ideas, patterns, decisions
   ```
   "What architectural patterns did we discuss?"
   ```

3. **Problem-Solution Format**: Search for issues and resolutions
   ```
   "Find problems we encountered with database connections"
   ```

4. **Question Format**: Use natural language questions
   ```
   "How did we implement user authentication?"
   ```

### Query Optimization Tips

- **Use 2-8 words** for optimal semantic matching
- **Include context words** like "implementation", "problem", "decision"
- **Combine technical and conceptual terms**: "JWT authentication security"
- **Ask follow-up questions** to refine results

## Next Steps

🎉 **Great work!** You've mastered basic search operations with workspace-qdrant-mcp.

**What's next:**
- [Verification and Testing](04-verification.md) - Complete system verification
- [Document Management](../basic-usage/02-document-management.md) - Advanced document operations
- [Search Strategies](../basic-usage/03-search-strategies.md) - Master advanced search techniques

## Quick Reference

### Search Command Examples
```bash
# Through Claude
"Search my project for [query]"
"Find information about [topic]"  
"What do I have documented about [subject]?"

# Performance testing
workspace-qdrant-test --benchmark
workspace-qdrant-health --analyze
```

### Search Quality Factors
- **Semantic similarity**: Conceptual relevance
- **Keyword matching**: Exact term presence  
- **RRF scoring**: Combined ranking algorithm
- **Cross-collection**: Searches all collections automatically

---

**Need help?** Check [Search Strategies Deep Dive](../basic-usage/03-search-strategies.md) or [Performance Problems](../troubleshooting/02-performance-problems.md).