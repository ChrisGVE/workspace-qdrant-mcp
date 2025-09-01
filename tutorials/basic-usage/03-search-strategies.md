# Search Strategies

## Objectives
- Master advanced search techniques for maximum information retrieval
- Understand hybrid search optimization and query formulation  
- Learn context-aware search patterns for different use cases
- Implement iterative search refinement strategies
- Optimize search performance and quality

## Prerequisites
- [Getting Started](../getting-started/) section completed
- [Collections Deep Dive](01-collections-deep-dive.md) completed
- [Document Management](02-document-management.md) completed
- Substantial content stored across multiple collections (20+ documents)

## Overview
Effective search is the key to unlocking the value in your knowledge base. This tutorial teaches sophisticated search strategies that go beyond basic queries to find exactly what you need quickly and accurately.

**Estimated time**: 45-60 minutes

## Step 1: Understanding Hybrid Search Mechanics

### Semantic vs Keyword Search Components

workspace-qdrant-mcp uses Reciprocal Rank Fusion (RRF) to combine two search approaches:

#### Semantic Search (Vector Similarity)
- **Strengths**: Understands meaning, finds conceptually related content
- **Use cases**: Exploring ideas, finding related concepts, broad research

```bash
# Semantic search examples - finds conceptually similar content
"Search for user authentication concepts"
# Finds: JWT tokens, OAuth, session management, security middleware

"Find information about database performance"  
# Finds: query optimization, indexing, caching, connection pooling
```

#### Keyword Search (Text Matching)
- **Strengths**: Precise term matching, exact phrases, technical terms
- **Use cases**: Finding specific APIs, error codes, configuration values

```bash
# Keyword search examples - finds exact matches
"Search for '/api/auth/login' endpoint"
# Finds: exact endpoint documentation, implementation code

"Find error code 422 handling"
# Finds: specific error handling code, documentation
```

#### Hybrid Fusion (RRF Combined)
- **Best of both**: Semantic understanding + keyword precision
- **Automatic optimization**: No manual tuning required

```bash
# Hybrid search examples - combines both approaches
"Search for authentication endpoint implementation problems"
# Finds: both conceptual issues (authentication concepts) 
#        and specific problems (endpoint errors, implementation bugs)
```

## Step 2: Query Formulation Strategies

### Effective Query Construction

#### Length Optimization
```bash
# Optimal: 2-8 words with mix of concepts and specifics
"JWT authentication security best practices"
"Database connection timeout troubleshooting"  
"API rate limiting implementation guide"

# Too short: Lacks context
"JWT"
"Database"

# Too long: Dilutes search focus
"I need to find information about implementing JWT authentication with proper security considerations and error handling for our new API endpoints that will be used by mobile and web clients"
```

#### Concept + Specific Pattern
```bash
# Pattern: [Concept] + [Specific Implementation/Problem]
"authentication middleware configuration"
"database migration rollback procedure" 
"API error handling best practices"
"user interface responsive design patterns"

# Pattern: [Technology] + [Use Case/Problem]
"Redis caching session storage"
"PostgreSQL performance query optimization"
"React component state management"
```

### Query Types and Use Cases

#### Exploratory Queries
```bash
# Finding related concepts and ideas
"What approaches exist for user session management?"
"Find architectural patterns for microservices communication"
"Search for database design considerations"

# Use when: Starting research, looking for options, exploring topics
```

#### Specific Problem-Solving Queries
```bash
# Finding exact solutions to known problems
"JWT token expires too quickly error handling"
"Database connection pool exhausted fix"
"React component re-render optimization"

# Use when: Debugging specific issues, implementing known solutions
```

#### Implementation Guidance Queries
```bash
# Finding how-to information and procedures
"How to implement rate limiting for API"
"Steps for database schema migration"
"Process for deploying Docker containers"

# Use when: Need step-by-step guidance, implementation procedures
```

#### Reference and Lookup Queries
```bash
# Finding specific facts, configurations, or examples
"API endpoint for user profile updates"
"Database connection string format"
"Environment variables for production deployment"

# Use when: Need specific factual information, configuration values
```

## Step 3: Context-Aware Search Patterns

### Development Phase-Specific Searches

#### Planning Phase
```bash
# Requirements and design exploration
"Search for user requirements and specifications"
"Find architectural decisions and design patterns"
"What security considerations were documented?"

# Focus: Broad exploration, understanding requirements, design options
```

#### Implementation Phase
```bash
# Code examples and technical guidance
"Find code examples for authentication middleware"
"Search for API implementation patterns"
"What database schema designs were chosen?"

# Focus: Specific technical guidance, implementation examples
```

#### Testing Phase
```bash
# Test strategies and debugging
"Find test cases for authentication flows"
"Search for debugging approaches and troubleshooting"
"What performance benchmarks were established?"

# Focus: Quality assurance, testing procedures, problem resolution
```

#### Deployment Phase
```bash
# Infrastructure and operations
"Search for deployment procedures and configurations"
"Find monitoring and alerting setup guides"
"What production issues have been encountered?"

# Focus: Operational procedures, infrastructure, monitoring
```

### Problem-Solving Search Workflows

#### The Funnel Approach
```bash
# 1. Start broad - understand the domain
"Search for database performance issues"

# 2. Narrow to specific technology
"Search for PostgreSQL query performance problems"  

# 3. Focus on specific symptoms
"Search for PostgreSQL slow query timeout errors"

# 4. Find exact solutions
"Search for PostgreSQL query timeout configuration fixes"
```

#### The Context Expansion Approach
```bash
# 1. Start with specific problem
"Search for JWT token validation failing"

# 2. Expand to related components  
"Search for authentication middleware and JWT issues"

# 3. Broaden to system context
"Search for authentication system architecture and security"

# 4. Include operational context
"Search for authentication monitoring and troubleshooting procedures"
```

## Step 4: Advanced Search Techniques

### Multi-Perspective Search

#### Technical + Business Perspective
```bash
# Technical perspective
"Search for user authentication implementation details"

# Business perspective  
"Search for user authentication security requirements and compliance"

# Combined insights provide complete picture
```

#### Current + Historical Perspective
```bash
# Current state
"Search for current API authentication implementation"

# Historical context
"Search for authentication system evolution and previous implementations"

# Understanding both informs better decisions
```

### Iterative Search Refinement

#### Search Strategy Evolution
```bash
# Initial search: General exploration
"Search for API performance optimization"

# Refine based on results: Focus on specific area
"Search for API response time optimization techniques"

# Further refinement: Specific technology
"Search for Node.js API response caching strategies"

# Final focus: Implementation details
"Search for Redis caching implementation for Node.js APIs"
```

#### Search Result Analysis
```bash
# After each search, analyze results:
# 1. What relevant information was found?
# 2. What gaps exist in the results?
# 3. What additional context is needed?
# 4. How can the next query be improved?

# Use insights to formulate better follow-up queries
```

### Cross-Collection Search Strategies

#### Collection-Aware Query Formulation
```bash
# Leverage automatic cross-collection search
"Search all my documentation for authentication patterns"
# Searches: project-docs, scratchbook, references, global collections

# Understand which collections contribute to results
"Where did you find information about JWT implementation?"
# Claude will specify: "From your project-docs and scratchbook collections"
```

#### Collection-Specific Follow-ups
```bash
# If general search provides overview, dive into specific collections
"Search my scratchbook for personal notes about authentication issues"
"Find official documentation about authentication in my project docs"
"Look for authentication best practices in my references collection"
```

## Step 5: Search Quality Optimization

### Query Quality Assessment

#### Good Query Characteristics
```bash
# Specific enough to be focused
"JWT authentication middleware error handling"

# General enough to find related content  
"authentication error handling patterns"

# Includes both concepts and implementation terms
"Redis session storage implementation guide"

# Natural language that matches document content
"How to handle database connection timeouts"
```

#### Query Improvement Techniques
```bash
# Original query: "authentication problems"
# Improved: "JWT authentication validation failures"

# Original query: "database issues"  
# Improved: "PostgreSQL connection pool timeout problems"

# Original query: "API stuff"
# Improved: "REST API rate limiting implementation"
```

### Search Result Evaluation

#### Quality Indicators
```bash
# High-quality results show:
1. Relevant content that matches query intent
2. Appropriate mix of general concepts and specific details
3. Multiple perspectives on the topic
4. Recent and up-to-date information
5. Cross-references to related topics

# Example of good results for "authentication security":
- JWT token security best practices (concepts)
- Authentication middleware implementation (specifics)
- Security audit findings (real-world context)
- Related OAuth integration documentation (cross-references)
```

#### Result Analysis Questions
```bash
# For each search, ask:
1. Did I find what I was looking for?
2. What additional information would be helpful?
3. Are there gaps in my understanding?
4. What related topics should I explore?
5. How can I improve my next search?
```

### Performance Optimization

#### Search Performance Monitoring
```bash
# Monitor search performance
workspace-qdrant-test --benchmark --focus search

# Analyze slow queries
wqutil analyze-search-performance --slow-queries --days 7

# Optimize if needed
wqutil optimize-search-index --collections my-project-docs,my-project-scratchbook
```

#### Query Performance Tips
```bash
# Faster queries:
- Use specific terms (faster semantic matching)
- Avoid very long queries (reduces processing time)
- Search specific collections when appropriate

# Examples:
Fast: "JWT middleware implementation"
Slower: "comprehensive authentication and authorization middleware implementation patterns"
```

## Step 6: Specialized Search Scenarios

### Code and Technical Search

#### Finding Code Examples
```bash
# Effective queries for code search
"authentication middleware function implementation"
"database connection pool configuration code"
"API error handling middleware examples"
"JWT token validation function code"

# Include context for better matches
"Python Flask authentication middleware code example"
"Node.js Express rate limiting implementation"
```

#### Technical Troubleshooting
```bash
# Problem + symptoms pattern
"database connection timeout error symptoms and solutions"
"JWT token expired error handling implementation"
"API response slow performance investigation steps"

# Include error messages when available
"error 'connection timeout' database PostgreSQL troubleshooting"
```

### Documentation and Learning Search

#### Learning and Understanding
```bash
# Concept exploration queries
"explain authentication flow and security considerations"
"database design principles and best practices"
"API architecture patterns for scalable systems"

# How-to and tutorial queries
"step by step guide for implementing JWT authentication"
"tutorial for setting up database connection pooling"
```

#### Reference and Specification Search
```bash
# API and specification queries
"API endpoint specification for user authentication"
"database schema definition for user management"
"configuration parameters for Redis session storage"

# Standards and guidelines queries  
"coding standards for authentication implementation"
"security guidelines for API development"
```

### Research and Analysis Search

#### Comparative Analysis
```bash
# Comparison queries
"OAuth vs JWT authentication comparison analysis"
"PostgreSQL vs MongoDB performance characteristics"
"REST vs GraphQL API design trade-offs"

# Decision rationale queries
"why did we choose JWT over session-based authentication"
"database selection criteria and decision reasoning"
```

#### Trend and Evolution Search
```bash
# Historical evolution queries
"authentication system evolution and improvements over time"
"API design changes and version migration history"
"database schema migration history and lessons learned"

# Future planning queries
"roadmap for authentication system improvements"
"planned API enhancements and feature development"
```

## Step 7: Search Workflow Integration

### Daily Development Workflow

#### Morning Review Pattern
```bash
# 1. Check recent work context
"Search for recent development notes and progress updates"

# 2. Review current priorities
"Search for in-progress tasks and implementation plans"

# 3. Find relevant resources
"Search for documentation related to today's development tasks"
```

#### Problem-Solving Pattern
```bash
# 1. Define the problem
"Search for similar problems encountered before"

# 2. Find existing solutions
"Search for implementation patterns and solution approaches"

# 3. Get implementation details
"Search for code examples and technical specifications"

# 4. Validate approach
"Search for best practices and security considerations"
```

#### Knowledge Capture Pattern
```bash
# 1. Research background
"Search for existing knowledge about the topic"

# 2. Identify gaps  
"Search for missing information or incomplete documentation"

# 3. Document new insights
# Store new findings using document management best practices

# 4. Cross-reference
"Search to verify new information doesn't conflict with existing knowledge"
```

### Team Collaboration Search

#### Onboarding New Team Members
```bash
# Essential knowledge queries
"Search for getting started guides and setup procedures"
"Find architectural decisions and system design documentation"
"Search for coding standards and development processes"

# Historical context queries
"Search for project evolution and major milestone decisions"
"Find lessons learned and common pitfalls documentation"
```

#### Code Review and Knowledge Transfer
```bash
# Background research for reviews
"Search for design patterns used in this component"
"Find security guidelines relevant to this implementation"
"Search for performance considerations for this type of code"

# Knowledge sharing queries
"Search for similar implementations in other parts of the system"
"Find related documentation that reviewers should be aware of"
```

## Troubleshooting Search Issues

### Poor Search Results

#### Diagnosis Steps
```bash
# 1. Check collection health
wqutil collection-health my-project-docs

# 2. Verify document content
wqutil sample-documents my-project-docs --count 5

# 3. Test embedding quality
workspace-qdrant-test --component embedding

# 4. Analyze search performance
wqutil search-quality --test-queries "authentication,database,API"
```

#### Improvement Strategies
```bash
# Improve document quality
# - Add more descriptive content
# - Include relevant keywords
# - Use consistent terminology

# Optimize search queries  
# - Use specific but not overly narrow terms
# - Include both concepts and implementation details
# - Try different query formulations

# Consider embedding model upgrade
export FASTEMBED_MODEL="BAAI/bge-base-en-v1.5"
# Reindex collections after model change
```

### Search Performance Issues

```bash
# Monitor search latency
workspace-qdrant-health --watch --focus search

# Optimize slow collections
wqutil optimize-collection my-project-docs --focus search-performance

# Check system resources
wqutil system-resources --focus search-workload
```

## Best Practices Summary

### Query Formulation
1. **Optimal Length**: Use 2-8 words for best semantic matching
2. **Mix Concepts and Specifics**: Include both broad topics and specific terms
3. **Natural Language**: Write queries as you would ask a colleague
4. **Iterative Refinement**: Start broad, then narrow based on results

### Search Strategy
1. **Multi-Perspective**: Search from different angles (technical, business, historical)
2. **Context-Aware**: Adapt search style to current development phase
3. **Cross-Collection**: Leverage automatic search across all collections
4. **Result Analysis**: Always evaluate and learn from search results

### Performance Optimization
1. **Monitor Quality**: Regularly assess search result relevance
2. **Optimize Collections**: Keep collections healthy and well-indexed
3. **Document Quality**: Maintain high-quality, searchable content
4. **Query Performance**: Use efficient query patterns

## Next Steps

🎉 **Outstanding!** You've mastered advanced search strategies for workspace-qdrant-mcp.

**Complete Basic Usage:**
- [Scratchbook Usage](04-scratchbook-usage.md) - Optimize your personal development journal

**Advance to Integration:**
- [Claude Desktop Integration](../integration-guides/01-claude-desktop.md)
- [Claude Code Integration](../integration-guides/02-claude-code.md)

**Apply to Real Scenarios:**
- [Software Development Project](../use-cases/01-development-project.md)
- [Personal Knowledge Management](../use-cases/03-personal-knowledge.md)

## Quick Reference

### Query Patterns
```bash
# Concept + Implementation
"[technology] [implementation aspect]"
"JWT authentication middleware"

# Problem + Solution  
"[problem description] [solution/fix]"
"database timeout troubleshooting"

# Process + Context
"[process] [context/technology]"  
"deployment Docker containers"
```

### Search Workflow
```bash
1. Start broad: "authentication concepts"
2. Narrow focus: "JWT authentication implementation"  
3. Get specific: "JWT middleware error handling"
4. Find examples: "JWT validation code examples"
```

---

**Ready to optimize your scratchbook usage?** Continue to [Scratchbook Usage](04-scratchbook-usage.md).