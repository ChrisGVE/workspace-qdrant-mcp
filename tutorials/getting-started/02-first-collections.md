# First Steps with Collections

## Objectives
- Understand how workspace-qdrant-mcp automatically creates collections
- Learn collection naming conventions
- Create your first project-specific collections
- Understand the difference between project and global collections

## Prerequisites
- [Installation and Setup](01-installation-setup.md) completed successfully
- Qdrant server running and accessible
- Basic understanding of vector databases

## Overview
Collections are the foundation of workspace-qdrant-mcp's organization system. Unlike traditional Qdrant usage where you manually manage collections, workspace-qdrant-mcp automatically creates project-scoped collections based on your workspace and configuration.

**Estimated time**: 15-20 minutes

## Step 1: Understanding Automatic Collection Creation

Navigate to any Git repository (or create a test one):

```bash
# Create a test project
mkdir my-test-project
cd my-test-project
git init
echo "# My Test Project" > README.md
git add README.md
git commit -m "Initial commit"
```

Now check what collections would be created:

```bash
# Check workspace status
wqutil workspace-status
```

**Expected output**:
```
🏗️ Workspace Status for: my-test-project

📁 Project Information:
   - Name: my-test-project  
   - Type: Git Repository
   - Path: /path/to/my-test-project
   - Branch: main

📊 Collections (Auto-created):
   ✅ my-test-project-scratchbook   (Personal notes & ideas)
   ✅ my-test-project-project       (Project documentation) 
   🌐 references                   (Global collection)

📋 Configuration:
   - COLLECTIONS: project
   - GLOBAL_COLLECTIONS: references
   - GITHUB_USER: (not set)
```

## Step 2: Collection Naming Conventions

workspace-qdrant-mcp follows a consistent naming pattern:

### Always Created Collections
- **`{project-name}-scratchbook`** - Your personal development journal
  - Notes, ideas, TODOs, meeting minutes
  - Code snippets and implementation patterns
  - Bug reports and troubleshooting notes

### Project Collections (Based on COLLECTIONS env var)
- **`{project-name}-{suffix}`** format
- Common suffixes: `project`, `docs`, `tests`, `api`, `frontend`, `backend`

### Global Collections (Based on GLOBAL_COLLECTIONS env var)  
- **`{name}`** - shared across all projects
- Common names: `references`, `standards`, `docs`, `tools`

### Examples with Different Configurations

```bash
# Configuration: COLLECTIONS="docs,tests,api"
# GLOBAL_COLLECTIONS="references,standards"
# Project: "ecommerce-app"

Collections created:
- ecommerce-app-scratchbook  (automatic)
- ecommerce-app-docs         (from COLLECTIONS)
- ecommerce-app-tests        (from COLLECTIONS)  
- ecommerce-app-api          (from COLLECTIONS)
- references                 (from GLOBAL_COLLECTIONS)
- standards                  (from GLOBAL_COLLECTIONS)
```

## Step 3: Verify Collection Creation

Start Claude Desktop or Claude Code and test collection creation by storing a document:

### Using Claude Desktop

In a new conversation, try:

```
"Store this note in my project scratchbook: This is my first note about the my-test-project. I want to experiment with automatic project detection and collection management."
```

### Using Claude Code

```bash
# From your project directory
claude

# Then ask Claude:
# "Store a test document in my project scratchbook"
```

**Expected response**: Claude should confirm the document was stored and mention the collection name.

## Step 4: List Created Collections

Check what collections were actually created:

```bash
# List all collections
wqutil list-collections
```

**Expected output**:
```
📚 Available Collections:

Project Collections:
✅ my-test-project-scratchbook (1 documents, 384 dim)
✅ my-test-project-project (0 documents, 384 dim)

Global Collections:  
✅ references (0 documents, 384 dim)

Total: 3 collections, 1 documents
```

## Step 5: Understanding Collection Purposes

Each collection type has a specific purpose:

### Scratchbook Collections
**Purpose**: Personal development journal for the project
**What to store**:
```bash
# Example content for scratchbook
"Meeting notes from architecture review - decided to use microservices"
"TODO: Implement rate limiting for API endpoints"  
"Bug found: memory leak in worker threads, use weak references"
"Code pattern: async context managers for database connections"
"Research: OAuth 2.0 vs JWT for authentication - JWT chosen for simplicity"
```

### Project Collections  
**Purpose**: Official project documentation and code
**What to store**:
```bash
# Example content for project collection
"API documentation for user authentication endpoints"
"Database schema documentation"
"Deployment guide and infrastructure setup"
"Architecture decision records (ADRs)"
"User stories and requirements documentation"
```

### Global Collections
**Purpose**: Cross-project resources and standards  
**What to store**:
```bash
# Example content for global collections
"Company coding standards and best practices"
"Reusable code libraries and utilities"
"Security guidelines and compliance requirements"  
"Infrastructure templates and configurations"
"Third-party service documentation"
```

## Step 6: Test Search Across Collections

Store content in different collections and test cross-collection search:

### Store in Scratchbook
```
Claude: "Store this in my scratchbook: Found a great authentication pattern using JWT tokens with refresh rotation"
```

### Store in Project Collection  
```
Claude: "Store this in my project collection: User Authentication API - Endpoints for login, logout, token refresh, and user profile"
```

### Search Across All Collections
```
Claude: "Search for anything related to authentication in my project"
```

**Expected result**: Claude should find and return content from both collections, showing that search works across all project collections automatically.

## Step 7: Working with Subprojects

If you have Git submodules and a configured GitHub username, additional collections are created:

### Setup Subproject Example
```bash
# From your main project
git submodule add https://github.com/yourusername/frontend-module frontend
git submodule add https://github.com/yourusername/backend-module backend

# Check workspace status
wqutil workspace-status
```

**Expected output with GITHUB_USER set**:
```
📁 Main Project: my-test-project
📁 Subprojects Detected:
   - frontend (github.com/yourusername/frontend-module)
   - backend (github.com/yourusername/backend-module)

📊 Collections Created:
   my-test-project-scratchbook
   my-test-project-project
   frontend-scratchbook
   frontend-project  
   backend-scratchbook
   backend-project
   references
```

## Troubleshooting

### Collections Not Created
**Issue**: No collections appear when storing documents

**Solutions**:
```bash
# Check if you're in a Git repository
git status

# Check workspace detection
wqutil workspace-status

# Verify Qdrant connection  
workspace-qdrant-test --component qdrant
```

### Project Name Detection Issues
**Issue**: Unexpected project names or collection names

**Solutions**:
```bash
# Check current directory name
basename $(pwd)

# Collections use sanitized directory names
# "My-Project!" becomes "my-project"
```

### Missing Global Collections
**Issue**: Global collections not created

**Solutions**:
```bash
# Check GLOBAL_COLLECTIONS environment variable
echo $GLOBAL_COLLECTIONS

# Set if missing
export GLOBAL_COLLECTIONS="references,standards"

# Restart Claude to pick up changes
```

## Configuration Examples

### Minimal Configuration
```bash
# Simplest setup - just project and scratchbook collections
export COLLECTIONS="project"
# Creates: {project}-scratchbook, {project}-project
```

### Development Team Configuration  
```bash  
# Comprehensive setup for development teams
export COLLECTIONS="docs,tests,api,frontend,backend"
export GLOBAL_COLLECTIONS="standards,tools,references"
# Creates rich collection structure for organized teams
```

### Research Configuration
```bash
# Optimized for research and documentation
export COLLECTIONS="papers,notes,data"
export GLOBAL_COLLECTIONS="references,bibliography"
# Creates specialized collections for academic work
```

## Next Steps

🎉 **Excellent!** You now understand how workspace-qdrant-mcp automatically manages collections.

**What's next:**
- [Basic Search Operations](03-basic-search.md) - Learn effective search strategies
- [Verification and Testing](04-verification.md) - Test your complete setup
- [Collections Deep Dive](../basic-usage/01-collections-deep-dive.md) - Advanced collection management

## Quick Reference

### Key Commands
```bash
wqutil workspace-status      # Check project detection
wqutil list-collections     # List all collections
workspace-qdrant-test       # Test full system
```

### Collection Naming Pattern
- **Scratchbook**: `{project}-scratchbook` (automatic)
- **Project**: `{project}-{suffix}` (from COLLECTIONS)  
- **Global**: `{name}` (from GLOBAL_COLLECTIONS)

---

**Need help?** Check [Common Issues](../troubleshooting/01-common-issues.md) or the [API Reference](../../API.md).