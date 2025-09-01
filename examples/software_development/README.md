# Software Development Examples

Practical examples for using workspace-qdrant-mcp in software development workflows, including code documentation, architecture decisions, and project onboarding.

## 🎯 Overview

This section demonstrates how workspace-qdrant-mcp enhances software development workflows by:

- **Code Documentation** - Automated documentation search and management
- **Architecture Decisions** - Tracking and retrieving design decisions  
- **Project Onboarding** - New team member knowledge transfer
- **Code Review** - Contextual code review with historical knowledge
- **Bug Tracking** - Linking issues to relevant code and documentation

## 🏗️ Examples Structure

```
software_development/
├── README.md                    # This file
├── code_documentation/          # Code docs workflow
│   ├── sample_project/         # Example codebase
│   ├── documentation_config.py  # Doc automation
│   └── claude_prompts.md       # Ready-to-use prompts
├── architecture_decisions/      # ADR workflow
│   ├── sample_adrs/           # Example ADRs
│   ├── adr_templates/         # ADR templates
│   └── decision_search.py     # Search decisions
├── project_onboarding/         # New team member setup
│   ├── onboarding_checklist.md # Step-by-step guide
│   ├── knowledge_extraction.py # Extract project knowledge
│   └── interactive_guide.py   # Interactive onboarding
├── code_review/                # Code review enhancement
│   ├── review_context.py      # Contextual reviews
│   ├── historical_analysis.py # Pattern analysis
│   └── claude_review_prompts.md
└── bug_tracking/               # Enhanced bug tracking
    ├── issue_context.py       # Link issues to code
    ├── solution_database.py   # Solution knowledge base
    └── debugging_workflows.py
```

## 🚀 Quick Start

### 1. Basic Setup

```bash
# Navigate to software development examples
cd examples/software_development

# Install dependencies
pip install -r requirements.txt

# Configure workspace-qdrant-mcp
export COLLECTIONS="project,docs,tests,architecture"
export GLOBAL_COLLECTIONS="patterns,solutions,best-practices"
```

### 2. Initialize Sample Project

```bash
# Set up the sample project
python setup_sample_project.py

# Ingest sample documentation
workspace-qdrant-ingest sample_project/ --collection myproject-docs

# Test the setup
python test_integration.py
```

### 3. Claude Integration

In Claude Desktop or Claude Code, try these commands:

**Code Documentation:**
- "Search my project documentation for authentication patterns"
- "Find all API endpoint documentation"
- "Show me examples of error handling in this project"

**Architecture Search:**
- "What architecture decisions have we made about caching?"
- "Find documentation about our microservices communication patterns"
- "Show me the reasoning behind our database choice"

## 📚 Example Workflows

### Code Documentation Workflow

**Automated Documentation Management:**

```python
# documentation_config.py - Automated doc processing
from workspace_qdrant_mcp.client import WorkspaceClient
import os
import ast

class DocumentationManager:
    def __init__(self):
        self.client = WorkspaceClient()
        
    def extract_code_documentation(self, source_dir):
        """Extract docstrings and comments from Python files"""
        docs = []
        
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    docs.extend(self.parse_python_file(file_path))
        
        return docs
    
    def parse_python_file(self, file_path):
        """Parse Python file for documentation"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            docs = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    doc = ast.get_docstring(node)
                    if doc:
                        docs.append({
                            'type': 'docstring',
                            'name': node.name,
                            'file': file_path,
                            'content': doc,
                            'line_number': node.lineno
                        })
                        
            return docs
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
    
    def store_documentation(self, docs, collection='project-docs'):
        """Store documentation in Qdrant"""
        for doc in docs:
            metadata = {
                'type': doc['type'],
                'name': doc['name'],
                'file': doc['file'],
                'line_number': doc.get('line_number', 0)
            }
            
            self.client.store(
                content=doc['content'],
                metadata=metadata,
                collection=collection
            )

# Usage example
if __name__ == "__main__":
    dm = DocumentationManager()
    docs = dm.extract_code_documentation('./sample_project/')
    dm.store_documentation(docs)
    print(f"Stored {len(docs)} documentation entries")
```

### Architecture Decisions Recording

**ADR (Architecture Decision Record) Management:**

```python
# adr_manager.py - Architecture Decision Records
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ArchitectureDecision:
    title: str
    status: str  # proposed, accepted, deprecated, superseded
    context: str
    decision: str
    consequences: str
    date: datetime
    tags: List[str]

class ADRManager:
    def __init__(self, client):
        self.client = client
        self.adr_template = """
# ADR-{number}: {title}

**Status:** {status}
**Date:** {date}
**Tags:** {tags}

## Context
{context}

## Decision
{decision}

## Consequences
{consequences}

## References
- Related ADRs: {related}
- Implementation: {implementation}
"""

    def create_adr(self, adr: ArchitectureDecision) -> str:
        """Create new ADR and store in knowledge base"""
        adr_number = self._get_next_adr_number()
        
        adr_content = self.adr_template.format(
            number=adr_number,
            title=adr.title,
            status=adr.status,
            date=adr.date.strftime("%Y-%m-%d"),
            tags=", ".join(adr.tags),
            context=adr.context,
            decision=adr.decision,
            consequences=adr.consequences,
            related="TBD",
            implementation="TBD"
        )
        
        # Store in both file system and Qdrant
        adr_file = f"architecture_decisions/sample_adrs/adr-{adr_number:03d}-{adr.title.lower().replace(' ', '-')}.md"
        os.makedirs(os.path.dirname(adr_file), exist_ok=True)
        
        with open(adr_file, 'w') as f:
            f.write(adr_content)
            
        # Store in Qdrant for searchable access
        metadata = {
            'type': 'architecture_decision',
            'adr_number': adr_number,
            'title': adr.title,
            'status': adr.status,
            'date': adr.date.isoformat(),
            'tags': adr.tags,
            'file_path': adr_file
        }
        
        self.client.store(
            content=adr_content,
            metadata=metadata,
            collection='architecture-decisions'
        )
        
        return adr_file
    
    def search_decisions(self, query: str, tags: List[str] = None) -> List[Dict]:
        """Search architecture decisions"""
        search_metadata = {'type': 'architecture_decision'}
        if tags:
            search_metadata['tags'] = tags
            
        results = self.client.search(
            query=query,
            collection='architecture-decisions',
            metadata_filter=search_metadata,
            limit=10
        )
        
        return results
    
    def _get_next_adr_number(self) -> int:
        """Get the next ADR number"""
        adr_dir = "architecture_decisions/sample_adrs"
        if not os.path.exists(adr_dir):
            return 1
            
        existing_files = [f for f in os.listdir(adr_dir) if f.startswith('adr-')]
        if not existing_files:
            return 1
            
        numbers = []
        for f in existing_files:
            try:
                num = int(f.split('-')[1])
                numbers.append(num)
            except (IndexError, ValueError):
                continue
                
        return max(numbers, default=0) + 1

# Sample ADR creation
if __name__ == "__main__":
    from workspace_qdrant_mcp.client import WorkspaceClient
    
    client = WorkspaceClient()
    adr_manager = ADRManager(client)
    
    # Create sample ADR
    sample_adr = ArchitectureDecision(
        title="Use FastAPI for REST API",
        status="accepted",
        context="We need a high-performance Python web framework for our REST API that supports modern Python features like type hints and async/await.",
        decision="We will use FastAPI as our primary web framework for building REST APIs.",
        consequences="Positive: Excellent performance, automatic OpenAPI documentation, type safety. Negative: Relatively new ecosystem compared to Flask/Django.",
        date=datetime.now(),
        tags=["api", "framework", "python", "performance"]
    )
    
    adr_file = adr_manager.create_adr(sample_adr)
    print(f"Created ADR: {adr_file}")
```

### Project Onboarding System

**Automated Knowledge Extraction for New Team Members:**

```python
# onboarding_assistant.py - Interactive onboarding
class OnboardingAssistant:
    def __init__(self, client):
        self.client = client
        
    def extract_project_knowledge(self, project_path: str):
        """Extract comprehensive project knowledge"""
        knowledge_areas = {
            'setup': self._extract_setup_info(project_path),
            'architecture': self._extract_architecture_info(project_path),
            'dependencies': self._extract_dependencies(project_path),
            'testing': self._extract_testing_info(project_path),
            'deployment': self._extract_deployment_info(project_path),
            'troubleshooting': self._extract_troubleshooting_info(project_path)
        }
        
        # Store each area in Qdrant with proper metadata
        for area, info in knowledge_areas.items():
            if info:
                self.client.store(
                    content=info,
                    metadata={
                        'type': 'onboarding',
                        'area': area,
                        'project': os.path.basename(project_path),
                        'extracted_date': datetime.now().isoformat()
                    },
                    collection='onboarding-knowledge'
                )
        
        return knowledge_areas
    
    def generate_onboarding_checklist(self, project_name: str) -> str:
        """Generate personalized onboarding checklist"""
        # Search for existing onboarding knowledge
        setup_info = self.client.search(
            query="setup installation requirements",
            collection='onboarding-knowledge',
            metadata_filter={'project': project_name, 'area': 'setup'}
        )
        
        arch_info = self.client.search(
            query="architecture overview components",
            collection='onboarding-knowledge',
            metadata_filter={'project': project_name, 'area': 'architecture'}
        )
        
        checklist = f"""
# {project_name} Onboarding Checklist

## Phase 1: Environment Setup
- [ ] Clone repository: `git clone <repo-url>`
- [ ] Install dependencies (see setup documentation)
- [ ] Configure environment variables
- [ ] Run tests to verify setup
- [ ] Access development database/services

## Phase 2: Architecture Understanding
- [ ] Review architecture documentation
- [ ] Understand core components and their interactions
- [ ] Review ADRs (Architecture Decision Records)
- [ ] Understand data flow and API contracts

## Phase 3: Development Workflow
- [ ] Set up IDE/editor with project configurations
- [ ] Understand branching strategy and PR process
- [ ] Review coding standards and style guides
- [ ] Set up local development workflow

## Phase 4: Domain Knowledge
- [ ] Review business requirements and user stories
- [ ] Understand key business processes
- [ ] Review existing feature documentation
- [ ] Identify key stakeholders and contacts

## Phase 5: Hands-on Practice
- [ ] Complete 'good first issue' tickets
- [ ] Participate in code review process
- [ ] Deploy to staging environment
- [ ] Shadow experienced team member

## Resources
"""
        
        if setup_info:
            checklist += "\n### Setup Information:\n"
            for result in setup_info[:3]:  # Top 3 results
                checklist += f"- {result['content'][:200]}...\n"
        
        if arch_info:
            checklist += "\n### Architecture Information:\n"
            for result in arch_info[:3]:
                checklist += f"- {result['content'][:200]}...\n"
        
        return checklist
    
    def _extract_setup_info(self, project_path: str) -> str:
        """Extract setup and installation information"""
        setup_files = ['README.md', 'INSTALL.md', 'setup.py', 'requirements.txt', 
                      'package.json', 'Dockerfile', 'docker-compose.yml']
        
        setup_info = []
        for file in setup_files:
            file_path = os.path.join(project_path, file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    setup_info.append(f"=== {file} ===\n{content}\n")
        
        return "\n".join(setup_info) if setup_info else None
    
    # Additional extraction methods...
    def _extract_architecture_info(self, project_path: str) -> str:
        """Extract architecture documentation"""
        arch_files = []
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if any(keyword in file.lower() for keyword in 
                      ['architecture', 'design', 'overview', 'adr']):
                    arch_files.append(os.path.join(root, file))
        
        arch_content = []
        for file_path in arch_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    arch_content.append(f"=== {os.path.basename(file_path)} ===\n{content}\n")
            except UnicodeDecodeError:
                continue
                
        return "\n".join(arch_content) if arch_content else None

# Usage example
if __name__ == "__main__":
    from workspace_qdrant_mcp.client import WorkspaceClient
    
    client = WorkspaceClient()
    assistant = OnboardingAssistant(client)
    
    # Extract knowledge for current project
    knowledge = assistant.extract_project_knowledge('./sample_project/')
    
    # Generate onboarding checklist
    checklist = assistant.generate_onboarding_checklist('sample-project')
    
    with open('onboarding_checklist.md', 'w') as f:
        f.write(checklist)
    
    print("Onboarding knowledge extracted and checklist generated!")
```

## 💡 Claude Interaction Prompts

### Ready-to-Use Prompts

**Code Documentation Search:**
```
Search my project documentation for:
- Authentication and authorization patterns
- Database schema and migration patterns
- API endpoint documentation and examples
- Error handling and logging strategies
- Testing patterns and best practices
```

**Architecture Decision Search:**
```
Find architecture decisions about:
- Technology stack choices and rationale
- Database design decisions
- API design patterns
- Security architecture decisions
- Performance optimization choices
```

**Code Review Context:**
```
For this code review, please:
1. Search for similar patterns in our codebase
2. Check if this follows our established architecture decisions
3. Verify it matches our coding standards and best practices
4. Look for related test cases and documentation
5. Identify any potential security or performance concerns
```

### Advanced Workflows

**Contextual Code Review:**
```python
# Use in Claude Code when reviewing PRs
"""
Before reviewing this PR, please gather context by:

1. Search our architecture decisions for relevant patterns
2. Find similar implementations in our codebase
3. Check our coding standards for this type of change
4. Look for existing tests that cover similar functionality
5. Review any related documentation or ADRs

Then provide a comprehensive review considering our project's context.
"""
```

## 📊 Best Practices

### Collection Organization

**Recommended collection structure for software development:**

```bash
# Project-specific collections
export COLLECTIONS="project,docs,tests,architecture,apis"

# Global collections for team knowledge
export GLOBAL_COLLECTIONS="patterns,best-practices,solutions,templates"

# Example result:
# myproject-project        # Core project documentation
# myproject-docs          # API docs, guides, tutorials  
# myproject-tests         # Test documentation and examples
# myproject-architecture  # ADRs and architecture docs
# myproject-apis          # API specifications and examples
# patterns                # Reusable code patterns
# best-practices         # Team coding standards
# solutions              # Common problem solutions
# templates              # Project templates and boilerplate
```

### Automated Documentation

**Set up automated documentation ingestion:**

```bash
# Add to your CI/CD pipeline
workspace-qdrant-ingest ./docs --collection $PROJECT_NAME-docs
workspace-qdrant-ingest ./api --collection $PROJECT_NAME-apis
workspace-qdrant-ingest ./architecture --collection $PROJECT_NAME-architecture
```

### Code Review Integration

**Enhance code reviews with contextual information:**

1. **Pre-review context gathering** - Search for similar patterns
2. **Historical analysis** - Review related past decisions  
3. **Pattern verification** - Ensure consistency with codebase
4. **Knowledge sharing** - Document review insights

## 🔗 Integration Examples

- **[VS Code Integration](../integrations/vscode/README.md)** - Workspace setup and snippets
- **[Automation Scripts](../integrations/automation/README.md)** - CI/CD integration
- **[Performance Optimization](../performance_optimization/README.md)** - Large codebase handling

---

**Next Steps:**
1. Try the [Project Onboarding Example](project_onboarding/)
2. Set up [Architecture Decision Recording](architecture_decisions/)
3. Explore [Code Review Enhancement](code_review/)