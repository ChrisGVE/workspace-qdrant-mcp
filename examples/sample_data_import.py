#!/usr/bin/env python3
"""
sample_data_import.py - Import sample data for testing workspace-qdrant-mcp examples

This script creates sample data across different domains to demonstrate
workspace-qdrant-mcp functionality with realistic examples.
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

def create_sample_software_development_data() -> List[Dict[str, Any]]:
    """Create sample software development documents."""
    return [
        {
            "content": """# Authentication Service Architecture Decision Record

**Date:** 2024-01-15
**Status:** Accepted
**Decision:** Use JWT tokens with RSA256 signing for stateless authentication

## Context
We need a scalable authentication system that can handle multiple services without shared state.

## Decision
Implement JWT-based authentication with:
- RSA256 signing for security
- 24-hour token expiration
- Refresh token mechanism
- Role-based access control

## Consequences
- **Positive:** Stateless, scalable, secure
- **Negative:** Token size larger than session IDs, requires key management
""",
            "metadata": {
                "type": "architecture_decision",
                "title": "JWT Authentication System",
                "author": "Alice Johnson",
                "status": "accepted",
                "date": "2024-01-15",
                "tags": ["authentication", "jwt", "architecture", "security"],
                "category": "backend"
            }
        },
        {
            "content": """# API Rate Limiting Implementation

## Overview
Implementation of rate limiting for our REST API to prevent abuse and ensure fair usage.

## Implementation Details
- Token bucket algorithm with Redis backend
- Per-user and per-IP rate limiting
- Configurable limits per endpoint
- Graceful degradation under high load

## Code Example
```python
@rate_limit(requests_per_minute=60, burst=10)
def api_endpoint(request):
    # API logic here
    pass
```

## Testing Strategy
- Unit tests for rate limiter logic
- Integration tests with Redis
- Load testing to verify limits
- Monitoring and alerting setup
""",
            "metadata": {
                "type": "technical_documentation",
                "title": "API Rate Limiting Implementation",
                "author": "Bob Smith",
                "created_date": "2024-01-20",
                "tags": ["api", "rate-limiting", "redis", "performance"],
                "category": "backend",
                "implementation_status": "completed"
            }
        },
        {
            "content": """# Frontend Component Library Guidelines

## Component Design Principles
1. **Atomic Design:** Components should be small and focused
2. **Accessibility First:** All components must meet WCAG 2.1 AA standards
3. **Consistent API:** Props should follow established naming conventions
4. **Documentation:** Each component needs usage examples

## File Structure
```
components/
├── atoms/          # Basic building blocks
├── molecules/      # Simple combinations
├── organisms/      # Complex components
└── templates/      # Page layouts
```

## Best Practices
- Use TypeScript for all components
- Include unit tests with React Testing Library
- Document props with JSDoc comments
- Follow the style guide for CSS-in-JS
""",
            "metadata": {
                "type": "development_guidelines",
                "title": "Frontend Component Library Guidelines",
                "author": "Carol Davis",
                "created_date": "2024-02-01",
                "tags": ["frontend", "react", "components", "guidelines"],
                "category": "frontend",
                "team": "ui-ux"
            }
        }
    ]

def create_sample_research_data() -> List[Dict[str, Any]]:
    """Create sample research documents."""
    return [
        {
            "content": """# Attention Mechanisms in Neural Networks: A Comprehensive Survey

## Abstract
This survey examines the evolution of attention mechanisms in deep learning, from early sequence-to-sequence models to modern transformer architectures. We analyze 150+ papers and identify key innovations that have shaped the field.

## Key Findings
1. **Scaled Dot-Product Attention** revolutionized the field by making attention computationally efficient
2. **Multi-Head Attention** allows models to attend to different representation subspaces
3. **Self-Attention** enables modeling long-range dependencies within sequences

## Future Directions
- Sparse attention patterns for longer sequences
- Attention mechanisms for multimodal inputs
- Interpretability and explainability of attention weights

## References
[1] Vaswani et al. "Attention Is All You Need" (2017)
[2] Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
[3] Luong et al. "Effective Approaches to Attention-based Neural Machine Translation" (2015)
""",
            "metadata": {
                "type": "research_paper",
                "title": "Attention Mechanisms in Neural Networks: A Comprehensive Survey",
                "authors": ["Dr. Emily Chen", "Prof. Michael Rodriguez"],
                "publication_date": "2024-03-15",
                "journal": "Journal of Machine Learning Research",
                "tags": ["attention", "neural-networks", "transformers", "survey"],
                "field": "machine-learning",
                "citation_count": 45,
                "peer_reviewed": True
            }
        },
        {
            "content": """# Literature Review: Federated Learning Privacy Mechanisms

## Research Question
How do different privacy-preserving mechanisms affect the performance and security of federated learning systems?

## Methodology
Systematic review of 75 papers published between 2020-2024, focusing on:
- Differential privacy implementations
- Secure aggregation protocols
- Homomorphic encryption approaches
- Multi-party computation techniques

## Key Insights
1. **Trade-off Analysis:** Strong privacy guarantees often come with performance costs
2. **Practical Implementations:** Many theoretical approaches struggle in real-world scenarios
3. **Emerging Trends:** Hybrid approaches combining multiple privacy mechanisms

## Research Gaps
- Limited evaluation on heterogeneous devices
- Insufficient analysis of communication costs
- Need for standardized privacy metrics

## Recommendations for Future Work
1. Develop more efficient secure aggregation protocols
2. Create benchmark datasets for privacy evaluation
3. Investigate personalized differential privacy
""",
            "metadata": {
                "type": "literature_review",
                "title": "Literature Review: Federated Learning Privacy Mechanisms",
                "author": "Sarah Kim",
                "created_date": "2024-02-28",
                "tags": ["federated-learning", "privacy", "differential-privacy", "literature-review"],
                "field": "privacy-ml",
                "papers_reviewed": 75,
                "status": "in-progress"
            }
        }
    ]

def create_sample_business_data() -> List[Dict[str, Any]]:
    """Create sample business documents."""
    return [
        {
            "content": """# Q1 2024 Product Strategy Meeting

**Date:** January 10, 2024
**Attendees:** CEO, CTO, VP Product, VP Engineering, VP Marketing
**Duration:** 2 hours

## Agenda
1. Q4 2023 Performance Review
2. Market Analysis and Competitive Landscape
3. Q1 2024 Product Roadmap
4. Resource Allocation

## Key Decisions
- **Product Focus:** Prioritize mobile app development over web platform
- **Market Strategy:** Target SMB segment with simplified pricing
- **Technology Stack:** Migrate to microservices architecture
- **Team Expansion:** Hire 3 senior engineers and 1 product manager

## Action Items
- [ ] **John (CTO):** Draft technical architecture plan by Jan 20
- [ ] **Sarah (VP Product):** Complete market research analysis by Jan 15
- [ ] **Mike (VP Eng):** Provide hiring timeline and budget by Jan 12
- [ ] **Lisa (VP Marketing):** Create go-to-market strategy by Jan 25

## Next Meeting
January 24, 2024 - Product roadmap finalization
""",
            "metadata": {
                "type": "meeting_notes",
                "title": "Q1 2024 Product Strategy Meeting",
                "organizer": "John Smith",
                "meeting_date": "2024-01-10",
                "attendees": ["CEO", "CTO", "VP Product", "VP Engineering", "VP Marketing"],
                "tags": ["strategy", "product", "quarterly-planning"],
                "action_items_count": 4,
                "follow_up_required": True
            }
        },
        {
            "content": """# Customer Onboarding Process Documentation

## Overview
Standardized process for onboarding new enterprise customers to ensure consistent experience and reduce time-to-value.

## Process Steps

### Phase 1: Initial Setup (Days 1-3)
1. Welcome email with account details
2. Assign dedicated customer success manager
3. Schedule kickoff call within 24 hours
4. Provide access to knowledge base and training materials

### Phase 2: Configuration (Days 4-14)
1. Technical setup and integration
2. Data migration assistance
3. Custom configuration based on use case
4. Initial training sessions for admin users

### Phase 3: Launch Preparation (Days 15-21)
1. User acceptance testing
2. Final configuration review
3. Go-live planning and rollout strategy
4. Success metrics definition

### Phase 4: Launch & Support (Days 22-30)
1. Go-live execution
2. Daily check-ins for first week
3. Performance monitoring and optimization
4. Feedback collection and process improvement

## Success Metrics
- Time to first value: < 21 days
- Customer satisfaction score: > 4.5/5
- Feature adoption rate: > 80% within 30 days
- Support ticket volume: < 5 per customer in first month
""",
            "metadata": {
                "type": "process_documentation",
                "title": "Customer Onboarding Process",
                "owner": "Customer Success Team",
                "last_updated": "2024-03-01",
                "version": "2.1",
                "tags": ["onboarding", "customer-success", "process"],
                "approval_status": "approved",
                "review_date": "2024-06-01"
            }
        }
    ]

def create_sample_personal_data() -> List[Dict[str, Any]]:
    """Create sample personal knowledge documents."""
    return [
        {
            "content": """# Learning Plan: Advanced Python Concepts

## Goal
Master advanced Python programming concepts to improve code quality and performance in my projects.

## Topics to Cover

### 1. Metaclasses and Descriptors
- Understanding `__new__` vs `__init__`
- Creating custom metaclasses
- Property descriptors and data validation
- **Resources:** "Effective Python" by Brett Slatkin, Chapter 6

### 2. Concurrency and Parallelism
- asyncio and async/await patterns
- Threading vs multiprocessing trade-offs
- GIL understanding and workarounds
- **Resources:** "Python Tricks" by Dan Bader, AsyncIO documentation

### 3. Performance Optimization
- Profiling with cProfile and line_profiler
- Memory optimization techniques
- Cython for performance-critical code
- **Resources:** "High Performance Python" by Micha Gorelick

## Study Schedule
- **Week 1-2:** Metaclasses and descriptors
- **Week 3-4:** Concurrency patterns
- **Week 5-6:** Performance optimization
- **Week 7:** Practice project implementation

## Success Criteria
- [ ] Complete all reading materials
- [ ] Implement 3 practical examples for each topic
- [ ] Apply concepts to current work project
- [ ] Write blog post summarizing key learnings
""",
            "metadata": {
                "type": "learning_plan",
                "title": "Learning Plan: Advanced Python Concepts",
                "created_date": "2024-01-05",
                "target_completion": "2024-03-01",
                "tags": ["python", "learning", "programming", "advanced"],
                "category": "professional-development",
                "status": "in-progress",
                "progress": 30
            }
        },
        {
            "content": """# Daily Note - March 15, 2024

## Today's Focus
- [ ] Complete Python metaclass tutorial
- [ ] Review PR feedback from team lead  
- [ ] Prepare presentation for next week's tech talk
- [x] Update project documentation

## Ideas & Insights
- **Metaclass realization:** They're not as complex as I thought - just classes that create classes
- **Code review insight:** Adding type hints significantly improves code readability
- **Presentation idea:** Demo live coding vs explaining concepts - live coding is more engaging

## Learnings
- Learned about `__init_subclass__` as an alternative to metaclasses
- Discovered `dataclasses.field()` for better default value handling
- Found a great resource for async programming patterns

## Tomorrow's Preparation
- Finish metaclass exercises
- Review async/await best practices
- Outline tech talk structure

## Gratitude
- Grateful for patient code reviewers who provide detailed feedback
- Appreciate having interesting technical challenges to solve
- Thankful for the learning resources available online

---
**Energy Level:** 8/10
**Mood:** Focused and motivated
**Weather:** Sunny, 72°F
""",
            "metadata": {
                "type": "daily_note",
                "date": "2024-03-15",
                "tags": ["daily", "programming", "learning", "reflection"],
                "mood": "focused",
                "energy_level": 8,
                "completed_tasks": 1,
                "total_tasks": 4
            }
        },
        {
            "content": """# Project Idea: Personal Knowledge Graph Visualizer

## Concept
Build an interactive web application that visualizes connections between notes, ideas, and concepts in a personal knowledge base.

## Core Features
1. **Node Visualization:** Notes as nodes, relationships as edges
2. **Interactive Exploration:** Click and drag to explore connections
3. **Search and Filter:** Find specific topics or note types
4. **Automatic Linking:** Suggest connections based on content similarity
5. **Timeline View:** Show knowledge evolution over time

## Technical Approach
- **Frontend:** React with D3.js for visualization
- **Backend:** FastAPI with graph database (Neo4j)
- **AI Integration:** Use embeddings to find semantic relationships
- **Data Source:** Import from various note-taking apps

## Implementation Plan
### Phase 1: MVP (4 weeks)
- Basic graph visualization
- Simple node/edge creation
- File import functionality

### Phase 2: Intelligence (4 weeks)
- Automatic relationship detection
- Search and filtering
- Note content preview

### Phase 3: Advanced Features (4 weeks)
- Timeline visualization
- Knowledge gap detection
- Export and sharing capabilities

## Potential Impact
- Better understanding of personal knowledge connections
- Identify gaps in learning and research
- Visual approach to knowledge management
- Could be useful for researchers and knowledge workers

## Next Steps
- [x] Research existing solutions (Obsidian, Roam Research)
- [ ] Create technical architecture diagram
- [ ] Set up development environment
- [ ] Build basic graph visualization prototype
""",
            "metadata": {
                "type": "project_idea",
                "title": "Personal Knowledge Graph Visualizer",
                "created_date": "2024-02-20",
                "tags": ["project", "visualization", "knowledge-management", "web-app"],
                "category": "side-project",
                "status": "planning",
                "estimated_effort": "12 weeks",
                "excitement_level": 9
            }
        }
    ]

def import_sample_data(client, project_name: str, domain: str = "all"):
    """Import sample data for specified domain(s)."""
    
    data_generators = {
        "software_development": create_sample_software_development_data,
        "research": create_sample_research_data,
        "business": create_sample_business_data,
        "personal": create_sample_personal_data
    }
    
    if domain == "all":
        domains_to_import = data_generators.keys()
    else:
        domains_to_import = [domain] if domain in data_generators else []
    
    if not domains_to_import:
        print(f"❌ Unknown domain: {domain}")
        return
    
    total_imported = 0
    
    for domain_name in domains_to_import:
        print(f"\n📚 Importing {domain_name} sample data...")
        
        sample_data = data_generators[domain_name]()
        
        for i, doc in enumerate(sample_data):
            try:
                # Add project-specific metadata
                doc["metadata"]["project"] = project_name
                doc["metadata"]["sample_data"] = True
                doc["metadata"]["import_date"] = datetime.now().isoformat()
                doc["metadata"]["domain"] = domain_name
                
                # Determine collection name
                collection = f"{project_name}-{domain_name.replace('_', '-')}"
                
                # Store document
                result = client.store(
                    content=doc["content"],
                    metadata=doc["metadata"],
                    collection=collection
                )
                
                if result:
                    print(f"  ✅ Imported: {doc['metadata'].get('title', f'Document {i+1}')}")
                    total_imported += 1
                else:
                    print(f"  ❌ Failed to import document {i+1}")
                    
            except Exception as e:
                print(f"  ❌ Error importing document {i+1}: {str(e)}")
    
    print(f"\n🎉 Import complete! Total documents imported: {total_imported}")
    
    # Create summary document
    summary_content = f"""# Sample Data Import Summary

**Project:** {project_name}
**Import Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Domains Imported:** {', '.join(domains_to_import)}
**Total Documents:** {total_imported}

## Collections Created
{chr(10).join(f"- {project_name}-{domain.replace('_', '-')}" for domain in domains_to_import)}

## Usage with Claude
Try these example searches:

**Software Development:**
- "Find architecture decisions about authentication"
- "Search for API documentation and guidelines"

**Research:**
- "Show me papers about attention mechanisms"
- "Find literature reviews on privacy in machine learning"

**Business:**
- "Search for meeting notes about product strategy"
- "Find process documentation for customer onboarding"

**Personal:**
- "Show me learning plans and study materials"
- "Find project ideas related to visualization"

## Next Steps
1. Explore the imported data using Claude
2. Add your own content to the collections
3. Try the example workflows in the documentation
"""
    
    client.store(
        content=summary_content,
        metadata={
            "type": "import_summary",
            "title": f"Sample Data Import - {project_name}",
            "project": project_name,
            "import_date": datetime.now().isoformat(),
            "domains_imported": list(domains_to_import),
            "total_documents": total_imported
        },
        collection=f"{project_name}-scratchbook"
    )
    
    print(f"📋 Import summary saved to {project_name}-scratchbook collection")

def main():
    """Main entry point for sample data import."""
    parser = argparse.ArgumentParser(description="Import sample data for workspace-qdrant-mcp examples")
    parser.add_argument("--project-name", required=True, help="Project name for collections")
    parser.add_argument("--domain", default="all", 
                       choices=["all", "software_development", "research", "business", "personal"],
                       help="Domain to import (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be imported without importing")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print(f"🔍 DRY RUN - Sample data import for project '{args.project_name}'")
        print(f"Domain: {args.domain}")
        
        data_generators = {
            "software_development": create_sample_software_development_data,
            "research": create_sample_research_data,
            "business": create_sample_business_data,
            "personal": create_sample_personal_data
        }
        
        if args.domain == "all":
            domains = data_generators.keys()
        else:
            domains = [args.domain]
        
        for domain in domains:
            if domain in data_generators:
                sample_data = data_generators[domain]()
                print(f"\n📚 {domain} ({len(sample_data)} documents):")
                for doc in sample_data:
                    title = doc["metadata"].get("title", "Untitled")
                    doc_type = doc["metadata"].get("type", "unknown")
                    print(f"  • {title} ({doc_type})")
        
        return
    
    try:
        from workspace_qdrant_mcp.client import WorkspaceClient
        client = WorkspaceClient()
        print(f"✅ Connected to workspace-qdrant-mcp")
    except Exception as e:
        print(f"❌ Failed to connect to workspace-qdrant-mcp: {e}")
        print("Make sure the MCP server is running and accessible")
        return
    
    try:
        import_sample_data(client, args.project_name, args.domain)
        print(f"\n🚀 Sample data import complete!")
        print(f"You can now use Claude to search and interact with the imported data.")
        print(f"Try: 'Search my {args.project_name} project for architecture decisions'")
    except Exception as e:
        print(f"❌ Import failed: {e}")

if __name__ == "__main__":
    main()