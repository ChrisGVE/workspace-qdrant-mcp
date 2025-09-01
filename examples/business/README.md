# Business Examples

Practical examples for using workspace-qdrant-mcp in business workflows, including meeting notes, knowledge bases, and document management.

## 🎯 Overview

This section demonstrates how workspace-qdrant-mcp enhances business workflows by:

- **Meeting Management** - Capture, organize, and search meeting notes and action items
- **Knowledge Base** - Build and maintain organizational knowledge repositories
- **Document Management** - Organize business documents with intelligent search
- **Team Collaboration** - Share knowledge and insights across teams
- **Decision Tracking** - Track business decisions and their outcomes
- **Project Management** - Document project progress and lessons learned

## 🏗️ Examples Structure

```
business/
├── README.md                          # This file
├── meeting_management/                # Meeting workflows
│   ├── meeting_capture.py            # Meeting note automation
│   ├── action_item_tracker.py        # Action item management
│   ├── meeting_templates/             # Meeting note templates
│   └── agenda_generator.py           # Smart agenda generation
├── knowledge_base/                    # Organizational knowledge
│   ├── knowledge_manager.py          # Knowledge base management
│   ├── expertise_mapping.py          # Team expertise tracking
│   ├── best_practices.py             # Best practice documentation
│   └── process_documentation.py      # Process and procedure docs
├── document_management/               # Business document workflows
│   ├── document_classifier.py        # Automatic document classification
│   ├── contract_analyzer.py          # Contract and agreement analysis
│   ├── policy_manager.py             # Policy document management
│   └── compliance_tracker.py         # Compliance documentation
├── team_workflows/                    # Team collaboration
│   ├── project_knowledge.py          # Project-specific knowledge
│   ├── onboarding_system.py         # Employee onboarding
│   ├── skill_inventory.py           # Team skills and capabilities
│   └── communication_hub.py         # Team communication history
└── analytics/                        # Business intelligence
    ├── meeting_analytics.py          # Meeting effectiveness analysis
    ├── knowledge_gaps.py             # Identify knowledge gaps
    ├── collaboration_insights.py     # Team collaboration patterns
    └── decision_impact.py            # Decision outcome tracking
```

## 🚀 Quick Start

### 1. Business Setup

```bash
# Navigate to business examples
cd examples/business

# Install business-specific dependencies
pip install -r requirements.txt

# Configure collections for business workflow
export COLLECTIONS="meetings,documents,projects,processes"
export GLOBAL_COLLECTIONS="knowledge-base,policies,templates,best-practices"
```

### 2. Initialize Business Environment

```python
# business_setup.py - Initialize business collections
from workspace_qdrant_mcp.client import WorkspaceClient
from datetime import datetime

def setup_business_collections():
    """Initialize collections for business workflows."""
    client = WorkspaceClient()
    
    # Create business-specific collections
    collections = {
        'meetings': 'Meeting notes, agendas, and action items',
        'documents': 'Business documents and reports',
        'projects': 'Project documentation and updates',
        'processes': 'Business processes and procedures',
        'knowledge-base': 'Organizational knowledge and expertise',
        'policies': 'Company policies and guidelines',
        'templates': 'Document templates and frameworks',
        'best-practices': 'Documented best practices and lessons learned',
        'decisions': 'Business decisions and their rationale',
        'team-updates': 'Team status updates and communications'
    }
    
    for collection, description in collections.items():
        print(f"📁 Creating collection: {collection}")
        # In a real implementation, you'd create the collection
        # client.create_collection(collection, description)
    
    # Store initial organizational information
    org_info = f"""
# Organization Setup - {datetime.now().strftime('%Y-%m-%d')}

## Collections Initialized
{chr(10).join(f"- **{name}**: {desc}" for name, desc in collections.items())}

## Usage Guidelines

**Meeting Notes**: Store in 'meetings' collection with metadata including date, attendees, and meeting type
**Project Updates**: Store in 'projects' collection with project name and status
**Processes**: Document standard operating procedures in 'processes' collection
**Knowledge Sharing**: Store institutional knowledge in 'knowledge-base' collection

## Getting Started

1. Use Claude to store meeting notes: "Store this meeting summary in my meetings collection"
2. Search across all business documents: "Find all documents about quarterly planning"
3. Retrieve action items: "Show me all pending action items from last week's meetings"
    """
    
    client.store(
        content=org_info,
        metadata={
            'type': 'organization_setup',
            'setup_date': datetime.now().isoformat(),
            'collections_count': len(collections)
        },
        collection='knowledge-base'
    )
    
    return client

if __name__ == "__main__":
    client = setup_business_collections()
    print("🏢 Business environment ready!")
```

### 3. Claude Integration for Business

In Claude Desktop or Claude Code, try these business commands:

**Meeting Management:**
- "Store this meeting summary in my meetings collection: [meeting notes]"
- "Find all action items assigned to John from last week's meetings"
- "Search for meetings about the Q4 planning project"

**Knowledge Retrieval:**
- "What's our standard process for client onboarding?"
- "Find all documents related to the marketing campaign"
- "Show me best practices for project management in our organization"

**Document Management:**
- "Search all contracts for clauses about data retention"
- "Find policy documents updated in the last 6 months"
- "What are our compliance requirements for customer data?"

## 📚 Example Workflows

### Meeting Management System

**Comprehensive Meeting Capture and Action Item Tracking:**

```python
# meeting_management.py - Complete meeting management system
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

class MeetingType(Enum):
    """Types of business meetings."""
    STANDUP = "standup"
    PLANNING = "planning"
    REVIEW = "review"
    ALL_HANDS = "all_hands"
    ONE_ON_ONE = "one_on_one"
    CLIENT = "client"
    BOARD = "board"
    TRAINING = "training"

class ActionStatus(Enum):
    """Status of action items."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"

@dataclass
class ActionItem:
    """Represents an action item from a meeting."""
    id: str
    description: str
    assignee: str
    due_date: datetime
    priority: str = "medium"  # low, medium, high
    status: ActionStatus = ActionStatus.PENDING
    meeting_id: str = ""
    created_date: datetime = field(default_factory=datetime.now)
    completed_date: Optional[datetime] = None
    notes: str = ""
    dependencies: List[str] = field(default_factory=list)

@dataclass
class Meeting:
    """Represents a business meeting with comprehensive details."""
    id: str
    title: str
    date: datetime
    duration_minutes: int
    meeting_type: MeetingType
    attendees: List[str]
    organizer: str
    agenda: List[str] = field(default_factory=list)
    notes: str = ""
    action_items: List[ActionItem] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    follow_up_required: bool = False
    recording_url: Optional[str] = None
    transcript: Optional[str] = None

class MeetingManager:
    """
    Comprehensive meeting management system.
    
    Provides tools for capturing meeting notes, tracking action items,
    and analyzing meeting effectiveness across the organization.
    """
    
    def __init__(self, client):
        self.client = client
        self.meetings_db = {}  # In production, use proper database
        self.action_items_db = {}
        
    def capture_meeting(self, meeting: Meeting) -> str:
        """
        Capture and store comprehensive meeting information.
        
        Args:
            meeting: Meeting object with all details
            
        Returns:
            Meeting ID for reference
        """
        # Store meeting in database
        self.meetings_db[meeting.id] = meeting
        
        # Create comprehensive meeting content for search
        content = self._format_meeting_content(meeting)
        
        # Store in Qdrant with rich metadata
        metadata = {
            'type': 'meeting',
            'meeting_id': meeting.id,
            'title': meeting.title,
            'date': meeting.date.isoformat(),
            'meeting_type': meeting.meeting_type.value,
            'organizer': meeting.organizer,
            'attendees': meeting.attendees,
            'duration_minutes': meeting.duration_minutes,
            'action_items_count': len(meeting.action_items),
            'decisions_count': len(meeting.decisions),
            'follow_up_required': meeting.follow_up_required
        }
        
        self.client.store(
            content=content,
            metadata=metadata,
            collection='meetings'
        )
        
        # Store individual action items
        for action_item in meeting.action_items:
            self._store_action_item(action_item)
        
        print(f"📝 Meeting '{meeting.title}' captured successfully")
        return meeting.id
    
    def _format_meeting_content(self, meeting: Meeting) -> str:
        """Format meeting content for searchability."""
        content = f"""
# {meeting.title}

**Date:** {meeting.date.strftime('%Y-%m-%d %H:%M')}
**Type:** {meeting.meeting_type.value.title()}
**Duration:** {meeting.duration_minutes} minutes
**Organizer:** {meeting.organizer}

## Attendees
{chr(10).join(f"- {attendee}" for attendee in meeting.attendees)}

## Agenda
{chr(10).join(f"{i+1}. {item}" for i, item in enumerate(meeting.agenda))}

## Meeting Notes
{meeting.notes}

## Decisions Made
{chr(10).join(f"- {decision}" for decision in meeting.decisions)}

## Action Items
{chr(10).join(f"- **{item.assignee}**: {item.description} (Due: {item.due_date.strftime('%Y-%m-%d')})" for item in meeting.action_items)}

## Next Steps
{chr(10).join(f"- {step}" for step in meeting.next_steps)}

## Follow-up Required
{meeting.follow_up_required}
        """.strip()
        
        if meeting.transcript:
            content += f"\n\n## Transcript\n{meeting.transcript}"
        
        return content
    
    def _store_action_item(self, action_item: ActionItem):
        """Store individual action item for tracking."""
        self.action_items_db[action_item.id] = action_item
        
        content = f"""
# Action Item: {action_item.description}

**Assigned to:** {action_item.assignee}
**Due Date:** {action_item.due_date.strftime('%Y-%m-%d')}
**Priority:** {action_item.priority}
**Status:** {action_item.status.value}

## Details
{action_item.description}

## Notes
{action_item.notes}

**Meeting:** {action_item.meeting_id}
**Created:** {action_item.created_date.strftime('%Y-%m-%d')}
        """
        
        metadata = {
            'type': 'action_item',
            'action_id': action_item.id,
            'assignee': action_item.assignee,
            'due_date': action_item.due_date.isoformat(),
            'priority': action_item.priority,
            'status': action_item.status.value,
            'meeting_id': action_item.meeting_id,
            'is_overdue': action_item.due_date < datetime.now(),
            'days_until_due': (action_item.due_date - datetime.now()).days
        }
        
        self.client.store(
            content=content,
            metadata=metadata,
            collection='meetings'
        )
    
    def parse_meeting_notes(self, raw_notes: str, meeting_metadata: Dict[str, Any]) -> Meeting:
        """
        Parse raw meeting notes into structured Meeting object.
        
        Uses NLP techniques to extract action items, decisions, and structure.
        """
        # Extract basic information
        meeting_id = f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Parse action items using regex patterns
        action_items = self._extract_action_items(raw_notes, meeting_id)
        
        # Parse decisions
        decisions = self._extract_decisions(raw_notes)
        
        # Parse agenda items
        agenda = self._extract_agenda(raw_notes)
        
        # Create meeting object
        meeting = Meeting(
            id=meeting_id,
            title=meeting_metadata.get('title', 'Untitled Meeting'),
            date=meeting_metadata.get('date', datetime.now()),
            duration_minutes=meeting_metadata.get('duration', 60),
            meeting_type=MeetingType(meeting_metadata.get('type', 'planning')),
            attendees=meeting_metadata.get('attendees', []),
            organizer=meeting_metadata.get('organizer', 'Unknown'),
            agenda=agenda,
            notes=raw_notes,
            action_items=action_items,
            decisions=decisions
        )
        
        return meeting
    
    def _extract_action_items(self, notes: str, meeting_id: str) -> List[ActionItem]:
        """Extract action items from meeting notes."""
        action_items = []
        
        # Common patterns for action items
        patterns = [
            r"(?i)action(?:\s+item)?[:\s]+(.+?)(?=\n|\Z)",
            r"(?i)(?:TODO|to do)[:\s]+(.+?)(?=\n|\Z)",
            r"(?i)(@\w+)\s+(?:will|to|should)\s+(.+?)(?=\n|\Z)",
            r"(?i)(\w+)\s+(?:responsible for|assigned to)\s+(.+?)(?=\n|\Z)"
        ]
        
        action_id_counter = 1
        
        for pattern in patterns:
            matches = re.finditer(pattern, notes, re.MULTILINE)
            for match in matches:
                # Extract assignee and description
                if len(match.groups()) == 1:
                    description = match.group(1).strip()
                    assignee = "Unassigned"
                else:
                    assignee = match.group(1).strip().lstrip('@')
                    description = match.group(2).strip()
                
                # Default due date (1 week from now)
                due_date = datetime.now() + timedelta(days=7)
                
                action_item = ActionItem(
                    id=f"{meeting_id}_action_{action_id_counter}",
                    description=description,
                    assignee=assignee,
                    due_date=due_date,
                    meeting_id=meeting_id
                )
                
                action_items.append(action_item)
                action_id_counter += 1
        
        return action_items
    
    def _extract_decisions(self, notes: str) -> List[str]:
        """Extract decisions from meeting notes."""
        decisions = []
        
        patterns = [
            r"(?i)decision[:\s]+(.+?)(?=\n|\Z)",
            r"(?i)(?:we )?decided[:\s]+(.+?)(?=\n|\Z)",
            r"(?i)agreed[:\s]+(.+?)(?=\n|\Z)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, notes, re.MULTILINE)
            for match in matches:
                decision = match.group(1).strip()
                if decision not in decisions:
                    decisions.append(decision)
        
        return decisions
    
    def _extract_agenda(self, notes: str) -> List[str]:
        """Extract agenda items from meeting notes."""
        agenda = []
        
        # Look for numbered or bulleted lists at the beginning
        agenda_section = re.search(r"(?i)agenda[:\s]*\n((?:.*\n)*?)(?=\n\n|\Z)", notes)
        
        if agenda_section:
            agenda_text = agenda_section.group(1)
            items = re.findall(r"(?:^\d+\.|\-|\*)\s+(.+?)(?=\n|$)", agenda_text, re.MULTILINE)
            agenda = [item.strip() for item in items]
        
        return agenda
    
    def get_action_items_by_assignee(self, assignee: str, status: ActionStatus = None) -> List[Dict]:
        """Get action items assigned to specific person."""
        filters = {
            'type': 'action_item',
            'assignee': assignee
        }
        
        if status:
            filters['status'] = status.value
        
        results = self.client.search(
            query="action item",
            collection='meetings',
            metadata_filter=filters,
            limit=50
        )
        
        return results
    
    def get_overdue_action_items(self) -> List[Dict]:
        """Get all overdue action items."""
        results = self.client.search(
            query="action item",
            collection='meetings',
            metadata_filter={
                'type': 'action_item',
                'is_overdue': True
            },
            limit=100
        )
        
        return results
    
    def generate_meeting_summary(self, meeting_id: str) -> str:
        """Generate executive summary of a meeting."""
        if meeting_id not in self.meetings_db:
            return "Meeting not found"
        
        meeting = self.meetings_db[meeting_id]
        
        summary = f"""
# Meeting Summary: {meeting.title}

**Date:** {meeting.date.strftime('%Y-%m-%d')}
**Duration:** {meeting.duration_minutes} minutes
**Attendees:** {len(meeting.attendees)} participants

## Key Outcomes
- **Decisions Made:** {len(meeting.decisions)}
- **Action Items:** {len(meeting.action_items)}
- **Follow-up Required:** {'Yes' if meeting.follow_up_required else 'No'}

## Action Items Summary
{chr(10).join(f"- {item.assignee}: {item.description[:50]}..." for item in meeting.action_items)}

## Next Meeting
{meeting.next_steps[0] if meeting.next_steps else 'TBD'}
        """
        
        return summary.strip()
    
    def analyze_meeting_patterns(self, date_range: tuple = None) -> Dict[str, Any]:
        """Analyze meeting patterns and effectiveness."""
        # Get all meetings in date range
        filters = {'type': 'meeting'}
        
        results = self.client.search(
            query="meeting",
            collection='meetings',
            metadata_filter=filters,
            limit=1000
        )
        
        # Analyze patterns
        meeting_types = {}
        attendee_frequency = {}
        duration_analysis = []
        action_item_completion = {'completed': 0, 'pending': 0, 'overdue': 0}
        
        for result in results:
            metadata = result.get('metadata', {})
            
            # Meeting type analysis
            m_type = metadata.get('meeting_type', 'unknown')
            meeting_types[m_type] = meeting_types.get(m_type, 0) + 1
            
            # Duration analysis
            duration = metadata.get('duration_minutes', 0)
            if duration > 0:
                duration_analysis.append(duration)
            
            # Attendee frequency
            attendees = metadata.get('attendees', [])
            for attendee in attendees:
                attendee_frequency[attendee] = attendee_frequency.get(attendee, 0) + 1
        
        # Calculate statistics
        avg_duration = sum(duration_analysis) / len(duration_analysis) if duration_analysis else 0
        
        return {
            'total_meetings': len(results),
            'meeting_types': meeting_types,
            'average_duration': round(avg_duration, 1),
            'most_frequent_attendees': dict(sorted(attendee_frequency.items(), 
                                                 key=lambda x: x[1], reverse=True)[:10]),
            'action_item_summary': action_item_completion
        }

# Usage example
if __name__ == "__main__":
    from workspace_qdrant_mcp.client import WorkspaceClient
    
    client = WorkspaceClient()
    mm = MeetingManager(client)
    
    # Sample meeting notes
    raw_notes = """
    # Team Planning Meeting - Q4 2024
    
    Attendees: John Smith, Sarah Johnson, Mike Chen, Lisa Williams
    
    ## Agenda
    1. Q4 budget review
    2. Project status updates
    3. Resource planning
    
    ## Discussion
    We reviewed the Q4 budget and found we're 15% under budget.
    
    Sarah presented the status of the customer portal project - 80% complete.
    Mike updated on the mobile app development - facing some technical challenges.
    
    ## Decisions
    - Approved additional $50k for mobile development
    - Decided to extend customer portal launch by 2 weeks
    
    ## Action Items
    - @john will review budget allocations by Friday
    - @sarah to coordinate with QA team for portal testing
    - @mike will document technical challenges and solutions
    - @lisa responsible for updating stakeholder communications
    
    Meeting adjourned at 3:30 PM.
    """
    
    # Parse and capture meeting
    meeting_metadata = {
        'title': 'Team Planning Meeting - Q4 2024',
        'date': datetime.now(),
        'duration': 90,
        'type': 'planning',
        'attendees': ['John Smith', 'Sarah Johnson', 'Mike Chen', 'Lisa Williams'],
        'organizer': 'John Smith'
    }
    
    meeting = mm.parse_meeting_notes(raw_notes, meeting_metadata)
    meeting_id = mm.capture_meeting(meeting)
    
    print(f"📝 Meeting captured: {meeting_id}")
    print(f"🎯 Action items created: {len(meeting.action_items)}")
    
    # Generate summary
    summary = mm.generate_meeting_summary(meeting_id)
    print("\n" + summary)
```

### Knowledge Base Management

**Organizational Knowledge Repository:**

```python
# knowledge_base.py - Organizational knowledge management
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

class KnowledgeType(Enum):
    """Types of organizational knowledge."""
    PROCESS = "process"
    BEST_PRACTICE = "best_practice"
    LESSON_LEARNED = "lesson_learned"
    EXPERTISE = "expertise"
    FAQ = "faq"
    POLICY = "policy"
    TEMPLATE = "template"
    DECISION = "decision"

@dataclass
class KnowledgeItem:
    """Represents a piece of organizational knowledge."""
    id: str
    title: str
    content: str
    knowledge_type: KnowledgeType
    author: str
    department: str
    tags: List[str]
    created_date: datetime
    last_updated: datetime
    version: str = "1.0"
    related_items: List[str] = None
    approval_status: str = "draft"  # draft, approved, deprecated
    access_level: str = "internal"  # public, internal, confidential
    usage_count: int = 0
    feedback_score: float = 0.0

class KnowledgeBase:
    """
    Comprehensive organizational knowledge management system.
    
    Manages the creation, organization, and retrieval of institutional
    knowledge including processes, best practices, and lessons learned.
    """
    
    def __init__(self, client, organization_name: str):
        self.client = client
        self.organization_name = organization_name
        self.knowledge_items = {}  # In production, use proper database
        
        # Initialize knowledge taxonomy
        self._initialize_taxonomy()
    
    def _initialize_taxonomy(self):
        """Initialize organizational knowledge taxonomy."""
        taxonomy = {
            'departments': [
                'Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 
                'Operations', 'Legal', 'Customer Success'
            ],
            'knowledge_categories': {
                'Processes': ['Standard Operating Procedures', 'Workflows', 'Guidelines'],
                'Best Practices': ['Coding Standards', 'Communication', 'Project Management'],
                'Lessons Learned': ['Project Retrospectives', 'Incident Reports', 'Case Studies'],
                'Policies': ['HR Policies', 'Security Policies', 'Compliance']
            },
            'access_levels': ['Public', 'Internal', 'Confidential', 'Restricted']
        }
        
        # Store taxonomy in knowledge base
        content = f"""
# {self.organization_name} Knowledge Taxonomy

## Departments
{chr(10).join(f"- {dept}" for dept in taxonomy['departments'])}

## Knowledge Categories
"""
        for category, subcategories in taxonomy['knowledge_categories'].items():
            content += f"\n### {category}\n"
            content += chr(10).join(f"- {sub}" for sub in subcategories)
        
        content += f"""

## Access Levels
{chr(10).join(f"- {level}" for level in taxonomy['access_levels'])}

## Usage Guidelines

**Creating Knowledge Items:**
1. Choose appropriate knowledge type
2. Use clear, descriptive titles
3. Tag with relevant keywords
4. Assign to correct department
5. Set appropriate access level

**Maintaining Knowledge:**
- Review and update regularly
- Archive outdated information
- Link related knowledge items
- Collect usage feedback
        """
        
        self.client.store(
            content=content,
            metadata={
                'type': 'knowledge_taxonomy',
                'organization': self.organization_name,
                'created_date': datetime.now().isoformat()
            },
            collection='knowledge-base'
        )
    
    def create_knowledge_item(self, knowledge_item: KnowledgeItem) -> str:
        """Create and store a new knowledge item."""
        # Store in local database
        self.knowledge_items[knowledge_item.id] = knowledge_item
        
        # Create comprehensive content for search
        content = f"""
# {knowledge_item.title}

**Type:** {knowledge_item.knowledge_type.value.title()}
**Department:** {knowledge_item.department}
**Author:** {knowledge_item.author}
**Version:** {knowledge_item.version}
**Status:** {knowledge_item.approval_status}

## Content
{knowledge_item.content}

## Tags
{', '.join(knowledge_item.tags)}

## Metadata
- **Created:** {knowledge_item.created_date.strftime('%Y-%m-%d')}
- **Last Updated:** {knowledge_item.last_updated.strftime('%Y-%m-%d')}
- **Access Level:** {knowledge_item.access_level}
- **Usage Count:** {knowledge_item.usage_count}
- **Rating:** {knowledge_item.feedback_score:.1f}/5.0
        """
        
        metadata = {
            'type': 'knowledge_item',
            'knowledge_id': knowledge_item.id,
            'title': knowledge_item.title,
            'knowledge_type': knowledge_item.knowledge_type.value,
            'author': knowledge_item.author,
            'department': knowledge_item.department,
            'tags': knowledge_item.tags,
            'created_date': knowledge_item.created_date.isoformat(),
            'last_updated': knowledge_item.last_updated.isoformat(),
            'approval_status': knowledge_item.approval_status,
            'access_level': knowledge_item.access_level,
            'version': knowledge_item.version
        }
        
        self.client.store(
            content=content,
            metadata=metadata,
            collection='knowledge-base'
        )
        
        print(f"📚 Knowledge item created: {knowledge_item.title}")
        return knowledge_item.id
    
    def search_knowledge(self, query: str, filters: Dict[str, Any] = None) -> List[Dict]:
        """Search knowledge base with advanced filtering."""
        search_metadata = {'type': 'knowledge_item'}
        
        if filters:
            search_metadata.update(filters)
        
        results = self.client.search(
            query=query,
            collection='knowledge-base',
            metadata_filter=search_metadata,
            limit=20
        )
        
        # Update usage counts
        for result in results:
            knowledge_id = result.get('metadata', {}).get('knowledge_id')
            if knowledge_id in self.knowledge_items:
                self.knowledge_items[knowledge_id].usage_count += 1
        
        return results
    
    def get_knowledge_by_department(self, department: str) -> List[Dict]:
        """Get all knowledge items for a specific department."""
        return self.search_knowledge(
            query="*",
            filters={'department': department}
        )
    
    def get_expertise_map(self) -> Dict[str, List[str]]:
        """Create expertise map showing who knows what in the organization."""
        expertise_results = self.search_knowledge(
            query="*",
            filters={'knowledge_type': 'expertise'}
        )
        
        expertise_map = {}
        
        for result in expertise_results:
            metadata = result.get('metadata', {})
            author = metadata.get('author', 'Unknown')
            tags = metadata.get('tags', [])
            
            if author not in expertise_map:
                expertise_map[author] = []
            
            expertise_map[author].extend(tags)
        
        # Remove duplicates and sort
        for person, skills in expertise_map.items():
            expertise_map[person] = sorted(list(set(skills)))
        
        return expertise_map
    
    def generate_knowledge_gaps_report(self) -> str:
        """Identify potential knowledge gaps in the organization."""
        # Get all knowledge items
        all_knowledge = self.search_knowledge(query="*")
        
        # Analyze coverage by department and type
        department_coverage = {}
        type_coverage = {}
        
        for item in all_knowledge:
            metadata = item.get('metadata', {})
            dept = metadata.get('department', 'Unknown')
            k_type = metadata.get('knowledge_type', 'unknown')
            
            department_coverage[dept] = department_coverage.get(dept, 0) + 1
            type_coverage[k_type] = type_coverage.get(k_type, 0) + 1
        
        report = f"""
# Knowledge Gaps Analysis - {datetime.now().strftime('%Y-%m-%d')}

## Department Coverage
{chr(10).join(f"- **{dept}**: {count} items" for dept, count in sorted(department_coverage.items(), key=lambda x: x[1], reverse=True))}

## Knowledge Type Distribution
{chr(10).join(f"- **{k_type.title()}**: {count} items" for k_type, count in sorted(type_coverage.items(), key=lambda x: x[1], reverse=True))}

## Potential Gaps

### Departments with Low Coverage
{chr(10).join(f"- {dept} ({count} items)" for dept, count in department_coverage.items() if count < 5)}

### Missing Knowledge Types
- Areas that may need more documentation
- Consider creating templates for common processes
- Encourage best practice sharing

## Recommendations

1. **Focus on departments** with fewer than 5 knowledge items
2. **Create process documentation** for critical workflows
3. **Capture lessons learned** from completed projects
4. **Document expertise** of key team members
5. **Regular knowledge reviews** to keep information current
        """
        
        return report.strip()
    
    def create_best_practice(self, title: str, content: str, author: str, 
                           department: str, tags: List[str]) -> str:
        """Create a best practice knowledge item."""
        best_practice = KnowledgeItem(
            id=f"bp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=title,
            content=content,
            knowledge_type=KnowledgeType.BEST_PRACTICE,
            author=author,
            department=department,
            tags=tags,
            created_date=datetime.now(),
            last_updated=datetime.now()
        )
        
        return self.create_knowledge_item(best_practice)
    
    def create_process_documentation(self, title: str, steps: List[str], 
                                   owner: str, department: str) -> str:
        """Create process documentation."""
        content = f"""
## Process Overview
{title}

## Process Steps
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(steps))}

## Process Owner
{owner}

## Last Review Date
{datetime.now().strftime('%Y-%m-%d')}

## Related Documents
- [List any related policies, templates, or guides]

## Notes
- [Any additional notes or considerations]
        """
        
        process_doc = KnowledgeItem(
            id=f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=title,
            content=content,
            knowledge_type=KnowledgeType.PROCESS,
            author=owner,
            department=department,
            tags=["process", "documentation", department.lower()],
            created_date=datetime.now(),
            last_updated=datetime.now()
        )
        
        return self.create_knowledge_item(process_doc)
    
    def capture_lesson_learned(self, project_name: str, lesson: str, 
                             author: str, category: str) -> str:
        """Capture a lesson learned from a project or incident."""
        content = f"""
## Project/Context
{project_name}

## Lesson Learned
{lesson}

## Category
{category}

## Application
How this lesson can be applied in future projects or situations.

## Related Knowledge
- [Link to related processes, best practices, or documentation]
        """
        
        lesson_item = KnowledgeItem(
            id=f"lesson_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=f"Lesson from {project_name}: {lesson[:50]}...",
            content=content,
            knowledge_type=KnowledgeType.LESSON_LEARNED,
            author=author,
            department="Operations",  # Default department
            tags=["lesson-learned", category.lower(), project_name.lower().replace(' ', '-')],
            created_date=datetime.now(),
            last_updated=datetime.now()
        )
        
        return self.create_knowledge_item(lesson_item)

# Usage example
if __name__ == "__main__":
    from workspace_qdrant_mcp.client import WorkspaceClient
    
    client = WorkspaceClient()
    kb = KnowledgeBase(client, "TechCorp Inc.")
    
    # Create sample best practice
    kb.create_best_practice(
        title="Code Review Best Practices",
        content="""
        **Effective code reviews improve code quality and team collaboration.**
        
        ## Guidelines:
        1. Review code within 24 hours of PR creation
        2. Focus on logic, not style (use automated tools for style)
        3. Be constructive in feedback
        4. Test the code before approving
        5. Document complex decisions in comments
        
        ## Checklist:
        - [ ] Code follows team conventions
        - [ ] Tests are included and pass
        - [ ] Documentation is updated
        - [ ] No sensitive data is exposed
        - [ ] Performance impact considered
        """,
        author="Sarah Johnson",
        department="Engineering",
        tags=["code-review", "engineering", "quality", "collaboration"]
    )
    
    # Search knowledge base
    results = kb.search_knowledge("code review best practices")
    print(f"Found {len(results)} knowledge items about code review")
    
    # Generate gaps report
    gaps_report = kb.generate_knowledge_gaps_report()
    print("\n" + gaps_report)
```

## 💡 Claude Interaction Prompts

### Business-Focused Prompts

**Meeting Management:**
```
Help me with meeting management:
- Store these meeting notes with action items and decisions
- Find all action items assigned to [person] from last month
- Search for meetings about the [project name] project
- Show me overdue action items from team meetings
- Generate a summary of this week's key decisions
```

**Knowledge Retrieval:**
```
Search our knowledge base for:
- Standard process for client onboarding
- Best practices for project management
- Lessons learned from similar projects
- Company policies about [specific topic]
- Who has expertise in [technology/area]
```

**Document Management:**
```
Help me organize business documents:
- Find all contracts with [client name]
- Search policy documents updated in the last quarter
- Locate templates for [document type]
- Show me compliance requirements for [area]
- Find all documents related to [project/initiative]
```

### Advanced Business Workflows

**Strategic Planning Support:**
```python
# Use in Claude Code for strategic planning
"""
I'm preparing for strategic planning. Please help me gather:

1. All meeting notes from executive team meetings this quarter
2. Key decisions made about product direction
3. Lessons learned from major projects this year
4. Market research and competitive analysis documents
5. Customer feedback and satisfaction data
6. Team capacity and capability assessments

Focus on strategic insights and decision-making context.
"""
```

## 📊 Best Practices

### Collection Organization

**Recommended collection structure for business:**

```bash
# Business-specific collections
export COLLECTIONS="meetings,projects,processes,reports"

# Global business collections  
export GLOBAL_COLLECTIONS="knowledge-base,policies,templates,decisions"

# Example result:
# mycompany-meetings        # Meeting notes and action items
# mycompany-projects        # Project documentation and updates
# mycompany-processes       # Standard operating procedures
# mycompany-reports         # Business reports and analytics
# knowledge-base           # Organizational knowledge repository
# policies                 # Company policies and guidelines
# templates               # Document templates and frameworks
# decisions               # Business decisions and rationale
```

### Automated Business Workflows

**Set up automated knowledge capture:**

```bash
# Daily meeting note processing
python meeting_processor.py --process-new-meetings

# Weekly action item reminders
python action_tracker.py --send-reminders --overdue-only

# Monthly knowledge base health check
python knowledge_maintenance.py --check-outdated --suggest-updates
```

### Team Collaboration Integration

**Enhance team collaboration:**

1. **Shared knowledge** - Central repository for team expertise
2. **Meeting continuity** - Link related meetings and decisions
3. **Action accountability** - Track and remind about commitments
4. **Institutional memory** - Preserve important decisions and context

## 🔗 Integration Examples

- **[VS Code Integration](../integrations/vscode/README.md)** - Business workspace setup
- **[Automation Scripts](../integrations/automation/README.md)** - Meeting and document automation
- **[Performance Optimization](../performance_optimization/README.md)** - Large document collections

---

**Next Steps:**
1. Try the [Meeting Management Example](meeting_management/)
2. Set up [Knowledge Base System](knowledge_base/)
3. Explore [Team Collaboration](team_workflows/)