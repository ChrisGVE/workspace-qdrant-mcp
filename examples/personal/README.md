# Personal Examples

Practical examples for using workspace-qdrant-mcp in personal knowledge management workflows, including note-taking, learning, and idea organization.

## 🎯 Overview

This section demonstrates how workspace-qdrant-mcp enhances personal productivity and learning by:

- **Digital Note-Taking** - Capture and organize thoughts, ideas, and insights
- **Learning Management** - Track learning progress and build knowledge connections
- **Idea Development** - Nurture ideas from conception to implementation  
- **Personal Wiki** - Build a comprehensive personal knowledge base
- **Journal Integration** - Reflect and track personal growth
- **Project Planning** - Organize personal projects and goals

## 🏗️ Examples Structure

```
personal/
├── README.md                      # This file
├── note_taking/                   # Digital note-taking systems
│   ├── zettelkasten.py           # Zettelkasten note-taking method
│   ├── daily_notes.py            # Daily note capture and organization
│   ├── meeting_notes.py          # Personal meeting notes
│   └── note_templates/           # Note templates and formats
├── learning/                      # Learning and skill development
│   ├── learning_tracker.py       # Track courses, books, and progress
│   ├── concept_mapping.py        # Build knowledge concept maps
│   ├── skill_development.py      # Skill tracking and improvement
│   └── resource_library.py       # Learning resource organization
├── idea_management/               # Idea capture and development
│   ├── idea_capture.py           # Quick idea capture system
│   ├── project_brainstorm.py     # Project ideation and development
│   ├── creative_process.py       # Creative workflow management
│   └── innovation_journal.py     # Innovation and inspiration tracking
├── personal_wiki/                 # Comprehensive knowledge base
│   ├── wiki_builder.py           # Personal wiki construction
│   ├── knowledge_graph.py        # Personal knowledge relationships
│   ├── reference_manager.py      # Personal reference collection
│   └── topic_exploration.py      # Deep topic exploration
└── productivity/                  # Personal productivity systems
    ├── goal_tracking.py           # Goal setting and tracking
    ├── habit_formation.py         # Habit tracking and development
    ├── reflection_system.py       # Personal reflection and review
    └── life_logging.py            # Comprehensive life logging
```

## 🚀 Quick Start

### 1. Personal Setup

```bash
# Navigate to personal examples
cd examples/personal

# Install personal productivity dependencies
pip install -r requirements.txt

# Configure collections for personal workflow
export COLLECTIONS="notes,ideas,learning,projects"
export GLOBAL_COLLECTIONS="references,templates,inspiration,archive"
```

### 2. Initialize Personal Environment

```python
# personal_setup.py - Initialize personal knowledge system
from workspace_qdrant_mcp.client import WorkspaceClient
from datetime import datetime

def setup_personal_collections():
    """Initialize collections for personal knowledge management."""
    client = WorkspaceClient()
    
    # Create personal collections
    collections = {
        'notes': 'Daily notes, thoughts, and observations',
        'ideas': 'Creative ideas and project concepts',
        'learning': 'Learning progress, courses, and insights',
        'projects': 'Personal projects and goals',
        'references': 'Useful articles, links, and resources',
        'templates': 'Note templates and frameworks',
        'inspiration': 'Inspirational content and quotes',
        'archive': 'Archived content and completed projects',
        'journal': 'Personal journal and reflection entries',
        'goals': 'Personal and professional goals'
    }
    
    for collection, description in collections.items():
        print(f"📝 Creating collection: {collection}")
        # In real implementation: client.create_collection(collection, description)
    
    # Create initial welcome note
    welcome_content = f"""
# Welcome to Your Personal Knowledge System

Created on {datetime.now().strftime('%Y-%m-%d')}

## Collections Overview
{chr(10).join(f"- **{name}**: {desc}" for name, desc in collections.items())}

## Getting Started

### Daily Workflow
1. **Morning**: Review yesterday's notes and set today's intentions
2. **Throughout day**: Capture ideas and insights as they occur
3. **Evening**: Reflect on the day and plan tomorrow

### Weekly Review
- Review and connect related notes
- Archive completed projects
- Plan upcoming learning goals
- Update project status

### Monthly Reflection
- Analyze learning progress
- Review goal advancement
- Update knowledge connections
- Archive old content

## Quick Commands for Claude

**Capture**: "Store this thought in my ideas collection: [content]"
**Search**: "Find my notes about [topic]"
**Connect**: "Show me related notes about [subject]"
**Review**: "What did I learn about [topic] this month?"
    """
    
    client.store(
        content=welcome_content,
        metadata={
            'type': 'system_welcome',
            'created_date': datetime.now().isoformat(),
            'collections_count': len(collections)
        },
        collection='notes'
    )
    
    return client

if __name__ == "__main__":
    client = setup_personal_collections()
    print("🌟 Personal knowledge system ready!")
```

### 3. Claude Integration for Personal Use

In Claude Desktop or Claude Code, try these personal commands:

**Note-Taking:**
- "Store this insight in my notes: [your thought or observation]"
- "Find all my notes about productivity and focus techniques"
- "Show me what I wrote about this topic last month"

**Learning:**
- "Track my progress in learning Python: completed functions chapter"
- "Find all my notes about machine learning concepts"
- "What books am I currently reading and what are my key takeaways?"

**Ideas:**
- "Capture this project idea: [describe your idea]"
- "Find all my ideas related to mobile app development"
- "Show me ideas I haven't developed yet"

## 📚 Example Workflows

### Zettelkasten Note-Taking System

**Comprehensive Implementation of the Zettelkasten Method:**

```python
# zettelkasten.py - Digital Zettelkasten implementation
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ZettelNote:
    """Represents a single note in the Zettelkasten system."""
    id: str
    title: str
    content: str
    tags: Set[str] = field(default_factory=set)
    links: Set[str] = field(default_factory=set)  # Links to other notes
    backlinks: Set[str] = field(default_factory=set)  # Notes that link to this one
    created_date: datetime = field(default_factory=datetime.now)
    modified_date: datetime = field(default_factory=datetime.now)
    note_type: str = "permanent"  # permanent, literature, fleeting, project
    source: Optional[str] = None  # Source reference if applicable
    status: str = "active"  # active, archived, draft

class ZettelkastenSystem:
    """
    Digital implementation of the Zettelkasten note-taking method.
    
    The Zettelkasten (slip box) method creates a network of interconnected notes
    that build knowledge organically through connections and associations.
    """
    
    def __init__(self, client):
        self.client = client
        self.notes = {}  # In production, use proper database
        self.tag_index = {}  # Tag to note IDs mapping
        self.link_graph = {}  # Note relationship graph
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the Zettelkasten system with foundational concepts."""
        # Create index note
        index_note = self.create_note(
            title="Zettelkasten Index",
            content="""
# Zettelkasten Index

This is the main index for your personal Zettelkasten system.

## Core Principles

1. **Atomic Notes**: Each note contains one idea
2. **Linking**: Connect related concepts through links
3. **Tagging**: Use tags for topic categorization  
4. **Continuous Development**: Notes evolve over time

## Note Types

- **Permanent Notes**: Fully developed ideas and concepts
- **Literature Notes**: Summaries and insights from sources
- **Fleeting Notes**: Quick thoughts and observations
- **Project Notes**: Project-specific information

## Getting Started

1. Capture fleeting notes throughout the day
2. Develop fleeting notes into permanent notes
3. Link related concepts together
4. Review and expand connections regularly

## Navigation

Use search and tags to navigate your knowledge network.
The system grows more valuable as connections multiply.
            """,
            note_type="permanent",
            tags={"index", "system", "zettelkasten"}
        )
        
        print(f"📋 Zettelkasten system initialized with index note: {index_note}")
    
    def create_note(self, title: str, content: str, note_type: str = "permanent", 
                   tags: Set[str] = None, source: str = None) -> str:
        """
        Create a new Zettel note.
        
        Args:
            title: Note title
            content: Note content (can include [[wiki-style]] links)
            note_type: Type of note (permanent, literature, fleeting, project)
            tags: Set of tags for categorization
            source: Source reference if applicable
            
        Returns:
            Note ID
        """
        # Generate unique ID (timestamp-based for chronological ordering)
        note_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"
        
        # Extract wiki-style links from content
        links = self._extract_links(content)
        
        # Create note
        note = ZettelNote(
            id=note_id,
            title=title,
            content=content,
            tags=tags or set(),
            links=links,
            note_type=note_type,
            source=source
        )
        
        # Store note
        self.notes[note_id] = note
        
        # Update indexes
        self._update_tag_index(note)
        self._update_link_graph(note)
        
        # Store in Qdrant for search
        self._store_note_in_qdrant(note)
        
        print(f"📝 Created {note_type} note: {title}")
        return note_id
    
    def _extract_links(self, content: str) -> Set[str]:
        """Extract [[wiki-style]] links from note content."""
        link_pattern = r'\[\[([^\]]+)\]\]'
        matches = re.findall(link_pattern, content)
        return set(matches)
    
    def _update_tag_index(self, note: ZettelNote):
        """Update the tag index with the new note."""
        for tag in note.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(note.id)
    
    def _update_link_graph(self, note: ZettelNote):
        """Update the link graph with note connections."""
        # Add outgoing links
        self.link_graph[note.id] = note.links.copy()
        
        # Update backlinks for linked notes
        for linked_note_title in note.links:
            # Find note ID by title (simplified - in production, maintain title->ID mapping)
            linked_note_id = self._find_note_by_title(linked_note_title)
            if linked_note_id:
                if linked_note_id not in self.notes:
                    continue
                self.notes[linked_note_id].backlinks.add(note.id)
    
    def _find_note_by_title(self, title: str) -> Optional[str]:
        """Find note ID by title."""
        for note_id, note in self.notes.items():
            if note.title.lower() == title.lower():
                return note_id
        return None
    
    def _store_note_in_qdrant(self, note: ZettelNote):
        """Store note in Qdrant for semantic search."""
        # Create searchable content
        content = f"""
Title: {note.title}

{note.content}

Tags: {', '.join(note.tags)}

Type: {note.note_type}

Links: {', '.join(note.links)}

Source: {note.source or 'Personal'}
        """.strip()
        
        metadata = {
            'type': 'zettel_note',
            'note_id': note.id,
            'title': note.title,
            'note_type': note.note_type,
            'tags': list(note.tags),
            'created_date': note.created_date.isoformat(),
            'modified_date': note.modified_date.isoformat(),
            'links_count': len(note.links),
            'backlinks_count': len(note.backlinks),
            'status': note.status,
            'source': note.source
        }
        
        self.client.store(
            content=content,
            metadata=metadata,
            collection='notes'
        )
    
    def link_notes(self, source_note_id: str, target_note_title: str, 
                  bidirectional: bool = False) -> bool:
        """
        Create a link between two notes.
        
        Args:
            source_note_id: ID of the source note
            target_note_title: Title of the target note
            bidirectional: Whether to create a bidirectional link
            
        Returns:
            True if link was created successfully
        """
        if source_note_id not in self.notes:
            print(f"❌ Source note {source_note_id} not found")
            return False
        
        target_note_id = self._find_note_by_title(target_note_title)
        if not target_note_id:
            print(f"❌ Target note '{target_note_title}' not found")
            return False
        
        # Add link to source note
        source_note = self.notes[source_note_id]
        source_note.links.add(target_note_title)
        source_note.modified_date = datetime.now()
        
        # Add backlink to target note
        target_note = self.notes[target_note_id]
        target_note.backlinks.add(source_note_id)
        
        # Update link graph
        self._update_link_graph(source_note)
        
        # Create bidirectional link if requested
        if bidirectional:
            source_note.backlinks.add(target_note_id)
            target_note.links.add(source_note.title)
        
        # Update in Qdrant
        self._store_note_in_qdrant(source_note)
        self._store_note_in_qdrant(target_note)
        
        print(f"🔗 Linked '{source_note.title}' to '{target_note_title}'")
        return True
    
    def search_notes(self, query: str, note_type: str = None, 
                    tags: List[str] = None) -> List[Dict]:
        """
        Search notes using semantic search.
        
        Args:
            query: Search query
            note_type: Filter by note type
            tags: Filter by tags
            
        Returns:
            List of matching notes
        """
        filters = {'type': 'zettel_note'}
        
        if note_type:
            filters['note_type'] = note_type
        
        if tags:
            filters['tags'] = tags
        
        results = self.client.search(
            query=query,
            collection='notes',
            metadata_filter=filters,
            limit=20
        )
        
        return results
    
    def get_note_connections(self, note_id: str) -> Dict[str, List[str]]:
        """Get all connections (links and backlinks) for a note."""
        if note_id not in self.notes:
            return {}
        
        note = self.notes[note_id]
        
        return {
            'outgoing_links': list(note.links),
            'incoming_links': [self.notes[bid].title for bid in note.backlinks if bid in self.notes],
            'related_tags': list(note.tags)
        }
    
    def get_notes_by_tag(self, tag: str) -> List[ZettelNote]:
        """Get all notes with a specific tag."""
        if tag not in self.tag_index:
            return []
        
        return [self.notes[note_id] for note_id in self.tag_index[tag] 
                if note_id in self.notes]
    
    def develop_fleeting_note(self, fleeting_note_id: str, 
                            expanded_content: str, new_title: str = None) -> str:
        """
        Develop a fleeting note into a permanent note.
        
        Args:
            fleeting_note_id: ID of the fleeting note to develop
            expanded_content: Expanded and refined content
            new_title: New title (optional)
            
        Returns:
            ID of the new permanent note
        """
        if fleeting_note_id not in self.notes:
            print(f"❌ Fleeting note {fleeting_note_id} not found")
            return None
        
        fleeting_note = self.notes[fleeting_note_id]
        
        # Create permanent note
        permanent_note_id = self.create_note(
            title=new_title or fleeting_note.title,
            content=expanded_content,
            note_type="permanent",
            tags=fleeting_note.tags,
            source=fleeting_note.source
        )
        
        # Archive the fleeting note
        fleeting_note.status = "archived"
        fleeting_note.modified_date = datetime.now()
        
        # Add link from permanent to fleeting note for history
        self.link_notes(permanent_note_id, fleeting_note.title)
        
        print(f"🌱 Developed fleeting note into permanent note: {permanent_note_id}")
        return permanent_note_id
    
    def generate_note_network_map(self) -> str:
        """Generate a visualization of the note network."""
        network_analysis = {
            'total_notes': len(self.notes),
            'note_types': {},
            'most_connected': [],
            'orphaned_notes': [],
            'tag_distribution': {}
        }
        
        # Analyze note types
        for note in self.notes.values():
            note_type = note.note_type
            network_analysis['note_types'][note_type] = network_analysis['note_types'].get(note_type, 0) + 1
        
        # Find most connected notes
        connection_counts = []
        for note in self.notes.values():
            total_connections = len(note.links) + len(note.backlinks)
            connection_counts.append((note.title, total_connections))
        
        network_analysis['most_connected'] = sorted(connection_counts, 
                                                  key=lambda x: x[1], reverse=True)[:10]
        
        # Find orphaned notes (no connections)
        for note in self.notes.values():
            if len(note.links) == 0 and len(note.backlinks) == 0:
                network_analysis['orphaned_notes'].append(note.title)
        
        # Analyze tag distribution
        for note in self.notes.values():
            for tag in note.tags:
                network_analysis['tag_distribution'][tag] = network_analysis['tag_distribution'].get(tag, 0) + 1
        
        # Generate report
        report = f"""
# Zettelkasten Network Analysis

## Overview
- **Total Notes**: {network_analysis['total_notes']}
- **Note Types**: {', '.join(f"{type}: {count}" for type, count in network_analysis['note_types'].items())}

## Most Connected Notes
{chr(10).join(f"{i+1}. {title} ({connections} connections)" for i, (title, connections) in enumerate(network_analysis['most_connected']))}

## Orphaned Notes
{chr(10).join(f"- {title}" for title in network_analysis['orphaned_notes'][:10])}
{f"... and {len(network_analysis['orphaned_notes']) - 10} more" if len(network_analysis['orphaned_notes']) > 10 else ""}

## Popular Tags
{chr(10).join(f"- {tag}: {count} notes" for tag, count in sorted(network_analysis['tag_distribution'].items(), key=lambda x: x[1], reverse=True)[:10])}

## Recommendations
- Consider connecting orphaned notes to the network
- Develop fleeting notes into permanent notes
- Review and expand highly connected notes
- Create structure notes for popular topics
        """
        
        return report.strip()
    
    def daily_review(self) -> str:
        """Generate a daily review of recent activity."""
        today = datetime.now().date()
        
        # Get today's notes
        todays_notes = [note for note in self.notes.values() 
                       if note.created_date.date() == today]
        
        # Get fleeting notes to develop
        fleeting_notes = [note for note in self.notes.values() 
                         if note.note_type == "fleeting" and note.status == "active"]
        
        review = f"""
# Daily Zettelkasten Review - {today.strftime('%Y-%m-%d')}

## Today's Activity
- **Notes Created**: {len(todays_notes)}
- **Note Types**: {', '.join(set(note.note_type for note in todays_notes))}

## Fleeting Notes to Develop
{chr(10).join(f"- {note.title} (Created: {note.created_date.strftime('%m-%d')})" for note in fleeting_notes[:5])}

## Recent Notes
{chr(10).join(f"- **{note.title}** ({note.note_type})" for note in todays_notes)}

## Tomorrow's Focus
- Review and connect today's notes
- Develop fleeting notes into permanent notes
- Look for emerging patterns and themes
        """
        
        return review.strip()

# Usage example
if __name__ == "__main__":
    from workspace_qdrant_mcp.client import WorkspaceClient
    
    client = WorkspaceClient()
    zk = ZettelkastenSystem(client)
    
    # Create some example notes
    productivity_note = zk.create_note(
        title="Productivity Systems",
        content="""
        # Productivity Systems
        
        Effective productivity systems help manage tasks, time, and attention.
        
        ## Key Principles
        - **Capture Everything**: Don't rely on memory
        - **Process Regularly**: Review and organize inputs
        - **Single Source**: Avoid multiple systems
        
        ## Related Concepts
        - [[Getting Things Done]]
        - [[Time Management]]
        - [[Focus Techniques]]
        
        ## Implementation
        The system must be simple enough to maintain consistently.
        """,
        note_type="permanent",
        tags={"productivity", "systems", "organization"}
    )
    
    gtd_note = zk.create_note(
        title="Getting Things Done",
        content="""
        # Getting Things Done (GTD)
        
        GTD is a productivity methodology by David Allen.
        
        ## Five Steps
        1. Capture
        2. Clarify  
        3. Organize
        4. Reflect
        5. Engage
        
        Links to [[Productivity Systems]] as a comprehensive approach.
        """,
        note_type="literature",
        tags={"productivity", "gtd", "methodology"},
        source="David Allen - Getting Things Done"
    )
    
    # Create fleeting note
    idea_note = zk.create_note(
        title="App Idea: Digital Zettelkasten",
        content="Create a mobile app for Zettelkasten note-taking with voice input and AI-powered linking suggestions.",
        note_type="fleeting",
        tags={"ideas", "apps", "zettelkasten"}
    )
    
    print(f"\n📊 Network Analysis:")
    print(zk.generate_note_network_map())
    
    print(f"\n📅 Daily Review:")
    print(zk.daily_review())
```

### Learning Tracker System

**Comprehensive Learning Progress Management:**

```python
# learning_tracker.py - Personal learning management system
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

class LearningType(Enum):
    """Types of learning activities."""
    BOOK = "book"
    COURSE = "course"
    ARTICLE = "article"
    VIDEO = "video"
    PODCAST = "podcast"
    PRACTICE = "practice"
    PROJECT = "project"
    WORKSHOP = "workshop"

class LearningStatus(Enum):
    """Status of learning activities."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

@dataclass
class LearningResource:
    """Represents a learning resource or activity."""
    id: str
    title: str
    learning_type: LearningType
    author_instructor: str
    description: str
    status: LearningStatus = LearningStatus.PLANNED
    start_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    progress_percentage: float = 0.0
    time_invested: int = 0  # minutes
    rating: Optional[float] = None  # 1-5 stars
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    key_insights: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    related_resources: List[str] = field(default_factory=list)
    difficulty_level: str = "intermediate"  # beginner, intermediate, advanced

class LearningTracker:
    """
    Comprehensive personal learning management system.
    
    Tracks learning progress, captures insights, and helps identify
    knowledge gaps and learning patterns.
    """
    
    def __init__(self, client):
        self.client = client
        self.resources = {}  # In production, use proper database
        self.learning_goals = {}
        self.skills_inventory = {}
        
        # Initialize learning dashboard
        self._initialize_dashboard()
    
    def _initialize_dashboard(self):
        """Initialize the learning dashboard."""
        dashboard_content = f"""
# Personal Learning Dashboard

Welcome to your comprehensive learning tracking system!

## Learning Philosophy
- **Continuous Learning**: Embrace lifelong learning across all areas
- **Deliberate Practice**: Focus on challenging, progressive skill development  
- **Reflection**: Regular review and synthesis of learnings
- **Application**: Connect learning to real-world practice

## Learning Categories
- **Technical Skills**: Programming, tools, methodologies
- **Professional Development**: Leadership, communication, business
- **Personal Interest**: Hobbies, creative pursuits, general knowledge
- **Health & Wellness**: Physical and mental well-being

## Dashboard Overview
Created: {datetime.now().strftime('%Y-%m-%d')}

### Quick Stats
- Active Learning Resources: 0
- Completed This Month: 0
- Total Learning Hours: 0
- Current Focus Areas: To be defined

## Getting Started
1. Add learning resources using `add_learning_resource()`
2. Update progress regularly with `update_progress()`
3. Capture insights with `add_insight()`
4. Review weekly with `generate_learning_report()`
        """
        
        self.client.store(
            content=dashboard_content,
            metadata={
                'type': 'learning_dashboard',
                'created_date': datetime.now().isoformat()
            },
            collection='learning'
        )
    
    def add_learning_resource(self, resource: LearningResource) -> str:
        """Add a new learning resource to track."""
        # Store in local database
        self.resources[resource.id] = resource
        
        # Create searchable content
        content = f"""
# {resource.title}

**Type:** {resource.learning_type.value.title()}
**Instructor/Author:** {resource.author_instructor}
**Status:** {resource.status.value.title()}
**Difficulty:** {resource.difficulty_level.title()}

## Description
{resource.description}

## Progress
- **Completion:** {resource.progress_percentage}%
- **Time Invested:** {resource.time_invested} minutes
- **Started:** {resource.start_date.strftime('%Y-%m-%d') if resource.start_date else 'Not started'}
- **Completed:** {resource.completion_date.strftime('%Y-%m-%d') if resource.completion_date else 'In progress'}

## Tags
{', '.join(resource.tags)}

## Key Insights
{chr(10).join(f"- {insight}" for insight in resource.key_insights)}

## Action Items
{chr(10).join(f"- {item}" for item in resource.action_items)}

## Notes
{resource.notes}

## Rating
{f"{resource.rating}/5 stars" if resource.rating else "Not rated yet"}
        """
        
        metadata = {
            'type': 'learning_resource',
            'resource_id': resource.id,
            'title': resource.title,
            'learning_type': resource.learning_type.value,
            'status': resource.status.value,
            'author_instructor': resource.author_instructor,
            'progress_percentage': resource.progress_percentage,
            'time_invested': resource.time_invested,
            'difficulty_level': resource.difficulty_level,
            'tags': resource.tags,
            'start_date': resource.start_date.isoformat() if resource.start_date else None,
            'completion_date': resource.completion_date.isoformat() if resource.completion_date else None,
            'rating': resource.rating
        }
        
        self.client.store(
            content=content,
            metadata=metadata,
            collection='learning'
        )
        
        print(f"📚 Added learning resource: {resource.title}")
        return resource.id
    
    def update_progress(self, resource_id: str, progress_percentage: float, 
                       time_spent: int = 0, notes: str = "") -> bool:
        """Update progress on a learning resource."""
        if resource_id not in self.resources:
            print(f"❌ Resource {resource_id} not found")
            return False
        
        resource = self.resources[resource_id]
        
        # Update progress
        old_progress = resource.progress_percentage
        resource.progress_percentage = min(progress_percentage, 100.0)
        resource.time_invested += time_spent
        
        if notes:
            resource.notes += f"\n\n**{datetime.now().strftime('%Y-%m-%d')}**: {notes}"
        
        # Update status based on progress
        if resource.progress_percentage >= 100.0 and resource.status != LearningStatus.COMPLETED:
            resource.status = LearningStatus.COMPLETED
            resource.completion_date = datetime.now()
        elif resource.progress_percentage > 0 and resource.status == LearningStatus.PLANNED:
            resource.status = LearningStatus.IN_PROGRESS
            resource.start_date = datetime.now()
        
        # Store updated resource
        self._store_resource_in_qdrant(resource)
        
        # Create progress log entry
        progress_log = f"""
# Progress Update: {resource.title}

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Progress:** {old_progress}% → {resource.progress_percentage}%
**Time Spent Today:** {time_spent} minutes
**Total Time:** {resource.time_invested} minutes
**Status:** {resource.status.value.title()}

## Notes
{notes}
        """
        
        self.client.store(
            content=progress_log,
            metadata={
                'type': 'progress_update',
                'resource_id': resource_id,
                'date': datetime.now().isoformat(),
                'progress_percentage': resource.progress_percentage,
                'time_spent': time_spent
            },
            collection='learning'
        )
        
        print(f"📈 Updated progress for {resource.title}: {resource.progress_percentage}%")
        return True
    
    def add_insight(self, resource_id: str, insight: str, category: str = "general") -> bool:
        """Add a key insight from learning."""
        if resource_id not in self.resources:
            print(f"❌ Resource {resource_id} not found")
            return False
        
        resource = self.resources[resource_id]
        resource.key_insights.append(insight)
        
        # Create insight entry
        insight_content = f"""
# Learning Insight: {resource.title}

**Category:** {category}
**Date:** {datetime.now().strftime('%Y-%m-%d')}

## Insight
{insight}

## Context
From learning resource: {resource.title} by {resource.author_instructor}

## Application
How can this insight be applied in practice?

## Connections
How does this relate to other knowledge or experiences?
        """
        
        self.client.store(
            content=insight_content,
            metadata={
                'type': 'learning_insight',
                'resource_id': resource_id,
                'category': category,
                'date': datetime.now().isoformat(),
                'resource_title': resource.title
            },
            collection='learning'
        )
        
        # Update resource in storage
        self._store_resource_in_qdrant(resource)
        
        print(f"💡 Added insight for {resource.title}")
        return True
    
    def _store_resource_in_qdrant(self, resource: LearningResource):
        """Store or update resource in Qdrant."""
        content = f"""
# {resource.title}

**Type:** {resource.learning_type.value.title()}
**Instructor/Author:** {resource.author_instructor}
**Status:** {resource.status.value.title()}

## Description
{resource.description}

## Key Insights
{chr(10).join(f"- {insight}" for insight in resource.key_insights)}

## Notes
{resource.notes}
        """
        
        metadata = {
            'type': 'learning_resource',
            'resource_id': resource.id,
            'title': resource.title,
            'learning_type': resource.learning_type.value,
            'status': resource.status.value,
            'progress_percentage': resource.progress_percentage,
            'time_invested': resource.time_invested,
            'tags': resource.tags
        }
        
        self.client.store(
            content=content,
            metadata=metadata,
            collection='learning'
        )
    
    def search_learning_content(self, query: str, learning_type: str = None, 
                              status: str = None) -> List[Dict]:
        """Search learning content and insights."""
        filters = {}
        
        if learning_type:
            filters['learning_type'] = learning_type
        if status:
            filters['status'] = status
        
        results = self.client.search(
            query=query,
            collection='learning',
            metadata_filter=filters,
            limit=20
        )
        
        return results
    
    def generate_learning_report(self, period: str = "week") -> str:
        """Generate learning progress report."""
        # Calculate date range
        if period == "week":
            start_date = datetime.now() - timedelta(days=7)
        elif period == "month":
            start_date = datetime.now() - timedelta(days=30)
        else:
            start_date = datetime.now() - timedelta(days=365)
        
        # Analyze progress
        active_resources = [r for r in self.resources.values() 
                          if r.status == LearningStatus.IN_PROGRESS]
        completed_resources = [r for r in self.resources.values() 
                             if r.status == LearningStatus.COMPLETED and 
                             r.completion_date and r.completion_date >= start_date]
        
        total_time = sum(r.time_invested for r in self.resources.values())
        period_time = sum(r.time_invested for r in completed_resources)
        
        # Generate report
        report = f"""
# Learning Report - {period.title()} Ending {datetime.now().strftime('%Y-%m-%d')}

## Summary
- **Active Learning Resources**: {len(active_resources)}
- **Completed This {period.title()}**: {len(completed_resources)}
- **Total Learning Time**: {total_time // 60} hours {total_time % 60} minutes
- **This {period.title()}'s Learning Time**: {period_time // 60} hours {period_time % 60} minutes

## Active Learning
{chr(10).join(f"- **{r.title}** ({r.progress_percentage:.1f}% complete)" for r in active_resources)}

## Recent Completions
{chr(10).join(f"- **{r.title}** - {r.rating}/5 stars" if r.rating else f"- **{r.title}** - Not rated" for r in completed_resources)}

## Learning Insights This {period.title()}
{self._get_recent_insights(start_date)}

## Recommendations
- Continue with active learning resources
- Review and rate completed resources
- Plan next learning goals based on interests and gaps
- Schedule regular learning time blocks

## Next {period.title()}'s Goals
- Complete in-progress resources
- Add new learning resources
- Practice and apply recent insights
        """
        
        return report.strip()
    
    def _get_recent_insights(self, start_date: datetime) -> str:
        """Get recent learning insights."""
        # Search for recent insights
        insights = self.client.search(
            query="learning insight",
            collection='learning',
            metadata_filter={'type': 'learning_insight'},
            limit=10
        )
        
        # Filter by date and format
        recent_insights = []
        for insight in insights:
            metadata = insight.get('metadata', {})
            insight_date_str = metadata.get('date')
            if insight_date_str:
                insight_date = datetime.fromisoformat(insight_date_str)
                if insight_date >= start_date:
                    content = insight.get('content', '')
                    # Extract the insight text (simplified)
                    lines = content.split('\n')
                    insight_text = ""
                    for i, line in enumerate(lines):
                        if line.strip() == "## Insight" and i+1 < len(lines):
                            insight_text = lines[i+1].strip()
                            break
                    
                    if insight_text:
                        recent_insights.append(f"- {insight_text}")
        
        return '\n'.join(recent_insights[:5]) if recent_insights else "- No insights captured this period"
    
    def set_learning_goal(self, goal: str, target_date: datetime, 
                         related_resources: List[str] = None) -> str:
        """Set a learning goal with timeline."""
        goal_id = f"goal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        goal_content = f"""
# Learning Goal: {goal}

**Target Date:** {target_date.strftime('%Y-%m-%d')}
**Created:** {datetime.now().strftime('%Y-%m-%d')}

## Related Resources
{chr(10).join(f"- {resource}" for resource in related_resources) if related_resources else "To be identified"}

## Success Criteria
- [ ] Define specific success criteria
- [ ] Identify learning resources
- [ ] Create learning plan
- [ ] Track progress regularly

## Progress Notes
Goal created. Next: Define specific success criteria and learning plan.
        """
        
        self.client.store(
            content=goal_content,
            metadata={
                'type': 'learning_goal',
                'goal_id': goal_id,
                'goal': goal,
                'target_date': target_date.isoformat(),
                'created_date': datetime.now().isoformat(),
                'status': 'active'
            },
            collection='learning'
        )
        
        print(f"🎯 Set learning goal: {goal}")
        return goal_id

# Usage example
if __name__ == "__main__":
    from workspace_qdrant_mcp.client import WorkspaceClient
    
    client = WorkspaceClient()
    lt = LearningTracker(client)
    
    # Add sample learning resources
    python_course = LearningResource(
        id="python_advanced_2024",
        title="Advanced Python Programming",
        learning_type=LearningType.COURSE,
        author_instructor="Jane Smith",
        description="Comprehensive course covering advanced Python concepts including decorators, metaclasses, asyncio, and performance optimization.",
        tags=["python", "programming", "advanced", "software-development"],
        difficulty_level="advanced"
    )
    
    resource_id = lt.add_learning_resource(python_course)
    
    # Update progress
    lt.update_progress(resource_id, 25.0, 120, "Completed modules on decorators and context managers. Great examples and exercises.")
    
    # Add insight
    lt.add_insight(resource_id, "Context managers are powerful for resource management. The __enter__ and __exit__ methods provide clean setup and teardown.", "programming")
    
    # Set learning goal
    lt.set_learning_goal("Master advanced Python concepts for backend development", datetime.now() + timedelta(days=90), [python_course.title])
    
    # Generate report
    report = lt.generate_learning_report("week")
    print(report)
```

## 💡 Claude Interaction Prompts

### Personal Knowledge Management Prompts

**Note-Taking and Ideas:**
```
Help me with personal note-taking:
- Store this thought in my ideas collection: [your idea or observation]
- Find all my notes about [topic] from the last month
- Show me related notes and concepts about [subject]
- Connect this new idea to my existing notes about [topic]
- What patterns do you see in my recent notes?
```

**Learning and Growth:**
```
Track my learning progress:
- Update my progress on [course/book]: [progress details]
- Add this insight from my learning: [key insight or takeaway]
- Find all my notes about [skill or topic] I'm learning
- What learning resources have I completed this month?
- Help me plan my next learning goals based on my interests
```

**Personal Reflection:**
```
Support my personal reflection:
- Store this journal entry about [topic or experience]
- What themes appear across my recent reflections?
- Find my previous thoughts about [life area or challenge]
- Help me track progress on my personal goals
- Show me insights from my past experiences with [situation]
```

### Advanced Personal Workflows

**Knowledge Synthesis:**
```python
# Use in Claude Code for connecting personal knowledge
"""
I want to synthesize my personal knowledge about [topic]. Please:

1. Search all my notes and ideas related to this topic
2. Find connections between different concepts and insights
3. Identify gaps in my understanding or experience
4. Suggest areas for further exploration or learning
5. Help me create a comprehensive overview combining my insights

Focus on personal insights, experiences, and learning over time.
"""
```

## 📊 Best Practices

### Collection Organization

**Recommended collection structure for personal use:**

```bash
# Personal collections
export COLLECTIONS="notes,ideas,learning,journal"

# Global personal collections
export GLOBAL_COLLECTIONS="references,templates,inspiration,goals"

# Example result:
# mylife-notes              # Daily notes and observations
# mylife-ideas              # Creative ideas and projects
# mylife-learning           # Learning progress and insights
# mylife-journal            # Personal reflection and growth
# references                # Useful articles and resources
# templates                 # Note templates and frameworks
# inspiration               # Inspirational content and quotes
# goals                     # Personal and professional goals
```

### Daily Personal Workflow

**Establish consistent daily practices:**

```bash
# Morning routine
python daily_note.py --create-today
python goal_tracker.py --review-daily-goals

# Evening routine
python reflection_system.py --daily-review
python learning_tracker.py --log-progress
```

### Personal Knowledge Development

**Build knowledge systematically:**

1. **Capture immediately** - Don't lose fleeting thoughts
2. **Process regularly** - Develop raw notes into insights
3. **Connect actively** - Link related concepts and ideas
4. **Review periodically** - Weekly and monthly reviews
5. **Apply continuously** - Use knowledge in real situations

## 🔗 Integration Examples

- **[VS Code Integration](../integrations/vscode/README.md)** - Personal workspace setup
- **[Automation Scripts](../integrations/automation/README.md)** - Automated capture and organization
- **[Performance Optimization](../performance_optimization/README.md)** - Large personal knowledge bases

---

**Next Steps:**
1. Try the [Zettelkasten System](note_taking/)
2. Set up [Learning Tracking](learning/)
3. Explore [Idea Management](idea_management/)