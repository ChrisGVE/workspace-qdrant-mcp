# Research Examples

Practical examples for using workspace-qdrant-mcp in academic research workflows, including paper management, citation tracking, and literature reviews.

## 🎯 Overview

This section demonstrates how workspace-qdrant-mcp enhances academic research workflows by:

- **Paper Management** - Organize and search through academic papers and publications
- **Citation Tracking** - Track citations, references, and build citation networks
- **Literature Reviews** - Systematic literature review support and analysis
- **Research Notes** - Capture and organize research insights and findings
- **Collaboration** - Share research knowledge with team members
- **Grant Writing** - Support for grant proposals and funding applications

## 🏗️ Examples Structure

```
research/
├── README.md                       # This file
├── paper_management/               # Academic paper workflows
│   ├── paper_ingestion.py         # Import papers and metadata
│   ├── citation_extractor.py      # Extract citations and references
│   ├── arxiv_integration.py       # ArXiv API integration
│   └── sample_papers/             # Sample research papers
├── literature_review/              # Systematic literature review
│   ├── systematic_review.py       # Structured review process
│   ├── review_templates/          # Review templates and checklists
│   ├── search_strategies.py       # Literature search strategies
│   └── analysis_tools.py          # Analysis and synthesis tools
├── research_notes/                 # Research note-taking
│   ├── lab_notebook.py            # Digital lab notebook
│   ├── meeting_notes.py           # Research meeting notes
│   ├── hypothesis_tracking.py     # Hypothesis management
│   └── note_templates/            # Note templates
├── collaboration/                  # Research collaboration
│   ├── shared_knowledge.py        # Shared research knowledge base
│   ├── annotation_system.py       # Collaborative annotations
│   └── review_workflows.py        # Peer review workflows
└── grant_writing/                  # Grant application support
    ├── proposal_tracker.py        # Track proposals and deadlines
    ├── literature_synthesis.py    # Synthesize literature for grants
    └── impact_analysis.py         # Research impact analysis
```

## 🚀 Quick Start

### 1. Research Setup

```bash
# Navigate to research examples
cd examples/research

# Install research-specific dependencies
pip install -r requirements.txt

# Configure collections for research workflow
export COLLECTIONS="papers,notes,reviews,grants"
export GLOBAL_COLLECTIONS="citations,methodologies,datasets"
```

### 2. Initialize Research Environment

```python
# research_setup.py - Initialize research collections
from workspace_qdrant_mcp.client import WorkspaceClient

def setup_research_collections():
    """Initialize collections for academic research workflow."""
    client = WorkspaceClient()
    
    # Create research-specific collections
    collections = {
        'papers': 'Academic papers and publications',
        'citations': 'Citation network and references',
        'notes': 'Research notes and insights',
        'reviews': 'Literature reviews and analyses',
        'methodologies': 'Research methodologies and protocols',
        'datasets': 'Research datasets and data sources',
        'grants': 'Grant proposals and funding information'
    }
    
    for collection, description in collections.items():
        client.create_collection(collection, description)
        print(f"✅ Created collection: {collection}")
    
    return client

if __name__ == "__main__":
    client = setup_research_collections()
    print("🔬 Research environment ready!")
```

### 3. Claude Integration for Research

In Claude Desktop or Claude Code, try these research commands:

**Paper Management:**
- "Search my papers for machine learning applications in healthcare"
- "Find all papers by [Author Name] in my collection"
- "Show me papers related to natural language processing from 2023"

**Literature Review:**
- "What methodologies have been used in sentiment analysis research?"
- "Find gaps in the current literature on reinforcement learning"
- "Summarize the main findings from papers about transformers"

**Research Notes:**
- "Search my research notes for hypothesis about attention mechanisms"
- "Find all meeting notes discussing the experimental design"
- "Show me insights about data preprocessing techniques"

## 📚 Example Workflows

### Academic Paper Management

**Automated Paper Ingestion:**

```python
# paper_management.py - Comprehensive paper management system
import os
import json
import requests
import arxiv
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AcademicPaper:
    """Represents an academic paper with comprehensive metadata."""
    title: str
    authors: List[str]
    abstract: str
    publication_date: datetime
    venue: str  # Journal, conference, etc.
    paper_type: str  # journal, conference, preprint, thesis
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    local_path: Optional[str] = None
    keywords: List[str] = None
    citations: List[str] = None
    cited_by: List[str] = None
    methodology: str = None
    key_findings: str = None
    relevance_score: float = 0.0
    notes: str = None

class PaperManager:
    """
    Comprehensive academic paper management system.
    
    Provides tools for importing, organizing, and searching academic papers
    from various sources including arXiv, PubMed, and manual uploads.
    """
    
    def __init__(self, client, storage_path: str = "./papers"):
        self.client = client
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize arXiv client
        self.arxiv_client = arxiv.Client()
    
    def import_from_arxiv(self, query: str, max_results: int = 50) -> List[AcademicPaper]:
        """
        Import papers from arXiv based on search query.
        
        Args:
            query: Search query for arXiv (e.g., "cat:cs.AI AND ti:transformer")
            max_results: Maximum number of papers to retrieve
            
        Returns:
            List of imported AcademicPaper objects
        """
        print(f"🔍 Searching arXiv for: {query}")
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in self.arxiv_client.results(search):
            paper = AcademicPaper(
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                publication_date=result.published,
                venue="arXiv",
                paper_type="preprint",
                arxiv_id=result.entry_id.split('/')[-1],
                url=result.entry_id,
                keywords=result.categories
            )
            
            # Download PDF if available
            try:
                pdf_path = self.storage_path / f"{paper.arxiv_id.replace('/', '_')}.pdf"
                result.download_pdf(str(pdf_path))
                paper.local_path = str(pdf_path)
                print(f"📄 Downloaded: {paper.title[:60]}...")
            except Exception as e:
                print(f"⚠️  Could not download PDF for {paper.title[:60]}...: {e}")
            
            papers.append(paper)
            
            # Store in Qdrant for searchability
            self.store_paper(paper)
        
        print(f"✅ Imported {len(papers)} papers from arXiv")
        return papers
    
    def import_from_bibtex(self, bibtex_file: str) -> List[AcademicPaper]:
        """
        Import papers from BibTeX file.
        
        Args:
            bibtex_file: Path to BibTeX file
            
        Returns:
            List of imported papers
        """
        try:
            import bibtexparser
        except ImportError:
            raise ImportError("bibtexparser required: pip install bibtexparser")
        
        print(f"📚 Importing from BibTeX: {bibtex_file}")
        
        with open(bibtex_file) as bib_file:
            bib_database = bibtexparser.load(bib_file)
        
        papers = []
        for entry in bib_database.entries:
            # Extract publication date
            pub_date = None
            if 'year' in entry:
                try:
                    pub_date = datetime(int(entry['year']), 1, 1)
                except ValueError:
                    pub_date = datetime.now()
            else:
                pub_date = datetime.now()
            
            paper = AcademicPaper(
                title=entry.get('title', 'Unknown Title'),
                authors=self._parse_bibtex_authors(entry.get('author', '')),
                abstract=entry.get('abstract', ''),
                publication_date=pub_date,
                venue=entry.get('journal', entry.get('booktitle', 'Unknown Venue')),
                paper_type=entry.get('ENTRYTYPE', 'unknown'),
                doi=entry.get('doi'),
                url=entry.get('url'),
                keywords=entry.get('keywords', '').split(',') if entry.get('keywords') else []
            )
            
            papers.append(paper)
            self.store_paper(paper)
        
        print(f"✅ Imported {len(papers)} papers from BibTeX")
        return papers
    
    def store_paper(self, paper: AcademicPaper):
        """Store paper in Qdrant with comprehensive metadata."""
        # Create searchable content
        content = f"""
Title: {paper.title}

Authors: {', '.join(paper.authors)}

Abstract: {paper.abstract}

Venue: {paper.venue} ({paper.publication_date.year})

Keywords: {', '.join(paper.keywords or [])}

Key Findings: {paper.key_findings or 'Not analyzed'}

Notes: {paper.notes or 'No notes'}
        """.strip()
        
        metadata = {
            'type': 'academic_paper',
            'title': paper.title,
            'authors': paper.authors,
            'publication_year': paper.publication_date.year,
            'venue': paper.venue,
            'paper_type': paper.paper_type,
            'doi': paper.doi,
            'arxiv_id': paper.arxiv_id,
            'url': paper.url,
            'local_path': paper.local_path,
            'keywords': paper.keywords or [],
            'relevance_score': paper.relevance_score,
            'has_pdf': paper.local_path is not None
        }
        
        self.client.store(
            content=content,
            metadata=metadata,
            collection='papers'
        )
    
    def search_papers(self, query: str, filters: Dict[str, Any] = None) -> List[Dict]:
        """
        Search papers with advanced filtering.
        
        Args:
            query: Search query
            filters: Metadata filters (e.g., {'publication_year': 2023})
            
        Returns:
            List of matching papers
        """
        search_metadata = {'type': 'academic_paper'}
        if filters:
            search_metadata.update(filters)
        
        results = self.client.search(
            query=query,
            collection='papers',
            metadata_filter=search_metadata,
            limit=20
        )
        
        return results
    
    def analyze_paper_trends(self) -> Dict[str, Any]:
        """Analyze trends in the paper collection."""
        # Get all papers
        all_papers = self.client.search(
            query="*",
            collection='papers',
            metadata_filter={'type': 'academic_paper'},
            limit=1000
        )
        
        # Analyze trends
        year_counts = {}
        venue_counts = {}
        author_counts = {}
        keyword_counts = {}
        
        for paper_result in all_papers:
            metadata = paper_result.get('metadata', {})
            
            # Year analysis
            year = metadata.get('publication_year')
            if year:
                year_counts[year] = year_counts.get(year, 0) + 1
            
            # Venue analysis
            venue = metadata.get('venue')
            if venue:
                venue_counts[venue] = venue_counts.get(venue, 0) + 1
            
            # Author analysis
            authors = metadata.get('authors', [])
            for author in authors:
                author_counts[author] = author_counts.get(author, 0) + 1
            
            # Keyword analysis
            keywords = metadata.get('keywords', [])
            for keyword in keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        return {
            'total_papers': len(all_papers),
            'year_distribution': dict(sorted(year_counts.items(), reverse=True)[:10]),
            'top_venues': dict(sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_authors': dict(sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_keywords': dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        }
    
    def _parse_bibtex_authors(self, author_string: str) -> List[str]:
        """Parse BibTeX author string into list of author names."""
        if not author_string:
            return []
        
        # Simple parsing - in production, use more sophisticated parsing
        authors = [name.strip() for name in author_string.split(' and ')]
        return authors

# Usage example
if __name__ == "__main__":
    from workspace_qdrant_mcp.client import WorkspaceClient
    
    client = WorkspaceClient()
    pm = PaperManager(client)
    
    # Import recent NLP papers from arXiv
    papers = pm.import_from_arxiv("cat:cs.CL AND ti:transformer", max_results=10)
    
    # Search for specific topics
    results = pm.search_papers("attention mechanism in neural networks")
    print(f"Found {len(results)} papers about attention mechanisms")
    
    # Analyze collection trends
    trends = pm.analyze_paper_trends()
    print(f"Collection contains {trends['total_papers']} papers")
    print(f"Top keywords: {list(trends['top_keywords'].keys())[:5]}")
```

### Systematic Literature Review

**Structured Literature Review Process:**

```python
# literature_review.py - Systematic literature review support
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class ReviewStage(Enum):
    """Stages of systematic literature review process."""
    PLANNING = "planning"
    SEARCH = "search"
    SCREENING = "screening"
    EXTRACTION = "extraction"
    SYNTHESIS = "synthesis"
    REPORTING = "reporting"

@dataclass
class SearchStrategy:
    """Literature search strategy for systematic review."""
    databases: List[str]
    search_terms: List[str]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    date_range: tuple
    language: str = "English"

@dataclass
class ReviewPaper:
    """Paper in systematic review with review-specific metadata."""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    venue: str
    year: int
    
    # Review-specific fields
    included: Optional[bool] = None
    exclusion_reason: Optional[str] = None
    quality_score: Optional[float] = None
    data_extracted: bool = False
    reviewer_notes: str = ""
    extraction_data: Dict[str, Any] = field(default_factory=dict)

class LiteratureReview:
    """
    Systematic literature review management system.
    
    Supports PRISMA-compliant systematic literature reviews with
    structured search, screening, and data extraction workflows.
    """
    
    def __init__(self, client, review_title: str):
        self.client = client
        self.review_title = review_title
        self.review_id = review_title.lower().replace(' ', '_')
        self.current_stage = ReviewStage.PLANNING
        
        # Initialize review in knowledge base
        self._initialize_review()
    
    def _initialize_review(self):
        """Initialize the systematic review in the knowledge base."""
        review_metadata = {
            'type': 'systematic_review',
            'review_id': self.review_id,
            'title': self.review_title,
            'stage': self.current_stage.value,
            'created_date': datetime.now().isoformat()
        }
        
        content = f"""
# Systematic Literature Review: {self.review_title}

## Review Protocol

**Objective:** To be defined in planning stage

**Research Questions:** To be defined in planning stage

**Search Strategy:** To be developed

**Inclusion/Exclusion Criteria:** To be defined

## Progress Tracking

**Current Stage:** {self.current_stage.value}
**Created:** {datetime.now().strftime('%Y-%m-%d')}

## Notes

Initial review setup completed.
        """
        
        self.client.store(
            content=content,
            metadata=review_metadata,
            collection='reviews'
        )
    
    def define_search_strategy(self, strategy: SearchStrategy) -> str:
        """Define and document the search strategy."""
        self.current_stage = ReviewStage.SEARCH
        
        content = f"""
# Search Strategy: {self.review_title}

## Databases
{chr(10).join(f"- {db}" for db in strategy.databases)}

## Search Terms
{chr(10).join(f"- {term}" for term in strategy.search_terms)}

## Inclusion Criteria
{chr(10).join(f"- {criteria}" for criteria in strategy.inclusion_criteria)}

## Exclusion Criteria
{chr(10).join(f"- {criteria}" for criteria in strategy.exclusion_criteria)}

## Date Range
From: {strategy.date_range[0]}
To: {strategy.date_range[1]}

## Language
{strategy.language}

## Search Documentation
This search strategy follows PRISMA guidelines for systematic literature reviews.
Search terms were developed through preliminary scoping and expert consultation.
        """
        
        metadata = {
            'type': 'search_strategy',
            'review_id': self.review_id,
            'databases': strategy.databases,
            'search_terms': strategy.search_terms,
            'date_range': list(strategy.date_range)
        }
        
        self.client.store(
            content=content,
            metadata=metadata,
            collection='reviews'
        )
        
        return f"Search strategy defined for review: {self.review_title}"
    
    def screen_papers(self, papers: List[ReviewPaper], reviewer: str) -> Dict[str, Any]:
        """
        Screen papers for inclusion/exclusion.
        
        Args:
            papers: List of papers to screen
            reviewer: Name of the reviewer
            
        Returns:
            Screening results summary
        """
        self.current_stage = ReviewStage.SCREENING
        
        included_count = 0
        excluded_count = 0
        screening_results = []
        
        for paper in papers:
            # Store individual screening decision
            screening_content = f"""
# Screening Decision: {paper.title}

**Paper ID:** {paper.paper_id}
**Authors:** {', '.join(paper.authors)}
**Venue:** {paper.venue} ({paper.year})

## Abstract
{paper.abstract}

## Decision
**Included:** {paper.included}
**Exclusion Reason:** {paper.exclusion_reason or 'N/A'}

## Reviewer Notes
{paper.reviewer_notes}

**Reviewer:** {reviewer}
**Date:** {datetime.now().strftime('%Y-%m-%d')}
            """
            
            metadata = {
                'type': 'screening_decision',
                'review_id': self.review_id,
                'paper_id': paper.paper_id,
                'included': paper.included,
                'exclusion_reason': paper.exclusion_reason,
                'reviewer': reviewer,
                'screening_date': datetime.now().isoformat()
            }
            
            self.client.store(
                content=screening_content,
                metadata=metadata,
                collection='reviews'
            )
            
            if paper.included:
                included_count += 1
            else:
                excluded_count += 1
            
            screening_results.append({
                'paper_id': paper.paper_id,
                'title': paper.title,
                'included': paper.included,
                'exclusion_reason': paper.exclusion_reason
            })
        
        # Store screening summary
        summary_content = f"""
# Screening Summary: {self.review_title}

## Results
- **Total papers screened:** {len(papers)}
- **Included:** {included_count}
- **Excluded:** {excluded_count}
- **Inclusion rate:** {(included_count/len(papers)*100):.1f}%

## Reviewer
{reviewer}

## Date
{datetime.now().strftime('%Y-%m-%d')}
        """
        
        self.client.store(
            content=summary_content,
            metadata={
                'type': 'screening_summary',
                'review_id': self.review_id,
                'total_screened': len(papers),
                'included': included_count,
                'excluded': excluded_count,
                'reviewer': reviewer
            },
            collection='reviews'
        )
        
        return {
            'total_screened': len(papers),
            'included': included_count,
            'excluded': excluded_count,
            'inclusion_rate': included_count/len(papers)*100,
            'results': screening_results
        }
    
    def extract_data(self, paper: ReviewPaper, extraction_form: Dict[str, Any]) -> str:
        """Extract data from included papers."""
        self.current_stage = ReviewStage.EXTRACTION
        
        # Update paper with extraction data
        paper.data_extracted = True
        paper.extraction_data = extraction_form
        
        content = f"""
# Data Extraction: {paper.title}

**Paper ID:** {paper.paper_id}
**Authors:** {', '.join(paper.authors)}

## Extraction Form Data
"""
        
        for field, value in extraction_form.items():
            content += f"\n**{field}:** {value}"
        
        content += f"""

## Quality Assessment
**Quality Score:** {paper.quality_score}/10

## Extraction Notes
{paper.reviewer_notes}

**Extraction Date:** {datetime.now().strftime('%Y-%m-%d')}
        """
        
        metadata = {
            'type': 'data_extraction',
            'review_id': self.review_id,
            'paper_id': paper.paper_id,
            'quality_score': paper.quality_score,
            'extraction_data': extraction_form,
            'extraction_date': datetime.now().isoformat()
        }
        
        self.client.store(
            content=content,
            metadata=metadata,
            collection='reviews'
        )
        
        return f"Data extracted for paper: {paper.title}"
    
    def generate_synthesis_report(self) -> str:
        """Generate synthesis report from extracted data."""
        self.current_stage = ReviewStage.SYNTHESIS
        
        # Retrieve all extraction data
        extractions = self.client.search(
            query="*",
            collection='reviews',
            metadata_filter={
                'type': 'data_extraction',
                'review_id': self.review_id
            },
            limit=1000
        )
        
        # Analyze extracted data
        synthesis_data = self._analyze_extractions(extractions)
        
        # Generate synthesis report
        content = f"""
# Synthesis Report: {self.review_title}

## Overview
- **Number of included studies:** {synthesis_data['study_count']}
- **Data extraction completed:** {datetime.now().strftime('%Y-%m-%d')}

## Key Findings
{synthesis_data['key_findings']}

## Methodological Quality
- **Average quality score:** {synthesis_data['avg_quality']:.1f}/10
- **Quality range:** {synthesis_data['quality_range']}

## Study Characteristics
{synthesis_data['characteristics']}

## Implications
{synthesis_data['implications']}

## Limitations
{synthesis_data['limitations']}

## Recommendations for Future Research
{synthesis_data['recommendations']}
        """
        
        metadata = {
            'type': 'synthesis_report',
            'review_id': self.review_id,
            'study_count': synthesis_data['study_count'],
            'avg_quality': synthesis_data['avg_quality'],
            'synthesis_date': datetime.now().isoformat()
        }
        
        self.client.store(
            content=content,
            metadata=metadata,
            collection='reviews'
        )
        
        return content
    
    def _analyze_extractions(self, extractions: List[Dict]) -> Dict[str, Any]:
        """Analyze extraction data for synthesis."""
        study_count = len(extractions)
        quality_scores = []
        
        for extraction in extractions:
            metadata = extraction.get('metadata', {})
            if metadata.get('quality_score'):
                quality_scores.append(metadata['quality_score'])
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        quality_range = f"{min(quality_scores)}-{max(quality_scores)}" if quality_scores else "N/A"
        
        return {
            'study_count': study_count,
            'avg_quality': avg_quality,
            'quality_range': quality_range,
            'key_findings': "To be synthesized from individual studies",
            'characteristics': "Study characteristics to be analyzed",
            'implications': "Practical implications to be determined",
            'limitations': "Review limitations to be documented",
            'recommendations': "Future research directions to be identified"
        }
    
    def get_review_status(self) -> Dict[str, Any]:
        """Get current review status and progress."""
        # Count papers in each stage
        screening_decisions = self.client.search(
            query="*",
            collection='reviews',
            metadata_filter={
                'type': 'screening_decision',
                'review_id': self.review_id
            },
            limit=1000
        )
        
        extractions = self.client.search(
            query="*",
            collection='reviews',
            metadata_filter={
                'type': 'data_extraction',
                'review_id': self.review_id
            },
            limit=1000
        )
        
        included_count = sum(1 for d in screening_decisions 
                           if d.get('metadata', {}).get('included', False))
        
        return {
            'review_title': self.review_title,
            'current_stage': self.current_stage.value,
            'papers_screened': len(screening_decisions),
            'papers_included': included_count,
            'papers_extracted': len(extractions),
            'completion_rate': {
                'screening': len(screening_decisions) > 0,
                'extraction': len(extractions) == included_count if included_count > 0 else False,
                'synthesis': self.current_stage in [ReviewStage.SYNTHESIS, ReviewStage.REPORTING]
            }
        }

# Usage example
if __name__ == "__main__":
    from workspace_qdrant_mcp.client import WorkspaceClient
    
    client = WorkspaceClient()
    
    # Initialize systematic review
    review = LiteratureReview(client, "Machine Learning in Healthcare Diagnosis")
    
    # Define search strategy
    strategy = SearchStrategy(
        databases=["PubMed", "IEEE Xplore", "ACM Digital Library"],
        search_terms=["machine learning", "healthcare", "diagnosis", "classification"],
        inclusion_criteria=[
            "Peer-reviewed articles",
            "Published 2018-2023",
            "English language",
            "Focus on diagnostic applications"
        ],
        exclusion_criteria=[
            "Review articles",
            "Conference abstracts only",
            "Non-English publications"
        ],
        date_range=(2018, 2023)
    )
    
    review.define_search_strategy(strategy)
    print("✅ Systematic review initialized with search strategy")
```

## 💡 Claude Interaction Prompts

### Research-Focused Prompts

**Paper Discovery:**
```
Search my research collection for:
- Papers about [specific topic] published in the last 2 years
- All papers by [Author Name] with their key contributions
- Research methodologies used in [field] studies
- Papers with high citation counts in my collection
- Recent preprints related to my current research
```

**Literature Analysis:**
```
Analyze my literature collection to:
- Identify research gaps in [specific area]
- Compare methodologies across different studies
- Find conflicting findings or inconsistent results
- Trace the evolution of concepts over time
- Identify emerging trends and future directions
```

**Research Synthesis:**
```
Help me synthesize research findings about:
- Common patterns across multiple studies
- Effectiveness of different approaches
- Limitations and recommendations from various papers
- Theoretical frameworks used in the field
- Practical applications and real-world implementations
```

### Advanced Research Workflows

**Grant Writing Support:**
```python
# Use in Claude Code for grant proposal development
"""
I'm writing a grant proposal on [topic]. Please help me:

1. Search my literature collection for relevant background research
2. Identify key researchers and institutions in this field
3. Find recent breakthroughs and current challenges
4. Locate similar funded projects and their outcomes
5. Extract methodology details for my proposed approach
6. Identify potential collaborators based on my research network

Focus on papers from top-tier venues in the last 3 years.
"""
```

## 📊 Best Practices

### Collection Organization

**Recommended collection structure for academic research:**

```bash
# Research-specific collections
export COLLECTIONS="papers,notes,reviews,experiments"

# Global academic collections
export GLOBAL_COLLECTIONS="citations,methodologies,datasets,collaborations"

# Example result:
# myresearch-papers         # Project-specific papers
# myresearch-notes          # Research notes and insights
# myresearch-reviews        # Literature reviews and analyses
# myresearch-experiments    # Experimental data and results
# citations                 # Citation network and references
# methodologies            # Research methods and protocols
# datasets                 # Shared datasets and data sources
# collaborations           # Collaborative research projects
```

### Automated Research Workflows

**Set up automated paper monitoring:**

```bash
# Daily arXiv monitoring (add to cron job)
python paper_management.py --arxiv-query "cat:cs.AI" --daily-update

# Citation tracking
python citation_tracker.py --update-citations --notify-new

# Collaboration monitoring
python collaboration_tracker.py --check-coauthor-publications
```

### Research Note Integration

**Integrate with lab notebook systems:**

1. **Hypothesis tracking** - Link hypotheses to relevant literature
2. **Experimental design** - Reference methodologies from papers
3. **Results interpretation** - Compare with published findings
4. **Collaboration notes** - Document research meetings and decisions

## 🔗 Integration Examples

- **[VS Code Integration](../integrations/vscode/README.md)** - Research workspace setup
- **[Automation Scripts](../integrations/automation/README.md)** - Automated literature monitoring
- **[Performance Optimization](../performance_optimization/README.md)** - Large literature collections

---

**Next Steps:**
1. Try the [Paper Management Example](paper_management/)
2. Set up [Literature Review Workflows](literature_review/)
3. Explore [Research Collaboration](collaboration/)