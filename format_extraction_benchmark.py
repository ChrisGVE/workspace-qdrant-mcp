#!/usr/bin/env python3
"""
Document Format Extraction Quality Benchmark

This script systematically evaluates text extraction quality across different
document formats to establish optimal precedence rules for version management.

Research objectives:
1. Compare text extraction quality (completeness, accuracy, formatting preservation)
2. Measure processing speed and resource usage
3. Evaluate metadata extraction capabilities
4. Test error handling and edge cases
5. Establish format precedence rules

Formats tested:
- PDF (text-based and image-based)
- EPUB
- MOBI (if available)
- HTML
- Markdown
- Plain text
- Microsoft Word (if available)
"""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
import traceback

# Try to import various document processing libraries
AVAILABLE_LIBRARIES = {}

# PDF processing
try:
    import pypdf
    AVAILABLE_LIBRARIES['pypdf'] = pypdf.__version__
except ImportError:
    AVAILABLE_LIBRARIES['pypdf'] = None

try:
    import pdfplumber
    AVAILABLE_LIBRARIES['pdfplumber'] = pdfplumber.__version__
except ImportError:
    AVAILABLE_LIBRARIES['pdfplumber'] = None

try:
    import pymupdf  # fitz
    AVAILABLE_LIBRARIES['pymupdf'] = pymupdf.__version__
except ImportError:
    AVAILABLE_LIBRARIES['pymupdf'] = None

# EPUB processing
try:
    import ebooklib
    from ebooklib import epub
    AVAILABLE_LIBRARIES['ebooklib'] = "available"
except ImportError:
    AVAILABLE_LIBRARIES['ebooklib'] = None

# HTML processing
try:
    from bs4 import BeautifulSoup
    import bs4
    AVAILABLE_LIBRARIES['beautifulsoup4'] = bs4.__version__
except ImportError:
    AVAILABLE_LIBRARIES['beautifulsoup4'] = None

# General text extraction
try:
    import textract
    AVAILABLE_LIBRARIES['textract'] = "available"
except ImportError:
    AVAILABLE_LIBRARIES['textract'] = None

# Markdown processing
try:
    import markdown
    AVAILABLE_LIBRARIES['markdown'] = markdown.__version__
except ImportError:
    AVAILABLE_LIBRARIES['markdown'] = None

# Word document processing
try:
    from docx import Document as DocxDocument
    AVAILABLE_LIBRARIES['python-docx'] = "available"
except ImportError:
    AVAILABLE_LIBRARIES['python-docx'] = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentExtractionBenchmark:
    """Benchmark different document formats for text extraction quality."""
    
    def __init__(self, output_dir: str = "format_benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        
    def create_reference_content(self) -> str:
        """Create reference content for testing."""
        return """# Document Format Extraction Test

## Introduction
This document contains various types of content to test text extraction quality across different formats.

### Text Formatting
This paragraph contains **bold text**, *italic text*, and `inline code`.

### Code Block
```python
def hello_world():
    print("Hello, World!")
    return 42
```

### Lists
Unordered list:
- Item 1
- Item 2
  - Nested item A
  - Nested item B
- Item 3

Ordered list:
1. First item
2. Second item
3. Third item

### Table
| Format | Extraction Quality | Processing Speed |
|--------|-------------------|------------------|
| PDF    | Good              | Medium           |
| EPUB   | Excellent         | Fast             |
| HTML   | Very Good         | Fast             |

### Special Characters
Unicode test: café, naïve, résumé, 中文, 日本語, العربية

Mathematical notation: E = mc², ∑, ∫, √, ≤, ≥

### Conclusion
This document tests various formatting elements to evaluate extraction quality.
"""

    def create_test_documents(self) -> Dict[str, Path]:
        """Create test documents in various formats."""
        test_docs = {}
        content = self.create_reference_content()
        
        # Create plain text version
        txt_path = self.output_dir / "test_document.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        test_docs['txt'] = txt_path
        
        # Create Markdown version
        md_path = self.output_dir / "test_document.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
        test_docs['md'] = md_path
        
        # Create HTML version
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Document Format Extraction Test</title>
    <meta charset="UTF-8">
</head>
<body>
{self._markdown_to_html(content)}
</body>
</html>"""
        html_path = self.output_dir / "test_document.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        test_docs['html'] = html_path
        
        # Check if PDF was created by external script
        pdf_path = self.output_dir / "test_document.pdf"
        if pdf_path.exists():
            test_docs['pdf'] = pdf_path
            logger.info(f"Found existing PDF: {pdf_path}")
        else:
            logger.info("PDF not found - PDF extraction tests will be skipped")
        
        # Create EPUB version if possible
        epub_path = self._create_epub(content)
        if epub_path:
            test_docs['epub'] = epub_path
            logger.info(f"Created EPUB: {epub_path}")
        else:
            logger.info("EPUB creation failed - EPUB extraction tests will be skipped")
        
        logger.info(f"Created test documents in: {self.output_dir}")
        return test_docs
    
    def _markdown_to_html(self, content: str) -> str:
        """Convert markdown to HTML (basic conversion)."""
        if AVAILABLE_LIBRARIES.get('markdown'):
            return markdown.markdown(content, extensions=['tables', 'codehilite'])
        
        # Basic manual conversion
        lines = content.split('\n')
        html_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                html_lines.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('## '):
                html_lines.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                html_lines.append(f'<h3>{line[4:]}</h3>')
            elif line.startswith('- '):
                html_lines.append(f'<li>{line[2:]}</li>')
            elif line:
                html_lines.append(f'<p>{line}</p>')
            else:
                html_lines.append('<br>')
                
        return '\n'.join(html_lines)
    
    def _create_epub(self, content: str) -> Optional[Path]:
        """Create an EPUB version of the test document."""
        if not AVAILABLE_LIBRARIES.get('ebooklib'):
            return None
            
        try:
            from ebooklib import epub
            
            book = epub.EpubBook()
            
            # Set metadata
            book.set_identifier('test-document-001')
            book.set_title('Document Format Extraction Test')
            book.set_language('en')
            book.add_author('Benchmark Test')
            
            # Create chapter
            chapter = epub.EpubHtml(
                title='Test Content',
                file_name='test.xhtml',
                lang='en'
            )
            
            # Convert markdown to HTML for EPUB
            html_content = self._markdown_to_html(content)
            chapter.content = f'<html><head><title>Test Content</title></head><body>{html_content}</body></html>'
            
            # Add chapter to book
            book.add_item(chapter)
            
            # Define Table of Contents
            book.toc = (epub.Link("test.xhtml", "Test Content", "test"),)
            
            # Add default NCX and Nav file
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            
            # Define CSS style
            style = 'BODY {color: black;}'
            nav_css = epub.EpubItem(
                uid="nav", 
                file_name="style/nav.css", 
                media_type="text/css", 
                content=style
            )
            book.add_item(nav_css)
            
            # Basic spine
            book.spine = ['nav', chapter]
            
            # Write EPUB file
            epub_path = self.output_dir / "test_document.epub"
            epub.write_epub(str(epub_path), book, {})
            
            return epub_path
            
        except Exception as e:
            logger.warning(f"Failed to create EPUB: {e}")
            return None

    async def extract_text_txt(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from plain text file."""
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            extraction_time = time.time() - start_time
            
            return {
                'format': 'txt',
                'success': True,
                'content': content,
                'extraction_time': extraction_time,
                'content_length': len(content),
                'word_count': len(content.split()),
                'metadata': {
                    'encoding': 'utf-8',
                    'file_size': file_path.stat().st_size
                },
                'error': None
            }
        except Exception as e:
            return {
                'format': 'txt',
                'success': False,
                'content': '',
                'extraction_time': time.time() - start_time,
                'error': str(e)
            }

    async def extract_text_html(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from HTML file."""
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            if AVAILABLE_LIBRARIES.get('beautifulsoup4'):
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                content = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in content.splitlines())
                content = '\n'.join(line for line in lines if line)
                
                metadata = {
                    'title': soup.title.string if soup.title else None,
                    'meta_tags': len(soup.find_all('meta')),
                    'links': len(soup.find_all('a')),
                    'images': len(soup.find_all('img'))
                }
            else:
                # Basic HTML tag removal
                import re
                content = re.sub('<[^<]+?>', '', html_content)
                metadata = {'parser': 'regex_basic'}
            
            extraction_time = time.time() - start_time
            
            return {
                'format': 'html',
                'success': True,
                'content': content,
                'extraction_time': extraction_time,
                'content_length': len(content),
                'word_count': len(content.split()),
                'metadata': metadata,
                'error': None
            }
        except Exception as e:
            return {
                'format': 'html',
                'success': False,
                'content': '',
                'extraction_time': time.time() - start_time,
                'error': str(e)
            }

    async def extract_text_pdf_pypdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF using pypdf."""
        start_time = time.time()
        
        if not AVAILABLE_LIBRARIES.get('pypdf'):
            return {
                'format': 'pdf_pypdf',
                'success': False,
                'content': '',
                'extraction_time': 0,
                'error': 'pypdf not available'
            }
        
        try:
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                
                content_parts = []
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        content_parts.append(page_text)
                
                content = '\n\n'.join(content_parts)
                
                # Extract metadata
                metadata = {}
                if reader.metadata:
                    for key, value in reader.metadata.items():
                        if isinstance(value, str):
                            metadata[key.lstrip('/')] = value
                
                extraction_time = time.time() - start_time
                
                return {
                    'format': 'pdf_pypdf',
                    'success': True,
                    'content': content,
                    'extraction_time': extraction_time,
                    'content_length': len(content),
                    'word_count': len(content.split()),
                    'metadata': {
                        'pages': len(reader.pages),
                        'encrypted': reader.is_encrypted,
                        **metadata
                    },
                    'error': None
                }
        except Exception as e:
            return {
                'format': 'pdf_pypdf',
                'success': False,
                'content': '',
                'extraction_time': time.time() - start_time,
                'error': str(e)
            }

    async def extract_text_pdf_pdfplumber(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF using pdfplumber."""
        start_time = time.time()
        
        if not AVAILABLE_LIBRARIES.get('pdfplumber'):
            return {
                'format': 'pdf_pdfplumber',
                'success': False,
                'content': '',
                'extraction_time': 0,
                'error': 'pdfplumber not available'
            }
        
        try:
            import pdfplumber
            
            with pdfplumber.open(file_path) as pdf:
                content_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content_parts.append(page_text)
                
                content = '\n\n'.join(content_parts)
                
                extraction_time = time.time() - start_time
                
                return {
                    'format': 'pdf_pdfplumber',
                    'success': True,
                    'content': content,
                    'extraction_time': extraction_time,
                    'content_length': len(content),
                    'word_count': len(content.split()),
                    'metadata': {
                        'pages': len(pdf.pages),
                        'metadata': pdf.metadata if pdf.metadata else {}
                    },
                    'error': None
                }
        except Exception as e:
            return {
                'format': 'pdf_pdfplumber',
                'success': False,
                'content': '',
                'extraction_time': time.time() - start_time,
                'error': str(e)
            }

    async def extract_text_pdf_pymupdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF using PyMuPDF."""
        start_time = time.time()
        
        if not AVAILABLE_LIBRARIES.get('pymupdf'):
            return {
                'format': 'pdf_pymupdf',
                'success': False,
                'content': '',
                'extraction_time': 0,
                'error': 'pymupdf not available'
            }
        
        try:
            import fitz  # pymupdf
            
            doc = fitz.open(file_path)
            content_parts = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    content_parts.append(page_text)
            
            content = '\n\n'.join(content_parts)
            
            metadata = doc.metadata
            doc.close()
            
            extraction_time = time.time() - start_time
            
            return {
                'format': 'pdf_pymupdf',
                'success': True,
                'content': content,
                'extraction_time': extraction_time,
                'content_length': len(content),
                'word_count': len(content.split()),
                'metadata': {
                    'pages': doc.page_count,
                    **metadata
                },
                'error': None
            }
        except Exception as e:
            return {
                'format': 'pdf_pymupdf',
                'success': False,
                'content': '',
                'extraction_time': time.time() - start_time,
                'error': str(e)
            }
    
    async def extract_text_epub(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from EPUB file."""
        start_time = time.time()
        
        if not AVAILABLE_LIBRARIES.get('ebooklib'):
            return {
                'format': 'epub',
                'success': False,
                'content': '',
                'extraction_time': 0,
                'error': 'ebooklib not available'
            }
        
        try:
            from ebooklib import epub
            from bs4 import BeautifulSoup
            
            book = epub.read_epub(str(file_path))
            
            content_parts = []
            metadata = {}
            
            # Extract metadata
            try:
                # Try different ways to get metadata
                if hasattr(book, 'get_metadata'):
                    try:
                        # Try with namespace
                        dc_meta = book.get_metadata('DC')
                        for item in dc_meta:
                            if len(item) >= 2:
                                key = item[0].split(':')[-1] if ':' in item[0] else item[0]
                                metadata[f'dc_{key.lower()}'] = item[1]
                    except (TypeError, AttributeError):
                        # Alternative metadata extraction
                        pass
                
                # Try direct attribute access
                if hasattr(book, 'title') and book.title:
                    metadata['title'] = str(book.title)
                if hasattr(book, 'author') and book.author:
                    metadata['author'] = str(book.author)
                if hasattr(book, 'language') and book.language:
                    metadata['language'] = str(book.language)
            except Exception as e:
                logger.warning(f"Failed to extract EPUB metadata: {e}")
                metadata['metadata_error'] = str(e)
            
            # Extract text from all items
            for item in book.get_items():
                if item.get_type() == 9:  # EPUB HTML item
                    try:
                        soup = BeautifulSoup(item.get_content(), 'html.parser')
                        text = soup.get_text()
                        if text.strip():
                            content_parts.append(text.strip())
                    except Exception as e:
                        logger.warning(f"Failed to extract text from EPUB item: {e}")
            
            content = '\n\n'.join(content_parts)
            
            extraction_time = time.time() - start_time
            
            return {
                'format': 'epub',
                'success': True,
                'content': content,
                'extraction_time': extraction_time,
                'content_length': len(content),
                'word_count': len(content.split()),
                'metadata': metadata,
                'error': None
            }
        except Exception as e:
            return {
                'format': 'epub',
                'success': False,
                'content': '',
                'extraction_time': time.time() - start_time,
                'error': str(e)
            }

    def calculate_extraction_quality(self, reference: str, extracted: str) -> Dict[str, float]:
        """Calculate extraction quality metrics."""
        if not extracted:
            return {
                'completeness': 0.0,
                'accuracy': 0.0,
                'similarity': 0.0
            }
        
        # Normalize texts for comparison
        ref_words = set(reference.lower().split())
        ext_words = set(extracted.lower().split())
        
        # Completeness: what percentage of reference words were extracted
        completeness = len(ref_words.intersection(ext_words)) / len(ref_words) if ref_words else 0.0
        
        # Accuracy: what percentage of extracted words are from the reference
        accuracy = len(ref_words.intersection(ext_words)) / len(ext_words) if ext_words else 0.0
        
        # Simple similarity based on character-level comparison
        ref_chars = set(reference.lower())
        ext_chars = set(extracted.lower())
        similarity = len(ref_chars.intersection(ext_chars)) / len(ref_chars.union(ext_chars)) if ref_chars.union(ext_chars) else 0.0
        
        return {
            'completeness': completeness,
            'accuracy': accuracy,
            'similarity': similarity
        }

    async def run_benchmark(self, test_docs: Dict[str, Path]) -> List[Dict[str, Any]]:
        """Run extraction benchmark on all formats."""
        results = []
        reference_content = self.create_reference_content()
        
        # Test plain text extraction
        if 'txt' in test_docs:
            result = await self.extract_text_txt(test_docs['txt'])
            if result['success']:
                quality = self.calculate_extraction_quality(reference_content, result['content'])
                result.update(quality)
            results.append(result)
        
        # Test Markdown extraction (treat as text)
        if 'md' in test_docs:
            result = await self.extract_text_txt(test_docs['md'])
            result['format'] = 'md'
            if result['success']:
                quality = self.calculate_extraction_quality(reference_content, result['content'])
                result.update(quality)
            results.append(result)
        
        # Test HTML extraction
        if 'html' in test_docs:
            result = await self.extract_text_html(test_docs['html'])
            if result['success']:
                quality = self.calculate_extraction_quality(reference_content, result['content'])
                result.update(quality)
            results.append(result)
        
        # Test PDF extraction with different libraries
        if 'pdf' in test_docs:
            # Test with pypdf
            result = await self.extract_text_pdf_pypdf(test_docs['pdf'])
            if result['success']:
                quality = self.calculate_extraction_quality(reference_content, result['content'])
                result.update(quality)
            results.append(result)
            
            # Test with pdfplumber
            result = await self.extract_text_pdf_pdfplumber(test_docs['pdf'])
            if result['success']:
                quality = self.calculate_extraction_quality(reference_content, result['content'])
                result.update(quality)
            results.append(result)
            
            # Test with pymupdf
            result = await self.extract_text_pdf_pymupdf(test_docs['pdf'])
            if result['success']:
                quality = self.calculate_extraction_quality(reference_content, result['content'])
                result.update(quality)
            results.append(result)
        
        # Test EPUB extraction
        if 'epub' in test_docs:
            result = await self.extract_text_epub(test_docs['epub'])
            if result['success']:
                quality = self.calculate_extraction_quality(reference_content, result['content'])
                result.update(quality)
            results.append(result)
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results and generate recommendations."""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {'error': 'No successful extractions'}
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in ['completeness', 'accuracy', 'similarity', 'extraction_time']:
            values = [r.get(metric, 0) for r in successful_results if metric in r]
            if values:
                avg_metrics[f'avg_{metric}'] = sum(values) / len(values)
                avg_metrics[f'max_{metric}'] = max(values)
                avg_metrics[f'min_{metric}'] = min(values)
        
        # Rank formats by overall quality
        quality_scores = []
        for result in successful_results:
            if all(key in result for key in ['completeness', 'accuracy', 'similarity']):
                # Weighted quality score
                quality_score = (
                    result['completeness'] * 0.4 +
                    result['accuracy'] * 0.3 +
                    result['similarity'] * 0.3
                )
                quality_scores.append({
                    'format': result['format'],
                    'quality_score': quality_score,
                    'extraction_time': result['extraction_time'],
                    'completeness': result['completeness'],
                    'accuracy': result['accuracy'],
                    'similarity': result['similarity']
                })
        
        # Sort by quality score (descending)
        quality_scores.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return {
            'total_formats_tested': len(results),
            'successful_extractions': len(successful_results),
            'failed_extractions': len(results) - len(successful_results),
            'average_metrics': avg_metrics,
            'format_rankings': quality_scores,
            'library_availability': AVAILABLE_LIBRARIES
        }
    
    def generate_precedence_rules(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate format precedence rules based on analysis."""
        if 'format_rankings' not in analysis:
            return {'error': 'No format rankings available'}
        
        rankings = analysis['format_rankings']
        
        # Base precedence rules
        precedence_rules = {
            'primary_criteria': 'quality_score',
            'secondary_criteria': 'extraction_speed',
            'format_precedence': [],
            'thresholds': {
                'minimum_quality': 0.7,
                'maximum_extraction_time': 10.0
            },
            'recommendations': []
        }
        
        for rank, format_info in enumerate(rankings):
            precedence_rules['format_precedence'].append({
                'rank': rank + 1,
                'format': format_info['format'],
                'quality_score': format_info['quality_score'],
                'extraction_time': format_info['extraction_time'],
                'recommended': format_info['quality_score'] >= 0.7
            })
        
        # Generate specific recommendations
        if rankings:
            best_format = rankings[0]
            precedence_rules['recommendations'].append(
                f"Primary recommendation: {best_format['format']} "
                f"(quality: {best_format['quality_score']:.3f})"
            )
        
        # Speed recommendations
        fast_formats = [f for f in rankings if f['extraction_time'] < 1.0]
        if fast_formats:
            precedence_rules['recommendations'].append(
                f"Fastest extraction: {fast_formats[0]['format']} "
                f"({fast_formats[0]['extraction_time']:.3f}s)"
            )
        
        return precedence_rules
    
    async def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any], 
                          precedence_rules: Dict[str, Any]) -> None:
        """Save benchmark results to files."""
        # Save raw results
        results_file = self.output_dir / "extraction_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save analysis
        analysis_file = self.output_dir / "analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save precedence rules
        precedence_file = self.output_dir / "precedence_rules.json"
        with open(precedence_file, 'w') as f:
            json.dump(precedence_rules, f, indent=2)
        
        # Generate summary report
        report_file = self.output_dir / "benchmark_report.md"
        await self.generate_report(results, analysis, precedence_rules, report_file)
        
        logger.info(f"Results saved to {self.output_dir}/")
    
    async def generate_report(self, results: List[Dict[str, Any]], analysis: Dict[str, Any], 
                             precedence_rules: Dict[str, Any], report_file: Path) -> None:
        """Generate comprehensive benchmark report."""
        with open(report_file, 'w') as f:
            f.write("# Document Format Extraction Benchmark Report\n\n")
            
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total formats tested:** {analysis.get('total_formats_tested', 0)}\n")
            f.write(f"- **Successful extractions:** {analysis.get('successful_extractions', 0)}\n")
            f.write(f"- **Failed extractions:** {analysis.get('failed_extractions', 0)}\n\n")
            
            # Library Availability
            f.write("## Library Availability\n\n")
            for lib, version in AVAILABLE_LIBRARIES.items():
                status = "✅ Available" if version else "❌ Not Available"
                f.write(f"- **{lib}:** {status}\n")
            f.write("\n")
            
            # Format Rankings
            if 'format_rankings' in analysis:
                f.write("## Format Quality Rankings\n\n")
                f.write("| Rank | Format | Quality Score | Extraction Time | Completeness | Accuracy | Similarity |\n")
                f.write("|------|--------|---------------|-----------------|--------------|-----------|------------|\n")
                
                for rank, fmt in enumerate(analysis['format_rankings']):
                    f.write(f"| {rank + 1} | {fmt['format']} | {fmt['quality_score']:.3f} | "
                           f"{fmt['extraction_time']:.3f}s | {fmt['completeness']:.3f} | "
                           f"{fmt['accuracy']:.3f} | {fmt['similarity']:.3f} |\n")
                f.write("\n")
            
            # Precedence Rules
            if 'format_precedence' in precedence_rules:
                f.write("## Format Precedence Rules\n\n")
                f.write("Based on the benchmark results, the following precedence rules are recommended:\n\n")
                
                for rule in precedence_rules['format_precedence']:
                    status = "✅ Recommended" if rule['recommended'] else "❌ Not Recommended"
                    f.write(f"{rule['rank']}. **{rule['format']}** - {status}\n")
                    f.write(f"   - Quality Score: {rule['quality_score']:.3f}\n")
                    f.write(f"   - Extraction Time: {rule['extraction_time']:.3f}s\n\n")
            
            # Recommendations
            if 'recommendations' in precedence_rules:
                f.write("## Implementation Recommendations\n\n")
                for rec in precedence_rules['recommendations']:
                    f.write(f"- {rec}\n")
                f.write("\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            for result in results:
                f.write(f"### {result['format']}\n\n")
                if result['success']:
                    f.write(f"- **Status:** ✅ Success\n")
                    f.write(f"- **Content Length:** {result.get('content_length', 0)} characters\n")
                    f.write(f"- **Word Count:** {result.get('word_count', 0)} words\n")
                    f.write(f"- **Extraction Time:** {result.get('extraction_time', 0):.3f} seconds\n")
                    if 'completeness' in result:
                        f.write(f"- **Completeness:** {result['completeness']:.3f}\n")
                        f.write(f"- **Accuracy:** {result['accuracy']:.3f}\n")
                        f.write(f"- **Similarity:** {result['similarity']:.3f}\n")
                else:
                    f.write(f"- **Status:** ❌ Failed\n")
                    f.write(f"- **Error:** {result.get('error', 'Unknown error')}\n")
                f.write("\n")

async def main():
    """Run the document format extraction benchmark."""
    print("🔍 Starting Document Format Extraction Benchmark...")
    print(f"📚 Available libraries: {sum(1 for v in AVAILABLE_LIBRARIES.values() if v)}/{len(AVAILABLE_LIBRARIES)}")
    
    benchmark = DocumentExtractionBenchmark()
    
    try:
        # Create test documents
        print("📝 Creating test documents...")
        test_docs = benchmark.create_test_documents()
        
        # Run benchmark
        print("⚡ Running extraction benchmark...")
        results = await benchmark.run_benchmark(test_docs)
        
        # Analyze results
        print("📊 Analyzing results...")
        analysis = benchmark.analyze_results(results)
        
        # Generate precedence rules
        print("🎯 Generating precedence rules...")
        precedence_rules = benchmark.generate_precedence_rules(analysis)
        
        # Save results
        print("💾 Saving results...")
        await benchmark.save_results(results, analysis, precedence_rules)
        
        print(f"✅ Benchmark complete! Results saved to: {benchmark.output_dir}/")
        
        # Print summary
        if 'format_rankings' in analysis and analysis['format_rankings']:
            best_format = analysis['format_rankings'][0]
            print(f"🏆 Best format: {best_format['format']} (quality: {best_format['quality_score']:.3f})")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))