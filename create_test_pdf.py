#!/usr/bin/env python3
"""Create a test PDF document for format comparison."""

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from pathlib import Path

def create_test_pdf():
    """Create a test PDF with the same content as other formats."""
    
    if not HAS_REPORTLAB:
        print("reportlab not available. Creating simple text-based PDF alternative.")
        return None
    
    pdf_path = Path("format_benchmark_results/test_document.pdf")
    
    # Create PDF document
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    subheading_style = styles['Heading3']
    normal_style = styles['Normal']
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=10,
        backgroundColor=colors.lightgrey,
        borderWidth=1,
        borderColor=colors.grey,
        leftIndent=20,
        rightIndent=20,
        spaceAbove=12,
        spaceBelow=12,
    )
    
    # Title
    story.append(Paragraph("Document Format Extraction Test", title_style))
    story.append(Spacer(1, 12))
    
    # Introduction
    story.append(Paragraph("Introduction", heading_style))
    story.append(Paragraph(
        "This document contains various types of content to test text extraction quality across different formats.",
        normal_style
    ))
    story.append(Spacer(1, 12))
    
    # Text formatting
    story.append(Paragraph("Text Formatting", subheading_style))
    story.append(Paragraph(
        "This paragraph contains <b>bold text</b>, <i>italic text</i>, and <font name='Courier'>inline code</font>.",
        normal_style
    ))
    story.append(Spacer(1, 12))
    
    # Code block
    story.append(Paragraph("Code Block", subheading_style))
    story.append(Paragraph(
        'def hello_world():<br/>&nbsp;&nbsp;&nbsp;&nbsp;print("Hello, World!")<br/>&nbsp;&nbsp;&nbsp;&nbsp;return 42',
        code_style
    ))
    story.append(Spacer(1, 12))
    
    # Lists
    story.append(Paragraph("Lists", subheading_style))
    story.append(Paragraph("Unordered list:", normal_style))
    story.append(Paragraph("• Item 1", normal_style))
    story.append(Paragraph("• Item 2", normal_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• Nested item A", normal_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• Nested item B", normal_style))
    story.append(Paragraph("• Item 3", normal_style))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Ordered list:", normal_style))
    story.append(Paragraph("1. First item", normal_style))
    story.append(Paragraph("2. Second item", normal_style))
    story.append(Paragraph("3. Third item", normal_style))
    story.append(Spacer(1, 12))
    
    # Table
    story.append(Paragraph("Table", subheading_style))
    table_data = [
        ['Format', 'Extraction Quality', 'Processing Speed'],
        ['PDF', 'Good', 'Medium'],
        ['EPUB', 'Excellent', 'Fast'],
        ['HTML', 'Very Good', 'Fast']
    ]
    
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 12))
    
    # Special characters
    story.append(Paragraph("Special Characters", subheading_style))
    story.append(Paragraph(
        "Unicode test: café, naïve, résumé, 中文, 日本語, العربية", 
        normal_style
    ))
    story.append(Paragraph(
        "Mathematical notation: E = mc², ∑, ∫, √, ≤, ≥", 
        normal_style
    ))
    story.append(Spacer(1, 12))
    
    # Conclusion
    story.append(Paragraph("Conclusion", subheading_style))
    story.append(Paragraph(
        "This document tests various formatting elements to evaluate extraction quality.",
        normal_style
    ))
    
    # Build PDF
    doc.build(story)
    print(f"Created test PDF: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    if HAS_REPORTLAB:
        create_test_pdf()
    else:
        print("Install reportlab to create PDF: pip install reportlab")