#!/usr/bin/env python3
"""Generate minimal valid binary test corpus files.

Creates PDF, DOCX, PPTX, ODT, ODP, EPUB, and RTF files using only
Python standard library (zipfile, struct). All content is self-created
and public domain.

Run: python3 generate_binaries.py
"""

import os
import zipfile
import struct
import zlib
from pathlib import Path

CORPUS_DIR = Path(__file__).parent


def write_file(path: Path, content: bytes | str):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode) as f:
        f.write(content)
    print(f"  Created {path.relative_to(CORPUS_DIR)}")


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def create_simple_pdf(text: str) -> bytes:
    """Create a minimal valid PDF with text content."""
    # Page content stream
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    stream_len = len(stream)

    objects = []
    # Obj 1: Catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")
    # Obj 2: Pages
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj")
    # Obj 3: Page
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>\nendobj"
    )
    # Obj 4: Content stream
    objects.append(
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode()
        + stream
        + b"\nendstream\nendobj"
    )
    # Obj 5: Font
    objects.append(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj"
    )

    # Build PDF
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
        pdf.extend(b"\n")

    # Cross-reference table
    xref_offset = len(pdf)
    pdf.extend(b"xref\n")
    pdf.extend(f"0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets:
        pdf.extend(f"{off:010d} 00000 n \n".encode())

    # Trailer
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(pdf)


def create_pdf_with_image() -> bytes:
    """Create a PDF with a minimal 2x2 pixel RGB image XObject."""
    # 2x2 red pixel image (raw RGB)
    img_data = b"\xff\x00\x00" * 4  # 4 red pixels
    img_len = len(img_data)

    # Page content: draw image
    stream = b"q 100 0 0 100 72 600 cm /Img1 Do Q BT /F1 12 Tf 72 720 Td (PDF with image) Tj ET"
    stream_len = len(stream)

    objects = []
    # Obj 1: Catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")
    # Obj 2: Pages
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj")
    # Obj 3: Page
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> "
        b"/XObject << /Img1 6 0 R >> >> >>\nendobj"
    )
    # Obj 4: Content stream
    objects.append(
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode()
        + stream
        + b"\nendstream\nendobj"
    )
    # Obj 5: Font (Type1 for regression test)
    objects.append(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj"
    )
    # Obj 6: Image XObject
    objects.append(
        f"6 0 obj\n<< /Type /XObject /Subtype /Image "
        f"/Width 2 /Height 2 /ColorSpace /DeviceRGB "
        f"/BitsPerComponent 8 /Length {img_len} >>\nstream\n".encode()
        + img_data
        + b"\nendstream\nendobj"
    )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj if isinstance(obj, bytes) else obj.encode())
        pdf.extend(b"\n")

    xref_offset = len(pdf)
    pdf.extend(b"xref\n")
    pdf.extend(f"0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets:
        pdf.extend(f"{off:010d} 00000 n \n".encode())

    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(pdf)


# ---------------------------------------------------------------------------
# Office Open XML helpers (DOCX, PPTX)
# ---------------------------------------------------------------------------

CONTENT_TYPES_DOCX = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

RELS_DOCX = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>"""

WORD_RELS = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
</Relationships>"""

WORD_RELS_WITH_IMAGE = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    Target="media/image1.png"/>
</Relationships>"""


def docx_document_xml(text: str) -> str:
    return f"""\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>{text}</w:t></w:r></w:p>
  </w:body>
</w:document>"""


def create_minimal_png() -> bytes:
    """Create a minimal valid 1x1 red PNG."""

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: 1x1, 8-bit RGB
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    # IDAT: single red pixel (filter byte 0 + RGB)
    raw = b"\x00\xff\x00\x00"
    compressed = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


def create_docx(text: str, with_image: bool = False) -> bytes:
    """Create a minimal valid DOCX file."""
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES_DOCX)
        zf.writestr("_rels/.rels", RELS_DOCX)
        zf.writestr("word/_rels/document.xml.rels",
                     WORD_RELS_WITH_IMAGE if with_image else WORD_RELS)
        zf.writestr("word/document.xml", docx_document_xml(text))
        if with_image:
            zf.writestr("word/media/image1.png", create_minimal_png())
    return buf.getvalue()


CONTENT_TYPES_PPTX = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml"
    ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slides/slide1.xml"
    ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
</Types>"""

RELS_PPTX = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="ppt/presentation.xml"/>
</Relationships>"""

PPT_RELS = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
    Target="slides/slide1.xml"/>
</Relationships>"""

PRESENTATION_XML = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:sldIdLst>
    <p:sldId id="256" r:id="rId1"/>
  </p:sldIdLst>
</p:presentation>"""

SLIDE1_XML = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr/>
      <p:sp>
        <p:nvSpPr><p:cNvPr id="2" name="Title 1"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr/>
        <p:txBody>
          <a:bodyPr/>
          <a:p><a:r><a:t>Sample Presentation Slide</a:t></a:r></a:p>
        </p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
</p:sld>"""


def create_pptx() -> bytes:
    """Create a minimal valid PPTX file."""
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES_PPTX)
        zf.writestr("_rels/.rels", RELS_PPTX)
        zf.writestr("ppt/_rels/presentation.xml.rels", PPT_RELS)
        zf.writestr("ppt/presentation.xml", PRESENTATION_XML)
        zf.writestr("ppt/slides/slide1.xml", SLIDE1_XML)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# OpenDocument Format (ODT, ODP)
# ---------------------------------------------------------------------------

ODF_MANIFEST_ODT = """\
<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"
                   manifest:version="1.2">
  <manifest:file-entry manifest:media-type="application/vnd.oasis.opendocument.text"
                       manifest:full-path="/"/>
  <manifest:file-entry manifest:media-type="text/xml" manifest:full-path="content.xml"/>
</manifest:manifest>"""

ODF_MANIFEST_ODP = """\
<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"
                   manifest:version="1.2">
  <manifest:file-entry manifest:media-type="application/vnd.oasis.opendocument.presentation"
                       manifest:full-path="/"/>
  <manifest:file-entry manifest:media-type="text/xml" manifest:full-path="content.xml"/>
</manifest:manifest>"""


def odt_content_xml(text: str) -> str:
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    office:version="1.2">
  <office:body>
    <office:text>
      <text:p>{text}</text:p>
    </office:text>
  </office:body>
</office:document-content>"""


def odp_content_xml() -> str:
    return """\
<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
    xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
    office:version="1.2">
  <office:body>
    <office:presentation>
      <draw:page draw:name="Slide 1" draw:master-page-name="Default"
                 presentation:presentation-page-layout-name="AL1T0">
        <draw:frame draw:layer="layout" draw:text-style-name="P1"
                    svg:width="20cm" svg:height="3cm" svg:x="2cm" svg:y="5cm">
          <draw:text-box>
            <text:p>Sample ODP Presentation Slide</text:p>
          </draw:text-box>
        </draw:frame>
      </draw:page>
    </office:presentation>
  </office:body>
</office:document-content>"""


def create_odt(text: str) -> bytes:
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("META-INF/manifest.xml", ODF_MANIFEST_ODT)
        zf.writestr("content.xml", odt_content_xml(text))
    return buf.getvalue()


def create_odp() -> bytes:
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.presentation")
        zf.writestr("META-INF/manifest.xml", ODF_MANIFEST_ODP)
        zf.writestr("content.xml", odp_content_xml())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------

EPUB_CONTAINER = """\
<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

EPUB_OPF = """\
<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">test-corpus-epub-001</dc:identifier>
    <dc:title>Sample Test EPUB</dc:title>
    <dc:language>en</dc:language>
    <meta property="dcterms:modified">2024-01-15T00:00:00Z</meta>
  </metadata>
  <manifest>
    <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="chapter2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="cover" href="images/cover.png" media-type="image/png"/>
  </manifest>
  <spine>
    <itemref idref="chapter1"/>
    <itemref idref="chapter2"/>
  </spine>
</package>"""

EPUB_NAV = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Navigation</title></head>
<body>
<nav epub:type="toc">
  <ol>
    <li><a href="chapter1.xhtml">Chapter 1: Introduction</a></li>
    <li><a href="chapter2.xhtml">Chapter 2: Content</a></li>
  </ol>
</nav>
</body>
</html>"""

EPUB_CH1 = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 1</title></head>
<body>
<h1>Chapter 1: Introduction</h1>
<p>This is a sample EPUB chapter for testing the workspace-qdrant document processor.</p>
<p>It contains basic text content to verify EPUB ingestion and chunking.</p>
</body>
</html>"""

EPUB_CH2 = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 2</title></head>
<body>
<h1>Chapter 2: Content</h1>
<p>This second chapter tests multi-chapter EPUB processing.</p>
<p>The document processor should extract text from each chapter separately.</p>
</body>
</html>"""


def create_epub() -> bytes:
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # mimetype must be first and uncompressed
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", EPUB_CONTAINER)
        zf.writestr("OEBPS/content.opf", EPUB_OPF)
        zf.writestr("OEBPS/nav.xhtml", EPUB_NAV)
        zf.writestr("OEBPS/chapter1.xhtml", EPUB_CH1)
        zf.writestr("OEBPS/chapter2.xhtml", EPUB_CH2)
        zf.writestr("OEBPS/images/cover.png", create_minimal_png())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# RTF
# ---------------------------------------------------------------------------

def create_rtf(text: str) -> str:
    """Create a minimal valid RTF file."""
    # Escape special RTF chars
    escaped = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return (
        r"{\rtf1\ansi\deff0"
        r"{\fonttbl{\f0\froman Times New Roman;}}"
        r"{\colortbl;\red0\green0\blue0;}"
        f"\\pard\\plain\\f0\\fs24 {escaped}"
        r"\par}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating binary test corpus files...\n")

    docs = CORPUS_DIR / "docs"
    slides = CORPUS_DIR / "slides"
    ebooks = CORPUS_DIR / "ebooks"

    # PDFs
    print("PDFs:")
    write_file(docs / "simple.pdf", create_simple_pdf(
        "This is a simple text-only PDF for testing document ingestion."
    ))
    write_file(docs / "images.pdf", create_pdf_with_image())
    # fonts.pdf reuses simple_pdf which already has Type1 font reference
    write_file(docs / "fonts.pdf", create_simple_pdf(
        "This PDF uses a Type1 font for regression testing."
    ))

    # DOCX
    print("\nDOCX:")
    write_file(docs / "sample.docx", create_docx(
        "This is a sample DOCX document for testing Word document ingestion."
    ))
    write_file(docs / "formatted.docx", create_docx(
        "This DOCX contains an embedded image for testing image detection.",
        with_image=True,
    ))

    # ODT
    print("\nODT:")
    write_file(docs / "sample.odt", create_odt(
        "This is a sample ODT document for testing OpenDocument text ingestion."
    ))

    # RTF
    print("\nRTF:")
    write_file(docs / "sample.rtf", create_rtf(
        "This is a sample RTF document for testing Rich Text Format ingestion."
    ))

    # Slides
    print("\nSlides:")
    write_file(slides / "sample.pptx", create_pptx())
    write_file(slides / "sample.odp", create_odp())

    # EPUB
    print("\nEPUB:")
    write_file(ebooks / "sample.epub", create_epub())

    print("\nDone! All binary files generated.")


if __name__ == "__main__":
    main()
