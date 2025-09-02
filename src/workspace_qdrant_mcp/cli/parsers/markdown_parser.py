
from ...observability import get_logger
logger = get_logger(__name__)
"""
Markdown document parser.

This parser handles Markdown files (.md, .markdown) with support for
extracting structured content, metadata from frontmatter, and converting
to plain text while preserving important structural information.
"""

import logging
import re
from pathlib import Path
from typing import Any, Union

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import markdown
    from markdown.extensions import codehilite, tables, toc

    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

from .base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class MarkdownParser(DocumentParser):
    """
    Parser for Markdown documents.

    Handles Markdown files with support for:
        - YAML frontmatter extraction
        - Structure analysis (headings, lists, code blocks)
        - Link and image extraction
        - Table of contents generation
        - Plain text conversion while preserving structure

    The parser can work in two modes:
        1. Simple mode: Basic regex-based parsing (no dependencies)
        2. Full mode: Complete Markdown processing (requires python-markdown)
    """

    @property
    def supported_extensions(self) -> list[str]:
        """Supported Markdown file extensions."""
        return [".md", ".markdown", ".mdown", ".mkd", ".mkdn"]

    @property
    def format_name(self) -> str:
        """Human-readable format name."""
        return "Markdown"

    async def parse(
        self,
        file_path: str | Path,
        extract_frontmatter: bool = True,
        preserve_structure: bool = True,
        include_code_blocks: bool = True,
        include_links: bool = False,
        extract_toc: bool = False,
        **options: Any,
    ) -> ParsedDocument:
        """
        Parse a Markdown file.

        Args:
            file_path: Path to the Markdown file
            extract_frontmatter: Whether to extract YAML frontmatter
            preserve_structure: Whether to preserve heading structure in text
            include_code_blocks: Whether to include code block content
            include_links: Whether to include link URLs in content
            extract_toc: Whether to generate table of contents
            **options: Additional parsing options

        Returns:
            ParsedDocument with extracted content and metadata
        """
        file_path = Path(file_path)
        self.validate_file(file_path)

        parsing_info = {"parser_mode": "full" if HAS_MARKDOWN else "simple"}

        try:
            # Read the file
            with open(file_path, encoding="utf-8", errors="replace") as f:
                raw_content = f.read()

            # Extract frontmatter if present
            frontmatter: dict[str, Any] = {}
            content = raw_content

            if extract_frontmatter and raw_content.startswith("---"):
                frontmatter, content = self._extract_frontmatter(raw_content)
                parsing_info["has_frontmatter"] = bool(frontmatter)

            # Parse Markdown content
            if HAS_MARKDOWN:
                parsed_content, structure_info = await self._parse_with_markdown_lib(
                    content,
                    preserve_structure,
                    include_code_blocks,
                    include_links,
                    extract_toc,
                )
                parsing_info.update(structure_info)
            else:
                parsed_content, structure_info = await self._parse_with_regex(
                    content, preserve_structure, include_code_blocks, include_links
                )
                parsing_info.update(structure_info)

            # Generate comprehensive metadata
            additional_metadata: dict[str, str | int | float | bool] = {
                "parser": self.format_name,
                "heading_count": structure_info.get("heading_count", 0),
                "code_block_count": structure_info.get("code_block_count", 0),
                "link_count": structure_info.get("link_count", 0),
                "image_count": structure_info.get("image_count", 0),
                "list_count": structure_info.get("list_count", 0),
                "word_count": len(parsed_content.split()) if parsed_content else 0,
                "has_frontmatter": bool(frontmatter),
            }

            # Add frontmatter fields to metadata
            if frontmatter:
                for key, value in frontmatter.items():
                    # Prefix frontmatter fields to avoid conflicts
                    if isinstance(value, str | int | float | bool):
                        additional_metadata[f"fm_{key}"] = value
                    elif isinstance(value, list) and all(
                        isinstance(x, str) for x in value
                    ):
                        additional_metadata[f"fm_{key}"] = ", ".join(value)

            return ParsedDocument.create(
                content=parsed_content,
                file_path=file_path,
                file_type="markdown",
                additional_metadata=additional_metadata,
                parsing_info=parsing_info,
            )

        except Exception as e:
            logger.error(f"Failed to parse Markdown file {file_path}: {e}")
            raise RuntimeError(f"Markdown parsing failed: {e}") from e

    def _extract_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """
        Extract YAML frontmatter from Markdown content.

        Args:
            content: Raw Markdown content

        Returns:
            Tuple of (frontmatter_dict, remaining_content)
        """
        if not content.startswith("---"):
            return {}, content

        # Find the end of frontmatter
        end_match = re.search(r"\n---\n", content)
        if not end_match:
            return {}, content

        frontmatter_yaml = content[4 : end_match.start()]  # Skip initial '---'
        remaining_content = content[end_match.end() :]

        # Parse YAML frontmatter
        frontmatter: dict[str, Any] = {}
        if HAS_YAML:
            try:
                frontmatter = yaml.safe_load(frontmatter_yaml) or {}
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse YAML frontmatter: {e}")
                # Store as raw text if YAML parsing fails
                frontmatter = {"raw_frontmatter": frontmatter_yaml}
        else:
            # Simple key-value extraction without YAML parser
            frontmatter = self._extract_simple_frontmatter(frontmatter_yaml)

        return frontmatter, remaining_content

    def _extract_simple_frontmatter(self, yaml_text: str) -> dict[str, str]:
        """Extract simple key-value pairs without YAML parser."""
        frontmatter = {}
        for line in yaml_text.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip("\"'")  # Remove quotes
                if key and value:
                    frontmatter[key] = value
        return frontmatter

    async def _parse_with_markdown_lib(
        self,
        content: str,
        preserve_structure: bool,
        include_code_blocks: bool,
        include_links: bool,
        extract_toc: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Parse Markdown using the python-markdown library."""

        # Configure markdown extensions
        extensions = ["tables", "fenced_code"]
        if extract_toc:
            extensions.append("toc")

        # Create markdown processor
        md = markdown.Markdown(extensions=extensions)

        # Convert to HTML first to analyze structure
        md.convert(content)

        # Extract structure information
        structure_info = self._analyze_markdown_structure(content)

        # Convert to plain text
        if preserve_structure:
            text_content = self._convert_to_structured_text(
                content, include_code_blocks, include_links
            )
        else:
            text_content = self._strip_markdown(
                content, include_code_blocks, include_links
            )

        # Add table of contents if requested
        if extract_toc and hasattr(md, "toc"):
            structure_info["table_of_contents"] = md.toc

        return text_content, structure_info

    async def _parse_with_regex(
        self,
        content: str,
        preserve_structure: bool,
        include_code_blocks: bool,
        include_links: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Parse Markdown using regex-based approach (fallback)."""

        structure_info = self._analyze_markdown_structure(content)

        if preserve_structure:
            text_content = self._convert_to_structured_text(
                content, include_code_blocks, include_links
            )
        else:
            text_content = self._strip_markdown(
                content, include_code_blocks, include_links
            )

        return text_content, structure_info

    def _analyze_markdown_structure(self, content: str) -> dict[str, Any]:
        """Analyze Markdown structure and extract metrics."""

        # Count headings
        heading_pattern = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
        headings = heading_pattern.findall(content)

        # Count code blocks
        code_block_pattern = re.compile(r"```[\s\S]*?```|`[^`]+`", re.MULTILINE)
        code_blocks = code_block_pattern.findall(content)

        # Count images first
        image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        images = image_pattern.findall(content)

        # Count links (excluding image links by using negative lookbehind)
        link_pattern = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
        links = link_pattern.findall(content)

        # Count lists
        list_pattern = re.compile(r"^[\s]*[-*+]\s|^[\s]*\d+\.\s", re.MULTILINE)
        list_items = list_pattern.findall(content)

        # Extract heading hierarchy
        heading_levels = []
        for heading in headings:
            level = len(heading) - len(heading.lstrip("#"))
            heading_levels.append(level)

        return {
            "heading_count": len(headings),
            "code_block_count": len(code_blocks),
            "link_count": len(links),
            "image_count": len(images),
            "list_count": len(list_items),
            "max_heading_level": max(heading_levels) if heading_levels else 0,
            "min_heading_level": min(heading_levels) if heading_levels else 0,
            "heading_levels": heading_levels,
        }

    def _convert_to_structured_text(
        self, content: str, include_code_blocks: bool, include_links: bool
    ) -> str:
        """Convert Markdown to structured plain text preserving hierarchy."""

        lines = content.split("\n")
        text_lines = []
        in_code_block = False

        for line in lines:
            original_line = line
            line = line.strip()

            # Handle code blocks
            if line.startswith("```"):
                in_code_block = not in_code_block
                if include_code_blocks:
                    text_lines.append(
                        f"[CODE BLOCK {'START' if in_code_block else 'END'}]"
                    )
                continue

            if in_code_block:
                if include_code_blocks:
                    text_lines.append(original_line)
                continue

            # Convert headings
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                title = line.lstrip("# ").strip()
                indent = "  " * (level - 1)
                text_lines.append(f"{indent}{title}")
                text_lines.append(f"{indent}{'-' * len(title)}")
                continue

            # Handle lists
            if re.match(r"^[-*+]\s", line):
                text_lines.append(f"• {line[2:]}")
                continue

            if re.match(r"^\d+\.\s", line):
                text_lines.append(line)
                continue

            # Handle links
            if include_links:
                line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", line)
            else:
                line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)

            # Handle images
            line = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"[IMAGE: \1]", line)

            # Handle inline code
            line = re.sub(r"`([^`]+)`", r"\1", line)

            # Handle bold/italic
            line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            line = re.sub(r"\*([^*]+)\*", r"\1", line)
            line = re.sub(r"__([^_]+)__", r"\1", line)
            line = re.sub(r"_([^_]+)_", r"\1", line)

            # Add cleaned line
            if line:
                text_lines.append(line)
            elif text_lines and text_lines[-1].strip():  # Preserve paragraph breaks
                text_lines.append("")

        return "\n".join(text_lines).strip()

    def _strip_markdown(
        self, content: str, include_code_blocks: bool, include_links: bool
    ) -> str:
        """Strip all Markdown formatting and return plain text."""

        # Remove code blocks
        if not include_code_blocks:
            content = re.sub(r"```[\s\S]*?```", "", content, flags=re.MULTILINE)
        else:
            # Keep code block content but remove markers
            content = re.sub(r"```[\w]*\n?([\s\S]*?)```", r"\1", content)

        # Remove inline code markers
        content = re.sub(r"`([^`]+)`", r"\1", content)

        # Handle links
        if include_links:
            content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", content)
        else:
            content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)

        # Remove images
        content = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"", content)

        # Remove headings markers
        content = re.sub(r"^#{1,6}\s+", "", content, flags=re.MULTILINE)

        # Remove bold/italic
        content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)
        content = re.sub(r"\*([^*]+)\*", r"\1", content)
        content = re.sub(r"__([^_]+)__", r"\1", content)
        content = re.sub(r"_([^_]+)_", r"\1", content)

        # Remove list markers
        content = re.sub(r"^[\s]*[-*+]\s+", "", content, flags=re.MULTILINE)
        content = re.sub(r"^[\s]*\d+\.\s+", "", content, flags=re.MULTILINE)

        # Remove horizontal rules
        content = re.sub(r"^---+\s*$", "", content, flags=re.MULTILINE)

        # Clean up excessive whitespace
        content = re.sub(r"\n\s*\n", "\n\n", content)
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content.strip()

    def get_parsing_options(self) -> dict[str, dict[str, Any]]:
        """Get available parsing options for Markdown files."""
        return {
            "extract_frontmatter": {
                "type": bool,
                "default": True,
                "description": "Whether to extract YAML frontmatter",
            },
            "preserve_structure": {
                "type": bool,
                "default": True,
                "description": "Whether to preserve heading structure in text",
            },
            "include_code_blocks": {
                "type": bool,
                "default": True,
                "description": "Whether to include code block content",
            },
            "include_links": {
                "type": bool,
                "default": False,
                "description": "Whether to include link URLs in content",
            },
            "extract_toc": {
                "type": bool,
                "default": False,
                "description": "Whether to generate table of contents (requires python-markdown)",
            },
        }
