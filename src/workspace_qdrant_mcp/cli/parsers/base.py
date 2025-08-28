"""
Base document parser interface and data structures.

This module defines the common interface and data structures used by all
document parsers in the workspace-qdrant-mcp ingestion system. It provides
a consistent API for extracting text content and metadata from various
file formats.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime


@dataclass
class ParsedDocument:
    """
    Container for parsed document content and metadata.
    
    Represents the result of parsing a document file, containing the extracted
    text content along with comprehensive metadata for indexing and retrieval.
    
    Attributes:
        content: Extracted text content from the document
        file_path: Original file path (for reference and deduplication)
        file_type: Document format identifier (e.g., 'pdf', 'markdown', 'text')
        metadata: Additional metadata dictionary with file-specific information
        content_hash: SHA256 hash of the content for deduplication
        parsed_at: Timestamp when parsing was completed
        file_size: Size of the original file in bytes
        parsing_info: Information about the parsing process (optional)
    """
    
    content: str
    file_path: str
    file_type: str
    metadata: Dict[str, Union[str, int, float, bool]]
    content_hash: str
    parsed_at: str
    file_size: int
    parsing_info: Optional[Dict[str, Union[str, int, float]]] = None
    
    @classmethod
    def create(
        cls,
        content: str,
        file_path: Union[str, Path],
        file_type: str,
        additional_metadata: Optional[Dict] = None,
        parsing_info: Optional[Dict] = None
    ) -> "ParsedDocument":
        """
        Create a ParsedDocument with auto-generated metadata.
        
        Args:
            content: Extracted text content
            file_path: Path to the original file
            file_type: Format identifier (e.g., 'pdf', 'markdown')
            additional_metadata: Extra metadata to include
            parsing_info: Information about the parsing process
            
        Returns:
            ParsedDocument instance with complete metadata
        """
        file_path = Path(file_path)
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Base metadata
        metadata = {
            "filename": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "content_length": len(content),
            "line_count": content.count('\n') + 1 if content else 0,
        }
        
        # Add file stats if file exists
        try:
            stat = file_path.stat()
            metadata.update({
                "file_size": stat.st_size,
                "file_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "file_created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            })
            file_size = stat.st_size
        except (OSError, FileNotFoundError):
            file_size = len(content.encode('utf-8'))  # Approximate
        
        # Add any additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return cls(
            content=content,
            file_path=str(file_path),
            file_type=file_type,
            metadata=metadata,
            content_hash=content_hash,
            parsed_at=datetime.utcnow().isoformat(),
            file_size=file_size,
            parsing_info=parsing_info
        )


class DocumentParser(ABC):
    """
    Abstract base class for document format parsers.
    
    Defines the interface that all document parsers must implement to provide
    consistent text extraction capabilities across different file formats.
    
    Each parser should:
        1. Validate that it can handle the given file format
        2. Extract text content efficiently and accurately
        3. Generate relevant metadata for the document
        4. Handle errors gracefully with informative messages
        5. Provide format-specific parsing options when needed
    """
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        List of file extensions this parser can handle.
        
        Returns:
            List of file extensions (including the dot, e.g., ['.txt', '.md'])
        """
        pass
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """
        Human-readable name of the format this parser handles.
        
        Returns:
            Format name string (e.g., 'Plain Text', 'PDF Document')
        """
        pass
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this parser can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this parser can handle the file format
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_extensions
    
    @abstractmethod
    async def parse(
        self, 
        file_path: Union[str, Path],
        **options
    ) -> ParsedDocument:
        """
        Parse a document file and extract its text content.
        
        Args:
            file_path: Path to the file to parse
            **options: Parser-specific options
            
        Returns:
            ParsedDocument with extracted content and metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
            RuntimeError: If parsing fails due to file corruption or other issues
        """
        pass
    
    def get_parsing_options(self) -> Dict[str, Dict]:
        """
        Get available parsing options for this format.
        
        Returns:
            Dictionary of option names mapped to their configuration:
            {
                "option_name": {
                    "type": type,
                    "default": default_value,
                    "description": "Description of what this option does"
                }
            }
        """
        return {}
    
    def validate_file(self, file_path: Union[str, Path]) -> None:
        """
        Validate that a file can be parsed.
        
        Args:
            file_path: Path to the file to validate
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
            RuntimeError: If file is corrupted or unreadable
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
            
        if not self.can_parse(file_path):
            raise ValueError(
                f"File format not supported by {self.format_name} parser: {file_path.suffix}"
            )