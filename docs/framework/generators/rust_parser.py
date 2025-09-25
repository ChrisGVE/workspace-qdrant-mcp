"""Rust documentation parser for extracting rustdoc information."""

import json
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from .ast_parser import DocumentationNode, MemberType


class RustDocParser:
    """Parser for Rust documentation using rustdoc JSON output."""

    def __init__(self):
        """Initialize the Rust documentation parser."""
        pass

    def parse_crate(self, crate_path: Union[str, Path]) -> Optional[DocumentationNode]:
        """Parse a Rust crate and extract documentation.

        Args:
            crate_path: Path to the Rust crate (directory containing Cargo.toml)

        Returns:
            DocumentationNode representing the crate, or None if parsing fails
        """
        crate_path = Path(crate_path)
        if not (crate_path / "Cargo.toml").exists():
            raise FileNotFoundError(f"Cargo.toml not found in {crate_path}")

        try:
            # Generate rustdoc JSON output
            result = subprocess.run([
                "cargo", "doc", "--no-deps", "--output-format", "json"
            ], cwd=crate_path, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"Warning: rustdoc failed for {crate_path}: {result.stderr}")
                return None

            # Parse JSON output (simplified for now)
            # In a full implementation, this would parse the rustdoc JSON format
            return DocumentationNode(
                name=crate_path.name,
                member_type=MemberType.MODULE,
                docstring=f"Rust crate: {crate_path.name}",
                source_file=str(crate_path),
                metadata={"type": "rust_crate", "path": str(crate_path)}
            )

        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            print(f"Warning: Could not parse Rust crate {crate_path}: {e}")
            return None

    def parse_directory(self, directory_path: Union[str, Path],
                       recursive: bool = True) -> List[DocumentationNode]:
        """Parse all Rust crates in a directory.

        Args:
            directory_path: Path to the directory to parse
            recursive: Whether to parse subdirectories recursively

        Returns:
            List of DocumentationNode objects for each crate
        """
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        crates = []
        pattern = "**/Cargo.toml" if recursive else "Cargo.toml"

        for cargo_toml in directory_path.glob(pattern):
            crate_path = cargo_toml.parent
            try:
                crate_node = self.parse_crate(crate_path)
                if crate_node:
                    crates.append(crate_node)
            except Exception as e:
                print(f"Warning: Could not parse {crate_path}: {e}")
                continue

        return crates


def extract_crate_info(crate_path: Union[str, Path]) -> Optional[DocumentationNode]:
    """Convenience function to extract documentation from a Rust crate.

    Args:
        crate_path: Path to the Rust crate directory

    Returns:
        DocumentationNode for the crate, or None if parsing fails
    """
    parser = RustDocParser()
    return parser.parse_crate(crate_path)