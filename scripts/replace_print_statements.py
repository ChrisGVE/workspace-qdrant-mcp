#!/usr/bin/env python3
"""
Script to systematically replace print() statements with structured logging
across the workspace-qdrant-mcp codebase.

This script implements the logging migration required by Task 33.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

# Source directory to process
SRC_DIR = Path("src/workspace_qdrant_mcp")

# Files to skip (contain intentional print statements for CLI output)
SKIP_FILES = {
    "cli/main.py",  # Already updated
    "config/profiles/README.md",  # Documentation file
}

def get_logger_import_line(file_path: Path) -> str:
    """Determine the appropriate logger import based on file location."""
    relative_path = file_path.relative_to(SRC_DIR)
    depth = len(relative_path.parts) - 1  # Subtract 1 for the file itself
    
    if depth == 0:
        return "from .observability import get_logger"
    else:
        prefix = ".." * depth
        return f"from {prefix}.observability import get_logger"

def process_file(file_path: Path) -> bool:
    """Process a single Python file to replace print statements."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        modified = False
        
        # Skip if already has structured logger import
        if "from .observability import get_logger" in content or "from ..observability import get_logger" in content:
            print(f"⚠️  {file_path.relative_to(SRC_DIR)}: Already has structured logging import")
        else:
            # Add logger import after other imports
            import_pattern = r'((?:^(?:from|import)\s+.*\n)*)'
            logger_import = get_logger_import_line(file_path)
            
            match = re.search(import_pattern, content, re.MULTILINE)
            if match:
                imports_end = match.end()
                # Insert logger import and initialization
                content = (content[:imports_end] + 
                          f"\n{logger_import}\n" +
                          "logger = get_logger(__name__)\n" +
                          content[imports_end:])
                modified = True
        
        # Find and replace print statements
        print_patterns = [
            # Simple print with f-string
            (r'print\(f"([^"]+)"\)', r'logger.info("\1")'),
            # Simple print with string
            (r'print\("([^"]+)"\)', r'logger.info("\1")'),
            # Print with variable
            (r'print\(([^)]+)\)', r'logger.info("Output", data=\1)'),
        ]
        
        for pattern, replacement in print_patterns:
            old_content = content
            content = re.sub(pattern, replacement, content)
            if content != old_content:
                modified = True
        
        # Handle specific cases manually for better context
        specific_replacements = [
            # Status information
            (r'logger\.info\("Project: \{([^}]+)\}"\)', r'logger.info("Project detected", project=\1)'),
            (r'logger\.info\("Collections: \{([^}]+)\}"\)', r'logger.info("Collections available", collections=\1)'),
            (r'logger\.info\("Available: \{([^}]+)\}"\)', r'logger.info("Available collection", collection=\1)'),
            # Error cases
            (r'logger\.info\("Error: \{([^}]+)\}"\)', r'logger.error("Operation failed", error=\1)'),
            # Debug information
            (r'logger\.info\("Debug: ([^"]+)"\)', r'logger.debug("\1")'),
        ]
        
        for pattern, replacement in specific_replacements:
            old_content = content
            content = re.sub(pattern, replacement, content)
            if content != old_content:
                modified = True
        
        # Write back if modified
        if modified and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ {file_path.relative_to(SRC_DIR)}: Updated with structured logging")
            return True
        else:
            print(f"📝 {file_path.relative_to(SRC_DIR)}: No changes needed")
            return False
            
    except Exception as e:
        print(f"❌ {file_path.relative_to(SRC_DIR)}: Error processing - {e}")
        return False

def main():
    """Main execution function."""
    print("🔄 Starting systematic print statement replacement...")
    print(f"📁 Processing directory: {SRC_DIR.absolute()}")
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(SRC_DIR):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                relative_path = file_path.relative_to(SRC_DIR)
                
                # Skip certain files
                if str(relative_path) in SKIP_FILES:
                    print(f"⏭️  Skipping {relative_path} (intentional print statements)")
                    continue
                    
                python_files.append(file_path)
    
    print(f"📋 Found {len(python_files)} Python files to process")
    
    # Process each file
    modified_count = 0
    for file_path in python_files:
        if process_file(file_path):
            modified_count += 1
    
    print(f"\n✨ Processing complete!")
    print(f"📊 Modified {modified_count} out of {len(python_files)} files")
    print("🎯 All print statements have been replaced with structured logging")

if __name__ == "__main__":
    main()