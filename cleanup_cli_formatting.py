#!/usr/bin/env python3
"""
CLI Formatting Cleanup Script

This script removes emoji characters, Rich markup, and other visual decorations 
from CLI command files to improve terminal compatibility and accessibility.

Part of Task 49: CLI UX improvements
"""

import re
from pathlib import Path


def clean_cli_formatting(file_path: Path) -> bool:
    """Clean formatting issues from a single CLI file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Remove common emojis used in CLI output
        emoji_replacements = {
            '❌': 'Error:',
            '✅': 'Success:',
            '📄': '',
            '📁': '',
            '🚀': '',
            '💾': '',
            '🎯': '',
            '🔧': '',
            '⚡': '',
            '📊': '',
            '🔒': '',
            '📈': '',
            '📋': '',
            '🔍': '',
        }
        
        for emoji, replacement in emoji_replacements.items():
            if emoji in content:
                if replacement:
                    content = content.replace(emoji, replacement + ' ')
                else:
                    content = content.replace(emoji, '')
        
        # Remove Rich markup patterns
        rich_markup_patterns = [
            r'\[/?\w+\]',  # [red], [/red], [bold], etc.
            r'\[.*?\]',    # Any remaining bracketed markup
        ]
        
        for pattern in rich_markup_patterns:
            content = re.sub(pattern, '', content)
        
        # Clean up extra spaces and inconsistent formatting
        content = re.sub(r'\s+Error:', ' Error:', content)
        content = re.sub(r'\s+Success:', ' Success:', content)
        content = re.sub(r'Error:\s+', 'Error: ', content)
        content = re.sub(r'Success:\s+', 'Success: ', content)
        
        # Fix common patterns where emojis were at start of strings
        content = re.sub(r'print\(f"Error:\s+([^"]+)"\)', r'print(f"Error: \1")', content)
        content = re.sub(r'print\(f"Success:\s+([^"]+)"\)', r'print(f"Success: \1")', content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Cleaned: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Clean all CLI command files."""
    cli_commands_dir = Path("src/workspace_qdrant_mcp/cli/commands")
    
    if not cli_commands_dir.exists():
        print(f"CLI commands directory not found: {cli_commands_dir}")
        return
    
    print("Starting CLI formatting cleanup...")
    
    files_changed = 0
    files_processed = 0
    
    # Process all Python files in the CLI commands directory
    for py_file in cli_commands_dir.glob("*.py"):
        if py_file.name != "__init__.py":
            files_processed += 1
            if clean_cli_formatting(py_file):
                files_changed += 1
    
    print(f"\nCleanup complete:")
    print(f"  Files processed: {files_processed}")
    print(f"  Files changed: {files_changed}")
    
    if files_changed > 0:
        print("\nNext steps:")
        print("1. Review the changes with 'git diff'")
        print("2. Test CLI commands to ensure they work correctly")
        print("3. Commit the changes with 'git add . && git commit'")


if __name__ == "__main__":
    main()