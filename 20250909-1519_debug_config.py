#!/usr/bin/env python3
"""Debug script to check what configuration is actually loaded."""

import sys
from pathlib import Path

# Add the src directory to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from workspace_qdrant_mcp.core.ingestion_config import IngestionConfigManager
import yaml

def debug_config():
    """Debug the configuration loading."""
    print("🔍 Debugging ingestion configuration loading...")
    
    manager = IngestionConfigManager()
    
    # Check what config file is found
    config_path = manager._find_ingestion_config()
    print(f"Found config file: {config_path}")
    
    if config_path:
        print(f"\nReading config file: {config_path}")
        with config_path.open('r') as f:
            content = f.read()
        print(f"First 500 characters:\n{content[:500]}...")
        
        # Parse it
        config_data = yaml.safe_load(content)
        print(f"\nParsed config keys: {list(config_data.keys()) if config_data else 'None'}")
        
        if config_data and "ignore_patterns" in config_data:
            patterns = config_data["ignore_patterns"]
            print(f"Ignore patterns keys: {list(patterns.keys())}")
            if "directories" in patterns:
                dirs = patterns["directories"]
                print(f"Number of directory patterns: {len(dirs)}")
                print(f"First 5 directory patterns: {dirs[:5]}")
    
    # Load using manager
    config = manager.load_config()
    print(f"\nLoaded config - enabled: {config.enabled}")
    print(f"Ignore pattern directories: {len(config.ignore_patterns.directories)}")
    print(f"First 5 dirs: {config.ignore_patterns.directories[:5]}")
    
    # Test the pattern matching directly
    path = Path("node_modules/package/index.js")
    result = manager._matches_ignore_patterns(path)
    print(f"\nPattern matching test for 'node_modules/package/index.js': {result}")
    
    # Check each pattern type
    patterns = config.ignore_patterns
    path_str = str(path)
    path_parts = path.parts
    print(f"Path parts: {path_parts}")
    
    # Check dot files
    dot_result = patterns.dot_files and any(part.startswith('.') for part in path_parts)
    print(f"Dot files check: {dot_result}")
    
    # Check directory patterns
    for dir_pattern in patterns.directories[:10]:  # Check first 10
        match_result = any(part == dir_pattern for part in path_parts)
        if match_result:
            print(f"Directory pattern '{dir_pattern}' matches!")
            break
    else:
        print("No directory pattern matches found in first 10")

if __name__ == "__main__":
    debug_config()