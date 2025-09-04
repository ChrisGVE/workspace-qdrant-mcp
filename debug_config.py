#!/usr/bin/env python3
"""Debug script to trace configuration loading and collection creation."""

import sys
import os
sys.path.append('src')

from workspace_qdrant_mcp.core.config import Config
from workspace_qdrant_mcp.utils.project_detection import ProjectDetector

def debug_configuration():
    print("=== Configuration Debug ===")
    
    # Test 1: Default config (no file)
    print("\n1. Default Configuration (no config file):")
    config_default = Config()
    print(f"  Collection suffixes: {config_default.workspace.collection_suffixes}")
    print(f"  Effective collection suffixes: {config_default.workspace.effective_collection_suffixes}")
    print(f"  Auto create collections: {config_default.workspace.auto_create_collections}")
    print(f"  Target collection suffix: '{config_default.auto_ingestion.target_collection_suffix}'")
    
    # Test 2: With YAML config
    print("\n2. With YAML Configuration:")
    try:
        config_yaml = Config(config_file="workspace_qdrant_config.yaml")
        print(f"  Collection suffixes: {config_yaml.workspace.collection_suffixes}")
        print(f"  Effective collection suffixes: {config_yaml.workspace.effective_collection_suffixes}")
        print(f"  Auto create collections: {config_yaml.workspace.auto_create_collections}")
        print(f"  Target collection suffix: '{config_yaml.auto_ingestion.target_collection_suffix}'")
    except Exception as e:
        print(f"  Error loading YAML config: {e}")
    
    # Test 3: Project detection
    print("\n3. Project Detection:")
    detector = ProjectDetector()
    project_info = detector.get_project_info()
    project_name = project_info['main_project']
    print(f"  Project name: {project_name}")
    
    # Test 4: Expected collection names
    print("\n4. Expected Collection Names:")
    if 'config_yaml' in locals():
        config = config_yaml
        print(f"  Using YAML config")
    else:
        config = config_default
        print(f"  Using default config")
    
    if config.workspace.auto_create_collections:
        print(f"  Auto-create enabled, would create:")
        for suffix in config.workspace.effective_collection_suffixes:
            collection_name = f"{project_name}-{suffix}"
            print(f"    - {collection_name}")
        
        target_suffix = config.auto_ingestion.target_collection_suffix
        target_collection = f"{project_name}-{target_suffix}" if target_suffix else "NO TARGET"
        print(f"  Target collection for auto-ingestion: {target_collection}")
    else:
        print(f"  Auto-create disabled, no collections would be created")

if __name__ == "__main__":
    debug_configuration()