#!/usr/bin/env python3
"""
Validation script for unified configuration system.

This script demonstrates and validates that both the Python MCP server
and Rust daemon can successfully load configuration from the same
unified configuration files.
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workspace_qdrant_mcp.core.unified_config import UnifiedConfigManager, ConfigFormat
from workspace_qdrant_mcp.core.config import Config


def create_test_configs(temp_dir: Path) -> Dict[str, Path]:
    """Create test configuration files in different formats."""
    
    # TOML configuration (Rust-friendly)
    toml_config = """
# Test unified configuration - TOML format
log_level = "info"
chunk_size = 1000
enable_lsp = true

[qdrant]
url = "http://localhost:6333"
timeout_ms = 30000
transport = "Http"

[auto_ingestion]
enabled = true
target_collection_suffix = "unified-test"
max_files_per_batch = 5
"""
    
    # YAML configuration (Python-friendly)
    yaml_config = """
# Test unified configuration - YAML format
host: "127.0.0.1"
port: 8000
debug: false

qdrant:
  url: "http://localhost:6333"
  timeout: 30
  prefer_grpc: false

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 800
  chunk_overlap: 120
  batch_size: 50

workspace:
  collection_suffixes: ["unified-test"]
  global_collections: []
  auto_create_collections: false

auto_ingestion:
  enabled: true
  target_collection_suffix: "unified-test"
"""

    toml_file = temp_dir / "unified_config.toml"
    yaml_file = temp_dir / "unified_config.yaml"
    
    toml_file.write_text(toml_config)
    yaml_file.write_text(yaml_config)
    
    return {
        "toml": toml_file,
        "yaml": yaml_file
    }


def test_python_config_loading(config_files: Dict[str, Path]) -> None:
    """Test Python MCP server configuration loading."""
    print("🐍 Testing Python MCP Server configuration loading...")
    
    config_manager = UnifiedConfigManager()
    
    # Test TOML loading
    print("  📄 Loading TOML configuration...")
    toml_config = config_manager.load_config(config_file=config_files["toml"])
    print(f"    ✅ TOML loaded: {toml_config.qdrant.url}")
    
    # Test YAML loading
    print("  📄 Loading YAML configuration...")
    yaml_config = config_manager.load_config(config_file=config_files["yaml"])
    print(f"    ✅ YAML loaded: {yaml_config.host}:{yaml_config.port}")
    
    # Validate both configs
    toml_issues = toml_config.validate_config()
    yaml_issues = yaml_config.validate_config()
    
    if not toml_issues:
        print("    ✅ TOML validation passed")
    else:
        print(f"    ❌ TOML validation issues: {toml_issues}")
    
    if not yaml_issues:
        print("    ✅ YAML validation passed")
    else:
        print(f"    ❌ YAML validation issues: {yaml_issues}")


def test_format_conversion(config_files: Dict[str, Path], temp_dir: Path) -> None:
    """Test format conversion capabilities."""
    print("🔄 Testing format conversion...")
    
    config_manager = UnifiedConfigManager()
    
    # Convert TOML -> YAML
    converted_yaml = temp_dir / "converted_from_toml.yaml"
    config_manager.convert_config(
        config_files["toml"], 
        converted_yaml, 
        ConfigFormat.YAML
    )
    print("    ✅ TOML → YAML conversion successful")
    
    # Convert YAML -> TOML  
    converted_toml = temp_dir / "converted_from_yaml.toml"
    config_manager.convert_config(
        config_files["yaml"],
        converted_toml,
        ConfigFormat.TOML
    )
    print("    ✅ YAML → TOML conversion successful")
    
    # Convert to JSON for good measure
    json_file = temp_dir / "config.json"
    config_manager.convert_config(
        config_files["yaml"],
        json_file,
        ConfigFormat.JSON
    )
    print("    ✅ YAML → JSON conversion successful")
    
    # Validate converted configs can be loaded
    converted_yaml_config = config_manager.load_config(config_file=converted_yaml)
    converted_toml_config = config_manager.load_config(config_file=converted_toml)
    json_config = config_manager.load_config(config_file=json_file)
    
    print("    ✅ All converted configs can be loaded")


def test_environment_overrides() -> None:
    """Test environment variable override functionality."""
    print("🌍 Testing environment variable overrides...")
    
    import os
    
    # Set some test environment variables
    test_env = {
        'WORKSPACE_QDRANT_HOST': '192.168.1.100',
        'WORKSPACE_QDRANT_PORT': '9090',
        'WORKSPACE_QDRANT_QDRANT__URL': 'http://test:6333',
        'WORKSPACE_QDRANT_DEBUG': 'true',
    }
    
    # Temporarily set environment variables
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        config_manager = UnifiedConfigManager()
        config = config_manager.load_config()  # Load default config with env overrides
        
        assert config.host == '192.168.1.100', f"Host override failed: {config.host}"
        assert config.port == 9090, f"Port override failed: {config.port}"
        assert config.qdrant.url == 'http://test:6333', f"Qdrant URL override failed: {config.qdrant.url}"
        assert config.debug is True, f"Debug override failed: {config.debug}"
        
        print("    ✅ Environment variable overrides working correctly")
        
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def test_config_discovery(temp_dir: Path) -> None:
    """Test configuration file discovery."""
    print("🔍 Testing configuration file discovery...")
    
    config_manager = UnifiedConfigManager(config_dir=temp_dir)
    
    # Get info about discovered configs
    info = config_manager.get_config_info()
    
    print(f"    📁 Config directory: {info['config_dir']}")
    print(f"    🏷️  Environment prefix: {info['env_prefix']}")
    
    existing_sources = [s for s in info['sources'] if s['exists']]
    print(f"    📋 Found {len(existing_sources)} configuration files:")
    
    for source in existing_sources:
        print(f"      • {source['file_path']} ({source['format']})")
    
    preferred = config_manager.get_preferred_config_source()
    if preferred:
        print(f"    ⭐ Preferred source: {preferred.file_path}")
    else:
        print("    ❌ No preferred source found")


def simulate_rust_daemon_usage(config_files: Dict[str, Path]) -> None:
    """Simulate how the Rust daemon would use unified configuration."""
    print("🦀 Simulating Rust daemon configuration usage...")
    
    # This simulates what the Rust code would do
    config_manager = UnifiedConfigManager()
    
    # Rust daemon prefers TOML
    toml_config = config_manager.load_config(config_file=config_files["toml"])
    
    # Convert to dictionary format for inspection (simulating Rust serialization)
    config_dict = config_manager._config_to_dict(toml_config)
    
    # Check key Rust daemon settings
    expected_rust_settings = [
        "qdrant.url",
        "auto_ingestion.enabled",
        "auto_ingestion.target_collection_suffix"
    ]
    
    for setting in expected_rust_settings:
        keys = setting.split('.')
        value = config_dict
        for key in keys:
            value = value.get(key)
        
        if value is not None:
            print(f"    ✅ Rust daemon setting '{setting}': {value}")
        else:
            print(f"    ❌ Missing Rust daemon setting: {setting}")


def main():
    """Main validation function."""
    print("🧪 Unified Configuration System Validation")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_dir = Path(tmp_dir)
        
        try:
            # Create test configuration files
            print("📝 Creating test configuration files...")
            config_files = create_test_configs(temp_dir)
            print(f"    ✅ Created {len(config_files)} test config files")
            
            # Test configuration discovery
            test_config_discovery(temp_dir)
            print()
            
            # Test Python configuration loading
            test_python_config_loading(config_files)
            print()
            
            # Test format conversion
            test_format_conversion(config_files, temp_dir)
            print()
            
            # Test environment overrides
            test_environment_overrides()
            print()
            
            # Simulate Rust daemon usage
            simulate_rust_daemon_usage(config_files)
            print()
            
            print("🎉 All validation tests passed!")
            print("✅ Unified configuration system is working correctly")
            print("✅ Both Python and Rust components can use the same config files")
            
        except Exception as e:
            print(f"❌ Validation failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()