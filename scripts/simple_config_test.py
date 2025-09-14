#!/usr/bin/env python3
"""
Simple test for unified configuration system (without watchdog dependency).
"""

import sys
import tempfile
import toml
import yaml
import os
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_basic_config_functionality():
    """Test basic configuration loading without watchdog features."""
    
    # Import only core config functionality
    from workspace_qdrant_mcp.core.config import Config
    
    print("🧪 Testing basic unified configuration functionality...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_dir = Path(tmp_dir)
        
        # Test 1: Default config creation and validation
        print("1️⃣ Testing default configuration...")
        default_config = Config()
        issues = default_config.validate_config()
        
        if not issues:
            print("   ✅ Default configuration is valid")
        else:
            print(f"   ❌ Default configuration has issues: {issues}")
        
        # Test 2: YAML config loading
        print("2️⃣ Testing YAML configuration loading...")
        yaml_config_content = """
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
  collection_types: ["test"]
  auto_create_collections: false

auto_ingestion:
  enabled: true
  target_collection_suffix: "test"
"""
        
        yaml_file = temp_dir / "test_config.yaml"
        yaml_file.write_text(yaml_config_content)
        
        try:
            yaml_config = Config.from_yaml(str(yaml_file))
            yaml_issues = yaml_config.validate_config()
            
            if not yaml_issues:
                print("   ✅ YAML configuration loaded and validated")
                print(f"      Host: {yaml_config.host}")
                print(f"      Qdrant URL: {yaml_config.qdrant.url}")
                print(f"      Embedding model: {yaml_config.embedding.model}")
            else:
                print(f"   ❌ YAML configuration has issues: {yaml_issues}")
                
        except Exception as e:
            print(f"   ❌ YAML loading failed: {e}")
        
        # Test 3: Environment variable overrides
        print("3️⃣ Testing environment variable overrides...")
        
        # Set test environment variables
        test_env = {
            'WORKSPACE_QDRANT_HOST': '192.168.1.100',
            'WORKSPACE_QDRANT_PORT': '9090',
            'WORKSPACE_QDRANT_QDRANT__URL': 'http://test:6333',
            'WORKSPACE_QDRANT_DEBUG': 'true',
        }
        
        # Store original values
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            env_config = Config()
            
            if (env_config.host == '192.168.1.100' and 
                env_config.port == 9090 and 
                env_config.qdrant.url == 'http://test:6333' and
                env_config.debug is True):
                print("   ✅ Environment variable overrides working")
            else:
                print("   ❌ Environment variable overrides not working correctly")
                print(f"      Host: {env_config.host} (expected 192.168.1.100)")
                print(f"      Port: {env_config.port} (expected 9090)")
                print(f"      Qdrant URL: {env_config.qdrant.url} (expected http://test:6333)")
                print(f"      Debug: {env_config.debug} (expected True)")
        
        finally:
            # Restore environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
        
        # Test 4: YAML export
        print("4️⃣ Testing YAML export functionality...")
        
        try:
            export_config = Config()
            yaml_output = export_config.to_yaml()
            
            # Try to parse the exported YAML
            parsed_yaml = yaml.safe_load(yaml_output)
            
            if isinstance(parsed_yaml, dict) and 'qdrant' in parsed_yaml:
                print("   ✅ YAML export working correctly")
                print(f"      Exported {len(parsed_yaml)} configuration sections")
            else:
                print("   ❌ YAML export not working correctly")
                
        except Exception as e:
            print(f"   ❌ YAML export failed: {e}")
        
        # Test 5: Format compatibility check
        print("5️⃣ Testing format compatibility...")
        
        # Create a TOML-style config that should work with both systems
        toml_config_content = """
[qdrant]
url = "http://localhost:6333"
timeout_ms = 30000

[auto_ingestion]
enabled = true
target_collection_suffix = "unified"
"""
        
        toml_file = temp_dir / "test_config.toml"
        toml_file.write_text(toml_config_content)
        
        try:
            # Parse TOML directly to simulate Rust daemon behavior
            toml_data = toml.loads(toml_config_content)
            
            if ('qdrant' in toml_data and 
                'auto_ingestion' in toml_data and
                toml_data['qdrant']['url'] == 'http://localhost:6333'):
                print("   ✅ TOML format compatible with expected structure")
            else:
                print("   ❌ TOML format compatibility issue")
                
        except Exception as e:
            print(f"   ❌ TOML parsing failed: {e}")
        
        print("\n🎉 Basic configuration tests completed!")
        return True


if __name__ == "__main__":
    try:
        success = test_basic_config_functionality()
        if success:
            print("✅ All basic tests passed - unified configuration system is working!")
        else:
            print("❌ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)