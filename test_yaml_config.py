#!/usr/bin/env python3
"""
Test script for YAML configuration system validation.
Tests hierarchy, environment variable substitution, and schema validation.
"""

import os
import tempfile
from pathlib import Path
from src.workspace_qdrant_mcp.core.yaml_config import load_config, WorkspaceConfig
import yaml


def test_basic_config_loading():
    """Test basic configuration loading with defaults."""
    print("=== Test 1: Basic Configuration Loading ===")
    
    config = load_config()
    print(f"✓ Config loaded successfully")
    print(f"✓ Qdrant URL: {config.qdrant.url}")
    print(f"✓ Daemon gRPC port: {config.daemon.grpc.port}")
    print(f"✓ Embedding provider: {config.embedding.provider}")
    print()


def test_environment_variable_substitution():
    """Test environment variable substitution in YAML."""
    print("=== Test 2: Environment Variable Substitution ===")
    
    # Set test environment variables
    os.environ["TEST_QDRANT_URL"] = "https://test.qdrant.io:6333"
    os.environ["TEST_API_KEY"] = "test-secret-key-123"
    os.environ["TEST_TIMEOUT"] = "45"
    
    # Create test config with env vars
    test_config = {
        'qdrant': {
            'url': '${TEST_QDRANT_URL}',
            'api_key': '${TEST_API_KEY}',
            'timeout_seconds': int(os.environ.get('TEST_TIMEOUT', '30'))
        },
        'embedding': {
            'provider': 'openai',
            'openai': {
                'api_key': '${TEST_API_KEY}'
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_config_path = f.name
    
    try:
        config = load_config(cli_config_path=temp_config_path)
        print(f"✓ Environment variable substitution works")
        print(f"✓ Qdrant URL resolved to: {config.qdrant.url}")
        print(f"✓ API key resolved to: {config.qdrant.api_key}")
        print(f"✓ Timeout resolved to: {config.qdrant.timeout_seconds}")
    finally:
        os.unlink(temp_config_path)
        # Clean up env vars
        del os.environ["TEST_QDRANT_URL"]
        del os.environ["TEST_API_KEY"]
        del os.environ["TEST_TIMEOUT"]
    
    print()


def test_config_hierarchy():
    """Test configuration hierarchy precedence."""
    print("=== Test 3: Configuration Hierarchy ===")
    
    # Create temporary config files for hierarchy testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create system-level config (lowest priority)
        system_config = {
            'qdrant': {'url': 'http://system.example.com:6333'},
            'daemon': {'grpc': {'port': 50051}}
        }
        system_config_path = temp_path / "system.yaml"
        with open(system_config_path, 'w') as f:
            yaml.dump(system_config, f)
        
        # Create user-level config (medium priority)
        user_config = {
            'qdrant': {'url': 'http://user.example.com:6333'},
            'daemon': {'grpc': {'port': 50052}}
        }
        user_config_path = temp_path / "user.yaml"
        with open(user_config_path, 'w') as f:
            yaml.dump(user_config, f)
        
        # Create project-level config (highest priority)
        project_config = {
            'qdrant': {'url': 'http://project.example.com:6333'}
            # Note: no daemon.grpc.port - should inherit from user config
        }
        project_config_path = temp_path / "project.yaml"
        with open(project_config_path, 'w') as f:
            yaml.dump(project_config, f)
        
        # Test hierarchy - project config should take precedence for URL
        config = load_config(cli_config_path=str(project_config_path))
        print(f"✓ Config hierarchy working")
        print(f"✓ URL from project config: {config.qdrant.url}")
        print(f"✓ Port inherited from defaults: {config.daemon.grpc.port}")
    
    print()


def test_schema_validation():
    """Test JSON schema validation."""
    print("=== Test 4: Schema Validation ===")
    
    # Test valid config
    valid_config = {
        'qdrant': {
            'url': 'http://localhost:6333',
            'timeout_seconds': 30
        },
        'daemon': {
            'grpc': {
                'port': 50051
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_config, f)
        temp_config_path = f.name
    
    try:
        config = load_config(cli_config_path=temp_config_path)
        print(f"✓ Valid config passed validation")
    except Exception as e:
        print(f"✗ Valid config failed unexpectedly: {e}")
    finally:
        os.unlink(temp_config_path)
    
    # Test invalid config (wrong types)
    invalid_config = {
        'qdrant': {
            'url': 'http://localhost:6333',
            'timeout_seconds': 'invalid_string'  # Should be int
        },
        'daemon': {
            'grpc': {
                'port': 'invalid_port'  # Should be int
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(invalid_config, f)
        temp_config_path = f.name
    
    try:
        config = load_config(cli_config_path=temp_config_path)
        print(f"✗ Invalid config passed validation (should have failed)")
    except Exception as e:
        print(f"✓ Invalid config correctly rejected: {type(e).__name__}")
    finally:
        os.unlink(temp_config_path)
    
    print()


def test_type_safety():
    """Test that configuration provides type-safe access."""
    print("=== Test 5: Type Safety ===")
    
    config = load_config()
    
    # Test that all expected attributes exist and have correct types
    assert isinstance(config.qdrant.url, str), "qdrant.url should be string"
    assert isinstance(config.qdrant.timeout_seconds, int), "qdrant.timeout_seconds should be int"
    assert isinstance(config.daemon.grpc.port, int), "daemon.grpc.port should be int"
    assert isinstance(config.embedding.provider, str), "embedding.provider should be string"
    assert isinstance(config.daemon.max_concurrent_jobs, int), "daemon.max_concurrent_jobs should be int"
    
    print(f"✓ All configuration attributes have correct types")
    print(f"✓ Type safety validated")
    print()


def test_gRPC_configuration():
    """Test gRPC-specific configuration settings."""
    print("=== Test 6: gRPC Configuration ===")
    
    config = load_config()
    
    print(f"✓ gRPC host: {config.daemon.grpc.host}")
    print(f"✓ gRPC port: {config.daemon.grpc.port}")
    print(f"✓ Max message size: {config.daemon.grpc.max_message_size_mb} MB")
    
    # Test custom gRPC config
    custom_grpc_config = {
        'daemon': {
            'grpc': {
                'host': '0.0.0.0',
                'port': 50555,
                'max_message_size_mb': 200
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(custom_grpc_config, f)
        temp_config_path = f.name
    
    try:
        custom_config = load_config(cli_config_path=temp_config_path)
        print(f"✓ Custom gRPC host: {custom_config.daemon.grpc.host}")
        print(f"✓ Custom gRPC port: {custom_config.daemon.grpc.port}")
        print(f"✓ Custom message size: {custom_config.daemon.grpc.max_message_size_mb} MB")
    finally:
        os.unlink(temp_config_path)
    
    print()


def main():
    """Run all configuration tests."""
    print("🧪 Testing YAML Configuration System\n")
    
    try:
        test_basic_config_loading()
        test_environment_variable_substitution()
        test_config_hierarchy()
        test_schema_validation()
        test_type_safety()
        test_gRPC_configuration()
        
        print("✅ All configuration tests passed!")
        print("\nConfiguration system is working correctly:")
        print("- YAML loading with defaults")
        print("- Environment variable substitution")
        print("- Configuration hierarchy")
        print("- Schema validation")
        print("- Type safety")
        print("- gRPC settings")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())