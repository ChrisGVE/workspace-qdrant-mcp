#!/usr/bin/env python3
"""
Test Client Discovery Mechanism Integration

This test validates the integration between the Python service discovery client
and the daemon client to ensure project-aware daemon discovery works correctly.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Optional

from src.workspace_qdrant_mcp.core.service_discovery import (
    ServiceDiscoveryClient, 
    ServiceEndpoint, 
    DiscoveryConfig
)
from src.workspace_qdrant_mcp.core.daemon_client import DaemonClient, create_project_client
from src.workspace_qdrant_mcp.utils.project_detection import ProjectDetector


async def test_service_discovery_client():
    """Test basic service discovery client functionality."""
    print("=== Testing Service Discovery Client ===")
    
    # Create temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir)
        
        # Create discovery config with test registry
        config = DiscoveryConfig(
            registry_path=test_path / "test_registry.json",
            discovery_timeout=2.0,
            health_check_timeout=2.0
        )
        
        client = ServiceDiscoveryClient(config)
        
        # Test 1: Register a mock endpoint
        print("Testing endpoint registration...")
        endpoint = ServiceEndpoint(
            host="127.0.0.1",
            port=50051,
            project_id="test-project-123",
            service_name="test-daemon"
        )
        
        await client.register_daemon(endpoint)
        
        # Test 2: Discover the registered endpoint
        print("Testing endpoint discovery...")
        discovered = await client.discover_daemon_for_project(temp_dir)
        
        if discovered:
            print(f"✓ Discovered endpoint: {discovered.address}")
            print(f"  Project ID: {discovered.project_id}")
            print(f"  Service: {discovered.service_name}")
        else:
            print("✗ No endpoint discovered")
        
        # Test 3: List available daemons
        print("Testing daemon listing...")
        daemons = await client.list_available_daemons()
        print(f"✓ Found {len(daemons)} daemon(s)")
        
        for daemon in daemons:
            print(f"  - {daemon.address} ({daemon.project_id})")
        
        # Test 4: Registry persistence
        print("Testing registry persistence...")
        
        # Create new client to test loading
        client2 = ServiceDiscoveryClient(config)
        await client2._load_registry()
        
        if "test-project-123" in client2.endpoints_cache:
            print("✓ Registry persistence works")
        else:
            print("✗ Registry persistence failed")


async def test_daemon_client_integration():
    """Test daemon client integration with service discovery."""
    print("\n=== Testing Daemon Client Integration ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir)
        
        # Create a mock project structure
        (test_path / ".git").mkdir()
        
        # Test 1: Create project-specific client
        print("Testing project-specific client creation...")
        client = create_project_client(str(test_path))
        
        connection_info = client.get_connection_info()
        print(f"✓ Client created for project: {connection_info['project_path']}")
        print(f"  Discovery used: {connection_info['discovery_used']}")
        print(f"  Strategy: {connection_info['discovery_strategy']}")
        
        # Test 2: Project identification
        print("Testing project identification...")
        detector = ProjectDetector()
        identifier = detector.create_daemon_identifier(str(test_path))
        project_id = identifier.generate_identifier()
        
        print(f"✓ Project ID generated: {project_id}")
        
        # Test 3: Mock daemon registration and discovery
        print("Testing mock daemon discovery...")
        
        # Register a mock daemon for this project
        discovery_client = ServiceDiscoveryClient()
        mock_endpoint = ServiceEndpoint(
            host="127.0.0.1",
            port=50052,
            project_id=project_id,
            service_name="workspace-qdrant-daemon",
            health_status="healthy"
        )
        
        await discovery_client.register_daemon(mock_endpoint)
        
        # Try daemon discovery (this will fail to connect but should find endpoint)
        print("Attempting daemon connection with discovery...")
        try:
            await client.connect()
            print("✓ Connected successfully")
        except Exception as e:
            print(f"✗ Connection failed (expected): {e}")
            
            # Check if discovery was attempted
            updated_info = client.get_connection_info()
            if updated_info['discovery_used']:
                print("✓ Service discovery was used")
                print(f"  Discovered endpoint: {updated_info['endpoint']}")
                print(f"  Project ID: {updated_info['project_id']}")
            else:
                print("✗ Service discovery was not used")


async def test_multi_project_isolation():
    """Test that different projects get different daemon instances."""
    print("\n=== Testing Multi-Project Isolation ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        
        # Create two different project directories
        project1_path = base_path / "project1"
        project2_path = base_path / "project2"
        
        project1_path.mkdir()
        project2_path.mkdir()
        (project1_path / ".git").mkdir()
        (project2_path / ".git").mkdir()
        
        # Create clients for each project
        client1 = create_project_client(str(project1_path))
        client2 = create_project_client(str(project2_path))
        
        # Get project identifiers
        detector = ProjectDetector()
        id1 = detector.create_daemon_identifier(str(project1_path)).generate_identifier()
        id2 = detector.create_daemon_identifier(str(project2_path)).generate_identifier()
        
        print(f"Project 1 ID: {id1}")
        print(f"Project 2 ID: {id2}")
        
        if id1 != id2:
            print("✓ Different projects get different identifiers")
        else:
            print("✗ Projects have the same identifier")
        
        # Test connection info
        info1 = client1.get_connection_info()
        info2 = client2.get_connection_info()
        
        print(f"Client 1 project path: {info1['project_path']}")
        print(f"Client 2 project path: {info2['project_path']}")
        
        if info1['project_path'] != info2['project_path']:
            print("✓ Clients have different project paths")
        else:
            print("✗ Clients have the same project path")


async def test_discovery_fallback():
    """Test discovery fallback mechanisms."""
    print("\n=== Testing Discovery Fallback ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir)
        
        # Create client with discovery disabled
        config = DiscoveryConfig(
            network_discovery_enabled=False,
            discovery_timeout=1.0
        )
        
        discovery_client = ServiceDiscoveryClient(config)
        
        # Try to discover non-existent daemon (should fail)
        print("Testing discovery of non-existent daemon...")
        endpoint = await discovery_client.discover_daemon_for_project(str(test_path))
        
        if endpoint is None:
            print("✓ Discovery correctly returned None for non-existent daemon")
        else:
            print(f"✗ Unexpected endpoint found: {endpoint.address}")
        
        # Test daemon client fallback to configuration
        print("Testing daemon client fallback to configuration...")
        
        client = create_project_client(str(test_path))
        
        # This should attempt discovery, fail, then fallback to config
        try:
            await client.connect()
            print("✓ Connection succeeded")
        except Exception as e:
            print(f"✗ Connection failed (expected): {e}")
            
            # Verify it tried discovery first
            info = client.get_connection_info()
            print(f"  Discovery used: {info['discovery_used']}")
            print(f"  Strategy: {info['discovery_strategy']}")
            print(f"  Endpoint: {info['endpoint']}")


async def run_all_tests():
    """Run all client discovery tests."""
    print("Client Discovery Mechanism Integration Tests")
    print("=" * 50)
    
    try:
        await test_service_discovery_client()
        await test_daemon_client_integration()
        await test_multi_project_isolation()
        await test_discovery_fallback()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_tests())