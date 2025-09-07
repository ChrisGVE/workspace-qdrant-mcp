#!/usr/bin/env python3
"""
Test script for Incremental Update Mechanism (Task #123).

This script demonstrates and tests the incremental processing system
with real file operations and change detection.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from typing import List

from src.workspace_qdrant_mcp.core.incremental_processor import (
    IncrementalProcessor,
    ProcessingPriority,
    create_incremental_processor,
)
from src.workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager


async def create_test_files(temp_dir: Path, count: int = 5) -> List[str]:
    """Create test files for incremental processing."""
    file_paths = []
    
    for i in range(count):
        file_path = temp_dir / f"test_file_{i:02d}.py"
        content = f"""# Test file {i}
def function_{i}():
    \"\"\"Test function {i}.\"\"\"
    return {i}

class TestClass_{i}:
    \"\"\"Test class {i}.\"\"\"
    
    def __init__(self):
        self.value = {i}
    
    def get_value(self):
        return self.value

# End of file {i}
"""
        file_path.write_text(content)
        file_paths.append(str(file_path))
        print(f"Created: {file_path}")
    
    return file_paths


async def modify_files(file_paths: List[str], indices: List[int]):
    """Modify specific test files."""
    for i in indices:
        if i < len(file_paths):
            file_path = Path(file_paths[i])
            content = file_path.read_text()
            
            # Add modification
            modified_content = content + f"""
# Modified at {time.time()}
def modified_function_{i}():
    return "modified_{i}"
"""
            
            time.sleep(0.1)  # Ensure different mtime
            file_path.write_text(modified_content)
            print(f"Modified: {file_path}")


async def delete_files(file_paths: List[str], indices: List[int]):
    """Delete specific test files."""
    for i in indices:
        if i < len(file_paths):
            file_path = Path(file_paths[i])
            if file_path.exists():
                file_path.unlink()
                print(f"Deleted: {file_path}")


class MockQdrantClient:
    """Mock Qdrant client for testing."""
    
    def __init__(self):
        self.operations = []
    
    def upsert(self, collection_name: str, points):
        self.operations.append(("upsert", collection_name, len(points)))
        print(f"Mock Qdrant: Upserted {len(points)} points to {collection_name}")
    
    def delete(self, collection_name: str, points_selector):
        point_count = len(points_selector.points) if hasattr(points_selector, 'points') else 1
        self.operations.append(("delete", collection_name, point_count))
        print(f"Mock Qdrant: Deleted {point_count} points from {collection_name}")
    
    def set_payload(self, collection_name: str, payload, points):
        point_count = len(points) if isinstance(points, list) else 1
        self.operations.append(("set_payload", collection_name, point_count))
        print(f"Mock Qdrant: Updated payload for {point_count} points in {collection_name}")


async def test_incremental_processing():
    """Test the complete incremental processing workflow."""
    print("=" * 60)
    print("Testing Incremental Update Mechanism (Task #123)")
    print("=" * 60)
    
    # Create temporary directory and database
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "test_state.db"
    
    try:
        print(f"Test directory: {temp_dir}")
        print(f"Database path: {db_path}")
        
        # Initialize state manager
        print("\n1. Initializing SQLite State Manager...")
        state_manager = SQLiteStateManager(str(db_path))
        success = await state_manager.initialize()
        assert success, "Failed to initialize state manager"
        print("✓ State manager initialized")
        
        # Create mock Qdrant client
        qdrant_client = MockQdrantClient()
        
        # Create incremental processor
        print("\n2. Creating Incremental Processor...")
        processor = await create_incremental_processor(
            state_manager=state_manager,
            qdrant_client=qdrant_client
        )
        print("✓ Incremental processor created and initialized")
        
        # Create initial test files
        print("\n3. Creating initial test files...")
        file_paths = await create_test_files(temp_dir, count=5)
        
        # First processing - all files should be new
        print("\n4. Processing initial files (should detect all as CREATED)...")
        priority_patterns = {
            "*_00.py": ProcessingPriority.HIGH,
            "*_01.py": ProcessingPriority.LOW,
        }
        
        result1 = await processor.process_changes(
            file_paths,
            collections=["test_collection"],
            priority_patterns=priority_patterns
        )
        
        print(f"✓ Processing result:")
        print(f"  - Processed: {len(result1.processed)} files")
        print(f"  - Failed: {len(result1.failed)} files")
        print(f"  - Skipped: {len(result1.skipped)} files")
        print(f"  - Processing time: {result1.processing_time_ms:.2f}ms")
        print(f"  - Qdrant operations: {result1.qdrant_operations}")
        
        # Wait a moment
        await asyncio.sleep(1)
        
        # Modify some files
        print("\n5. Modifying files 1 and 3...")
        await modify_files(file_paths, [1, 3])
        
        # Second processing - should only process modified files
        print("\n6. Processing changes (should detect MODIFIED files only)...")
        result2 = await processor.process_changes(
            file_paths,
            collections=["test_collection"],
            priority_patterns=priority_patterns
        )
        
        print(f"✓ Processing result:")
        print(f"  - Processed: {len(result2.processed)} files")
        print(f"  - Failed: {len(result2.failed)} files")
        print(f"  - Skipped: {len(result2.skipped)} files")
        print(f"  - Processing time: {result2.processing_time_ms:.2f}ms")
        print(f"  - Qdrant operations: {result2.qdrant_operations}")
        
        # Delete a file
        print("\n7. Deleting file 2...")
        await delete_files(file_paths, [2])
        
        # Third processing - should detect deletion
        print("\n8. Processing changes (should detect DELETED file)...")
        result3 = await processor.process_changes(
            file_paths,
            collections=["test_collection"]
        )
        
        print(f"✓ Processing result:")
        print(f"  - Processed: {len(result3.processed)} files")
        print(f"  - Failed: {len(result3.failed)} files")
        print(f"  - Skipped: {len(result3.skipped)} files")
        print(f"  - Processing time: {result3.processing_time_ms:.2f}ms")
        print(f"  - Qdrant operations: {result3.qdrant_operations}")
        
        # Test no changes
        print("\n9. Processing without changes (should skip all)...")
        result4 = await processor.process_changes(
            [fp for fp in file_paths if Path(fp).exists()],  # Only existing files
            collections=["test_collection"]
        )
        
        print(f"✓ Processing result:")
        print(f"  - Processed: {len(result4.processed)} files")
        print(f"  - Failed: {len(result4.failed)} files")
        print(f"  - Skipped: {len(result4.skipped)} files")
        print(f"  - Processing time: {result4.processing_time_ms:.2f}ms")
        
        # Get processing statistics
        print("\n10. Processing Statistics...")
        stats = await processor.get_processing_statistics()
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
        print("\n11. Qdrant Operations Summary:")
        for op_type, collection, count in qdrant_client.operations:
            print(f"  - {op_type}: {count} points in {collection}")
        
        print("\n" + "=" * 60)
        print("✓ Incremental processing test completed successfully!")
        print("=" * 60)
        
        # Close state manager
        await state_manager.close()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\nCleaned up: {temp_dir}")
        except Exception as e:
            print(f"Warning: Failed to cleanup {temp_dir}: {e}")


async def test_performance_benchmark():
    """Test performance with larger file sets."""
    print("\n" + "=" * 60)
    print("Performance Benchmark Test")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "perf_test.db"
    
    try:
        # Initialize components
        state_manager = SQLiteStateManager(str(db_path))
        await state_manager.initialize()
        
        qdrant_client = MockQdrantClient()
        processor = await create_incremental_processor(
            state_manager=state_manager,
            qdrant_client=qdrant_client
        )
        
        # Create many test files
        print("Creating 50 test files...")
        file_paths = []
        start_time = time.time()
        
        for i in range(50):
            file_path = temp_dir / f"perf_test_{i:03d}.py"
            content = f"# Performance test file {i}\n" * 20
            file_path.write_text(content)
            file_paths.append(str(file_path))
        
        creation_time = time.time() - start_time
        print(f"✓ Created 50 files in {creation_time:.2f}s")
        
        # Process all files
        print("\nProcessing all files...")
        start_time = time.time()
        
        result = await processor.process_changes(
            file_paths,
            batch_size=20
        )
        
        processing_time = time.time() - start_time
        print(f"✓ Processed {len(result.processed)} files in {processing_time:.2f}s")
        print(f"  - Processing rate: {len(result.processed) / processing_time:.1f} files/sec")
        
        # Modify subset and test incremental performance
        print("\nModifying 10 files...")
        await modify_files(file_paths, list(range(5, 15)))
        
        print("Processing incremental changes...")
        start_time = time.time()
        
        result = await processor.process_changes(
            file_paths,
            batch_size=20
        )
        
        incremental_time = time.time() - start_time
        print(f"✓ Incremental processing: {len(result.processed)} processed, {len(result.skipped)} skipped")
        print(f"  - Incremental time: {incremental_time:.2f}s")
        print(f"  - Speedup: {processing_time / max(incremental_time, 0.001):.1f}x faster")
        
        await state_manager.close()
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


async def main():
    """Run all incremental processor tests."""
    print("Incremental Update Mechanism Test Suite")
    print("Task #123 - Real-time Development Workflow Optimization")
    
    await test_incremental_processing()
    await test_performance_benchmark()
    
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())