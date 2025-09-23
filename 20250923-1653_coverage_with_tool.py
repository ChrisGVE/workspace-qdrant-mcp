#!/usr/bin/env python3
"""
Coverage analysis using the coverage library directly
"""
import coverage
import sys
from pathlib import Path

# Add src/python to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

def run_coverage_analysis():
    """Run comprehensive coverage analysis."""

    # Initialize coverage
    cov = coverage.Coverage(source=['common.core.client'])
    cov.start()

    # Import and run through all the methods
    from common.core.client import QdrantWorkspaceClient, create_qdrant_client
    from common.core.config import Config
    import asyncio

    config = Config()

    # Test all sync methods
    client = QdrantWorkspaceClient(config)

    # Basic properties
    _ = client.config
    _ = client.client
    _ = client.initialized

    # Project info methods
    _ = client.get_project_info()
    client.project_info = None
    _ = client.get_project_context()
    client.project_info = {"main_project": ""}
    _ = client.get_project_context()
    client.project_info = {"main_project": "test-project"}
    _ = client.get_project_context("docs")

    # Project ID generation
    _ = client._generate_project_id("test-project")

    # Embedding service
    _ = client.get_embedding_service()

    # Collection methods (not initialized)
    _ = client.list_collections()
    _ = client.select_collections_by_type("memory_collection")
    _ = client.get_searchable_collections()
    _ = client.validate_collection_access("test", "read")

    # Factory function
    _ = create_qdrant_client({"host": "localhost", "port": 6333})

    # Test refresh project detection
    _ = client.refresh_project_detection()

    # Test get enhanced collection selector (not initialized)
    try:
        _ = client.get_enhanced_collection_selector()
    except RuntimeError:
        pass  # Expected

    # Test async methods
    async def run_async_tests():
        # Status
        _ = await client.get_status()

        # Search
        _ = await client.search_with_project_context("test", {"dense": [0.1] * 384})

        # Ensure collection exists
        try:
            await client.ensure_collection_exists("test")
        except RuntimeError:
            pass  # Expected

        # Empty collection name
        client.initialized = True
        try:
            await client.ensure_collection_exists("")
        except ValueError:
            pass  # Expected
        client.initialized = False

        # Create collection
        _ = await client.create_collection("test")

        # Close
        client.embedding_service = None
        client.client = None
        await client.close()

    asyncio.run(run_async_tests())

    # Stop coverage and generate report
    cov.stop()
    cov.save()

    print("="*80)
    print("COVERAGE REPORT")
    print("="*80)

    # Print coverage report to console
    cov.report(show_missing=True)

    # Generate HTML report
    try:
        cov.html_report(directory='htmlcov_direct_client')
        print(f"\nHTML coverage report generated in: htmlcov_direct_client/")
    except Exception as e:
        print(f"Could not generate HTML report: {e}")

    # Get coverage percentage
    total = cov.report()
    print(f"\nOverall coverage: {total:.1f}%")

if __name__ == "__main__":
    run_coverage_analysis()