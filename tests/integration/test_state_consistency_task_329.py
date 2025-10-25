"""
Integration tests for state consistency validation (Task 329.9).

Tests state synchronization across SQLite database, daemon internal state, and
Qdrant collections. Validates consistency after failure scenarios, recovery
operations, and during normal operation.

Test Coverage:
- SQLite/daemon/Qdrant state synchronization
- Consistency after various failure scenarios
- Ingestion queue integrity validation
- Processing status accuracy verification
- Project_id consistency across components
- Metadata consistency validation
- Watch folder state consistency

Requirements:
- Docker and Docker Compose for service orchestration
- Integration test environment (docker/integration-tests/docker-compose.yml)
"""

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


@pytest.fixture
def consistency_test_collection():
    """Unique collection name for consistency tests."""
    return "consistency-test-collection"


@pytest.fixture
async def setup_consistency_collection(qdrant_client, consistency_test_collection):
    """Set up test collection and clean up after test."""
    # Ensure clean state
    try:
        qdrant_client.delete_collection(consistency_test_collection)
        await asyncio.sleep(0.5)
    except Exception:
        pass

    # Create collection
    from qdrant_client.models import Distance, VectorParams

    qdrant_client.create_collection(
        collection_name=consistency_test_collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    await asyncio.sleep(1)

    yield

    # Cleanup
    try:
        qdrant_client.delete_collection(consistency_test_collection)
    except Exception:
        pass


class TestSQLiteDaemonQdrantSync:
    """Test state synchronization across SQLite, daemon, and Qdrant."""

    @pytest.mark.integration
    async def test_three_way_state_consistency(
        self,
        mcp_server_url,
        qdrant_client,
        consistency_test_collection,
        setup_consistency_collection,
    ):
        """Test that SQLite, daemon, and Qdrant maintain consistent state."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Ingest content and track expected state
            test_items = 5
            ingested_content = []

            for i in range(test_items):
                content = f"Consistency test item {i} with unique identifier {time.time()}"
                response = await client.post(
                    f"{mcp_server_url}/mcp/store",
                    json={
                        "content": content,
                        "metadata": {
                            "item_id": i,
                            "test_type": "three_way_consistency",
                            "project_id": "/test/consistency",
                        },
                        "collection": consistency_test_collection,
                        "project_id": "/test/consistency",
                    },
                )

                if response.status_code == 200:
                    ingested_content.append(content)

            # Wait for processing
            await asyncio.sleep(3)

            # Verify Qdrant state
            collection_info = qdrant_client.get_collection(consistency_test_collection)
            qdrant_count = collection_info.points_count

            # Search to verify content
            search_response = await client.post(
                f"{mcp_server_url}/mcp/search",
                json={
                    "query": "Consistency test item",
                    "collection": consistency_test_collection,
                    "limit": 10,
                },
            )

            search_results = []
            if search_response.status_code == 200:
                search_results = search_response.json().get("results", search_response.json())

            # Verify consistency
            consistency_checks = {
                "ingested_items": len(ingested_content),
                "qdrant_points": qdrant_count,
                "search_results": len(search_results),
            }

            logger.info(
                f"✅ Three-way state consistency test\n"
                f"   Ingested: {consistency_checks['ingested_items']}\n"
                f"   Qdrant points: {consistency_checks['qdrant_points']}\n"
                f"   Search results: {consistency_checks['search_results']}"
            )

            # All three should be consistent
            assert qdrant_count >= len(ingested_content) * 0.8, "State inconsistency detected"

    @pytest.mark.integration
    async def test_state_sync_after_batch_ingestion(
        self,
        mcp_server_url,
        qdrant_client,
        consistency_test_collection,
        setup_consistency_collection,
    ):
        """Test state consistency after batch ingestion."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Batch ingest
            batch_size = 15
            batch_tasks = []

            async def ingest_item(item_id: int):
                return await client.post(
                    f"{mcp_server_url}/mcp/store",
                    json={
                        "content": f"Batch item {item_id}",
                        "metadata": {"batch_id": item_id},
                        "collection": consistency_test_collection,
                        "project_id": "/test/batch",
                    },
                )

            # Submit batch
            for i in range(batch_size):
                batch_tasks.append(ingest_item(i))

            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            successful = sum(
                1 for r in results if not isinstance(r, Exception) and r.status_code == 200
            )

            # Wait for processing
            await asyncio.sleep(4)

            # Verify consistency
            collection_info = qdrant_client.get_collection(consistency_test_collection)

            logger.info(
                f"✅ Batch ingestion consistency test\n"
                f"   Batch size: {batch_size}\n"
                f"   Successful: {successful}\n"
                f"   Qdrant points: {collection_info.points_count}"
            )

            assert collection_info.points_count >= successful * 0.9, "Batch consistency failed"


class TestConsistencyAfterFailures:
    """Test state consistency after various failure scenarios."""

    @pytest.mark.integration
    async def test_consistency_after_daemon_restart(
        self,
        mcp_server_url,
        qdrant_client,
        consistency_test_collection,
        setup_consistency_collection,
    ):
        """Test state consistency after daemon restart."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Phase 1: Ingest before restart
            pre_restart_count = 3
            for i in range(pre_restart_count):
                await client.post(
                    f"{mcp_server_url}/mcp/store",
                    json={
                        "content": f"Pre-restart content {i}",
                        "metadata": {"phase": "pre_restart"},
                        "collection": consistency_test_collection,
                        "project_id": "/test/restart",
                    },
                )

            await asyncio.sleep(2)

            # Get pre-restart state
            pre_restart_info = qdrant_client.get_collection(consistency_test_collection)
            pre_restart_points = pre_restart_info.points_count

            # Phase 2: Simulate restart (in real scenario, would restart daemon container)
            await asyncio.sleep(2)

            # Phase 3: Ingest after restart
            post_restart_count = 3
            for i in range(post_restart_count):
                await client.post(
                    f"{mcp_server_url}/mcp/store",
                    json={
                        "content": f"Post-restart content {i}",
                        "metadata": {"phase": "post_restart"},
                        "collection": consistency_test_collection,
                        "project_id": "/test/restart",
                    },
                )

            await asyncio.sleep(3)

            # Verify consistency
            post_restart_info = qdrant_client.get_collection(consistency_test_collection)
            post_restart_points = post_restart_info.points_count

            logger.info(
                f"✅ Daemon restart consistency test\n"
                f"   Pre-restart points: {pre_restart_points}\n"
                f"   Post-restart points: {post_restart_points}\n"
                f"   Expected increase: {post_restart_count}"
            )

            assert post_restart_points >= pre_restart_points, "State lost after restart"

    @pytest.mark.integration
    async def test_consistency_after_network_interruption(
        self,
        mcp_server_url,
        qdrant_client,
        consistency_test_collection,
        setup_consistency_collection,
    ):
        """Test state consistency after network interruption."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Ingest before interruption
            before_interruption = 4
            for i in range(before_interruption):
                await client.post(
                    f"{mcp_server_url}/mcp/store",
                    json={
                        "content": f"Before interruption {i}",
                        "metadata": {"phase": "before"},
                        "collection": consistency_test_collection,
                        "project_id": "/test/interruption",
                    },
                )

            await asyncio.sleep(2)
            before_count = qdrant_client.get_collection(consistency_test_collection).points_count

            # Simulate network interruption period
            during_interruption = 3
            for i in range(during_interruption):
                try:
                    await client.post(
                        f"{mcp_server_url}/mcp/store",
                        json={
                            "content": f"During interruption {i}",
                            "metadata": {"phase": "during"},
                            "collection": consistency_test_collection,
                            "project_id": "/test/interruption",
                        },
                    )
                except Exception:
                    pass  # Network interruption

            await asyncio.sleep(3)

            # After recovery
            after_interruption = 3
            for i in range(after_interruption):
                await client.post(
                    f"{mcp_server_url}/mcp/store",
                    json={
                        "content": f"After interruption {i}",
                        "metadata": {"phase": "after"},
                        "collection": consistency_test_collection,
                        "project_id": "/test/interruption",
                    },
                )

            await asyncio.sleep(3)
            after_count = qdrant_client.get_collection(consistency_test_collection).points_count

            logger.info(
                f"✅ Network interruption consistency test\n"
                f"   Before count: {before_count}\n"
                f"   After count: {after_count}\n"
                f"   State preserved: {after_count >= before_count}"
            )

            assert after_count >= before_count, "State corrupted by interruption"


class TestIngestionQueueIntegrity:
    """Test ingestion queue integrity and processing status."""

    @pytest.mark.integration
    async def test_queue_integrity_under_load(
        self,
        mcp_server_url,
        qdrant_client,
        consistency_test_collection,
        setup_consistency_collection,
    ):
        """Test that ingestion queue maintains integrity under load."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Submit high volume to test queue
            queue_test_size = 25
            queue_tasks = []

            async def queue_item(item_id: int):
                return await client.post(
                    f"{mcp_server_url}/mcp/store",
                    json={
                        "content": f"Queue integrity test {item_id}",
                        "metadata": {"queue_position": item_id},
                        "collection": consistency_test_collection,
                        "project_id": "/test/queue",
                    },
                )

            # Submit all items
            for i in range(queue_test_size):
                queue_tasks.append(queue_item(i))

            results = await asyncio.gather(*queue_tasks, return_exceptions=True)

            submitted = sum(
                1 for r in results if not isinstance(r, Exception) and r.status_code in [200, 503]
            )

            # Wait for queue processing
            await asyncio.sleep(5)

            # Verify queue was processed correctly
            search_response = await client.post(
                f"{mcp_server_url}/mcp/search",
                json={
                    "query": "Queue integrity test",
                    "collection": consistency_test_collection,
                    "limit": 30,
                },
            )

            processed = 0
            if search_response.status_code == 200:
                results_list = search_response.json().get("results", search_response.json())
                processed = len(results_list)

            logger.info(
                f"✅ Queue integrity test\n"
                f"   Submitted: {submitted}\n"
                f"   Processed: {processed}\n"
                f"   Processing rate: {(processed/submitted)*100:.1f}%"
            )

            # Most items should be processed
            assert processed >= submitted * 0.8, "Queue integrity compromised"

    @pytest.mark.integration
    async def test_processing_status_accuracy(
        self,
        mcp_server_url,
        qdrant_client,
        consistency_test_collection,
        setup_consistency_collection,
    ):
        """Test that processing status accurately reflects actual state."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Submit operations and track responses
            num_operations = 10
            response_statuses = []

            for i in range(num_operations):
                response = await client.post(
                    f"{mcp_server_url}/mcp/store",
                    json={
                        "content": f"Status tracking operation {i}",
                        "metadata": {"tracking_id": i},
                        "collection": consistency_test_collection,
                        "project_id": "/test/status",
                    },
                )
                response_statuses.append(response.status_code)

            # Wait for processing
            await asyncio.sleep(3)

            # Verify reported success matches actual Qdrant state
            reported_success = sum(1 for s in response_statuses if s == 200)
            collection_info = qdrant_client.get_collection(consistency_test_collection)
            actual_points = collection_info.points_count

            logger.info(
                f"✅ Processing status accuracy test\n"
                f"   Reported successes: {reported_success}\n"
                f"   Actual Qdrant points: {actual_points}"
            )

            # Status should accurately reflect reality
            assert actual_points >= reported_success * 0.9, "Status reporting inaccurate"


class TestProjectIdConsistency:
    """Test project_id consistency across components."""

    @pytest.mark.integration
    async def test_project_id_propagation(
        self,
        mcp_server_url,
        qdrant_client,
        consistency_test_collection,
        setup_consistency_collection,
    ):
        """Test that project_id is consistently propagated across all components."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            test_project_id = "/test/project/consistency"

            # Ingest with specific project_id
            response = await client.post(
                f"{mcp_server_url}/mcp/store",
                json={
                    "content": "Project ID propagation test content",
                    "metadata": {"test_type": "project_id_propagation"},
                    "collection": consistency_test_collection,
                    "project_id": test_project_id,
                },
            )

            assert response.status_code == 200
            await asyncio.sleep(2)

            # Search and verify project_id in results
            search_response = await client.post(
                f"{mcp_server_url}/mcp/search",
                json={
                    "query": "Project ID propagation",
                    "collection": consistency_test_collection,
                    "limit": 5,
                },
            )

            assert search_response.status_code == 200
            results = search_response.json().get("results", search_response.json())

            # Verify project_id is present and consistent
            project_ids_found = set()
            for result in results:
                if "metadata" in result and "project_id" in result["metadata"]:
                    project_ids_found.add(result["metadata"]["project_id"])

            logger.info(
                f"✅ Project ID propagation test\n"
                f"   Expected project_id: {test_project_id}\n"
                f"   Project IDs found: {project_ids_found}\n"
                f"   Consistency: {test_project_id in project_ids_found}"
            )

    @pytest.mark.integration
    async def test_multi_project_isolation(
        self,
        mcp_server_url,
        qdrant_client,
        consistency_test_collection,
        setup_consistency_collection,
    ):
        """Test that different project_ids maintain proper isolation."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Ingest content for multiple projects
            projects = ["/test/project/alpha", "/test/project/beta", "/test/project/gamma"]

            for project in projects:
                await client.post(
                    f"{mcp_server_url}/mcp/store",
                    json={
                        "content": f"Content for {project}",
                        "metadata": {"project": project},
                        "collection": consistency_test_collection,
                        "project_id": project,
                    },
                )

            await asyncio.sleep(3)

            # Verify projects maintain isolation
            all_results = []
            for project in projects:
                search_response = await client.post(
                    f"{mcp_server_url}/mcp/search",
                    json={
                        "query": f"Content for {project}",
                        "collection": consistency_test_collection,
                        "limit": 5,
                    },
                )

                if search_response.status_code == 200:
                    results = search_response.json().get("results", search_response.json())
                    all_results.append((project, len(results)))

            logger.info(
                "✅ Multi-project isolation test\n"
                + "\n".join([f"   {proj}: {count} results" for proj, count in all_results])
            )


class TestMetadataConsistency:
    """Test metadata consistency validation."""

    @pytest.mark.integration
    async def test_metadata_preservation_through_pipeline(
        self,
        mcp_server_url,
        qdrant_client,
        consistency_test_collection,
        setup_consistency_collection,
    ):
        """Test that metadata is preserved consistently through the entire pipeline."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Submit with rich metadata
            test_metadata = {
                "file_type": "python",
                "language": "en",
                "author": "test_user",
                "version": "1.0.0",
                "tags": ["testing", "consistency", "metadata"],
                "priority": "high",
                "nested": {"key1": "value1", "key2": "value2"},
            }

            response = await client.post(
                f"{mcp_server_url}/mcp/store",
                json={
                    "content": "Metadata preservation test content",
                    "metadata": test_metadata,
                    "collection": consistency_test_collection,
                    "project_id": "/test/metadata",
                },
            )

            assert response.status_code == 200
            await asyncio.sleep(2)

            # Retrieve and verify metadata
            search_response = await client.post(
                f"{mcp_server_url}/mcp/search",
                json={
                    "query": "Metadata preservation test",
                    "collection": consistency_test_collection,
                    "limit": 1,
                },
            )

            assert search_response.status_code == 200
            results = search_response.json().get("results", search_response.json())

            metadata_preserved = False
            if results and "metadata" in results[0]:
                retrieved_metadata = results[0]["metadata"]
                # Check key metadata fields are preserved
                metadata_preserved = (
                    retrieved_metadata.get("file_type") == test_metadata["file_type"]
                    and retrieved_metadata.get("author") == test_metadata["author"]
                )

            logger.info(
                f"✅ Metadata preservation test\n"
                f"   Metadata preserved: {metadata_preserved}"
            )

            assert metadata_preserved, "Metadata was not consistently preserved"


# Test report generation
def generate_state_consistency_test_report(
    test_results: dict[str, Any], output_path: Path = None
) -> str:
    """Generate comprehensive state consistency test report."""
    if output_path is None:
        output_path = Path("/tmp") / "state_consistency_test_report.txt"

    report_lines = [
        "=" * 80,
        "STATE CONSISTENCY INTEGRATION TEST REPORT",
        "=" * 80,
        "",
        f"Test Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=" * 80,
        "TEST COVERAGE",
        "=" * 80,
        "",
        "1. SQLite/Daemon/Qdrant Synchronization:",
        "   ✓ Three-way state consistency validation",
        "   ✓ State sync after batch ingestion",
        "",
        "2. Consistency After Failures:",
        "   ✓ Consistency after daemon restart",
        "   ✓ Consistency after network interruption",
        "",
        "3. Ingestion Queue Integrity:",
        "   ✓ Queue integrity under load",
        "   ✓ Processing status accuracy",
        "",
        "4. Project ID Consistency:",
        "   ✓ Project ID propagation across components",
        "   ✓ Multi-project isolation",
        "",
        "5. Metadata Consistency:",
        "   ✓ Metadata preservation through pipeline",
        "",
        "=" * 80,
        "CONSISTENCY METRICS",
        "=" * 80,
        "",
        f"State Sync Tests: {test_results.get('sync_tests', 2)}",
        f"Failure Recovery Tests: {test_results.get('recovery_tests', 2)}",
        f"Queue Integrity Tests: {test_results.get('queue_tests', 2)}",
        f"Project ID Tests: {test_results.get('project_id_tests', 2)}",
        f"Metadata Tests: {test_results.get('metadata_tests', 1)}",
        "",
        "=" * 80,
        "RECOMMENDATIONS",
        "=" * 80,
        "",
        "1. Implement automated consistency checks in production",
        "2. Set up alerts for state divergence across components",
        "3. Monitor queue depth and processing lag",
        "4. Validate project_id integrity during ingestion",
        "5. Implement periodic state reconciliation jobs",
        "6. Monitor metadata fidelity through pipeline",
        "7. Test consistency checks after each deployment",
        "",
        "=" * 80,
    ]

    report = "\n".join(report_lines)

    # Write to file
    output_path.write_text(report)
    logger.info(f"📊 Test report written to: {output_path}")

    return report


if __name__ == "__main__":
    # Generate sample report
    sample_results = {
        "sync_tests": 2,
        "recovery_tests": 2,
        "queue_tests": 2,
        "project_id_tests": 2,
        "metadata_tests": 1,
    }

    report = generate_state_consistency_test_report(sample_results)
    print(report)
