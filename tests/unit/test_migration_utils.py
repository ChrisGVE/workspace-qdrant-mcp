"""
Unit tests for migration utilities.

This module tests all components of the collection migration system including
analysis, planning, execution, rollback, and reporting functionality.
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from src.python.common.memory.migration_utils import (
    CollectionStructureAnalyzer,
    MigrationPlanner,
    BatchMigrator,
    RollbackManager,
    MigrationReporter,
    CollectionMigrationManager,
    CollectionInfo,
    CollectionPattern,
    MigrationPlan,
    MigrationResult,
    MigrationPhase,
)


class TestCollectionStructureAnalyzer:
    """Test the CollectionStructureAnalyzer class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Qdrant client."""
        client = Mock()
        return client

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.workspace.global_collections = ['scratchbook', 'global']
        config.workspace.effective_collection_types = ['docs', 'code', 'notes']
        return config

    @pytest.fixture
    def analyzer(self, mock_client, mock_config):
        """Create a CollectionStructureAnalyzer instance."""
        return CollectionStructureAnalyzer(mock_client, mock_config)

    def test_analyze_naming_pattern_suffix_based(self, analyzer):
        """Test suffix-based pattern detection."""
        pattern, project, suffix = analyzer._analyze_naming_pattern("my-project-docs")
        assert pattern == CollectionPattern.SUFFIX_BASED
        assert project == "my-project"
        assert suffix == "docs"

    def test_analyze_naming_pattern_global(self, analyzer):
        """Test global collection pattern detection."""
        pattern, project, suffix = analyzer._analyze_naming_pattern("scratchbook")
        assert pattern == CollectionPattern.GLOBAL
        assert project is None
        assert suffix is None

    def test_analyze_naming_pattern_project_based(self, analyzer):
        """Test project-based pattern detection."""
        pattern, project, suffix = analyzer._analyze_naming_pattern("myproject")
        assert pattern == CollectionPattern.PROJECT_BASED
        assert project == "myproject"
        assert suffix is None

    def test_analyze_naming_pattern_unknown(self, analyzer):
        """Test unknown pattern detection."""
        pattern, project, suffix = analyzer._analyze_naming_pattern("x")
        assert pattern == CollectionPattern.UNKNOWN
        assert project is None
        assert suffix is None

    def test_looks_like_project_name(self, analyzer):
        """Test project name heuristics."""
        assert analyzer._looks_like_project_name("my-project")
        assert analyzer._looks_like_project_name("workspace_app")
        assert analyzer._looks_like_project_name("project123")
        assert not analyzer._looks_like_project_name("x")
        assert not analyzer._looks_like_project_name("ab")

    def test_get_collection_stats(self, analyzer, mock_client):
        """Test collection statistics retrieval."""
        # Mock collection info
        collection_info = Mock()
        collection_info.points_count = 1000
        collection_info.vectors_count = 1000
        collection_info.config.params.vectors = {'': Mock(size=384)}
        
        mock_client.get_collection.return_value = collection_info

        stats = analyzer._get_collection_stats("test-collection")
        
        assert stats['point_count'] == 1000
        assert stats['vector_count'] == 1000
        assert stats['size_mb'] > 0  # Should calculate estimated size

    def test_get_collection_stats_error_handling(self, analyzer, mock_client):
        """Test error handling in collection stats."""
        mock_client.get_collection.side_effect = Exception("Connection error")
        
        stats = analyzer._get_collection_stats("test-collection")
        
        assert stats['point_count'] == 0
        assert stats['vector_count'] == 0
        assert stats['size_mb'] == 0.0

    def test_calculate_migration_priority(self, analyzer):
        """Test migration priority calculation."""
        # High priority: suffix-based with data
        priority = analyzer._calculate_migration_priority(CollectionPattern.SUFFIX_BASED, 1000)
        assert priority == 1

        # Medium priority: project-based with data
        priority = analyzer._calculate_migration_priority(CollectionPattern.PROJECT_BASED, 1000)
        assert priority == 2

        # Low priority: everything else
        priority = analyzer._calculate_migration_priority(CollectionPattern.UNKNOWN, 100)
        assert priority == 3

    @pytest.mark.asyncio
    async def test_analyze_metadata_structure(self, analyzer, mock_client):
        """Test metadata structure analysis."""
        # Mock points with various metadata
        mock_points = [
            Mock(payload={'field1': 'value1', 'field2': 'value2'}),
            Mock(payload={'field1': 'value1', 'project_id': 'test'}),
            Mock(payload={'field3': 'value3'})
        ]
        
        mock_client.scroll.return_value = (mock_points, None)
        
        metadata_keys, has_project_metadata = await analyzer._analyze_metadata_structure("test-collection")
        
        assert 'field1' in metadata_keys
        assert 'field2' in metadata_keys
        assert 'field3' in metadata_keys
        assert 'project_id' in metadata_keys
        assert has_project_metadata

    @pytest.mark.asyncio
    async def test_analyze_single_collection(self, analyzer, mock_client):
        """Test single collection analysis."""
        # Mock collection info
        collection_info = Mock()
        collection_info.points_count = 500
        collection_info.vectors_count = 500
        
        mock_client.get_collection.return_value = collection_info
        mock_client.scroll.return_value = ([], None)
        
        with patch.object(analyzer, '_get_collection_stats') as mock_stats:
            mock_stats.return_value = {
                'point_count': 500,
                'vector_count': 500,
                'size_mb': 10.5,
                'created_at': None,
                'last_modified': None
            }
            
            info = await analyzer._analyze_single_collection("my-project-docs")
            
            assert info.name == "my-project-docs"
            assert info.pattern == CollectionPattern.SUFFIX_BASED
            assert info.project_name == "my-project"
            assert info.suffix == "docs"
            assert info.point_count == 500


class TestMigrationPlanner:
    """Test the MigrationPlanner class."""

    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock analyzer."""
        return Mock()

    @pytest.fixture
    def mock_collision_detector(self):
        """Create a mock collision detector."""
        return Mock()

    @pytest.fixture
    def planner(self, mock_analyzer, mock_collision_detector):
        """Create a MigrationPlanner instance."""
        return MigrationPlanner(mock_analyzer, mock_collision_detector)

    @pytest.fixture
    def sample_collections(self):
        """Create sample collection data."""
        return [
            CollectionInfo(
                name="project1-docs",
                pattern=CollectionPattern.SUFFIX_BASED,
                project_name="project1",
                suffix="docs",
                point_count=1000,
                migration_priority=1
            ),
            CollectionInfo(
                name="project1-code",
                pattern=CollectionPattern.SUFFIX_BASED,
                project_name="project1",
                suffix="code",
                point_count=500,
                migration_priority=1
            ),
            CollectionInfo(
                name="project2",
                pattern=CollectionPattern.PROJECT_BASED,
                project_name="project2",
                point_count=2000,
                migration_priority=2
            )
        ]

    @pytest.mark.asyncio
    async def test_create_migration_plan(self, planner, sample_collections):
        """Test migration plan creation."""
        plan = await planner.create_migration_plan(sample_collections)
        
        assert len(plan.source_collections) == 3  # All collections need migration
        assert plan.total_points_to_migrate == 3500
        assert len(plan.target_collections) == 3
        assert plan.batch_size > 0
        assert plan.parallel_batches > 0

    @pytest.mark.asyncio
    async def test_generate_target_names(self, planner, sample_collections):
        """Test target name generation."""
        target_names = await planner._generate_target_names(sample_collections)
        
        assert len(target_names) == 3
        assert "project1-docs" in target_names  # Suffix-based keeps same name
        assert "project1-code" in target_names
        assert "project2-documents" in target_names  # Project-based gets suffix

    def test_calculate_migration_order(self, planner, sample_collections):
        """Test migration order calculation."""
        order, dependencies = planner._calculate_migration_order(sample_collections)
        
        assert len(order) == 3
        # Should be ordered by priority then size
        assert order[0] in ["project1-docs", "project1-code"]  # Priority 1 first
        assert order[-1] == "project2"  # Priority 2 last

    def test_estimate_duration(self, planner, sample_collections):
        """Test duration estimation."""
        duration = planner._estimate_duration(sample_collections)
        
        assert duration > 0
        # Should include base time + overhead
        expected_base = 3500 / 1000  # 3.5 minutes for points
        expected_overhead = 3 * 2  # 6 minutes for collection overhead
        assert duration >= expected_base + expected_overhead

    def test_optimize_batch_config(self, planner):
        """Test batch configuration optimization."""
        # Small dataset
        batch_size, parallel = planner._optimize_batch_config(5000)
        assert batch_size == 500
        assert parallel == 2

        # Medium dataset
        batch_size, parallel = planner._optimize_batch_config(50000)
        assert batch_size == 1000
        assert parallel == 3

        # Large dataset
        batch_size, parallel = planner._optimize_batch_config(500000)
        assert batch_size == 2000
        assert parallel == 4

        # Very large dataset
        batch_size, parallel = planner._optimize_batch_config(2000000)
        assert batch_size == 5000
        assert parallel == 5


class TestBatchMigrator:
    """Test the BatchMigrator class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock workspace client."""
        client = Mock()
        client.client = Mock()  # Access to underlying client
        return client

    @pytest.fixture
    def mock_metadata_schema(self):
        """Create a mock metadata schema."""
        return Mock()

    @pytest.fixture
    def migrator(self, mock_client, mock_metadata_schema):
        """Create a BatchMigrator instance."""
        return BatchMigrator(mock_client, mock_metadata_schema)

    @pytest.fixture
    def sample_collection(self):
        """Create sample collection info."""
        return CollectionInfo(
            name="test-project-docs",
            pattern=CollectionPattern.SUFFIX_BASED,
            project_name="test-project",
            suffix="docs",
            point_count=1000
        )

    @pytest.fixture
    def sample_plan(self):
        """Create sample migration plan."""
        return MigrationPlan(
            batch_size=100,
            parallel_batches=2
        )

    def test_inject_project_metadata(self, migrator, sample_collection):
        """Test project metadata injection."""
        # Mock source points
        source_points = [
            Mock(
                id="point1",
                vector=[1.0, 2.0, 3.0],
                payload={'original_field': 'value1'}
            ),
            Mock(
                id="point2",
                vector=[4.0, 5.0, 6.0],
                payload={'original_field': 'value2'}
            )
        ]

        enhanced_points = migrator._inject_project_metadata(source_points, sample_collection)
        
        assert len(enhanced_points) == 2
        
        # Check first enhanced point
        point1 = enhanced_points[0]
        assert point1.id == "point1"
        assert point1.vector == [1.0, 2.0, 3.0]
        assert point1.payload['original_field'] == 'value1'
        assert point1.payload['project_id'] == 'test-project'
        assert point1.payload['project_name'] == 'test-project'
        assert point1.payload['collection_suffix'] == 'docs'
        assert 'migrated_at' in point1.payload
        assert point1.payload['migration_source'] == 'test-project-docs'

    @pytest.mark.asyncio
    async def test_ensure_target_collection_exists(self, migrator, mock_client, sample_collection):
        """Test target collection creation when it doesn't exist."""
        # Mock collection doesn't exist
        mock_client.get_collection.side_effect = ResponseHandlingException("Not found")
        
        # Mock source collection info
        source_info = Mock()
        source_info.config.params.vectors = {'default': Mock(size=384)}
        source_info.config.params.sparse_vectors = None
        mock_client.get_collection.side_effect = [ResponseHandlingException("Not found"), source_info]
        
        mock_client.create_collection = AsyncMock()
        
        await migrator._ensure_target_collection("target-collection", sample_collection)
        
        mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_target_collection_already_exists(self, migrator, mock_client, sample_collection):
        """Test when target collection already exists."""
        # Mock collection exists
        mock_client.get_collection.return_value = Mock()
        mock_client.create_collection = AsyncMock()
        
        await migrator._ensure_target_collection("target-collection", sample_collection)
        
        mock_client.create_collection.assert_not_called()


class TestRollbackManager:
    """Test the RollbackManager class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Qdrant client."""
        return Mock()

    @pytest.fixture
    def temp_backup_dir(self, tmp_path):
        """Create a temporary backup directory."""
        return tmp_path / "backups"

    @pytest.fixture
    def rollback_manager(self, mock_client, temp_backup_dir):
        """Create a RollbackManager instance."""
        return RollbackManager(mock_client, temp_backup_dir)

    @pytest.mark.asyncio
    async def test_create_backup(self, rollback_manager, mock_client, temp_backup_dir):
        """Test backup creation."""
        # Mock points data
        mock_points = [
            Mock(id="1", vector=[1.0, 2.0], payload={'field': 'value1'}),
            Mock(id="2", vector=[3.0, 4.0], payload={'field': 'value2'})
        ]
        mock_client.scroll.return_value = (mock_points, None)
        
        backup_file = await rollback_manager.create_backup("test-collection", "migration-123")
        
        assert Path(backup_file).exists()
        assert "test-collection" in backup_file
        assert "migration-123" in backup_file
        
        # Verify backup content
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)
        
        assert backup_data['collection_name'] == "test-collection"
        assert backup_data['migration_id'] == "migration-123"
        assert backup_data['point_count'] == 2
        assert len(backup_data['points']) == 2

    @pytest.mark.asyncio
    async def test_restore_backup(self, rollback_manager, mock_client, temp_backup_dir):
        """Test backup restoration."""
        # Create a backup file
        backup_data = {
            'collection_name': 'test-collection',
            'migration_id': 'migration-123',
            'created_at': datetime.now().isoformat(),
            'point_count': 2,
            'points': [
                {'id': '1', 'vector': [1.0, 2.0], 'payload': {'field': 'value1'}},
                {'id': '2', 'vector': [3.0, 4.0], 'payload': {'field': 'value2'}}
            ]
        }
        
        backup_file = temp_backup_dir / "test_backup.json"
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f)
        
        # Mock client operations
        mock_client.scroll.return_value = ([], None)  # No existing points
        mock_client.upsert = Mock()
        
        success = await rollback_manager.restore_backup(str(backup_file))
        
        assert success
        mock_client.upsert.assert_called()


class TestMigrationReporter:
    """Test the MigrationReporter class."""

    @pytest.fixture
    def temp_report_dir(self, tmp_path):
        """Create a temporary report directory."""
        return tmp_path / "reports"

    @pytest.fixture
    def reporter(self, temp_report_dir):
        """Create a MigrationReporter instance."""
        return MigrationReporter(temp_report_dir)

    @pytest.fixture
    def sample_plan(self):
        """Create sample migration plan."""
        return MigrationPlan(
            plan_id="plan-123",
            total_points_to_migrate=1000,
            estimated_duration_minutes=5.0,
            source_collections=[
                CollectionInfo(
                    name="test-collection",
                    pattern=CollectionPattern.SUFFIX_BASED,
                    project_name="test",
                    suffix="docs",
                    point_count=1000
                )
            ]
        )

    @pytest.fixture
    def sample_result(self):
        """Create sample migration result."""
        return MigrationResult(
            plan_id="plan-123",
            execution_id="exec-456",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            success=True,
            phase=MigrationPhase.COMPLETED,
            collections_migrated=1,
            points_migrated=1000,
            points_failed=0
        )

    @pytest.mark.asyncio
    async def test_generate_migration_report(self, reporter, sample_plan, sample_result, temp_report_dir):
        """Test migration report generation."""
        report_file = await reporter.generate_migration_report(sample_plan, sample_result)
        
        assert report_file.exists()
        assert "migration_report" in report_file.name
        assert sample_result.execution_id in report_file.name
        
        # Verify report content
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        assert report_data['migration_summary']['plan_id'] == "plan-123"
        assert report_data['migration_summary']['execution_id'] == "exec-456"
        assert report_data['migration_summary']['overall_success'] is True
        assert report_data['execution_results']['collections_migrated'] == 1
        assert report_data['execution_results']['points_migrated'] == 1000

    def test_generate_summary_text(self, reporter, temp_report_dir):
        """Test summary text generation."""
        # Create a test report file
        report_data = {
            'migration_summary': {
                'execution_id': 'exec-123',
                'overall_success': True,
                'total_duration_seconds': 300.0,
                'final_phase': 'completed'
            },
            'execution_results': {
                'collections_migrated': 2,
                'points_migrated': 5000,
                'points_failed': 10,
                'success_rate_percent': 99.8
            },
            'performance_metrics': {
                'points_per_second': 16.7,
                'analysis_duration_seconds': 30.0,
                'migration_duration_seconds': 250.0,
                'validation_duration_seconds': 20.0
            },
            'issues_and_warnings': {
                'errors': ['Error 1'],
                'warnings': ['Warning 1', 'Warning 2']
            }
        }
        
        report_file = temp_report_dir / "test_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f)
        
        summary = reporter.generate_summary_text(report_file)
        
        assert 'exec-123' in summary
        assert 'SUCCESS' in summary
        assert '5,000' in summary
        assert '99.8%' in summary
        assert 'Errors: 1' in summary
        assert 'Warnings: 2' in summary


class TestCollectionMigrationManager:
    """Test the CollectionMigrationManager class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock workspace client."""
        client = Mock()
        client.client = Mock()
        return client

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return Mock()

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories."""
        return {
            'backup': tmp_path / "backups",
            'report': tmp_path / "reports"
        }

    @pytest.fixture
    def manager(self, mock_client, mock_config, temp_dirs):
        """Create a CollectionMigrationManager instance."""
        return CollectionMigrationManager(
            mock_client,
            mock_config,
            backup_dir=temp_dirs['backup'],
            report_dir=temp_dirs['report']
        )

    @pytest.mark.asyncio
    async def test_analyze_collections(self, manager):
        """Test collection analysis delegation."""
        with patch.object(manager.analyzer, 'analyze_all_collections') as mock_analyze:
            mock_analyze.return_value = [Mock()]
            
            result = await manager.analyze_collections()
            
            mock_analyze.assert_called_once()
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_create_migration_plan(self, manager):
        """Test migration plan creation delegation."""
        collections = [Mock()]
        
        with patch.object(manager.planner, 'create_migration_plan') as mock_plan:
            mock_plan.return_value = Mock()
            
            result = await manager.create_migration_plan(collections)
            
            mock_plan.assert_called_once_with(collections)

    @pytest.mark.asyncio
    async def test_execute_migration_success(self, manager):
        """Test successful migration execution."""
        # Create a simple plan
        plan = MigrationPlan(
            plan_id="test-plan",
            source_collections=[],
            target_collections=[],
            create_backups=False,
            enable_validation=False
        )
        
        result = await manager.execute_migration(plan)
        
        assert result.plan_id == "test-plan"
        assert result.success is True
        assert result.phase == MigrationPhase.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_migration_with_error(self, manager):
        """Test migration execution with error."""
        plan = MigrationPlan(
            plan_id="test-plan",
            source_collections=[Mock()],  # Will cause error in migration
            target_collections=["target"],
            create_backups=False,
            enable_validation=False
        )
        
        # Mock migrator to raise error
        with patch.object(manager.migrator, 'migrate_collection', side_effect=Exception("Test error")):
            result = await manager.execute_migration(plan)
            
            assert result.plan_id == "test-plan"
            assert result.success is False
            assert result.phase == MigrationPhase.FAILED
            assert "Test error" in result.error_message


# Integration tests
class TestMigrationIntegration:
    """Integration tests for the migration system."""

    @pytest.mark.asyncio
    async def test_full_migration_workflow(self):
        """Test complete migration workflow from analysis to reporting."""
        # This would require a real Qdrant instance for full integration testing
        # For now, we'll test the workflow with mocks
        
        # Mock client and config
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace.global_collections = ['scratchbook']
        mock_config.workspace.effective_collection_types = ['docs', 'code']
        
        # Create manager
        manager = CollectionMigrationManager(mock_client, mock_config)
        
        # Mock the analyzer to return test collections
        test_collections = [
            CollectionInfo(
                name="test-project-docs",
                pattern=CollectionPattern.SUFFIX_BASED,
                project_name="test-project",
                suffix="docs",
                point_count=100,
                migration_priority=1
            )
        ]
        
        with patch.object(manager.analyzer, 'analyze_all_collections', return_value=test_collections):
            # 1. Analyze collections
            collections = await manager.analyze_collections()
            assert len(collections) == 1
            assert collections[0].name == "test-project-docs"
            
            # 2. Create migration plan
            plan = await manager.create_migration_plan(collections)
            assert plan.total_points_to_migrate == 100
            assert len(plan.source_collections) == 1
            
            # 3. Execute migration (mocked)
            with patch.object(manager.migrator, 'migrate_collection') as mock_migrate:
                mock_migrate.return_value = {
                    'source': 'test-project-docs',
                    'target': 'test-project-docs',
                    'success': True,
                    'points_migrated': 100,
                    'points_failed': 0,
                    'batches_processed': 1,
                    'batches_failed': 0,
                    'errors': []
                }
                
                result = await manager.execute_migration(plan)
                assert result.success is True
                assert result.points_migrated == 100
                
                # 4. Generate report
                report_file = await manager.generate_migration_report(plan, result)
                assert report_file.exists()


if __name__ == "__main__":
    pytest.main([__file__])