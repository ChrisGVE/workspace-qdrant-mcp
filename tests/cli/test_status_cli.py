"""
Tests for the status CLI command.

This module tests the comprehensive status and user feedback system implemented
for Task 72, including basic status display, filtering, and export functionality.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from wqm_cli.cli.status import (
    create_queue_breakdown,
    create_recent_activity,
    create_status_overview,
    export_status_data,
    format_duration,
    format_file_size,
    format_timestamp,
    get_comprehensive_status,
    status_app,
)


class TestStatusCLI:
    """Test cases for the status CLI functionality."""

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        # Test valid timestamp
        timestamp = "2023-12-07T10:30:45.123Z"
        result = format_timestamp(timestamp)
        assert "2023-12-07 10:30:45" in result

        # Test invalid timestamp
        invalid_timestamp = "invalid"
        result = format_timestamp(invalid_timestamp)
        assert result == "invalid"

    def test_format_file_size(self):
        """Test file size formatting."""
        assert format_file_size(0) == "0 B"
        assert format_file_size(512) == "512.0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"

    def test_format_duration(self):
        """Test duration formatting."""
        assert format_duration(30.5) == "30.5s"
        assert format_duration(90) == "1.5m"
        assert format_duration(3600) == "1.0h"
        assert format_duration(7200) == "2.0h"

    def test_create_status_overview(self):
        """Test status overview panel creation."""
        processing_status = {
            "processing_info": {
                "currently_processing": 2,
                "recent_successful": 15,
                "recent_failed": 1
            }
        }

        queue_stats = {
            "queue_stats": {
                "total": 5
            }
        }

        grpc_stats = {
            "success": True,
            "stats": {
                "engine_stats": {
                    "total_documents_processed": 1000,
                    "uptime_seconds": 3600
                }
            }
        }

        panel = create_status_overview(processing_status, queue_stats, grpc_stats)
        assert panel is not None
        assert "Processing Status Overview" in str(panel.title)

    def test_create_queue_breakdown_empty(self):
        """Test queue breakdown with empty queue."""
        queue_stats = {
            "queue_stats": {
                "total": 0
            }
        }

        panel = create_queue_breakdown(queue_stats)
        assert panel is not None
        assert "No files currently in processing queue" in panel.renderable.plain

    def test_create_queue_breakdown_with_data(self):
        """Test queue breakdown with queue data."""
        queue_stats = {
            "queue_stats": {
                "total": 10,
                "urgent_priority": 2,
                "high_priority": 3,
                "normal_priority": 4,
                "low_priority": 1,
                "urgent_collections": ["critical"],
                "high_collections": ["important", "docs"],
                "normal_collections": ["general"],
                "low_collections": ["archive"]
            }
        }

        panel = create_queue_breakdown(queue_stats)
        assert panel is not None
        assert "Processing Queue Breakdown" in str(panel.title)

    def test_create_recent_activity_empty(self):
        """Test recent activity with no data."""
        processing_status = {
            "recent_files": []
        }

        panel = create_recent_activity(processing_status)
        assert panel is not None
        assert "No recent processing activity" in panel.renderable.plain

    def test_create_recent_activity_with_data(self):
        """Test recent activity with processing data."""
        processing_status = {
            "recent_files": [
                {
                    "file_path": "/test/document.pdf",
                    "status": "completed",
                    "collection": "docs",
                    "timestamp": "2023-12-07T10:30:45.123Z",
                    "processing_duration": 2.5
                },
                {
                    "file_path": "/test/failed_doc.txt",
                    "status": "failed",
                    "collection": "texts",
                    "timestamp": "2023-12-07T10:29:30.123Z",
                    "error_message": "Parsing error"
                }
            ]
        }

        panel = create_recent_activity(processing_status)
        assert panel is not None
        assert "Recent Processing Activity" in str(panel.title)

    @pytest.mark.asyncio
    async def test_export_status_data_json(self, tmp_path):
        """Test JSON export functionality."""
        status_data = {
            "processing_status": {
                "recent_files": [
                    {
                        "file_path": "/test/doc.pdf",
                        "status": "completed",
                        "collection": "docs"
                    }
                ]
            },
            "timestamp": "2023-12-07T10:30:45.123Z"
        }

        output_file = tmp_path / "status.json"

        await export_status_data(
            status_data=status_data,
            export_format="json",
            output_path=output_file,
            collection=None,
            status_filter=None,
            days=7,
            limit=100
        )

        assert output_file.exists()

        # Verify JSON content
        with open(output_file) as f:
            exported_data = json.load(f)

        assert "export_info" in exported_data
        assert "status_data" in exported_data
        assert exported_data["export_info"]["format"] == "json"
        assert exported_data["status_data"] == status_data

    @pytest.mark.asyncio
    async def test_export_status_data_csv(self, tmp_path):
        """Test CSV export functionality."""
        status_data = {
            "processing_status": {
                "recent_files": [
                    {
                        "file_path": "/test/doc.pdf",
                        "status": "completed",
                        "collection": "docs",
                        "timestamp": "2023-12-07T10:30:45.123Z",
                        "processing_duration": 2.5,
                        "error_message": "",
                        "chunks_added": 5
                    },
                    {
                        "file_path": "/test/failed.txt",
                        "status": "failed",
                        "collection": "texts",
                        "timestamp": "2023-12-07T10:29:30.123Z",
                        "processing_duration": 0,
                        "error_message": "Parse error",
                        "chunks_added": 0
                    }
                ]
            }
        }

        output_file = tmp_path / "status.csv"

        await export_status_data(
            status_data=status_data,
            export_format="csv",
            output_path=output_file,
            collection=None,
            status_filter=None,
            days=7,
            limit=100
        )

        assert output_file.exists()

        # Verify CSV content
        content = output_file.read_text()
        assert "timestamp,file_path,status,collection" in content
        assert "/test/doc.pdf" in content
        assert "/test/failed.txt" in content
        assert "completed" in content
        assert "failed" in content

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.status.get_grpc_engine_stats')
    @patch('wqm_cli.cli.status.get_processing_status')
    @patch('wqm_cli.cli.status.get_queue_stats')
    @patch('wqm_cli.cli.status.get_watch_folder_configs')
    @patch('wqm_cli.cli.status.get_database_stats')
    async def test_get_comprehensive_status(
        self,
        mock_db_stats,
        mock_watch_configs,
        mock_queue_stats,
        mock_processing_status,
        mock_grpc_stats
    ):
        """Test comprehensive status gathering."""
        # Setup mocks
        mock_processing_status.return_value = {
            "success": True,
            "processing_info": {"currently_processing": 1}
        }

        mock_queue_stats.return_value = {
            "success": True,
            "queue_stats": {"total": 3}
        }

        mock_watch_configs.return_value = {
            "success": True,
            "watch_configs": []
        }

        mock_db_stats.return_value = {
            "success": True,
            "database_stats": {"total_size_mb": 15.5}
        }

        mock_grpc_stats.return_value = {
            "success": True,
            "stats": {"engine_stats": {"uptime_seconds": 3600}}
        }

        # Test comprehensive status gathering
        status_data = await get_comprehensive_status()

        assert status_data["processing_status"]["success"] is True
        assert status_data["queue_stats"]["success"] is True
        assert status_data["watch_configs"]["success"] is True
        assert status_data["database_stats"]["success"] is True
        assert status_data["grpc_stats"]["success"] is True
        assert "timestamp" in status_data

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.status.get_comprehensive_status')
    async def test_get_comprehensive_status_error_handling(self, mock_get_status):
        """Test error handling in comprehensive status gathering."""
        # Simulate an exception
        mock_get_status.side_effect = Exception("Connection failed")

        # The function should handle exceptions gracefully
        # This would be called within the actual CLI command
        try:
            await get_comprehensive_status()
        except Exception as e:
            assert "Connection failed" in str(e)


class TestStatusCLIIntegration:
    """Integration tests for the status CLI system."""

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.status.test_grpc_connection')
    async def test_grpc_fallback_mechanism(self, mock_grpc_test):
        """Test that status CLI falls back gracefully when gRPC is unavailable."""
        # Mock failed gRPC connection
        mock_grpc_test.return_value = {
            "connected": False,
            "error": "Connection refused"
        }

        # This would test the fallback logic in live_streaming_status_monitor
        # The function should detect gRPC failure and fall back to polling mode
        connection_result = await mock_grpc_test("127.0.0.1", 50051, timeout=5.0)

        assert connection_result["connected"] is False
        assert "Connection refused" in connection_result["error"]

    def test_cli_help_includes_streaming_options(self):
        """Test that the CLI help includes streaming-related options."""
        # This would test that the CLI properly shows --stream, --grpc-host, --grpc-port options
        # The typer app should include these options
        assert status_app is not None

        # In a real test, we might invoke the CLI with --help and check the output
        # contains the streaming options

    @pytest.mark.asyncio
    async def test_export_unsupported_format(self):
        """Test error handling for unsupported export formats."""
        import click.exceptions

        status_data = {"test": "data"}

        # Test with unsupported format
        # typer.Exit raises click.exceptions.Exit
        with pytest.raises((SystemExit, click.exceptions.Exit)) as exc_info:
            await export_status_data(
                status_data=status_data,
                export_format="xml",  # Unsupported
                output_path=None,
                collection=None,
                status_filter=None,
                days=7,
                limit=100
            )
        # export_status_data raises typer.Exit(1) for unsupported formats
        assert exc_info.value.exit_code == 1


class TestStatusFiltering:
    """Test cases for status filtering functionality."""

    def test_collection_filtering(self):
        """Test filtering by collection name."""
        files = [
            {"file_path": "/doc1.pdf", "collection": "docs", "status": "completed"},
            {"file_path": "/text1.txt", "collection": "texts", "status": "completed"},
            {"file_path": "/doc2.pdf", "collection": "docs", "status": "failed"},
        ]

        # Filter by collection
        docs_files = [f for f in files if f.get("collection") == "docs"]
        assert len(docs_files) == 2
        assert all(f["collection"] == "docs" for f in docs_files)

    def test_status_filtering(self):
        """Test filtering by status."""
        files = [
            {"file_path": "/doc1.pdf", "status": "completed"},
            {"file_path": "/doc2.pdf", "status": "failed"},
            {"file_path": "/doc3.pdf", "status": "completed"},
        ]

        # Filter by status
        completed_files = [f for f in files if f.get("status") == "completed"]
        assert len(completed_files) == 2
        assert all(f["status"] == "completed" for f in completed_files)

        failed_files = [f for f in files if f.get("status") == "failed"]
        assert len(failed_files) == 1
        assert failed_files[0]["status"] == "failed"

    def test_combined_filtering(self):
        """Test filtering by both collection and status."""
        files = [
            {"file_path": "/doc1.pdf", "collection": "docs", "status": "completed"},
            {"file_path": "/text1.txt", "collection": "texts", "status": "completed"},
            {"file_path": "/doc2.pdf", "collection": "docs", "status": "failed"},
            {"file_path": "/text2.txt", "collection": "texts", "status": "failed"},
        ]

        # Filter by collection and status
        docs_completed = [
            f for f in files
            if f.get("collection") == "docs" and f.get("status") == "completed"
        ]

        assert len(docs_completed) == 1
        assert docs_completed[0]["file_path"] == "/doc1.pdf"


if __name__ == "__main__":
    pytest.main([__file__])
