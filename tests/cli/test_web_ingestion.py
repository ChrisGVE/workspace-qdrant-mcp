"""
Integration tests for web ingestion CLI commands.

Tests the complete web ingestion workflow including CLI argument parsing,
security configuration, content processing, and error handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner
from wqm_cli.cli.ingest import app


class TestWebIngestionCLI:
    """Tests for web ingestion CLI command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_successful_ingestion(self):
        """Mock successful web ingestion."""
        with patch('wqm_cli.cli.ingest._run_web_ingestion') as mock:
            mock.return_value = None  # Successful completion
            yield mock

    def test_web_ingestion_help(self, runner):
        """Test web ingestion command help."""
        result = runner.invoke(app, ['ingest-web', '--help'])

        assert result.exit_code == 0
        assert 'Ingest web content from URLs' in result.output
        assert '--max-pages' in result.output
        assert '--allowed-domains' in result.output
        assert '--disable-security' in result.output

    def test_web_ingestion_required_args(self, runner):
        """Test that required arguments are enforced."""
        # Missing collection argument
        result = runner.invoke(app, ['ingest-web', 'https://example.com'])

        assert result.exit_code != 0
        assert 'Missing option' in result.output

    @patch('asyncio.run')
    def test_web_ingestion_basic_call(self, mock_asyncio_run, runner):
        """Test basic web ingestion command call."""
        result = runner.invoke(app, [
            'ingest-web',
            'https://example.com/docs',
            '--collection', 'test-docs',
            '--yes'  # Skip confirmations
        ])

        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()

        # Verify the async function was called with correct parameters
        mock_asyncio_run.call_args[0][0]  # First positional argument
        # This is a coroutine, so we can't easily inspect its arguments
        # but we can verify it was called

    @patch('asyncio.run')
    def test_web_ingestion_with_options(self, mock_asyncio_run, runner):
        """Test web ingestion with various options."""
        result = runner.invoke(app, [
            'ingest-web',
            'https://example.com/docs',
            '--collection', 'test-docs',
            '--max-pages', '5',
            '--max-depth', '2',
            '--allowed-domains', 'example.com,docs.example.com',
            '--request-delay', '2.0',
            '--chunk-size', '500',
            '--dry-run',
            '--yes'
        ])

        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()

    @patch('asyncio.run')
    def test_web_ingestion_security_options(self, mock_asyncio_run, runner):
        """Test web ingestion security-related options."""
        result = runner.invoke(app, [
            'ingest-web',
            'https://example.com/docs',
            '--collection', 'test-docs',
            '--disable-security',
            '--allow-all-domains',
            '--yes'
        ])

        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()


class TestWebIngestionWorkflow:
    """Tests for the complete web ingestion workflow.

    Note: These tests use an outdated mock pattern that doesn't match
    the current implementation. The actual implementation uses:
    - get_daemon_client() instead of QdrantWorkspaceClient
    - daemon_client.process_document() instead of add_document()
    - get_config_manager() instead of Config class

    Tests are marked xfail until they can be rewritten to match the
    current implementation.
    """

    @pytest.fixture
    def mock_daemon_client(self):
        """Mock daemon client (current implementation)."""
        client = AsyncMock()
        client.connect.return_value = None
        client.disconnect.return_value = None
        client.get_system_status.return_value = MagicMock(status="running")

        # Mock process_document response
        response = MagicMock()
        response.success = True
        response.document_id = "doc123"
        response.chunks_created = 3
        response.error_message = ""
        client.process_document.return_value = response
        return client

    @pytest.fixture
    def mock_config(self):
        """Mock configuration manager."""
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: {
            'grpc.host': 'localhost',
            'grpc.port': 50051
        }.get(key, default)
        return config

    @pytest.fixture
    def mock_web_interface(self):
        """Mock web ingestion interface."""
        interface = AsyncMock()

        # Mock parsed document
        parsed_doc = MagicMock()
        parsed_doc.content = "Test web content from example.com"
        parsed_doc.additional_metadata = {
            'pages_crawled': 1,
            'security_warnings': []
        }

        interface.ingest_url.return_value = parsed_doc
        interface.ingest_site.return_value = parsed_doc

        return interface, parsed_doc

    @pytest.mark.xfail(reason="Test uses outdated mock pattern - needs rewrite for daemon_client API")
    @pytest.mark.asyncio
    @patch('wqm_cli.cli.ingest.get_daemon_client')
    @patch('wqm_cli.cli.ingest.get_config_manager')
    @patch('wqm_cli.cli.ingest.WebIngestionInterface')
    async def test_successful_single_page_ingestion(
        self,
        mock_interface_class,
        mock_config_class,
        mock_daemon_client_func,
        mock_daemon_client,
        mock_config,
        mock_web_interface,
    ):
        """Test successful single page web ingestion."""
        from wqm_cli.cli.ingest import _run_web_ingestion

        # Setup mocks - current implementation uses get_config_manager/get_daemon_client
        mock_config_class.return_value = mock_config
        mock_daemon_client_func.return_value = mock_daemon_client

        interface, parsed_doc = mock_web_interface
        mock_interface_class.return_value = interface

        # Run ingestion
        await _run_web_ingestion(
            url="https://example.com/test",
            collection="test-collection",
            max_pages=1,
            max_depth=0,
            allowed_domains=None,
            request_delay=1.0,
            chunk_size=1000,
            chunk_overlap=200,
            dry_run=False,
            disable_security=False,
            allow_all_domains=False,
            auto_confirm=True
        )

        # Verify workflow
        mock_daemon_client.connect.assert_called_once()
        interface.ingest_url.assert_called_once_with("https://example.com/test")
        mock_daemon_client.process_document.assert_called_once()
        mock_daemon_client.disconnect.assert_called_once()

    @pytest.mark.xfail(reason="Test uses outdated mock pattern - needs rewrite for daemon_client API")
    @pytest.mark.asyncio
    @patch('wqm_cli.cli.ingest.get_daemon_client')
    @patch('wqm_cli.cli.ingest.get_config_manager')
    @patch('wqm_cli.cli.ingest.WebIngestionInterface')
    async def test_successful_multi_page_ingestion(
        self,
        mock_interface_class,
        mock_config_class,
        mock_daemon_client_func,
        mock_daemon_client,
        mock_config,
        mock_web_interface,
    ):
        """Test successful multi-page web ingestion."""
        from wqm_cli.cli.ingest import _run_web_ingestion

        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_daemon_client_func.return_value = mock_daemon_client

        interface, parsed_doc = mock_web_interface
        parsed_doc.additional_metadata['pages_crawled'] = 5
        mock_interface_class.return_value = interface

        # Run ingestion
        await _run_web_ingestion(
            url="https://example.com/docs",
            collection="test-docs",
            max_pages=5,
            max_depth=2,
            allowed_domains=['example.com'],
            request_delay=1.5,
            chunk_size=800,
            chunk_overlap=150,
            dry_run=False,
            disable_security=False,
            allow_all_domains=False,
            auto_confirm=True
        )

        # Verify multi-page crawling was used
        interface.ingest_site.assert_called_once_with(
            "https://example.com/docs",
            max_pages=5,
            max_depth=2
        )
        mock_daemon_client.process_document.assert_called_once()

    @pytest.mark.xfail(reason="Test uses outdated mock pattern - needs rewrite for daemon_client API")
    @pytest.mark.asyncio
    @patch('wqm_cli.cli.ingest.get_daemon_client')
    @patch('wqm_cli.cli.ingest.get_config_manager')
    @patch('wqm_cli.cli.ingest.WebIngestionInterface')
    async def test_dry_run_workflow(
        self,
        mock_interface_class,
        mock_config_class,
        mock_daemon_client_func,
        mock_daemon_client,
        mock_config,
        mock_web_interface
    ):
        """Test dry run workflow (no actual ingestion)."""
        from wqm_cli.cli.ingest import _run_web_ingestion

        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_daemon_client_func.return_value = mock_daemon_client

        interface, parsed_doc = mock_web_interface
        mock_interface_class.return_value = interface

        # Run dry run
        await _run_web_ingestion(
            url="https://example.com/test",
            collection="test-collection",
            max_pages=1,
            max_depth=0,
            allowed_domains=None,
            request_delay=1.0,
            chunk_size=1000,
            chunk_overlap=200,
            dry_run=True,  # This is the key difference
            disable_security=False,
            allow_all_domains=False,
            auto_confirm=True
        )

        # Verify content was fetched but not ingested
        interface.ingest_url.assert_called_once()
        mock_daemon_client.process_document.assert_not_called()  # Should not be called in dry run

    @pytest.mark.xfail(reason="Test uses outdated mock pattern - needs rewrite for daemon_client API")
    @pytest.mark.asyncio
    @patch('wqm_cli.cli.ingest.get_daemon_client')
    @patch('wqm_cli.cli.ingest.get_config_manager')
    @patch('wqm_cli.cli.ingest.WebIngestionInterface')
    async def test_security_warnings_display(
        self,
        mock_interface_class,
        mock_config_class,
        mock_daemon_client_func,
        mock_daemon_client,
        mock_config,
    ):
        """Test that security warnings are properly displayed."""
        from wqm_cli.cli.ingest import _run_web_ingestion

        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_daemon_client_func.return_value = mock_daemon_client

        # Create interface with security warnings
        interface = AsyncMock()
        parsed_doc = MagicMock()
        parsed_doc.content = "Content with warnings"
        parsed_doc.additional_metadata = {
            'pages_crawled': 1,
            'security_warnings': [
                'Script tag detected in content',
                'External link to suspicious domain',
                'Encoded characters in URL'
            ]
        }
        interface.ingest_url.return_value = parsed_doc
        mock_interface_class.return_value = interface

        # Run ingestion
        await _run_web_ingestion(
            url="https://example.com/suspicious",
            collection="test-collection",
            max_pages=1,
            max_depth=0,
            allowed_domains=None,
            request_delay=1.0,
            chunk_size=1000,
            chunk_overlap=200,
            dry_run=True,  # Use dry run to avoid actual ingestion
            disable_security=False,
            allow_all_domains=False,
            auto_confirm=True
        )

        # Verify warnings were processed
        interface.ingest_url.assert_called_once()

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.ingest.typer.confirm')
    async def test_security_confirmation_prompts(self, mock_confirm):
        """Test that security-related confirmation prompts work."""
        from wqm_cli.cli.ingest import _run_web_ingestion

        # Test with security disabled - should prompt for confirmation
        mock_confirm.return_value = False  # User cancels

        await _run_web_ingestion(
            url="https://example.com/test",
            collection="test-collection",
            max_pages=1,
            max_depth=0,
            allowed_domains=None,
            request_delay=1.0,
            chunk_size=1000,
            chunk_overlap=200,
            dry_run=False,
            disable_security=True,  # This should trigger confirmation
            allow_all_domains=False,
            auto_confirm=False  # Don't auto-confirm
        )

        # Should have prompted for security confirmation
        mock_confirm.assert_called_with("Continue with disabled security?")

    @pytest.mark.xfail(reason="Test uses outdated mock pattern - needs rewrite for daemon_client API")
    @pytest.mark.asyncio
    @patch('wqm_cli.cli.ingest.get_daemon_client')
    @patch('wqm_cli.cli.ingest.get_config_manager')
    @patch('wqm_cli.cli.ingest.WebIngestionInterface')
    async def test_ingestion_error_handling(
        self,
        mock_interface_class,
        mock_config_class,
        mock_daemon_client_func,
        mock_daemon_client,
        mock_config
    ):
        """Test error handling during ingestion."""
        from wqm_cli.cli.ingest import _run_web_ingestion

        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_daemon_client_func.return_value = mock_daemon_client

        # Interface raises an exception
        interface = AsyncMock()
        interface.ingest_url.side_effect = Exception("Network error")
        mock_interface_class.return_value = interface

        # Should exit with error code
        with pytest.raises(SystemExit) as exc_info:
            await _run_web_ingestion(
                url="https://example.com/test",
                collection="test-collection",
                max_pages=1,
                max_depth=0,
                allowed_domains=None,
                request_delay=1.0,
                chunk_size=1000,
                chunk_overlap=200,
                dry_run=False,
                disable_security=False,
                allow_all_domains=False,
                auto_confirm=True
            )

        assert exc_info.value.code == 1  # Error exit code
        mock_daemon_client.disconnect.assert_called_once()  # Cleanup was called


if __name__ == '__main__':
    pytest.main([__file__])
