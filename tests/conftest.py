"""
Pytest configuration and fixtures for workspace-qdrant-mcp tests.

This module provides common test fixtures and configuration
for all test modules in the project.
"""

import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, Mock

import pytest


# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def temp_directory() -> AsyncGenerator[Path, None]:
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="wqm_test_")
    temp_path = Path(temp_dir)
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = Mock()
    client.search = AsyncMock(return_value=[])
    client.upsert = AsyncMock()
    client.delete = AsyncMock()
    client.get_collection = AsyncMock()
    client.create_collection = AsyncMock()
    return client


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    model = Mock()
    model.encode = Mock(return_value=[[0.1, 0.2, 0.3] * 128])  # 384-dim embedding
    return model


@pytest.fixture
async def sample_documents(temp_directory: Path) -> AsyncGenerator[list[Path], None]:
    """Create sample documents for testing."""
    documents = []
    
    # Create various file types
    txt_file = temp_directory / "sample.txt"
    txt_file.write_text("This is a sample text document for testing.")
    documents.append(txt_file)
    
    md_file = temp_directory / "sample.md"
    md_file.write_text("# Sample Markdown\n\nThis is a markdown document.")
    documents.append(md_file)
    
    pdf_placeholder = temp_directory / "sample.pdf" 
    pdf_placeholder.write_bytes(b"PDF placeholder content")
    documents.append(pdf_placeholder)
    
    yield documents


@pytest.fixture
def mock_ingestion_callback():
    """Mock ingestion callback for file watching tests."""
    return AsyncMock()


@pytest.fixture  
def mock_event_callback():
    """Mock event callback for file watching tests."""
    return Mock()


# Test configuration
pytest_plugins = ["pytest_asyncio"]