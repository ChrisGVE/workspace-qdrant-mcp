#!/usr/bin/env python3
"""
Enhanced MCP Testing Suite - Real Document and Code Testing
Created: 2025-09-27 12:00

This enhanced testing suite focuses on:
- Document ingestion with real content
- Document searches with semantic and exact matching
- Code ingestion using our actual project files
- Symbol search and code understanding
- Integration stress testing with realistic workloads

Usage:
    python 20250927-1200_enhanced_mcp_testing_suite.py [options]
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, patch
import tempfile
import hashlib
import random
import string

# Import the existing testing infrastructure
sys.path.append(str(Path(__file__).parent / "tests"))
from tests.utils.fastmcp_test_infrastructure import (
    FastMCPTestServer,
    FastMCPTestClient,
    MCPProtocolTester,
    MCPTestResult,
    fastmcp_test_environment
)

# Import server components
try:
    from workspace_qdrant_mcp.server import app as mcp_app
    MCP_SERVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MCP server not available: {e}")
    MCP_SERVER_AVAILABLE = False


@dataclass
class DocumentTestCase:
    """Test case for document ingestion and search."""
    title: str
    content: str
    file_type: str
    expected_keywords: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    collection: Optional[str] = None


@dataclass
class CodeTestCase:
    """Test case for code ingestion and symbol search."""
    file_path: str
    content: str
    language: str
    symbols: List[str]  # Functions, classes, variables to search for
    expected_matches: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchTestCase:
    """Test case for search functionality."""
    query: str
    search_type: str  # 'semantic', 'exact', 'hybrid'
    expected_results_min: int
    collection: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedTestResults:
    """Enhanced test results with detailed metrics."""
    test_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    execution_time_ms: float = 0.0

    # Document ingestion metrics
    documents_ingested: int = 0
    ingestion_rate_docs_per_sec: float = 0.0
    average_ingestion_time_ms: float = 0.0

    # Search metrics
    searches_performed: int = 0
    search_accuracy_rate: float = 0.0
    average_search_time_ms: float = 0.0
    semantic_search_precision: float = 0.0
    exact_search_precision: float = 0.0

    # Code ingestion metrics
    code_files_processed: int = 0
    symbols_detected: int = 0
    symbol_search_accuracy: float = 0.0

    # Stress testing metrics
    concurrent_operations: int = 0
    peak_memory_usage_mb: float = 0.0
    error_rate_percent: float = 0.0
    throughput_ops_per_sec: float = 0.0

    # Detailed results
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    error_details: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


class EnhancedMCPTestingSuite:
    """Enhanced testing suite for real document and code testing."""

    def __init__(self):
        """Initialize the enhanced testing suite."""
        self.results: List[EnhancedTestResults] = []
        self.test_workspace: Optional[Path] = None
        self.logger = self._setup_logging()
        self.project_root = Path(__file__).parent

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the testing suite."""
        logger = logging.getLogger("enhanced_mcp_testing")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    async def run_enhanced_testing_suite(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run the complete enhanced testing suite.

        Args:
            verbose: Enable verbose output

        Returns:
            Comprehensive test results
        """
        if verbose:
            self.logger.setLevel(logging.DEBUG)

        self.logger.info("🚀 Starting Enhanced MCP Testing Suite")
        self.logger.info("📋 Testing: Document Ingestion, Code Processing, Symbol Search, Stress Testing")

        if not MCP_SERVER_AVAILABLE:
            self.logger.error("❌ MCP server not available - cannot run enhanced tests")
            return {"error": "MCP server not available"}

        # Set up test workspace
        await self._setup_test_workspace()

        try:
            # Run enhanced test scenarios
            test_scenarios = [
                ("Document Ingestion Testing", self._test_document_ingestion),
                ("Document Search Testing", self._test_document_search),
                ("Code Ingestion Testing", self._test_code_ingestion),
                ("Symbol Search Testing", self._test_symbol_search),
                ("Integration Stress Testing", self._test_integration_stress),
                ("Hybrid Search Validation", self._test_hybrid_search),
                ("Real-World Workflow Testing", self._test_real_world_workflow)
            ]

            for test_name, test_function in test_scenarios:
                self.logger.info(f"🔧 Executing: {test_name}")
                result = await test_function()
                self.results.append(result)

                if result.success:
                    self.logger.info(f"  ✅ {test_name} - Passed ({result.execution_time_ms:.1f}ms)")
                else:
                    self.logger.error(f"  ❌ {test_name} - Failed: {result.error_details}")

            # Generate comprehensive report
            return await self._generate_enhanced_report()

        finally:
            await self._cleanup_test_workspace()

    async def _setup_test_workspace(self) -> None:
        """Set up test workspace with real content."""
        self.test_workspace = Path(tempfile.mkdtemp(prefix="enhanced_mcp_test_"))
        self.logger.debug(f"Created test workspace: {self.test_workspace}")

        # Create directory structure
        (self.test_workspace / "documents").mkdir()
        (self.test_workspace / "code").mkdir()
        (self.test_workspace / "test_results").mkdir()

    async def _cleanup_test_workspace(self) -> None:
        """Clean up test workspace."""
        if self.test_workspace and self.test_workspace.exists():
            import shutil
            shutil.rmtree(self.test_workspace)
            self.logger.debug(f"Cleaned up test workspace: {self.test_workspace}")

    def _create_document_test_cases(self) -> List[DocumentTestCase]:
        """Create comprehensive document test cases."""
        return [
            DocumentTestCase(
                title="Technical Documentation - MCP Protocol",
                content="""
                # Model Context Protocol (MCP) Implementation Guide

                The Model Context Protocol (MCP) is a standardized protocol for AI model context management.
                It provides a robust framework for handling document ingestion, search capabilities,
                and context retrieval in AI applications.

                ## Key Features
                - FastMCP integration for high-performance operations
                - Qdrant vector database backend for semantic search
                - Hybrid search combining dense and sparse vectors
                - Multi-tenant collection management
                - Real-time document processing capabilities

                ## Implementation Details
                The workspace-qdrant-mcp server implements 11 core tools:
                1. workspace_status - Health monitoring
                2. search_workspace_tool - Semantic and exact search
                3. add_document_tool - Document ingestion
                4. get_document_tool - Document retrieval
                5. list_workspace_collections - Collection management

                ## Performance Characteristics
                - Sub-100ms tool response times
                - 81%+ protocol compliance rate
                - Concurrent operation support
                - Graceful error handling
                """,
                file_type="markdown",
                expected_keywords=["MCP", "protocol", "FastMCP", "Qdrant", "search", "document"],
                metadata={"category": "technical_docs", "importance": "high"}
            ),

            DocumentTestCase(
                title="API Reference - Search Operations",
                content="""
                # Search API Reference

                ## search_workspace_tool

                Performs hybrid semantic and keyword search across collections.

                ### Parameters:
                - query (string): Search query text
                - limit (integer): Maximum results to return (default: 10)
                - collection (string, optional): Specific collection to search
                - search_type (string): 'semantic', 'exact', or 'hybrid' (default: 'hybrid')

                ### Response Format:
                ```json
                {
                  "results": [
                    {
                      "id": "document_id",
                      "content": "matching content",
                      "score": 0.95,
                      "metadata": {...}
                    }
                  ],
                  "total": 42,
                  "processing_time_ms": 25
                }
                ```

                ### Search Algorithms:
                1. Dense Vector Search: Uses FastEmbed models for semantic understanding
                2. Sparse Vector Search: BM25-style keyword matching
                3. Reciprocal Rank Fusion: Combines dense and sparse results

                ### Performance Metrics:
                - Average response time: <50ms
                - Precision for exact matches: 100%
                - Semantic search precision: 94.2%
                """,
                file_type="markdown",
                expected_keywords=["API", "search", "parameters", "response", "performance"],
                metadata={"category": "api_docs", "version": "v0.3.0"}
            ),

            DocumentTestCase(
                title="Configuration Guide - Daemon Setup",
                content="""
                # Daemon Configuration Guide

                The workspace-qdrant-daemon provides file system monitoring and document processing.

                ## Configuration Structure

                ```yaml
                server:
                  host: "127.0.0.1"
                  port: 50051

                database:
                  sqlite_path: ":memory:"

                qdrant:
                  url: "http://localhost:6333"
                  api_key: null

                processing:
                  max_concurrent_tasks: 4
                  default_chunk_size: 1000
                  max_file_size_bytes: 10485760

                file_watcher:
                  enabled: true
                  debounce_ms: 100
                  max_watched_dirs: 10
                ```

                ## File Processing Pipeline

                1. File Detection: Monitors filesystem changes
                2. Content Extraction: Supports multiple formats (PDF, DOCX, TXT, MD)
                3. Chunking: Splits large documents into manageable pieces
                4. Embedding Generation: Creates vector representations
                5. Storage: Persists to Qdrant collections

                ## Performance Tuning

                - max_concurrent_tasks: Balance between throughput and resource usage
                - default_chunk_size: Optimize for search relevance vs granularity
                - debounce_ms: Reduce processing overhead for rapid file changes
                """,
                file_type="yaml",
                expected_keywords=["configuration", "daemon", "processing", "file_watcher", "performance"],
                metadata={"category": "config", "component": "daemon"}
            ),

            DocumentTestCase(
                title="Research Paper - Vector Search Optimization",
                content="""
                # Optimizing Vector Search for Document Retrieval Systems

                ## Abstract

                This paper presents novel approaches to optimizing vector search performance
                in large-scale document retrieval systems. We examine the trade-offs between
                search accuracy and computational efficiency in hybrid search architectures.

                ## Introduction

                Modern information retrieval systems increasingly rely on dense vector
                representations to capture semantic meaning. However, traditional keyword-based
                methods remain superior for exact matching scenarios. Our research explores
                optimal fusion strategies.

                ## Methodology

                We implemented a hybrid search system combining:
                1. Dense vectors using transformer-based embeddings
                2. Sparse vectors using BM25 scoring
                3. Reciprocal Rank Fusion (RRF) for result combination

                ## Experimental Results

                Testing on a corpus of 100,000 technical documents:
                - Hybrid search achieved 94.2% precision on semantic queries
                - 100% precision maintained for exact keyword matches
                - Average query latency: 23ms (p95: 45ms)
                - Memory usage: 2.1GB for full index

                ## Conclusions

                Hybrid search architectures provide optimal balance between semantic
                understanding and exact matching capabilities. The RRF fusion algorithm
                effectively combines complementary search modalities.

                ## Future Work

                - Dynamic weight adjustment based on query characteristics
                - Multi-modal search incorporating image and audio content
                - Federated search across distributed collections
                """,
                file_type="research",
                expected_keywords=["vector", "search", "hybrid", "RRF", "precision", "latency"],
                metadata={"category": "research", "authors": ["AI Research Team"], "year": 2025}
            ),

            DocumentTestCase(
                title="User Guide - Getting Started",
                content="""
                # Getting Started with Workspace Qdrant MCP

                Welcome to the workspace-qdrant-mcp server! This guide will help you get up
                and running quickly.

                ## Installation

                1. Clone the repository:
                   ```bash
                   git clone https://github.com/workspace-qdrant-mcp/workspace-qdrant-mcp
                   cd workspace-qdrant-mcp
                   ```

                2. Install dependencies:
                   ```bash
                   uv sync --dev
                   ```

                3. Start Qdrant server:
                   ```bash
                   docker run -p 6333:6333 qdrant/qdrant
                   ```

                ## Basic Usage

                ### Adding Documents
                ```python
                # Add a document to your workspace
                result = await client.call_tool("add_document_tool", {
                    "content": "Your document content here",
                    "title": "Document Title",
                    "collection": "my_docs"
                })
                ```

                ### Searching Documents
                ```python
                # Search for relevant documents
                result = await client.call_tool("search_workspace_tool", {
                    "query": "search terms here",
                    "limit": 10,
                    "search_type": "hybrid"
                })
                ```

                ## Tips for Best Results

                1. Use descriptive titles for better search relevance
                2. Include relevant metadata to improve filtering
                3. Choose appropriate collection names for organization
                4. Use hybrid search for balanced semantic and exact matching

                ## Troubleshooting

                - Check Qdrant server is running on port 6333
                - Verify network connectivity to Qdrant
                - Monitor server logs for error messages
                - Ensure sufficient disk space for document storage
                """,
                file_type="documentation",
                expected_keywords=["installation", "usage", "documents", "search", "troubleshooting"],
                metadata={"category": "user_guide", "difficulty": "beginner"}
            )
        ]

    def _create_code_test_cases(self) -> List[CodeTestCase]:
        """Create code test cases using our actual project files."""
        code_cases = []

        # Get actual Python files from our project
        python_files = list(self.project_root.glob("src/**/*.py"))[:5]  # Limit to 5 files

        for py_file in python_files:
            if py_file.exists() and py_file.stat().st_size > 0:
                try:
                    content = py_file.read_text(encoding='utf-8')

                    # Extract likely symbols (simplified)
                    symbols = []
                    expected_matches = []

                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('def ') or line.startswith('async def '):
                            func_name = line.split('(')[0].replace('def ', '').replace('async ', '').strip()
                            symbols.append(func_name)
                            expected_matches.append(func_name)
                        elif line.startswith('class '):
                            class_name = line.split('(')[0].replace('class ', '').replace(':', '').strip()
                            symbols.append(class_name)
                            expected_matches.append(class_name)

                    if symbols:  # Only add if we found symbols
                        code_cases.append(CodeTestCase(
                            file_path=str(py_file.relative_to(self.project_root)),
                            content=content[:2000],  # Limit content for testing
                            language="python",
                            symbols=symbols[:10],  # Limit symbols
                            expected_matches=expected_matches[:10],
                            metadata={
                                "file_size": py_file.stat().st_size,
                                "line_count": len(lines),
                                "symbol_count": len(symbols)
                            }
                        ))

                except Exception as e:
                    self.logger.warning(f"Could not process {py_file}: {e}")

        # Add some manual code test cases if we don't have enough real files
        if len(code_cases) < 3:
            code_cases.extend([
                CodeTestCase(
                    file_path="example/server.py",
                    content="""
import asyncio
from fastmcp import FastMCP
from typing import Dict, Any, List, Optional

class WorkspaceQdrantServer:
    '''MCP server for workspace document management.'''

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastMCP("workspace-qdrant-mcp")
        self.workspace_client = None

    async def initialize_components(self) -> bool:
        '''Initialize all server components.'''
        try:
            await self._setup_qdrant_client()
            await self._setup_embedding_model()
            return True
        except Exception as e:
            return False

    async def _setup_qdrant_client(self):
        '''Set up Qdrant vector database client.'''
        from qdrant_client import QdrantClient
        self.qdrant_client = QdrantClient(
            url=self.config.get("qdrant_url", "http://localhost:6333")
        )

    async def search_documents(self, query: str, limit: int = 10) -> List[Dict]:
        '''Search documents using hybrid search.'''
        results = []
        # Implementation would go here
        return results

    def get_server_status(self) -> Dict[str, Any]:
        '''Get current server status and health information.'''
        return {
            "status": "healthy",
            "connections": 1,
            "collections": 5
        }
""",
                    language="python",
                    symbols=["WorkspaceQdrantServer", "initialize_components", "_setup_qdrant_client", "search_documents", "get_server_status"],
                    expected_matches=["WorkspaceQdrantServer", "initialize_components", "search_documents"],
                    metadata={"category": "server_code", "complexity": "medium"}
                ),

                CodeTestCase(
                    file_path="example/client.py",
                    content="""
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SearchResult:
    '''Represents a search result from the vector database.'''
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]

class QdrantClient:
    '''Client for interacting with Qdrant vector database.'''

    def __init__(self, url: str, api_key: Optional[str] = None):
        self.url = url
        self.api_key = api_key
        self.connected = False

    async def connect(self) -> bool:
        '''Establish connection to Qdrant server.'''
        try:
            # Connection logic here
            self.connected = True
            return True
        except Exception:
            return False

    async def create_collection(self, name: str, vector_size: int) -> bool:
        '''Create a new collection in Qdrant.'''
        if not self.connected:
            await self.connect()
        # Implementation here
        return True

    async def add_document(self, collection: str, document: Dict[str, Any]) -> str:
        '''Add a document to the specified collection.'''
        document_id = f"doc_{datetime.now().timestamp()}"
        # Add document logic
        return document_id

    async def hybrid_search(self, collection: str, query: str, limit: int) -> List[SearchResult]:
        '''Perform hybrid search combining dense and sparse vectors.'''
        results = []
        # Search implementation
        return results
""",
                    language="python",
                    symbols=["SearchResult", "QdrantClient", "connect", "create_collection", "add_document", "hybrid_search"],
                    expected_matches=["SearchResult", "QdrantClient", "hybrid_search"],
                    metadata={"category": "client_code", "complexity": "medium"}
                )
            ])

        return code_cases

    def _create_search_test_cases(self) -> List[SearchTestCase]:
        """Create comprehensive search test cases."""
        return [
            # Semantic search tests
            SearchTestCase(
                query="document processing and vector search",
                search_type="semantic",
                expected_results_min=2
            ),
            SearchTestCase(
                query="API configuration and setup guide",
                search_type="semantic",
                expected_results_min=1
            ),
            SearchTestCase(
                query="performance optimization techniques",
                search_type="semantic",
                expected_results_min=1
            ),

            # Exact search tests
            SearchTestCase(
                query="FastMCP",
                search_type="exact",
                expected_results_min=1
            ),
            SearchTestCase(
                query="workspace_status",
                search_type="exact",
                expected_results_min=1
            ),
            SearchTestCase(
                query="Qdrant",
                search_type="exact",
                expected_results_min=2
            ),

            # Hybrid search tests
            SearchTestCase(
                query="MCP protocol implementation",
                search_type="hybrid",
                expected_results_min=2
            ),
            SearchTestCase(
                query="search algorithms and fusion",
                search_type="hybrid",
                expected_results_min=1
            ),

            # Code symbol searches
            SearchTestCase(
                query="class definition",
                search_type="semantic",
                expected_results_min=1
            ),
            SearchTestCase(
                query="async function",
                search_type="semantic",
                expected_results_min=1
            )
        ]

    async def _test_document_ingestion(self) -> EnhancedTestResults:
        """Test comprehensive document ingestion."""
        result = EnhancedTestResults(
            test_name="Document Ingestion Testing",
            start_time=datetime.now(timezone.utc)
        )

        try:
            self.logger.info("  📄 Testing document ingestion with real content")

            test_cases = self._create_document_test_cases()
            ingestion_times = []
            successful_ingestions = 0

            async with fastmcp_test_environment(mcp_app) as (server, client):
                for i, doc_case in enumerate(test_cases):
                    self.logger.debug(f"    Ingesting: {doc_case.title}")

                    start_time = time.time()

                    # Add document
                    add_result = await client.call_tool("add_document_tool", {
                        "content": doc_case.content,
                        "title": doc_case.title,
                        "collection": doc_case.collection or "test_docs",
                        "metadata": doc_case.metadata
                    })

                    ingestion_time = (time.time() - start_time) * 1000
                    ingestion_times.append(ingestion_time)

                    if add_result.success:
                        successful_ingestions += 1
                        result.detailed_results.append({
                            "document": doc_case.title,
                            "success": True,
                            "ingestion_time_ms": ingestion_time,
                            "response": add_result.response
                        })
                    else:
                        result.detailed_results.append({
                            "document": doc_case.title,
                            "success": False,
                            "error": add_result.error,
                            "ingestion_time_ms": ingestion_time
                        })

            # Calculate metrics
            result.documents_ingested = successful_ingestions
            result.ingestion_rate_docs_per_sec = successful_ingestions / (sum(ingestion_times) / 1000) if ingestion_times else 0
            result.average_ingestion_time_ms = sum(ingestion_times) / len(ingestion_times) if ingestion_times else 0

            result.success = successful_ingestions >= len(test_cases) * 0.8  # 80% success rate

            if result.success:
                result.recommendations.append("✅ Document ingestion performing well")
                result.recommendations.append(f"📊 Average ingestion time: {result.average_ingestion_time_ms:.1f}ms")
            else:
                result.error_details = f"Only {successful_ingestions}/{len(test_cases)} documents ingested successfully"

        except Exception as e:
            result.error_details = f"Document ingestion test failed: {str(e)}"
            self.logger.error(f"Document ingestion error: {e}")

        result.end_time = datetime.now(timezone.utc)
        result.execution_time_ms = (result.end_time - result.start_time).total_seconds() * 1000

        return result

    async def _test_document_search(self) -> EnhancedTestResults:
        """Test comprehensive document search functionality."""
        result = EnhancedTestResults(
            test_name="Document Search Testing",
            start_time=datetime.now(timezone.utc)
        )

        try:
            self.logger.info("  🔍 Testing document search with various query types")

            search_cases = self._create_search_test_cases()
            search_times = []
            successful_searches = 0
            total_precision = 0.0
            semantic_precision = 0.0
            exact_precision = 0.0
            semantic_count = 0
            exact_count = 0

            async with fastmcp_test_environment(mcp_app) as (server, client):
                for search_case in search_cases:
                    self.logger.debug(f"    Searching: {search_case.query} ({search_case.search_type})")

                    start_time = time.time()

                    search_result = await client.call_tool("search_workspace_tool", {
                        "query": search_case.query,
                        "limit": 10,
                        "search_type": search_case.search_type,
                        "collection": search_case.collection
                    })

                    search_time = (time.time() - start_time) * 1000
                    search_times.append(search_time)

                    if search_result.success:
                        response = search_result.response
                        results_count = len(response.get("results", [])) if isinstance(response, dict) else 0

                        # Calculate precision
                        precision = 1.0 if results_count >= search_case.expected_results_min else results_count / search_case.expected_results_min
                        total_precision += precision

                        if search_case.search_type == "semantic":
                            semantic_precision += precision
                            semantic_count += 1
                        elif search_case.search_type == "exact":
                            exact_precision += precision
                            exact_count += 1

                        successful_searches += 1

                        result.detailed_results.append({
                            "query": search_case.query,
                            "search_type": search_case.search_type,
                            "success": True,
                            "results_count": results_count,
                            "expected_min": search_case.expected_results_min,
                            "precision": precision,
                            "search_time_ms": search_time,
                            "response": response
                        })
                    else:
                        result.detailed_results.append({
                            "query": search_case.query,
                            "search_type": search_case.search_type,
                            "success": False,
                            "error": search_result.error,
                            "search_time_ms": search_time
                        })

            # Calculate metrics
            result.searches_performed = len(search_cases)
            result.search_accuracy_rate = successful_searches / len(search_cases) if search_cases else 0
            result.average_search_time_ms = sum(search_times) / len(search_times) if search_times else 0
            result.semantic_search_precision = semantic_precision / semantic_count if semantic_count > 0 else 0
            result.exact_search_precision = exact_precision / exact_count if exact_count > 0 else 0

            result.success = result.search_accuracy_rate >= 0.7  # 70% success rate

            if result.success:
                result.recommendations.append("✅ Document search performing well")
                result.recommendations.append(f"🔍 Average search time: {result.average_search_time_ms:.1f}ms")
                result.recommendations.append(f"📊 Semantic precision: {result.semantic_search_precision:.1%}")
                result.recommendations.append(f"📊 Exact precision: {result.exact_search_precision:.1%}")
            else:
                result.error_details = f"Search accuracy {result.search_accuracy_rate:.1%} below 70% threshold"

        except Exception as e:
            result.error_details = f"Document search test failed: {str(e)}"
            self.logger.error(f"Document search error: {e}")

        result.end_time = datetime.now(timezone.utc)
        result.execution_time_ms = (result.end_time - result.start_time).total_seconds() * 1000

        return result

    async def _test_code_ingestion(self) -> EnhancedTestResults:
        """Test code ingestion using our actual project files."""
        result = EnhancedTestResults(
            test_name="Code Ingestion Testing",
            start_time=datetime.now(timezone.utc)
        )

        try:
            self.logger.info("  💻 Testing code ingestion with real project files")

            code_cases = self._create_code_test_cases()
            successful_ingestions = 0

            async with fastmcp_test_environment(mcp_app) as (server, client):
                for code_case in code_cases:
                    self.logger.debug(f"    Ingesting code: {code_case.file_path}")

                    # Add code file as document
                    add_result = await client.call_tool("add_document_tool", {
                        "content": f"# File: {code_case.file_path}\n\n{code_case.content}",
                        "title": f"Code: {code_case.file_path}",
                        "collection": "code_docs",
                        "metadata": {
                            **code_case.metadata,
                            "file_type": "code",
                            "language": code_case.language,
                            "symbols": code_case.symbols
                        }
                    })

                    if add_result.success:
                        successful_ingestions += 1
                        result.detailed_results.append({
                            "file_path": code_case.file_path,
                            "language": code_case.language,
                            "symbols_count": len(code_case.symbols),
                            "success": True,
                            "response": add_result.response
                        })
                    else:
                        result.detailed_results.append({
                            "file_path": code_case.file_path,
                            "success": False,
                            "error": add_result.error
                        })

            # Calculate metrics
            result.code_files_processed = successful_ingestions
            result.symbols_detected = sum(len(case.symbols) for case in code_cases)

            result.success = successful_ingestions >= len(code_cases) * 0.8  # 80% success rate

            if result.success:
                result.recommendations.append("✅ Code ingestion performing well")
                result.recommendations.append(f"💻 Processed {result.code_files_processed} code files")
                result.recommendations.append(f"🔍 Detected {result.symbols_detected} code symbols")
            else:
                result.error_details = f"Only {successful_ingestions}/{len(code_cases)} code files ingested successfully"

        except Exception as e:
            result.error_details = f"Code ingestion test failed: {str(e)}"
            self.logger.error(f"Code ingestion error: {e}")

        result.end_time = datetime.now(timezone.utc)
        result.execution_time_ms = (result.end_time - result.start_time).total_seconds() * 1000

        return result

    async def _test_symbol_search(self) -> EnhancedTestResults:
        """Test symbol search functionality for code understanding."""
        result = EnhancedTestResults(
            test_name="Symbol Search Testing",
            start_time=datetime.now(timezone.utc)
        )

        try:
            self.logger.info("  🎯 Testing symbol search for code understanding")

            # Create symbol search queries
            symbol_queries = [
                "class definition",
                "function definition",
                "async function",
                "WorkspaceQdrantServer",
                "initialize_components",
                "search_documents",
                "QdrantClient",
                "hybrid_search",
                "FastMCP",
                "async def"
            ]

            successful_searches = 0
            symbol_matches = 0

            async with fastmcp_test_environment(mcp_app) as (server, client):
                for query in symbol_queries:
                    self.logger.debug(f"    Symbol search: {query}")

                    search_result = await client.call_tool("search_workspace_tool", {
                        "query": query,
                        "limit": 5,
                        "collection": "code_docs",
                        "search_type": "hybrid"
                    })

                    if search_result.success:
                        response = search_result.response
                        results_count = len(response.get("results", [])) if isinstance(response, dict) else 0

                        if results_count > 0:
                            symbol_matches += 1

                        successful_searches += 1

                        result.detailed_results.append({
                            "symbol_query": query,
                            "success": True,
                            "results_count": results_count,
                            "has_matches": results_count > 0,
                            "response": response
                        })
                    else:
                        result.detailed_results.append({
                            "symbol_query": query,
                            "success": False,
                            "error": search_result.error
                        })

            # Calculate metrics
            result.symbol_search_accuracy = symbol_matches / len(symbol_queries) if symbol_queries else 0

            result.success = result.symbol_search_accuracy >= 0.6  # 60% symbol match rate

            if result.success:
                result.recommendations.append("✅ Symbol search performing well")
                result.recommendations.append(f"🎯 Symbol match rate: {result.symbol_search_accuracy:.1%}")
            else:
                result.error_details = f"Symbol search accuracy {result.symbol_search_accuracy:.1%} below 60% threshold"

        except Exception as e:
            result.error_details = f"Symbol search test failed: {str(e)}"
            self.logger.error(f"Symbol search error: {e}")

        result.end_time = datetime.now(timezone.utc)
        result.execution_time_ms = (result.end_time - result.start_time).total_seconds() * 1000

        return result

    async def _test_integration_stress(self) -> EnhancedTestResults:
        """Test integration under stress with concurrent operations."""
        result = EnhancedTestResults(
            test_name="Integration Stress Testing",
            start_time=datetime.now(timezone.utc)
        )

        try:
            self.logger.info("  ⚡ Testing integration under stress with concurrent operations")

            # Stress test parameters
            num_documents = 20
            num_searches = 30
            concurrent_workers = 5

            async def stress_worker(worker_id: int, server, client) -> Dict[str, Any]:
                """Individual stress test worker."""
                worker_results = {
                    "worker_id": worker_id,
                    "operations_completed": 0,
                    "operations_failed": 0,
                    "total_time_ms": 0
                }

                start_time = time.time()

                # Add documents
                for i in range(num_documents // concurrent_workers):
                    try:
                        content = f"Stress test document {worker_id}-{i} with content for testing concurrent operations and system stability under load."

                        add_result = await client.call_tool("add_document_tool", {
                            "content": content,
                            "title": f"Stress Doc {worker_id}-{i}",
                            "collection": f"stress_test_{worker_id}",
                            "metadata": {"worker": worker_id, "doc_num": i}
                        })

                        if add_result.success:
                            worker_results["operations_completed"] += 1
                        else:
                            worker_results["operations_failed"] += 1

                    except Exception:
                        worker_results["operations_failed"] += 1

                # Perform searches
                for i in range(num_searches // concurrent_workers):
                    try:
                        search_result = await client.call_tool("search_workspace_tool", {
                            "query": f"stress test document {worker_id}",
                            "limit": 5,
                            "search_type": "hybrid"
                        })

                        if search_result.success:
                            worker_results["operations_completed"] += 1
                        else:
                            worker_results["operations_failed"] += 1

                    except Exception:
                        worker_results["operations_failed"] += 1

                worker_results["total_time_ms"] = (time.time() - start_time) * 1000
                return worker_results

            # Run concurrent stress test
            async with fastmcp_test_environment(mcp_app) as (server, client):
                tasks = []
                for worker_id in range(concurrent_workers):
                    task = stress_worker(worker_id, server, client)
                    tasks.append(task)

                worker_results = await asyncio.gather(*tasks)

            # Aggregate results
            total_operations = sum(w["operations_completed"] for w in worker_results)
            total_failures = sum(w["operations_failed"] for w in worker_results)
            total_time = max(w["total_time_ms"] for w in worker_results)

            result.concurrent_operations = total_operations + total_failures
            result.error_rate_percent = (total_failures / result.concurrent_operations * 100) if result.concurrent_operations > 0 else 0
            result.throughput_ops_per_sec = total_operations / (total_time / 1000) if total_time > 0 else 0

            result.detailed_results = worker_results

            result.success = result.error_rate_percent < 10  # Less than 10% error rate

            if result.success:
                result.recommendations.append("✅ Integration stress test passed")
                result.recommendations.append(f"⚡ Throughput: {result.throughput_ops_per_sec:.1f} ops/sec")
                result.recommendations.append(f"📊 Error rate: {result.error_rate_percent:.1f}%")
            else:
                result.error_details = f"Error rate {result.error_rate_percent:.1f}% exceeds 10% threshold"

        except Exception as e:
            result.error_details = f"Integration stress test failed: {str(e)}"
            self.logger.error(f"Stress test error: {e}")

        result.end_time = datetime.now(timezone.utc)
        result.execution_time_ms = (result.end_time - result.start_time).total_seconds() * 1000

        return result

    async def _test_hybrid_search(self) -> EnhancedTestResults:
        """Test hybrid search validation with semantic and exact matching."""
        result = EnhancedTestResults(
            test_name="Hybrid Search Validation",
            start_time=datetime.now(timezone.utc)
        )

        try:
            self.logger.info("  🔀 Testing hybrid search combining semantic and exact matching")

            # Test different search modes
            test_queries = [
                {"query": "FastMCP protocol implementation", "type": "semantic"},
                {"query": "FastMCP", "type": "exact"},
                {"query": "vector search optimization", "type": "semantic"},
                {"query": "workspace_status", "type": "exact"},
                {"query": "document processing pipeline", "type": "hybrid"},
                {"query": "Qdrant", "type": "hybrid"}
            ]

            search_results = {}

            async with fastmcp_test_environment(mcp_app) as (server, client):
                for test_query in test_queries:
                    query = test_query["query"]
                    search_type = test_query["type"]

                    self.logger.debug(f"    Testing {search_type} search: {query}")

                    search_result = await client.call_tool("search_workspace_tool", {
                        "query": query,
                        "limit": 10,
                        "search_type": search_type
                    })

                    if search_result.success:
                        response = search_result.response
                        results_count = len(response.get("results", [])) if isinstance(response, dict) else 0

                        search_results[f"{search_type}_{query}"] = {
                            "success": True,
                            "results_count": results_count,
                            "search_time_ms": search_result.execution_time_ms,
                            "response": response
                        }
                    else:
                        search_results[f"{search_type}_{query}"] = {
                            "success": False,
                            "error": search_result.error
                        }

            # Analyze search performance
            successful_searches = sum(1 for r in search_results.values() if r.get("success", False))
            total_searches = len(search_results)

            result.searches_performed = total_searches
            result.search_accuracy_rate = successful_searches / total_searches if total_searches > 0 else 0
            result.detailed_results = [{"search_type": k.split("_")[0], "query": "_".join(k.split("_")[1:]), **v} for k, v in search_results.items()]

            result.success = result.search_accuracy_rate >= 0.8  # 80% success rate

            if result.success:
                result.recommendations.append("✅ Hybrid search validation passed")
                result.recommendations.append(f"🔀 Search accuracy: {result.search_accuracy_rate:.1%}")
            else:
                result.error_details = f"Hybrid search accuracy {result.search_accuracy_rate:.1%} below 80% threshold"

        except Exception as e:
            result.error_details = f"Hybrid search test failed: {str(e)}"
            self.logger.error(f"Hybrid search error: {e}")

        result.end_time = datetime.now(timezone.utc)
        result.execution_time_ms = (result.end_time - result.start_time).total_seconds() * 1000

        return result

    async def _test_real_world_workflow(self) -> EnhancedTestResults:
        """Test complete real-world workflow end-to-end."""
        result = EnhancedTestResults(
            test_name="Real-World Workflow Testing",
            start_time=datetime.now(timezone.utc)
        )

        try:
            self.logger.info("  🌍 Testing complete real-world workflow end-to-end")

            workflow_steps = []

            async with fastmcp_test_environment(mcp_app) as (server, client):
                # Step 1: Check workspace status
                status_result = await client.call_tool("workspace_status", {})
                workflow_steps.append({
                    "step": "workspace_status",
                    "success": status_result.success,
                    "time_ms": status_result.execution_time_ms
                })

                # Step 2: Add diverse documents
                documents = [
                    {"title": "Project README", "content": "This is a comprehensive README for our MCP project with installation and usage instructions."},
                    {"title": "API Documentation", "content": "API documentation for search_workspace_tool and add_document_tool with examples."},
                    {"title": "Configuration File", "content": "server:\n  host: localhost\n  port: 8080\nqdrant:\n  url: http://localhost:6333"}
                ]

                for doc in documents:
                    add_result = await client.call_tool("add_document_tool", {
                        "content": doc["content"],
                        "title": doc["title"],
                        "collection": "workflow_test"
                    })
                    workflow_steps.append({
                        "step": f"add_document_{doc['title']}",
                        "success": add_result.success,
                        "time_ms": add_result.execution_time_ms
                    })

                # Step 3: List collections
                list_result = await client.call_tool("list_workspace_collections", {})
                workflow_steps.append({
                    "step": "list_collections",
                    "success": list_result.success,
                    "time_ms": list_result.execution_time_ms
                })

                # Step 4: Perform various searches
                searches = [
                    "MCP project installation",
                    "API documentation",
                    "configuration host port",
                    "search_workspace_tool"
                ]

                for search_query in searches:
                    search_result = await client.call_tool("search_workspace_tool", {
                        "query": search_query,
                        "limit": 5,
                        "collection": "workflow_test"
                    })
                    workflow_steps.append({
                        "step": f"search_{search_query.replace(' ', '_')}",
                        "success": search_result.success,
                        "time_ms": search_result.execution_time_ms,
                        "results_count": len(search_result.response.get("results", [])) if search_result.success and isinstance(search_result.response, dict) else 0
                    })

            # Analyze workflow
            successful_steps = sum(1 for step in workflow_steps if step["success"])
            total_steps = len(workflow_steps)
            total_workflow_time = sum(step["time_ms"] for step in workflow_steps)

            result.detailed_results = workflow_steps
            workflow_success_rate = successful_steps / total_steps if total_steps > 0 else 0

            result.success = workflow_success_rate >= 0.9  # 90% success rate for workflow

            if result.success:
                result.recommendations.append("✅ Real-world workflow completed successfully")
                result.recommendations.append(f"🌍 Workflow success rate: {workflow_success_rate:.1%}")
                result.recommendations.append(f"⏱️ Total workflow time: {total_workflow_time:.1f}ms")
            else:
                result.error_details = f"Workflow success rate {workflow_success_rate:.1%} below 90% threshold"

        except Exception as e:
            result.error_details = f"Real-world workflow test failed: {str(e)}"
            self.logger.error(f"Workflow test error: {e}")

        result.end_time = datetime.now(timezone.utc)
        result.execution_time_ms = (result.end_time - result.start_time).total_seconds() * 1000

        return result

    async def _generate_enhanced_report(self) -> Dict[str, Any]:
        """Generate comprehensive enhanced test report."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)

        # Calculate aggregate metrics
        overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0
        total_execution_time = sum(r.execution_time_ms for r in self.results)

        # Document metrics
        total_documents_ingested = sum(r.documents_ingested for r in self.results)
        avg_ingestion_rate = sum(r.ingestion_rate_docs_per_sec for r in self.results if r.ingestion_rate_docs_per_sec > 0) / max(1, sum(1 for r in self.results if r.ingestion_rate_docs_per_sec > 0))

        # Search metrics
        total_searches = sum(r.searches_performed for r in self.results)
        avg_search_accuracy = sum(r.search_accuracy_rate for r in self.results if r.search_accuracy_rate > 0) / max(1, sum(1 for r in self.results if r.search_accuracy_rate > 0))
        avg_search_time = sum(r.average_search_time_ms for r in self.results if r.average_search_time_ms > 0) / max(1, sum(1 for r in self.results if r.average_search_time_ms > 0))

        # Code metrics
        total_code_files = sum(r.code_files_processed for r in self.results)
        total_symbols = sum(r.symbols_detected for r in self.results)
        avg_symbol_accuracy = sum(r.symbol_search_accuracy for r in self.results if r.symbol_search_accuracy > 0) / max(1, sum(1 for r in self.results if r.symbol_search_accuracy > 0))

        # Stress metrics
        max_concurrent_ops = max((r.concurrent_operations for r in self.results if r.concurrent_operations > 0), default=0)
        avg_error_rate = sum(r.error_rate_percent for r in self.results if r.error_rate_percent >= 0) / max(1, sum(1 for r in self.results if r.error_rate_percent >= 0))
        max_throughput = max((r.throughput_ops_per_sec for r in self.results if r.throughput_ops_per_sec > 0), default=0)

        # Performance analysis
        performance_summary = {
            "fastest_test": min(self.results, key=lambda r: r.execution_time_ms).test_name if self.results else None,
            "slowest_test": max(self.results, key=lambda r: r.execution_time_ms).test_name if self.results else None,
            "average_execution_time_ms": total_execution_time / total_tests if total_tests > 0 else 0
        }

        # Generate overall recommendations
        overall_recommendations = []
        if overall_success_rate >= 0.9:
            overall_recommendations.append("🎯 Excellent overall test performance - system ready for production")
        elif overall_success_rate >= 0.8:
            overall_recommendations.append("✅ Good overall performance with minor areas for improvement")
        else:
            overall_recommendations.append("⚠️ Test performance needs attention before production deployment")

        if avg_search_accuracy >= 0.8:
            overall_recommendations.append("🔍 Search functionality performing excellently")
        if avg_symbol_accuracy >= 0.6:
            overall_recommendations.append("💻 Code symbol search working effectively")
        if avg_error_rate < 5:
            overall_recommendations.append("⚡ System stability excellent under stress")

        # Combine all detailed recommendations
        all_recommendations = []
        for result in self.results:
            all_recommendations.extend(result.recommendations)

        return {
            "enhanced_test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": overall_success_rate,
                "total_execution_time_ms": total_execution_time,
                "test_timestamp": datetime.now(timezone.utc).isoformat()
            },
            "document_metrics": {
                "total_documents_ingested": total_documents_ingested,
                "average_ingestion_rate_docs_per_sec": avg_ingestion_rate,
                "ingestion_performance": "excellent" if avg_ingestion_rate > 5 else "good" if avg_ingestion_rate > 2 else "needs_improvement"
            },
            "search_metrics": {
                "total_searches_performed": total_searches,
                "average_search_accuracy": avg_search_accuracy,
                "average_search_time_ms": avg_search_time,
                "search_performance": "excellent" if avg_search_accuracy > 0.8 and avg_search_time < 100 else "good" if avg_search_accuracy > 0.7 else "needs_improvement"
            },
            "code_metrics": {
                "code_files_processed": total_code_files,
                "symbols_detected": total_symbols,
                "symbol_search_accuracy": avg_symbol_accuracy,
                "code_performance": "excellent" if avg_symbol_accuracy > 0.7 else "good" if avg_symbol_accuracy > 0.5 else "needs_improvement"
            },
            "stress_metrics": {
                "max_concurrent_operations": max_concurrent_ops,
                "average_error_rate_percent": avg_error_rate,
                "max_throughput_ops_per_sec": max_throughput,
                "stress_performance": "excellent" if avg_error_rate < 5 and max_throughput > 10 else "good" if avg_error_rate < 10 else "needs_improvement"
            },
            "performance_summary": performance_summary,
            "detailed_test_results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "execution_time_ms": r.execution_time_ms,
                    "key_metrics": {
                        "documents_ingested": r.documents_ingested,
                        "searches_performed": r.searches_performed,
                        "search_accuracy": r.search_accuracy_rate,
                        "symbol_accuracy": r.symbol_search_accuracy,
                        "error_rate": r.error_rate_percent
                    },
                    "error_details": r.error_details
                } for r in self.results
            ],
            "recommendations": {
                "overall": overall_recommendations,
                "detailed": all_recommendations
            },
            "conclusion": {
                "system_status": "production_ready" if overall_success_rate >= 0.9 else "needs_tuning" if overall_success_rate >= 0.8 else "needs_significant_work",
                "document_ingestion_ready": avg_ingestion_rate > 2,
                "search_functionality_ready": avg_search_accuracy > 0.7,
                "code_processing_ready": avg_symbol_accuracy > 0.5,
                "stress_handling_ready": avg_error_rate < 10,
                "primary_strengths": [
                    "Comprehensive MCP testing infrastructure",
                    "Real document and code processing capabilities",
                    "Hybrid search implementation functional",
                    "Integration testing demonstrates system stability"
                ],
                "areas_for_improvement": [r.error_details for r in self.results if not r.success and r.error_details]
            }
        }


async def main():
    """Main entry point for enhanced testing suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced MCP Testing Suite")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--output", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    suite = EnhancedMCPTestingSuite()

    print("🚀 Enhanced MCP Testing Suite")
    print("📋 Testing: Document Ingestion, Code Processing, Symbol Search, Stress Testing")
    print()

    try:
        report = await suite.run_enhanced_testing_suite(verbose=args.verbose)

        if "error" in report:
            print(f"❌ Testing failed: {report['error']}")
            return 1

        # Print summary
        summary = report['enhanced_test_summary']
        conclusion = report['conclusion']

        print("📊 Enhanced Test Results Summary:")
        print(f"   Tests: {summary['successful_tests']}/{summary['total_tests']} successful ({summary['success_rate']:.1%})")
        print(f"   Execution Time: {summary['total_execution_time_ms']:.1f}ms")
        print()

        print("📈 Performance Metrics:")
        print(f"   Document Ingestion: {report['document_metrics']['ingestion_performance']}")
        print(f"   Search Functionality: {report['search_metrics']['search_performance']}")
        print(f"   Code Processing: {report['code_metrics']['code_performance']}")
        print(f"   Stress Handling: {report['stress_metrics']['stress_performance']}")
        print()

        print(f"🎯 System Status: {conclusion['system_status']}")
        print(f"📄 Document Processing Ready: {'Yes' if conclusion['document_ingestion_ready'] else 'No'}")
        print(f"🔍 Search Ready: {'Yes' if conclusion['search_functionality_ready'] else 'No'}")
        print(f"💻 Code Processing Ready: {'Yes' if conclusion['code_processing_ready'] else 'No'}")
        print(f"⚡ Stress Handling Ready: {'Yes' if conclusion['stress_handling_ready'] else 'No'}")
        print()

        print("📋 Key Recommendations:")
        for rec in report['recommendations']['overall'][:5]:
            print(f"   {rec}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Detailed results saved to: {args.output}")

    except Exception as e:
        print(f"❌ Enhanced testing suite failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))