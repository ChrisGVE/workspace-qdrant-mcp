"""
MCP Message Generator

Generates dummy MCP (Model Context Protocol) messages for all 30+ tools
in the workspace-qdrant-mcp server including document management, search,
collection management, multi-tenant operations, scratchbook, and system tools.
"""

import random
import uuid
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import json


@dataclass
class McpToolSpec:
    """Specification for an MCP tool and its parameters."""
    name: str
    category: str
    parameters: Dict[str, Dict[str, Any]]
    description: str


class McpMessageGenerator:
    """Generates realistic MCP messages for all server tools."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        self.seed = seed or int(time.time())
        random.seed(self.seed)

        self._init_tool_specs()

    def _init_tool_specs(self):
        """Initialize specifications for all MCP tools."""
        self.tools = {
            # Document Management Tools
            "add_document": McpToolSpec(
                name="add_document",
                category="document_management",
                parameters={
                    "content": {"type": "string", "required": True},
                    "title": {"type": "string", "required": False},
                    "metadata": {"type": "object", "required": False},
                    "collection_name": {"type": "string", "required": False}
                },
                description="Add a document to the workspace"
            ),

            "get_document": McpToolSpec(
                name="get_document",
                category="document_management",
                parameters={
                    "document_id": {"type": "string", "required": True},
                    "include_content": {"type": "boolean", "required": False}
                },
                description="Retrieve a document from the workspace"
            ),

            "add_document_with_project_context": McpToolSpec(
                name="add_document_with_project_context",
                category="document_management",
                parameters={
                    "content": {"type": "string", "required": True},
                    "file_path": {"type": "string", "required": True},
                    "project_context": {"type": "object", "required": False}
                },
                description="Add document with automatic project context detection"
            ),

            # Search Operations
            "search_workspace": McpToolSpec(
                name="search_workspace",
                category="search",
                parameters={
                    "query": {"type": "string", "required": True},
                    "limit": {"type": "integer", "required": False},
                    "collection_filter": {"type": "string", "required": False}
                },
                description="Search across workspace collections"
            ),

            "search_workspace_by_project": McpToolSpec(
                name="search_workspace_by_project",
                category="search",
                parameters={
                    "query": {"type": "string", "required": True},
                    "project_name": {"type": "string", "required": True},
                    "workspace_types": {"type": "array", "required": False}
                },
                description="Search within specific project collections"
            ),

            "hybrid_search_advanced": McpToolSpec(
                name="hybrid_search_advanced",
                category="search",
                parameters={
                    "query": {"type": "string", "required": True},
                    "dense_weight": {"type": "number", "required": False},
                    "sparse_weight": {"type": "number", "required": False},
                    "rerank": {"type": "boolean", "required": False},
                    "filters": {"type": "object", "required": False}
                },
                description="Advanced hybrid search with configurable weights"
            ),

            # Collection Management
            "list_collections": McpToolSpec(
                name="list_collections",
                category="collection_management",
                parameters={
                    "include_stats": {"type": "boolean", "required": False},
                    "project_filter": {"type": "string", "required": False}
                },
                description="List all workspace collections"
            ),

            "create_workspace_collection": McpToolSpec(
                name="create_workspace_collection",
                category="collection_management",
                parameters={
                    "collection_name": {"type": "string", "required": True},
                    "project_name": {"type": "string", "required": False},
                    "vector_size": {"type": "integer", "required": False}
                },
                description="Create a new workspace collection"
            ),

            "get_workspace_collection_info": McpToolSpec(
                name="get_workspace_collection_info",
                category="collection_management",
                parameters={
                    "collection_name": {"type": "string", "required": True},
                    "include_vectors": {"type": "boolean", "required": False}
                },
                description="Get detailed collection information"
            ),

            # Multi-Tenant Operations
            "initialize_project_workspace_collections": McpToolSpec(
                name="initialize_project_workspace_collections",
                category="multitenant",
                parameters={
                    "project_name": {"type": "string", "required": True},
                    "workspace_types": {"type": "array", "required": False},
                    "auto_detect_structure": {"type": "boolean", "required": False}
                },
                description="Initialize project workspace collections"
            ),

            # Memory Tools
            "add_memory_collection": McpToolSpec(
                name="add_memory_collection",
                category="memory",
                parameters={
                    "collection_name": {"type": "string", "required": True},
                    "content": {"type": "string", "required": True},
                    "tags": {"type": "array", "required": False}
                },
                description="Add content to memory collection"
            ),

            "search_memory_collections": McpToolSpec(
                name="search_memory_collections",
                category="memory",
                parameters={
                    "query": {"type": "string", "required": True},
                    "collection_names": {"type": "array", "required": False},
                    "limit": {"type": "integer", "required": False}
                },
                description="Search across memory collections"
            ),

            # Scratchbook Tools
            "update_scratchbook": McpToolSpec(
                name="update_scratchbook",
                category="scratchbook",
                parameters={
                    "content": {"type": "string", "required": True},
                    "section": {"type": "string", "required": False},
                    "project_context": {"type": "string", "required": False}
                },
                description="Update scratchbook with new content"
            ),

            "search_scratchbook": McpToolSpec(
                name="search_scratchbook",
                category="scratchbook",
                parameters={
                    "query": {"type": "string", "required": True},
                    "project_filter": {"type": "string", "required": False},
                    "date_range": {"type": "object", "required": False}
                },
                description="Search scratchbook entries"
            ),

            # Watch Management
            "setup_folder_watch": McpToolSpec(
                name="setup_folder_watch",
                category="watch_management",
                parameters={
                    "folder_path": {"type": "string", "required": True},
                    "recursive": {"type": "boolean", "required": False},
                    "file_patterns": {"type": "array", "required": False},
                    "auto_process": {"type": "boolean", "required": False}
                },
                description="Setup folder monitoring for automatic processing"
            ),

            "list_watched_folders": McpToolSpec(
                name="list_watched_folders",
                category="watch_management",
                parameters={
                    "active_only": {"type": "boolean", "required": False}
                },
                description="List all monitored folders"
            ),

            # System Tools
            "workspace_status": McpToolSpec(
                name="workspace_status",
                category="system",
                parameters={
                    "detailed": {"type": "boolean", "required": False},
                    "include_metrics": {"type": "boolean", "required": False}
                },
                description="Get comprehensive workspace status"
            ),

            "get_server_info": McpToolSpec(
                name="get_server_info",
                category="system",
                parameters={
                    "include_config": {"type": "boolean", "required": False}
                },
                description="Get server information and configuration"
            ),

            # Advanced Search Tools
            "type_search": McpToolSpec(
                name="type_search",
                category="advanced_search",
                parameters={
                    "type_query": {"type": "string", "required": True},
                    "language_filter": {"type": "string", "required": False},
                    "project_scope": {"type": "string", "required": False}
                },
                description="Search for type definitions and declarations"
            ),

            "symbol_resolver": McpToolSpec(
                name="symbol_resolver",
                category="advanced_search",
                parameters={
                    "symbol_name": {"type": "string", "required": True},
                    "context_file": {"type": "string", "required": False},
                    "resolution_depth": {"type": "integer", "required": False}
                },
                description="Resolve symbol definitions and references"
            ),

            "code_search": McpToolSpec(
                name="code_search",
                category="advanced_search",
                parameters={
                    "code_query": {"type": "string", "required": True},
                    "language": {"type": "string", "required": False},
                    "file_patterns": {"type": "array", "required": False}
                },
                description="Search for code patterns and implementations"
            ),

            # Dependency Analysis
            "dependency_analyzer": McpToolSpec(
                name="dependency_analyzer",
                category="analysis",
                parameters={
                    "file_path": {"type": "string", "required": True},
                    "analysis_depth": {"type": "integer", "required": False},
                    "include_transitive": {"type": "boolean", "required": False}
                },
                description="Analyze file and project dependencies"
            ),

            # Degradation Aware Tools
            "degradation_aware_search": McpToolSpec(
                name="degradation_aware_search",
                category="resilient_search",
                parameters={
                    "query": {"type": "string", "required": True},
                    "fallback_strategies": {"type": "array", "required": False},
                    "quality_threshold": {"type": "number", "required": False}
                },
                description="Search with automatic quality degradation handling"
            ),

            # Enhanced State Tools
            "enhanced_state_management": McpToolSpec(
                name="enhanced_state_management",
                category="state",
                parameters={
                    "operation": {"type": "string", "required": True},
                    "state_data": {"type": "object", "required": False},
                    "persistence_level": {"type": "string", "required": False}
                },
                description="Advanced workspace state management"
            ),

            # Research Tools
            "research_query": McpToolSpec(
                name="research_query",
                category="research",
                parameters={
                    "research_topic": {"type": "string", "required": True},
                    "sources": {"type": "array", "required": False},
                    "depth": {"type": "string", "required": False}
                },
                description="Perform research queries across collections"
            ),

            # Performance Benchmarking
            "performance_benchmark": McpToolSpec(
                name="performance_benchmark",
                category="performance",
                parameters={
                    "benchmark_type": {"type": "string", "required": True},
                    "iterations": {"type": "integer", "required": False},
                    "concurrency": {"type": "integer", "required": False}
                },
                description="Run performance benchmarks on operations"
            ),

            # gRPC Integration Tools
            "grpc_health_check": McpToolSpec(
                name="grpc_health_check",
                category="grpc_integration",
                parameters={
                    "service_name": {"type": "string", "required": False},
                    "timeout": {"type": "number", "required": False}
                },
                description="Check gRPC service health status"
            ),

            "grpc_service_stats": McpToolSpec(
                name="grpc_service_stats",
                category="grpc_integration",
                parameters={
                    "service_filter": {"type": "array", "required": False},
                    "time_window": {"type": "string", "required": False}
                },
                description="Get gRPC service statistics and metrics"
            )
        }

    def generate_tool_request(self, tool_name: str, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an MCP tool request message.

        Args:
            tool_name: Name of the MCP tool
            custom_params: Custom parameters to override defaults

        Returns:
            MCP tool request message
        """
        if tool_name not in self.tools:
            return self._generate_unknown_tool_request(tool_name, custom_params)

        tool_spec = self.tools[tool_name]

        # Generate realistic parameters
        params = {}
        if custom_params:
            params.update(custom_params)

        # Fill in missing required parameters with realistic values
        for param_name, param_spec in tool_spec.parameters.items():
            if param_name not in params:
                if param_spec.get("required", False):
                    params[param_name] = self._generate_parameter_value(param_name, param_spec)
                elif random.random() > 0.5:  # 50% chance to include optional parameters
                    params[param_name] = self._generate_parameter_value(param_name, param_spec)

        return {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            },
            "meta": {
                "timestamp": int(time.time()),
                "source": "test_client",
                "category": tool_spec.category,
                "description": tool_spec.description
            }
        }

    def _generate_parameter_value(self, param_name: str, param_spec: Dict[str, Any]) -> Any:
        """Generate realistic parameter values based on parameter name and type."""
        param_type = param_spec.get("type", "string")

        if param_type == "string":
            return self._generate_string_parameter(param_name)
        elif param_type == "integer":
            return self._generate_integer_parameter(param_name)
        elif param_type == "number":
            return self._generate_number_parameter(param_name)
        elif param_type == "boolean":
            return random.choice([True, False])
        elif param_type == "array":
            return self._generate_array_parameter(param_name)
        elif param_type == "object":
            return self._generate_object_parameter(param_name)
        else:
            return f"unknown_type_{param_type}"

    def _generate_string_parameter(self, param_name: str) -> str:
        """Generate realistic string parameters based on parameter name."""
        string_generators = {
            "content": lambda: self._generate_document_content(),
            "title": lambda: f"Test Document {random.randint(1, 1000)}",
            "query": lambda: self._generate_search_query(),
            "document_id": lambda: str(uuid.uuid4()),
            "collection_name": lambda: f"test_collection_{random.randint(100, 999)}",
            "project_name": lambda: random.choice(["workspace-qdrant-mcp", "test-project", "demo-app"]),
            "file_path": lambda: f"/tmp/test_file_{random.randint(1, 100)}.txt",
            "folder_path": lambda: f"/tmp/watch_folder_{random.randint(1, 10)}",
            "symbol_name": lambda: random.choice(["function_name", "ClassName", "CONSTANT_VALUE"]),
            "code_query": lambda: random.choice(["def main", "class Config", "import logging"]),
            "research_topic": lambda: "machine learning algorithms for text processing",
            "benchmark_type": lambda: random.choice(["search", "insert", "update", "delete"]),
            "service_name": lambda: random.choice(["DocumentProcessor", "SearchService", "MemoryService"]),
            "type_query": lambda: random.choice(["function", "class", "interface", "struct"]),
            "section": lambda: random.choice(["notes", "ideas", "todo", "research"]),
            "language": lambda: random.choice(["python", "rust", "javascript", "typescript"]),
            "operation": lambda: random.choice(["save", "load", "reset", "backup"]),
            "context_file": lambda: f"/src/main_{random.randint(1, 10)}.py"
        }

        generator = string_generators.get(param_name)
        if generator:
            return generator()
        else:
            return f"test_{param_name}_{random.randint(1, 100)}"

    def _generate_integer_parameter(self, param_name: str) -> int:
        """Generate realistic integer parameters."""
        integer_ranges = {
            "limit": (1, 100),
            "vector_size": [384, 768, 1536],
            "analysis_depth": (1, 5),
            "resolution_depth": (1, 10),
            "iterations": (10, 1000),
            "concurrency": (1, 20)
        }

        if param_name in integer_ranges:
            range_or_list = integer_ranges[param_name]
            if isinstance(range_or_list, list):
                return random.choice(range_or_list)
            else:
                return random.randint(*range_or_list)
        else:
            return random.randint(1, 100)

    def _generate_number_parameter(self, param_name: str) -> float:
        """Generate realistic number parameters."""
        number_ranges = {
            "dense_weight": (0.0, 1.0),
            "sparse_weight": (0.0, 1.0),
            "quality_threshold": (0.5, 1.0),
            "timeout": (1.0, 30.0)
        }

        range_tuple = number_ranges.get(param_name, (0.0, 1.0))
        return random.uniform(*range_tuple)

    def _generate_array_parameter(self, param_name: str) -> List[Any]:
        """Generate realistic array parameters."""
        array_generators = {
            "workspace_types": lambda: random.sample(
                ["docs", "notes", "scratchbook", "knowledge", "context", "memory"],
                random.randint(1, 3)
            ),
            "tags": lambda: [f"tag_{i}" for i in range(random.randint(1, 5))],
            "collection_names": lambda: [f"collection_{i}" for i in range(random.randint(1, 3))],
            "file_patterns": lambda: random.sample(["*.py", "*.rs", "*.js", "*.md", "*.txt"], random.randint(1, 3)),
            "fallback_strategies": lambda: random.sample(["fuzzy", "keyword", "semantic"], random.randint(1, 2)),
            "sources": lambda: random.sample(["documents", "scratchbook", "memory"], random.randint(1, 2)),
            "service_filter": lambda: random.sample(
                ["DocumentProcessor", "SearchService", "MemoryService"],
                random.randint(1, 2)
            )
        }

        generator = array_generators.get(param_name)
        if generator:
            return generator()
        else:
            return [f"item_{i}" for i in range(random.randint(1, 3))]

    def _generate_object_parameter(self, param_name: str) -> Dict[str, Any]:
        """Generate realistic object parameters."""
        object_generators = {
            "metadata": lambda: {
                "author": random.choice(["Alice", "Bob", "Charlie"]),
                "type": random.choice(["document", "note", "code"]),
                "language": random.choice(["python", "rust", "javascript"]),
                "created_at": int(time.time()),
                "version": str(random.randint(1, 10))
            },
            "project_context": lambda: {
                "project_name": random.choice(["workspace-qdrant-mcp", "test-project"]),
                "git_branch": random.choice(["main", "develop", "feature/test"]),
                "current_file": f"/src/test_{random.randint(1, 10)}.py",
                "workspace_root": "/tmp/workspace"
            },
            "filters": lambda: {
                "project_id": random.choice(["proj_1", "proj_2", "proj_3"]),
                "document_type": random.choice(["text", "code", "markdown"]),
                "date_range": {
                    "start": int(time.time() - 86400),
                    "end": int(time.time())
                }
            },
            "date_range": lambda: {
                "start": int(time.time() - 86400 * random.randint(1, 30)),
                "end": int(time.time())
            },
            "state_data": lambda: {
                "current_state": random.choice(["active", "idle", "processing"]),
                "data": {"key": "value", "counter": random.randint(1, 100)},
                "timestamp": int(time.time())
            }
        }

        generator = object_generators.get(param_name)
        if generator:
            return generator()
        else:
            return {"key": "value", "number": random.randint(1, 100)}

    def _generate_document_content(self) -> str:
        """Generate realistic document content."""
        content_templates = [
            "This is a test document containing information about {topic}. "
            "It includes details about implementation, usage examples, and best practices. "
            "The document was created for testing purposes and contains {length} characters.",

            "# {title}\n\n## Overview\n\nThis document describes {topic} implementation. "
            "Key features include:\n\n- Feature 1: High performance\n- Feature 2: Scalability\n"
            "- Feature 3: Reliability\n\n## Usage\n\n```python\n# Example code\ndef main():\n    pass\n```",

            "Project: {project}\nAuthor: {author}\nDate: {date}\n\n"
            "Description: {description}\n\nImplementation notes:\n{notes}",

            "Function documentation for {function_name}:\n\n"
            "Parameters:\n- param1: Description of parameter 1\n- param2: Description of parameter 2\n\n"
            "Returns:\n- return_value: Description of return value\n\n"
            "Examples:\n{examples}"
        ]

        template = random.choice(content_templates)
        variables = {
            "topic": random.choice(["machine learning", "web development", "data processing", "API design"]),
            "title": f"Document {random.randint(1, 1000)}",
            "length": random.randint(100, 5000),
            "project": random.choice(["workspace-qdrant-mcp", "test-project", "demo-app"]),
            "author": random.choice(["Alice", "Bob", "Charlie", "Test User"]),
            "date": time.strftime("%Y-%m-%d"),
            "description": "A test document for communication framework validation",
            "notes": "Implementation details and considerations for the test framework",
            "function_name": random.choice(["process_document", "search_workspace", "create_collection"]),
            "examples": "# Example usage\nresult = function_call()\nprint(result)"
        }

        try:
            return template.format(**variables)
        except KeyError:
            return template

    def _generate_search_query(self) -> str:
        """Generate realistic search queries."""
        queries = [
            "machine learning algorithms for text classification",
            "python async programming patterns",
            "rust memory management best practices",
            "gRPC service implementation examples",
            "vector database query optimization",
            "document processing pipeline design",
            "hybrid search algorithm comparison",
            "multi-tenant architecture patterns",
            "API design principles and practices",
            "microservices communication protocols",
            "error handling strategies in distributed systems",
            "performance monitoring and alerting",
            "data consistency in distributed databases",
            "authentication and authorization patterns",
            "testing strategies for complex systems"
        ]
        return random.choice(queries)

    def _generate_unknown_tool_request(self, tool_name: str, custom_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate request for unknown tool."""
        params = custom_params or {}
        if not params:
            params = {
                "generic_param": f"value_{random.randint(1, 100)}",
                "test_data": random.choice(["test_value_1", "test_value_2", "test_value_3"])
            }

        return {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            },
            "meta": {
                "timestamp": int(time.time()),
                "source": "test_client",
                "category": "unknown",
                "description": f"Unknown tool: {tool_name}"
            }
        }

    def generate_notification(self, callback_type: str, grpc_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate MCP notification message from gRPC callback.

        Args:
            callback_type: Type of callback (document_processed, search_results, etc.)
            grpc_response: gRPC response data

        Returns:
            MCP notification message
        """
        notification_generators = {
            "document_processed": self._generate_document_processed_notification,
            "search_results": self._generate_search_results_notification,
            "error_notification": self._generate_error_notification,
            "progress_update": self._generate_progress_notification,
            "health_status": self._generate_health_notification
        }

        generator = notification_generators.get(callback_type, self._generate_generic_notification)
        return generator(callback_type, grpc_response)

    def _generate_document_processed_notification(self, callback_type: str, grpc_response: Dict[str, Any]) -> Dict[str, Any]:
        """Generate document processing completion notification."""
        return {
            "jsonrpc": "2.0",
            "method": "notifications/document_processed",
            "params": {
                "document_id": grpc_response.get("document_id", str(uuid.uuid4())),
                "status": grpc_response.get("status", "COMPLETED"),
                "processing_time_ms": grpc_response.get("processing_time_ms", random.randint(100, 5000)),
                "extracted_text_length": grpc_response.get("extracted_text_length", random.randint(100, 50000)),
                "embeddings_generated": grpc_response.get("embeddings_generated", True),
                "error_details": grpc_response.get("error_details")
            },
            "meta": {
                "timestamp": int(time.time()),
                "source": "rust_daemon",
                "callback_type": callback_type
            }
        }

    def _generate_search_results_notification(self, callback_type: str, grpc_response: Dict[str, Any]) -> Dict[str, Any]:
        """Generate search results notification."""
        return {
            "jsonrpc": "2.0",
            "method": "notifications/search_completed",
            "params": {
                "query_id": grpc_response.get("query_id", str(uuid.uuid4())),
                "results_count": grpc_response.get("results_count", random.randint(0, 100)),
                "search_time_ms": grpc_response.get("search_time_ms", random.randint(10, 1000)),
                "results": grpc_response.get("results", []),
                "has_more_results": random.choice([True, False])
            },
            "meta": {
                "timestamp": int(time.time()),
                "source": "rust_daemon",
                "callback_type": callback_type
            }
        }

    def _generate_error_notification(self, callback_type: str, grpc_response: Dict[str, Any]) -> Dict[str, Any]:
        """Generate error notification."""
        return {
            "jsonrpc": "2.0",
            "method": "notifications/error",
            "params": {
                "error_code": grpc_response.get("error_code", "INTERNAL_ERROR"),
                "error_message": grpc_response.get("error_message", "An error occurred during processing"),
                "service_name": grpc_response.get("service_name", "unknown_service"),
                "retry_possible": grpc_response.get("retry_possible", False),
                "error_details": {
                    "timestamp": int(time.time()),
                    "context": "test_framework_simulation"
                }
            },
            "meta": {
                "timestamp": int(time.time()),
                "source": "rust_daemon",
                "callback_type": callback_type,
                "severity": "error"
            }
        }

    def _generate_progress_notification(self, callback_type: str, grpc_response: Dict[str, Any]) -> Dict[str, Any]:
        """Generate progress update notification."""
        return {
            "jsonrpc": "2.0",
            "method": "notifications/progress_update",
            "params": {
                "operation_id": grpc_response.get("operation_id", str(uuid.uuid4())),
                "progress_percent": grpc_response.get("progress_percent", random.randint(0, 100)),
                "current_step": grpc_response.get("current_step", "processing"),
                "estimated_remaining_time_ms": grpc_response.get("estimated_remaining_time_ms", random.randint(1000, 60000)),
                "status_message": f"Processing step: {grpc_response.get('current_step', 'processing')}"
            },
            "meta": {
                "timestamp": int(time.time()),
                "source": "rust_daemon",
                "callback_type": callback_type
            }
        }

    def _generate_health_notification(self, callback_type: str, grpc_response: Dict[str, Any]) -> Dict[str, Any]:
        """Generate health status notification."""
        return {
            "jsonrpc": "2.0",
            "method": "notifications/health_status",
            "params": {
                "service_name": grpc_response.get("service_name", "unknown_service"),
                "status": grpc_response.get("status", "HEALTHY"),
                "metrics": grpc_response.get("metrics", {}),
                "health_check_time": int(time.time()),
                "alert_level": random.choice(["info", "warning", "error"])
            },
            "meta": {
                "timestamp": int(time.time()),
                "source": "rust_daemon",
                "callback_type": callback_type
            }
        }

    def _generate_generic_notification(self, callback_type: str, grpc_response: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic notification."""
        return {
            "jsonrpc": "2.0",
            "method": f"notifications/{callback_type}",
            "params": {
                "data": grpc_response,
                "notification_type": callback_type,
                "timestamp": int(time.time())
            },
            "meta": {
                "timestamp": int(time.time()),
                "source": "rust_daemon",
                "callback_type": callback_type
            }
        }

    def generate_tool_response(self, tool_name: str, success: bool = True, response_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate MCP tool response message.

        Args:
            tool_name: Name of the tool
            success: Whether the response indicates success
            response_data: Custom response data

        Returns:
            MCP tool response message
        """
        if success:
            result = response_data or self._generate_success_result(tool_name)
            return {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "result": result,
                "meta": {
                    "timestamp": int(time.time()),
                    "tool_name": tool_name,
                    "execution_time_ms": random.randint(10, 1000)
                }
            }
        else:
            error = response_data or self._generate_error_result(tool_name)
            return {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "error": error,
                "meta": {
                    "timestamp": int(time.time()),
                    "tool_name": tool_name,
                    "execution_time_ms": random.randint(10, 1000)
                }
            }

    def _generate_success_result(self, tool_name: str) -> Dict[str, Any]:
        """Generate success result for tool response."""
        success_generators = {
            "add_document": lambda: {
                "document_id": str(uuid.uuid4()),
                "status": "added",
                "collection": "test_collection",
                "vector_count": random.randint(1, 100)
            },
            "search_workspace": lambda: {
                "results": [
                    {
                        "document_id": str(uuid.uuid4()),
                        "score": random.uniform(0.0, 1.0),
                        "content": "Sample search result content",
                        "metadata": {"type": "document", "author": "test"}
                    }
                    for _ in range(random.randint(1, 10))
                ],
                "total_count": random.randint(1, 100),
                "search_time_ms": random.randint(10, 500)
            },
            "list_collections": lambda: {
                "collections": [
                    {
                        "name": f"collection_{i}",
                        "vector_count": random.randint(0, 1000),
                        "status": "active"
                    }
                    for i in range(random.randint(1, 5))
                ],
                "total_count": random.randint(1, 10)
            },
            "workspace_status": lambda: {
                "status": "healthy",
                "collections": random.randint(1, 10),
                "total_documents": random.randint(100, 10000),
                "memory_usage_mb": random.randint(100, 2048),
                "uptime_seconds": random.randint(0, 86400)
            }
        }

        generator = success_generators.get(tool_name)
        if generator:
            return generator()
        else:
            return {
                "status": "success",
                "message": f"Tool {tool_name} executed successfully",
                "data": {"result": "generic_success_result"}
            }

    def _generate_error_result(self, tool_name: str) -> Dict[str, Any]:
        """Generate error result for tool response."""
        error_codes = [
            -32601,  # Method not found
            -32602,  # Invalid params
            -32603,  # Internal error
            -1001,   # Service unavailable
            -1002,   # Timeout
            -1003    # Resource not found
        ]

        error_messages = [
            "Tool execution failed due to invalid parameters",
            "Service temporarily unavailable",
            "Request timeout exceeded",
            "Resource not found",
            "Internal processing error",
            "Authentication required",
            "Permission denied"
        ]

        return {
            "code": random.choice(error_codes),
            "message": random.choice(error_messages),
            "data": {
                "tool_name": tool_name,
                "error_details": "Simulated error for testing purposes",
                "retry_after": random.randint(1, 60)
            }
        }

    def generate_batch_request(self, tool_names: List[str], batch_size: int = 10) -> Dict[str, Any]:
        """
        Generate batch MCP request with multiple tool calls.

        Args:
            tool_names: List of tool names to include in batch
            batch_size: Number of tool calls in the batch

        Returns:
            Batch MCP request
        """
        batch_requests = []

        for i in range(batch_size):
            tool_name = random.choice(tool_names)
            request = self.generate_tool_request(tool_name)
            request["id"] = f"batch_{i}"
            batch_requests.append(request)

        return {
            "jsonrpc": "2.0",
            "method": "batch_request",
            "params": {
                "requests": batch_requests,
                "batch_id": str(uuid.uuid4()),
                "parallel": random.choice([True, False])
            },
            "meta": {
                "timestamp": int(time.time()),
                "batch_size": batch_size,
                "source": "test_client"
            }
        }

    def get_all_tool_names(self) -> List[str]:
        """Get list of all available tool names."""
        return list(self.tools.keys())

    def get_tools_by_category(self, category: str) -> List[str]:
        """Get all tool names in a specific category."""
        return [name for name, spec in self.tools.items() if spec.category == category]

    def get_tool_categories(self) -> List[str]:
        """Get list of all available categories."""
        return list(set(spec.category for spec in self.tools.values()))