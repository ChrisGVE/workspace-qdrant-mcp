"""
Custom test assertions and validators.

Provides domain-specific assertions for testing workspace-qdrant-mcp
components including search results, collections, and protocol compliance.
"""

from typing import Dict, List, Any, Optional


def assert_search_results_valid(
    results: List[Dict[str, Any]],
    min_score: float = 0.0,
    expected_fields: Optional[List[str]] = None,
):
    """
    Assert search results are valid and well-formed.

    Args:
        results: Search results to validate
        min_score: Minimum expected score
        expected_fields: Required fields in each result

    Raises:
        AssertionError: If validation fails
    """
    assert isinstance(results, list), "Results must be a list"

    if expected_fields is None:
        expected_fields = ["id", "score", "payload"]

    for i, result in enumerate(results):
        assert isinstance(
            result, dict
        ), f"Result {i} must be a dict, got {type(result)}"

        # Check required fields
        for field in expected_fields:
            assert field in result, f"Result {i} missing required field '{field}'"

        # Validate score
        if "score" in result:
            score = result["score"]
            assert isinstance(
                score, (int, float)
            ), f"Score must be numeric, got {type(score)}"
            assert (
                score >= min_score
            ), f"Score {score} below minimum {min_score} at result {i}"

        # Validate payload
        if "payload" in result:
            assert isinstance(
                result["payload"], dict
            ), f"Payload must be dict at result {i}"


def assert_collection_valid(
    collection: Dict[str, Any],
    expected_name: Optional[str] = None,
    min_points: Optional[int] = None,
):
    """
    Assert collection info is valid.

    Args:
        collection: Collection information dict
        expected_name: Expected collection name
        min_points: Minimum expected number of points

    Raises:
        AssertionError: If validation fails
    """
    assert isinstance(collection, dict), "Collection must be a dict"
    assert "name" in collection, "Collection missing 'name' field"

    if expected_name:
        assert (
            collection["name"] == expected_name
        ), f"Expected name '{expected_name}', got '{collection['name']}'"

    if "points_count" in collection and min_points is not None:
        count = collection["points_count"]
        assert (
            count >= min_points
        ), f"Collection has {count} points, expected at least {min_points}"


def assert_mcp_response_valid(
    response: Dict[str, Any], expected_id: Optional[int] = None
):
    """
    Assert MCP protocol response is valid.

    Args:
        response: MCP response dict
        expected_id: Expected request ID

    Raises:
        AssertionError: If validation fails
    """
    assert isinstance(response, dict), "Response must be a dict"
    assert "jsonrpc" in response, "Response missing 'jsonrpc' field"
    assert response["jsonrpc"] == "2.0", "Invalid JSON-RPC version"
    assert "id" in response, "Response missing 'id' field"

    if expected_id is not None:
        assert response["id"] == expected_id, f"Expected id {expected_id}"

    # Must have either result or error
    assert (
        "result" in response or "error" in response
    ), "Response must have 'result' or 'error'"

    # Validate error format if present
    if "error" in response:
        error = response["error"]
        assert isinstance(error, dict), "Error must be a dict"
        assert "code" in error, "Error missing 'code' field"
        assert "message" in error, "Error missing 'message' field"


def assert_tool_definition_valid(tool: Dict[str, Any]):
    """
    Assert MCP tool definition is valid.

    Args:
        tool: Tool definition dict

    Raises:
        AssertionError: If validation fails
    """
    assert isinstance(tool, dict), "Tool definition must be a dict"

    required_fields = ["name", "description", "inputSchema"]
    for field in required_fields:
        assert field in tool, f"Tool missing required field '{field}'"

    # Validate input schema
    schema = tool["inputSchema"]
    assert isinstance(schema, dict), "inputSchema must be a dict"
    assert "type" in schema, "inputSchema missing 'type' field"
    assert schema["type"] == "object", "inputSchema type must be 'object'"

    if "properties" in schema:
        assert isinstance(
            schema["properties"], dict
        ), "inputSchema.properties must be dict"

    if "required" in schema:
        assert isinstance(
            schema["required"], list
        ), "inputSchema.required must be list"


def assert_hybrid_search_results_valid(
    results: List[Dict[str, Any]],
    query: str,
    min_results: int = 0,
    check_relevance: bool = True,
):
    """
    Assert hybrid search results are valid and relevant.

    Args:
        results: Hybrid search results
        query: Original search query
        min_results: Minimum expected number of results
        check_relevance: Whether to check relevance scores

    Raises:
        AssertionError: If validation fails
    """
    assert isinstance(results, list), "Results must be a list"
    assert len(results) >= min_results, (
        f"Expected at least {min_results} results, got {len(results)}"
    )

    for i, result in enumerate(results):
        assert isinstance(result, dict), f"Result {i} must be a dict"

        # Check dense score
        if "dense_score" in result:
            dense_score = result["dense_score"]
            assert isinstance(
                dense_score, (int, float)
            ), f"Dense score must be numeric at result {i}"
            assert 0.0 <= dense_score <= 1.0, f"Dense score out of range at result {i}"

        # Check sparse score
        if "sparse_score" in result:
            sparse_score = result["sparse_score"]
            assert isinstance(
                sparse_score, (int, float)
            ), f"Sparse score must be numeric at result {i}"

        # Check fusion score
        if "fusion_score" in result:
            fusion_score = result["fusion_score"]
            assert isinstance(
                fusion_score, (int, float)
            ), f"Fusion score must be numeric at result {i}"

        # Scores should be in descending order
        if i > 0 and "fusion_score" in result and "fusion_score" in results[i - 1]:
            prev_score = results[i - 1]["fusion_score"]
            curr_score = result["fusion_score"]
            assert (
                curr_score <= prev_score
            ), f"Results not sorted by fusion_score at position {i}"


def assert_document_ingested(
    collection_info: Dict[str, Any], expected_min_documents: int = 1
):
    """
    Assert documents were successfully ingested.

    Args:
        collection_info: Collection information dict
        expected_min_documents: Minimum expected document count

    Raises:
        AssertionError: If validation fails
    """
    assert "points_count" in collection_info, "Collection info missing points_count"

    points_count = collection_info["points_count"]
    assert points_count >= expected_min_documents, (
        f"Expected at least {expected_min_documents} documents, "
        f"found {points_count}"
    )


def assert_vector_dimensions_match(
    vector: List[float], expected_dim: int, vector_name: str = "vector"
):
    """
    Assert vector has expected dimensions.

    Args:
        vector: Vector to check
        expected_dim: Expected number of dimensions
        vector_name: Name for error messages

    Raises:
        AssertionError: If validation fails
    """
    assert isinstance(vector, list), f"{vector_name} must be a list"
    assert all(isinstance(v, (int, float)) for v in vector), (
        f"{vector_name} must contain only numbers"
    )
    assert len(vector) == expected_dim, (
        f"{vector_name} has {len(vector)} dimensions, expected {expected_dim}"
    )


def assert_performance_acceptable(
    duration_ms: float,
    max_duration_ms: float,
    operation: str = "operation",
):
    """
    Assert operation completed within acceptable time.

    Args:
        duration_ms: Actual duration in milliseconds
        max_duration_ms: Maximum acceptable duration
        operation: Operation name for error messages

    Raises:
        AssertionError: If validation fails
    """
    assert duration_ms <= max_duration_ms, (
        f"{operation} took {duration_ms:.2f}ms, "
        f"exceeded limit of {max_duration_ms:.2f}ms"
    )