"""
Qdrant Data Generator

Generates dummy Qdrant database operations data including vectors, points,
collections, search queries, and metadata for testing database interactions.
"""

import random
import uuid
import time
import json
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class VectorSpec:
    """Specification for vector generation."""
    size: int
    value_range: Tuple[float, float] = (-1.0, 1.0)
    distribution: str = "normal"  # normal, uniform, sparse


@dataclass
class CollectionSpec:
    """Specification for collection configuration."""
    name: str
    vector_size: int
    distance_metric: str = "Cosine"
    optimizers_config: Optional[Dict[str, Any]] = None
    wal_config: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None


class QdrantDataGenerator:
    """Generates realistic Qdrant database operation data."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        self.seed = seed or int(time.time())
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.distance_metrics = ["Cosine", "Euclidean", "Dot"]
        self.vector_sizes = [384, 768, 1536, 2048]
        self.collection_types = ["documents", "notes", "scratchbook", "knowledge", "context", "memory"]

    def generate_vector(self, spec: VectorSpec) -> List[float]:
        """Generate a vector according to specification."""
        if spec.distribution == "normal":
            vector = np.random.normal(0, 0.5, spec.size)
        elif spec.distribution == "uniform":
            vector = np.random.uniform(spec.value_range[0], spec.value_range[1], spec.size)
        elif spec.distribution == "sparse":
            vector = np.zeros(spec.size)
            # Make 10% of values non-zero
            non_zero_indices = np.random.choice(spec.size, size=spec.size // 10, replace=False)
            vector[non_zero_indices] = np.random.normal(0, 1.0, len(non_zero_indices))
        else:
            vector = np.random.normal(0, 0.5, spec.size)

        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        # Clamp to range if specified
        vector = np.clip(vector, spec.value_range[0], spec.value_range[1])

        return vector.tolist()

    def generate_point(
        self,
        point_id: Optional[Union[int, str]] = None,
        vector_spec: Optional[VectorSpec] = None,
        include_payload: bool = True,
        include_sparse_vector: bool = False
    ) -> Dict[str, Any]:
        """Generate a Qdrant point (document with vector and metadata)."""
        if point_id is None:
            point_id = str(uuid.uuid4())

        if vector_spec is None:
            vector_spec = VectorSpec(size=random.choice(self.vector_sizes))

        point_data = {
            "id": point_id,
            "vector": self.generate_vector(vector_spec)
        }

        if include_sparse_vector:
            sparse_spec = VectorSpec(size=vector_spec.size, distribution="sparse")
            sparse_vector = self.generate_vector(sparse_spec)
            # Convert to sparse format (indices and values)
            non_zero_indices = np.nonzero(sparse_vector)[0]
            point_data["sparse_vector"] = {
                "indices": non_zero_indices.tolist(),
                "values": [sparse_vector[i] for i in non_zero_indices]
            }

        if include_payload:
            point_data["payload"] = self.generate_payload()

        return point_data

    def generate_payload(self) -> Dict[str, Any]:
        """Generate realistic metadata payload."""
        payload_templates = [
            {
                "title": f"Document {random.randint(1, 10000)}",
                "content": self._generate_content_sample(),
                "author": random.choice(["Alice", "Bob", "Charlie", "Diana", "Edward"]),
                "type": random.choice(["document", "note", "code", "comment", "question"]),
                "language": random.choice(["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]),
                "created_at": int(time.time() - random.randint(0, 86400 * 365)),  # Up to 1 year ago
                "updated_at": int(time.time() - random.randint(0, 86400 * 30)),   # Up to 30 days ago
                "tags": random.sample(["ml", "ai", "python", "rust", "data", "search", "vector", "database"],
                                    random.randint(1, 4)),
                "project": random.choice(["workspace-qdrant-mcp", "test-project", "demo-app", "research-project"]),
                "file_path": f"/src/{random.choice(['main', 'lib', 'utils', 'core'])}.{random.choice(['py', 'rs', 'js', 'md'])}",
                "size_bytes": random.randint(100, 100000),
                "complexity_score": random.uniform(0.1, 1.0),
                "importance": random.choice(["low", "medium", "high", "critical"]),
                "is_public": random.choice([True, False])
            },
            {
                "function_name": f"process_{random.choice(['data', 'request', 'response', 'query'])}",
                "module": f"{random.choice(['core', 'utils', 'api', 'db'])}.{random.choice(['py', 'rs', 'js'])}",
                "line_start": random.randint(1, 1000),
                "line_end": random.randint(1001, 2000),
                "parameters": random.randint(0, 8),
                "return_type": random.choice(["void", "str", "int", "bool", "Dict", "List", "Optional"]),
                "complexity": random.randint(1, 10),
                "test_coverage": random.uniform(0.0, 1.0),
                "documentation_quality": random.choice(["poor", "fair", "good", "excellent"]),
                "last_modified": int(time.time() - random.randint(0, 86400 * 7)),
                "git_hash": uuid.uuid4().hex[:8],
                "branch": random.choice(["main", "develop", "feature/test", "hotfix/urgent"])
            },
            {
                "note_id": str(uuid.uuid4()),
                "category": random.choice(["meeting", "idea", "todo", "reference", "research"]),
                "priority": random.randint(1, 5),
                "status": random.choice(["draft", "active", "completed", "archived"]),
                "related_notes": [str(uuid.uuid4()) for _ in range(random.randint(0, 3))],
                "mentions": [f"@{name}" for name in random.sample(["alice", "bob", "charlie"], random.randint(0, 2))],
                "due_date": int(time.time() + random.randint(0, 86400 * 30)) if random.random() > 0.7 else None,
                "estimated_hours": random.uniform(0.5, 40.0) if random.random() > 0.5 else None,
                "links": [f"https://example.com/resource_{i}" for i in range(random.randint(0, 3))],
                "location": random.choice(["office", "home", "client_site", "remote"]) if random.random() > 0.8 else None
            }
        ]

        base_payload = random.choice(payload_templates)

        # Add some common fields to all payloads
        base_payload.update({
            "indexed_at": int(time.time()),
            "vector_model": random.choice(["all-MiniLM-L6-v2", "all-mpnet-base-v2", "text-embedding-ada-002"]),
            "processing_version": f"v{random.randint(1, 5)}.{random.randint(0, 10)}.{random.randint(0, 20)}",
            "checksum": uuid.uuid4().hex[:16]
        })

        return base_payload

    def generate_collection_config(self, collection_spec: CollectionSpec) -> Dict[str, Any]:
        """Generate Qdrant collection configuration."""
        config = {
            "vectors": {
                "size": collection_spec.vector_size,
                "distance": collection_spec.distance_metric
            },
            "optimizers_config": collection_spec.optimizers_config or {
                "deleted_threshold": random.uniform(0.1, 0.3),
                "vacuum_min_vector_number": random.randint(1000, 10000),
                "default_segment_number": random.randint(1, 8),
                "max_segment_size": random.randint(100000, 1000000),
                "memmap_threshold": random.randint(50000, 200000),
                "indexing_threshold": random.randint(10000, 100000),
                "flush_interval_sec": random.randint(1, 10),
                "max_optimization_threads": random.randint(1, 8)
            },
            "wal_config": collection_spec.wal_config or {
                "wal_capacity_mb": random.randint(32, 512),
                "wal_segments_ahead": random.randint(0, 5)
            },
            "hnsw_config": {
                "m": random.choice([16, 32, 64]),
                "ef_construct": random.choice([100, 200, 400]),
                "full_scan_threshold": random.randint(1000, 50000),
                "max_indexing_threads": random.randint(1, 8),
                "on_disk": random.choice([True, False])
            },
            "quantization_config": collection_spec.quantization_config,
            "replication_factor": random.randint(1, 3),
            "write_consistency_factor": random.randint(1, 2),
            "init_from": None,
            "on_disk_payload": random.choice([True, False])
        }

        # Add quantization config randomly
        if random.random() > 0.7 and config["quantization_config"] is None:
            config["quantization_config"] = {
                "scalar": {
                    "type": random.choice(["int8", "uint8"]),
                    "quantile": random.uniform(0.9, 0.99),
                    "always_ram": random.choice([True, False])
                }
            }

        return config

    def generate_search_request(
        self,
        collection_name: Optional[str] = None,
        vector_spec: Optional[VectorSpec] = None,
        include_sparse: bool = False
    ) -> Dict[str, Any]:
        """Generate a search request."""
        if collection_name is None:
            collection_name = f"{random.choice(self.collection_types)}_collection_{random.randint(1, 100)}"

        if vector_spec is None:
            vector_spec = VectorSpec(size=random.choice(self.vector_sizes))

        request = {
            "collection_name": collection_name,
            "vector": self.generate_vector(vector_spec),
            "limit": random.randint(1, 100),
            "offset": random.randint(0, 1000) if random.random() > 0.8 else 0,
            "with_payload": random.choice([True, False]),
            "with_vector": random.choice([True, False]),
            "score_threshold": random.uniform(0.1, 0.9) if random.random() > 0.5 else None
        }

        if include_sparse:
            sparse_spec = VectorSpec(size=vector_spec.size, distribution="sparse")
            sparse_vector = self.generate_vector(sparse_spec)
            non_zero_indices = np.nonzero(sparse_vector)[0]
            request["sparse_vector"] = {
                "indices": non_zero_indices.tolist(),
                "values": [sparse_vector[i] for i in non_zero_indices]
            }

        # Add filters
        if random.random() > 0.4:
            request["filter"] = self.generate_search_filter()

        # Add search params
        if random.random() > 0.6:
            request["params"] = {
                "hnsw_ef": random.randint(64, 512),
                "exact": random.choice([True, False])
            }

        return request

    def generate_search_filter(self) -> Dict[str, Any]:
        """Generate search filter conditions."""
        filter_conditions = []

        # Must conditions
        if random.random() > 0.6:
            must_conditions = []

            # Text match
            if random.random() > 0.7:
                must_conditions.append({
                    "key": "type",
                    "match": {"value": random.choice(["document", "note", "code"])}
                })

            # Range filter
            if random.random() > 0.8:
                must_conditions.append({
                    "key": "created_at",
                    "range": {
                        "gte": int(time.time() - 86400 * random.randint(1, 365)),
                        "lte": int(time.time())
                    }
                })

            # Exist filter
            if random.random() > 0.9:
                must_conditions.append({
                    "key": random.choice(["author", "project", "file_path"]),
                    "exists": {}
                })

            if must_conditions:
                filter_conditions.append({"must": must_conditions})

        # Should conditions
        if random.random() > 0.8:
            should_conditions = []

            # Multiple value match
            should_conditions.append({
                "key": "tags",
                "match": {"any": random.sample(["ml", "ai", "python", "rust", "search"], random.randint(1, 3))}
            })

            if should_conditions:
                filter_conditions.append({"should": should_conditions})

        # Must not conditions
        if random.random() > 0.9:
            must_not_conditions = []

            must_not_conditions.append({
                "key": "status",
                "match": {"value": "archived"}
            })

            if must_not_conditions:
                filter_conditions.append({"must_not": must_not_conditions})

        if len(filter_conditions) == 1:
            return filter_conditions[0]
        elif len(filter_conditions) > 1:
            return {"must": filter_conditions}
        else:
            return {}

    def generate_search_result(
        self,
        query_vector: List[float],
        num_results: int = 10
    ) -> Dict[str, Any]:
        """Generate realistic search results."""
        results = []

        for i in range(num_results):
            # Generate result vector similar to query (higher similarity)
            result_vector = np.array(query_vector) + np.random.normal(0, 0.1, len(query_vector))
            result_vector = result_vector / np.linalg.norm(result_vector)  # Normalize

            # Calculate similarity score
            similarity = np.dot(query_vector, result_vector)
            # Add some randomness and ensure decreasing scores
            score = similarity * random.uniform(0.9, 1.0) * (1.0 - i * 0.05)

            result = {
                "id": str(uuid.uuid4()),
                "version": random.randint(1, 100),
                "score": max(0.0, score),
                "payload": self.generate_payload(),
                "vector": result_vector.tolist() if random.random() > 0.5 else None
            }

            results.append(result)

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return {
            "result": results,
            "status": "ok",
            "time": random.uniform(0.001, 0.1)  # Search time in seconds
        }

    def generate_batch_operation(
        self,
        operation_type: str,
        batch_size: int = 100,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate batch operation request."""
        if collection_name is None:
            collection_name = f"batch_collection_{random.randint(1, 10)}"

        operations = []

        for i in range(batch_size):
            if operation_type == "upsert":
                operations.append({
                    "upsert": {
                        "points": [self.generate_point()]
                    }
                })
            elif operation_type == "delete":
                operations.append({
                    "delete": {
                        "points": [str(uuid.uuid4())]
                    }
                })
            elif operation_type == "update_payload":
                operations.append({
                    "set_payload": {
                        "points": [str(uuid.uuid4())],
                        "payload": {
                            "updated_at": int(time.time()),
                            "batch_id": str(uuid.uuid4())
                        }
                    }
                })

        return {
            "collection_name": collection_name,
            "operations": operations,
            "ordered": random.choice([True, False])
        }

    def generate_hybrid_search_request(
        self,
        collection_name: Optional[str] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> Dict[str, Any]:
        """Generate hybrid search request with dense and sparse vectors."""
        if collection_name is None:
            collection_name = f"hybrid_collection_{random.randint(1, 10)}"

        vector_size = random.choice(self.vector_sizes)

        # Dense vector
        dense_spec = VectorSpec(size=vector_size, distribution="normal")
        dense_vector = self.generate_vector(dense_spec)

        # Sparse vector
        sparse_spec = VectorSpec(size=vector_size, distribution="sparse")
        sparse_vector = self.generate_vector(sparse_spec)
        sparse_indices = np.nonzero(sparse_vector)[0]

        return {
            "collection_name": collection_name,
            "prefetch": [
                {
                    "query": dense_vector,
                    "using": "dense",
                    "limit": random.randint(50, 200)
                },
                {
                    "query": {
                        "indices": sparse_indices.tolist(),
                        "values": [sparse_vector[i] for i in sparse_indices]
                    },
                    "using": "sparse",
                    "limit": random.randint(50, 200)
                }
            ],
            "query": {
                "fusion": "rrf",
                "prefetch": [
                    {"fusion": "rrf", "weight": dense_weight},
                    {"fusion": "rrf", "weight": sparse_weight}
                ]
            },
            "limit": random.randint(1, 50),
            "with_payload": True,
            "with_vector": random.choice([True, False])
        }

    def generate_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Generate collection information response."""
        return {
            "result": {
                "status": random.choice(["green", "yellow", "red"]),
                "optimizer_status": {
                    "ok": random.choice([True, False]),
                    "error": "" if random.random() > 0.1 else "Optimization error occurred"
                },
                "vectors_count": random.randint(0, 1000000),
                "indexed_vectors_count": random.randint(0, 1000000),
                "points_count": random.randint(0, 1000000),
                "segments_count": random.randint(1, 20),
                "config": {
                    "params": {
                        "vectors": {
                            "size": random.choice(self.vector_sizes),
                            "distance": random.choice(self.distance_metrics)
                        },
                        "shard_number": random.randint(1, 8),
                        "replication_factor": random.randint(1, 3)
                    },
                    "hnsw_config": {
                        "m": random.choice([16, 32, 64]),
                        "ef_construct": random.choice([100, 200, 400])
                    },
                    "optimizer_config": {
                        "deleted_threshold": random.uniform(0.1, 0.3),
                        "vacuum_min_vector_number": random.randint(1000, 10000)
                    }
                },
                "payload_schema": {
                    "title": {"data_type": "keyword"},
                    "content": {"data_type": "text"},
                    "author": {"data_type": "keyword"},
                    "created_at": {"data_type": "integer"},
                    "tags": {"data_type": "keyword"},
                    "project": {"data_type": "keyword"}
                }
            },
            "status": "ok",
            "time": random.uniform(0.001, 0.05)
        }

    def generate_cluster_info(self) -> Dict[str, Any]:
        """Generate cluster information."""
        peers = []
        for i in range(random.randint(1, 5)):
            peers.append({
                "uri": f"http://node-{i}:6333",
                "id": random.randint(1000000, 9999999),
                "state": random.choice(["Active", "Dead", "Partial"])
            })

        return {
            "result": {
                "peer_id": random.randint(1000000, 9999999),
                "peers": peers,
                "raft_info": {
                    "term": random.randint(1, 1000),
                    "commit": random.randint(1, 100000),
                    "pending_operations": random.randint(0, 100),
                    "leader": random.randint(1000000, 9999999),
                    "role": random.choice(["Leader", "Follower", "Candidate"])
                }
            },
            "status": "ok",
            "time": random.uniform(0.001, 0.05)
        }

    def _generate_content_sample(self) -> str:
        """Generate sample content for documents."""
        content_samples = [
            "This document describes the implementation of a vector search system using hybrid approaches.",
            "Function documentation for processing user queries and returning relevant results.",
            "Configuration guidelines for setting up the workspace environment and dependencies.",
            "API endpoint specifications for document management and search operations.",
            "Performance optimization strategies for large-scale vector operations.",
            "Error handling patterns for distributed system communication.",
            "Testing methodologies for validating search accuracy and system reliability.",
            "Deployment instructions for production environment setup.",
            "Monitoring and alerting configuration for system health tracking.",
            "Data migration procedures for upgrading to new system versions."
        ]

        base_content = random.choice(content_samples)
        # Add some variation
        additional_info = f" Generated at {time.strftime('%Y-%m-%d %H:%M:%S')} for testing purposes."

        return base_content + additional_info

    def generate_telemetry_data(self) -> Dict[str, Any]:
        """Generate system telemetry and metrics data."""
        return {
            "result": {
                "app": {
                    "name": "qdrant",
                    "version": f"v1.{random.randint(0, 10)}.{random.randint(0, 20)}"
                },
                "collections": {
                    "count": random.randint(1, 100),
                    "vectors_total": random.randint(1000, 10000000),
                    "segments_total": random.randint(10, 1000)
                },
                "cluster": {
                    "enabled": random.choice([True, False]),
                    "peers_total": random.randint(1, 5),
                    "peers_online": random.randint(1, 5)
                },
                "requests": {
                    "rest_per_second": random.uniform(0.0, 1000.0),
                    "grpc_per_second": random.uniform(0.0, 1000.0)
                },
                "system": {
                    "memory_used_bytes": random.randint(100_000_000, 8_000_000_000),
                    "memory_total_bytes": random.randint(4_000_000_000, 32_000_000_000),
                    "disk_used_bytes": random.randint(1_000_000_000, 100_000_000_000),
                    "disk_total_bytes": random.randint(10_000_000_000, 1_000_000_000_000),
                    "cpu_used_percent": random.uniform(0.0, 100.0)
                }
            },
            "status": "ok",
            "time": random.uniform(0.001, 0.05)
        }

    def generate_scroll_request(
        self,
        collection_name: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Generate scroll request for iterating through collection."""
        if collection_name is None:
            collection_name = f"scroll_collection_{random.randint(1, 10)}"

        request = {
            "collection_name": collection_name,
            "limit": batch_size,
            "with_payload": random.choice([True, False]),
            "with_vector": random.choice([True, False])
        }

        # Add optional filters
        if random.random() > 0.5:
            request["filter"] = self.generate_search_filter()

        # Add offset for pagination
        if random.random() > 0.3:
            request["offset"] = str(uuid.uuid4())

        return request

    def generate_recommendation_request(
        self,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate recommendation request."""
        if collection_name is None:
            collection_name = f"recommend_collection_{random.randint(1, 10)}"

        # Generate positive and negative examples
        positive_examples = [str(uuid.uuid4()) for _ in range(random.randint(1, 5))]
        negative_examples = [str(uuid.uuid4()) for _ in range(random.randint(0, 3))]

        return {
            "collection_name": collection_name,
            "positive": positive_examples,
            "negative": negative_examples,
            "limit": random.randint(1, 50),
            "with_payload": random.choice([True, False]),
            "with_vector": random.choice([True, False]),
            "score_threshold": random.uniform(0.1, 0.9) if random.random() > 0.7 else None,
            "using": random.choice(["dense", "sparse"]) if random.random() > 0.8 else None,
            "lookup_from": {
                "collection": collection_name,
                "vector": random.choice(["dense", "sparse"]) if random.random() > 0.9 else None
            } if random.random() > 0.8 else None
        }

    def generate_alias_operations(self) -> Dict[str, Any]:
        """Generate collection alias operations."""
        operations = []
        operation_types = ["create_alias", "delete_alias", "rename_alias"]

        for _ in range(random.randint(1, 3)):
            op_type = random.choice(operation_types)

            if op_type == "create_alias":
                operations.append({
                    "create_alias": {
                        "collection_name": f"collection_{random.randint(1, 100)}",
                        "alias_name": f"alias_{random.randint(1, 100)}"
                    }
                })
            elif op_type == "delete_alias":
                operations.append({
                    "delete_alias": {
                        "alias_name": f"alias_{random.randint(1, 100)}"
                    }
                })
            elif op_type == "rename_alias":
                operations.append({
                    "rename_alias": {
                        "old_alias_name": f"old_alias_{random.randint(1, 100)}",
                        "new_alias_name": f"new_alias_{random.randint(1, 100)}"
                    }
                })

        return {
            "actions": operations
        }

    def generate_snapshot_operations(self, collection_name: str) -> Dict[str, Any]:
        """Generate snapshot creation and management operations."""
        return {
            "collection_name": collection_name,
            "snapshot_description": f"Snapshot created at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "metadata": {
                "created_by": random.choice(["admin", "system", "backup_service"]),
                "backup_type": random.choice(["full", "incremental", "differential"]),
                "compression": random.choice([True, False]),
                "encryption": random.choice([True, False])
            }
        }

    def generate_error_responses(self, error_type: str = "random") -> Dict[str, Any]:
        """Generate various error response scenarios."""
        if error_type == "random":
            error_type = random.choice([
                "collection_not_found", "invalid_vector", "timeout",
                "internal_error", "insufficient_memory", "invalid_filter"
            ])

        error_responses = {
            "collection_not_found": {
                "status": {"error": "Collection not found"},
                "time": random.uniform(0.001, 0.01)
            },
            "invalid_vector": {
                "status": {"error": "Invalid vector dimension"},
                "time": random.uniform(0.001, 0.01)
            },
            "timeout": {
                "status": {"error": "Operation timed out"},
                "time": random.uniform(10.0, 30.0)
            },
            "internal_error": {
                "status": {"error": "Internal server error"},
                "time": random.uniform(0.001, 0.1)
            },
            "insufficient_memory": {
                "status": {"error": "Insufficient memory for operation"},
                "time": random.uniform(0.1, 1.0)
            },
            "invalid_filter": {
                "status": {"error": "Invalid filter condition"},
                "time": random.uniform(0.001, 0.01)
            }
        }

        return error_responses.get(error_type, error_responses["internal_error"])

    def generate_load_test_dataset(
        self,
        num_points: int = 1000,
        vector_size: int = 768,
        include_payload: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate large dataset for load testing."""
        vector_spec = VectorSpec(size=vector_size)
        points = []

        for i in range(num_points):
            point = self.generate_point(
                point_id=i,
                vector_spec=vector_spec,
                include_payload=include_payload
            )
            points.append(point)

        return points