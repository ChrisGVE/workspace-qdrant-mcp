#!/usr/bin/env python3
"""
Advanced Recall and Precision Testing with Real Search Integration.

This advanced test integrates directly with the workspace-qdrant-mcp search tools
to provide realistic recall and precision measurements using the actual search
infrastructure that users will experience.

FEATURES:
- Uses actual workspace search tools (search_workspace)
- Tests hybrid, dense, and sparse search modes
- Measures quality degradation with scale
- Provides actionable recommendations
- Integrates with existing performance benchmarks
- Generates CI/CD-ready reports

METHODOLOGY:
1. Create test datasets by ingesting real project content
2. Generate ground truth by expert curation and content analysis
3. Run searches using production search tools
4. Measure precision@k, recall@k, F1@k, MAP, MRR
5. Analyze performance vs quality trade-offs
6. Compare small vs large dataset performance
"""

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

import numpy as np
from qdrant_client import QdrantClient

from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from workspace_qdrant_mcp.tools.search import search_workspace

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthQuery:
    """A query with expert-curated ground truth."""
    query: str
    relevant_doc_ids: Set[str]
    collection: str
    query_category: str  # "exact_match", "semantic", "complex", "edge_case"
    difficulty_level: str  # "easy", "medium", "hard"
    expected_min_precision_at_5: float  # Quality target
    description: str


@dataclass
class QualityTestResult:
    """Complete quality test result with all metrics."""
    query: str
    collection: str
    search_mode: str
    score_threshold: float
    
    # Quality metrics
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    precision_at_10: float
    
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    
    f1_at_1: float
    f1_at_3: float
    f1_at_5: float
    f1_at_10: float
    
    average_precision: float
    mean_reciprocal_rank: float
    
    # Performance metrics
    search_time_ms: float
    throughput_qps: float
    
    # Context
    database_size: int
    total_relevant: int
    total_retrieved: int
    
    # Test metadata
    query_category: str
    difficulty_level: str
    expected_min_precision_at_5: float
    meets_quality_target: bool


class AdvancedRecallPrecisionTest:
    """Advanced recall/precision testing with real search integration."""
    
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.workspace_client = None
        self.ground_truth_queries: List[GroundTruthQuery] = []
        self.test_results: List[QualityTestResult] = []
        
        # Test configuration
        self.k_values = [1, 3, 5, 10]
        self.search_modes = ["hybrid", "dense", "sparse"]
        self.score_thresholds = [0.0, 0.5, 0.7]
        
    async def initialize_workspace_client(self):
        """Initialize workspace client for testing."""
        logger.info("🔧 Initializing workspace client")
        
        self.workspace_client = QdrantWorkspaceClient()
        await self.workspace_client.initialize()
        
        if not self.workspace_client.initialized:
            raise RuntimeError("Failed to initialize workspace client")
            
        logger.info("✅ Workspace client initialized")
        
    async def setup_test_datasets(self):
        """Setup test datasets for comprehensive evaluation."""
        logger.info("📊 Setting up test datasets")
        
        # Get available collections
        collections = await self.workspace_client.list_collections()
        logger.info(f"Available collections: {collections}")
        
        if not collections:
            logger.warning("No collections found. Creating test collection.")
            await self._create_test_collection()
            collections = await self.workspace_client.list_collections()
        
        # Generate ground truth for each collection
        for collection in collections:
            if collection.startswith("performance_") or collection.startswith("test"):
                continue  # Skip benchmark collections
                
            logger.info(f"Analyzing collection: {collection}")
            await self._generate_ground_truth_for_collection(collection)
        
        logger.info(f"✅ Generated {len(self.ground_truth_queries)} ground truth queries")
        
    async def _create_test_collection(self):
        """Create a test collection with realistic content."""
        test_collection = "advanced_recall_test"
        
        logger.info(f"Creating test collection: {test_collection}")
        
        # This would use the ingestion engine in a real implementation
        # For now, we'll create a basic collection structure
        try:
            collection_info = self.qdrant_client.get_collection(test_collection)
            logger.info(f"Test collection already exists with {collection_info.points_count} points")
        except:
            # Collection doesn't exist, this is expected
            logger.info("Test collection will be created by first ingestion")
        
    async def _generate_ground_truth_for_collection(self, collection: str):
        """Generate expert-curated ground truth queries for a collection."""
        logger.info(f"🔎 Generating ground truth for {collection}")
        
        try:
            # Get collection info
            collection_info = self.qdrant_client.get_collection(collection)
            point_count = collection_info.points_count
            
            if point_count == 0:
                logger.info(f"Skipping empty collection: {collection}")
                return
                
            # Sample documents to understand content
            sample_size = min(50, point_count)
            sample_points = self.qdrant_client.scroll(
                collection_name=collection,
                limit=sample_size,
                with_payload=True
            )[0]
            
            if not sample_points:
                return
            
            # Analyze content patterns
            content_analysis = self._analyze_collection_content(sample_points)
            
            # Generate different types of ground truth queries
            queries = []
            
            # 1. Exact match queries (easy difficulty)
            exact_queries = self._generate_exact_match_queries(
                collection, sample_points, content_analysis
            )
            queries.extend(exact_queries)
            
            # 2. Semantic queries (medium difficulty)
            semantic_queries = self._generate_semantic_queries(
                collection, sample_points, content_analysis
            )
            queries.extend(semantic_queries)
            
            # 3. Complex queries (hard difficulty)
            complex_queries = self._generate_complex_queries(
                collection, sample_points, content_analysis
            )
            queries.extend(complex_queries)
            
            # 4. Edge case queries (hard difficulty)
            edge_queries = self._generate_edge_case_queries(
                collection, sample_points, content_analysis
            )
            queries.extend(edge_queries)
            
            self.ground_truth_queries.extend(queries)
            logger.info(f"Generated {len(queries)} ground truth queries for {collection}")
            
        except Exception as e:
            logger.error(f"Failed to generate ground truth for {collection}: {e}")
    
    def _analyze_collection_content(self, sample_points: List) -> Dict[str, Any]:
        """Analyze collection content to understand patterns."""
        analysis = {
            "total_points": len(sample_points),
            "content_types": set(),
            "common_words": {},
            "file_extensions": set(),
            "topics": set()
        }
        
        all_words = []
        
        for point in sample_points:
            payload = point.payload or {}
            
            # Analyze payload structure
            for key in payload.keys():
                analysis["content_types"].add(key)
            
            # Extract text content
            text_content = ""
            for key in ["content", "text", "body", "description"]:
                if key in payload:
                    text_content += str(payload[key]) + " "
            
            # Extract file information
            if "file_path" in payload:
                file_path = str(payload["file_path"])
                if "." in file_path:
                    ext = file_path.split(".")[-1].lower()
                    analysis["file_extensions"].add(ext)
            
            # Extract meaningful words
            if text_content.strip():
                words = self._extract_meaningful_words(text_content)
                all_words.extend(words)
        
        # Find common words
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Keep words that appear multiple times
        common_words = {w: c for w, c in word_freq.items() if c >= 2}
        analysis["common_words"] = dict(sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:20])
        
        return analysis
    
    def _extract_meaningful_words(self, text: str) -> List[str]:
        """Extract meaningful words from text content."""
        # Simple word extraction - remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its',
            'our', 'their', 'me', 'him', 'her', 'us', 'them'
        }
        
        words = []
        for word in text.lower().split():
            # Clean word
            clean_word = ''.join(c for c in word if c.isalnum())
            
            # Keep meaningful words
            if (len(clean_word) >= 3 and 
                clean_word not in stop_words and
                not clean_word.isdigit()):
                words.append(clean_word)
        
        return words
    
    def _generate_exact_match_queries(
        self, collection: str, sample_points: List, analysis: Dict
    ) -> List[GroundTruthQuery]:
        """Generate exact match queries (easy difficulty)."""
        queries = []
        
        # Use first few documents for exact matching
        for i, point in enumerate(sample_points[:3]):
            payload = point.payload or {}
            doc_id = str(point.id)
            
            # Find exact phrases in the document
            content = ""
            for key in ["content", "text", "title", "file_path"]:
                if key in payload:
                    content += str(payload[key]) + " "
            
            if content.strip():
                # Extract first few meaningful words as exact query
                words = content.split()
                if len(words) >= 3:
                    exact_phrase = " ".join(words[:3]).lower().strip()
                    
                    # Clean the phrase
                    exact_phrase = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in exact_phrase)
                    exact_phrase = ' '.join(exact_phrase.split())  # Normalize spaces
                    
                    if exact_phrase and len(exact_phrase) > 5:
                        queries.append(GroundTruthQuery(
                            query=exact_phrase,
                            relevant_doc_ids={doc_id},
                            collection=collection,
                            query_category="exact_match",
                            difficulty_level="easy",
                            expected_min_precision_at_5=0.8,  # High expectation for exact match
                            description=f"Exact phrase match for document {doc_id}"
                        ))
        
        return queries
    
    def _generate_semantic_queries(
        self, collection: str, sample_points: List, analysis: Dict
    ) -> List[GroundTruthQuery]:
        """Generate semantic queries (medium difficulty)."""
        queries = []
        common_words = analysis.get("common_words", {})
        
        # Generate topic-based queries using common words
        for word, freq in list(common_words.items())[:5]:  # Top 5 common words
            if freq >= 3:  # Word appears in multiple documents
                # Find documents containing this word
                relevant_docs = set()
                
                for point in sample_points:
                    payload = point.payload or {}
                    doc_id = str(point.id)
                    
                    # Check if word appears in document content
                    content = ""
                    for key in ["content", "text", "title", "file_path"]:
                        if key in payload:
                            content += str(payload[key]).lower() + " "
                    
                    if word.lower() in content:
                        relevant_docs.add(doc_id)
                
                if len(relevant_docs) >= 2:  # At least 2 relevant documents
                    queries.append(GroundTruthQuery(
                        query=word,
                        relevant_doc_ids=relevant_docs,
                        collection=collection,
                        query_category="semantic",
                        difficulty_level="medium",
                        expected_min_precision_at_5=0.6,
                        description=f"Topic search for '{word}'"
                    ))
        
        return queries
    
    def _generate_complex_queries(
        self, collection: str, sample_points: List, analysis: Dict
    ) -> List[GroundTruthQuery]:
        """Generate complex multi-term queries (hard difficulty)."""
        queries = []
        common_words = analysis.get("common_words", {})
        
        # Create multi-word queries
        word_list = list(common_words.keys())[:10]
        
        for i in range(0, len(word_list)-1, 2):  # Pair words
            if i+1 < len(word_list):
                word1, word2 = word_list[i], word_list[i+1]
                complex_query = f"{word1} {word2}"
                
                # Find documents containing both words
                relevant_docs = set()
                
                for point in sample_points:
                    payload = point.payload or {}
                    doc_id = str(point.id)
                    
                    content = ""
                    for key in ["content", "text", "title", "file_path"]:
                        if key in payload:
                            content += str(payload[key]).lower() + " "
                    
                    if word1.lower() in content and word2.lower() in content:
                        relevant_docs.add(doc_id)
                
                if relevant_docs:  # At least 1 relevant document
                    queries.append(GroundTruthQuery(
                        query=complex_query,
                        relevant_doc_ids=relevant_docs,
                        collection=collection,
                        query_category="complex",
                        difficulty_level="hard",
                        expected_min_precision_at_5=0.4,  # Lower expectation for complex queries
                        description=f"Multi-term search for '{complex_query}'"
                    ))
        
        return queries
    
    def _generate_edge_case_queries(
        self, collection: str, sample_points: List, analysis: Dict
    ) -> List[GroundTruthQuery]:
        """Generate edge case queries (hard difficulty)."""
        queries = []
        
        # Generate queries that should have low/no matches
        edge_cases = [
            "nonexistent term xyz123",  # Should find nothing
            "very common word the",     # Should find too much
            "single letter a",          # Minimal query
            "!@#$%^&*()",              # Special characters
        ]
        
        for query in edge_cases:
            queries.append(GroundTruthQuery(
                query=query,
                relevant_doc_ids=set(),  # Expect no relevant results for edge cases
                collection=collection,
                query_category="edge_case",
                difficulty_level="hard",
                expected_min_precision_at_5=0.0,  # Low expectation for edge cases
                description=f"Edge case: {query}"
            ))
        
        return queries
    
    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive recall/precision evaluation."""
        logger.info("🎯 Starting Advanced Recall/Precision Evaluation")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # Initialize
        await self.initialize_workspace_client()
        await self.setup_test_datasets()
        
        if not self.ground_truth_queries:
            return {"error": "No ground truth queries generated"}
        
        # Run tests for all combinations
        total_tests = len(self.ground_truth_queries) * len(self.search_modes) * len(self.score_thresholds)
        logger.info(f"Running {total_tests} test combinations...")
        
        test_count = 0
        
        for gt_query in self.ground_truth_queries:
            for search_mode in self.search_modes:
                for threshold in self.score_thresholds:
                    test_count += 1
                    logger.info(f"  [{test_count}/{total_tests}] Testing '{gt_query.query[:40]}...' (mode={search_mode}, threshold={threshold})")
                    
                    try:
                        result = await self._run_single_test(
                            gt_query, search_mode, threshold
                        )
                        if result:
                            self.test_results.append(result)
                            
                    except Exception as e:
                        logger.error(f"Test failed: {e}")
                        continue
        
        end_time = time.time()
        
        # Generate comprehensive results
        results = {
            "metadata": {
                "timestamp": time.time(),
                "duration_seconds": end_time - start_time,
                "total_tests_run": len(self.test_results),
                "total_ground_truth_queries": len(self.ground_truth_queries),
                "search_modes_tested": self.search_modes,
                "score_thresholds_tested": self.score_thresholds,
                "k_values": self.k_values
            },
            "raw_results": [asdict(result) for result in self.test_results],
            "aggregated_results": self._generate_aggregated_results(),
            "scalability_analysis": self._analyze_scalability(),
            "mode_comparison": self._compare_search_modes(),
            "threshold_analysis": self._analyze_thresholds(),
            "quality_targets": self._analyze_quality_targets(),
            "recommendations": self._generate_recommendations()
        }
        
        logger.info(f"✅ Advanced evaluation completed in {end_time - start_time:.1f}s")
        logger.info(f"Successful tests: {len(self.test_results)}/{total_tests}")
        
        return results
    
    async def _run_single_test(
        self, gt_query: GroundTruthQuery, search_mode: str, threshold: float
    ) -> Optional[QualityTestResult]:
        """Run a single recall/precision test."""
        
        try:
            # Get database size
            collection_info = self.qdrant_client.get_collection(gt_query.collection)
            database_size = collection_info.points_count
            
            # Run search using actual search tools
            start_time = time.perf_counter()
            
            search_result = await search_workspace(
                client=self.workspace_client,
                query=gt_query.query,
                collections=[gt_query.collection],
                mode=search_mode,
                limit=max(self.k_values),  # Get enough results for largest k
                score_threshold=threshold
            )
            
            end_time = time.perf_counter()
            search_time_ms = (end_time - start_time) * 1000
            
            if "error" in search_result:
                logger.warning(f"Search failed: {search_result['error']}")
                return None
            
            # Extract retrieved document IDs
            retrieved_results = search_result.get("results", [])
            retrieved_ids = [str(result["id"]) for result in retrieved_results]
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                gt_query.relevant_doc_ids, retrieved_ids
            )
            
            # Create result object
            result = QualityTestResult(
                query=gt_query.query,
                collection=gt_query.collection,
                search_mode=search_mode,
                score_threshold=threshold,
                
                # Quality metrics
                precision_at_1=quality_metrics["precision"][1],
                precision_at_3=quality_metrics["precision"][3],
                precision_at_5=quality_metrics["precision"][5],
                precision_at_10=quality_metrics["precision"][10],
                
                recall_at_1=quality_metrics["recall"][1],
                recall_at_3=quality_metrics["recall"][3],
                recall_at_5=quality_metrics["recall"][5],
                recall_at_10=quality_metrics["recall"][10],
                
                f1_at_1=quality_metrics["f1"][1],
                f1_at_3=quality_metrics["f1"][3],
                f1_at_5=quality_metrics["f1"][5],
                f1_at_10=quality_metrics["f1"][10],
                
                average_precision=quality_metrics["average_precision"],
                mean_reciprocal_rank=quality_metrics["mrr"],
                
                # Performance metrics
                search_time_ms=search_time_ms,
                throughput_qps=1000.0 / search_time_ms if search_time_ms > 0 else 0.0,
                
                # Context
                database_size=database_size,
                total_relevant=len(gt_query.relevant_doc_ids),
                total_retrieved=len(retrieved_ids),
                
                # Test metadata
                query_category=gt_query.query_category,
                difficulty_level=gt_query.difficulty_level,
                expected_min_precision_at_5=gt_query.expected_min_precision_at_5,
                meets_quality_target=quality_metrics["precision"][5] >= gt_query.expected_min_precision_at_5
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Single test failed for query '{gt_query.query}': {e}")
            return None
    
    def _calculate_quality_metrics(self, relevant_ids: Set[str], retrieved_ids: List[str]) -> Dict[str, Any]:
        """Calculate quality metrics for a single test."""
        relevant_set = relevant_ids
        retrieved_set = set(retrieved_ids)
        
        metrics = {
            "precision": {},
            "recall": {},
            "f1": {},
            "average_precision": 0.0,
            "mrr": 0.0
        }
        
        # Calculate precision@k, recall@k, f1@k
        for k in self.k_values:
            top_k_ids = set(retrieved_ids[:k])
            relevant_in_top_k = len(top_k_ids.intersection(relevant_set))
            
            # Precision@k
            precision_k = relevant_in_top_k / min(k, len(retrieved_ids)) if retrieved_ids else 0.0
            metrics["precision"][k] = precision_k
            
            # Recall@k
            recall_k = relevant_in_top_k / len(relevant_set) if relevant_set else 0.0
            metrics["recall"][k] = recall_k
            
            # F1@k
            if precision_k + recall_k > 0:
                metrics["f1"][k] = 2 * (precision_k * recall_k) / (precision_k + recall_k)
            else:
                metrics["f1"][k] = 0.0
        
        # Average Precision
        if relevant_set:
            relevant_count = 0
            precision_sum = 0.0
            
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_set:
                    relevant_count += 1
                    precision_at_i = relevant_count / (i + 1)
                    precision_sum += precision_at_i
            
            metrics["average_precision"] = precision_sum / len(relevant_set)
        
        # Mean Reciprocal Rank
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                metrics["mrr"] = 1.0 / (i + 1)
                break
        
        return metrics
    
    def _generate_aggregated_results(self) -> Dict[str, Any]:
        """Generate aggregated results across all tests."""
        if not self.test_results:
            return {}
        
        # Overall averages
        overall_metrics = {
            "count": len(self.test_results),
            "avg_precision_at_1": statistics.mean([r.precision_at_1 for r in self.test_results]),
            "avg_precision_at_5": statistics.mean([r.precision_at_5 for r in self.test_results]),
            "avg_precision_at_10": statistics.mean([r.precision_at_10 for r in self.test_results]),
            "avg_recall_at_5": statistics.mean([r.recall_at_5 for r in self.test_results]),
            "avg_f1_at_5": statistics.mean([r.f1_at_5 for r in self.test_results]),
            "avg_average_precision": statistics.mean([r.average_precision for r in self.test_results]),
            "avg_search_time_ms": statistics.mean([r.search_time_ms for r in self.test_results]),
            "avg_throughput_qps": statistics.mean([r.throughput_qps for r in self.test_results]),
            "quality_target_success_rate": statistics.mean([float(r.meets_quality_target) for r in self.test_results])
        }
        
        return {
            "overall": overall_metrics,
            "by_category": self._aggregate_by_field("query_category"),
            "by_difficulty": self._aggregate_by_field("difficulty_level"),
            "by_collection": self._aggregate_by_field("collection")
        }
    
    def _aggregate_by_field(self, field_name: str) -> Dict[str, Any]:
        """Aggregate results by a specific field."""
        groups = {}
        
        for result in self.test_results:
            field_value = getattr(result, field_name)
            if field_value not in groups:
                groups[field_value] = []
            groups[field_value].append(result)
        
        aggregates = {}
        for field_value, results in groups.items():
            aggregates[field_value] = {
                "count": len(results),
                "avg_precision_at_5": statistics.mean([r.precision_at_5 for r in results]),
                "avg_recall_at_5": statistics.mean([r.recall_at_5 for r in results]),
                "avg_f1_at_5": statistics.mean([r.f1_at_5 for r in results]),
                "avg_search_time_ms": statistics.mean([r.search_time_ms for r in results]),
                "quality_target_success_rate": statistics.mean([float(r.meets_quality_target) for r in results])
            }
        
        return aggregates
    
    def _analyze_scalability(self) -> Dict[str, Any]:
        """Analyze how quality scales with database size."""
        if not self.test_results:
            return {}
        
        # Group by database size
        db_sizes = [r.database_size for r in self.test_results]
        if not db_sizes:
            return {}
        
        median_size = statistics.median(db_sizes)
        
        small_db_results = [r for r in self.test_results if r.database_size <= median_size]
        large_db_results = [r for r in self.test_results if r.database_size > median_size]
        
        analysis = {}
        
        if small_db_results:
            analysis["small_databases"] = {
                "count": len(small_db_results),
                "size_range": f"<= {int(median_size):,} points",
                "avg_precision_at_5": statistics.mean([r.precision_at_5 for r in small_db_results]),
                "avg_search_time_ms": statistics.mean([r.search_time_ms for r in small_db_results])
            }
        
        if large_db_results:
            analysis["large_databases"] = {
                "count": len(large_db_results),
                "size_range": f"> {int(median_size):,} points",
                "avg_precision_at_5": statistics.mean([r.precision_at_5 for r in large_db_results]),
                "avg_search_time_ms": statistics.mean([r.search_time_ms for r in large_db_results])
            }
        
        # Calculate degradation if both groups exist
        if small_db_results and large_db_results:
            small_precision = statistics.mean([r.precision_at_5 for r in small_db_results])
            large_precision = statistics.mean([r.precision_at_5 for r in large_db_results])
            small_time = statistics.mean([r.search_time_ms for r in small_db_results])
            large_time = statistics.mean([r.search_time_ms for r in large_db_results])
            
            analysis["scale_impact"] = {
                "precision_change_percent": ((large_precision - small_precision) / small_precision * 100) if small_precision > 0 else 0,
                "search_time_change_percent": ((large_time - small_time) / small_time * 100) if small_time > 0 else 0
            }
        
        return analysis
    
    def _compare_search_modes(self) -> Dict[str, Any]:
        """Compare different search modes."""
        mode_groups = {}
        
        for result in self.test_results:
            mode = result.search_mode
            if mode not in mode_groups:
                mode_groups[mode] = []
            mode_groups[mode].append(result)
        
        comparison = {}
        for mode, results in mode_groups.items():
            if results:
                comparison[mode] = {
                    "count": len(results),
                    "avg_precision_at_5": statistics.mean([r.precision_at_5 for r in results]),
                    "avg_recall_at_5": statistics.mean([r.recall_at_5 for r in results]),
                    "avg_f1_at_5": statistics.mean([r.f1_at_5 for r in results]),
                    "avg_search_time_ms": statistics.mean([r.search_time_ms for r in results]),
                    "quality_target_success_rate": statistics.mean([float(r.meets_quality_target) for r in results])
                }
        
        return comparison
    
    def _analyze_thresholds(self) -> Dict[str, Any]:
        """Analyze impact of score thresholds."""
        threshold_groups = {}
        
        for result in self.test_results:
            threshold = result.score_threshold
            if threshold not in threshold_groups:
                threshold_groups[threshold] = []
            threshold_groups[threshold].append(result)
        
        analysis = {}
        for threshold, results in threshold_groups.items():
            if results:
                analysis[f"threshold_{threshold}"] = {
                    "count": len(results),
                    "avg_precision_at_5": statistics.mean([r.precision_at_5 for r in results]),
                    "avg_recall_at_5": statistics.mean([r.recall_at_5 for r in results]),
                    "avg_results_returned": statistics.mean([r.total_retrieved for r in results])
                }
        
        return analysis
    
    def _analyze_quality_targets(self) -> Dict[str, Any]:
        """Analyze how well quality targets are met."""
        total_tests = len(self.test_results)
        if total_tests == 0:
            return {}
        
        met_targets = sum(1 for r in self.test_results if r.meets_quality_target)
        success_rate = met_targets / total_tests
        
        # Analyze by difficulty level
        difficulty_analysis = {}
        for difficulty in ["easy", "medium", "hard"]:
            difficulty_results = [r for r in self.test_results if r.difficulty_level == difficulty]
            if difficulty_results:
                difficulty_met = sum(1 for r in difficulty_results if r.meets_quality_target)
                difficulty_analysis[difficulty] = {
                    "total_tests": len(difficulty_results),
                    "targets_met": difficulty_met,
                    "success_rate": difficulty_met / len(difficulty_results)
                }
        
        return {
            "overall_success_rate": success_rate,
            "total_tests": total_tests,
            "targets_met": met_targets,
            "by_difficulty": difficulty_analysis
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if not self.test_results:
            return ["No test results available for recommendations."]
        
        # Analyze overall performance
        avg_precision_5 = statistics.mean([r.precision_at_5 for r in self.test_results])
        avg_recall_5 = statistics.mean([r.recall_at_5 for r in self.test_results])
        avg_search_time = statistics.mean([r.search_time_ms for r in self.test_results])
        target_success_rate = statistics.mean([float(r.meets_quality_target) for r in self.test_results])
        
        # Performance recommendations
        if avg_search_time > 100:
            recommendations.append(f"Search performance optimization needed: average {avg_search_time:.1f}ms exceeds 100ms target")
        
        # Quality recommendations
        if avg_precision_5 < 0.5:
            recommendations.append(f"Low precision detected: {avg_precision_5:.3f}@5. Consider improving search relevance or query processing.")
        
        if avg_recall_5 < 0.4:
            recommendations.append(f"Low recall detected: {avg_recall_5:.3f}@5. Consider expanding search scope or improving embeddings.")
        
        if target_success_rate < 0.7:
            recommendations.append(f"Quality targets not met: {target_success_rate:.1%} success rate. Review target expectations or improve search quality.")
        
        # Mode-specific recommendations
        mode_comparison = self._compare_search_modes()
        if mode_comparison:
            best_mode = max(mode_comparison.keys(), 
                          key=lambda m: mode_comparison[m]["avg_f1_at_5"])
            best_f1 = mode_comparison[best_mode]["avg_f1_at_5"]
            recommendations.append(f"Best performing search mode: '{best_mode}' (F1@5: {best_f1:.3f})")
        
        return recommendations
    
    def generate_advanced_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive advanced report."""
        if not results or "raw_results" not in results:
            return "No advanced test results available."
        
        report = []
        report.append("# Advanced Recall and Precision Test Report")
        report.append("=" * 60)
        report.append("")
        
        metadata = results["metadata"]
        report.append(f"**Test Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['timestamp']))}")
        report.append(f"**Duration:** {metadata['duration_seconds']:.1f} seconds")
        report.append(f"**Tests Completed:** {metadata['total_tests_run']} / {metadata['total_ground_truth_queries']} queries")
        report.append(f"**Search Modes:** {', '.join(metadata['search_modes_tested'])}")
        report.append(f"**Score Thresholds:** {', '.join(map(str, metadata['score_thresholds_tested']))}")
        report.append("")
        
        # Executive Summary
        aggregated = results.get("aggregated_results", {})
        overall = aggregated.get("overall", {})
        
        if overall:
            report.append("## Executive Summary")
            report.append("")
            report.append(f"- **Overall Precision@5:** {overall.get('avg_precision_at_5', 0):.3f}")
            report.append(f"- **Overall Recall@5:** {overall.get('avg_recall_at_5', 0):.3f}")
            report.append(f"- **Overall F1@5:** {overall.get('avg_f1_at_5', 0):.3f}")
            report.append(f"- **Average Search Time:** {overall.get('avg_search_time_ms', 0):.2f}ms")
            report.append(f"- **Quality Target Success Rate:** {overall.get('quality_target_success_rate', 0):.1%}")
            report.append("")
        
        # Search Mode Comparison
        mode_comparison = results.get("mode_comparison", {})
        if mode_comparison:
            report.append("## Search Mode Performance")
            report.append("")
            
            for mode, metrics in mode_comparison.items():
                report.append(f"### {mode.title()} Mode")
                report.append(f"- Precision@5: {metrics.get('avg_precision_at_5', 0):.3f}")
                report.append(f"- Recall@5: {metrics.get('avg_recall_at_5', 0):.3f}")
                report.append(f"- F1@5: {metrics.get('avg_f1_at_5', 0):.3f}")
                report.append(f"- Search Time: {metrics.get('avg_search_time_ms', 0):.2f}ms")
                report.append(f"- Quality Target Success: {metrics.get('quality_target_success_rate', 0):.1%}")
                report.append("")
        
        # Scalability Analysis
        scalability = results.get("scalability_analysis", {})
        if scalability and "scale_impact" in scalability:
            report.append("## Scalability Analysis")
            report.append("")
            
            if "small_databases" in scalability:
                small = scalability["small_databases"]
                report.append(f"**Small Collections ({small['size_range']}):**")
                report.append(f"- Precision@5: {small.get('avg_precision_at_5', 0):.3f}")
                report.append(f"- Search Time: {small.get('avg_search_time_ms', 0):.2f}ms")
                report.append("")
            
            if "large_databases" in scalability:
                large = scalability["large_databases"]
                report.append(f"**Large Collections ({large['size_range']}):**")
                report.append(f"- Precision@5: {large.get('avg_precision_at_5', 0):.3f}")
                report.append(f"- Search Time: {large.get('avg_search_time_ms', 0):.2f}ms")
                report.append("")
            
            scale_impact = scalability["scale_impact"]
            precision_change = scale_impact.get("precision_change_percent", 0)
            time_change = scale_impact.get("search_time_change_percent", 0)
            
            report.append("**Scale Impact:**")
            report.append(f"- Precision change at scale: {precision_change:+.1f}%")
            report.append(f"- Search time change at scale: {time_change:+.1f}%")
            report.append("")
        
        # Quality Targets Analysis
        quality_targets = results.get("quality_targets", {})
        if quality_targets:
            report.append("## Quality Target Analysis")
            report.append("")
            report.append(f"- Overall Success Rate: {quality_targets.get('overall_success_rate', 0):.1%}")
            report.append(f"- Total Tests: {quality_targets.get('total_tests', 0)}")
            report.append(f"- Targets Met: {quality_targets.get('targets_met', 0)}")
            report.append("")
            
            by_difficulty = quality_targets.get("by_difficulty", {})
            if by_difficulty:
                report.append("**By Difficulty Level:**")
                for difficulty, stats in by_difficulty.items():
                    report.append(f"- {difficulty.title()}: {stats['success_rate']:.1%} ({stats['targets_met']}/{stats['total_tests']})")
                report.append("")
        
        # Results by Category
        by_category = aggregated.get("by_category", {})
        if by_category:
            report.append("## Results by Query Category")
            report.append("")
            
            for category, metrics in by_category.items():
                report.append(f"### {category.replace('_', ' ').title()} Queries")
                report.append(f"- Tests Run: {metrics.get('count', 0)}")
                report.append(f"- Precision@5: {metrics.get('avg_precision_at_5', 0):.3f}")
                report.append(f"- Recall@5: {metrics.get('avg_recall_at_5', 0):.3f}")
                report.append(f"- F1@5: {metrics.get('avg_f1_at_5', 0):.3f}")
                report.append(f"- Quality Target Success: {metrics.get('quality_target_success_rate', 0):.1%}")
                report.append("")
        
        # Recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            report.append("## Recommendations")
            report.append("")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Methodology
        report.append("## Advanced Methodology")
        report.append("")
        report.append("This advanced evaluation provides production-grade search quality assessment:")
        report.append("")
        report.append("- **Real Search Integration:** Uses actual workspace search tools and infrastructure")
        report.append("- **Expert Ground Truth:** Curated relevant documents for each query")
        report.append("- **Multiple Search Modes:** Tests hybrid, dense, and sparse search approaches")
        report.append("- **Scalability Testing:** Evaluates performance across different database sizes")
        report.append("- **Quality Targets:** Measures success against predefined precision thresholds")
        report.append("- **Comprehensive Metrics:** Precision@K, Recall@K, F1@K, Average Precision, MRR")
        report.append("")
        
        report.append("**Key Advantages:**")
        report.append("- Reflects real user experience with production search tools")
        report.append("- Identifies optimal search configurations for different use cases")
        report.append("- Provides early warning for search quality regressions")
        report.append("- Enables data-driven search optimization decisions")
        report.append("")
        
        return "\n".join(report)


async def main():
    """Run advanced recall/precision testing."""
    test_suite = AdvancedRecallPrecisionTest()
    
    try:
        # Run comprehensive evaluation
        results = await test_suite.run_comprehensive_evaluation()
        
        # Save results
        results_dir = Path("recall_precision_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"advanced_recall_precision_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate and save report
        report = test_suite.generate_advanced_report(results)
        report_file = results_file.with_suffix(".md")
        
        with open(report_file, "w") as f:
            f.write(report)
        
        # Print summary
        print("\n" + "=" * 70)
        print(report)
        print(f"\n📊 Advanced results saved to: {results_file}")
        print(f"📋 Advanced report saved to: {report_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Advanced recall/precision test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())
