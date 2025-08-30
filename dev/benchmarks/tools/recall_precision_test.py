#!/usr/bin/env python3
"""
Simple Recall and Precision Benchmark Tool for workspace-qdrant-mcp.

This tool provides a streamlined approach to measuring search quality metrics
alongside existing performance benchmarks. It focuses on practical evaluation
of search accuracy with real-world datasets.

USAGE:
    python recall_precision_test.py
    
OUTPUT:
    - JSON results with precision@k, recall@k, F1@k metrics
    - Human-readable report with recommendations
    - Integration with existing performance baseline data

METHODOLOGY:
    1. Use existing collections as test data
    2. Generate focused test queries with known relevant documents
    3. Measure precision, recall, and F1 scores at different k values
    4. Compare performance vs quality trade-offs
    5. Provide actionable recommendations for search configuration
"""

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import numpy as np
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


@dataclass
class SimpleQualityMetrics:
    """Simplified search quality metrics for practical use."""
    
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    
    f1_at_1: float
    f1_at_5: float
    f1_at_10: float
    
    average_precision: float
    mean_reciprocal_rank: float
    
    search_time_ms: float
    throughput_qps: float
    
    total_relevant: int
    total_retrieved: int
    database_size: int


@dataclass 
class TestCase:
    """Simple test case with query and expected relevant documents."""
    
    query: str
    collection: str
    expected_docs: Set[str]  # Expected relevant document IDs
    query_type: str  # "exact", "semantic", "fuzzy"
    description: str


class SimplifiedRecallPrecisionTest:
    """Simplified recall and precision testing focused on practical insights."""
    
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.test_cases: List[TestCase] = []
        self.collections = []
        self.results = {}
    
    def setup_test_environment(self):
        """Setup test environment using existing collections."""
        logger.info("🚀 Setting up Simplified Recall/Precision Test")
        
        # Discover existing collections
        collections_response = self.client.get_collections()
        self.collections = [c.name for c in collections_response.collections if c.name != "performance_baseline"]
        
        logger.info(f"📊 Found {len(self.collections)} collections: {self.collections}")
        
        # Generate test cases based on existing data
        self._generate_realistic_test_cases()
        
        logger.info(f"✅ Generated {len(self.test_cases)} test cases")
    
    def _generate_realistic_test_cases(self):
        """Generate realistic test cases using existing collection data."""
        logger.info("🔍 Generating realistic test cases from existing data")
        
        # Test cases for existing collections
        test_cases = []
        
        for collection in self.collections:
            try:
                # Get collection info
                collection_info = self.client.get_collection(collection)
                point_count = collection_info.points_count
                
                if point_count == 0:
                    logger.info(f"Skipping empty collection: {collection}")
                    continue
                
                logger.info(f"Analyzing collection '{collection}' with {point_count} points")
                
                # Sample some documents to understand content
                sample_points = self.client.scroll(
                    collection_name=collection,
                    limit=min(20, point_count),  # Sample up to 20 documents
                    with_payload=True
                )[0]
                
                if not sample_points:
                    continue
                
                # Generate test cases based on document content
                collection_test_cases = self._generate_collection_test_cases(
                    collection, sample_points
                )
                test_cases.extend(collection_test_cases)
                
            except Exception as e:
                logger.warning(f"Failed to analyze collection {collection}: {e}")
                continue
        
        self.test_cases = test_cases
    
    def _generate_collection_test_cases(
        self, collection: str, sample_points: List
    ) -> List[TestCase]:
        """Generate test cases for a specific collection based on its content."""
        test_cases = []
        
        # Extract content patterns from samples
        content_samples = []
        doc_ids = []
        
        for point in sample_points:
            doc_id = str(point.id)
            payload = point.payload or {}
            
            # Extract searchable content
            content = ""
            if "content" in payload:
                content = str(payload["content"])
            elif "text" in payload:
                content = str(payload["text"])
            elif "file_path" in payload:
                content = str(payload["file_path"])
            
            if content.strip():
                content_samples.append(content)
                doc_ids.append(doc_id)
        
        if not content_samples:
            return test_cases
        
        # Generate different types of test queries
        
        # 1. Exact match test (easy)
        if doc_ids:
            # Pick a random document for exact matching
            target_doc = doc_ids[0]
            target_content = content_samples[0]
            
            # Extract a meaningful phrase for exact search
            words = target_content.split()
            if len(words) >= 3:
                exact_phrase = " ".join(words[:3])  # First 3 words
                test_cases.append(TestCase(
                    query=exact_phrase,
                    collection=collection,
                    expected_docs={target_doc},
                    query_type="exact",
                    description=f"Exact phrase search in {collection}"
                ))
        
        # 2. Semantic/topic-based test (medium)
        # Try to identify common themes in the collection
        all_words = []
        for content in content_samples:
            # Extract meaningful words (skip common words)
            words = [w.lower().strip('.,!?"()[]{}') for w in content.split() 
                    if len(w) > 3 and w.lower() not in {'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'after', 'our', 'two', 'new', 'may', 'way', 'who', 'make', 'most', 'now', 'old', 'see', 'him', 'had', 'has', 'his', 'her', 'how', 'man', 'day', 'get', 'use', 'use', 'work', 'life', 'only', 'its', 'also', 'back', 'other', 'many', 'well', 'such'}]
            all_words.extend(words)
        
        if all_words:
            # Find most common meaningful words
            word_freq = {}
            for word in all_words:
                if word.isalpha():  # Only alphabetic words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top words that appear multiple times
            common_words = [(w, c) for w, c in word_freq.items() if c >= 2]
            common_words.sort(key=lambda x: x[1], reverse=True)
            
            if common_words:
                # Create semantic query using top words
                top_word = common_words[0][0]
                
                # Find documents that should match this query
                expected_matches = set()
                for i, content in enumerate(content_samples):
                    if top_word.lower() in content.lower():
                        expected_matches.add(doc_ids[i])
                
                if expected_matches:
                    test_cases.append(TestCase(
                        query=top_word,
                        collection=collection,
                        expected_docs=expected_matches,
                        query_type="semantic",
                        description=f"Topic-based search for '{top_word}' in {collection}"
                    ))
        
        # 3. Fuzzy/partial match test (hard)
        if doc_ids and content_samples:
            # Pick a document and create a partial/fuzzy query
            target_content = content_samples[0]
            target_doc = doc_ids[0]
            
            words = target_content.split()
            if len(words) >= 5:
                # Create query with some words from the document
                fuzzy_query = f"{words[0]} {words[2]}".lower()  # Non-consecutive words
                
                test_cases.append(TestCase(
                    query=fuzzy_query,
                    collection=collection,
                    expected_docs={target_doc},
                    query_type="fuzzy",
                    description=f"Partial match search in {collection}"
                ))
        
        logger.info(f"Generated {len(test_cases)} test cases for collection '{collection}'")
        return test_cases
    
    def calculate_quality_metrics(
        self, 
        expected_docs: Set[str], 
        retrieved_results: List, 
        search_time_ms: float,
        database_size: int
    ) -> SimpleQualityMetrics:
        """Calculate simplified quality metrics."""
        
        # Extract retrieved document IDs
        retrieved_ids = []
        for result in retrieved_results:
            if hasattr(result, 'id'):
                retrieved_ids.append(str(result.id))
            elif isinstance(result, dict) and 'id' in result:
                retrieved_ids.append(str(result['id']))
        
        retrieved_set = set(retrieved_ids)
        
        # Basic counts
        total_relevant = len(expected_docs)
        total_retrieved = len(retrieved_ids)
        
        if total_retrieved == 0:
            # No results - return zero metrics
            return SimpleQualityMetrics(
                precision_at_1=0.0, precision_at_5=0.0, precision_at_10=0.0,
                recall_at_1=0.0, recall_at_5=0.0, recall_at_10=0.0,
                f1_at_1=0.0, f1_at_5=0.0, f1_at_10=0.0,
                average_precision=0.0, mean_reciprocal_rank=0.0,
                search_time_ms=search_time_ms,
                throughput_qps=1000.0 / search_time_ms if search_time_ms > 0 else 0.0,
                total_relevant=total_relevant, total_retrieved=total_retrieved,
                database_size=database_size
            )
        
        # Calculate metrics at different k values
        k_values = [1, 5, 10]
        precision_at_k = {}
        recall_at_k = {}
        f1_at_k = {}
        
        for k in k_values:
            top_k_ids = set(retrieved_ids[:k])
            relevant_in_top_k = len(top_k_ids.intersection(expected_docs))
            
            # Precision@k
            precision_k = relevant_in_top_k / min(k, total_retrieved)
            precision_at_k[k] = precision_k
            
            # Recall@k
            recall_k = relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0
            recall_at_k[k] = recall_k
            
            # F1@k
            if precision_k + recall_k > 0:
                f1_at_k[k] = 2 * (precision_k * recall_k) / (precision_k + recall_k)
            else:
                f1_at_k[k] = 0.0
        
        # Average Precision
        average_precision = self._calculate_average_precision(expected_docs, retrieved_ids)
        
        # Mean Reciprocal Rank
        mean_reciprocal_rank = self._calculate_reciprocal_rank(expected_docs, retrieved_ids)
        
        return SimpleQualityMetrics(
            precision_at_1=precision_at_k[1],
            precision_at_5=precision_at_k[5], 
            precision_at_10=precision_at_k[10],
            recall_at_1=recall_at_k[1],
            recall_at_5=recall_at_k[5],
            recall_at_10=recall_at_k[10],
            f1_at_1=f1_at_k[1],
            f1_at_5=f1_at_k[5],
            f1_at_10=f1_at_k[10],
            average_precision=average_precision,
            mean_reciprocal_rank=mean_reciprocal_rank,
            search_time_ms=search_time_ms,
            throughput_qps=1000.0 / search_time_ms if search_time_ms > 0 else 0.0,
            total_relevant=total_relevant,
            total_retrieved=total_retrieved,
            database_size=database_size
        )
    
    def _calculate_average_precision(self, expected_ids: Set[str], retrieved_ids: List[str]) -> float:
        """Calculate Average Precision."""
        if not expected_ids:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(expected_ids)
    
    def _calculate_reciprocal_rank(self, expected_ids: Set[str], retrieved_ids: List[str]) -> float:
        """Calculate Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    def run_quality_benchmark(self) -> Dict[str, Any]:
        """Run simplified recall/precision benchmark."""
        logger.info("🎯 Starting Simplified Recall/Precision Benchmark")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Setup test environment
        self.setup_test_environment()
        
        if not self.test_cases:
            return {"error": "No test cases generated"}
        
        results = {
            "metadata": {
                "timestamp": time.time(),
                "total_test_cases": len(self.test_cases),
                "collections_tested": self.collections,
                "qdrant_host": "localhost:6333"
            },
            "test_results": [],
            "summary_by_collection": {},
            "summary_by_query_type": {},
            "overall_summary": {}
        }
        
        # Run each test case
        for i, test_case in enumerate(self.test_cases):
            logger.info(f"  [{i+1}/{len(self.test_cases)}] Testing: '{test_case.query}' in {test_case.collection}")
            
            try:
                # Get database size
                collection_info = self.client.get_collection(test_case.collection)
                database_size = collection_info.points_count
                
                # Create a simple query vector (in a real implementation, this would use embeddings)
                # For demonstration, we'll use scroll with payload filtering instead
                start_search_time = time.perf_counter()
                
                # Simple text search using scroll and payload filtering
                search_results = self._perform_simple_search(
                    collection_name=test_case.collection,
                    query=test_case.query,
                    limit=10
                )
                
                end_search_time = time.perf_counter()
                search_time_ms = (end_search_time - start_search_time) * 1000
                
                # Calculate quality metrics
                metrics = self.calculate_quality_metrics(
                    expected_docs=test_case.expected_docs,
                    retrieved_results=search_results,
                    search_time_ms=search_time_ms,
                    database_size=database_size
                )
                
                # Store result
                test_result = {
                    "test_case": {
                        "query": test_case.query,
                        "collection": test_case.collection,
                        "query_type": test_case.query_type,
                        "description": test_case.description,
                        "expected_relevant_count": len(test_case.expected_docs)
                    },
                    "metrics": {
                        "precision_at_1": metrics.precision_at_1,
                        "precision_at_5": metrics.precision_at_5,
                        "precision_at_10": metrics.precision_at_10,
                        "recall_at_1": metrics.recall_at_1,
                        "recall_at_5": metrics.recall_at_5,
                        "recall_at_10": metrics.recall_at_10,
                        "f1_at_1": metrics.f1_at_1,
                        "f1_at_5": metrics.f1_at_5,
                        "f1_at_10": metrics.f1_at_10,
                        "average_precision": metrics.average_precision,
                        "mean_reciprocal_rank": metrics.mean_reciprocal_rank,
                        "search_time_ms": metrics.search_time_ms,
                        "throughput_qps": metrics.throughput_qps,
                        "database_size": metrics.database_size,
                        "total_relevant": metrics.total_relevant,
                        "total_retrieved": metrics.total_retrieved
                    }
                }
                
                results["test_results"].append(test_result)
                
                logger.info(f"    Precision@5: {metrics.precision_at_5:.3f}, Recall@5: {metrics.recall_at_5:.3f}, Time: {metrics.search_time_ms:.2f}ms")
                
            except Exception as e:
                logger.error(f"Test case failed: {e}")
                continue
        
        # Generate summaries
        if results["test_results"]:
            results["summary_by_collection"] = self._summarize_by_collection(results["test_results"])
            results["summary_by_query_type"] = self._summarize_by_query_type(results["test_results"])
            results["overall_summary"] = self._generate_overall_summary(results["test_results"])
        
        end_time = time.time()
        results["metadata"]["duration_seconds"] = end_time - start_time
        
        logger.info(f"✅ Recall/Precision benchmark completed in {end_time - start_time:.1f}s")
        
        return results
    
    def _perform_simple_search(self, collection_name: str, query: str, limit: int = 10) -> List:
        """Perform simple search using scroll and text matching."""
        # This is a simplified search implementation
        # In a real system, this would use proper vector search
        
        try:
            # Get all points and filter by text content
            all_points = self.client.scroll(
                collection_name=collection_name,
                limit=100,  # Get more points to search through
                with_payload=True
            )[0]
            
            # Simple text matching
            query_lower = query.lower()
            matching_points = []
            
            for point in all_points:
                payload = point.payload or {}
                
                # Check various payload fields for matches
                content_text = ""
                for key in ["content", "text", "file_path", "title"]:
                    if key in payload:
                        content_text += str(payload[key]).lower() + " "
                
                # Simple scoring based on word matches
                score = 0.0
                query_words = query_lower.split()
                for word in query_words:
                    if word in content_text:
                        score += 1.0
                
                if score > 0:
                    # Create a simple result object
                    result = type('Result', (), {
                        'id': point.id,
                        'score': score / len(query_words),  # Normalize score
                        'payload': payload
                    })
                    matching_points.append(result)
            
            # Sort by score and return top results
            matching_points.sort(key=lambda x: x.score, reverse=True)
            return matching_points[:limit]
            
        except Exception as e:
            logger.warning(f"Simple search failed: {e}")
            return []
    
    def _summarize_by_collection(self, test_results: List[Dict]) -> Dict[str, Any]:
        """Summarize results by collection."""
        collection_groups = {}
        
        for result in test_results:
            collection = result["test_case"]["collection"]
            if collection not in collection_groups:
                collection_groups[collection] = []
            collection_groups[collection].append(result["metrics"])
        
        summaries = {}
        for collection, metrics_list in collection_groups.items():
            summaries[collection] = self._calculate_average_metrics(metrics_list)
        
        return summaries
    
    def _summarize_by_query_type(self, test_results: List[Dict]) -> Dict[str, Any]:
        """Summarize results by query type."""
        type_groups = {}
        
        for result in test_results:
            query_type = result["test_case"]["query_type"]
            if query_type not in type_groups:
                type_groups[query_type] = []
            type_groups[query_type].append(result["metrics"])
        
        summaries = {}
        for query_type, metrics_list in type_groups.items():
            summaries[query_type] = self._calculate_average_metrics(metrics_list)
        
        return summaries
    
    def _generate_overall_summary(self, test_results: List[Dict]) -> Dict[str, Any]:
        """Generate overall summary statistics."""
        all_metrics = [result["metrics"] for result in test_results]
        overall_avg = self._calculate_average_metrics(all_metrics)
        
        # Find best and worst performers
        best_precision = max(test_results, key=lambda r: r["metrics"]["precision_at_5"])
        worst_precision = min(test_results, key=lambda r: r["metrics"]["precision_at_5"])
        
        fastest_search = min(test_results, key=lambda r: r["metrics"]["search_time_ms"])
        slowest_search = max(test_results, key=lambda r: r["metrics"]["search_time_ms"])
        
        return {
            "total_tests": len(test_results),
            "average_metrics": overall_avg,
            "best_precision_test": {
                "query": best_precision["test_case"]["query"],
                "collection": best_precision["test_case"]["collection"],
                "precision_at_5": best_precision["metrics"]["precision_at_5"]
            },
            "worst_precision_test": {
                "query": worst_precision["test_case"]["query"],
                "collection": worst_precision["test_case"]["collection"],
                "precision_at_5": worst_precision["metrics"]["precision_at_5"]
            },
            "fastest_search": {
                "query": fastest_search["test_case"]["query"],
                "search_time_ms": fastest_search["metrics"]["search_time_ms"]
            },
            "slowest_search": {
                "query": slowest_search["test_case"]["query"],
                "search_time_ms": slowest_search["metrics"]["search_time_ms"]
            },
            "recommendations": self._generate_recommendations(test_results)
        }
    
    def _calculate_average_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Calculate average metrics from a list of metric dictionaries."""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        metric_keys = metrics_list[0].keys()
        
        for key in metric_keys:
            values = [m[key] for m in metrics_list if isinstance(m[key], (int, float))]
            if values:
                avg_metrics[key] = statistics.mean(values)
        
        return avg_metrics
    
    def _generate_recommendations(self, test_results: List[Dict]) -> List[str]:
        """Generate practical recommendations based on test results."""
        recommendations = []
        
        if not test_results:
            return ["No test results available for recommendations"]
        
        # Analyze average performance
        all_metrics = [r["metrics"] for r in test_results]
        avg_metrics = self._calculate_average_metrics(all_metrics)
        
        avg_precision_5 = avg_metrics.get("precision_at_5", 0)
        avg_recall_5 = avg_metrics.get("recall_at_5", 0)
        avg_f1_5 = avg_metrics.get("f1_at_5", 0)
        avg_search_time = avg_metrics.get("search_time_ms", 0)
        
        # Performance recommendations
        if avg_search_time > 50:  # Slower than 50ms
            recommendations.append(f"Search performance optimization needed: avg {avg_search_time:.1f}ms > 50ms target")
        elif avg_search_time < 10:
            recommendations.append(f"Excellent search performance: avg {avg_search_time:.1f}ms")
        
        # Quality recommendations
        if avg_precision_5 < 0.5:
            recommendations.append(f"Low precision detected (avg {avg_precision_5:.3f}@5). Consider improving search algorithm or query processing.")
        elif avg_precision_5 > 0.8:
            recommendations.append(f"Excellent search precision (avg {avg_precision_5:.3f}@5)")
        
        if avg_recall_5 < 0.3:
            recommendations.append(f"Low recall detected (avg {avg_recall_5:.3f}@5). Consider expanding search scope or improving embeddings.")
        elif avg_recall_5 > 0.7:
            recommendations.append(f"Good search recall (avg {avg_recall_5:.3f}@5)")
        
        # Query type analysis
        type_summary = {}
        for result in test_results:
            query_type = result["test_case"]["query_type"]
            if query_type not in type_summary:
                type_summary[query_type] = []
            type_summary[query_type].append(result["metrics"]["precision_at_5"])
        
        for query_type, precision_values in type_summary.items():
            avg_precision = statistics.mean(precision_values)
            if avg_precision < 0.5:
                recommendations.append(f"'{query_type}' queries underperforming (avg precision {avg_precision:.3f}). Focus optimization here.")
            elif avg_precision > 0.8:
                recommendations.append(f"'{query_type}' queries performing well (avg precision {avg_precision:.3f})")
        
        return recommendations
    
    def generate_simple_report(self, results: Dict[str, Any]) -> str:
        """Generate a simple, actionable report."""
        if not results or "test_results" not in results:
            return "No test results available."
        
        report = []
        report.append("# Simple Recall/Precision Test Report")
        report.append("=" * 50)
        report.append("")
        
        metadata = results["metadata"]
        report.append(f"**Test Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['timestamp']))}")
        report.append(f"**Duration:** {metadata['duration_seconds']:.1f} seconds")
        report.append(f"**Collections:** {', '.join(metadata['collections_tested'])}")
        report.append(f"**Total Tests:** {metadata['total_test_cases']}")
        report.append("")
        
        # Overall Summary
        overall = results.get("overall_summary", {})
        avg_metrics = overall.get("average_metrics", {})
        
        if avg_metrics:
            report.append("## Overall Performance")
            report.append("")
            report.append(f"- **Precision@5:** {avg_metrics.get('precision_at_5', 0):.3f} ({avg_metrics.get('precision_at_5', 0)*100:.1f}% of retrieved docs were relevant)")
            report.append(f"- **Recall@5:** {avg_metrics.get('recall_at_5', 0):.3f} ({avg_metrics.get('recall_at_5', 0)*100:.1f}% of relevant docs were found)")
            report.append(f"- **F1@5:** {avg_metrics.get('f1_at_5', 0):.3f} (balance of precision and recall)")
            report.append(f"- **Average Precision:** {avg_metrics.get('average_precision', 0):.3f}")
            report.append(f"- **Search Time:** {avg_metrics.get('search_time_ms', 0):.2f}ms")
            report.append(f"- **Throughput:** {avg_metrics.get('throughput_qps', 0):.1f} queries/sec")
            report.append("")
        
        # Performance by Collection
        collection_summary = results.get("summary_by_collection", {})
        if collection_summary:
            report.append("## Results by Collection")
            report.append("")
            
            for collection, metrics in collection_summary.items():
                report.append(f"### {collection}")
                report.append(f"- Precision@5: {metrics.get('precision_at_5', 0):.3f}")
                report.append(f"- Recall@5: {metrics.get('recall_at_5', 0):.3f}")
                report.append(f"- Search Time: {metrics.get('search_time_ms', 0):.2f}ms")
                report.append("")
        
        # Performance by Query Type
        type_summary = results.get("summary_by_query_type", {})
        if type_summary:
            report.append("## Results by Query Type")
            report.append("")
            
            for query_type, metrics in type_summary.items():
                report.append(f"### {query_type.title()} Queries")
                report.append(f"- Precision@5: {metrics.get('precision_at_5', 0):.3f}")
                report.append(f"- Recall@5: {metrics.get('recall_at_5', 0):.3f}")
                report.append(f"- F1@5: {metrics.get('f1_at_5', 0):.3f}")
                report.append("")
        
        # Best/Worst Performers
        if "best_precision_test" in overall:
            report.append("## Notable Results")
            report.append("")
            
            best = overall["best_precision_test"]
            worst = overall["worst_precision_test"]
            fastest = overall["fastest_search"]
            slowest = overall["slowest_search"]
            
            report.append(f"**Best Precision:** '{best['query']}' in {best['collection']} (P@5: {best['precision_at_5']:.3f})")
            report.append(f"**Worst Precision:** '{worst['query']}' in {worst['collection']} (P@5: {worst['precision_at_5']:.3f})")
            report.append(f"**Fastest Search:** '{fastest['query']}' ({fastest['search_time_ms']:.2f}ms)")
            report.append(f"**Slowest Search:** '{slowest['query']}' ({slowest['search_time_ms']:.2f}ms)")
            report.append("")
        
        # Recommendations
        recommendations = overall.get("recommendations", [])
        if recommendations:
            report.append("## Recommendations")
            report.append("")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Methodology
        report.append("## Methodology")
        report.append("")
        report.append("This simplified test evaluates search quality using:")
        report.append("- **Real collections:** Uses existing Qdrant collections as test data")
        report.append("- **Generated queries:** Creates realistic test queries based on actual content")
        report.append("- **Ground truth:** Identifies expected relevant documents for each query")
        report.append("- **Standard metrics:** Precision@K, Recall@K, F1@K, Average Precision")
        report.append("- **Performance tracking:** Measures search latency alongside quality")
        report.append("")
        
        report.append("**Note:** This is a simplified evaluation. For production use, consider more sophisticated")
        report.append("embedding-based search and larger test datasets.")
        report.append("")
        
        return "\n".join(report)


def main():
    """Run the simplified recall/precision test."""
    test = SimplifiedRecallPrecisionTest()
    
    try:
        # Run benchmark
        results = test.run_quality_benchmark()
        
        # Save results
        results_dir = Path("recall_precision_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"simple_recall_precision_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate and save report
        report = test.generate_simple_report(results)
        report_file = results_file.with_suffix(".md")
        
        with open(report_file, "w") as f:
            f.write(report)
        
        # Print summary
        print("\n" + "=" * 60)
        print(report)
        print(f"\n📊 Results saved to: {results_file}")
        print(f"📋 Report saved to: {report_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Recall/Precision test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main()
