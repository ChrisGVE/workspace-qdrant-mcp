#!/usr/bin/env python3
"""
Recall and Precision Testing Demonstration for workspace-qdrant-mcp.

This demonstration shows how to implement comprehensive recall and precision
testing methodology for search systems. It provides a complete framework
that can be integrated with any search backend.

KEY FEATURES:
- Ground truth generation methodology
- Standard information retrieval metrics (Precision@K, Recall@K, F1@K, MAP, MRR)
- Performance vs quality trade-off analysis
- Scalability assessment framework
- CI/CD integration ready

METHODOLOGY DEMONSTRATION:
1. Create synthetic test datasets with known relevance
2. Generate diverse query types (exact, semantic, complex, edge cases)
3. Simulate search results with realistic patterns
4. Calculate comprehensive quality metrics
5. Analyze performance characteristics
6. Generate actionable recommendations

This serves as a blueprint for implementing recall/precision testing
against the actual workspace-qdrant-mcp search infrastructure.
"""

import json
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


@dataclass
class TestDocument:
    """Represents a test document with metadata."""
    id: str
    title: str
    content: str
    category: str
    keywords: List[str]
    file_type: str


@dataclass
class TestQuery:
    """Represents a test query with ground truth."""
    query: str
    relevant_doc_ids: Set[str]
    query_type: str  # "exact", "semantic", "complex", "edge_case"
    difficulty: str  # "easy", "medium", "hard"
    expected_min_precision_at_5: float
    description: str


@dataclass
class QualityMetrics:
    """Quality metrics for a single test."""
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
    total_relevant: int
    total_retrieved: int
    meets_target: bool


class RecallPrecisionDemo:
    """Demonstration of comprehensive recall/precision testing methodology."""
    
    def __init__(self):
        self.test_documents: List[TestDocument] = []
        self.test_queries: List[TestQuery] = []
        self.results: List[QualityMetrics] = []
        
        # Create synthetic test data
        self._create_synthetic_dataset()
        self._generate_test_queries()
    
    def _create_synthetic_dataset(self):
        """Create synthetic test dataset representing realistic workspace content."""
        print("📊 Creating synthetic test dataset...")
        
        # Programming-related documents
        programming_docs = [
            TestDocument(
                id="doc_001",
                title="Python Function Documentation",
                content="This document explains how to write Python functions with proper documentation. Include docstrings, type hints, and error handling.",
                category="programming",
                keywords=["python", "function", "documentation", "docstring"],
                file_type="md"
            ),
            TestDocument(
                id="doc_002", 
                title="Search Algorithm Implementation",
                content="Implementation of vector search algorithms using embeddings. Covers dense vectors, sparse vectors, and hybrid search approaches.",
                category="programming",
                keywords=["search", "algorithm", "vector", "embedding"],
                file_type="py"
            ),
            TestDocument(
                id="doc_003",
                title="Database Query Optimization", 
                content="Best practices for optimizing database queries, including indexing strategies and query execution plans.",
                category="programming",
                keywords=["database", "query", "optimization", "indexing"],
                file_type="md"
            )
        ]
        
        # Configuration documents
        config_docs = [
            TestDocument(
                id="doc_004",
                title="Environment Configuration",
                content="Configuration settings for development and production environments. Includes API keys, database connections, and feature flags.",
                category="config",
                keywords=["config", "environment", "api", "database"],
                file_type="json"
            ),
            TestDocument(
                id="doc_005",
                title="Docker Configuration",
                content="Docker configuration for containerized deployment. Includes Dockerfile, docker-compose, and environment setup.",
                category="config",
                keywords=["docker", "container", "deployment", "setup"],
                file_type="yml"
            )
        ]
        
        # Documentation files
        doc_docs = [
            TestDocument(
                id="doc_006",
                title="API Reference Guide",
                content="Complete API reference with endpoints, parameters, and response formats. Includes authentication and rate limiting.",
                category="documentation",
                keywords=["api", "reference", "endpoint", "authentication"],
                file_type="md"
            ),
            TestDocument(
                id="doc_007",
                title="Installation Instructions",
                content="Step-by-step installation guide for the workspace-qdrant-mcp project. Covers dependencies, setup, and troubleshooting.",
                category="documentation",
                keywords=["installation", "setup", "dependencies", "troubleshooting"],
                file_type="md"
            )
        ]
        
        # Test files
        test_docs = [
            TestDocument(
                id="doc_008",
                title="Unit Test Suite",
                content="Comprehensive unit tests for search functionality. Tests precision, recall, performance, and edge cases.",
                category="testing",
                keywords=["test", "unittest", "search", "precision", "recall"],
                file_type="py"
            ),
            TestDocument(
                id="doc_009",
                title="Performance Benchmarks",
                content="Performance benchmarking tools and results. Measures search latency, throughput, and quality metrics.",
                category="testing", 
                keywords=["benchmark", "performance", "latency", "quality"],
                file_type="py"
            )
        ]
        
        # Edge case documents
        edge_docs = [
            TestDocument(
                id="doc_010",
                title="Special Characters Test",
                content="Test document with special characters: @#$%^&*()! This tests search robustness.",
                category="edge_case",
                keywords=["special", "characters", "test", "robustness"],
                file_type="txt"
            ),
            TestDocument(
                id="doc_011",
                title="Empty Content Document",
                content="",
                category="edge_case",
                keywords=[],
                file_type="txt"
            )
        ]
        
        self.test_documents = programming_docs + config_docs + doc_docs + test_docs + edge_docs
        print(f"✅ Created {len(self.test_documents)} synthetic documents")
    
    def _generate_test_queries(self):
        """Generate diverse test queries with ground truth."""
        print("🔍 Generating test queries with ground truth...")
        
        test_queries = []
        
        # 1. Exact match queries (easy difficulty)
        exact_queries = [
            TestQuery(
                query="Python function documentation",
                relevant_doc_ids={"doc_001"},
                query_type="exact",
                difficulty="easy",
                expected_min_precision_at_5=0.8,
                description="Exact phrase match for Python documentation"
            ),
            TestQuery(
                query="Docker configuration",
                relevant_doc_ids={"doc_005"},
                query_type="exact",
                difficulty="easy",
                expected_min_precision_at_5=0.8,
                description="Exact match for Docker config"
            )
        ]
        
        # 2. Semantic queries (medium difficulty)
        semantic_queries = [
            TestQuery(
                query="search algorithms and vector embeddings",
                relevant_doc_ids={"doc_002", "doc_008"},  # Search implementation + tests
                query_type="semantic",
                difficulty="medium",
                expected_min_precision_at_5=0.6,
                description="Semantic search for search-related content"
            ),
            TestQuery(
                query="setup installation guide",
                relevant_doc_ids={"doc_007", "doc_005"},  # Installation + Docker setup
                query_type="semantic",
                difficulty="medium",
                expected_min_precision_at_5=0.6,
                description="Setup and installation related documents"
            ),
            TestQuery(
                query="performance testing benchmarks",
                relevant_doc_ids={"doc_009", "doc_008"},  # Benchmarks + tests
                query_type="semantic",
                difficulty="medium", 
                expected_min_precision_at_5=0.6,
                description="Performance and testing related content"
            )
        ]
        
        # 3. Complex queries (hard difficulty)
        complex_queries = [
            TestQuery(
                query="API authentication database optimization",
                relevant_doc_ids={"doc_006", "doc_003", "doc_004"},  # API + DB + Config
                query_type="complex",
                difficulty="hard",
                expected_min_precision_at_5=0.4,
                description="Complex multi-topic query"
            ),
            TestQuery(
                query="testing documentation with examples",
                relevant_doc_ids={"doc_008", "doc_009", "doc_006", "doc_007"},  # Multiple docs
                query_type="complex",
                difficulty="hard",
                expected_min_precision_at_5=0.5,
                description="Broad query covering multiple document types"
            )
        ]
        
        # 4. Edge case queries (hard difficulty)
        edge_queries = [
            TestQuery(
                query="nonexistent topic xyz123",
                relevant_doc_ids=set(),  # No relevant docs
                query_type="edge_case",
                difficulty="hard",
                expected_min_precision_at_5=0.0,
                description="Query with no relevant results"
            ),
            TestQuery(
                query="@#$%^&*()",
                relevant_doc_ids={"doc_010"},  # Special characters doc
                query_type="edge_case",
                difficulty="hard",
                expected_min_precision_at_5=0.2,
                description="Special characters query"
            ),
            TestQuery(
                query="a",  # Single character
                relevant_doc_ids=set(),  # Too broad, no specific relevance
                query_type="edge_case",
                difficulty="hard",
                expected_min_precision_at_5=0.0,
                description="Single character query"
            )
        ]
        
        self.test_queries = exact_queries + semantic_queries + complex_queries + edge_queries
        print(f"✅ Generated {len(self.test_queries)} test queries")
    
    def _simulate_search_results(self, query: TestQuery, search_mode: str = "hybrid") -> Tuple[List[str], float]:
        """Simulate search results with realistic patterns."""
        
        # Simulate search latency based on complexity
        if query.difficulty == "easy":
            search_time = random.uniform(1.0, 5.0)  # 1-5ms for easy queries
        elif query.difficulty == "medium":
            search_time = random.uniform(3.0, 15.0)  # 3-15ms for medium queries
        else:
            search_time = random.uniform(10.0, 50.0)  # 10-50ms for hard queries
        
        # Simulate different search modes affecting results
        mode_factors = {
            "exact": 1.2,  # Better for exact matches
            "semantic": 1.0,  # Balanced
            "hybrid": 1.1   # Slightly better overall
        }
        
        query_mode_factor = mode_factors.get(search_mode, 1.0)
        
        # Generate realistic search results
        results = []
        all_doc_ids = [doc.id for doc in self.test_documents]
        
        # First, add some relevant documents (if any)
        relevant_docs = list(query.relevant_doc_ids)
        if relevant_docs:
            # Add relevant docs with high probability but not certainty (realistic)
            for doc_id in relevant_docs:
                # Probability of finding relevant doc depends on query type and difficulty
                if query.query_type == "exact" and query.difficulty == "easy":
                    prob = 0.95 * query_mode_factor
                elif query.query_type == "semantic" and query.difficulty == "medium":
                    prob = 0.7 * query_mode_factor
                elif query.query_type == "complex":
                    prob = 0.5 * query_mode_factor
                else:  # edge_case
                    prob = 0.2 * query_mode_factor
                
                if random.random() < min(prob, 1.0):
                    results.append(doc_id)
        
        # Add some non-relevant documents (noise)
        non_relevant_docs = [doc_id for doc_id in all_doc_ids if doc_id not in query.relevant_doc_ids]
        
        # Add noise based on query difficulty
        if query.difficulty == "easy":
            noise_count = random.randint(0, 2)  # Low noise for easy queries
        elif query.difficulty == "medium":
            noise_count = random.randint(1, 5)  # Medium noise
        else:
            noise_count = random.randint(2, 8)  # High noise for hard queries
        
        noise_docs = random.sample(non_relevant_docs, min(noise_count, len(non_relevant_docs)))
        results.extend(noise_docs)
        
        # Shuffle and limit results (realistic search behavior)
        random.shuffle(results)
        results = results[:10]  # Typical search result limit
        
        return results, search_time
    
    def calculate_quality_metrics(
        self, 
        relevant_doc_ids: Set[str], 
        retrieved_doc_ids: List[str],
        search_time_ms: float,
        expected_min_precision: float
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        
        relevant_set = relevant_doc_ids
        retrieved_list = retrieved_doc_ids
        
        # Calculate precision@k, recall@k, f1@k for different k values
        k_values = [1, 5, 10]
        precision = {}
        recall = {}
        f1 = {}
        
        for k in k_values:
            top_k = set(retrieved_list[:k])
            relevant_in_top_k = len(top_k.intersection(relevant_set))
            
            # Precision@k
            p_k = relevant_in_top_k / min(k, len(retrieved_list)) if retrieved_list else 0.0
            precision[k] = p_k
            
            # Recall@k
            r_k = relevant_in_top_k / len(relevant_set) if relevant_set else 0.0
            recall[k] = r_k
            
            # F1@k
            if p_k + r_k > 0:
                f1[k] = 2 * (p_k * r_k) / (p_k + r_k)
            else:
                f1[k] = 0.0
        
        # Average Precision
        average_precision = 0.0
        if relevant_set and retrieved_list:
            relevant_count = 0
            precision_sum = 0.0
            
            for i, doc_id in enumerate(retrieved_list):
                if doc_id in relevant_set:
                    relevant_count += 1
                    precision_at_i = relevant_count / (i + 1)
                    precision_sum += precision_at_i
            
            average_precision = precision_sum / len(relevant_set)
        
        # Mean Reciprocal Rank
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_list):
            if doc_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        
        # Check if quality target is met
        meets_target = precision[5] >= expected_min_precision
        
        return QualityMetrics(
            precision_at_1=precision[1],
            precision_at_5=precision[5],
            precision_at_10=precision[10],
            recall_at_1=recall[1],
            recall_at_5=recall[5],
            recall_at_10=recall[10],
            f1_at_1=f1[1],
            f1_at_5=f1[5],
            f1_at_10=f1[10],
            average_precision=average_precision,
            mean_reciprocal_rank=mrr,
            search_time_ms=search_time_ms,
            total_relevant=len(relevant_set),
            total_retrieved=len(retrieved_list),
            meets_target=meets_target
        )
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive recall/precision evaluation demonstration."""
        print("🎯 Running Comprehensive Recall/Precision Evaluation Demo")
        print("=" * 65)
        
        start_time = time.time()
        
        # Test different search modes
        search_modes = ["exact", "semantic", "hybrid"]
        
        all_results = []
        
        # Run tests for each query and search mode
        total_tests = len(self.test_queries) * len(search_modes)
        test_count = 0
        
        for query in self.test_queries:
            for search_mode in search_modes:
                test_count += 1
                print(f"  [{test_count:2d}/{total_tests}] Testing '{query.query[:40]:<40}' ({search_mode})")
                
                # Simulate search
                retrieved_ids, search_time = self._simulate_search_results(query, search_mode)
                
                # Calculate metrics
                metrics = self.calculate_quality_metrics(
                    relevant_doc_ids=query.relevant_doc_ids,
                    retrieved_doc_ids=retrieved_ids,
                    search_time_ms=search_time,
                    expected_min_precision=query.expected_min_precision_at_5
                )
                
                # Store result with context
                result_record = {
                    "query": query.query,
                    "query_type": query.query_type,
                    "difficulty": query.difficulty,
                    "search_mode": search_mode,
                    "expected_min_precision_at_5": query.expected_min_precision_at_5,
                    "metrics": asdict(metrics)
                }
                
                all_results.append(result_record)
                
                # Show quick result
                status = "✅" if metrics.meets_target else "❌"
                print(f"    {status} P@5: {metrics.precision_at_5:.3f}, R@5: {metrics.recall_at_5:.3f}, Time: {metrics.search_time_ms:.1f}ms")
        
        end_time = time.time()
        
        # Generate comprehensive analysis
        analysis = self._analyze_results(all_results)
        
        results = {
            "metadata": {
                "timestamp": time.time(),
                "duration_seconds": end_time - start_time,
                "total_tests": len(all_results),
                "total_documents": len(self.test_documents),
                "total_queries": len(self.test_queries),
                "search_modes": search_modes
            },
            "raw_results": all_results,
            "analysis": analysis
        }
        
        print(f"\n✅ Demo evaluation completed in {end_time - start_time:.1f}s")
        return results
    
    def _analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze results and generate insights."""
        
        # Overall statistics
        all_metrics = [r["metrics"] for r in results]
        
        overall_stats = {
            "avg_precision_at_1": statistics.mean([m["precision_at_1"] for m in all_metrics]),
            "avg_precision_at_5": statistics.mean([m["precision_at_5"] for m in all_metrics]),
            "avg_precision_at_10": statistics.mean([m["precision_at_10"] for m in all_metrics]),
            "avg_recall_at_5": statistics.mean([m["recall_at_5"] for m in all_metrics]),
            "avg_f1_at_5": statistics.mean([m["f1_at_5"] for m in all_metrics]),
            "avg_search_time_ms": statistics.mean([m["search_time_ms"] for m in all_metrics]),
            "quality_target_success_rate": statistics.mean([float(m["meets_target"]) for m in all_metrics])
        }
        
        # Analysis by query type
        by_query_type = {}
        for query_type in ["exact", "semantic", "complex", "edge_case"]:
            type_results = [r for r in results if r["query_type"] == query_type]
            if type_results:
                type_metrics = [r["metrics"] for r in type_results]
                by_query_type[query_type] = {
                    "count": len(type_results),
                    "avg_precision_at_5": statistics.mean([m["precision_at_5"] for m in type_metrics]),
                    "avg_recall_at_5": statistics.mean([m["recall_at_5"] for m in type_metrics]),
                    "avg_f1_at_5": statistics.mean([m["f1_at_5"] for m in type_metrics]),
                    "success_rate": statistics.mean([float(m["meets_target"]) for m in type_metrics])
                }
        
        # Analysis by search mode
        by_search_mode = {}
        for search_mode in ["exact", "semantic", "hybrid"]:
            mode_results = [r for r in results if r["search_mode"] == search_mode]
            if mode_results:
                mode_metrics = [r["metrics"] for r in mode_results]
                by_search_mode[search_mode] = {
                    "count": len(mode_results),
                    "avg_precision_at_5": statistics.mean([m["precision_at_5"] for m in mode_metrics]),
                    "avg_recall_at_5": statistics.mean([m["recall_at_5"] for m in mode_metrics]),
                    "avg_f1_at_5": statistics.mean([m["f1_at_5"] for m in mode_metrics]),
                    "avg_search_time_ms": statistics.mean([m["search_time_ms"] for m in mode_metrics]),
                    "success_rate": statistics.mean([float(m["meets_target"]) for m in mode_metrics])
                }
        
        # Analysis by difficulty
        by_difficulty = {}
        for difficulty in ["easy", "medium", "hard"]:
            diff_results = [r for r in results if r["difficulty"] == difficulty]
            if diff_results:
                diff_metrics = [r["metrics"] for r in diff_results]
                by_difficulty[difficulty] = {
                    "count": len(diff_results),
                    "avg_precision_at_5": statistics.mean([m["precision_at_5"] for m in diff_metrics]),
                    "avg_recall_at_5": statistics.mean([m["recall_at_5"] for m in diff_metrics]),
                    "success_rate": statistics.mean([float(m["meets_target"]) for m in diff_metrics])
                }
        
        # Find best and worst performers
        best_precision = max(results, key=lambda r: r["metrics"]["precision_at_5"])
        worst_precision = min(results, key=lambda r: r["metrics"]["precision_at_5"])
        fastest_search = min(results, key=lambda r: r["metrics"]["search_time_ms"])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_stats, by_search_mode, by_query_type, by_difficulty
        )
        
        return {
            "overall_statistics": overall_stats,
            "by_query_type": by_query_type,
            "by_search_mode": by_search_mode,
            "by_difficulty": by_difficulty,
            "best_performers": {
                "best_precision": {
                    "query": best_precision["query"],
                    "search_mode": best_precision["search_mode"],
                    "precision_at_5": best_precision["metrics"]["precision_at_5"]
                },
                "worst_precision": {
                    "query": worst_precision["query"],
                    "search_mode": worst_precision["search_mode"],
                    "precision_at_5": worst_precision["metrics"]["precision_at_5"]
                },
                "fastest_search": {
                    "query": fastest_search["query"],
                    "search_mode": fastest_search["search_mode"],
                    "search_time_ms": fastest_search["metrics"]["search_time_ms"]
                }
            },
            "recommendations": recommendations
        }
    
    def _generate_recommendations(self, overall: Dict, by_mode: Dict, by_type: Dict, by_diff: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Overall performance assessment
        avg_precision = overall["avg_precision_at_5"]
        avg_recall = overall["avg_recall_at_5"]
        avg_time = overall["avg_search_time_ms"]
        success_rate = overall["quality_target_success_rate"]
        
        # Performance recommendations
        if avg_time > 20:
            recommendations.append(f"Search latency optimization needed: {avg_time:.1f}ms average exceeds 20ms target")
        elif avg_time < 5:
            recommendations.append(f"Excellent search performance: {avg_time:.1f}ms average")
        
        # Quality recommendations
        if avg_precision < 0.5:
            recommendations.append(f"Low precision detected: {avg_precision:.3f}@5. Consider improving relevance ranking.")
        if avg_recall < 0.4:
            recommendations.append(f"Low recall detected: {avg_recall:.3f}@5. Consider expanding search coverage.")
        
        if success_rate < 0.7:
            recommendations.append(f"Quality targets not met: {success_rate:.1%} success rate. Review expectations or improve search.")
        
        # Mode-specific recommendations
        if by_mode:
            best_mode = max(by_mode.keys(), key=lambda m: by_mode[m]["avg_f1_at_5"])
            best_f1 = by_mode[best_mode]["avg_f1_at_5"]
            recommendations.append(f"Best performing search mode: '{best_mode}' (F1@5: {best_f1:.3f})")
        
        # Query type recommendations
        if by_type:
            for query_type, stats in by_type.items():
                if stats["success_rate"] < 0.5:
                    recommendations.append(f"'{query_type}' queries underperforming: {stats['success_rate']:.1%} success rate")
        
        # Difficulty analysis
        if by_diff:
            for difficulty, stats in by_diff.items():
                if difficulty == "easy" and stats["success_rate"] < 0.8:
                    recommendations.append(f"Easy queries should perform better: {stats['success_rate']:.1%} success rate")
                elif difficulty == "hard" and stats["success_rate"] > 0.3:
                    recommendations.append(f"Hard queries performing well: {stats['success_rate']:.1%} success rate")
        
        return recommendations
    
    def generate_demo_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive demonstration report."""
        
        report = []
        report.append("# Recall and Precision Testing Methodology Demonstration")
        report.append("=" * 65)
        report.append("")
        
        metadata = results["metadata"]
        analysis = results["analysis"]
        
        report.append(f"**Demo Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['timestamp']))}")
        report.append(f"**Duration:** {metadata['duration_seconds']:.1f} seconds")
        report.append(f"**Tests Run:** {metadata['total_tests']}")
        report.append(f"**Documents:** {metadata['total_documents']}")
        report.append(f"**Queries:** {metadata['total_queries']}")
        report.append(f"**Search Modes:** {', '.join(metadata['search_modes'])}")
        report.append("")
        
        # Executive Summary
        overall = analysis["overall_statistics"]
        report.append("## Executive Summary")
        report.append("")
        report.append(f"- **Average Precision@5:** {overall['avg_precision_at_5']:.3f}")
        report.append(f"- **Average Recall@5:** {overall['avg_recall_at_5']:.3f}")
        report.append(f"- **Average F1@5:** {overall['avg_f1_at_5']:.3f}")
        report.append(f"- **Average Search Time:** {overall['avg_search_time_ms']:.2f}ms")
        report.append(f"- **Quality Target Success Rate:** {overall['quality_target_success_rate']:.1%}")
        report.append("")
        
        # Performance by Search Mode
        by_mode = analysis["by_search_mode"]
        report.append("## Performance by Search Mode")
        report.append("")
        
        for mode, stats in by_mode.items():
            report.append(f"### {mode.title()} Mode")
            report.append(f"- Tests: {stats['count']}")
            report.append(f"- Precision@5: {stats['avg_precision_at_5']:.3f}")
            report.append(f"- Recall@5: {stats['avg_recall_at_5']:.3f}")
            report.append(f"- F1@5: {stats['avg_f1_at_5']:.3f}")
            report.append(f"- Avg Time: {stats['avg_search_time_ms']:.2f}ms")
            report.append(f"- Success Rate: {stats['success_rate']:.1%}")
            report.append("")
        
        # Performance by Query Type
        by_type = analysis["by_query_type"]
        report.append("## Performance by Query Type")
        report.append("")
        
        for query_type, stats in by_type.items():
            report.append(f"### {query_type.replace('_', ' ').title()} Queries")
            report.append(f"- Tests: {stats['count']}")
            report.append(f"- Precision@5: {stats['avg_precision_at_5']:.3f}")
            report.append(f"- Recall@5: {stats['avg_recall_at_5']:.3f}")
            report.append(f"- F1@5: {stats['avg_f1_at_5']:.3f}")
            report.append(f"- Success Rate: {stats['success_rate']:.1%}")
            report.append("")
        
        # Performance by Difficulty
        by_diff = analysis["by_difficulty"]
        report.append("## Performance by Difficulty Level")
        report.append("")
        
        for difficulty, stats in by_diff.items():
            report.append(f"### {difficulty.title()} Queries")
            report.append(f"- Tests: {stats['count']}")
            report.append(f"- Precision@5: {stats['avg_precision_at_5']:.3f}")
            report.append(f"- Recall@5: {stats['avg_recall_at_5']:.3f}")
            report.append(f"- Success Rate: {stats['success_rate']:.1%}")
            report.append("")
        
        # Best Performers
        best = analysis["best_performers"]
        report.append("## Notable Results")
        report.append("")
        report.append(f"**Best Precision:** '{best['best_precision']['query']}' using {best['best_precision']['search_mode']} mode (P@5: {best['best_precision']['precision_at_5']:.3f})")
        report.append(f"**Worst Precision:** '{best['worst_precision']['query']}' using {best['worst_precision']['search_mode']} mode (P@5: {best['worst_precision']['precision_at_5']:.3f})")
        report.append(f"**Fastest Search:** '{best['fastest_search']['query']}' using {best['fastest_search']['search_mode']} mode ({best['fastest_search']['search_time_ms']:.1f}ms)")
        report.append("")
        
        # Recommendations
        recommendations = analysis["recommendations"]
        if recommendations:
            report.append("## Recommendations")
            report.append("")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Methodology
        report.append("## Methodology Demonstration")
        report.append("")
        report.append("This demonstration illustrates a complete recall/precision testing framework:")
        report.append("")
        report.append("### 1. Ground Truth Generation")
        report.append("- Created synthetic dataset with realistic content categories")
        report.append("- Manually curated relevant document sets for each query")
        report.append("- Established quality targets based on query difficulty")
        report.append("")
        report.append("### 2. Comprehensive Test Coverage")
        report.append("- **Exact Match:** Direct phrase matching (easy difficulty)")
        report.append("- **Semantic:** Topic-based queries (medium difficulty)")
        report.append("- **Complex:** Multi-concept queries (hard difficulty)")
        report.append("- **Edge Cases:** Unusual or challenging queries (hard difficulty)")
        report.append("")
        report.append("### 3. Quality Metrics")
        report.append("- **Precision@K:** Accuracy of returned results")
        report.append("- **Recall@K:** Coverage of relevant documents")
        report.append("- **F1@K:** Balanced precision/recall measure")
        report.append("- **Average Precision:** Area under precision-recall curve")
        report.append("- **Mean Reciprocal Rank:** Position of first relevant result")
        report.append("")
        report.append("### 4. Performance Integration")
        report.append("- Search latency measurement")
        report.append("- Quality vs speed trade-off analysis")
        report.append("- Search mode comparison (exact, semantic, hybrid)")
        report.append("")
        report.append("### 5. Actionable Analysis")
        report.append("- Success rate against quality targets")
        report.append("- Performance breakdown by query characteristics")
        report.append("- Specific recommendations for optimization")
        report.append("")
        
        # Implementation Notes
        report.append("## Implementation for workspace-qdrant-mcp")
        report.append("")
        report.append("To implement this methodology for the actual system:")
        report.append("")
        report.append("1. **Replace simulation with real search calls:**")
        report.append("   - Use `search_workspace()` function from the actual codebase")
        report.append("   - Test hybrid, dense, and sparse search modes")
        report.append("   - Measure actual search latency")
        report.append("")
        report.append("2. **Create realistic test datasets:**")
        report.append("   - Ingest actual project files using ingestion engine")
        report.append("   - Add external project datasets for scale testing")
        report.append("   - Generate embeddings using FastEmbed")
        report.append("")
        report.append("3. **Establish ground truth:**")
        report.append("   - Expert curation of query-document relevance")
        report.append("   - Content analysis to identify relevant documents")
        report.append("   - Quality targets based on use case requirements")
        report.append("")
        report.append("4. **Integrate with CI/CD:**")
        report.append("   - Automated quality regression detection")
        report.append("   - Performance threshold monitoring")
        report.append("   - Regular evaluation against growing datasets")
        report.append("")
        
        return "\n".join(report)


def main():
    """Run the recall/precision testing demonstration."""
    
    # Create and run demo
    demo = RecallPrecisionDemo()
    results = demo.run_comprehensive_evaluation()
    
    # Save results
    results_dir = Path("recall_precision_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    results_file = results_dir / f"recall_precision_demo_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    report = demo.generate_demo_report(results)
    report_file = results_file.with_suffix(".md")
    
    with open(report_file, "w") as f:
        f.write(report)
    
    # Display summary
    print("\n" + "=" * 70)
    print(report)
    print(f"\n📊 Demo results saved to: {results_file}")
    print(f"📋 Demo report saved to: {report_file}")
    
    return results


if __name__ == "__main__":
    random.seed(42)  # For reproducible demo results
    main()
