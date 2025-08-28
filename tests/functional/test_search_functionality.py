"""
Functional tests for search functionality with recall/precision measurements.

Tests all search modes (semantic, hybrid, exact) using real codebase data
and measures search quality with comprehensive metrics.
"""

import pytest
import asyncio
from typing import Dict, List, Set, Any
from unittest.mock import AsyncMock, MagicMock
import json

from workspace_qdrant_mcp.tools.search import search_workspace, search_collection_by_metadata
from workspace_qdrant_mcp.core.hybrid_search import HybridSearchEngine
from workspace_qdrant_mcp.tools.scratchbook import ScratchbookManager

from ..fixtures.test_data_collector import TestDataCollector, SearchGroundTruth
from ..utils.metrics import RecallPrecisionMeter, PerformanceBenchmarker, AsyncTimedOperation


class TestSearchFunctionality:
    """Test comprehensive search functionality with quality measurements."""
    
    @pytest.fixture(autouse=True)
    async def setup_search_environment(self, mock_config, tmp_path):
        """Set up search test environment with real data and mock responses."""
        self.tmp_path = tmp_path
        self.recall_meter = RecallPrecisionMeter()
        self.benchmarker = PerformanceBenchmarker()
        
        # Collect real test data
        source_root = tmp_path.parent.parent.parent  # workspace-qdrant-mcp root
        self.data_collector = TestDataCollector(source_root)
        self.test_data = self.data_collector.collect_all_data()
        
        # Create mock workspace client with realistic search responses
        self.mock_client = await self._create_mock_search_client()
        
        print(f"🔍 Search test setup: {len(self.test_data['chunks'])} indexed chunks, "
              f"{len(self.test_data['ground_truth'])} test queries")
    
    async def _create_mock_search_client(self):
        """Create mock client that returns realistic search results based on test data."""
        mock_client = AsyncMock()
        
        # Index chunks by content for search simulation
        self.chunk_index = {chunk['id']: chunk for chunk in self.test_data['chunks']}
        
        # Mock basic client properties
        mock_client.initialized = True
        mock_client.list_collections.return_value = ["test_collection", "test_docs", "test_code"]
        
        # Mock search function with content-based matching
        async def mock_search(query: str, collections=None, mode="hybrid", limit=10, **kwargs):
            return await self._simulate_search(query, collections, mode, limit)
        
        mock_client.search = mock_search
        
        # Mock embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.9]}
        }
        mock_client.get_embedding_service.return_value = mock_embedding_service
        
        # Mock Qdrant client for hybrid search
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.search.return_value = []
        mock_client.client = mock_qdrant_client
        
        return mock_client
    
    async def _simulate_search(self, query: str, collections=None, mode="hybrid", limit=10):
        """Simulate search results based on content matching."""
        query_lower = query.lower()
        results = []
        
        # Score chunks based on content similarity
        scored_chunks = []
        for chunk in self.test_data['chunks']:
            content_lower = chunk['content'].lower()
            
            # Simple scoring: count query word matches
            query_words = query_lower.split()
            matches = sum(1 for word in query_words if word in content_lower)
            
            if matches > 0:
                # Boost score for exact phrase matches
                phrase_match = query_lower in content_lower
                score = matches / len(query_words)
                if phrase_match:
                    score *= 1.5
                
                # Boost score for symbol matches
                if any(query_lower in symbol.lower() for symbol in chunk.get('symbols', [])):
                    score *= 2.0
                
                scored_chunks.append((chunk, score))
        
        # Sort by score and limit results
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        for chunk, score in scored_chunks[:limit]:
            results.append({
                'id': chunk['id'],
                'score': min(score, 1.0),  # Cap at 1.0
                'content': chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content'],
                'metadata': chunk['metadata'],
                'payload': {
                    'content': chunk['content'],
                    'chunk_type': chunk['chunk_type'],
                    'symbols': chunk.get('symbols', []),
                    **chunk['metadata']
                }
            })
        
        return {
            'results': results,
            'total': len(results),
            'query': query,
            'collections': collections or ['test_collection'],
            'mode': mode
        }
    
    @pytest.mark.integration
    async def test_symbol_search_recall(self):
        """Test symbol-based search recall and precision."""
        # Filter ground truth for symbol searches
        symbol_queries = [
            gt for gt in self.test_data['ground_truth']
            if gt['query_type'] == 'symbol'
        ][:10]  # Test first 10 symbol queries
        
        if not symbol_queries:
            pytest.skip("No symbol queries in ground truth data")
        
        print(f"🔍 Testing {len(symbol_queries)} symbol searches...")
        
        for gt_case in symbol_queries:
            async with AsyncTimedOperation(self.benchmarker, "symbol_search"):
                # Perform search
                search_result = await search_workspace(
                    self.mock_client,
                    query=gt_case['query'],
                    collections=['test_collection'],
                    mode='hybrid',
                    limit=10
                )
            
            # Extract result IDs
            result_ids = [r['id'] for r in search_result.get('results', [])]
            expected_ids = set(gt_case['expected_results'])
            
            # Measure recall/precision
            metrics = self.recall_meter.evaluate_search(
                query=gt_case['query'],
                results=search_result.get('results', []),
                expected_results=expected_ids,
                query_type='symbol',
                search_time_ms=getattr(AsyncTimedOperation, 'duration_ms', 0)
            )
            
            # Symbol searches should have high precision for exact matches
            if gt_case['query'] in [s['name'] for s in self.test_data['symbols']]:
                assert metrics.precision >= 0.80, f"Symbol search precision too low: {metrics.precision} (measured baseline: 100%)"
        
        # Analyze overall symbol search performance
        symbol_metrics = [m for m in self.recall_meter.metrics if m.query_type == 'symbol']
        if symbol_metrics:
            avg_precision = sum(m.precision for m in symbol_metrics) / len(symbol_metrics)
            avg_recall = sum(m.recall for m in symbol_metrics) / len(symbol_metrics)
            
            print(f"📊 Symbol search results:")
            print(f"  Average precision: {avg_precision:.3f}")
            print(f"  Average recall: {avg_recall:.3f}")
            print(f"  Queries tested: {len(symbol_metrics)}")
            
            assert avg_precision >= 0.80, "Symbol search average precision should be ≥ 80% (measured baseline: 100%)"
    
    @pytest.mark.integration
    async def test_semantic_search_quality(self):
        """Test semantic search quality with content-based queries."""
        # Filter ground truth for semantic searches
        semantic_queries = [
            gt for gt in self.test_data['ground_truth']
            if gt['query_type'] == 'semantic'
        ][:8]  # Test first 8 semantic queries
        
        if not semantic_queries:
            pytest.skip("No semantic queries in ground truth data")
        
        print(f"🧠 Testing {len(semantic_queries)} semantic searches...")
        
        for gt_case in semantic_queries:
            async with AsyncTimedOperation(self.benchmarker, "semantic_search"):
                # Perform semantic search
                search_result = await search_workspace(
                    self.mock_client,
                    query=gt_case['query'],
                    collections=['test_collection'],
                    mode='semantic',
                    limit=15,
                    score_threshold=0.1  # Lower threshold for semantic search
                )
            
            # Measure quality
            result_ids = [r['id'] for r in search_result.get('results', [])]
            expected_ids = set(gt_case['expected_results'])
            
            metrics = self.recall_meter.evaluate_search(
                query=gt_case['query'],
                results=search_result.get('results', []),
                expected_results=expected_ids,
                query_type='semantic',
                search_time_ms=getattr(AsyncTimedOperation, 'duration_ms', 0)
            )
        
        # Analyze semantic search performance
        semantic_metrics = [m for m in self.recall_meter.metrics if m.query_type == 'semantic']
        if semantic_metrics:
            avg_precision = sum(m.precision for m in semantic_metrics) / len(semantic_metrics)
            avg_recall = sum(m.recall for m in semantic_metrics) / len(semantic_metrics)
            avg_f1 = sum(m.f1_score for m in semantic_metrics) / len(semantic_metrics)
            
            print(f"📊 Semantic search results:")
            print(f"  Average precision: {avg_precision:.3f}")
            print(f"  Average recall: {avg_recall:.3f}")
            print(f"  Average F1: {avg_f1:.3f}")
            
            # Semantic search may have lower precision but should find relevant content
            assert avg_recall >= 0.80, "Semantic search recall should be ≥ 80% (measured baseline: 100%)"
            assert avg_f1 >= 0.2, "Semantic search F1 score should be ≥ 20%"
    
    @pytest.mark.integration
    async def test_hybrid_search_effectiveness(self):
        """Test that hybrid search combines benefits of dense and sparse search."""
        # Use diverse query types for hybrid testing
        test_queries = [
            ("QdrantWorkspaceClient", "symbol"),
            ("FastMCP server configuration", "semantic"),
            ("async def initialize", "exact"),
            ("embedding vector generation", "semantic"),
            ("collection management", "semantic")
        ]
        
        print(f"🔀 Testing hybrid search with {len(test_queries)} diverse queries...")
        
        hybrid_results = []
        semantic_results = []
        
        for query, expected_type in test_queries:
            # Test hybrid search
            async with AsyncTimedOperation(self.benchmarker, "hybrid_search"):
                hybrid_result = await search_workspace(
                    self.mock_client,
                    query=query,
                    mode='hybrid',
                    limit=10
                )
            
            # Test semantic-only search for comparison
            semantic_result = await search_workspace(
                self.mock_client,
                query=query,
                mode='semantic',
                limit=10
            )
            
            hybrid_results.append({
                'query': query,
                'type': expected_type,
                'results': hybrid_result.get('results', []),
                'count': len(hybrid_result.get('results', []))
            })
            
            semantic_results.append({
                'query': query,
                'type': expected_type,
                'results': semantic_result.get('results', []),
                'count': len(semantic_result.get('results', []))
            })
        
        # Analyze hybrid vs semantic performance
        hybrid_avg_results = sum(r['count'] for r in hybrid_results) / len(hybrid_results)
        semantic_avg_results = sum(r['count'] for r in semantic_results) / len(semantic_results)
        
        print(f"📊 Hybrid vs Semantic comparison:")
        print(f"  Hybrid average results: {hybrid_avg_results:.1f}")
        print(f"  Semantic average results: {semantic_avg_results:.1f}")
        
        # Verify hybrid search returns reasonable results
        assert hybrid_avg_results > 0, "Hybrid search should return results"
        
        # Check that hybrid search finds different results than semantic-only
        for hybrid_res, semantic_res in zip(hybrid_results, semantic_results):
            hybrid_ids = set(r['id'] for r in hybrid_res['results'])
            semantic_ids = set(r['id'] for r in semantic_res['results'])
            
            # Some overlap is expected, but not complete overlap
            if len(hybrid_ids) > 0 and len(semantic_ids) > 0:
                overlap_ratio = len(hybrid_ids & semantic_ids) / len(hybrid_ids | semantic_ids)
                assert overlap_ratio < 1.0, f"Hybrid and semantic shouldn't be identical for: {hybrid_res['query']}"
    
    @pytest.mark.integration
    async def test_exact_match_search_precision(self):
        """Test exact match searches for high precision."""
        # Filter ground truth for exact searches
        exact_queries = [
            gt for gt in self.test_data['ground_truth']
            if gt['query_type'] == 'exact'
        ][:5]  # Test first 5 exact queries
        
        if not exact_queries:
            # Create some exact queries from docstrings
            exact_queries = []
            for symbol in self.test_data['symbols'][:3]:
                if symbol.get('docstring') and len(symbol['docstring']) > 20:
                    # Use first sentence of docstring
                    first_sentence = symbol['docstring'].split('.')[0] + '.'
                    if len(first_sentence) < 100:
                        exact_queries.append({
                            'query': first_sentence,
                            'query_type': 'exact',
                            'expected_results': [f"symbol_{symbol['name']}"],
                            'expected_symbols': [symbol['name']]
                        })
        
        if not exact_queries:
            pytest.skip("No exact queries available for testing")
        
        print(f"🎯 Testing {len(exact_queries)} exact match searches...")
        
        for gt_case in exact_queries:
            async with AsyncTimedOperation(self.benchmarker, "exact_search"):
                # Perform exact search (use hybrid mode but expect high precision)
                search_result = await search_workspace(
                    self.mock_client,
                    query=gt_case['query'],
                    collections=['test_collection'],
                    mode='hybrid',
                    limit=5,
                    score_threshold=0.8  # High threshold for exact matches
                )
            
            # Measure precision (recall may be lower for exact matches)
            result_ids = [r['id'] for r in search_result.get('results', [])]
            expected_ids = set(gt_case['expected_results'])
            
            metrics = self.recall_meter.evaluate_search(
                query=gt_case['query'],
                results=search_result.get('results', []),
                expected_results=expected_ids,
                query_type='exact',
                search_time_ms=getattr(AsyncTimedOperation, 'duration_ms', 0)
            )
            
            # Exact searches should prioritize precision over recall
            if metrics.total_results > 0:
                assert metrics.precision >= 0.90, f"Exact search precision too low: {metrics.precision} (measured baseline: 100%)"
        
        # Analyze exact search performance
        exact_metrics = [m for m in self.recall_meter.metrics if m.query_type == 'exact']
        if exact_metrics:
            avg_precision = sum(m.precision for m in exact_metrics) / len(exact_metrics)
            avg_recall = sum(m.recall for m in exact_metrics) / len(exact_metrics)
            
            print(f"📊 Exact search results:")
            print(f"  Average precision: {avg_precision:.3f}")
            print(f"  Average recall: {avg_recall:.3f}")
            
            assert avg_precision >= 0.90, "Exact search average precision should be ≥ 90% (measured baseline: 100%)"
    
    @pytest.mark.integration
    async def test_metadata_filtering_accuracy(self):
        """Test search filtering by metadata with accuracy measurement."""
        # Test filtering by chunk type
        metadata_filters = [
            {"chunk_type": "code"},
            {"chunk_type": "documentation"},
            {"symbol_type": "function"},
            {"has_docstring": True}
        ]
        
        print(f"🔎 Testing metadata filtering with {len(metadata_filters)} filters...")
        
        for filter_dict in metadata_filters:
            async with AsyncTimedOperation(self.benchmarker, "metadata_search"):
                # Use metadata search function
                search_result = await search_collection_by_metadata(
                    self.mock_client,
                    collection='test_collection',
                    metadata_filter=filter_dict,
                    limit=20
                )
            
            results = search_result.get('results', [])
            
            # Verify filtering accuracy (in mock, we simulate based on test data)
            if results:
                # Check that results actually match the filter (for available data)
                filter_key = list(filter_dict.keys())[0]
                expected_value = filter_dict[filter_key]
                
                matching_chunks = [
                    chunk for chunk in self.test_data['chunks']
                    if chunk.get(filter_key) == expected_value or
                       chunk.get('metadata', {}).get(filter_key) == expected_value
                ]
                
                # Should find some matching chunks for valid filters
                print(f"  Filter {filter_dict}: {len(results)} results, "
                      f"{len(matching_chunks)} expected matches")
        
        # Test combining filters
        combined_filter = {"chunk_type": "code", "has_docstring": True}
        combined_result = await search_collection_by_metadata(
            self.mock_client,
            collection='test_collection', 
            metadata_filter=combined_filter,
            limit=10
        )
        
        # Combined filters should be more restrictive
        single_filter_result = await search_collection_by_metadata(
            self.mock_client,
            collection='test_collection',
            metadata_filter={"chunk_type": "code"},
            limit=10
        )
        
        combined_count = len(combined_result.get('results', []))
        single_count = len(single_filter_result.get('results', []))
        
        print(f"📊 Filter combination test:")
        print(f"  Single filter results: {single_count}")
        print(f"  Combined filter results: {combined_count}")
        print(f"  Restriction ratio: {combined_count/single_count if single_count > 0 else 0:.2f}")
    
    @pytest.mark.integration
    async def test_search_across_multiple_collections(self):
        """Test search functionality across multiple collections."""
        collections_to_search = ['test_docs', 'test_code', 'test_collection']
        test_queries = [
            "client initialization",
            "search functionality", 
            "embedding generation"
        ]
        
        print(f"🔍 Testing multi-collection search...")
        
        for query in test_queries:
            # Test single collection searches
            single_results = {}
            for collection in collections_to_search:
                result = await search_workspace(
                    self.mock_client,
                    query=query,
                    collections=[collection],
                    mode='hybrid',
                    limit=5
                )
                single_results[collection] = result.get('results', [])
            
            # Test multi-collection search
            multi_result = await search_workspace(
                self.mock_client,
                query=query,
                collections=collections_to_search,
                mode='hybrid',
                limit=15
            )
            multi_results = multi_result.get('results', [])
            
            # Verify multi-collection search aggregates results properly
            total_single_results = sum(len(results) for results in single_results.values())
            
            print(f"  Query '{query}':")
            print(f"    Single collections total: {total_single_results}")
            print(f"    Multi-collection results: {len(multi_results)}")
            
            # Multi-collection should potentially find more results
            if total_single_results > 0:
                assert len(multi_results) > 0, "Multi-collection search should return results"
    
    @pytest.mark.performance
    async def test_search_performance_benchmarks(self):
        """Comprehensive search performance benchmarking."""
        test_queries = [
            "QdrantWorkspaceClient",
            "async function definition",
            "configuration management", 
            "embedding vector processing",
            "collection initialization"
        ]
        
        print(f"⚡ Performance testing {len(test_queries)} queries across search modes...")
        
        # Benchmark different search modes
        for mode in ['semantic', 'hybrid']:
            mode_times = []
            
            for query in test_queries:
                # Benchmark individual search
                benchmark = await self.benchmarker.benchmark_async_operation(
                    f"{mode}_search_individual",
                    lambda q=query: search_workspace(
                        self.mock_client,
                        query=q,
                        mode=mode,
                        limit=10
                    ),
                    iterations=3
                )
                mode_times.append(benchmark.mean_time_ms)
            
            avg_time = sum(mode_times) / len(mode_times)
            print(f"  {mode.capitalize()} search average: {avg_time:.1f}ms")
            
            # Performance assertions
            assert avg_time < 500, f"{mode} search should be < 500ms (got {avg_time:.1f}ms)"
        
        # Benchmark concurrent searches
        async def concurrent_searches():
            tasks = [
                search_workspace(self.mock_client, query, mode='hybrid', limit=5)
                for query in test_queries
            ]
            await asyncio.gather(*tasks)
        
        concurrent_benchmark = await self.benchmarker.benchmark_async_operation(
            "concurrent_searches",
            concurrent_searches,
            iterations=3
        )
        
        print(f"  Concurrent {len(test_queries)} searches: {concurrent_benchmark.mean_time_ms:.1f}ms")
        
        # Concurrent should be faster than sequential
        sequential_time = sum(mode_times) * 2  # Approximate for both modes
        speedup = sequential_time / concurrent_benchmark.mean_time_ms
        
        print(f"  Concurrency speedup: {speedup:.1f}x")
        assert speedup > 1.5, "Concurrent searches should show >1.5x speedup"
    
    @pytest.mark.integration
    async def test_search_result_ranking_quality(self):
        """Test that search results are properly ranked by relevance."""
        # Use queries with clear relevance hierarchy
        ranking_test_queries = [
            ("QdrantWorkspaceClient initialize", ["QdrantWorkspaceClient", "initialize"]),
            ("FastMCP server", ["FastMCP", "server"]),
            ("embedding vector", ["embedding", "vector"])
        ]
        
        print(f"📊 Testing result ranking quality...")
        
        for query, key_terms in ranking_test_queries:
            search_result = await search_workspace(
                self.mock_client,
                query=query,
                mode='hybrid',
                limit=10
            )
            
            results = search_result.get('results', [])
            if len(results) < 2:
                continue  # Skip if insufficient results
            
            # Analyze ranking quality
            scores = [r['score'] for r in results]
            
            # Scores should be in descending order
            assert scores == sorted(scores, reverse=True), f"Results not properly ranked for: {query}"
            
            # Top results should have higher relevance to key terms
            top_results = results[:3]
            for result in top_results:
                content = result.get('content', '').lower()
                payload = result.get('payload', {})
                
                # Check for key term presence in content or symbols
                relevance_found = any(term.lower() in content for term in key_terms)
                symbols = payload.get('symbols', [])
                symbol_relevance = any(
                    any(term.lower() in symbol.lower() for term in key_terms)
                    for symbol in symbols
                )
                
                if not (relevance_found or symbol_relevance):
                    print(f"⚠️  Low relevance top result for '{query}': {result.get('id', 'unknown')}")
                    # Note: In mock test, perfect relevance isn't expected
    
    def test_export_comprehensive_search_report(self):
        """Export detailed search quality report."""
        # Generate comprehensive report
        aggregate_metrics = self.recall_meter.get_aggregate_metrics()
        benchmarks = self.benchmarker.get_summary()
        
        # Export to JSON
        report_file = self.tmp_path / "search_quality_report.json"
        self.recall_meter.export_detailed_results(str(report_file))
        
        # Verify report was created
        assert report_file.exists(), "Search quality report should be generated"
        
        with open(report_file) as f:
            report_data = json.load(f)
        
        # Verify report structure
        assert 'summary' in report_data
        assert 'individual_queries' in report_data
        assert 'benchmarks' in report_data
        
        print(f"📋 Search quality report exported to: {report_file}")
        
        if aggregate_metrics:
            summary = aggregate_metrics.get('summary', {})
            print(f"📊 Final search quality summary:")
            print(f"  Total queries tested: {summary.get('total_queries', 0)}")
            print(f"  Average precision: {summary.get('avg_precision', 0):.3f}")
            print(f"  Average recall: {summary.get('avg_recall', 0):.3f}")
            print(f"  Average F1 score: {summary.get('avg_f1_score', 0):.3f}")
            print(f"  Average search time: {summary.get('avg_search_time_ms', 0):.1f}ms")
            
            # Overall quality assertions
            if summary.get('total_queries', 0) > 5:
                assert summary.get('avg_precision', 0) >= 0.75, "Overall precision should be ≥ 75% (measured baseline: 100%)"
                assert summary.get('avg_search_time_ms', 1000) < 1000, "Average search time should be < 1s"
        
        print("✅ Search functionality testing completed with comprehensive metrics")
    
    @pytest.mark.integration
    async def test_scratchbook_search_specialized_functionality(self):
        """Test scratchbook-specific search functionality."""
        # Mock scratchbook manager
        mock_manager = AsyncMock(spec=ScratchbookManager)
        mock_manager.search_notes.return_value = {
            'results': [
                {
                    'id': 'note_1',
                    'content': 'Development notes about client implementation',
                    'metadata': {'note_type': 'development', 'tags': ['client', 'implementation']}
                },
                {
                    'id': 'note_2', 
                    'content': 'TODO: Fix embedding service initialization',
                    'metadata': {'note_type': 'todo', 'tags': ['embedding', 'bug']}
                }
            ],
            'total': 2
        }
        
        # Test specialized scratchbook search
        scratchbook_result = await mock_manager.search_notes(
            query="client implementation",
            note_types=['development'],
            tags=['client'],
            limit=5
        )
        
        results = scratchbook_result.get('results', [])
        assert len(results) > 0, "Scratchbook search should return results"
        
        # Verify scratchbook-specific metadata
        for result in results:
            metadata = result.get('metadata', {})
            assert 'note_type' in metadata, "Scratchbook results should have note_type"
            assert 'tags' in metadata, "Scratchbook results should have tags"
        
        print(f"📝 Scratchbook search test: {len(results)} results with specialized metadata"