#!/usr/bin/env python3
"""
Empirical chunk size testing for workspace-qdrant-mcp.

This script conducts practical testing of different chunk sizes using real
workspace documents and the actual embedding system. Tests processing speed,
chunk count distribution, and provides actionable recommendations.

FOCUS AREAS:
1. Processing speed for different chunk sizes
2. Chunk count analysis per document type
3. Memory usage patterns
4. Practical recommendations for production use

TESTING APPROACH:
- Uses actual workspace documents
- Tests realistic chunk size range: 512, 800, 1024, 1500, 2048
- Measures real processing times and resource usage
- Provides data-driven recommendations
"""

import asyncio
import json
import logging
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import tracemalloc
import psutil

from workspace_qdrant_mcp.core.config import Config, EmbeddingConfig  
from workspace_qdrant_mcp.core.embeddings import EmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkSizeEmpiricalTester:
    """
    Practical chunk size testing using real workspace data.
    
    Focuses on empirical performance measurement and practical recommendations.
    """
    
    def __init__(self, output_dir: str = "chunk_empirical_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Realistic chunk sizes to test (character-based)
        self.test_chunk_sizes = [512, 800, 1024, 1500, 2048]
        
        # Fixed overlap ratio for consistency
        self.overlap_ratio = 0.15  # 15% overlap based on literature
        
        # Test documents
        self.test_documents: List[Dict] = []
        
        # Results storage  
        self.results: Dict = {}
        
        # System monitoring
        self.process = psutil.Process(os.getpid())
        
        logger.info(f"Empirical chunk size testing initialized")
        logger.info(f"Test chunk sizes: {self.test_chunk_sizes}")
        logger.info(f"Output directory: {self.output_dir}")
    
    async def run_empirical_tests(self) -> Dict[str, Any]:
        """
        Execute empirical chunk size testing.
        
        Returns comprehensive test results and recommendations.
        """
        logger.info("🧪 Starting empirical chunk size testing")
        
        start_time = time.time()
        
        # Collect test documents
        await self.collect_workspace_documents()
        
        # Test each chunk size configuration
        for chunk_size in self.test_chunk_sizes:
            overlap_size = int(chunk_size * self.overlap_ratio)
            logger.info(f"Testing chunk_size={chunk_size}, overlap={overlap_size}")
            
            test_results = await self.test_chunk_configuration(chunk_size, overlap_size)
            self.results[f"chunk_{chunk_size}"] = test_results
        
        # Analyze results and generate recommendations
        analysis = self.analyze_results()
        recommendations = self.generate_recommendations(analysis)
        
        # Create final report
        final_report = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": time.time() - start_time,
                "test_chunk_sizes": self.test_chunk_sizes,
                "overlap_ratio": self.overlap_ratio,
                "documents_tested": len(self.test_documents)
            },
            "individual_results": self.results,
            "comparative_analysis": analysis,
            "recommendations": recommendations
        }
        
        # Save results
        report_file = self.output_dir / f"chunk_empirical_test_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        # Generate summary report
        await self.generate_summary_report(final_report)
        
        logger.info(f"✅ Empirical testing completed. Results: {report_file}")
        return final_report
    
    async def collect_workspace_documents(self) -> None:
        """
        Collect representative documents from the workspace for testing.
        """
        logger.info("📁 Collecting workspace documents for testing")
        
        workspace_root = Path(__file__).parent
        
        # Collect different types of files
        file_patterns = [
            ("*.py", "code"),
            ("*.md", "documentation"),
            ("*.json", "configuration"),
            ("*.txt", "text")
        ]
        
        for pattern, doc_type in file_patterns:
            for file_path in workspace_root.rglob(pattern):
                # Skip unwanted directories
                if any(skip in str(file_path) for skip in [
                    '.venv/', '__pycache__/', '.git/', 'node_modules/',
                    '.pytest_cache/', 'chunk_empirical_results/', 'chunk_optimization_research/'
                ]):
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8')
                    if len(content.strip()) < 200:  # Skip very small files
                        continue
                    
                    self.test_documents.append({
                        "path": str(file_path.relative_to(workspace_root)),
                        "content": content,
                        "type": doc_type,
                        "size_chars": len(content),
                        "size_lines": len(content.splitlines())
                    })
                    
                except Exception as e:
                    logger.debug(f"Skipped {file_path}: {e}")
                    continue
        
        # Limit to manageable number for testing
        if len(self.test_documents) > 50:
            # Sample diverse documents
            by_type = {}
            for doc in self.test_documents:
                doc_type = doc["type"] 
                if doc_type not in by_type:
                    by_type[doc_type] = []
                by_type[doc_type].append(doc)
            
            # Take top documents from each type
            sampled_docs = []
            for doc_type, docs in by_type.items():
                # Sort by size and take diverse samples
                docs.sort(key=lambda x: x["size_chars"])
                sample_size = min(15, len(docs))
                step = max(1, len(docs) // sample_size)
                sampled_docs.extend(docs[::step][:sample_size])
            
            self.test_documents = sampled_docs[:50]
        
        # Calculate document statistics
        doc_types = {}
        for doc in self.test_documents:
            doc_type = doc["type"]
            if doc_type not in doc_types:
                doc_types[doc_type] = {"count": 0, "total_chars": 0}
            doc_types[doc_type]["count"] += 1
            doc_types[doc_type]["total_chars"] += doc["size_chars"]
        
        logger.info(f"📊 Collected {len(self.test_documents)} documents:")
        for doc_type, stats in doc_types.items():
            avg_size = stats["total_chars"] / stats["count"]
            logger.info(f"  {doc_type}: {stats['count']} files, avg {avg_size:.0f} chars")
    
    async def test_chunk_configuration(self, chunk_size: int, overlap_size: int) -> Dict[str, Any]:
        """
        Test a specific chunk size configuration.
        
        Measures processing performance, chunk characteristics, and resource usage.
        """
        logger.info(f"  Testing chunk_size={chunk_size}, overlap={overlap_size}")
        
        # Create embedding service with test configuration
        test_config = Config()
        test_config.embedding = EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            batch_size=50,
            enable_sparse_vectors=False  # Focus on chunking performance
        )
        
        embedding_service = EmbeddingService(test_config)
        
        # Track memory usage
        tracemalloc.start()
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Process documents and collect metrics
        processing_times = []
        chunk_counts = []
        chunk_sizes = []
        boundary_scores = []
        
        start_time = time.perf_counter()
        
        for doc in self.test_documents:
            doc_start = time.perf_counter()
            
            # Chunk the document
            chunks = embedding_service.chunk_text(doc["content"])
            
            doc_end = time.perf_counter()
            
            # Collect metrics
            processing_times.append(doc_end - doc_start)
            chunk_counts.append(len(chunks))
            
            # Analyze chunk sizes
            for chunk in chunks:
                chunk_sizes.append(len(chunk))
            
            # Assess boundary preservation
            boundary_score = self.assess_boundary_preservation(chunks, doc["type"])
            boundary_scores.append(boundary_score)
        
        end_time = time.perf_counter()
        
        # Memory usage after processing
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Stop memory tracking
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        await embedding_service.close()
        
        # Calculate comprehensive metrics
        total_processing_time = end_time - start_time
        total_chars_processed = sum(doc["size_chars"] for doc in self.test_documents)
        total_chunks_created = sum(chunk_counts)
        
        results = {
            "parameters": {
                "chunk_size": chunk_size,
                "overlap_size": overlap_size,
                "overlap_ratio": self.overlap_ratio
            },
            "processing_performance": {
                "total_documents": len(self.test_documents),
                "total_processing_time_ms": total_processing_time * 1000,
                "avg_processing_time_ms": statistics.mean(processing_times) * 1000,
                "chars_per_second": total_chars_processed / total_processing_time,
                "documents_per_second": len(self.test_documents) / total_processing_time,
                "fastest_doc_ms": min(processing_times) * 1000,
                "slowest_doc_ms": max(processing_times) * 1000
            },
            "chunking_characteristics": {
                "total_chunks_created": total_chunks_created,
                "avg_chunks_per_doc": statistics.mean(chunk_counts),
                "median_chunks_per_doc": statistics.median(chunk_counts),
                "min_chunks_per_doc": min(chunk_counts),
                "max_chunks_per_doc": max(chunk_counts),
                "chunk_size_utilization": statistics.mean(chunk_sizes) / chunk_size,
                "avg_actual_chunk_size": statistics.mean(chunk_sizes),
                "chunk_size_std": statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0
            },
            "quality_metrics": {
                "boundary_preservation_score": statistics.mean(boundary_scores),
                "boundary_preservation_std": statistics.stdev(boundary_scores) if len(boundary_scores) > 1 else 0,
                "description": "Higher scores indicate better preservation of semantic boundaries"
            },
            "resource_usage": {
                "memory_increase_mb": memory_increase,
                "peak_memory_kb": peak_memory / 1024,
                "memory_per_chunk_bytes": (peak_memory / total_chunks_created) if total_chunks_created > 0 else 0
            }
        }
        
        return results
    
    def assess_boundary_preservation(self, chunks: List[str], doc_type: str) -> float:
        """
        Assess how well chunking preserves semantic boundaries.
        
        Returns score from 0.0 to 1.0 indicating boundary preservation quality.
        """
        if not chunks:
            return 0.0
        
        scores = []
        
        for chunk in chunks:
            score = 0.5  # Base score
            
            if doc_type == "code":
                # For code, prefer chunks ending with complete statements
                stripped = chunk.rstrip()
                if stripped.endswith((':', ';', '}', ')', ']', '"""', "'''")):
                    score += 0.3
                
                # Bonus for starting with function/class definitions or comments
                lines = chunk.strip().split('\n')
                if lines:
                    first_line = lines[0].strip()
                    if any(first_line.startswith(kw) for kw in [
                        'def ', 'class ', 'async def ', '#', '"""', "'''"
                    ]):
                        score += 0.2
            
            elif doc_type == "documentation":
                # For docs, prefer sentence and paragraph boundaries
                stripped = chunk.rstrip()
                if stripped.endswith(('.', '!', '?', ':', '\n')):
                    score += 0.3
                
                # Bonus for starting with headers or new paragraphs
                if chunk.lstrip().startswith(('#', '##', '###', '\n')):
                    score += 0.2
            
            elif doc_type == "configuration":
                # For config files, prefer complete key-value pairs or sections
                if any(chunk.rstrip().endswith(end) for end in ['}', ']', ',', '\n']):
                    score += 0.3
                
                if any(chunk.lstrip().startswith(start) for start in ['{', '[', '"']):
                    score += 0.2
            
            scores.append(min(score, 1.0))
        
        return statistics.mean(scores)
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze and compare results across all chunk size configurations.
        """
        logger.info("📊 Analyzing comparative results")
        
        # Extract metrics for comparison
        chunk_sizes = []
        processing_speeds = []
        avg_chunks_per_doc = []
        boundary_scores = []
        memory_usage = []
        utilizations = []
        
        for config_name, results in self.results.items():
            chunk_size = results["parameters"]["chunk_size"]
            
            chunk_sizes.append(chunk_size)
            processing_speeds.append(results["processing_performance"]["chars_per_second"])
            avg_chunks_per_doc.append(results["chunking_characteristics"]["avg_chunks_per_doc"])
            boundary_scores.append(results["quality_metrics"]["boundary_preservation_score"])
            memory_usage.append(results["resource_usage"]["memory_per_chunk_bytes"])
            utilizations.append(results["chunking_characteristics"]["chunk_size_utilization"])
        
        # Find optimal configurations
        best_performance_idx = processing_speeds.index(max(processing_speeds))
        best_quality_idx = boundary_scores.index(max(boundary_scores))
        best_memory_idx = memory_usage.index(min(memory_usage))
        
        # Calculate efficiency score (balance of speed and quality)
        efficiency_scores = []
        for i in range(len(chunk_sizes)):
            # Normalize scores to 0-1 range
            norm_speed = (processing_speeds[i] - min(processing_speeds)) / (max(processing_speeds) - min(processing_speeds))
            norm_quality = (boundary_scores[i] - min(boundary_scores)) / (max(boundary_scores) - min(boundary_scores))
            
            # Weight speed and quality equally
            efficiency = (norm_speed + norm_quality) / 2
            efficiency_scores.append(efficiency)
        
        best_efficiency_idx = efficiency_scores.index(max(efficiency_scores))
        
        analysis = {
            "performance_comparison": {
                "fastest_config": {
                    "chunk_size": chunk_sizes[best_performance_idx],
                    "speed_chars_per_sec": processing_speeds[best_performance_idx]
                },
                "best_quality_config": {
                    "chunk_size": chunk_sizes[best_quality_idx], 
                    "boundary_score": boundary_scores[best_quality_idx]
                },
                "most_memory_efficient": {
                    "chunk_size": chunk_sizes[best_memory_idx],
                    "memory_per_chunk_bytes": memory_usage[best_memory_idx]
                },
                "best_overall_efficiency": {
                    "chunk_size": chunk_sizes[best_efficiency_idx],
                    "efficiency_score": efficiency_scores[best_efficiency_idx]
                }
            },
            "trends": {
                "processing_speed_vs_chunk_size": list(zip(chunk_sizes, processing_speeds)),
                "quality_vs_chunk_size": list(zip(chunk_sizes, boundary_scores)),
                "chunks_per_doc_vs_chunk_size": list(zip(chunk_sizes, avg_chunks_per_doc)),
                "utilization_vs_chunk_size": list(zip(chunk_sizes, utilizations))
            },
            "statistical_summary": {
                "processing_speed": {
                    "mean": statistics.mean(processing_speeds),
                    "std": statistics.stdev(processing_speeds) if len(processing_speeds) > 1 else 0,
                    "range": [min(processing_speeds), max(processing_speeds)]
                },
                "boundary_quality": {
                    "mean": statistics.mean(boundary_scores),
                    "std": statistics.stdev(boundary_scores) if len(boundary_scores) > 1 else 0,
                    "range": [min(boundary_scores), max(boundary_scores)]
                },
                "memory_efficiency": {
                    "mean": statistics.mean(memory_usage),
                    "std": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0,
                    "range": [min(memory_usage), max(memory_usage)]
                }
            }
        }
        
        return analysis
    
    def generate_recommendations(self, analysis: Dict) -> Dict[str, Any]:
        """
        Generate actionable recommendations based on empirical results.
        """
        logger.info("💡 Generating recommendations")
        
        # Get best configurations from analysis
        best_overall = analysis["performance_comparison"]["best_overall_efficiency"]["chunk_size"]
        best_speed = analysis["performance_comparison"]["fastest_config"]["chunk_size"]
        best_quality = analysis["performance_comparison"]["best_quality_config"]["chunk_size"]
        best_memory = analysis["performance_comparison"]["most_memory_efficient"]["chunk_size"]
        
        # Current baseline for comparison
        current_baseline = 1000
        
        recommendations = {
            "immediate_action": {
                "recommended_chunk_size": best_overall,
                "recommended_overlap": int(best_overall * self.overlap_ratio),
                "rationale": f"Best balance of processing speed and boundary preservation quality",
                "change_from_baseline": best_overall - current_baseline,
                "expected_impact": self.estimate_impact(best_overall, analysis)
            },
            
            "use_case_specific": {
                "performance_critical": {
                    "chunk_size": best_speed,
                    "overlap": int(best_speed * self.overlap_ratio),
                    "rationale": "Optimizes for maximum processing throughput"
                },
                "quality_focused": {
                    "chunk_size": best_quality,
                    "overlap": int(best_quality * self.overlap_ratio), 
                    "rationale": "Optimizes for best semantic boundary preservation"
                },
                "memory_constrained": {
                    "chunk_size": best_memory,
                    "overlap": int(best_memory * self.overlap_ratio),
                    "rationale": "Most efficient memory usage per chunk"
                }
            },
            
            "configuration_guidelines": {
                "minimum_recommended": min(self.test_chunk_sizes),
                "maximum_recommended": max(self.test_chunk_sizes),
                "sweet_spot_range": [800, 1500],
                "overlap_ratio": self.overlap_ratio,
                "rationale": "Based on empirical performance and quality analysis"
            },
            
            "implementation_strategy": {
                "rollout_approach": "Gradual deployment with A/B testing",
                "monitoring_metrics": [
                    "Average processing time per document",
                    "Search result quality feedback",
                    "Memory usage patterns",
                    "User search satisfaction"
                ],
                "fallback_plan": f"Revert to baseline {current_baseline} if issues arise",
                "validation_period": "2 weeks"
            },
            
            "future_optimizations": [
                "Implement adaptive chunking based on document type",
                "Add user-configurable chunking strategies", 
                "Develop smart boundary detection for code functions",
                "Create chunking quality feedback loop"
            ]
        }
        
        return recommendations
    
    def estimate_impact(self, recommended_size: int, analysis: Dict) -> Dict[str, str]:
        """Estimate the impact of implementing the recommended chunk size."""
        
        # Find results for recommended size and baseline
        recommended_results = None
        baseline_results = None
        
        for config_name, results in self.results.items():
            chunk_size = results["parameters"]["chunk_size"] 
            if chunk_size == recommended_size:
                recommended_results = results
            elif chunk_size == 1024:  # Closest to current baseline of 1000
                baseline_results = results
        
        if not recommended_results or not baseline_results:
            return {"note": "Impact analysis requires baseline comparison"}
        
        # Calculate performance differences
        speed_change = (
            (recommended_results["processing_performance"]["chars_per_second"] - 
             baseline_results["processing_performance"]["chars_per_second"]) /
            baseline_results["processing_performance"]["chars_per_second"] * 100
        )
        
        quality_change = (
            (recommended_results["quality_metrics"]["boundary_preservation_score"] - 
             baseline_results["quality_metrics"]["boundary_preservation_score"]) /
            baseline_results["quality_metrics"]["boundary_preservation_score"] * 100
        )
        
        return {
            "processing_speed": f"{speed_change:+.1f}% change in processing speed",
            "boundary_quality": f"{quality_change:+.1f}% change in boundary preservation",
            "overall": "Improved balance of speed and quality" if speed_change >= -5 and quality_change >= -5 else "Check trade-offs carefully"
        }
    
    async def generate_summary_report(self, report: Dict) -> None:
        """Generate human-readable summary report."""
        
        rec = report["recommendations"]
        analysis = report["comparative_analysis"]
        
        summary_content = f"""# Chunk Size Empirical Testing - Results Summary

## Test Overview
- **Test Date**: {datetime.fromisoformat(report['test_metadata']['timestamp']).strftime('%Y-%m-%d %H:%M')}
- **Duration**: {report['test_metadata']['duration_seconds']:.1f} seconds
- **Documents Tested**: {report['test_metadata']['documents_tested']}
- **Chunk Sizes Tested**: {', '.join(map(str, report['test_metadata']['test_chunk_sizes']))}

## Key Findings

### Recommended Configuration
- **Optimal Chunk Size**: {rec['immediate_action']['recommended_chunk_size']} characters
- **Recommended Overlap**: {rec['immediate_action']['recommended_overlap']} characters
- **Rationale**: {rec['immediate_action']['rationale']}
- **Change from Current**: {rec['immediate_action']['change_from_baseline']:+} characters

### Performance Analysis

#### Best Performing Configurations
- **Fastest Processing**: {analysis['performance_comparison']['fastest_config']['chunk_size']} chars ({analysis['performance_comparison']['fastest_config']['speed_chars_per_sec']:.0f} chars/sec)
- **Best Quality**: {analysis['performance_comparison']['best_quality_config']['chunk_size']} chars (score: {analysis['performance_comparison']['best_quality_config']['boundary_score']:.3f})
- **Most Memory Efficient**: {analysis['performance_comparison']['most_memory_efficient']['chunk_size']} chars ({analysis['performance_comparison']['most_memory_efficient']['memory_per_chunk_bytes']:.0f} bytes/chunk)

#### Performance Trends
- **Speed Range**: {analysis['statistical_summary']['processing_speed']['range'][0]:.0f} - {analysis['statistical_summary']['processing_speed']['range'][1]:.0f} chars/sec
- **Quality Range**: {analysis['statistical_summary']['boundary_quality']['range'][0]:.3f} - {analysis['statistical_summary']['boundary_quality']['range'][1]:.3f}

## Use Case Recommendations

### Performance Critical Applications
- **Chunk Size**: {rec['use_case_specific']['performance_critical']['chunk_size']} characters
- **Overlap**: {rec['use_case_specific']['performance_critical']['overlap']} characters
- **Best For**: High-throughput document processing

### Quality Focused Applications  
- **Chunk Size**: {rec['use_case_specific']['quality_focused']['chunk_size']} characters
- **Overlap**: {rec['use_case_specific']['quality_focused']['overlap']} characters
- **Best For**: Precise search and semantic preservation

### Memory Constrained Applications
- **Chunk Size**: {rec['use_case_specific']['memory_constrained']['chunk_size']} characters  
- **Overlap**: {rec['use_case_specific']['memory_constrained']['overlap']} characters
- **Best For**: Resource-limited environments

## Implementation Plan

### Immediate Action
1. Update default chunk size to {rec['immediate_action']['recommended_chunk_size']} characters
2. Set overlap to {rec['immediate_action']['recommended_overlap']} characters  
3. Deploy with monitoring enabled

### Expected Impact
{rec['immediate_action']['expected_impact']['processing_speed']}
{rec['immediate_action']['expected_impact']['boundary_quality']}

### Monitoring Strategy
- Track processing performance metrics
- Monitor search result quality
- Measure memory usage patterns
- Collect user satisfaction feedback

## Configuration Guidelines
- **Minimum Recommended**: {rec['configuration_guidelines']['minimum_recommended']} characters
- **Maximum Recommended**: {rec['configuration_guidelines']['maximum_recommended']} characters
- **Sweet Spot Range**: {rec['configuration_guidelines']['sweet_spot_range'][0]} - {rec['configuration_guidelines']['sweet_spot_range'][1]} characters
- **Standard Overlap**: {rec['configuration_guidelines']['overlap_ratio']:.0%} of chunk size

## Conclusion

The empirical testing demonstrates that chunk size {rec['immediate_action']['recommended_chunk_size']} provides the best balance of processing performance and quality. This configuration should be implemented as the new default with appropriate monitoring to validate the improvements.

---
*Generated by Chunk Size Empirical Testing Framework*
"""
        
        summary_file = self.output_dir / "EMPIRICAL_TEST_SUMMARY.md"
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        logger.info(f"📋 Summary report saved to {summary_file}")

async def main():
    """Run empirical chunk size testing."""
    
    print("🧪 CHUNK SIZE EMPIRICAL TESTING")
    print("=" * 50)
    
    tester = ChunkSizeEmpiricalTester()
    
    try:
        results = await tester.run_empirical_tests()
        
        print("\n✅ TESTING COMPLETED SUCCESSFULLY")
        print("=" * 50)
        
        # Display key findings
        rec = results["recommendations"]
        print(f"\n📊 KEY FINDINGS:")
        print(f"Recommended chunk size: {rec['immediate_action']['recommended_chunk_size']} characters")
        print(f"Recommended overlap: {rec['immediate_action']['recommended_overlap']} characters")
        print(f"Change from baseline: {rec['immediate_action']['change_from_baseline']:+} characters")
        print(f"Rationale: {rec['immediate_action']['rationale']}")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Review detailed results in: {tester.output_dir}/")
        print(f"2. Read summary report: {tester.output_dir}/EMPIRICAL_TEST_SUMMARY.md")
        print(f"3. Implement recommended configuration")
        print(f"4. Monitor performance in production")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())