#!/usr/bin/env python3
"""
Quick chunk size analysis for workspace-qdrant-mcp.

Focused analysis to determine optimal chunk sizes without full embedding processing.
Tests chunking behavior directly to provide immediate recommendations.
"""

import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Import just the embeddings service for chunking
import sys
sys.path.append('src')

from workspace_qdrant_mcp.core.config import Config, EmbeddingConfig
from workspace_qdrant_mcp.core.embeddings import EmbeddingService

def analyze_chunk_sizes():
    """Quick analysis of chunk size effectiveness."""
    
    print("🔍 Quick Chunk Size Analysis")
    print("=" * 40)
    
    # Test configurations
    test_chunk_sizes = [512, 800, 1024, 1500, 2048]
    overlap_ratio = 0.15
    
    # Collect sample documents  
    workspace_root = Path('.')
    test_documents = []
    
    print("📁 Collecting sample documents...")
    
    # Get a smaller sample of diverse files
    for pattern in ["*.py", "*.md", "*.json"]:
        for file_path in workspace_root.rglob(pattern):
            if any(skip in str(file_path) for skip in [
                '.venv/', '__pycache__/', '.git/', 'node_modules/',
                'tests/', 'chunk_empirical', 'chunk_optimization'
            ]):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                if 200 <= len(content) <= 50000:  # Reasonable size range
                    test_documents.append({
                        "path": str(file_path),
                        "content": content,
                        "size_chars": len(content),
                        "type": "code" if file_path.suffix == ".py" else 
                               "docs" if file_path.suffix == ".md" else "config"
                    })
                    
                if len(test_documents) >= 15:  # Limit for quick analysis
                    break
            except:
                continue
                
        if len(test_documents) >= 15:
            break
    
    print(f"📊 Testing {len(test_documents)} documents")
    
    # Analyze each chunk size
    results = {}
    
    for chunk_size in test_chunk_sizes:
        overlap_size = int(chunk_size * overlap_ratio)
        print(f"\nTesting chunk_size={chunk_size}, overlap={overlap_size}")
        
        # Create embedding service (without initialization)
        config = Config()
        config.embedding = EmbeddingConfig(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size
        )
        service = EmbeddingService(config)
        
        # Test chunking on documents
        chunk_counts = []
        chunk_sizes_actual = []
        processing_times = []
        boundary_scores = []
        
        for doc in test_documents:
            start_time = time.perf_counter()
            
            chunks = service.chunk_text(doc["content"])
            
            end_time = time.perf_counter()
            
            # Collect metrics
            processing_times.append(end_time - start_time)
            chunk_counts.append(len(chunks))
            
            for chunk in chunks:
                chunk_sizes_actual.append(len(chunk))
            
            # Simple boundary score
            boundary_score = assess_boundaries(chunks, doc["type"])
            boundary_scores.append(boundary_score)
        
        # Calculate metrics
        total_processing_time = sum(processing_times)
        total_chars = sum(doc["size_chars"] for doc in test_documents)
        total_chunks = sum(chunk_counts)
        
        results[chunk_size] = {
            "chunk_size": chunk_size,
            "overlap": overlap_size,
            "processing_speed": total_chars / total_processing_time if total_processing_time > 0 else 0,
            "avg_chunks_per_doc": statistics.mean(chunk_counts),
            "total_chunks": total_chunks,
            "avg_actual_chunk_size": statistics.mean(chunk_sizes_actual) if chunk_sizes_actual else 0,
            "size_utilization": (statistics.mean(chunk_sizes_actual) / chunk_size) if chunk_sizes_actual else 0,
            "boundary_quality": statistics.mean(boundary_scores),
            "processing_time_ms": total_processing_time * 1000
        }
        
        print(f"  Speed: {results[chunk_size]['processing_speed']:.0f} chars/sec")
        print(f"  Avg chunks/doc: {results[chunk_size]['avg_chunks_per_doc']:.1f}")
        print(f"  Boundary quality: {results[chunk_size]['boundary_quality']:.3f}")
        print(f"  Size utilization: {results[chunk_size]['size_utilization']:.1%}")
    
    # Analysis and recommendations
    print("\n📊 ANALYSIS RESULTS")
    print("=" * 40)
    
    # Find best performers
    by_speed = max(results.items(), key=lambda x: x[1]['processing_speed'])
    by_quality = max(results.items(), key=lambda x: x[1]['boundary_quality']) 
    by_utilization = max(results.items(), key=lambda x: x[1]['size_utilization'])
    
    # Calculate efficiency score (balanced approach)
    for chunk_size, data in results.items():
        # Normalize metrics to 0-1 scale
        speeds = [r['processing_speed'] for r in results.values()]
        qualities = [r['boundary_quality'] for r in results.values()]
        
        norm_speed = (data['processing_speed'] - min(speeds)) / (max(speeds) - min(speeds)) if max(speeds) > min(speeds) else 0.5
        norm_quality = (data['boundary_quality'] - min(qualities)) / (max(qualities) - min(qualities)) if max(qualities) > min(qualities) else 0.5
        
        data['efficiency_score'] = (norm_speed + norm_quality) / 2
    
    by_efficiency = max(results.items(), key=lambda x: x[1]['efficiency_score'])
    
    print(f"🚀 Fastest Processing: {by_speed[0]} chars ({by_speed[1]['processing_speed']:.0f} chars/sec)")
    print(f"🎯 Best Quality: {by_quality[0]} chars (score: {by_quality[1]['boundary_quality']:.3f})")
    print(f"📈 Best Utilization: {by_utilization[0]} chars ({by_utilization[1]['size_utilization']:.1%})")
    print(f"⚖️ Best Overall: {by_efficiency[0]} chars (efficiency: {by_efficiency[1]['efficiency_score']:.3f})")
    
    # Detailed comparison table
    print("\n📋 DETAILED COMPARISON")
    print("-" * 80)
    print(f"{'Size':<6} {'Speed':<12} {'Quality':<10} {'Chunks/Doc':<12} {'Utilization':<12} {'Efficiency':<10}")
    print("-" * 80)
    
    for chunk_size in test_chunk_sizes:
        data = results[chunk_size]
        print(f"{chunk_size:<6} {data['processing_speed']:<12.0f} {data['boundary_quality']:<10.3f} "
              f"{data['avg_chunks_per_doc']:<12.1f} {data['size_utilization']:<12.1%} {data['efficiency_score']:<10.3f}")
    
    # Generate recommendations
    current_baseline = 1000
    recommended = by_efficiency[0]
    
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 40)
    print(f"Current baseline: {current_baseline} characters")
    print(f"Recommended size: {recommended} characters")
    print(f"Recommended overlap: {int(recommended * overlap_ratio)} characters")
    print(f"Change: {recommended - current_baseline:+} characters")
    
    improvement_speed = ((results[recommended]['processing_speed'] - 
                         results.get(1024, results[min(results.keys(), key=lambda x: abs(x - current_baseline))])['processing_speed']) /
                        results.get(1024, results[min(results.keys(), key=lambda x: abs(x - current_baseline))])['processing_speed'] * 100)
    
    improvement_quality = ((results[recommended]['boundary_quality'] - 
                           results.get(1024, results[min(results.keys(), key=lambda x: abs(x - current_baseline))])['boundary_quality']) /
                          results.get(1024, results[min(results.keys(), key=lambda x: abs(x - current_baseline))])['boundary_quality'] * 100)
    
    print(f"Expected speed change: {improvement_speed:+.1f}%")
    print(f"Expected quality change: {improvement_quality:+.1f}%")
    
    # Use case specific recommendations
    print(f"\n🎯 USE CASE RECOMMENDATIONS")
    print("-" * 40)
    print(f"Performance Critical: {by_speed[0]} characters (fastest processing)")
    print(f"Quality Focused: {by_quality[0]} characters (best boundaries)")  
    print(f"Balanced/Production: {by_efficiency[0]} characters (best overall)")
    print(f"Memory Constrained: {min(test_chunk_sizes)} characters (smallest chunks)")
    
    # Save results
    output_dir = Path("chunk_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_documents": len(test_documents),
        "configurations_tested": test_chunk_sizes,
        "results": results,
        "recommendations": {
            "current_baseline": current_baseline,
            "recommended_chunk_size": recommended,
            "recommended_overlap": int(recommended * overlap_ratio),
            "rationale": f"Best balance of speed ({results[recommended]['processing_speed']:.0f} chars/sec) and quality ({results[recommended]['boundary_quality']:.3f})",
            "use_cases": {
                "performance": by_speed[0],
                "quality": by_quality[0], 
                "balanced": by_efficiency[0],
                "memory": min(test_chunk_sizes)
            }
        }
    }
    
    with open(output_dir / "chunk_analysis_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_dir}/chunk_analysis_report.json")
    return report

def assess_boundaries(chunks: List[str], doc_type: str) -> float:
    """Simple boundary quality assessment."""
    if not chunks:
        return 0.0
    
    scores = []
    for chunk in chunks:
        score = 0.5  # Base
        
        if doc_type == "code":
            if any(chunk.rstrip().endswith(end) for end in [':', ';', '}', ')', ']']):
                score += 0.3
            if any(chunk.strip().startswith(start) for start in ['def ', 'class ', '#']):
                score += 0.2
                
        elif doc_type == "docs":
            if any(chunk.rstrip().endswith(end) for end in ['.', '!', '?']):
                score += 0.3
            if chunk.strip().startswith('#'):
                score += 0.2
                
        elif doc_type == "config":
            if any(chunk.rstrip().endswith(end) for end in ['}', ']', ',']):
                score += 0.3
                
        scores.append(min(score, 1.0))
    
    return statistics.mean(scores)

if __name__ == "__main__":
    analyze_chunk_sizes()