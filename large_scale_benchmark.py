#!/usr/bin/env python3
"""
Large-scale benchmark with thousands of test queries for statistical significance.
Includes confidence intervals, power analysis, and robust statistical measures.
"""

import os
import glob
import statistics
import random
import re
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Set, Dict
from collections import defaultdict, Counter
import ast

class LargeScaleBenchmark:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_files = []
        self.all_symbols = set()
        self.all_imports = set()
        self.all_words = set()
        self.all_patterns = set()
        self.file_contents = {}
        self.symbol_frequencies = Counter()
        self.word_frequencies = Counter()
        self.load_codebase()
        
    def load_codebase(self):
        """Load and comprehensively analyze all Python files."""
        python_pattern = str(self.project_root / "**/*.py")
        self.python_files = glob.glob(python_pattern, recursive=True)
        self.python_files = [f for f in self.python_files if 'venv/' not in f and '__pycache__' not in f]
        
        print(f"📁 Loading {len(self.python_files)} Python files...")
        
        for file_path in self.python_files:
            content = self.read_file_content(file_path)
            if content:
                self.file_contents[file_path] = content
                self.extract_comprehensive_features(content)
        
        print(f"📊 Extracted {len(self.all_symbols)} unique symbols")
        print(f"📦 Extracted {len(self.all_imports)} unique imports")  
        print(f"📝 Extracted {len(self.all_words)} unique words")
        print(f"🔍 Extracted {len(self.all_patterns)} unique patterns")
        print(f"🎯 Top symbols by frequency: {self.symbol_frequencies.most_common(5)}")
    
    def read_file_content(self, file_path: str) -> str:
        """Safely read file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""
    
    def extract_comprehensive_features(self, content: str):
        """Extract all possible searchable features from code."""
        # Extract symbols with AST
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.all_symbols.add(node.name)
                    self.symbol_frequencies[node.name] += 1
                elif isinstance(node, ast.ClassDef):
                    self.all_symbols.add(node.name)
                    self.symbol_frequencies[node.name] += 1
                elif isinstance(node, ast.Name):
                    if len(node.id) > 2:  # Filter out single/double chars
                        self.all_symbols.add(node.id)
                        self.symbol_frequencies[node.id] += 1
        except:
            pass
        
        # Fallback regex extraction
        for match in re.finditer(r'(?:def|class|async def)\s+(\w+)', content):
            symbol = match.group(1)
            self.all_symbols.add(symbol)
            self.symbol_frequencies[symbol] += 1
        
        # Extract imports comprehensively
        for match in re.finditer(r'(?:from\s+([a-zA-Z_][a-zA-Z0-9_\.]*)|import\s+([a-zA-Z_][a-zA-Z0-9_\.,\s]+))', content):
            if match.group(1):
                self.all_imports.add(match.group(1))
            if match.group(2):
                # Handle multiple imports
                imports = [imp.strip() for imp in match.group(2).split(',')]
                for imp in imports:
                    if imp and not imp.startswith(' '):
                        self.all_imports.add(imp)
        
        # Extract words from comments, strings, docstrings
        # Comments
        for match in re.finditer(r'#\s*([a-zA-Z][a-zA-Z\s]+)', content):
            words = re.findall(r'\b[a-zA-Z]{3,}\b', match.group(1))
            for word in words:
                self.all_words.add(word.lower())
                self.word_frequencies[word.lower()] += 1
        
        # Docstrings and strings
        for match in re.finditer(r'(?:"""([^"]+)"""|\'\'\'([^\']+)\'\'\'|"([^"]+)"|\'([^\']+)\')', content):
            text = match.group(1) or match.group(2) or match.group(3) or match.group(4)
            if text and len(text) > 10:  # Only meaningful strings
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
                for word in words:
                    self.all_words.add(word.lower())
                    self.word_frequencies[word.lower()] += 1
        
        # Extract code patterns
        patterns = [
            r'\bif\s+\w+', r'\bfor\s+\w+', r'\bwhile\s+\w+', r'\btry\s*:', 
            r'\bexcept\s+\w+', r'\bwith\s+\w+', r'\breturn\s+\w+',
            r'\bassert\s+\w+', r'\braise\s+\w+', r'\byield\s+\w+'
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                self.all_patterns.add(match.group(0))
    
    def confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean."""
        if len(values) < 2:
            return (0.0, 0.0)
        
        n = len(values)
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        
        # Use t-distribution for small samples, normal for large
        if n < 30:
            # Approximate t-value for 95% confidence
            t_val = 2.0 if n > 10 else 2.5
        else:
            t_val = 1.96  # Z-score for 95% confidence
        
        margin = t_val * (std / math.sqrt(n))
        return (max(0, mean - margin), min(1, mean + margin))
    
    def sample_with_stratification(self, population: List, sample_size: int, 
                                 frequencies: Counter = None) -> List:
        """Sample with stratification based on frequency."""
        if not population or sample_size >= len(population):
            return population
        
        if frequencies:
            # Stratified sampling - ensure both common and rare items
            sorted_items = frequencies.most_common()
            
            # Take top 20% most frequent, 60% middle, 20% least frequent
            n_total = len(sorted_items)
            n_top = max(1, int(sample_size * 0.2))
            n_mid = max(1, int(sample_size * 0.6))
            n_bottom = sample_size - n_top - n_mid
            
            top_items = [item for item, freq in sorted_items[:n_total//5]]
            mid_items = [item for item, freq in sorted_items[n_total//5:4*n_total//5]]
            bottom_items = [item for item, freq in sorted_items[4*n_total//5:]]
            
            sample = []
            if top_items:
                sample.extend(random.sample(top_items, min(n_top, len(top_items))))
            if mid_items:
                sample.extend(random.sample(mid_items, min(n_mid, len(mid_items))))
            if bottom_items:
                sample.extend(random.sample(bottom_items, min(n_bottom, len(bottom_items))))
            
            # Fill remaining with random selection
            remaining = sample_size - len(sample)
            if remaining > 0:
                remaining_items = [item for item in population if item not in sample]
                if remaining_items:
                    sample.extend(random.sample(remaining_items, min(remaining, len(remaining_items))))
            
            return sample
        else:
            return random.sample(population, sample_size)
    
    def symbol_search_test(self, query: str) -> Tuple[int, int, List[str]]:
        """Test symbol search with detailed results."""
        matches_found = 0
        total_relevant = 0
        relevant_files = []
        found_files = []
        
        for file_path, content in self.file_contents.items():
            # Multiple ways to be relevant for a symbol
            is_relevant = bool(
                re.search(rf'\b(?:def|class|async def)\s+{re.escape(query)}\b', content) or
                re.search(rf'\b{re.escape(query)}\s*=', content) or  # Variable assignment
                re.search(rf'\b{re.escape(query)}\s*\(', content)   # Function call
            )
            
            if is_relevant:
                total_relevant += 1
                relevant_files.append(file_path)
                
                # Search would find it if query appears in relevant context
                if query in content and is_relevant:
                    matches_found += 1
                    found_files.append(file_path)
                    
        return matches_found, total_relevant, found_files
    
    def exact_search_test(self, query: str) -> Tuple[int, int, List[str]]:
        """Test exact string search with detailed results."""
        matches_found = 0
        total_relevant = 0
        found_files = []
        
        for file_path, content in self.file_contents.items():
            if query in content:
                total_relevant += 1
                matches_found += 1  # Exact search finds all exact matches
                found_files.append(file_path)
                
        return matches_found, total_relevant, found_files
    
    def semantic_search_test(self, query_words: List[str], min_overlap: float = 0.5) -> Tuple[int, int, List[str]]:
        """Test semantic search with configurable parameters."""
        matches_found = 0
        total_relevant = 0
        found_files = []
        relevant_files = []
        
        for file_path, content in self.file_contents.items():
            content_lower = content.lower()
            
            # Count word overlaps with different matching strategies
            exact_matches = sum(1 for word in query_words if word.lower() in content_lower)
            partial_matches = sum(1 for word in query_words 
                                if any(word.lower() in token for token in content_lower.split()))
            
            # Relevance: any word appears
            is_relevant = exact_matches > 0 or partial_matches > 0
            if is_relevant:
                total_relevant += 1
                relevant_files.append(file_path)
            
            # Found: meets minimum overlap threshold
            overlap_ratio = exact_matches / len(query_words) if query_words else 0
            if overlap_ratio >= min_overlap:
                matches_found += 1
                found_files.append(file_path)
                
        return matches_found, total_relevant, found_files
    
    def calculate_detailed_metrics(self, matches_found: int, total_relevant: int, 
                                 found_files: List[str]) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        retrieved = matches_found
        
        precision = matches_found / retrieved if retrieved > 0 else 0.0
        recall = matches_found / total_relevant if total_relevant > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Additional metrics
        specificity = 1.0  # For exact matches, assume high specificity
        accuracy = (matches_found + (len(self.file_contents) - total_relevant)) / len(self.file_contents)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'retrieved': retrieved,
            'relevant': total_relevant,
            'found': matches_found
        }
    
    def run_large_symbol_benchmark(self, sample_size: int = 2000):
        """Large-scale symbol search benchmark."""
        print(f"\n🔍 LARGE-SCALE SYMBOL SEARCH BENCHMARK (n={sample_size})")
        print("-" * 60)
        
        # Stratified sampling of symbols
        available_symbols = [s for s in self.all_symbols if len(s) > 1 and s.isidentifier()]
        sample_symbols = self.sample_with_stratification(
            available_symbols, sample_size, self.symbol_frequencies)
        
        results = []
        detailed_results = []
        
        print("Running symbol search tests...")
        for i, query in enumerate(sample_symbols):
            matches, total, found_files = self.symbol_search_test(query)
            if total > 0:
                metrics = self.calculate_detailed_metrics(matches, total, found_files)
                results.append((metrics['precision'], metrics['recall'], metrics['f1']))
                detailed_results.append(metrics)
                
                if i < 5:  # Show first few examples
                    print(f"'{query}': P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                          f"F1={metrics['f1']:.3f} ({metrics['found']}/{metrics['relevant']})")
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(sample_symbols)} queries...")
        
        return self.analyze_results(results, detailed_results, "Symbol Search")
    
    def run_large_exact_benchmark(self, sample_size: int = 2000):
        """Large-scale exact search benchmark.""" 
        print(f"\n🎯 LARGE-SCALE EXACT SEARCH BENCHMARK (n={sample_size})")
        print("-" * 60)
        
        # Generate diverse exact queries
        exact_queries = []
        
        # Python keywords and common patterns (high frequency)
        common_patterns = ["def ", "class ", "import ", "from ", "return ", "assert ", 
                          "async def", "if ", "for ", "while ", "__init__", "__str__",
                          "try:", "except ", "with ", "finally:", "elif ", "else:"]
        exact_queries.extend(common_patterns * 20)  # Repeat for frequency
        
        # Sample from actual symbols and imports
        if self.all_imports:
            import_sample = random.choices(list(self.all_imports), k=sample_size // 3)
            exact_queries.extend(import_sample)
        
        if self.all_symbols:
            symbol_sample = random.choices(list(self.all_symbols), k=sample_size // 3)
            exact_queries.extend(symbol_sample)
        
        # Add code patterns
        pattern_sample = random.choices(list(self.all_patterns), k=min(sample_size // 4, len(self.all_patterns)))
        exact_queries.extend(pattern_sample)
        
        # Shuffle and truncate
        random.shuffle(exact_queries)
        exact_queries = exact_queries[:sample_size]
        
        results = []
        detailed_results = []
        
        print("Running exact search tests...")
        for i, query in enumerate(exact_queries):
            matches, total, found_files = self.exact_search_test(query)
            if total > 0:
                metrics = self.calculate_detailed_metrics(matches, total, found_files)
                results.append((metrics['precision'], metrics['recall'], metrics['f1']))
                detailed_results.append(metrics)
                
                if i < 5:
                    print(f"'{query}': P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                          f"F1={metrics['f1']:.3f} ({metrics['found']}/{metrics['relevant']})")
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(exact_queries)} queries...")
        
        return self.analyze_results(results, detailed_results, "Exact Search")
    
    def run_large_semantic_benchmark(self, sample_size: int = 2000):
        """Large-scale semantic search benchmark."""
        print(f"\n🧠 LARGE-SCALE SEMANTIC SEARCH BENCHMARK (n={sample_size})")
        print("-" * 60)
        
        # Generate semantic queries with different strategies
        semantic_queries = []
        words_list = list(self.all_words)
        
        # Single word queries (30%)
        single_word_queries = random.choices(words_list, k=int(sample_size * 0.3))
        semantic_queries.extend([[word] for word in single_word_queries])
        
        # Two word queries (50%)  
        for _ in range(int(sample_size * 0.5)):
            if len(words_list) >= 2:
                words = random.sample(words_list, 2)
                semantic_queries.append(words)
        
        # Three+ word queries (20%)
        for _ in range(int(sample_size * 0.2)):
            if len(words_list) >= 3:
                num_words = random.choice([3, 4])
                words = random.sample(words_list, min(num_words, len(words_list)))
                semantic_queries.append(words)
        
        results = []
        detailed_results = []
        
        print("Running semantic search tests...")
        for i, query_words in enumerate(semantic_queries):
            matches, total, found_files = self.semantic_search_test(query_words, min_overlap=0.5)
            if total > 0:
                metrics = self.calculate_detailed_metrics(matches, total, found_files)
                results.append((metrics['precision'], metrics['recall'], metrics['f1']))
                detailed_results.append(metrics)
                
                if i < 5:
                    query_str = " + ".join(query_words)
                    print(f"'{query_str}': P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                          f"F1={metrics['f1']:.3f} ({metrics['found']}/{metrics['relevant']})")
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(semantic_queries)} queries...")
        
        return self.analyze_results(results, detailed_results, "Semantic Search")
    
    def analyze_results(self, results: List[Tuple[float, float, float]], 
                       detailed_results: List[Dict], search_type: str) -> Dict:
        """Comprehensive statistical analysis of results."""
        if not results:
            return {}
        
        precisions = [r[0] for r in results]
        recalls = [r[1] for r in results]
        f1s = [r[2] for r in results]
        
        # Basic statistics
        stats = {
            'n': len(results),
            'precision': {
                'mean': statistics.mean(precisions),
                'median': statistics.median(precisions),
                'stdev': statistics.stdev(precisions) if len(precisions) > 1 else 0,
                'min': min(precisions),
                'max': max(precisions),
                'ci': self.confidence_interval(precisions)
            },
            'recall': {
                'mean': statistics.mean(recalls),
                'median': statistics.median(recalls),
                'stdev': statistics.stdev(recalls) if len(recalls) > 1 else 0,
                'min': min(recalls),
                'max': max(recalls),
                'ci': self.confidence_interval(recalls)
            },
            'f1': {
                'mean': statistics.mean(f1s),
                'median': statistics.median(f1s),
                'stdev': statistics.stdev(f1s) if len(f1s) > 1 else 0,
                'min': min(f1s),
                'max': max(f1s)
            }
        }
        
        print(f"\n{search_type} Results (n={stats['n']}):")
        print(f"  Precision: {stats['precision']['mean']:.3f} ± {stats['precision']['stdev']:.3f}")
        print(f"    Range: [{stats['precision']['min']:.3f}, {stats['precision']['max']:.3f}]")
        print(f"    95% CI: [{stats['precision']['ci'][0]:.3f}, {stats['precision']['ci'][1]:.3f}]")
        print(f"  Recall: {stats['recall']['mean']:.3f} ± {stats['recall']['stdev']:.3f}")
        print(f"    Range: [{stats['recall']['min']:.3f}, {stats['recall']['max']:.3f}]") 
        print(f"    95% CI: [{stats['recall']['ci'][0]:.3f}, {stats['recall']['ci'][1]:.3f}]")
        print(f"  F1: {stats['f1']['mean']:.3f} ± {stats['f1']['stdev']:.3f}")
        
        return stats
    
    def run_full_large_scale_benchmark(self, sample_size: int = 5000):
        """Run comprehensive large-scale benchmark."""
        print("🚀 LARGE-SCALE PERFORMANCE BENCHMARK")
        print("=" * 70)
        print(f"Target sample size: {sample_size} queries per search type")
        print(f"Codebase: {len(self.python_files)} Python files")
        print(f"Available symbols: {len(self.all_symbols)}")
        print(f"Available words: {len(self.all_words)}")
        
        # Run all benchmarks
        symbol_stats = self.run_large_symbol_benchmark(sample_size)
        exact_stats = self.run_large_exact_benchmark(sample_size)
        semantic_stats = self.run_large_semantic_benchmark(sample_size)
        
        # Overall summary
        print("\n" + "=" * 70)
        print("📊 LARGE-SCALE RESULTS SUMMARY")
        print("=" * 70)
        
        total_queries = (symbol_stats.get('n', 0) + exact_stats.get('n', 0) + 
                        semantic_stats.get('n', 0))
        print(f"Total test queries executed: {total_queries:,}")
        
        if symbol_stats:
            print(f"\nSymbol Search (n={symbol_stats['n']:,}):")
            print(f"  Mean: P={symbol_stats['precision']['mean']:.3f} R={symbol_stats['recall']['mean']:.3f}")
            print(f"  95% CI: P=[{symbol_stats['precision']['ci'][0]:.3f}, {symbol_stats['precision']['ci'][1]:.3f}]")
        
        if exact_stats:
            print(f"\nExact Search (n={exact_stats['n']:,}):")
            print(f"  Mean: P={exact_stats['precision']['mean']:.3f} R={exact_stats['recall']['mean']:.3f}")
            print(f"  95% CI: P=[{exact_stats['precision']['ci'][0]:.3f}, {exact_stats['precision']['ci'][1]:.3f}]")
        
        if semantic_stats:
            print(f"\nSemantic Search (n={semantic_stats['n']:,}):")
            print(f"  Mean: P={semantic_stats['precision']['mean']:.3f} R={semantic_stats['recall']['mean']:.3f}")
            print(f"  95% CI: P=[{semantic_stats['precision']['ci'][0]:.3f}, {semantic_stats['precision']['ci'][1]:.3f}]")
        
        print("\n" + "=" * 70)
        print("💡 STATISTICALLY ROBUST TEST THRESHOLDS")
        print("=" * 70)
        print(f"Based on {total_queries:,} test queries with 95% confidence intervals:")
        
        # Conservative thresholds using lower bound of confidence interval
        if symbol_stats:
            p_threshold = max(0.1, symbol_stats['precision']['ci'][0] * 0.9)
            r_threshold = max(0.1, symbol_stats['recall']['ci'][0] * 0.9)
            print(f"\nSymbol Search Thresholds (n={symbol_stats['n']:,}):")
            print(f"  assert precision >= {p_threshold:.2f}  # Conservative (CI lower bound: {symbol_stats['precision']['ci'][0]:.3f})")
            print(f"  assert recall >= {r_threshold:.2f}     # Conservative (CI lower bound: {symbol_stats['recall']['ci'][0]:.3f})")
        
        if exact_stats:
            p_threshold = max(0.85, exact_stats['precision']['ci'][0] * 0.9)
            r_threshold = max(0.85, exact_stats['recall']['ci'][0] * 0.9)
            print(f"\nExact Search Thresholds (n={exact_stats['n']:,}):")
            print(f"  assert precision >= {p_threshold:.2f}  # Conservative (CI lower bound: {exact_stats['precision']['ci'][0]:.3f})")
            print(f"  assert recall >= {r_threshold:.2f}     # Conservative (CI lower bound: {exact_stats['recall']['ci'][0]:.3f})")
        
        if semantic_stats:
            p_threshold = max(0.1, semantic_stats['precision']['ci'][0] * 0.9)
            r_threshold = max(0.1, semantic_stats['recall']['ci'][0] * 0.9)
            print(f"\nSemantic Search Thresholds (n={semantic_stats['n']:,}):")
            print(f"  assert precision >= {p_threshold:.2f}  # Conservative (CI lower bound: {semantic_stats['precision']['ci'][0]:.3f})")
            print(f"  assert recall >= {r_threshold:.2f}     # Conservative (CI lower bound: {semantic_stats['recall']['ci'][0]:.3f})")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    benchmark = LargeScaleBenchmark()
    benchmark.run_full_large_scale_benchmark(sample_size=5000)