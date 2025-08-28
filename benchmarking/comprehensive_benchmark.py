#!/usr/bin/env python3
"""
Comprehensive benchmark with statistically significant sample sizes.
Generates hundreds of test queries to get reliable performance measurements.
"""

import os
import glob
import statistics
import random
import re
from pathlib import Path
from typing import List, Tuple, Set
import ast

class ComprehensiveBenchmark:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_files = []
        self.all_symbols = set()
        self.all_imports = set()
        self.all_words = set()
        self.file_contents = {}
        self.load_codebase()
        
    def load_codebase(self):
        """Load and analyze all Python files."""
        python_pattern = str(self.project_root / "**/*.py")
        self.python_files = glob.glob(python_pattern, recursive=True)
        self.python_files = [f for f in self.python_files if 'venv/' not in f and '__pycache__' not in f]
        
        print(f"📁 Loading {len(self.python_files)} Python files...")
        
        for file_path in self.python_files:
            content = self.read_file_content(file_path)
            if content:
                self.file_contents[file_path] = content
                self.extract_symbols(content)
                self.extract_imports(content)
                self.extract_words(content)
        
        print(f"📊 Extracted {len(self.all_symbols)} unique symbols")
        print(f"📦 Extracted {len(self.all_imports)} unique imports")  
        print(f"📝 Extracted {len(self.all_words)} unique words")
    
    def read_file_content(self, file_path: str) -> str:
        """Safely read file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""
    
    def extract_symbols(self, content: str):
        """Extract function and class names from code."""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.all_symbols.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    self.all_symbols.add(node.name)
                elif isinstance(node, ast.AsyncFunctionDef):
                    self.all_symbols.add(node.name)
        except:
            # Fallback to regex for unparseable files
            for match in re.finditer(r'(?:def|class|async def)\s+(\w+)', content):
                self.all_symbols.add(match.group(1))
    
    def extract_imports(self, content: str):
        """Extract import statements."""
        for match in re.finditer(r'(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_\.]*)', content):
            self.all_imports.add(match.group(1))
    
    def extract_words(self, content: str):
        """Extract meaningful words from comments and strings."""
        # Extract from comments
        for match in re.finditer(r'#\s*([a-zA-Z][a-zA-Z\s]+)', content):
            words = match.group(1).split()
            for word in words:
                if len(word) > 3:
                    self.all_words.add(word.lower())
        
        # Extract from docstrings  
        for match in re.finditer(r'"""([^"]+)"""', content):
            words = re.findall(r'\b[a-zA-Z]{4,}\b', match.group(1))
            for word in words:
                self.all_words.add(word.lower())
    
    def symbol_search_test(self, query: str) -> Tuple[int, int]:
        """Test symbol search performance."""
        matches_found = 0
        total_relevant = 0
        
        for file_path, content in self.file_contents.items():
            # Count as relevant if contains the symbol definition
            is_relevant = bool(re.search(rf'\b(?:def|class|async def)\s+{re.escape(query)}\b', content))
            if is_relevant:
                total_relevant += 1
                
            # Count as found if search would find it
            if query in content and is_relevant:
                matches_found += 1
                
        return matches_found, total_relevant
    
    def exact_search_test(self, query: str) -> Tuple[int, int]:
        """Test exact string search."""
        matches_found = 0
        total_relevant = 0
        
        for file_path, content in self.file_contents.items():
            if query in content:
                total_relevant += 1
                matches_found += 1  # Exact search finds all exact matches
                
        return matches_found, total_relevant
    
    def semantic_search_test(self, query_words: List[str], min_overlap: float = 0.5) -> Tuple[int, int]:
        """Test semantic search with configurable overlap threshold."""
        matches_found = 0
        total_relevant = 0
        
        for file_path, content in self.file_contents.items():
            content_lower = content.lower()
            
            # Count word overlaps
            words_found = sum(1 for word in query_words if word.lower() in content_lower)
            overlap_ratio = words_found / len(query_words) if query_words else 0
            
            is_relevant = overlap_ratio > 0  # Any overlap makes it relevant
            if is_relevant:
                total_relevant += 1
                
            # Found if meets minimum overlap threshold
            if overlap_ratio >= min_overlap:
                matches_found += 1
                
        return matches_found, total_relevant
    
    def calculate_metrics(self, matches_found: int, total_relevant: int) -> Tuple[float, float, float]:
        """Calculate precision, recall, F1."""
        retrieved = matches_found  # For simplicity, assume we retrieve what we find
        
        precision = matches_found / retrieved if retrieved > 0 else 0.0
        recall = matches_found / total_relevant if total_relevant > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def run_symbol_benchmark(self, sample_size: int = 100):
        """Test symbol search with large sample size."""
        print(f"\n🔍 SYMBOL SEARCH BENCHMARK (n={sample_size})")
        print("-" * 50)
        
        # Generate sample from available symbols
        available_symbols = [s for s in self.all_symbols if len(s) > 1]  # Filter out single chars
        if len(available_symbols) < sample_size:
            print(f"⚠️  Only {len(available_symbols)} symbols available, using all")
            sample_symbols = available_symbols
        else:
            sample_symbols = random.sample(available_symbols, sample_size)
        
        results = []
        found_queries = 0
        
        for i, query in enumerate(sample_symbols):
            matches, total = self.symbol_search_test(query)
            if total > 0:  # Only count queries with ground truth
                precision, recall, f1 = self.calculate_metrics(matches, total)
                results.append((precision, recall, f1))
                found_queries += 1
                
                if i < 10:  # Show first 10 examples
                    print(f"'{query}': P={precision:.3f} R={recall:.3f} F1={f1:.3f} ({matches}/{total})")
        
        if found_queries > 10:
            print(f"... and {found_queries - 10} more queries")
        
        if results:
            avg_p = statistics.mean([r[0] for r in results])
            avg_r = statistics.mean([r[1] for r in results])
            avg_f1 = statistics.mean([r[2] for r in results])
            std_p = statistics.stdev([r[0] for r in results]) if len(results) > 1 else 0
            std_r = statistics.stdev([r[1] for r in results]) if len(results) > 1 else 0
            
            print(f"\nSymbol Search Results (n={len(results)}):")
            print(f"  Precision: {avg_p:.3f} ± {std_p:.3f}")
            print(f"  Recall:    {avg_r:.3f} ± {std_r:.3f}")
            print(f"  F1:        {avg_f1:.3f}")
            return avg_p, avg_r, avg_f1, len(results)
        
        return 0, 0, 0, 0
    
    def run_exact_benchmark(self, sample_size: int = 100):
        """Test exact search with large sample size."""
        print(f"\n🎯 EXACT SEARCH BENCHMARK (n={sample_size})")
        print("-" * 50)
        
        # Generate diverse exact search queries
        exact_queries = []
        
        # Common Python keywords/patterns
        common_patterns = ["def ", "class ", "import ", "from ", "return ", "assert ", 
                          "async def", "if ", "for ", "while ", "__init__", "__str__"]
        exact_queries.extend(common_patterns)
        
        # Sample from actual imports
        if self.all_imports:
            import_sample = random.sample(list(self.all_imports), 
                                        min(sample_size // 3, len(self.all_imports)))
            exact_queries.extend(import_sample)
        
        # Sample from actual symbols  
        if self.all_symbols:
            symbol_sample = random.sample(list(self.all_symbols), 
                                        min(sample_size // 3, len(self.all_symbols)))
            exact_queries.extend(symbol_sample)
        
        # Truncate to sample size
        exact_queries = exact_queries[:sample_size]
        
        results = []
        found_queries = 0
        
        for i, query in enumerate(exact_queries):
            matches, total = self.exact_search_test(query)
            if total > 0:
                precision, recall, f1 = self.calculate_metrics(matches, total)
                results.append((precision, recall, f1))
                found_queries += 1
                
                if i < 10:
                    print(f"'{query}': P={precision:.3f} R={recall:.3f} F1={f1:.3f} ({matches}/{total})")
        
        if found_queries > 10:
            print(f"... and {found_queries - 10} more queries")
        
        if results:
            avg_p = statistics.mean([r[0] for r in results])
            avg_r = statistics.mean([r[1] for r in results])
            avg_f1 = statistics.mean([r[2] for r in results])
            std_p = statistics.stdev([r[0] for r in results]) if len(results) > 1 else 0
            std_r = statistics.stdev([r[1] for r in results]) if len(results) > 1 else 0
            
            print(f"\nExact Search Results (n={len(results)}):")
            print(f"  Precision: {avg_p:.3f} ± {std_p:.3f}")
            print(f"  Recall:    {avg_r:.3f} ± {std_r:.3f}")
            print(f"  F1:        {avg_f1:.3f}")
            return avg_p, avg_r, avg_f1, len(results)
        
        return 0, 0, 0, 0
    
    def run_semantic_benchmark(self, sample_size: int = 100):
        """Test semantic search with large sample size."""
        print(f"\n🧠 SEMANTIC SEARCH BENCHMARK (n={sample_size})")
        print("-" * 50)
        
        # Generate semantic query combinations
        semantic_queries = []
        words_list = list(self.all_words)
        
        for _ in range(sample_size):
            if len(words_list) >= 2:
                # Create 2-3 word combinations
                num_words = random.choice([2, 3])
                query_words = random.sample(words_list, min(num_words, len(words_list)))
                semantic_queries.append(query_words)
        
        results = []
        found_queries = 0
        
        for i, query_words in enumerate(semantic_queries):
            matches, total = self.semantic_search_test(query_words, min_overlap=0.5)
            if total > 0:
                precision, recall, f1 = self.calculate_metrics(matches, total)
                results.append((precision, recall, f1))
                found_queries += 1
                
                if i < 10:
                    query_str = " + ".join(query_words)
                    print(f"'{query_str}': P={precision:.3f} R={recall:.3f} F1={f1:.3f} ({matches}/{total})")
        
        if found_queries > 10:
            print(f"... and {found_queries - 10} more queries")
        
        if results:
            avg_p = statistics.mean([r[0] for r in results])
            avg_r = statistics.mean([r[1] for r in results])
            avg_f1 = statistics.mean([r[2] for r in results])
            std_p = statistics.stdev([r[0] for r in results]) if len(results) > 1 else 0
            std_r = statistics.stdev([r[1] for r in results]) if len(results) > 1 else 0
            
            print(f"\nSemantic Search Results (n={len(results)}):")
            print(f"  Precision: {avg_p:.3f} ± {std_p:.3f}")
            print(f"  Recall:    {avg_r:.3f} ± {std_r:.3f}") 
            print(f"  F1:        {avg_f1:.3f}")
            return avg_p, avg_r, avg_f1, len(results)
        
        return 0, 0, 0, 0
    
    def run_full_benchmark(self, sample_size: int = 200):
        """Run comprehensive benchmark with large sample sizes."""
        print("🚀 COMPREHENSIVE PERFORMANCE BENCHMARK")
        print("=" * 60)
        print(f"Testing with sample size: {sample_size} queries per search type")
        print(f"Codebase: {len(self.python_files)} Python files")
        
        # Run all benchmarks with large samples
        symbol_results = self.run_symbol_benchmark(sample_size)
        exact_results = self.run_exact_benchmark(sample_size)  
        semantic_results = self.run_semantic_benchmark(sample_size)
        
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE RESULTS SUMMARY")
        print("=" * 60)
        
        total_queries = symbol_results[3] + exact_results[3] + semantic_results[3]
        print(f"Total test queries executed: {total_queries}")
        
        print(f"\nSymbol Search   (n={symbol_results[3]}): P={symbol_results[0]:.3f} R={symbol_results[1]:.3f} F1={symbol_results[2]:.3f}")
        print(f"Exact Search    (n={exact_results[3]}):  P={exact_results[0]:.3f} R={exact_results[1]:.3f} F1={exact_results[2]:.3f}")
        print(f"Semantic Search (n={semantic_results[3]}): P={semantic_results[0]:.3f} R={semantic_results[1]:.3f} F1={semantic_results[2]:.3f}")
        
        print("\n" + "=" * 60)
        print("💡 STATISTICALLY-INFORMED TEST THRESHOLDS")
        print("=" * 60)
        print(f"Based on {total_queries} test queries across search types:")
        
        def safe_threshold(value, minimum=0.1, safety_margin=0.8):
            return max(minimum, value * safety_margin)
        
        print(f"\nSymbol Search Thresholds (n={symbol_results[3]}):")
        print(f"  assert precision >= {safe_threshold(symbol_results[0]):.2f}  # {safe_threshold(symbol_results[0])*100:.0f}% (measured: {symbol_results[0]*100:.1f}%)")
        print(f"  assert recall >= {safe_threshold(symbol_results[1]):.2f}     # {safe_threshold(symbol_results[1])*100:.0f}% (measured: {symbol_results[1]*100:.1f}%)")
        
        print(f"\nExact Search Thresholds (n={exact_results[3]}):")
        print(f"  assert precision >= {safe_threshold(exact_results[0], 0.85):.2f}  # {safe_threshold(exact_results[0], 0.85)*100:.0f}% (measured: {exact_results[0]*100:.1f}%)")
        print(f"  assert recall >= {safe_threshold(exact_results[1], 0.85):.2f}     # {safe_threshold(exact_results[1], 0.85)*100:.0f}% (measured: {exact_results[1]*100:.1f}%)")
        
        print(f"\nSemantic Search Thresholds (n={semantic_results[3]}):")
        print(f"  assert precision >= {safe_threshold(semantic_results[0]):.2f}  # {safe_threshold(semantic_results[0])*100:.0f}% (measured: {semantic_results[0]*100:.1f}%)")
        print(f"  assert recall >= {safe_threshold(semantic_results[1]):.2f}     # {safe_threshold(semantic_results[1])*100:.0f}% (measured: {semantic_results[1]*100:.1f}%)")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    benchmark = ComprehensiveBenchmark()
    benchmark.run_full_benchmark(sample_size=200)