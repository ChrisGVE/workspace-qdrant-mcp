#!/usr/bin/env python3
"""
🚨 CRITICAL: Blocking Issues Resolver for 100% Coverage Mission

This script identifies and provides actionable solutions for issues preventing 100% coverage achievement.
Focus: Fix the 58 Python test collection errors and set up Rust coverage infrastructure.
"""

import subprocess
import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockingIssuesResolver:
    """Identify and resolve critical blocking issues for coverage achievement"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.issues_found = []
        self.solutions_available = []

    def analyze_python_test_errors(self) -> List[Dict]:
        """Analyze the 58 Python test collection errors"""
        logger.info("🔍 Analyzing Python test collection errors...")

        try:
            # Run pytest with detailed error output
            cmd = ["uv", "run", "pytest", "--collect-only", "-v"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            output = result.stdout + result.stderr
            errors = []

            current_error = None
            error_details = []

            for line in output.split('\n'):
                if "ERROR" in line and any(x in line for x in ["test_", "tests/"]):
                    if current_error:
                        errors.append({
                            'file': current_error,
                            'details': '\n'.join(error_details),
                            'type': self._classify_error('\n'.join(error_details))
                        })
                    current_error = line.split("ERROR ")[-1]
                    error_details = []
                elif current_error and line.strip():
                    error_details.append(line)

            # Add the last error
            if current_error:
                errors.append({
                    'file': current_error,
                    'details': '\n'.join(error_details),
                    'type': self._classify_error('\n'.join(error_details))
                })

            logger.info(f"📊 Found {len(errors)} test collection errors")
            return errors

        except Exception as e:
            logger.error(f"💥 Failed to analyze test errors: {e}")
            return []

    def _classify_error(self, error_details: str) -> str:
        """Classify the type of error for targeted solutions"""
        error_details_lower = error_details.lower()

        if "modulenotfounderror" in error_details_lower or "no module named" in error_details_lower:
            return "IMPORT_ERROR"
        elif "syntaxerror" in error_details_lower:
            return "SYNTAX_ERROR"
        elif "attributeerror" in error_details_lower:
            return "ATTRIBUTE_ERROR"
        elif "cannot import name" in error_details_lower:
            return "IMPORT_NAME_ERROR"
        elif "indentationerror" in error_details_lower:
            return "INDENTATION_ERROR"
        elif "__init__" in error_details_lower and "constructor" in error_details_lower:
            return "CONSTRUCTOR_ERROR"
        else:
            return "OTHER_ERROR"

    def analyze_rust_infrastructure(self) -> Dict:
        """Analyze Rust testing and coverage infrastructure"""
        logger.info("🔍 Analyzing Rust infrastructure...")

        rust_dir = self.project_root / "rust-engine"
        issues = {}

        if not rust_dir.exists():
            issues['rust_directory'] = "Rust engine directory not found"
            return issues

        # Check if Cargo.toml exists
        cargo_toml = rust_dir / "Cargo.toml"
        if not cargo_toml.exists():
            issues['cargo_toml'] = "Cargo.toml not found in rust-engine"

        # Check for tarpaulin
        try:
            result = subprocess.run(
                ["cargo", "tarpaulin", "--version"],
                capture_output=True, text=True,
                cwd=rust_dir
            )
            if result.returncode != 0:
                issues['tarpaulin'] = "cargo-tarpaulin not installed for coverage"
        except FileNotFoundError:
            issues['cargo'] = "Cargo not found - Rust toolchain may not be installed"

        # Check if tests exist
        src_dir = rust_dir / "src"
        test_files = list(src_dir.rglob("*test*.rs")) if src_dir.exists() else []
        tests_dir = rust_dir / "tests"
        test_dir_files = list(tests_dir.rglob("*.rs")) if tests_dir.exists() else []

        if not test_files and not test_dir_files:
            issues['no_tests'] = "No Rust test files found"

        return issues

    def generate_python_solutions(self, errors: List[Dict]) -> List[str]:
        """Generate specific solutions for Python errors"""
        solutions = []

        # Group errors by type
        error_types = {}
        for error in errors:
            error_type = error['type']
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)

        for error_type, error_list in error_types.items():
            count = len(error_list)

            if error_type == "IMPORT_ERROR":
                solutions.append(f"""
🔧 IMPORT ERRORS ({count} files):
   Problem: Missing module imports or incorrect Python paths
   Solution:
   1. Add missing __init__.py files
   2. Fix PYTHONPATH configuration
   3. Update import statements to use absolute imports
   4. Run: find src -type d -exec touch {{}}/__init__.py \\;
   Files affected: {[e['file'] for e in error_list[:5]]}
""")

            elif error_type == "SYNTAX_ERROR":
                solutions.append(f"""
🔧 SYNTAX ERRORS ({count} files):
   Problem: Invalid Python syntax preventing test loading
   Solution:
   1. Run syntax checker: python -m py_compile <file>
   2. Fix syntax issues (missing colons, parentheses, etc.)
   3. Use black formatter: uv run black src/ tests/
   Files affected: {[e['file'] for e in error_list[:3]]}
""")

            elif error_type == "CONSTRUCTOR_ERROR":
                solutions.append(f"""
🔧 CONSTRUCTOR ERRORS ({count} files):
   Problem: Test classes with __init__ methods confuse pytest
   Solution:
   1. Remove __init__ from test classes
   2. Use @dataclass instead of classes with constructors
   3. Rename classes that aren't actually test classes
   Files affected: {[e['file'] for e in error_list[:3]]}
""")

        return solutions

    def generate_rust_solutions(self, issues: Dict) -> List[str]:
        """Generate specific solutions for Rust infrastructure issues"""
        solutions = []

        if 'tarpaulin' in issues:
            solutions.append("""
🔧 RUST COVERAGE SETUP:
   Problem: cargo-tarpaulin not installed
   Solution:
   1. Install tarpaulin: cargo install cargo-tarpaulin
   2. Alternative: Use cargo-llvm-cov: cargo install cargo-llvm-cov
   3. For CI: Add to Cargo.toml [dev-dependencies]: tarpaulin = "0.27"
""")

        if 'no_tests' in issues:
            solutions.append("""
🔧 RUST TEST INFRASTRUCTURE:
   Problem: No Rust tests found
   Solution:
   1. Create test files in rust-engine/tests/
   2. Add unit tests with #[cfg(test)] modules
   3. Example: Create integration_tests.rs with basic coverage
""")

        if 'cargo' in issues:
            solutions.append("""
🔧 RUST TOOLCHAIN:
   Problem: Rust/Cargo not properly installed
   Solution:
   1. Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   2. Update PATH: source ~/.cargo/env
   3. Verify: cargo --version
""")

        return solutions

    def create_immediate_action_plan(self) -> str:
        """Create an immediate action plan to unblock coverage progress"""
        python_errors = self.analyze_python_test_errors()
        rust_issues = self.analyze_rust_infrastructure()

        python_solutions = self.generate_python_solutions(python_errors)
        rust_solutions = self.generate_rust_solutions(rust_issues)

        action_plan = f"""
🚨 CRITICAL EMERGENCY ACTION PLAN 🚨
⏰ Generated: {Path().cwd().name} at {os.popen('date').read().strip()}

🎯 MISSION: Unblock path to 100% coverage achievement

📊 CURRENT BLOCKING ISSUES SUMMARY:
   Python Test Errors: {len(python_errors)} collection failures
   Rust Infrastructure: {len(rust_issues)} setup issues

🔥 IMMEDIATE ACTIONS REQUIRED (Priority Order):

PHASE 1 - PYTHON UNBLOCKING (Next 30 minutes):
{chr(10).join(python_solutions)}

PHASE 2 - RUST INFRASTRUCTURE (Next 30 minutes):
{chr(10).join(rust_solutions)}

PHASE 3 - VERIFICATION (Next 15 minutes):
🔧 VERIFICATION STEPS:
   1. Python: uv run pytest --collect-only (should show 0 errors)
   2. Rust: cd rust-engine && cargo test (should run tests)
   3. Coverage: uv run pytest --cov=src --cov-report=term
   4. Rust Coverage: cd rust-engine && cargo tarpaulin --all

🚨 SUCCESS CRITERIA:
   ✅ Python tests collect without errors
   ✅ Rust tests execute successfully
   ✅ Coverage reports generate for both languages
   ✅ Path clear for 100% coverage achievement

⏰ ESTIMATED TIME TO UNBLOCK: 75 minutes
🎯 EXPECTED COVERAGE JUMP: 15-25% increase once unblocked

📞 NEXT STEPS AFTER UNBLOCKING:
   1. Run enhanced monitoring: python 20250922-0926_CRITICAL_100_PERCENT_MONITOR.py
   2. Focus on highest-impact coverage areas
   3. Target 100% achievement within 24-48 hours

================================================================================
🚨 CRITICAL: Execute this plan immediately to achieve 100% coverage mission! 🚨
"""

        return action_plan

    def run_analysis(self):
        """Run complete blocking issues analysis"""
        logger.info("🚀 Starting CRITICAL blocking issues analysis...")

        action_plan = self.create_immediate_action_plan()

        # Write action plan to file
        plan_file = "20250922-0926_CRITICAL_ACTION_PLAN.md"
        with open(plan_file, "w") as f:
            f.write(action_plan)

        print(action_plan)
        logger.info(f"📝 Action plan saved to: {plan_file}")
        logger.info("🎯 Execute the plan to unblock 100% coverage achievement!")

if __name__ == "__main__":
    resolver = BlockingIssuesResolver()
    resolver.run_analysis()