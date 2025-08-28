#!/usr/bin/env python3
"""
Test runner script for workspace-qdrant-mcp.

Provides convenient commands to run different test categories and generate reports.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error running command: {e}")
        return 1


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for workspace-qdrant-mcp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                 # Run all tests
  python run_tests.py --unit                # Run unit tests only
  python run_tests.py --integration         # Run integration tests
  python run_tests.py --e2e                 # Run end-to-end tests
  python run_tests.py --coverage            # Run with coverage report
  python run_tests.py --fast                # Run fast tests only
  python run_tests.py --benchmark           # Run performance benchmarks
  python run_tests.py --lint                # Run linting only
  python run_tests.py --ci                  # Run CI pipeline locally
        """
    )
    
    # Test category options
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests only")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only (exclude slow tests)")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    
    # Report options
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--html-coverage", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--no-cov", action="store_true", help="Disable coverage collection")
    
    # Quality checks
    parser.add_argument("--lint", action="store_true", help="Run linting (ruff + black check)")
    parser.add_argument("--format", action="store_true", help="Format code with black")
    parser.add_argument("--type-check", action="store_true", help="Run mypy type checking")
    
    # CI/CD options
    parser.add_argument("--ci", action="store_true", help="Run full CI pipeline")
    parser.add_argument("--pre-commit", action="store_true", help="Run pre-commit checks")
    
    # Pytest options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-x", "--stop-on-first-failure", action="store_true", help="Stop on first failure")
    parser.add_argument("--pdb", action="store_true", help="Drop into debugger on failure")
    parser.add_argument("--parallel", "-n", type=int, help="Run tests in parallel (number of workers)")
    
    # File/pattern options
    parser.add_argument("files", nargs="*", help="Specific test files or patterns to run")
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    if not (project_root / "pyproject.toml").exists():
        print("Error: Must run from project root directory")
        return 1
    
    exit_codes = []
    
    # Handle formatting
    if args.format:
        cmd = ["black", "src/", "tests/", "run_tests.py"]
        exit_codes.append(run_command(cmd, "Formatting code with black"))
    
    # Handle linting
    if args.lint:
        # Run ruff
        cmd = ["ruff", "check", "src/", "tests/"]
        exit_codes.append(run_command(cmd, "Linting with ruff"))
        
        # Check black formatting
        cmd = ["black", "--check", "src/", "tests/", "run_tests.py"]
        exit_codes.append(run_command(cmd, "Checking code formatting"))
    
    # Handle type checking
    if args.type_check:
        cmd = ["mypy", "src/"]
        exit_codes.append(run_command(cmd, "Type checking with mypy"))
    
    # Handle pre-commit checks
    if args.pre_commit:
        for check in ["--format", "--lint", "--type-check", "--unit", "--coverage"]:
            sub_args = argparse.Namespace(**{check.lstrip('--').replace('-', '_'): True})
            # Recursively call with individual checks
            cmd = [sys.executable, __file__, check]
            exit_codes.append(run_command(cmd, f"Pre-commit check: {check}"))
        
        if any(exit_codes):
            print("\n❌ Pre-commit checks failed")
            return max(exit_codes)
        else:
            print("\n✅ All pre-commit checks passed")
            return 0
    
    # Handle CI pipeline
    if args.ci:
        ci_commands = [
            (["black", "--check", "src/", "tests/"], "Code formatting check"),
            (["ruff", "check", "src/", "tests/"], "Linting check"),
            (["mypy", "src/"], "Type checking"),
            (["pytest", "--cov", "--cov-fail-under=80", "-m", "not slow"], "Fast tests with coverage"),
        ]
        
        for cmd, desc in ci_commands:
            exit_code = run_command(cmd, desc)
            exit_codes.append(exit_code)
            if exit_code != 0:
                print(f"\n❌ CI pipeline failed at: {desc}")
                return exit_code
        
        print("\n✅ CI pipeline completed successfully")
        return 0
    
    # Build pytest command
    cmd = ["pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add stop on first failure
    if args.stop_on_first_failure:
        cmd.append("-x")
    
    # Add debugger
    if args.pdb:
        cmd.append("--pdb")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Add coverage options
    if not args.no_cov and (args.coverage or args.html_coverage or args.all or args.ci):
        cmd.append("--cov")
        
        if args.html_coverage:
            cmd.append("--cov-report=html")
        
        if args.coverage or args.all:
            cmd.append("--cov-report=term-missing")
    
    # Add test category markers
    test_markers = []
    
    if args.unit:
        test_markers.append("unit")
    elif args.integration:
        test_markers.append("integration")
    elif args.e2e:
        test_markers.append("e2e")
    elif args.fast:
        test_markers.append("not slow")
    elif args.benchmark:
        test_markers.append("benchmark")
    
    if test_markers:
        cmd.extend(["-m", " and ".join(test_markers)])
    
    # Add specific files if provided
    if args.files:
        cmd.extend(args.files)
    
    # Default to all tests if no specific category chosen
    if not any([args.unit, args.integration, args.e2e, args.fast, args.benchmark, args.files]):
        # Run all tests by default
        pass
    
    # Run tests if any test-related arguments were provided
    if any([args.all, args.unit, args.integration, args.e2e, args.fast, args.benchmark, args.coverage, args.files]):
        description = "Running tests"
        if args.unit:
            description = "Running unit tests"
        elif args.integration:
            description = "Running integration tests"
        elif args.e2e:
            description = "Running end-to-end tests"
        elif args.fast:
            description = "Running fast tests"
        elif args.benchmark:
            description = "Running performance benchmarks"
        
        exit_codes.append(run_command(cmd, description))
        
        # Show coverage report location if HTML was generated
        if args.html_coverage:
            print(f"\n📊 HTML coverage report generated: {project_root}/htmlcov/index.html")
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    # Return the highest exit code
    return max(exit_codes) if exit_codes else 0


if __name__ == "__main__":
    sys.exit(main())
