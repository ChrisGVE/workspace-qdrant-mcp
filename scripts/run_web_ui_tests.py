#!/usr/bin/env python3
"""Test runner script for web UI functional testing.

This script coordinates the testing of web UI functionality including:
1. CLI command validation
2. Web UI build and development server setup
3. Playwright browser automation tests
4. Integration testing between components

Usage:
    python scripts/run_web_ui_tests.py [--dev-server] [--integration] [--all]
"""

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional


class WebUITestRunner:
    """Coordinate web UI testing workflow."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.web_ui_path = project_root / "web-ui"
        self.test_results = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log test progress."""
        print(f"[{level}] {message}")
        
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, 
                   capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        self.log(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=capture_output,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out: {' '.join(cmd)}", "ERROR")
            raise
        except Exception as e:
            self.log(f"Command failed: {e}", "ERROR")
            raise
    
    def check_prerequisites(self) -> bool:
        """Check that all prerequisites are available."""
        self.log("Checking prerequisites...")
        
        # Check Python dependencies
        try:
            import pytest
            import playwright
            self.log("✓ Python test dependencies available")
        except ImportError as e:
            self.log(f"✗ Missing Python dependency: {e}", "ERROR")
            return False
        
        # Check web UI directory
        if not self.web_ui_path.exists():
            self.log(f"✗ Web UI directory not found: {self.web_ui_path}", "ERROR")
            self.log("Run: git submodule update --init --recursive", "INFO")
            return False
        
        self.log(f"✓ Web UI directory found: {self.web_ui_path}")
        
        # Check Node.js and npm
        try:
            result = self.run_command(["node", "--version"])
            if result.returncode == 0:
                self.log(f"✓ Node.js available: {result.stdout.strip()}")
            else:
                raise subprocess.CalledProcessError(result.returncode, "node")
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log("✗ Node.js not available", "ERROR")
            self.log("Install from: https://nodejs.org/", "INFO")
            return False
        
        try:
            result = self.run_command(["npm", "--version"])
            if result.returncode == 0:
                self.log(f"✓ npm available: {result.stdout.strip()}")
            else:
                raise subprocess.CalledProcessError(result.returncode, "npm")
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log("✗ npm not available", "ERROR")
            return False
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install Python and Node.js dependencies."""
        self.log("Installing dependencies...")
        
        # Install Python dependencies
        try:
            result = self.run_command([
                sys.executable, "-m", "pip", "install", "-e", ".[dev]"
            ])
            if result.returncode != 0:
                self.log("Failed to install Python dependencies", "ERROR")
                self.log(result.stderr, "ERROR")
                return False
            self.log("✓ Python dependencies installed")
        except Exception as e:
            self.log(f"Failed to install Python dependencies: {e}", "ERROR")
            return False
        
        # Install Playwright browsers
        try:
            result = self.run_command([sys.executable, "-m", "playwright", "install", "chromium"])
            if result.returncode != 0:
                self.log("Failed to install Playwright browsers", "ERROR")
                return False
            self.log("✓ Playwright browsers installed")
        except Exception as e:
            self.log(f"Failed to install Playwright browsers: {e}", "ERROR")
            return False
        
        # Install Node.js dependencies
        if (self.web_ui_path / "package.json").exists():
            try:
                result = self.run_command(["npm", "install"], cwd=self.web_ui_path)
                if result.returncode != 0:
                    self.log("Failed to install Node.js dependencies", "ERROR")
                    self.log(result.stderr, "ERROR")
                    return False
                self.log("✓ Node.js dependencies installed")
            except Exception as e:
                self.log(f"Failed to install Node.js dependencies: {e}", "ERROR")
                return False
        
        return True
    
    def run_cli_tests(self) -> bool:
        """Run CLI command tests."""
        self.log("Running CLI command tests...")
        
        try:
            result = self.run_command([
                sys.executable, "-m", "pytest",
                "tests/integration/test_web_commands_integration.py",
                "-v", "--tb=short"
            ])
            
            success = result.returncode == 0
            self.test_results.append(("CLI Tests", success, result.stdout, result.stderr))
            
            if success:
                self.log("✓ CLI tests passed")
            else:
                self.log("✗ CLI tests failed", "ERROR")
                self.log(result.stdout, "DEBUG")
                self.log(result.stderr, "DEBUG")
            
            return success
            
        except Exception as e:
            self.log(f"Failed to run CLI tests: {e}", "ERROR")
            self.test_results.append(("CLI Tests", False, "", str(e)))
            return False
    
    def build_web_ui(self) -> bool:
        """Build web UI for testing."""
        self.log("Building web UI...")
        
        if not (self.web_ui_path / "package.json").exists():
            self.log("No package.json found, skipping build", "WARN")
            return True
        
        try:
            result = self.run_command(["npm", "run", "build"], cwd=self.web_ui_path)
            if result.returncode != 0:
                self.log("Web UI build failed", "ERROR")
                self.log(result.stderr, "ERROR")
                return False
            
            self.log("✓ Web UI build completed")
            return True
            
        except Exception as e:
            self.log(f"Failed to build web UI: {e}", "ERROR")
            return False
    
    async def start_dev_server(self, port: int = 3000, timeout: int = 30) -> Optional[subprocess.Popen]:
        """Start development server for testing."""
        self.log(f"Starting development server on port {port}...")
        
        if not (self.web_ui_path / "package.json").exists():
            self.log("No package.json found, cannot start dev server", "ERROR")
            return None
        
        try:
            env = os.environ.copy()
            env["PORT"] = str(port)
            env["BROWSER"] = "none"  # Don't open browser
            
            process = subprocess.Popen(
                ["npm", "start"],
                cwd=self.web_ui_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to be ready
            import aiohttp
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"http://localhost:{port}", 
                                             timeout=aiohttp.ClientTimeout(total=2)) as response:
                            if response.status < 400:
                                self.log(f"✓ Development server ready at http://localhost:{port}")
                                return process
                except:
                    pass
                
                # Check if process failed
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    self.log("Development server failed to start", "ERROR")
                    self.log(stdout, "DEBUG")
                    self.log(stderr, "DEBUG")
                    return None
                
                await asyncio.sleep(1)
            
            self.log("Development server startup timeout", "ERROR")
            process.terminate()
            return None
            
        except Exception as e:
            self.log(f"Failed to start development server: {e}", "ERROR")
            return None
    
    def run_playwright_tests(self, dev_server_running: bool = False) -> bool:
        """Run Playwright browser automation tests."""
        self.log("Running Playwright tests...")
        
        test_cmd = [
            sys.executable, "-m", "pytest",
            "tests/playwright/test_web_ui_functionality.py",
            "-v", "--tb=short"
        ]
        
        if not dev_server_running:
            test_cmd.extend(["-m", "not requires_dev_server"])
        
        try:
            result = self.run_command(test_cmd)
            
            success = result.returncode == 0
            self.test_results.append(("Playwright Tests", success, result.stdout, result.stderr))
            
            if success:
                self.log("✓ Playwright tests passed")
            else:
                self.log("✗ Playwright tests failed", "ERROR")
                self.log(result.stdout, "DEBUG")
                self.log(result.stderr, "DEBUG")
            
            return success
            
        except Exception as e:
            self.log(f"Failed to run Playwright tests: {e}", "ERROR")
            self.test_results.append(("Playwright Tests", False, "", str(e)))
            return False
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        self.log("Running integration tests...")
        
        try:
            result = self.run_command([
                sys.executable, "-m", "pytest",
                "tests/integration/test_web_ui_integration.py",
                "-v", "--tb=short"
            ])
            
            success = result.returncode == 0
            self.test_results.append(("Integration Tests", success, result.stdout, result.stderr))
            
            if success:
                self.log("✓ Integration tests passed")
            else:
                self.log("✗ Integration tests failed", "ERROR")
                self.log(result.stdout, "DEBUG")
                self.log(result.stderr, "DEBUG")
            
            return success
            
        except Exception as e:
            self.log(f"Failed to run integration tests: {e}", "ERROR")
            self.test_results.append(("Integration Tests", False, "", str(e)))
            return False
    
    def print_summary(self):
        """Print test results summary."""
        self.log("=" * 60)
        self.log("TEST RESULTS SUMMARY")
        self.log("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, success, _, _ in self.test_results if success)
        
        for test_name, success, stdout, stderr in self.test_results:
            status = "✓ PASS" if success else "✗ FAIL"
            self.log(f"{test_name}: {status}")
        
        self.log(f"\nOverall: {passed_tests}/{total_tests} test suites passed")
        
        if passed_tests == total_tests:
            self.log("🎉 All tests passed!", "SUCCESS")
            return True
        else:
            self.log("❌ Some tests failed", "ERROR")
            return False


async def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run web UI functional tests")
    parser.add_argument("--cli-only", action="store_true", help="Run only CLI tests")
    parser.add_argument("--ui-only", action="store_true", help="Run only UI tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--dev-server", action="store_true", help="Start dev server for tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--no-install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Default to all tests if no specific test type selected
    if not any([args.cli_only, args.ui_only, args.integration]):
        args.all = True
    
    project_root = Path(__file__).parent.parent
    runner = WebUITestRunner(project_root)
    
    try:
        # Check prerequisites
        if not runner.check_prerequisites():
            runner.log("Prerequisites check failed", "ERROR")
            return 1
        
        # Install dependencies
        if not args.no_install:
            if not runner.install_dependencies():
                runner.log("Dependency installation failed", "ERROR")
                return 1
        
        # Run selected tests
        success = True
        dev_server = None
        
        try:
            # Start development server if needed
            if args.dev_server or args.all:
                dev_server = await runner.start_dev_server()
                if dev_server is None:
                    runner.log("Failed to start development server", "WARN")
                    dev_server_running = False
                else:
                    dev_server_running = True
            else:
                dev_server_running = False
            
            # Run CLI tests
            if args.cli_only or args.all:
                success &= runner.run_cli_tests()
            
            # Build web UI
            if args.ui_only or args.dev_server or args.all:
                success &= runner.build_web_ui()
            
            # Run Playwright tests
            if args.ui_only or args.all:
                success &= runner.run_playwright_tests(dev_server_running)
            
            # Run integration tests
            if args.integration or args.all:
                success &= runner.run_integration_tests()
            
        finally:
            # Clean up development server
            if dev_server:
                runner.log("Stopping development server...")
                dev_server.terminate()
                try:
                    dev_server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    dev_server.kill()
        
        # Print summary
        runner.print_summary()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        runner.log("Tests interrupted by user", "WARN")
        return 130
    except Exception as e:
        runner.log(f"Unexpected error: {e}", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))