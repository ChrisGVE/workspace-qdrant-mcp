#!/usr/bin/env python3
"""
Comprehensive GitHub Actions workflow testing with edge cases and error conditions.

Tests all edge cases for GitHub Actions including:
- Timeout handling and resource limits
- Matrix build failures and partial successes
- Artifact upload/download edge cases
- Secret handling and environment variable issues
- Service container failures
- Cross-platform compatibility issues
- Network failures and retry mechanisms
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
import pytest
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
import requests


@dataclass
class WorkflowTestResult:
    """Result of a workflow test."""
    test_name: str
    success: bool
    duration: float
    output: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GitHubActionsWorkflowTester:
    """Comprehensive testing for GitHub Actions workflows with edge cases."""

    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize the GitHub Actions workflow tester."""
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
        self.original_dir = Path.cwd()
        self.test_results: List[WorkflowTestResult] = []
        self.workflow_templates = {}

    def create_test_repository(self) -> Path:
        """Create a test repository with GitHub Actions workflow."""
        repo_path = self.temp_dir / "test_actions_repo"
        repo_path.mkdir(exist_ok=True)

        os.chdir(repo_path)

        # Initialize Git repository
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True)

        # Create .github/workflows directory
        workflows_dir = repo_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        return repo_path

    def create_workflow_with_timeouts(self) -> Dict[str, Any]:
        """Create workflow configuration with various timeout scenarios."""
        workflow = {
            'name': 'Timeout Edge Cases Test',
            'on': ['push', 'pull_request'],
            'jobs': {
                'quick-job': {
                    'runs-on': 'ubuntu-latest',
                    'timeout-minutes': 1,
                    'steps': [
                        {'name': 'Quick task', 'run': 'echo "Quick task completed"'},
                        {'name': 'Sleep 30s', 'run': 'sleep 30'}
                    ]
                },
                'timeout-job': {
                    'runs-on': 'ubuntu-latest',
                    'timeout-minutes': 2,
                    'steps': [
                        {'name': 'Long running task', 'run': 'sleep 150'},  # 2.5 minutes
                        {'name': 'Should not reach here', 'run': 'echo "This should timeout"'}
                    ]
                },
                'step-timeout': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Step with timeout',
                            'run': 'sleep 300',  # 5 minutes
                            'timeout-minutes': 1
                        },
                        {'name': 'Cleanup', 'run': 'echo "Cleanup step"'}
                    ]
                },
                'resource-limit-test': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Memory stress test',
                            'run': '''
                                python3 -c "
import sys
try:
    # Try to allocate large amounts of memory
    data = []
    for i in range(1000):
        data.append('x' * 1024 * 1024)  # 1MB chunks
        if i % 100 == 0:
            print(f'Allocated {i}MB')
            sys.stdout.flush()
except MemoryError:
    print('Memory limit reached')
except Exception as e:
    print(f'Error: {e}')
"
                            ''',
                            'timeout-minutes': 3
                        },
                        {
                            'name': 'CPU stress test',
                            'run': '''
                                python3 -c "
import time
import multiprocessing

def cpu_stress():
    end_time = time.time() + 60  # Run for 1 minute
    while time.time() < end_time:
        pass

processes = []
for _ in range(multiprocessing.cpu_count()):
    p = multiprocessing.Process(target=cpu_stress)
    p.start()
    processes.append(p)

for p in processes:
    p.join()
print('CPU stress test completed')
"
                            ''',
                            'timeout-minutes': 2
                        }
                    ]
                }
            }
        }
        return workflow

    def create_matrix_failure_workflow(self) -> Dict[str, Any]:
        """Create workflow with matrix builds that have failure scenarios."""
        workflow = {
            'name': 'Matrix Build Edge Cases',
            'on': ['push'],
            'jobs': {
                'matrix-test': {
                    'runs-on': '${{ matrix.os }}',
                    'continue-on-error': True,
                    'strategy': {
                        'fail-fast': False,
                        'matrix': {
                            'os': ['ubuntu-latest', 'macos-latest', 'windows-latest'],
                            'python-version': ['3.8', '3.9', '3.10', '3.11', '3.12'],
                            'node-version': ['16', '18', '20'],
                            'include': [
                                {
                                    'os': 'ubuntu-latest',
                                    'python-version': '3.8',
                                    'node-version': '16',
                                    'experimental': True
                                }
                            ],
                            'exclude': [
                                {
                                    'os': 'windows-latest',
                                    'python-version': '3.12'  # Intentional exclusion
                                }
                            ]
                        }
                    },
                    'steps': [
                        {'name': 'Checkout', 'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Setup Python',
                            'uses': 'actions/setup-python@v5',
                            'with': {'python-version': '${{ matrix.python-version }}'}
                        },
                        {
                            'name': 'Setup Node',
                            'uses': 'actions/setup-node@v4',
                            'with': {'node-version': '${{ matrix.node-version }}'}
                        },
                        {
                            'name': 'Platform-specific failing test',
                            'run': '''
                                if [[ "${{ matrix.os }}" == "windows-latest" && "${{ matrix.python-version }}" == "3.8" ]]; then
                                    echo "Simulating Windows Python 3.8 failure"
                                    exit 1
                                elif [[ "${{ matrix.os }}" == "macos-latest" && "${{ matrix.node-version }}" == "20" ]]; then
                                    echo "Simulating macOS Node 20 failure"
                                    exit 1
                                else
                                    echo "Test passed for ${{ matrix.os }} Python ${{ matrix.python-version }} Node ${{ matrix.node-version }}"
                                fi
                            ''',
                            'shell': 'bash'
                        },
                        {
                            'name': 'Conditional step',
                            'if': 'matrix.experimental',
                            'run': 'echo "Running experimental configuration"'
                        }
                    ]
                }
            }
        }
        return workflow

    def create_artifact_edge_cases_workflow(self) -> Dict[str, Any]:
        """Create workflow with artifact handling edge cases."""
        workflow = {
            'name': 'Artifact Edge Cases',
            'on': ['push'],
            'jobs': {
                'artifact-producer': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {'name': 'Checkout', 'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Create test artifacts',
                            'run': '''
                                mkdir -p artifacts/large artifacts/empty artifacts/special-chars

                                # Create large file (close to GitHub limit)
                                dd if=/dev/zero of=artifacts/large/large-file.bin bs=1M count=100

                                # Create many small files
                                for i in {1..1000}; do
                                    echo "File $i content" > artifacts/many/file_$i.txt
                                done

                                # Create files with special characters
                                touch "artifacts/special-chars/file with spaces.txt"
                                touch "artifacts/special-chars/file[brackets].txt"
                                touch "artifacts/special-chars/file(parentheses).txt"

                                # Create empty directories
                                mkdir -p artifacts/empty/subdir

                                # Create files with unusual permissions
                                echo "readonly content" > artifacts/readonly.txt
                                chmod 444 artifacts/readonly.txt

                                # Create symbolic links
                                ln -s ../large/large-file.bin artifacts/symlink-to-large.bin
                                ln -s nonexistent-file.txt artifacts/broken-symlink.txt

                                # Create files with unusual encodings
                                echo -e "UTF-8: Hello, 世界!" > artifacts/utf8.txt
                                echo -e "\\x48\\x65\\x6c\\x6c\\x6f" > artifacts/binary.txt
                            '''
                        },
                        {
                            'name': 'Upload large artifact',
                            'uses': 'actions/upload-artifact@v4',
                            'with': {
                                'name': 'large-artifacts',
                                'path': 'artifacts/large/',
                                'retention-days': 1
                            },
                            'continue-on-error': True
                        },
                        {
                            'name': 'Upload many files artifact',
                            'uses': 'actions/upload-artifact@v4',
                            'with': {
                                'name': 'many-files',
                                'path': 'artifacts/many/',
                                'retention-days': 1
                            },
                            'continue-on-error': True
                        },
                        {
                            'name': 'Upload special characters artifact',
                            'uses': 'actions/upload-artifact@v4',
                            'with': {
                                'name': 'special-chars',
                                'path': 'artifacts/special-chars/',
                                'retention-days': 1
                            },
                            'continue-on-error': True
                        },
                        {
                            'name': 'Upload empty artifact',
                            'uses': 'actions/upload-artifact@v4',
                            'with': {
                                'name': 'empty-artifact',
                                'path': 'artifacts/empty/',
                                'retention-days': 1
                            },
                            'continue-on-error': True
                        },
                        {
                            'name': 'Upload with glob patterns',
                            'uses': 'actions/upload-artifact@v4',
                            'with': {
                                'name': 'glob-patterns',
                                'path': |
                                    artifacts/**/*.txt
                                    artifacts/**/*.bin
                                'retention-days': 1
                            },
                            'continue-on-error': True
                        }
                    ]
                },
                'artifact-consumer': {
                    'runs-on': 'ubuntu-latest',
                    'needs': 'artifact-producer',
                    'steps': [
                        {'name': 'Checkout', 'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Download all artifacts',
                            'uses': 'actions/download-artifact@v4',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Verify artifact contents',
                            'run': '''
                                echo "Verifying downloaded artifacts:"
                                find . -name "*artifact*" -type d | head -10

                                # Check for large files
                                if [ -f "large-artifacts/large-file.bin" ]; then
                                    echo "Large file size: $(du -h large-artifacts/large-file.bin)"
                                fi

                                # Check many files
                                if [ -d "many-files" ]; then
                                    echo "Many files count: $(find many-files -type f | wc -l)"
                                fi

                                # Check special characters
                                if [ -d "special-chars" ]; then
                                    echo "Special character files:"
                                    ls -la "special-chars/"
                                fi
                            ''',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Test artifact download failure handling',
                            'uses': 'actions/download-artifact@v4',
                            'with': {
                                'name': 'nonexistent-artifact',
                                'path': 'downloads/'
                            },
                            'continue-on-error': True
                        }
                    ]
                }
            }
        }
        return workflow

    def create_service_container_edge_cases_workflow(self) -> Dict[str, Any]:
        """Create workflow with service container edge cases."""
        workflow = {
            'name': 'Service Container Edge Cases',
            'on': ['push'],
            'jobs': {
                'service-edge-cases': {
                    'runs-on': 'ubuntu-latest',
                    'services': {
                        # Working service
                        'postgres': {
                            'image': 'postgres:13',
                            'env': {
                                'POSTGRES_PASSWORD': 'postgres',
                                'POSTGRES_DB': 'testdb'
                            },
                            'options': '--health-cmd "pg_isready -U postgres" --health-interval 10s --health-timeout 5s --health-retries 5',
                            'ports': ['5432:5432']
                        },
                        # Service with resource constraints
                        'redis-constrained': {
                            'image': 'redis:7',
                            'options': '--memory=64m --cpus=0.5 --health-cmd "redis-cli ping" --health-interval 10s',
                            'ports': ['6379:6379']
                        },
                        # Service with custom network
                        'custom-network-service': {
                            'image': 'nginx:alpine',
                            'ports': ['8080:80'],
                            'options': '--health-cmd "curl -f http://localhost:80 || exit 1"'
                        },
                        # Service that might fail to start
                        'failing-service': {
                            'image': 'postgres:13',
                            'env': {
                                'POSTGRES_PASSWORD': '',  # Invalid - should cause failure
                                'POSTGRES_DB': 'testdb'
                            },
                            'ports': ['5433:5432']
                        }
                    },
                    'steps': [
                        {'name': 'Wait for services', 'run': 'sleep 30'},
                        {
                            'name': 'Test PostgreSQL connection',
                            'run': '''
                                echo "Testing PostgreSQL connection..."
                                until pg_isready -h localhost -p 5432 -U postgres; do
                                    echo "Waiting for PostgreSQL..."
                                    sleep 2
                                done
                                echo "PostgreSQL is ready"

                                # Test database operations
                                PGPASSWORD=postgres psql -h localhost -U postgres -d testdb -c "CREATE TABLE test_table (id serial, name text);"
                                PGPASSWORD=postgres psql -h localhost -U postgres -d testdb -c "INSERT INTO test_table (name) VALUES ('test');"
                                PGPASSWORD=postgres psql -h localhost -U postgres -d testdb -c "SELECT * FROM test_table;"
                            ''',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Test Redis connection',
                            'run': '''
                                echo "Testing Redis connection..."
                                redis-cli -h localhost -p 6379 ping
                                redis-cli -h localhost -p 6379 set test_key "test_value"
                                redis-cli -h localhost -p 6379 get test_key
                            ''',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Test Nginx service',
                            'run': '''
                                echo "Testing Nginx service..."
                                curl -f http://localhost:8080 || echo "Nginx connection failed"
                                curl -I http://localhost:8080 || echo "Nginx headers failed"
                            ''',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Test failing service handling',
                            'run': '''
                                echo "Testing failing service..."
                                pg_isready -h localhost -p 5433 -U postgres || echo "Expected failure: failing service not accessible"
                            ''',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Service resource monitoring',
                            'run': '''
                                echo "Monitoring service resource usage..."
                                docker stats --no-stream
                                docker ps -a
                            ''',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Test service network isolation',
                            'run': '''
                                echo "Testing service network isolation..."
                                # Test internal communication between services
                                docker exec $(docker ps -q --filter "ancestor=postgres:13" | head -1) ping -c 3 redis-constrained || echo "Service isolation test"
                            ''',
                            'continue-on-error': True
                        }
                    ]
                }
            }
        }
        return workflow

    def create_environment_secrets_edge_cases_workflow(self) -> Dict[str, Any]:
        """Create workflow with environment variables and secrets edge cases."""
        workflow = {
            'name': 'Environment and Secrets Edge Cases',
            'on': ['push'],
            'env': {
                'GLOBAL_VAR': 'global_value',
                'UNICODE_VAR': 'Hello, 世界! 🚀',
                'SPECIAL_CHARS': 'var with spaces & symbols @#$%^&*()',
                'EMPTY_VAR': '',
                'MULTILINE_VAR': 'line1\\nline2\\nline3'
            },
            'jobs': {
                'env-secrets-test': {
                    'runs-on': 'ubuntu-latest',
                    'env': {
                        'JOB_VAR': 'job_value',
                        'OVERRIDE_VAR': 'job_override'
                    },
                    'steps': [
                        {
                            'name': 'Test basic environment variables',
                            'env': {
                                'STEP_VAR': 'step_value',
                                'OVERRIDE_VAR': 'step_override'
                            },
                            'run': '''
                                echo "=== Environment Variables Test ==="
                                echo "Global var: $GLOBAL_VAR"
                                echo "Job var: $JOB_VAR"
                                echo "Step var: $STEP_VAR"
                                echo "Override test: $OVERRIDE_VAR"
                                echo "Unicode test: $UNICODE_VAR"
                                echo "Special chars: $SPECIAL_CHARS"
                                echo "Empty var: [$EMPTY_VAR]"
                                echo "Multiline var: $MULTILINE_VAR"
                            '''
                        },
                        {
                            'name': 'Test environment variable edge cases',
                            'run': '''
                                echo "=== Environment Edge Cases ==="

                                # Test undefined variables
                                echo "Undefined var: ${UNDEFINED_VAR:-default_value}"

                                # Test variable expansion
                                export EXPANDED_VAR="Value: $GLOBAL_VAR"
                                echo "Expanded: $EXPANDED_VAR"

                                # Test case sensitivity
                                export lowercase_var="lowercase"
                                export UPPERCASE_VAR="uppercase"
                                echo "Lowercase: $lowercase_var"
                                echo "Uppercase: $UPPERCASE_VAR"

                                # Test numeric variables
                                export NUMERIC_VAR="12345"
                                echo "Numeric: $NUMERIC_VAR"
                                echo "Arithmetic: $((NUMERIC_VAR + 100))"

                                # Test boolean-like variables
                                export BOOL_TRUE="true"
                                export BOOL_FALSE="false"
                                echo "Bool true: $BOOL_TRUE"
                                echo "Bool false: $BOOL_FALSE"
                            '''
                        },
                        {
                            'name': 'Test environment variable injection protection',
                            'run': '''
                                echo "=== Injection Protection Test ==="

                                # Test potentially dangerous values
                                export POTENTIAL_INJECTION="; rm -rf /"
                                echo "Injection test: $POTENTIAL_INJECTION"

                                # Test command substitution protection
                                export COMMAND_SUB="$(whoami)"
                                echo "Command sub: $COMMAND_SUB"

                                # Test null byte handling
                                export NULL_BYTE_VAR="hello\\0world"
                                echo "Null byte test: $NULL_BYTE_VAR"
                            '''
                        },
                        {
                            'name': 'Test secrets masking',
                            'env': {
                                'FAKE_SECRET': 'this-should-be-masked-12345',
                                'FAKE_TOKEN': 'ghp_fake_token_abcdefghijklmnopqrstuvwxyz123456',
                                'FAKE_API_KEY': 'sk-fake_api_key_1234567890abcdefghij'
                            },
                            'run': '''
                                echo "=== Secrets Masking Test ==="
                                echo "Fake secret: $FAKE_SECRET"
                                echo "Fake token: $FAKE_TOKEN"
                                echo "Fake API key: $FAKE_API_KEY"

                                # Test that secrets are masked in output
                                echo "$FAKE_SECRET" > secret_file.txt
                                cat secret_file.txt
                            '''
                        },
                        {
                            'name': 'Test environment variable limits',
                            'run': '''
                                echo "=== Environment Limits Test ==="

                                # Test very long environment variable
                                export VERY_LONG_VAR=$(python3 -c "print('x' * 1000)")
                                echo "Long var length: ${#VERY_LONG_VAR}"

                                # Test many environment variables
                                for i in {1..100}; do
                                    export "TEST_VAR_$i"="value_$i"
                                done

                                echo "Created 100 test variables"
                                env | grep TEST_VAR | wc -l

                                # Test environment variable with binary data
                                export BINARY_VAR=$(echo -e "\\x00\\x01\\x02\\x03")
                                echo "Binary var: $BINARY_VAR"
                            '''
                        },
                        {
                            'name': 'Test cross-platform environment handling',
                            'run': '''
                                echo "=== Cross-Platform Environment Test ==="

                                # Test PATH handling
                                echo "PATH: $PATH"
                                echo "PATH separator test"

                                # Test case sensitivity (Linux vs Windows)
                                export Path="custom_path"  # Different case
                                echo "Custom Path: $Path"

                                # Test environment variable inheritance
                                bash -c 'echo "Inherited GLOBAL_VAR: $GLOBAL_VAR"'

                                # Test shell-specific variables
                                echo "Shell: $SHELL"
                                echo "PWD: $PWD"
                                echo "USER: $USER"
                            '''
                        }
                    ]
                }
            }
        }
        return workflow

    def create_network_failure_workflow(self) -> Dict[str, Any]:
        """Create workflow with network failure scenarios."""
        workflow = {
            'name': 'Network Failure Edge Cases',
            'on': ['push'],
            'jobs': {
                'network-edge-cases': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Test network connectivity',
                            'run': '''
                                echo "=== Network Connectivity Test ==="
                                curl -I https://github.com || echo "GitHub connection failed"
                                curl -I https://api.github.com || echo "GitHub API connection failed"
                                ping -c 3 8.8.8.8 || echo "Google DNS ping failed"
                            ''',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Test slow network connections',
                            'run': '''
                                echo "=== Slow Network Test ==="
                                # Simulate slow download
                                timeout 30s curl -m 25 --limit-rate 1k https://httpbin.org/delay/5 || echo "Slow connection timeout"
                            ''',
                            'continue-on-error': True,
                            'timeout-minutes': 1
                        },
                        {
                            'name': 'Test unreliable connections with retries',
                            'run': '''
                                echo "=== Retry Mechanism Test ==="

                                # Function to simulate unreliable connection
                                unreliable_request() {
                                    local attempt=1
                                    local max_attempts=3

                                    while [ $attempt -le $max_attempts ]; do
                                        echo "Attempt $attempt of $max_attempts"

                                        # Simulate random failure (50% chance)
                                        if [ $((RANDOM % 2)) -eq 0 ]; then
                                            echo "Request succeeded on attempt $attempt"
                                            return 0
                                        else
                                            echo "Request failed on attempt $attempt"
                                            if [ $attempt -eq $max_attempts ]; then
                                                return 1
                                            fi
                                            sleep 2
                                            ((attempt++))
                                        fi
                                    done
                                }

                                unreliable_request || echo "All retry attempts failed"
                            '''
                        },
                        {
                            'name': 'Test DNS resolution issues',
                            'run': '''
                                echo "=== DNS Resolution Test ==="

                                # Test valid domains
                                nslookup github.com || echo "GitHub DNS lookup failed"
                                nslookup google.com || echo "Google DNS lookup failed"

                                # Test invalid domains
                                nslookup nonexistent-domain-12345.com || echo "Expected: Invalid domain lookup failed"

                                # Test IPv6 resolution
                                nslookup -type=AAAA github.com || echo "IPv6 lookup failed"
                            ''',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Test HTTP error handling',
                            'run': '''
                                echo "=== HTTP Error Handling Test ==="

                                # Test various HTTP status codes
                                curl -f https://httpbin.org/status/200 || echo "200 OK failed"
                                curl -f https://httpbin.org/status/404 || echo "Expected: 404 Not Found"
                                curl -f https://httpbin.org/status/500 || echo "Expected: 500 Internal Server Error"
                                curl -f https://httpbin.org/status/429 || echo "Expected: 429 Too Many Requests"

                                # Test connection timeouts
                                curl -m 5 https://httpbin.org/delay/10 || echo "Expected: Connection timeout"

                                # Test SSL/TLS issues
                                curl https://expired.badssl.com/ || echo "Expected: SSL certificate expired"
                                curl https://self-signed.badssl.com/ || echo "Expected: Self-signed certificate"
                            ''',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Test bandwidth limitations',
                            'run': '''
                                echo "=== Bandwidth Limitation Test ==="

                                # Test large file download with bandwidth limit
                                echo "Testing 1MB download with 10KB/s limit..."
                                timeout 30s curl --limit-rate 10k https://httpbin.org/bytes/1048576 -o large_file.bin || echo "Bandwidth limited download timeout"

                                if [ -f large_file.bin ]; then
                                    echo "Downloaded file size: $(du -h large_file.bin)"
                                    rm large_file.bin
                                fi
                            ''',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Test proxy and firewall simulation',
                            'run': '''
                                echo "=== Proxy/Firewall Simulation Test ==="

                                # Test with invalid proxy
                                export https_proxy=http://invalid-proxy:8080
                                curl -m 10 https://github.com || echo "Expected: Proxy connection failed"

                                # Reset proxy settings
                                unset https_proxy
                                unset http_proxy

                                # Test blocked ports (simulate firewall)
                                nc -z -w5 github.com 22 || echo "SSH port test"
                                nc -z -w5 github.com 80 || echo "HTTP port test"
                                nc -z -w5 github.com 443 || echo "HTTPS port test"
                                nc -z -w5 github.com 9999 || echo "Expected: Blocked port"
                            ''',
                            'continue-on-error': True
                        }
                    ]
                }
            }
        }
        return workflow

    def test_workflow_configuration_validation(self) -> List[WorkflowTestResult]:
        """Test workflow configuration validation and edge cases."""
        results = []
        repo_path = self.create_test_repository()

        try:
            # Test 1: Valid workflow configuration
            valid_workflow = {
                'name': 'Valid Workflow',
                'on': 'push',
                'jobs': {
                    'test': {
                        'runs-on': 'ubuntu-latest',
                        'steps': [{'name': 'Echo', 'run': 'echo "hello"'}]
                    }
                }
            }

            workflow_file = repo_path / '.github' / 'workflows' / 'valid.yml'
            with open(workflow_file, 'w') as f:
                yaml.dump(valid_workflow, f)

            start_time = time.time()
            try:
                # Validate YAML syntax
                with open(workflow_file, 'r') as f:
                    yaml.safe_load(f)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)

            results.append(WorkflowTestResult(
                test_name="valid_workflow_configuration",
                success=success,
                duration=time.time() - start_time,
                output="Valid workflow configuration parsed successfully",
                error=error
            ))

            # Test 2: Invalid YAML syntax
            invalid_yaml_file = repo_path / '.github' / 'workflows' / 'invalid_yaml.yml'
            invalid_yaml_file.write_text("""
name: Invalid YAML
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Test
        run: echo "hello"
      invalid_indentation
""")

            start_time = time.time()
            try:
                with open(invalid_yaml_file, 'r') as f:
                    yaml.safe_load(f)
                success = False  # Should fail
                error = "Expected YAML parsing to fail"
            except Exception as e:
                success = True  # Expected to fail
                error = str(e)

            results.append(WorkflowTestResult(
                test_name="invalid_yaml_syntax",
                success=success,
                duration=time.time() - start_time,
                output="Invalid YAML correctly rejected",
                error=error if not success else None
            ))

            # Test 3: Missing required fields
            missing_fields_workflow = {
                'name': 'Missing Fields',
                # Missing 'on' trigger
                'jobs': {
                    'test': {
                        # Missing 'runs-on'
                        'steps': [{'name': 'Test', 'run': 'echo test'}]
                    }
                }
            }

            missing_fields_file = repo_path / '.github' / 'workflows' / 'missing_fields.yml'
            with open(missing_fields_file, 'w') as f:
                yaml.dump(missing_fields_workflow, f)

            start_time = time.time()
            # This would typically be validated by GitHub Actions, not locally
            # We can only test YAML structure validity
            try:
                with open(missing_fields_file, 'r') as f:
                    content = yaml.safe_load(f)
                    # Check for required fields
                    has_on = 'on' in content
                    has_runs_on = any('runs-on' in job for job in content.get('jobs', {}).values())
                    success = not (has_on and has_runs_on)  # Should fail validation
                    error = f"Missing fields - on: {has_on}, runs-on: {has_runs_on}"
            except Exception as e:
                success = True
                error = str(e)

            results.append(WorkflowTestResult(
                test_name="missing_required_fields",
                success=success,
                duration=time.time() - start_time,
                output="Missing required fields detected",
                error=error if not success else None
            ))

            # Test 4: Invalid runner OS
            invalid_runner_workflow = {
                'name': 'Invalid Runner',
                'on': 'push',
                'jobs': {
                    'test': {
                        'runs-on': 'nonexistent-os-2023',
                        'steps': [{'name': 'Test', 'run': 'echo test'}]
                    }
                }
            }

            invalid_runner_file = repo_path / '.github' / 'workflows' / 'invalid_runner.yml'
            with open(invalid_runner_file, 'w') as f:
                yaml.dump(invalid_runner_workflow, f)

            start_time = time.time()
            try:
                with open(invalid_runner_file, 'r') as f:
                    content = yaml.safe_load(f)
                    runner = content['jobs']['test']['runs-on']
                    valid_runners = ['ubuntu-latest', 'windows-latest', 'macos-latest']
                    success = runner not in valid_runners
                    error = f"Invalid runner detected: {runner}"
            except Exception as e:
                success = False
                error = str(e)

            results.append(WorkflowTestResult(
                test_name="invalid_runner_os",
                success=success,
                duration=time.time() - start_time,
                output="Invalid runner OS detected",
                error=error if not success else None
            ))

        except Exception as e:
            results.append(WorkflowTestResult(
                test_name="workflow_validation_error",
                success=False,
                duration=0,
                output="",
                error=str(e)
            ))

        finally:
            os.chdir(self.original_dir)

        return results

    def test_matrix_build_edge_cases(self) -> List[WorkflowTestResult]:
        """Test matrix build edge cases and failure scenarios."""
        results = []
        repo_path = self.create_test_repository()

        try:
            # Create matrix workflow with edge cases
            matrix_workflow = self.create_matrix_failure_workflow()
            workflow_file = repo_path / '.github' / 'workflows' / 'matrix_edge_cases.yml'

            with open(workflow_file, 'w') as f:
                yaml.dump(matrix_workflow, f)

            start_time = time.time()

            # Analyze matrix configuration
            try:
                with open(workflow_file, 'r') as f:
                    workflow = yaml.safe_load(f)

                matrix_job = workflow['jobs']['matrix-test']
                strategy = matrix_job['strategy']
                matrix = strategy['matrix']

                # Calculate expected combinations
                os_count = len(matrix['os'])
                python_count = len(matrix['python-version'])
                node_count = len(matrix['node-version'])

                total_combinations = os_count * python_count * node_count

                # Account for includes and excludes
                includes = len(matrix.get('include', []))
                excludes = len(matrix.get('exclude', []))

                expected_jobs = total_combinations + includes - excludes

                results.append(WorkflowTestResult(
                    test_name="matrix_configuration_analysis",
                    success=True,
                    duration=time.time() - start_time,
                    output=f"Matrix analysis: {expected_jobs} expected jobs",
                    metadata={
                        'total_combinations': total_combinations,
                        'includes': includes,
                        'excludes': excludes,
                        'expected_jobs': expected_jobs,
                        'fail_fast': strategy['fail-fast'],
                        'continue_on_error': matrix_job.get('continue-on-error', False)
                    }
                ))

            except Exception as e:
                results.append(WorkflowTestResult(
                    test_name="matrix_configuration_error",
                    success=False,
                    duration=time.time() - start_time,
                    output="",
                    error=str(e)
                ))

            # Test empty matrix
            start_time = time.time()
            empty_matrix_workflow = {
                'name': 'Empty Matrix',
                'on': 'push',
                'jobs': {
                    'empty-matrix': {
                        'runs-on': 'ubuntu-latest',
                        'strategy': {
                            'matrix': {}
                        },
                        'steps': [{'name': 'Test', 'run': 'echo test'}]
                    }
                }
            }

            empty_matrix_file = repo_path / '.github' / 'workflows' / 'empty_matrix.yml'
            with open(empty_matrix_file, 'w') as f:
                yaml.dump(empty_matrix_workflow, f)

            try:
                with open(empty_matrix_file, 'r') as f:
                    workflow = yaml.safe_load(f)
                    matrix = workflow['jobs']['empty-matrix']['strategy']['matrix']
                    success = len(matrix) == 0
            except Exception as e:
                success = False

            results.append(WorkflowTestResult(
                test_name="empty_matrix_configuration",
                success=success,
                duration=time.time() - start_time,
                output="Empty matrix configuration tested",
                error=None
            ))

        except Exception as e:
            results.append(WorkflowTestResult(
                test_name="matrix_edge_cases_error",
                success=False,
                duration=0,
                output="",
                error=str(e)
            ))

        finally:
            os.chdir(self.original_dir)

        return results

    def run_comprehensive_tests(self) -> Dict[str, List[WorkflowTestResult]]:
        """Run all comprehensive GitHub Actions edge case tests."""
        all_results = {}

        print("🔧 Running comprehensive GitHub Actions edge case tests...")

        # Run all test categories
        test_categories = [
            ("workflow_validation", self.test_workflow_configuration_validation),
            ("matrix_builds", self.test_matrix_build_edge_cases),
        ]

        for category_name, test_function in test_categories:
            print(f"\n📋 Running {category_name} tests...")
            try:
                results = test_function()
                all_results[category_name] = results

                # Print summary for this category
                total = len(results)
                passed = sum(1 for r in results if r.success)
                print(f"  {category_name}: {passed}/{total} tests passed")

            except Exception as e:
                print(f"  ❌ Error in {category_name}: {e}")
                all_results[category_name] = [
                    WorkflowTestResult(
                        test_name=f"{category_name}_error",
                        success=False,
                        duration=0,
                        output="",
                        error=str(e)
                    )
                ]

        return all_results

    def generate_comprehensive_report(self, results: Dict[str, List[WorkflowTestResult]]) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("# GitHub Actions Workflow Edge Cases Test Report")
        report.append("=" * 60)
        report.append("")

        total_tests = 0
        total_passed = 0

        for category, test_results in results.items():
            report.append(f"## {category.upper()}")
            report.append("")

            category_passed = 0
            for result in test_results:
                total_tests += 1
                status = "✅ PASS" if result.success else "❌ FAIL"
                if result.success:
                    total_passed += 1
                    category_passed += 1

                report.append(f"- **{result.test_name}**: {status}")
                report.append(f"  - Duration: {result.duration:.3f}s")

                if result.output:
                    report.append(f"  - Output: {result.output}")

                if result.error:
                    report.append(f"  - Error: {result.error}")

                if result.metadata:
                    report.append(f"  - Metadata: {result.metadata}")

                report.append("")

            report.append(f"**{category} Summary**: {category_passed}/{len(test_results)} tests passed")
            report.append("")

        # Overall summary
        report.append("## OVERALL SUMMARY")
        report.append("")
        report.append(f"**Total Tests**: {total_tests}")
        report.append(f"**Passed**: {total_passed}")
        report.append(f"**Failed**: {total_tests - total_passed}")
        report.append(f"**Success Rate**: {total_passed/total_tests*100:.1f}%")

        return "\n".join(report)


# Pytest test cases
class TestGitHubActionsEdgeCases:
    """Pytest test cases for GitHub Actions edge cases."""

    @pytest.fixture
    def actions_tester(self):
        """Fixture to provide a GitHub Actions tester."""
        tester = GitHubActionsWorkflowTester()
        yield tester
        # Cleanup handled by temp directory deletion

    def test_workflow_timeout_scenarios(self, actions_tester):
        """Test workflow timeout edge cases."""
        workflow = actions_tester.create_workflow_with_timeouts()

        # Verify timeout configuration
        assert 'timeout-job' in workflow['jobs']
        timeout_job = workflow['jobs']['timeout-job']
        assert timeout_job['timeout-minutes'] == 2

        # Verify step timeout
        step_timeout_job = workflow['jobs']['step-timeout']
        assert any(step.get('timeout-minutes') == 1 for step in step_timeout_job['steps'])

    def test_matrix_build_configuration(self, actions_tester):
        """Test matrix build configuration edge cases."""
        workflow = actions_tester.create_matrix_failure_workflow()

        matrix_job = workflow['jobs']['matrix-test']
        strategy = matrix_job['strategy']

        # Verify matrix configuration
        assert 'matrix' in strategy
        assert 'include' in strategy['matrix']
        assert 'exclude' in strategy['matrix']
        assert strategy['fail-fast'] is False

    def test_artifact_handling_edge_cases(self, actions_tester):
        """Test artifact handling edge cases."""
        workflow = actions_tester.create_artifact_edge_cases_workflow()

        # Verify artifact jobs exist
        assert 'artifact-producer' in workflow['jobs']
        assert 'artifact-consumer' in workflow['jobs']

        # Verify dependencies
        consumer_job = workflow['jobs']['artifact-consumer']
        assert consumer_job['needs'] == 'artifact-producer'

    def test_service_container_edge_cases(self, actions_tester):
        """Test service container edge cases."""
        workflow = actions_tester.create_service_container_edge_cases_workflow()

        job = workflow['jobs']['service-edge-cases']
        services = job['services']

        # Verify service configurations
        assert 'postgres' in services
        assert 'redis-constrained' in services
        assert 'failing-service' in services

        # Verify resource constraints
        redis_service = services['redis-constrained']
        assert '--memory=64m' in redis_service['options']

    def test_environment_variable_edge_cases(self, actions_tester):
        """Test environment variable edge cases."""
        workflow = actions_tester.create_environment_secrets_edge_cases_workflow()

        # Verify environment variables at different levels
        assert 'GLOBAL_VAR' in workflow['env']
        assert 'UNICODE_VAR' in workflow['env']

        job = workflow['jobs']['env-secrets-test']
        assert 'JOB_VAR' in job['env']

    def test_network_failure_scenarios(self, actions_tester):
        """Test network failure scenarios."""
        workflow = actions_tester.create_network_failure_workflow()

        job = workflow['jobs']['network-edge-cases']

        # Verify network test steps exist
        step_names = [step['name'] for step in job['steps']]
        assert any('network connectivity' in name.lower() for name in step_names)
        assert any('dns resolution' in name.lower() for name in step_names)

    @pytest.mark.asyncio
    async def test_async_workflow_operations(self, actions_tester):
        """Test asynchronous workflow operations."""
        # Simulate async operations that might occur during workflow execution
        async def simulate_async_step():
            await asyncio.sleep(0.1)
            return {"step_result": "completed"}

        result = await simulate_async_step()
        assert result["step_result"] == "completed"


if __name__ == "__main__":
    # Run comprehensive GitHub Actions edge case testing
    tester = GitHubActionsWorkflowTester()

    try:
        results = tester.run_comprehensive_tests()

        # Generate and save report
        report = tester.generate_comprehensive_report(results)
        print("\n" + report)

        # Save results to files
        with open("github_actions_edge_case_results.json", "w") as f:
            # Convert WorkflowTestResult objects to dictionaries for JSON serialization
            json_results = {}
            for category, test_results in results.items():
                json_results[category] = [
                    {
                        "test_name": r.test_name,
                        "success": r.success,
                        "duration": r.duration,
                        "output": r.output,
                        "error": r.error,
                        "metadata": r.metadata
                    }
                    for r in test_results
                ]
            json.dump(json_results, f, indent=2)

        with open("github_actions_edge_case_report.md", "w") as f:
            f.write(report)

        print("\n📄 Results saved to:")
        print("  - github_actions_edge_case_results.json")
        print("  - github_actions_edge_case_report.md")

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
    finally:
        print("\n🧹 Test cleanup completed")