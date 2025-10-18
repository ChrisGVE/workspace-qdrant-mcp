"""
Automated dependency vulnerability scanning tests.

Tests Python and Rust dependencies for known security vulnerabilities
using pip-audit and cargo-audit.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.mark.security
class TestPythonDependencyVulnerabilities:
    """Test Python dependencies for security vulnerabilities."""

    def test_pip_audit_scan(self, project_root):
        """Run pip-audit to scan for vulnerabilities in Python dependencies."""
        try:
            # Try to run pip-audit
            result = subprocess.run(
                ["pip-audit", "--format=json"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )

            if result.returncode != 0:
                # pip-audit not installed or found vulnerabilities
                if "command not found" in result.stderr or "No module named" in result.stderr:
                    pytest.skip("pip-audit not installed. Install with: pip install pip-audit")

                # Parse JSON output if available
                try:
                    vulnerabilities = json.loads(result.stdout)
                    if vulnerabilities:
                        # Filter by severity
                        critical = []
                        high = []
                        medium = []
                        low = []

                        for vuln in vulnerabilities.get("dependencies", []):
                            for issue in vuln.get("vulnerabilities", []):
                                severity = issue.get("severity", "UNKNOWN").upper()
                                vuln_info = f"{vuln['name']}=={vuln['version']}: {issue.get('id', 'N/A')} - {issue.get('description', 'N/A')}"

                                if severity == "CRITICAL":
                                    critical.append(vuln_info)
                                elif severity == "HIGH":
                                    high.append(vuln_info)
                                elif severity == "MEDIUM":
                                    medium.append(vuln_info)
                                else:
                                    low.append(vuln_info)

                        # Fail on critical or high severity vulnerabilities
                        if critical or high:
                            error_msg = "Security vulnerabilities found:\n"
                            if critical:
                                error_msg += f"\nCRITICAL ({len(critical)}):\n" + "\n".join(critical)
                            if high:
                                error_msg += f"\nHIGH ({len(high)}):\n" + "\n".join(high)
                            if medium:
                                error_msg += f"\nMEDIUM ({len(medium)}) - Review recommended:\n" + "\n".join(medium)
                            if low:
                                error_msg += f"\nLOW ({len(low)}) - Informational:\n" + "\n".join(low)

                            pytest.fail(error_msg)
                except json.JSONDecodeError:
                    # Could not parse JSON, just check return code
                    pytest.fail(f"pip-audit found vulnerabilities. Output:\n{result.stdout}\n{result.stderr}")

            # No vulnerabilities found
            assert True

        except FileNotFoundError:
            pytest.skip("pip-audit not installed. Install with: pip install pip-audit")
        except subprocess.TimeoutExpired:
            pytest.fail("pip-audit scan timed out after 5 minutes")

    def test_safety_scan(self, project_root):
        """Run safety to scan for vulnerabilities in Python dependencies."""
        try:
            # Try to run safety check
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                cwd=project_root,
            )

            if result.returncode != 0:
                if "command not found" in result.stderr or "No module named" in result.stderr:
                    pytest.skip("safety not installed. Install with: pip install safety")

                # Parse JSON output
                try:
                    vulnerabilities = json.loads(result.stdout)
                    if vulnerabilities:
                        critical_high = []
                        medium_low = []

                        for vuln in vulnerabilities:
                            vuln_info = (
                                f"{vuln.get('package', 'N/A')}=={vuln.get('installed_version', 'N/A')}: "
                                f"{vuln.get('vulnerability_id', 'N/A')} - {vuln.get('advisory', 'N/A')}"
                            )

                            # Safety doesn't always provide severity, so treat all as potentially critical
                            severity = vuln.get('severity', 'UNKNOWN')
                            if severity in ['CRITICAL', 'HIGH', 'UNKNOWN']:
                                critical_high.append(vuln_info)
                            else:
                                medium_low.append(vuln_info)

                        if critical_high:
                            error_msg = f"Security vulnerabilities found:\n\nCRITICAL/HIGH ({len(critical_high)}):\n" + "\n".join(critical_high)
                            if medium_low:
                                error_msg += f"\n\nMEDIUM/LOW ({len(medium_low)}):\n" + "\n".join(medium_low)
                            pytest.fail(error_msg)

                except json.JSONDecodeError:
                    # Could not parse JSON
                    if result.stdout and "No known security vulnerabilities" not in result.stdout:
                        pytest.fail(f"safety found vulnerabilities. Output:\n{result.stdout}\n{result.stderr}")

            # No vulnerabilities found
            assert True

        except FileNotFoundError:
            pytest.skip("safety not installed. Install with: pip install safety")
        except subprocess.TimeoutExpired:
            pytest.fail("safety scan timed out after 5 minutes")

    def test_requirements_pinned_versions(self, project_root):
        """Verify that dependencies use pinned or minimum versions."""
        pyproject_path = project_root / "pyproject.toml"

        if not pyproject_path.exists():
            pytest.skip("pyproject.toml not found")

        # Read pyproject.toml
        import toml
        try:
            config = toml.load(pyproject_path)
        except Exception as e:
            pytest.fail(f"Failed to parse pyproject.toml: {e}")

        dependencies = config.get("project", {}).get("dependencies", [])

        # Check that dependencies have version constraints
        unpinned = []
        for dep in dependencies:
            # Skip comments
            if dep.strip().startswith("#"):
                continue

            # Check for version specifier
            if ">=" not in dep and "==" not in dep and "~=" not in dep and "^" not in dep and ">" not in dep:
                # No version constraint
                unpinned.append(dep)

        if unpinned:
            pytest.fail(
                f"Dependencies without version constraints found (security risk):\n" +
                "\n".join(unpinned) +
                "\n\nAll dependencies should have version constraints (>=, ==, ~=, etc.)"
            )

        # All dependencies have version constraints
        assert True


@pytest.mark.security
class TestRustDependencyVulnerabilities:
    """Test Rust dependencies for security vulnerabilities."""

    def test_cargo_audit_scan(self, project_root):
        """Run cargo-audit to scan for vulnerabilities in Rust dependencies."""
        # Find Cargo.toml files
        cargo_files = list(project_root.glob("**/Cargo.toml"))

        if not cargo_files:
            pytest.skip("No Cargo.toml files found")

        all_vulnerabilities = []

        for cargo_file in cargo_files:
            cargo_dir = cargo_file.parent

            try:
                # Run cargo audit with JSON output
                result = subprocess.run(
                    ["cargo", "audit", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes timeout
                    cwd=cargo_dir,
                )

                # Check if cargo-audit is installed
                if "cargo-audit" in result.stderr and "not installed" in result.stderr:
                    pytest.skip("cargo-audit not installed. Install with: cargo install cargo-audit")

                # Parse output (success or failure)
                try:
                    output_lines = result.stdout.strip().split('\n') if result.stdout else []

                    for line in output_lines:
                        if line.strip():
                            try:
                                audit_output = json.loads(line)

                                # Check for vulnerabilities
                                vulnerabilities = audit_output.get("vulnerabilities", {}).get("list", [])
                                if vulnerabilities:
                                    for vuln in vulnerabilities:
                                        advisory = vuln.get("advisory", {})
                                        package = vuln.get("package", {})

                                        vuln_info = {
                                            "file": str(cargo_file),
                                            "package": package.get("name", "N/A"),
                                            "version": package.get("version", "N/A"),
                                            "id": advisory.get("id", "N/A"),
                                            "severity": vuln.get("severity", "UNKNOWN"),
                                            "title": advisory.get("title", "N/A"),
                                            "description": advisory.get("description", "N/A"),
                                            "url": advisory.get("url", "N/A"),
                                        }

                                        all_vulnerabilities.append(vuln_info)
                            except json.JSONDecodeError:
                                # Skip lines that aren't JSON
                                continue

                except Exception as e:
                    # Could not parse output
                    if result.returncode != 0 and result.stderr:
                        # Non-zero exit with errors
                        all_vulnerabilities.append({
                            "file": str(cargo_file),
                            "error": "Failed to parse cargo audit output",
                            "output": result.stdout + "\n" + result.stderr
                        })

            except FileNotFoundError:
                pytest.skip("cargo not installed or not in PATH")
            except subprocess.TimeoutExpired:
                pytest.fail(f"cargo audit timed out for {cargo_file}")

        # Check for critical/high vulnerabilities
        if all_vulnerabilities:
            critical_high = [v for v in all_vulnerabilities if v.get("severity", "").upper() in ["CRITICAL", "HIGH"]]
            medium_low = [v for v in all_vulnerabilities if v.get("severity", "").upper() in ["MEDIUM", "LOW"]]
            errors = [v for v in all_vulnerabilities if "error" in v]

            # Only fail if there are actual vulnerabilities (not just errors)
            if critical_high or medium_low or errors:
                error_msg = "Rust dependency vulnerabilities found:\n\n"

                if critical_high:
                    error_msg += f"CRITICAL/HIGH ({len(critical_high)}):\n"
                    for vuln in critical_high:
                        error_msg += f"  - {vuln.get('package')}=={vuln.get('version')}: {vuln.get('id')} - {vuln.get('title')}\n"
                        error_msg += f"    {vuln.get('url')}\n"

                if medium_low:
                    error_msg += f"\nMEDIUM/LOW ({len(medium_low)}):\n"
                    for vuln in medium_low:
                        error_msg += f"  - {vuln.get('package')}=={vuln.get('version')}: {vuln.get('id')} - {vuln.get('title')}\n"

                if errors:
                    error_msg += f"\nERRORS ({len(errors)}):\n"
                    for err in errors:
                        error_msg += f"  - {err.get('file')}: {err.get('error')}\n"
                        if err.get('output'):
                            error_msg += f"    {err.get('output')[:200]}...\n"

                # Fail on any vulnerabilities or errors
                pytest.fail(error_msg)

        # No vulnerabilities found
        assert True

    def test_cargo_lock_exists(self, project_root):
        """Verify that Cargo.lock files exist for reproducible builds."""
        cargo_files = list(project_root.glob("**/Cargo.toml"))

        if not cargo_files:
            pytest.skip("No Cargo.toml files found")

        missing_locks = []

        for cargo_file in cargo_files:
            # Only check main/workspace Cargo.toml files
            # (workspace members don't need individual Cargo.lock files)
            try:
                import toml
                cargo_config = toml.load(cargo_file)

                # Check if this is a workspace root or standalone project
                is_workspace = "workspace" in cargo_config
                is_package = "package" in cargo_config
                is_member = False

                # Check if this is a workspace member
                if is_package and not is_workspace:
                    # Look for parent workspace
                    parent = cargo_file.parent.parent
                    while parent != project_root.parent:
                        parent_cargo = parent / "Cargo.toml"
                        if parent_cargo.exists():
                            try:
                                parent_config = toml.load(parent_cargo)
                                if "workspace" in parent_config:
                                    # This is a workspace member
                                    is_member = True
                                    break
                            except:
                                pass
                        parent = parent.parent

                # Only require Cargo.lock for workspace roots and standalone projects
                if (is_workspace or (is_package and not is_member)):
                    cargo_lock = cargo_file.parent / "Cargo.lock"
                    if not cargo_lock.exists():
                        missing_locks.append(str(cargo_file.parent))

            except Exception as e:
                # If we can't parse the Cargo.toml, skip it
                continue

        if missing_locks:
            pytest.fail(
                f"Cargo.lock files missing for workspace/standalone projects:\n" +
                "\n".join(missing_locks) +
                "\n\nRun 'cargo build' in these directories to generate Cargo.lock files"
            )

        # All required Cargo.lock files exist
        assert True


@pytest.mark.security
class TestDependencyScanAutomation:
    """Test automation and CI/CD integration for dependency scanning."""

    def test_scan_script_exists(self, project_root):
        """Verify that dependency scanning automation script exists."""
        # Check for various possible locations
        possible_scripts = [
            project_root / "scripts" / "security" / "scan_dependencies.py",
            project_root / "scripts" / "scan_dependencies.py",
            project_root / ".github" / "workflows" / "security.yml",
            project_root / ".github" / "workflows" / "dependency-scan.yml",
        ]

        found = any(script.exists() for script in possible_scripts)

        if not found:
            # This is informational - not a hard failure
            pytest.skip(
                "No dependency scanning automation script found. "
                "Consider creating scripts/security/scan_dependencies.py for CI/CD integration"
            )

        assert True

    def test_scan_configuration_valid(self, project_root):
        """Test that scan configuration is valid and up-to-date."""
        # Check pyproject.toml has proper dependency groups
        pyproject_path = project_root / "pyproject.toml"

        if not pyproject_path.exists():
            pytest.skip("pyproject.toml not found")

        import toml
        try:
            config = toml.load(pyproject_path)
        except Exception as e:
            pytest.fail(f"Failed to parse pyproject.toml: {e}")

        # Verify dependency groups exist
        project_config = config.get("project", {})

        assert "dependencies" in project_config, "No dependencies section found"

        optional_deps = project_config.get("optional-dependencies", {})
        assert "dev" in optional_deps, "No dev dependencies section found"

        # Check that security tools are in dev dependencies
        dev_deps = " ".join(optional_deps.get("dev", []))

        # These are recommended but not required
        security_tools = []
        if "pip-audit" not in dev_deps and "safety" not in dev_deps:
            security_tools.append("pip-audit or safety")

        if security_tools:
            # Informational only
            pytest.skip(
                f"Security scanning tools not in dev dependencies: {', '.join(security_tools)}\n"
                "Consider adding to pyproject.toml [project.optional-dependencies.dev]"
            )

        assert True


@pytest.mark.security
class TestDependencyUpdatePolicy:
    """Test dependency update policies and procedures."""

    def test_dependency_age_check(self, project_root):
        """Check for outdated dependencies (informational)."""
        try:
            # Try pip list --outdated
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0 and result.stdout:
                try:
                    outdated = json.loads(result.stdout)
                    if outdated:
                        # Group by severity (based on version difference)
                        major_updates = []
                        minor_updates = []

                        for pkg in outdated:
                            name = pkg.get("name", "")
                            current = pkg.get("version", "")
                            latest = pkg.get("latest_version", "")

                            # Simple version comparison
                            try:
                                curr_major = int(current.split(".")[0]) if current else 0
                                latest_major = int(latest.split(".")[0]) if latest else 0

                                if latest_major > curr_major:
                                    major_updates.append(f"{name}: {current} -> {latest}")
                                else:
                                    minor_updates.append(f"{name}: {current} -> {latest}")
                            except (ValueError, IndexError):
                                minor_updates.append(f"{name}: {current} -> {latest}")

                        if major_updates:
                            # Informational - major updates may have breaking changes
                            info_msg = f"Packages with major version updates available ({len(major_updates)}):\n" + "\n".join(major_updates[:10])
                            if len(major_updates) > 10:
                                info_msg += f"\n... and {len(major_updates) - 10} more"
                            # Don't fail, just inform
                            pytest.skip(info_msg)

                except json.JSONDecodeError:
                    pass

        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("pip list --outdated not available or timed out")

        assert True


# Security test markers are configured in pyproject.toml
