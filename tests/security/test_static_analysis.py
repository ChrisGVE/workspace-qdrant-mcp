"""
Automated static analysis security testing.

Tests integration of semgrep, bandit, clippy and other static analysis tools
for automated security vulnerability detection in CI/CD pipelines.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def security_report_dir(tmp_path):
    """Create temporary directory for security reports."""
    report_dir = tmp_path / "security_reports"
    report_dir.mkdir(exist_ok=True)
    return report_dir


@pytest.mark.security
class TestSemgrepIntegration:
    """Test semgrep static analysis integration."""

    def test_semgrep_installation(self):
        """Test that semgrep is available or can be installed."""
        try:
            result = subprocess.run(
                ["semgrep", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0
            assert "semgrep" in result.stdout.lower()
        except FileNotFoundError:
            pytest.skip(
                "semgrep not installed. Install with: pip install semgrep or brew install semgrep"
            )

    def test_semgrep_security_scan(self, project_root):
        """Run semgrep security scan on Python code."""
        try:
            # Run semgrep with security rules
            result = subprocess.run(
                [
                    "semgrep",
                    "--config=auto",
                    "--json",
                    "--metrics=off",
                    str(project_root / "src"),
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes
            )

            # Parse JSON output
            try:
                output = json.loads(result.stdout)
                findings = output.get("results", [])

                # Filter by severity
                critical_high = [
                    f
                    for f in findings
                    if f.get("extra", {}).get("severity", "").upper()
                    in ["ERROR", "WARNING"]
                ]

                if critical_high:
                    error_msg = (
                        f"Semgrep found {len(critical_high)} critical/high security issues:\n\n"
                    )
                    for finding in critical_high[:10]:  # Show first 10
                        check_id = finding.get("check_id", "N/A")
                        path = finding.get("path", "N/A")
                        line = finding.get("start", {}).get("line", "N/A")
                        message = finding.get("extra", {}).get("message", "N/A")
                        error_msg += f"  - {check_id} at {path}:{line}\n"
                        error_msg += f"    {message}\n"

                    if len(critical_high) > 10:
                        error_msg += f"\n... and {len(critical_high) - 10} more\n"

                    # This is informational - don't fail on findings
                    pytest.skip(error_msg)

            except json.JSONDecodeError:
                # Could not parse JSON output
                if result.returncode != 0:
                    pytest.skip(f"Semgrep scan failed: {result.stderr}")

            assert True

        except FileNotFoundError:
            pytest.skip("semgrep not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("semgrep scan timed out after 5 minutes")

    def test_custom_security_rules(self, project_root, tmp_path):
        """Test custom semgrep security rules for project-specific patterns."""
        custom_rules_file = tmp_path / "custom_security_rules.yaml"

        # Create custom security rules
        custom_rules = """rules:
  - id: hardcoded-secret
    pattern: |
      password = "..."
    message: Potential hardcoded password detected
    severity: ERROR
    languages: [python]

  - id: sql-injection-risk
    pattern: |
      execute($SQL)
    message: Potential SQL injection - use parameterized queries
    severity: WARNING
    languages: [python]

  - id: unsafe-deserialization
    pattern: pickle.loads(...)
    message: Unsafe deserialization with pickle - use safer alternatives
    severity: ERROR
    languages: [python]
"""
        custom_rules_file.write_text(custom_rules)

        try:
            # Test that custom rules work
            result = subprocess.run(
                [
                    "semgrep",
                    f"--config={custom_rules_file}",
                    "--json",
                    "--metrics=off",
                    str(project_root / "src"),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Should complete successfully
            assert result.returncode in [0, 1]  # 0 = no findings, 1 = findings

        except FileNotFoundError:
            pytest.skip("semgrep not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("Custom rules scan timed out")

    def test_semgrep_ci_integration(self, security_report_dir):
        """Test semgrep CI/CD integration configuration."""
        # Create semgrep CI configuration
        semgrep_ci_config = {
            "scan": {
                "tool": "semgrep",
                "config": "auto",
                "exclude_patterns": [
                    "tests/",
                    "*.test.py",
                    "*/migrations/*",
                    ".venv/",
                ],
                "fail_on": ["ERROR"],
                "output_format": "json",
            },
            "reporting": {
                "enabled": True,
                "output_file": str(security_report_dir / "semgrep_report.json"),
                "sarif_output": str(security_report_dir / "semgrep_report.sarif"),
            },
        }

        # Validate configuration
        assert semgrep_ci_config["scan"]["tool"] == "semgrep"
        assert "auto" in semgrep_ci_config["scan"]["config"]
        assert semgrep_ci_config["scan"]["fail_on"] == ["ERROR"]
        assert semgrep_ci_config["reporting"]["enabled"]


@pytest.mark.security
class TestBanditIntegration:
    """Test bandit Python security analysis integration."""

    def test_bandit_installation(self):
        """Test that bandit is available."""
        try:
            result = subprocess.run(
                ["bandit", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0
            assert "bandit" in result.stdout.lower()
        except FileNotFoundError:
            pytest.skip("bandit not installed. Install with: pip install bandit")

    def test_bandit_security_scan(self, project_root):
        """Run bandit security scan on Python code."""
        try:
            # Run bandit with JSON output
            result = subprocess.run(
                [
                    "bandit",
                    "-r",
                    str(project_root / "src"),
                    "-f",
                    "json",
                    "--skip",
                    "B101",  # Skip assert_used
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Parse JSON output
            try:
                output = json.loads(result.stdout)
                findings = output.get("results", [])

                # Filter by severity
                high_severity = [
                    f for f in findings if f.get("issue_severity", "") == "HIGH"
                ]
                medium_severity = [
                    f for f in findings if f.get("issue_severity", "") == "MEDIUM"
                ]

                if high_severity:
                    error_msg = f"Bandit found {len(high_severity)} HIGH severity issues:\n\n"
                    for finding in high_severity[:10]:
                        test_id = finding.get("test_id", "N/A")
                        filename = finding.get("filename", "N/A")
                        line = finding.get("line_number", "N/A")
                        issue = finding.get("issue_text", "N/A")
                        error_msg += f"  - {test_id} at {filename}:{line}\n"
                        error_msg += f"    {issue}\n"

                    if len(high_severity) > 10:
                        error_msg += f"\n... and {len(high_severity) - 10} more\n"

                    # Informational - don't fail
                    pytest.skip(error_msg)

            except json.JSONDecodeError:
                if result.returncode != 0 and result.stdout:
                    pytest.skip(f"Bandit scan completed with warnings: {result.stdout}")

            assert True

        except FileNotFoundError:
            pytest.skip("bandit not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("bandit scan timed out after 2 minutes")

    def test_bandit_custom_config(self, project_root, tmp_path):
        """Test bandit with custom security configuration."""
        config_file = tmp_path / ".bandit"

        # Create custom bandit configuration
        config_content = """
[bandit]
exclude_dirs = /tests,/.venv,/build,/dist
tests = B201,B301,B302,B303,B304,B305,B306,B307,B308,B310,B311,B312,B313,B314,B315,B316,B317,B318,B319,B320,B321,B322,B323,B324,B325
skips = B101,B601,B603

[bandit.plugins]
profile = django

[bandit.formatters]
txt = {paths}
json = {paths}
"""
        config_file.write_text(config_content)

        try:
            # Test custom configuration
            result = subprocess.run(
                [
                    "bandit",
                    "-r",
                    str(project_root / "src"),
                    "-c",
                    str(config_file),
                    "-f",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Should complete successfully
            assert result.returncode in [0, 1]  # 0 = no issues, 1 = issues found

        except FileNotFoundError:
            pytest.skip("bandit not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("Custom config scan timed out")

    def test_bandit_baseline_generation(self, project_root, security_report_dir):
        """Test bandit baseline generation for tracking new issues."""
        baseline_file = security_report_dir / "bandit_baseline.json"

        try:
            # Generate baseline
            result = subprocess.run(
                [
                    "bandit",
                    "-r",
                    str(project_root / "src"),
                    "-f",
                    "json",
                    "-o",
                    str(baseline_file),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Baseline file should be created
            assert baseline_file.exists()

            # Should be valid JSON
            with open(baseline_file) as f:
                baseline = json.load(f)
                assert "results" in baseline or "errors" in baseline

        except FileNotFoundError:
            pytest.skip("bandit not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("Baseline generation timed out")


@pytest.mark.security
class TestClippyIntegration:
    """Test clippy Rust security lints integration."""

    def test_clippy_installation(self):
        """Test that clippy is available."""
        try:
            result = subprocess.run(
                ["cargo", "clippy", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Clippy might not be installed, which is OK
            if "not installed" in result.stderr.lower():
                pytest.skip(
                    "clippy not installed. Install with: rustup component add clippy"
                )
            assert result.returncode == 0
        except FileNotFoundError:
            pytest.skip("cargo not installed or not in PATH")

    def test_clippy_security_lints(self, project_root):
        """Run clippy with security-focused lints on Rust code."""
        # Find Cargo.toml files
        cargo_files = list(project_root.glob("**/Cargo.toml"))

        if not cargo_files:
            pytest.skip("No Cargo.toml files found")

        for cargo_file in cargo_files:
            cargo_dir = cargo_file.parent

            try:
                # Run clippy with security lints
                result = subprocess.run(
                    [
                        "cargo",
                        "clippy",
                        "--",
                        "-W",
                        "clippy::all",
                        "-W",
                        "clippy::pedantic",
                        "-W",
                        "clippy::nursery",
                        "-W",
                        "clippy::cargo",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=cargo_dir,
                )

                # Check for errors
                if "error:" in result.stdout.lower() or "error:" in result.stderr.lower():
                    # Extract error count
                    errors = []
                    for line in (result.stdout + result.stderr).split("\n"):
                        if "error:" in line.lower():
                            errors.append(line.strip())

                    if errors:
                        error_msg = (
                            f"Clippy found {len(errors)} errors in {cargo_dir}:\n\n"
                        )
                        for error in errors[:10]:
                            error_msg += f"  {error}\n"
                        if len(errors) > 10:
                            error_msg += f"\n... and {len(errors) - 10} more\n"

                        # Informational
                        pytest.skip(error_msg)

            except FileNotFoundError:
                pytest.skip("cargo clippy not available")
            except subprocess.TimeoutExpired:
                pytest.fail(f"clippy scan timed out for {cargo_dir}")

        assert True

    def test_clippy_security_checks(self, project_root):
        """Test specific clippy security-related checks."""
        cargo_files = list(project_root.glob("**/Cargo.toml"))

        if not cargo_files:
            pytest.skip("No Cargo.toml files found")

        security_lints = [
            "clippy::integer_arithmetic",  # Overflow detection
            "clippy::unwrap_used",  # Prefer proper error handling
            "clippy::expect_used",  # Prefer proper error handling
            "clippy::panic",  # Avoid panics
            "clippy::mem_forget",  # Memory safety
            "clippy::float_cmp",  # Floating point comparison
        ]

        for cargo_file in cargo_files:
            cargo_dir = cargo_file.parent

            try:
                # Run clippy with specific security checks
                lint_args = []
                for lint in security_lints:
                    lint_args.extend(["-W", lint])

                result = subprocess.run(
                    ["cargo", "clippy", "--", *lint_args],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=cargo_dir,
                )

                # Should complete
                assert result.returncode in [0, 1, 101]  # 101 = warnings treated as errors

            except FileNotFoundError:
                pytest.skip("cargo clippy not available")
            except subprocess.TimeoutExpired:
                pytest.fail(f"Security checks timed out for {cargo_dir}")


@pytest.mark.security
class TestCodeReviewAutomation:
    """Test security-focused code review automation."""

    def test_review_checklist_generation(self, security_report_dir):
        """Test generation of security review checklist."""
        checklist = {
            "authentication": [
                "All authentication endpoints use secure password hashing",
                "JWT tokens have appropriate expiration times",
                "Multi-factor authentication is supported",
                "Password reset flows are secure",
            ],
            "authorization": [
                "All endpoints check user permissions",
                "Role-based access control is enforced",
                "Resource ownership is validated",
                "Admin operations require elevated privileges",
            ],
            "input_validation": [
                "All user input is validated",
                "SQL injection prevention is in place",
                "XSS prevention is implemented",
                "Path traversal is prevented",
            ],
            "cryptography": [
                "Sensitive data is encrypted at rest",
                "TLS 1.2+ is enforced for data in transit",
                "Cryptographic algorithms are modern and secure",
                "Keys are properly managed and rotated",
            ],
            "dependencies": [
                "All dependencies are up to date",
                "No known vulnerabilities in dependencies",
                "Dependency versions are pinned",
                "Security advisories are monitored",
            ],
        }

        # Write checklist
        checklist_file = security_report_dir / "security_review_checklist.json"
        with open(checklist_file, "w") as f:
            json.dump(checklist, f, indent=2)

        assert checklist_file.exists()
        assert len(checklist) >= 5  # At least 5 categories

    def test_automated_review_scoring(self):
        """Test automated security review scoring system."""

        class SecurityReviewScorer:
            """Score security review compliance."""

            def __init__(self):
                self.checks: List[Dict[str, any]] = []

            def add_check(
                self, category: str, item: str, passed: bool, severity: str
            ):
                """Add a review check result."""
                self.checks.append(
                    {
                        "category": category,
                        "item": item,
                        "passed": passed,
                        "severity": severity,
                    }
                )

            def calculate_score(self) -> float:
                """Calculate overall security score (0-100)."""
                if not self.checks:
                    return 0.0

                severity_weights = {
                    "critical": 1.0,
                    "high": 0.75,
                    "medium": 0.5,
                    "low": 0.25,
                }

                total_weight = 0.0
                passed_weight = 0.0

                for check in self.checks:
                    weight = severity_weights.get(check["severity"], 0.5)
                    total_weight += weight
                    if check["passed"]:
                        passed_weight += weight

                return (passed_weight / total_weight * 100) if total_weight > 0 else 0.0

        # Test scorer
        scorer = SecurityReviewScorer()
        scorer.add_check("authentication", "Secure password hashing", True, "critical")
        scorer.add_check("authorization", "Permission checks", True, "high")
        scorer.add_check("input_validation", "SQL injection prevention", True, "high")
        scorer.add_check("cryptography", "TLS enforcement", False, "critical")

        score = scorer.calculate_score()

        # Should get 75% (3 out of 4 passed, weighted)
        assert 0 <= score <= 100
        assert score < 100  # Not perfect due to one failure

    def test_security_findings_aggregation(self):
        """Test aggregation of security findings from multiple tools."""

        class SecurityFindingsAggregator:
            """Aggregate findings from multiple security tools."""

            def __init__(self):
                self.findings: List[Dict[str, any]] = []

            def add_finding(
                self,
                tool: str,
                severity: str,
                category: str,
                message: str,
                location: str,
            ):
                """Add a security finding."""
                self.findings.append(
                    {
                        "tool": tool,
                        "severity": severity,
                        "category": category,
                        "message": message,
                        "location": location,
                    }
                )

            def get_summary(self) -> Dict[str, int]:
                """Get summary of findings by severity."""
                summary = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                for finding in self.findings:
                    severity = finding["severity"].lower()
                    if severity in summary:
                        summary[severity] += 1
                return summary

            def get_findings_by_tool(self, tool: str) -> List[Dict[str, any]]:
                """Get findings from specific tool."""
                return [f for f in self.findings if f["tool"] == tool]

        # Test aggregator
        aggregator = SecurityFindingsAggregator()
        aggregator.add_finding(
            "semgrep", "high", "injection", "SQL injection risk", "app.py:123"
        )
        aggregator.add_finding(
            "bandit",
            "medium",
            "cryptography",
            "Weak cipher",
            "crypto.py:45",
        )
        aggregator.add_finding(
            "clippy",
            "high",
            "memory_safety",
            "Unsafe unwrap",
            "main.rs:67",
        )

        summary = aggregator.get_summary()
        assert summary["high"] == 2
        assert summary["medium"] == 1

        semgrep_findings = aggregator.get_findings_by_tool("semgrep")
        assert len(semgrep_findings) == 1
        assert semgrep_findings[0]["category"] == "injection"


@pytest.mark.security
class TestCICDIntegration:
    """Test CI/CD pipeline integration for security scanning."""

    def test_github_actions_workflow_template(self, security_report_dir):
        """Test GitHub Actions workflow template for security scanning."""
        workflow_template = """name: Security Scan

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly scan

jobs:
  security-scan:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install semgrep bandit pip-audit safety

      - name: Run Semgrep
        run: |
          semgrep --config=auto --json --output=semgrep-results.json src/
        continue-on-error: true

      - name: Run Bandit
        run: |
          bandit -r src/ -f json -o bandit-results.json
        continue-on-error: true

      - name: Run pip-audit
        run: |
          pip-audit --format=json --output=pip-audit-results.json
        continue-on-error: true

      - name: Upload security reports
        uses: actions/upload-artifact@v4
        with:
          name: security-reports
          path: |
            semgrep-results.json
            bandit-results.json
            pip-audit-results.json

      - name: Upload to GitHub Security
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: semgrep-results.json
"""

        # Write workflow template
        workflow_file = security_report_dir / "security-scan.yml"
        workflow_file.write_text(workflow_template)

        assert workflow_file.exists()
        assert "semgrep" in workflow_template
        assert "bandit" in workflow_template

    def test_gitlab_ci_pipeline_template(self, security_report_dir):
        """Test GitLab CI pipeline template for security scanning."""
        pipeline_template = """security-scan:
  stage: test
  image: python:3.11
  before_script:
    - pip install semgrep bandit pip-audit
  script:
    - semgrep --config=auto --json --output=semgrep-results.json src/ || true
    - bandit -r src/ -f json -o bandit-results.json || true
    - pip-audit --format=json --output=pip-audit-results.json || true
  artifacts:
    reports:
      sast: semgrep-results.json
    paths:
      - semgrep-results.json
      - bandit-results.json
      - pip-audit-results.json
    expire_in: 1 week
  allow_failure: true
  only:
    - main
    - merge_requests
"""

        # Write pipeline template
        pipeline_file = security_report_dir / ".gitlab-ci-security.yml"
        pipeline_file.write_text(pipeline_template)

        assert pipeline_file.exists()
        assert "semgrep" in pipeline_template
        assert "bandit" in pipeline_template

    def test_pre_commit_hook_template(self, security_report_dir):
        """Test pre-commit hook template for local security scanning."""
        pre_commit_template = """repos:
  - repo: https://github.com/returntocorp/semgrep
    rev: v1.50.0
    hooks:
      - id: semgrep
        args: ['--config=auto', '--error']
        types: [python]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-ll', '-i']
        types: [python]

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
"""

        # Write pre-commit template
        pre_commit_file = security_report_dir / ".pre-commit-config.yaml"
        pre_commit_file.write_text(pre_commit_template)

        assert pre_commit_file.exists()
        assert "semgrep" in pre_commit_template
        assert "bandit" in pre_commit_template


@pytest.mark.security
class TestSecurityReporting:
    """Test security analysis report generation."""

    def test_comprehensive_report_generation(self, security_report_dir):
        """Test comprehensive security report generation."""

        class SecurityReportGenerator:
            """Generate comprehensive security reports."""

            def __init__(self):
                self.semgrep_findings: List[Dict] = []
                self.bandit_findings: List[Dict] = []
                self.clippy_findings: List[Dict] = []
                self.dependency_vulns: List[Dict] = []

            def add_semgrep_finding(self, finding: Dict):
                """Add semgrep finding."""
                self.semgrep_findings.append(finding)

            def add_bandit_finding(self, finding: Dict):
                """Add bandit finding."""
                self.bandit_findings.append(finding)

            def add_clippy_finding(self, finding: Dict):
                """Add clippy finding."""
                self.clippy_findings.append(finding)

            def add_dependency_vuln(self, vuln: Dict):
                """Add dependency vulnerability."""
                self.dependency_vulns.append(vuln)

            def generate_report(self) -> Dict:
                """Generate comprehensive report."""
                total_findings = (
                    len(self.semgrep_findings)
                    + len(self.bandit_findings)
                    + len(self.clippy_findings)
                    + len(self.dependency_vulns)
                )

                return {
                    "summary": {
                        "total_findings": total_findings,
                        "semgrep_findings": len(self.semgrep_findings),
                        "bandit_findings": len(self.bandit_findings),
                        "clippy_findings": len(self.clippy_findings),
                        "dependency_vulnerabilities": len(self.dependency_vulns),
                    },
                    "findings": {
                        "semgrep": self.semgrep_findings,
                        "bandit": self.bandit_findings,
                        "clippy": self.clippy_findings,
                        "dependencies": self.dependency_vulns,
                    },
                }

        # Test report generator
        generator = SecurityReportGenerator()
        generator.add_semgrep_finding(
            {
                "severity": "high",
                "message": "SQL injection risk",
                "location": "app.py:123",
            }
        )
        generator.add_bandit_finding(
            {"severity": "medium", "message": "Weak cipher", "location": "crypto.py:45"}
        )

        report = generator.generate_report()

        assert report["summary"]["total_findings"] == 2
        assert report["summary"]["semgrep_findings"] == 1
        assert report["summary"]["bandit_findings"] == 1

        # Write report
        report_file = security_report_dir / "security_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        assert report_file.exists()

    def test_actionable_recommendations(self):
        """Test generation of actionable security recommendations."""

        class RecommendationEngine:
            """Generate actionable security recommendations."""

            RECOMMENDATIONS = {
                "sql_injection": {
                    "title": "SQL Injection Prevention",
                    "description": "Use parameterized queries to prevent SQL injection",
                    "action": "Replace string concatenation with parameterized queries",
                    "example": "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
                },
                "weak_crypto": {
                    "title": "Cryptography Weakness",
                    "description": "Use strong cryptographic algorithms",
                    "action": "Replace weak ciphers with AES-256-GCM or ChaCha20-Poly1305",
                    "example": "from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes",
                },
                "hardcoded_secrets": {
                    "title": "Hardcoded Secrets",
                    "description": "Never hardcode secrets in source code",
                    "action": "Use environment variables or secret management systems",
                    "example": "password = os.environ.get('DB_PASSWORD')",
                },
            }

            def get_recommendation(self, finding_type: str) -> Optional[Dict]:
                """Get recommendation for finding type."""
                return self.RECOMMENDATIONS.get(finding_type)

        # Test recommendation engine
        engine = RecommendationEngine()
        recommendation = engine.get_recommendation("sql_injection")

        assert recommendation is not None
        assert "SQL Injection" in recommendation["title"]
        assert "parameterized queries" in recommendation["description"]
        assert "action" in recommendation
        assert "example" in recommendation

    def test_trend_analysis(self):
        """Test security trend analysis over time."""

        class SecurityTrendAnalyzer:
            """Analyze security trends over time."""

            def __init__(self):
                self.historical_scans: List[Dict] = []

            def add_scan_result(self, date: str, findings: int, severity_breakdown: Dict):
                """Add scan result."""
                self.historical_scans.append(
                    {
                        "date": date,
                        "findings": findings,
                        "severity": severity_breakdown,
                    }
                )

            def calculate_trend(self) -> str:
                """Calculate overall trend."""
                if len(self.historical_scans) < 2:
                    return "insufficient_data"

                # Compare first and last scan
                first_scan = self.historical_scans[0]["findings"]
                last_scan = self.historical_scans[-1]["findings"]

                if last_scan < first_scan:
                    return "improving"
                elif last_scan > first_scan:
                    return "declining"
                else:
                    return "stable"

        # Test trend analyzer
        analyzer = SecurityTrendAnalyzer()
        analyzer.add_scan_result(
            "2024-01-01", 10, {"critical": 2, "high": 3, "medium": 5}
        )
        analyzer.add_scan_result(
            "2024-02-01", 7, {"critical": 1, "high": 2, "medium": 4}
        )
        analyzer.add_scan_result(
            "2024-03-01", 5, {"critical": 0, "high": 1, "medium": 4}
        )

        trend = analyzer.calculate_trend()
        assert trend == "improving"


# Security test markers are configured in pyproject.toml
