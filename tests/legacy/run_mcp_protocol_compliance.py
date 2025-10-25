#!/usr/bin/env python3
"""
Comprehensive MCP Protocol Compliance Test Runner (Task 285.8).

This script runs the complete MCP protocol compliance test suite and generates
detailed reports including compliance scores, violations, and recommendations.

Features:
    - Runs all MCP protocol compliance tests
    - Generates detailed compliance reports
    - Calculates coverage against MCP specification
    - Produces JSON and HTML reports
    - Supports CI/CD integration

Usage:
    # Run all tests with detailed report
    python run_mcp_protocol_compliance.py

    # Run with JSON report output
    python run_mcp_protocol_compliance.py --format json --output report.json

    # Run with CI/CD mode (exit code based on compliance)
    python run_mcp_protocol_compliance.py --ci --threshold 0.85
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))
sys.path.insert(0, str(project_root / "tests"))

from integration.test_mcp_protocol_compliance import (
    MCPProtocolComplianceTester,
)
from utils.fastmcp_test_infrastructure import (
    FastMCPTestServer,
    fastmcp_test_environment,
)
from workspace_qdrant_mcp.server import app


class MCPComplianceReportGenerator:
    """Generate comprehensive MCP protocol compliance reports."""

    def __init__(self, compliance_results: dict[str, Any]):
        """
        Initialize report generator.

        Args:
            compliance_results: Results from MCPProtocolComplianceTester
        """
        self.results = compliance_results
        self.timestamp = datetime.now(timezone.utc)

    def generate_text_report(self) -> str:
        """Generate human-readable text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("MCP PROTOCOL COMPLIANCE TEST REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {self.timestamp.isoformat()}")
        lines.append("")

        # Summary
        summary = self.results.get("summary", {})
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Overall Compliance: {summary.get('overall_compliance', 0):.1%}")
        lines.append(f"Compliance Level: {summary.get('compliance_level', 'unknown').upper()}")
        lines.append(f"Total Categories: {summary.get('total_categories', 0)}")
        lines.append(f"Total Violations: {summary.get('total_violations', 0)}")
        lines.append(f"Total Warnings: {summary.get('total_warnings', 0)}")
        lines.append("")

        # Category breakdown
        lines.append("CATEGORY BREAKDOWN")
        lines.append("-" * 80)

        categories = [
            ("Tool Call Validation", "tool_call_validation"),
            ("Resource Access", "resource_access"),
            ("Error Response Format", "error_response_format"),
            ("Capability Negotiation", "capability_negotiation"),
            ("Message Format", "message_format"),
            ("Protocol Version", "protocol_version"),
            ("Parameter Validation", "parameter_validation"),
            ("Unknown Tool Handling", "unknown_tool_handling"),
            ("Malformed Messages", "malformed_messages"),
        ]

        for category_name, category_key in categories:
            if category_key in self.results:
                category_data = self.results[category_key]
                score = category_data.get("compliance_score", 0)
                total = category_data.get("total_tests", 0)
                violations = len(category_data.get("violations", []))

                status = "✓" if score >= 0.9 else "⚠" if score >= 0.7 else "✗"
                lines.append(f"{status} {category_name:30s} {score:6.1%}  ({total} tests, {violations} violations)")

        lines.append("")

        # Violations
        violations = self.results.get("violations", [])
        if violations:
            lines.append("VIOLATIONS")
            lines.append("-" * 80)
            for i, violation in enumerate(violations, 1):
                lines.append(f"{i}. {violation}")
            lines.append("")

        # Warnings
        warnings = self.results.get("warnings", [])
        if warnings:
            lines.append("WARNINGS")
            lines.append("-" * 80)
            for i, warning in enumerate(warnings, 1):
                lines.append(f"{i}. {warning}")
            lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_json_report(self) -> dict[str, Any]:
        """Generate machine-readable JSON report."""
        return {
            "report_metadata": {
                "generated_at": self.timestamp.isoformat(),
                "report_version": "1.0",
                "test_suite": "MCP Protocol Compliance",
            },
            "summary": self.results.get("summary", {}),
            "categories": {
                key: self.results[key]
                for key in self.results
                if key not in ["summary", "violations", "warnings"]
            },
            "violations": self.results.get("violations", []),
            "warnings": self.results.get("warnings", []),
            "recommendations": self._generate_recommendations(),
        }

    def generate_html_report(self) -> str:
        """Generate HTML report."""
        summary = self.results.get("summary", {})
        compliance = summary.get("overall_compliance", 0)
        level = summary.get("compliance_level", "unknown")

        # Determine color based on compliance level
        color_map = {
            "excellent": "#28a745",
            "good": "#5cb85c",
            "acceptable": "#f0ad4e",
            "needs_improvement": "#d9534f",
        }
        color = color_map.get(level, "#6c757d")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MCP Protocol Compliance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid {color}; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .compliance-score {{ font-size: 48px; font-weight: bold; color: {color}; text-align: center; margin: 20px 0; }}
        .compliance-level {{ text-align: center; font-size: 24px; color: #666; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background-color: #343a40; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .pass {{ color: #28a745; font-weight: bold; }}
        .warn {{ color: #f0ad4e; font-weight: bold; }}
        .fail {{ color: #d9534f; font-weight: bold; }}
        .violation {{ background-color: #f8d7da; padding: 10px; margin: 5px 0; border-left: 4px solid #d9534f; }}
        .warning {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #f0ad4e; }}
        .recommendation {{ background-color: #d1ecf1; padding: 10px; margin: 5px 0; border-left: 4px solid #17a2b8; }}
        .timestamp {{ color: #999; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MCP Protocol Compliance Test Report</h1>
        <p class="timestamp">Generated: {self.timestamp.isoformat()}</p>

        <div class="summary">
            <div class="compliance-score">{compliance:.1%}</div>
            <div class="compliance-level">{level}</div>
        </div>

        <h2>Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Overall Compliance</td>
                <td>{compliance:.1%}</td>
            </tr>
            <tr>
                <td>Compliance Level</td>
                <td>{level.upper()}</td>
            </tr>
            <tr>
                <td>Total Categories</td>
                <td>{summary.get('total_categories', 0)}</td>
            </tr>
            <tr>
                <td>Total Violations</td>
                <td class="{'pass' if summary.get('total_violations', 0) == 0 else 'fail'}">{summary.get('total_violations', 0)}</td>
            </tr>
            <tr>
                <td>Total Warnings</td>
                <td class="{'pass' if summary.get('total_warnings', 0) == 0 else 'warn'}">{summary.get('total_warnings', 0)}</td>
            </tr>
        </table>

        <h2>Category Results</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Compliance Score</th>
                <th>Tests</th>
                <th>Violations</th>
                <th>Status</th>
            </tr>
"""

        categories = [
            ("Tool Call Validation", "tool_call_validation"),
            ("Resource Access", "resource_access"),
            ("Error Response Format", "error_response_format"),
            ("Capability Negotiation", "capability_negotiation"),
            ("Message Format", "message_format"),
            ("Protocol Version", "protocol_version"),
            ("Parameter Validation", "parameter_validation"),
            ("Unknown Tool Handling", "unknown_tool_handling"),
            ("Malformed Messages", "malformed_messages"),
        ]

        for category_name, category_key in categories:
            if category_key in self.results:
                category_data = self.results[category_key]
                score = category_data.get("compliance_score", 0)
                total = category_data.get("total_tests", 0)
                violations = len(category_data.get("violations", []))

                status_class = "pass" if score >= 0.9 else "warn" if score >= 0.7 else "fail"
                status_text = "✓ PASS" if score >= 0.9 else "⚠ WARNING" if score >= 0.7 else "✗ FAIL"

                html += f"""
            <tr>
                <td>{category_name}</td>
                <td>{score:.1%}</td>
                <td>{total}</td>
                <td class="{status_class}">{violations}</td>
                <td class="{status_class}">{status_text}</td>
            </tr>
"""

        html += """
        </table>
"""

        # Violations
        violations = self.results.get("violations", [])
        if violations:
            html += "<h2>Violations</h2>"
            for violation in violations:
                html += f'<div class="violation">{violation}</div>'

        # Warnings
        warnings = self.results.get("warnings", [])
        if warnings:
            html += "<h2>Warnings</h2>"
            for warning in warnings:
                html += f'<div class="warning">{warning}</div>'

        # Recommendations
        html += "<h2>Recommendations</h2>"
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            html += f'<div class="recommendation">{rec}</div>'

        html += """
    </div>
</body>
</html>
"""

        return html

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        summary = self.results.get("summary", {})
        compliance = summary.get("overall_compliance", 0)

        if compliance < 0.95:
            recommendations.append(
                "Review and address all protocol violations to improve compliance score"
            )

        # Category-specific recommendations
        for category_key, category_name in [
            ("tool_call_validation", "Tool Call Validation"),
            ("resource_access", "Resource Access"),
            ("error_response_format", "Error Response Format"),
            ("capability_negotiation", "Capability Negotiation"),
            ("message_format", "Message Format"),
            ("protocol_version", "Protocol Version"),
            ("parameter_validation", "Parameter Validation"),
            ("unknown_tool_handling", "Unknown Tool Handling"),
            ("malformed_messages", "Malformed Message Handling"),
        ]:
            if category_key in self.results:
                category_data = self.results[category_key]
                score = category_data.get("compliance_score", 0)
                violations = category_data.get("violations", [])

                if score < 0.9 and violations:
                    recommendations.append(
                        f"Improve {category_name}: {violations[0]}"
                    )

        if not recommendations:
            recommendations.append(
                "Excellent compliance! Continue maintaining MCP protocol standards"
            )

        return recommendations


async def run_compliance_tests() -> dict[str, Any]:
    """Run comprehensive MCP protocol compliance tests."""
    async with fastmcp_test_environment(app) as (server, client):
        tester = MCPProtocolComplianceTester(server)
        results = await tester.run_all_compliance_tests()
        return results


def main():
    """Main entry point for MCP protocol compliance test runner."""
    parser = argparse.ArgumentParser(
        description="Run MCP Protocol Compliance Tests"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "html", "all"],
        default="text",
        help="Report format (default: text)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: stdout for text, mcp_compliance_report.{format} for others)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI/CD mode: exit with non-zero code if compliance below threshold",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum compliance threshold for CI mode (default: 0.85)",
    )

    args = parser.parse_args()

    # Run tests
    print("Running MCP Protocol Compliance Tests...")
    results = asyncio.run(run_compliance_tests())

    # Generate report
    reporter = MCPComplianceReportGenerator(results)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if args.format == "text" or args.format == "all":
            output_path = None  # stdout
        else:
            output_path = Path(f"mcp_compliance_report.{args.format}")

    # Generate and save reports
    if args.format == "text" or args.format == "all":
        text_report = reporter.generate_text_report()
        if output_path and args.format == "text":
            output_path.write_text(text_report)
            print(f"\nReport saved to: {output_path}")
        else:
            print(text_report)
            if args.format == "all":
                Path("mcp_compliance_report.txt").write_text(text_report)

    if args.format == "json" or args.format == "all":
        json_report = reporter.generate_json_report()
        json_path = output_path if args.format == "json" else Path("mcp_compliance_report.json")
        json_path.write_text(json.dumps(json_report, indent=2))
        print(f"\nJSON report saved to: {json_path}")

    if args.format == "html" or args.format == "all":
        html_report = reporter.generate_html_report()
        html_path = output_path if args.format == "html" else Path("mcp_compliance_report.html")
        html_path.write_text(html_report)
        print(f"\nHTML report saved to: {html_path}")

    # CI/CD mode: exit based on compliance
    if args.ci:
        compliance = results.get("summary", {}).get("overall_compliance", 0)
        if compliance < args.threshold:
            print(
                f"\n❌ Compliance {compliance:.1%} below threshold {args.threshold:.1%}"
            )
            sys.exit(1)
        else:
            print(
                f"\n✅ Compliance {compliance:.1%} meets threshold {args.threshold:.1%}"
            )
            sys.exit(0)


if __name__ == "__main__":
    main()
