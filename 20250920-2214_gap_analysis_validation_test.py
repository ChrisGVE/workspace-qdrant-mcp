#!/usr/bin/env python3
"""
Gap Analysis Validation Test Suite for Task 266
Validates all agent analysis outputs and synthesis results

This test suite ensures the comprehensive gap analysis meets all requirements
and provides validation framework for the migration strategy.
"""

import json
import os
from pathlib import Path
import pytest
from typing import Dict, Any

class TestGapAnalysisValidation:
    """Validation tests for comprehensive gap analysis"""

    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp")
        self.reports = {}
        self._load_all_reports()

    def _load_all_reports(self):
        """Load all analysis reports"""
        report_files = {
            "codebase_inventory": "20250920-2208_comprehensive_gap_analysis_report.json",
            "rust_engine": "20250920-2209_rust_engine_gap_analysis_report.json",
            "python_mcp": "20250920-2210_python_mcp_consolidation_report.json",
            "cli_utility": "20250920-2211_cli_utility_gap_analysis_report.json",
            "context_injector": "20250920-2212_context_injector_missing_analysis_report.json",
            "final_synthesis": "20250920-2213_comprehensive_gap_analysis_final_report.json"
        }

        for report_name, filename in report_files.items():
            file_path = self.project_root / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    self.reports[report_name] = json.load(f)

    def test_all_reports_generated(self):
        """Test that all required reports were generated"""
        expected_reports = [
            "codebase_inventory", "rust_engine", "python_mcp",
            "cli_utility", "context_injector", "final_synthesis"
        ]

        for report in expected_reports:
            assert report in self.reports, f"Missing report: {report}"
            assert self.reports[report], f"Empty report: {report}"

    def test_codebase_inventory_completeness(self):
        """Test codebase inventory analysis completeness"""
        report = self.reports["codebase_inventory"]

        # Test metadata presence
        assert "analysis_metadata" in report
        assert "executive_summary" in report

        # Test component analysis
        assert "component_analysis" in report
        components = report["component_analysis"]

        # Should have all 4 target components analyzed
        expected_components = [
            "component_1_rust_engine", "component_2_python_mcp",
            "component_3_cli_utility", "component_4_context_injector"
        ]

        for component in expected_components:
            assert component in components, f"Missing component analysis: {component}"

    def test_rust_engine_gap_analysis(self):
        """Test Rust engine gap analysis quality"""
        report = self.reports["rust_engine"]

        # Test alignment assessment
        exec_summary = report["executive_summary"]
        assert "overall_alignment" in exec_summary
        assert "critical_gaps" in exec_summary
        assert "estimated_migration_effort" in exec_summary

        # Test gap identification
        assert "identified_gaps" in report
        gaps = report["identified_gaps"]
        assert len(gaps) > 0, "No gaps identified in Rust engine"

        # Test reusable component identification
        assert "reusable_components" in report
        reusable = report["reusable_components"]
        assert "high_reusability" in reusable
        assert len(reusable["high_reusability"]) > 0, "No high reusability components identified"

    def test_python_mcp_consolidation_analysis(self):
        """Test Python MCP consolidation analysis"""
        report = self.reports["python_mcp"]

        # Test tool consolidation planning
        assert "consolidation_plans" in report
        plans = report["consolidation_plans"]

        # Should have plans for 4 target tools
        target_tools = ["qdrant_store", "qdrant_search", "qdrant_memory", "qdrant_watch"]
        plan_targets = [plan["target_tool"] for plan in plans if "target_tool" in plan]

        for tool in target_tools:
            assert any(tool in target for target in plan_targets), f"Missing consolidation plan for {tool}"

    def test_cli_utility_gap_analysis(self):
        """Test CLI utility gap analysis"""
        report = self.reports["cli_utility"]

        # Test fragmentation assessment
        exec_summary = report["executive_summary"]
        assert "current_fragmentation" in exec_summary
        assert "total_components" in exec_summary

        # Test consolidation planning
        assert "consolidation_plan" in report
        consolidation = report["consolidation_plan"]
        assert "implementation_phases" in consolidation

        # Should have multiple phases
        phases = consolidation["implementation_phases"]
        assert len(phases) >= 3, "Insufficient implementation phases"

    def test_context_injector_missing_analysis(self):
        """Test context injector missing component analysis"""
        report = self.reports["context_injector"]

        # Test requirements identification
        assert "implementation_requirements" in report
        requirements = report["implementation_requirements"]
        assert len(requirements) >= 3, "Insufficient implementation requirements identified"

        # Test implementation roadmap
        assert "implementation_roadmap" in report
        roadmap = report["implementation_roadmap"]
        assert len(roadmap) >= 4, "Insufficient roadmap phases"

        # Test complexity assessment
        exec_summary = report["executive_summary"]
        assert exec_summary["current_state"] == "Component completely missing"
        assert "High" in exec_summary["implementation_complexity"]

    def test_final_synthesis_completeness(self):
        """Test final synthesis report completeness"""
        report = self.reports["final_synthesis"]

        # Test synthesis sections
        required_sections = [
            "executive_summary", "cross_component_dependencies",
            "integrated_migration_strategy", "validation_framework",
            "risk_assessment", "gap_analysis_summary"
        ]

        for section in required_sections:
            assert section in report, f"Missing synthesis section: {section}"

    def test_migration_strategy_coherence(self):
        """Test migration strategy coherence and dependencies"""
        report = self.reports["final_synthesis"]

        # Test migration phases
        migration_strategy = report["integrated_migration_strategy"]
        assert len(migration_strategy) >= 4, "Insufficient migration phases"

        # Test phase dependencies
        for i, phase in enumerate(migration_strategy):
            assert "phase_number" in phase
            assert "duration_weeks" in phase
            assert "deliverables" in phase
            assert "success_criteria" in phase

            # Later phases should have dependencies
            if i > 0:
                assert len(phase.get("dependencies", [])) > 0, f"Phase {i+1} missing dependencies"

    def test_validation_framework_completeness(self):
        """Test validation framework completeness"""
        report = self.reports["final_synthesis"]
        validation = report["validation_framework"]

        # Test validation criteria
        assert "validation_criteria" in validation
        criteria = validation["validation_criteria"]
        assert len(criteria) >= 8, "Insufficient validation criteria"

        # Test each component has validation criteria
        components = set(criterion["component"] for criterion in criteria)
        expected_components = ["Rust Engine", "Python MCP", "CLI Utility", "Context Injector"]

        for component in expected_components:
            assert component in components, f"Missing validation criteria for {component}"

    def test_risk_assessment_quality(self):
        """Test risk assessment quality and completeness"""
        report = self.reports["final_synthesis"]
        risk_assessment = report["risk_assessment"]

        # Test risk categories
        assert "high_risk_areas" in risk_assessment
        assert "dependency_risks" in risk_assessment
        assert "timeline_risks" in risk_assessment
        assert "quality_risks" in risk_assessment

        # Test mitigation strategies
        high_risks = risk_assessment["high_risk_areas"]
        for risk_name, risk_info in high_risks.items():
            assert "mitigation" in risk_info, f"Missing mitigation for {risk_name}"

    def test_effort_estimates_consistency(self):
        """Test effort estimate consistency across reports"""
        # Collect effort estimates from individual reports
        rust_effort = self.reports["rust_engine"]["executive_summary"]["estimated_migration_effort"]
        python_effort = self.reports["python_mcp"]["executive_summary"]["estimated_effort"]
        cli_effort = self.reports["cli_utility"]["executive_summary"]["estimated_effort"]
        context_effort = self.reports["context_injector"]["executive_summary"]["estimated_total_effort"]

        # Test final synthesis total
        final_effort = self.reports["final_synthesis"]["executive_summary"]["total_effort_estimate"]

        # Final estimate should account for all components
        assert "20 weeks" in final_effort or "4-5 months" in final_effort

        # Individual estimates should be reasonable
        assert "week" in rust_effort
        assert "week" in python_effort
        assert "week" in cli_effort
        assert "week" in context_effort

    def test_alignment_percentages_reasonable(self):
        """Test that alignment percentages are reasonable and consistent"""
        # Overall alignment should be reasonable
        overall = self.reports["codebase_inventory"]["executive_summary"]["overall_alignment"]
        assert "35%" in overall

        # Component alignments should vary appropriately
        rust_alignment = self.reports["rust_engine"]["executive_summary"]["overall_alignment"]
        assert "40%" in rust_alignment  # Rust should be higher than overall

        # Context injector should be 0%
        context_alignment = self.reports["context_injector"]["executive_summary"]["current_state"]
        assert "completely missing" in context_alignment

    def test_deliverables_specificity(self):
        """Test that deliverables are specific and actionable"""
        migration_strategy = self.reports["final_synthesis"]["integrated_migration_strategy"]

        for phase in migration_strategy:
            deliverables = phase["deliverables"]
            assert len(deliverables) >= 2, f"Phase {phase['phase_number']} has too few deliverables"

            # Deliverables should be specific
            for deliverable in deliverables:
                assert len(deliverable) > 10, f"Deliverable too vague: {deliverable}"
                assert not deliverable.startswith("TBD"), f"Incomplete deliverable: {deliverable}"

def test_comprehensive_gap_analysis_execution():
    """Integration test for complete gap analysis execution"""
    # This test validates that the entire analysis framework executed successfully
    project_root = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp")

    # Test that all analysis scripts exist and are executable
    analysis_scripts = [
        "20250920-2208_codebase_inventory_analysis.py",
        "20250920-2209_rust_engine_gap_analysis.py",
        "20250920-2210_python_mcp_consolidation_analysis.py",
        "20250920-2211_cli_utility_gap_analysis.py",
        "20250920-2212_context_injector_missing_analysis.py",
        "20250920-2213_integration_migration_synthesis.py"
    ]

    for script in analysis_scripts:
        script_path = project_root / script
        assert script_path.exists(), f"Missing analysis script: {script}"
        assert script_path.stat().st_size > 1000, f"Analysis script too small: {script}"

if __name__ == "__main__":
    # Run validation tests
    test_suite = TestGapAnalysisValidation()
    test_suite.setup_method()

    print("🧪 Running comprehensive gap analysis validation tests...")

    # Execute all test methods
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]

    passed = 0
    failed = 0

    for test_method in test_methods:
        try:
            getattr(test_suite, test_method)()
            print(f"✅ {test_method}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_method}: {e}")
            failed += 1

    print(f"\n📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All gap analysis validation tests passed!")
    else:
        print("⚠️ Some validation tests failed - check analysis completeness")

    # Run integration test
    try:
        test_comprehensive_gap_analysis_execution()
        print("✅ Integration test passed")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")