"""
Performance regression detection utilities.

Provides statistical analysis and regression detection for benchmark results,
extending pytest-benchmark's built-in comparison with:
- Statistical significance testing (t-test, Mann-Whitney U test)
- Performance threshold alerts
- Historical trend tracking
- CI/CD integration helpers

Usage:
    from regression_detection import RegressionDetector

    # Load baseline and current results
    detector = RegressionDetector(baseline_path=".benchmarks/baseline.json")
    detector.load_current_results(current_path=".benchmarks/current.json")

    # Detect regressions
    regressions = detector.detect_regressions(threshold_percent=5.0)

    # Generate report
    report = detector.generate_report()
    print(report)

    # Exit with error code if regressions found (for CI/CD)
    if regressions:
        sys.exit(1)
"""

import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ComparisonResult(Enum):
    """Result of benchmark comparison."""

    IMPROVED = "improved"  # Significantly faster
    REGRESSED = "regressed"  # Significantly slower
    STABLE = "stable"  # No significant change


@dataclass
class BenchmarkComparison:
    """Comparison between baseline and current benchmark."""

    name: str
    baseline_mean: float
    current_mean: float
    percent_change: float
    result: ComparisonResult
    p_value: float | None = None
    baseline_stddev: float | None = None
    current_stddev: float | None = None


@dataclass
class RegressionReport:
    """Summary of regression detection results."""

    total_benchmarks: int
    regressions: list[BenchmarkComparison]
    improvements: list[BenchmarkComparison]
    stable: list[BenchmarkComparison]
    timestamp: datetime
    baseline_path: str
    current_path: str
    threshold_percent: float


class StatisticalTests:
    """Statistical significance tests for benchmark comparisons."""

    @staticmethod
    def welch_t_test(
        baseline_data: list[float],
        current_data: list[float],
        alpha: float = 0.05,
    ) -> tuple[bool, float]:
        """
        Perform Welch's t-test for independent samples.

        Args:
            baseline_data: Baseline timing data
            current_data: Current timing data
            alpha: Significance level (default 0.05)

        Returns:
            Tuple of (is_significant, p_value)
        """
        if len(baseline_data) < 2 or len(current_data) < 2:
            return False, 1.0

        mean1 = statistics.mean(baseline_data)
        mean2 = statistics.mean(current_data)
        var1 = statistics.variance(baseline_data)
        var2 = statistics.variance(current_data)
        n1 = len(baseline_data)
        n2 = len(current_data)

        # Calculate Welch's t-statistic
        pooled_se = ((var1 / n1) + (var2 / n2)) ** 0.5
        if pooled_se == 0:
            return False, 1.0

        t_stat = abs((mean1 - mean2) / pooled_se)

        # Degrees of freedom (Welch-Satterthwaite equation)
        numerator = ((var1 / n1) + (var2 / n2)) ** 2
        denominator = (
            ((var1 / n1) ** 2 / (n1 - 1)) + ((var2 / n2) ** 2 / (n2 - 1))
        )
        if denominator == 0:
            return False, 1.0

        df = numerator / denominator

        # Simple approximation of p-value for large df
        # For more accuracy, use scipy.stats.t.sf(t_stat, df) * 2
        # This is a conservative approximation
        if df > 30:
            # Approximate using standard normal for large df
            if t_stat > 1.96:  # ~95% confidence
                p_value = 0.05 if t_stat < 2.58 else 0.01
            else:
                p_value = 0.5
        else:
            # Conservative estimate for smaller df
            critical_values = {5: 2.571, 10: 2.228, 20: 2.086, 30: 2.042}
            closest_df = min(critical_values.keys(), key=lambda x: abs(x - df))
            critical_t = critical_values[closest_df]
            p_value = 0.05 if t_stat > critical_t else 0.5

        is_significant = p_value < alpha

        return is_significant, p_value

    @staticmethod
    def mann_whitney_u_test(
        baseline_data: list[float],
        current_data: list[float],
        alpha: float = 0.05,
    ) -> tuple[bool, float]:
        """
        Simplified Mann-Whitney U test implementation.

        This is a non-parametric test that doesn't assume normal distribution.
        More robust for benchmark data with outliers.

        Args:
            baseline_data: Baseline timing data
            current_data: Current timing data
            alpha: Significance level

        Returns:
            Tuple of (is_significant, approximate_p_value)
        """
        if len(baseline_data) < 2 or len(current_data) < 2:
            return False, 1.0

        # Combine and rank all data
        n1 = len(baseline_data)
        n2 = len(current_data)
        combined = [(x, "baseline") for x in baseline_data] + [
            (x, "current") for x in current_data
        ]
        combined.sort(key=lambda x: x[0])

        # Assign ranks (with tie handling)
        ranks = {}
        current_rank = 1
        i = 0
        while i < len(combined):
            # Find all items with the same value (ties)
            j = i
            while j < len(combined) and combined[j][0] == combined[i][0]:
                j += 1

            # Average rank for ties
            avg_rank = (current_rank + current_rank + (j - i - 1)) / 2

            # Assign average rank to all tied items
            for k in range(i, j):
                ranks[k] = avg_rank

            current_rank += j - i
            i = j

        # Sum ranks for baseline group
        baseline_rank_sum = sum(
            ranks[i] for i in range(len(combined)) if combined[i][1] == "baseline"
        )

        # Calculate U statistic
        u1 = baseline_rank_sum - (n1 * (n1 + 1)) / 2
        u2 = n1 * n2 - u1
        u_stat = min(u1, u2)

        # Calculate mean and standard deviation of U under null hypothesis
        mean_u = n1 * n2 / 2
        std_u = ((n1 * n2 * (n1 + n2 + 1)) / 12) ** 0.5

        if std_u == 0:
            return False, 1.0

        # Calculate z-score
        z_score = abs((u_stat - mean_u) / std_u)

        # Approximate p-value using standard normal distribution
        # For z > 1.96, p < 0.05; for z > 2.58, p < 0.01
        if z_score > 2.58:
            p_value = 0.01
        elif z_score > 1.96:
            p_value = 0.05
        else:
            p_value = 0.5

        is_significant = p_value < alpha

        return is_significant, p_value


class RegressionDetector:
    """
    Detect performance regressions by comparing benchmark results.

    Supports both pytest-benchmark JSON format and custom formats.
    """

    def __init__(
        self,
        baseline_path: str | None = None,
        use_statistical_test: bool = True,
        test_method: str = "welch",
    ):
        """
        Initialize regression detector.

        Args:
            baseline_path: Path to baseline benchmark results JSON
            use_statistical_test: Whether to use statistical significance testing
            test_method: Statistical test to use ('welch' or 'mann_whitney')
        """
        self.baseline_path = baseline_path
        self.baseline_data: dict[str, Any] = {}
        self.current_data: dict[str, Any] = {}
        self.use_statistical_test = use_statistical_test
        self.test_method = test_method

    def load_baseline(self, path: str | None = None) -> None:
        """Load baseline benchmark results from JSON file."""
        baseline_file = Path(path or self.baseline_path)
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline file not found: {baseline_file}")

        with open(baseline_file) as f:
            self.baseline_data = json.load(f)

    def load_current_results(self, path: str) -> None:
        """Load current benchmark results from JSON file."""
        current_file = Path(path)
        if not current_file.exists():
            raise FileNotFoundError(f"Current results file not found: {current_file}")

        with open(current_file) as f:
            self.current_data = json.load(f)

    def _parse_pytest_benchmark_format(
        self, data: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """
        Parse pytest-benchmark JSON format.

        Returns:
            Dict mapping benchmark name to stats
        """
        benchmarks = {}
        if "benchmarks" in data:
            for bench in data["benchmarks"]:
                name = bench.get("name", bench.get("fullname", "unknown"))
                stats = bench.get("stats", {})
                benchmarks[name] = {
                    "mean": stats.get("mean", 0),
                    "stddev": stats.get("stddev", 0),
                    "median": stats.get("median", 0),
                    "min": stats.get("min", 0),
                    "max": stats.get("max", 0),
                    "data": stats.get("data", []),
                }

        return benchmarks

    def detect_regressions(
        self,
        threshold_percent: float = 5.0,
        significance_level: float = 0.05,
    ) -> list[BenchmarkComparison]:
        """
        Detect performance regressions.

        Args:
            threshold_percent: Minimum percent change to consider (default 5%)
            significance_level: Statistical significance level (default 0.05)

        Returns:
            List of benchmarks that regressed
        """
        if not self.baseline_data or not self.current_data:
            raise ValueError(
                "Baseline and current data must be loaded before detection"
            )

        baseline_benchmarks = self._parse_pytest_benchmark_format(self.baseline_data)
        current_benchmarks = self._parse_pytest_benchmark_format(self.current_data)

        comparisons = []

        # Compare common benchmarks
        for name in baseline_benchmarks.keys():
            if name not in current_benchmarks:
                continue  # Skip benchmarks not in current results

            baseline = baseline_benchmarks[name]
            current = current_benchmarks[name]

            baseline_mean = baseline["mean"]
            current_mean = current["mean"]

            # Calculate percent change (positive = slower, negative = faster)
            if baseline_mean == 0:
                continue

            percent_change = ((current_mean - baseline_mean) / baseline_mean) * 100

            # Determine if change is statistically significant
            is_significant = False
            p_value = None

            if self.use_statistical_test and baseline["data"] and current["data"]:
                if self.test_method == "welch":
                    is_significant, p_value = StatisticalTests.welch_t_test(
                        baseline["data"], current["data"], significance_level
                    )
                elif self.test_method == "mann_whitney":
                    is_significant, p_value = StatisticalTests.mann_whitney_u_test(
                        baseline["data"], current["data"], significance_level
                    )

            # Determine result
            if abs(percent_change) < threshold_percent:
                result = ComparisonResult.STABLE
            elif percent_change > 0:
                # Slower is regression
                if self.use_statistical_test:
                    result = (
                        ComparisonResult.REGRESSED
                        if is_significant
                        else ComparisonResult.STABLE
                    )
                else:
                    result = ComparisonResult.REGRESSED
            else:
                # Faster is improvement
                if self.use_statistical_test:
                    result = (
                        ComparisonResult.IMPROVED
                        if is_significant
                        else ComparisonResult.STABLE
                    )
                else:
                    result = ComparisonResult.IMPROVED

            comparison = BenchmarkComparison(
                name=name,
                baseline_mean=baseline_mean,
                current_mean=current_mean,
                percent_change=percent_change,
                result=result,
                p_value=p_value,
                baseline_stddev=baseline.get("stddev"),
                current_stddev=current.get("stddev"),
            )

            comparisons.append(comparison)

        # Filter for regressions only
        regressions = [c for c in comparisons if c.result == ComparisonResult.REGRESSED]

        return regressions

    def generate_report(
        self,
        threshold_percent: float = 5.0,
        significance_level: float = 0.05,
    ) -> RegressionReport:
        """
        Generate comprehensive regression report.

        Args:
            threshold_percent: Minimum percent change threshold
            significance_level: Statistical significance level

        Returns:
            RegressionReport with all comparison results
        """
        if not self.baseline_data or not self.current_data:
            raise ValueError("Baseline and current data must be loaded")

        baseline_benchmarks = self._parse_pytest_benchmark_format(self.baseline_data)
        current_benchmarks = self._parse_pytest_benchmark_format(self.current_data)

        regressions = []
        improvements = []
        stable = []

        # Compare all benchmarks
        for name in baseline_benchmarks.keys():
            if name not in current_benchmarks:
                continue

            baseline = baseline_benchmarks[name]
            current = current_benchmarks[name]

            baseline_mean = baseline["mean"]
            current_mean = current["mean"]

            if baseline_mean == 0:
                continue

            percent_change = ((current_mean - baseline_mean) / baseline_mean) * 100

            # Statistical significance
            is_significant = False
            p_value = None

            if self.use_statistical_test and baseline["data"] and current["data"]:
                if self.test_method == "welch":
                    is_significant, p_value = StatisticalTests.welch_t_test(
                        baseline["data"], current["data"], significance_level
                    )
                elif self.test_method == "mann_whitney":
                    is_significant, p_value = StatisticalTests.mann_whitney_u_test(
                        baseline["data"], current["data"], significance_level
                    )

            # Categorize result
            if abs(percent_change) < threshold_percent:
                result = ComparisonResult.STABLE
            elif percent_change > 0:
                if self.use_statistical_test:
                    result = (
                        ComparisonResult.REGRESSED
                        if is_significant
                        else ComparisonResult.STABLE
                    )
                else:
                    result = ComparisonResult.REGRESSED
            else:
                if self.use_statistical_test:
                    result = (
                        ComparisonResult.IMPROVED
                        if is_significant
                        else ComparisonResult.STABLE
                    )
                else:
                    result = ComparisonResult.IMPROVED

            comparison = BenchmarkComparison(
                name=name,
                baseline_mean=baseline_mean,
                current_mean=current_mean,
                percent_change=percent_change,
                result=result,
                p_value=p_value,
                baseline_stddev=baseline.get("stddev"),
                current_stddev=current.get("stddev"),
            )

            if result == ComparisonResult.REGRESSED:
                regressions.append(comparison)
            elif result == ComparisonResult.IMPROVED:
                improvements.append(comparison)
            else:
                stable.append(comparison)

        total_benchmarks = len(regressions) + len(improvements) + len(stable)

        report = RegressionReport(
            total_benchmarks=total_benchmarks,
            regressions=regressions,
            improvements=improvements,
            stable=stable,
            timestamp=datetime.now(),
            baseline_path=self.baseline_path or "unknown",
            current_path="current",
            threshold_percent=threshold_percent,
        )

        return report

    def print_report(
        self,
        report: RegressionReport,
        show_stable: bool = False,
        show_improvements: bool = True,
    ) -> None:
        """
        Print formatted regression report.

        Args:
            report: RegressionReport to print
            show_stable: Whether to show stable benchmarks
            show_improvements: Whether to show improvements
        """
        print("\n" + "=" * 80)
        print("PERFORMANCE REGRESSION DETECTION REPORT")
        print("=" * 80)
        print(f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Baseline: {report.baseline_path}")
        print(f"Threshold: {report.threshold_percent}%")
        print(f"Total Benchmarks: {report.total_benchmarks}")
        print()

        # Summary
        print(f"Regressions: {len(report.regressions)}")
        print(f"Improvements: {len(report.improvements)}")
        print(f"Stable: {len(report.stable)}")
        print()

        # Regressions
        if report.regressions:
            print("REGRESSIONS (slower than baseline):")
            print("-" * 80)
            for comp in sorted(
                report.regressions, key=lambda x: abs(x.percent_change), reverse=True
            ):
                print(f"  {comp.name}")
                print(f"    Baseline: {comp.baseline_mean*1000:.2f} ms")
                print(f"    Current:  {comp.current_mean*1000:.2f} ms")
                print(f"    Change:   {comp.percent_change:+.2f}%")
                if comp.p_value is not None:
                    print(f"    P-value:  {comp.p_value:.4f}")
                print()

        # Improvements
        if show_improvements and report.improvements:
            print("IMPROVEMENTS (faster than baseline):")
            print("-" * 80)
            for comp in sorted(
                report.improvements, key=lambda x: abs(x.percent_change), reverse=True
            ):
                print(f"  {comp.name}")
                print(f"    Baseline: {comp.baseline_mean*1000:.2f} ms")
                print(f"    Current:  {comp.current_mean*1000:.2f} ms")
                print(f"    Change:   {comp.percent_change:+.2f}%")
                if comp.p_value is not None:
                    print(f"    P-value:  {comp.p_value:.4f}")
                print()

        # Stable
        if show_stable and report.stable:
            print("STABLE (no significant change):")
            print("-" * 80)
            for comp in report.stable:
                print(f"  {comp.name}: {comp.percent_change:+.2f}%")
            print()

        print("=" * 80)

    def export_report_json(
        self, report: RegressionReport, output_path: str
    ) -> None:
        """
        Export regression report to JSON file.

        Args:
            report: RegressionReport to export
            output_path: Path to output JSON file
        """
        report_data = {
            "timestamp": report.timestamp.isoformat(),
            "baseline_path": report.baseline_path,
            "current_path": report.current_path,
            "threshold_percent": report.threshold_percent,
            "total_benchmarks": report.total_benchmarks,
            "summary": {
                "regressions": len(report.regressions),
                "improvements": len(report.improvements),
                "stable": len(report.stable),
            },
            "regressions": [
                {
                    "name": comp.name,
                    "baseline_mean": comp.baseline_mean,
                    "current_mean": comp.current_mean,
                    "percent_change": comp.percent_change,
                    "p_value": comp.p_value,
                }
                for comp in report.regressions
            ],
            "improvements": [
                {
                    "name": comp.name,
                    "baseline_mean": comp.baseline_mean,
                    "current_mean": comp.current_mean,
                    "percent_change": comp.percent_change,
                    "p_value": comp.p_value,
                }
                for comp in report.improvements
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)


def main():
    """CLI entry point for regression detection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect performance regressions in benchmark results"
    )
    parser.add_argument(
        "--baseline", required=True, help="Path to baseline benchmark JSON"
    )
    parser.add_argument(
        "--current", required=True, help="Path to current benchmark JSON"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Regression threshold percentage (default: 5.0)",
    )
    parser.add_argument(
        "--significance",
        type=float,
        default=0.05,
        help="Statistical significance level (default: 0.05)",
    )
    parser.add_argument(
        "--test-method",
        choices=["welch", "mann_whitney"],
        default="welch",
        help="Statistical test method (default: welch)",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable statistical significance testing",
    )
    parser.add_argument(
        "--export", help="Export report to JSON file"
    )
    parser.add_argument(
        "--show-stable",
        action="store_true",
        help="Show stable benchmarks in report",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with error code if regressions detected",
    )

    args = parser.parse_args()

    # Initialize detector
    detector = RegressionDetector(
        baseline_path=args.baseline,
        use_statistical_test=not args.no_stats,
        test_method=args.test_method,
    )

    try:
        # Load data
        detector.load_baseline()
        detector.load_current_results(args.current)

        # Generate report
        report = detector.generate_report(
            threshold_percent=args.threshold,
            significance_level=args.significance,
        )

        # Print report
        detector.print_report(
            report,
            show_stable=args.show_stable,
            show_improvements=True,
        )

        # Export if requested
        if args.export:
            detector.export_report_json(report, args.export)
            print(f"\nReport exported to: {args.export}")

        # Exit with error if regressions found
        if args.fail_on_regression and report.regressions:
            print(f"\nERROR: {len(report.regressions)} regression(s) detected!")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
