"""
Error Statistics and Reporting System

Provides comprehensive error statistics, trend analysis, and report generation
for monitoring error patterns and resolution metrics.

Features:
    - Detailed statistics with time series data
    - Trend analysis over configurable time periods
    - Top error identification
    - Acknowledgment and resolution metrics
    - Multiple export formats (JSON, Markdown)

Example:
    ```python
    from workspace_qdrant_mcp.core.error_statistics import ErrorReportGenerator

    # Initialize generator
    generator = ErrorReportGenerator()
    await generator.initialize()

    # Generate summary report
    summary = await generator.generate_summary_report(days=7)
    print(f"Total errors: {summary.total_count}")

    # Generate trend report
    trends = await generator.generate_trend_report(days=7, granularity='daily')
    for trend in trends.time_series:
        print(f"{trend.period}: {trend.count} errors")

    # Export report
    json_report = generator.export_report(summary, format='json')
    ```
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Literal

from loguru import logger

from .error_message_manager import ErrorMessageManager, ErrorStatistics


@dataclass
class TimePeriodData:
    """
    Data for a specific time period in trend analysis.

    Attributes:
        period: Time period identifier (ISO timestamp or date string)
        count: Total error count in this period
        by_severity: Breakdown by severity level
        by_category: Breakdown by category
    """
    period: str
    count: int
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)


@dataclass
class TopError:
    """
    Information about a frequently occurring error.

    Attributes:
        message: Error message
        count: Number of occurrences
        category: Error category
        severity: Error severity
        first_seen: When first occurred
        last_seen: When last occurred
        sample_context: Sample context from one occurrence
    """
    message: str
    count: int
    category: str
    severity: str
    first_seen: datetime
    last_seen: datetime
    sample_context: Optional[Dict[str, Any]] = None


@dataclass
class ResolutionMetrics:
    """
    Metrics about error acknowledgment and resolution.

    Attributes:
        total_errors: Total number of errors
        acknowledged_count: Number of acknowledged errors
        acknowledgment_rate: Percentage of errors acknowledged
        avg_time_to_acknowledge: Average time to acknowledgment (seconds)
        unacknowledged_count: Number of unacknowledged errors
    """
    total_errors: int
    acknowledged_count: int
    acknowledgment_rate: float
    avg_time_to_acknowledge: Optional[float] = None
    unacknowledged_count: int = 0


@dataclass
class DetailedErrorStatistics(ErrorStatistics):
    """
    Extended error statistics with additional analysis.

    Attributes:
        time_range_start: Start of analysis period
        time_range_end: End of analysis period
        error_rate_per_hour: Average errors per hour
        error_rate_per_day: Average errors per day
        top_errors: Most frequent error messages
    """
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    error_rate_per_hour: float = 0.0
    error_rate_per_day: float = 0.0
    top_errors: List[TopError] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary with all statistics
        """
        base_dict = super().to_dict()
        base_dict.update({
            "time_range_start": self.time_range_start.isoformat() if self.time_range_start else None,
            "time_range_end": self.time_range_end.isoformat() if self.time_range_end else None,
            "error_rate_per_hour": self.error_rate_per_hour,
            "error_rate_per_day": self.error_rate_per_day,
            "top_errors": [
                {
                    "message": e.message,
                    "count": e.count,
                    "category": e.category,
                    "severity": e.severity,
                    "first_seen": e.first_seen.isoformat(),
                    "last_seen": e.last_seen.isoformat(),
                    "sample_context": e.sample_context
                }
                for e in self.top_errors
            ]
        })
        return base_dict


@dataclass
class SummaryReport:
    """
    Summary report with aggregate statistics.

    Attributes:
        statistics: Detailed error statistics
        time_range_days: Number of days in analysis period
        generated_at: When report was generated
    """
    statistics: DetailedErrorStatistics
    time_range_days: int
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TrendReport:
    """
    Trend analysis report with time series data.

    Attributes:
        time_series: List of time period data points
        granularity: Time granularity ('hourly', 'daily', 'weekly')
        time_range_days: Number of days in analysis period
        generated_at: When report was generated
    """
    time_series: List[TimePeriodData]
    granularity: Literal['hourly', 'daily', 'weekly']
    time_range_days: int
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TopErrorsReport:
    """
    Report of most frequent errors.

    Attributes:
        top_errors: List of top errors sorted by frequency
        limit: Maximum number of errors in report
        generated_at: When report was generated
    """
    top_errors: List[TopError]
    limit: int
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ResolutionReport:
    """
    Report on error acknowledgment and resolution.

    Attributes:
        metrics: Resolution metrics
        time_range_days: Number of days in analysis period
        generated_at: When report was generated
    """
    metrics: ResolutionMetrics
    time_range_days: Optional[int] = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorReportGenerator:
    """
    Generator for comprehensive error reports and statistics.

    Provides methods to generate various types of reports including summary
    statistics, trend analysis, top errors, and resolution metrics.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        error_manager: Optional[ErrorMessageManager] = None
    ):
        """
        Initialize error report generator.

        Args:
            db_path: Optional custom database path
            error_manager: Optional pre-initialized ErrorMessageManager
        """
        self.error_manager = error_manager or ErrorMessageManager(db_path=db_path)
        self._initialized = False

    async def initialize(self):
        """Initialize the error report generator."""
        if self._initialized:
            return

        await self.error_manager.initialize()
        self._initialized = True
        logger.info("Error report generator initialized")

    async def close(self):
        """Close the error report generator."""
        if not self._initialized:
            return

        await self.error_manager.close()
        self._initialized = False
        logger.info("Error report generator closed")

    async def generate_summary_report(
        self,
        days: int = 7
    ) -> SummaryReport:
        """
        Generate summary report with aggregate statistics.

        Args:
            days: Number of days to analyze (default: 7)

        Returns:
            SummaryReport with detailed statistics
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        end_date = datetime.now(timezone.utc)

        # Get base statistics
        base_stats = await self.error_manager.get_error_stats(
            start_date=start_date,
            end_date=end_date
        )

        # Calculate error rates
        hours = days * 24
        error_rate_per_hour = base_stats.total_count / hours if hours > 0 else 0.0
        error_rate_per_day = base_stats.total_count / days if days > 0 else 0.0

        # Get top errors
        top_errors = await self._get_top_errors(
            start_date=start_date,
            end_date=end_date,
            limit=10
        )

        # Create detailed statistics
        detailed_stats = DetailedErrorStatistics(
            total_count=base_stats.total_count,
            by_severity=base_stats.by_severity,
            by_category=base_stats.by_category,
            unacknowledged_count=base_stats.unacknowledged_count,
            last_error_at=base_stats.last_error_at,
            time_range_start=start_date,
            time_range_end=end_date,
            error_rate_per_hour=error_rate_per_hour,
            error_rate_per_day=error_rate_per_day,
            top_errors=top_errors
        )

        return SummaryReport(
            statistics=detailed_stats,
            time_range_days=days
        )

    async def generate_trend_report(
        self,
        days: int = 7,
        granularity: Literal['hourly', 'daily', 'weekly'] = 'daily'
    ) -> TrendReport:
        """
        Generate trend analysis report with time series data.

        Args:
            days: Number of days to analyze (default: 7)
            granularity: Time granularity for analysis (default: 'daily')

        Returns:
            TrendReport with time series data
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        end_date = datetime.now(timezone.utc)

        # Determine time period duration
        if granularity == 'hourly':
            period_delta = timedelta(hours=1)
            periods = days * 24
        elif granularity == 'daily':
            period_delta = timedelta(days=1)
            periods = days
        elif granularity == 'weekly':
            period_delta = timedelta(weeks=1)
            periods = (days + 6) // 7  # Round up to weeks
        else:
            raise ValueError(f"Invalid granularity: {granularity}")

        # Generate time series data
        time_series: List[TimePeriodData] = []
        current_start = start_date

        for _ in range(periods):
            current_end = min(current_start + period_delta, end_date)

            # Get statistics for this period
            period_stats = await self.error_manager.get_error_stats(
                start_date=current_start,
                end_date=current_end
            )

            # Format period identifier
            if granularity == 'hourly':
                period_id = current_start.strftime("%Y-%m-%d %H:00")
            elif granularity == 'daily':
                period_id = current_start.strftime("%Y-%m-%d")
            else:  # weekly
                period_id = current_start.strftime("%Y-W%W")

            time_series.append(TimePeriodData(
                period=period_id,
                count=period_stats.total_count,
                by_severity=period_stats.by_severity,
                by_category=period_stats.by_category
            ))

            current_start = current_end

            if current_start >= end_date:
                break

        return TrendReport(
            time_series=time_series,
            granularity=granularity,
            time_range_days=days
        )

    async def generate_top_errors_report(
        self,
        limit: int = 10,
        days: Optional[int] = None
    ) -> TopErrorsReport:
        """
        Generate report of most frequent errors.

        Args:
            limit: Maximum number of errors to include (default: 10)
            days: Optional number of days to analyze (None = all time)

        Returns:
            TopErrorsReport with most frequent errors
        """
        start_date = None
        if days:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)

        top_errors = await self._get_top_errors(
            start_date=start_date,
            limit=limit
        )

        return TopErrorsReport(
            top_errors=top_errors,
            limit=limit
        )

    async def generate_resolution_report(
        self,
        days: Optional[int] = None
    ) -> ResolutionReport:
        """
        Generate report on error acknowledgment and resolution.

        Args:
            days: Optional number of days to analyze (None = all time)

        Returns:
            ResolutionReport with acknowledgment metrics
        """
        start_date = None
        end_date = None

        if days:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            end_date = datetime.now(timezone.utc)

        # Get overall statistics
        stats = await self.error_manager.get_error_stats(
            start_date=start_date,
            end_date=end_date
        )

        # Calculate acknowledgment metrics
        acknowledgment_rate = 0.0
        if stats.total_count > 0:
            acknowledged_count = stats.total_count - stats.unacknowledged_count
            acknowledgment_rate = (acknowledged_count / stats.total_count) * 100
        else:
            acknowledged_count = 0

        # Calculate average time to acknowledgment
        avg_time = await self._calculate_avg_acknowledgment_time(
            start_date=start_date,
            end_date=end_date
        )

        metrics = ResolutionMetrics(
            total_errors=stats.total_count,
            acknowledged_count=acknowledged_count,
            acknowledgment_rate=acknowledgment_rate,
            avg_time_to_acknowledge=avg_time,
            unacknowledged_count=stats.unacknowledged_count
        )

        return ResolutionReport(
            metrics=metrics,
            time_range_days=days
        )

    def export_report(
        self,
        report: Any,
        format: Literal['json', 'markdown'] = 'json'
    ) -> str:
        """
        Export report to specified format.

        Args:
            report: Report object to export
            format: Output format ('json' or 'markdown')

        Returns:
            Formatted report string
        """
        if format == 'json':
            return self._export_json(report)
        elif format == 'markdown':
            return self._export_markdown(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self, report: Any) -> str:
        """
        Export report as JSON.

        Args:
            report: Report object to export

        Returns:
            JSON string
        """
        # Convert dataclass to dict
        if isinstance(report, SummaryReport):
            data = {
                "type": "summary",
                "statistics": report.statistics.to_dict(),
                "time_range_days": report.time_range_days,
                "generated_at": report.generated_at.isoformat()
            }
        elif isinstance(report, TrendReport):
            data = {
                "type": "trend",
                "time_series": [
                    {
                        "period": tp.period,
                        "count": tp.count,
                        "by_severity": tp.by_severity,
                        "by_category": tp.by_category
                    }
                    for tp in report.time_series
                ],
                "granularity": report.granularity,
                "time_range_days": report.time_range_days,
                "generated_at": report.generated_at.isoformat()
            }
        elif isinstance(report, TopErrorsReport):
            data = {
                "type": "top_errors",
                "top_errors": [
                    {
                        "message": e.message,
                        "count": e.count,
                        "category": e.category,
                        "severity": e.severity,
                        "first_seen": e.first_seen.isoformat(),
                        "last_seen": e.last_seen.isoformat(),
                        "sample_context": e.sample_context
                    }
                    for e in report.top_errors
                ],
                "limit": report.limit,
                "generated_at": report.generated_at.isoformat()
            }
        elif isinstance(report, ResolutionReport):
            data = {
                "type": "resolution",
                "metrics": {
                    "total_errors": report.metrics.total_errors,
                    "acknowledged_count": report.metrics.acknowledged_count,
                    "acknowledgment_rate": report.metrics.acknowledgment_rate,
                    "avg_time_to_acknowledge": report.metrics.avg_time_to_acknowledge,
                    "unacknowledged_count": report.metrics.unacknowledged_count
                },
                "time_range_days": report.time_range_days,
                "generated_at": report.generated_at.isoformat()
            }
        else:
            raise ValueError(f"Unsupported report type: {type(report)}")

        return json.dumps(data, indent=2)

    def _export_markdown(self, report: Any) -> str:
        """
        Export report as Markdown.

        Args:
            report: Report object to export

        Returns:
            Markdown string
        """
        if isinstance(report, SummaryReport):
            return self._format_summary_markdown(report)
        elif isinstance(report, TrendReport):
            return self._format_trend_markdown(report)
        elif isinstance(report, TopErrorsReport):
            return self._format_top_errors_markdown(report)
        elif isinstance(report, ResolutionReport):
            return self._format_resolution_markdown(report)
        else:
            raise ValueError(f"Unsupported report type: {type(report)}")

    def _format_summary_markdown(self, report: SummaryReport) -> str:
        """Format summary report as Markdown."""
        lines = [
            f"# Error Summary Report",
            f"",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"**Time Range:** Last {report.time_range_days} days",
            f"",
            f"## Overview",
            f"",
            f"- **Total Errors:** {report.statistics.total_count}",
            f"- **Unacknowledged:** {report.statistics.unacknowledged_count}",
            f"- **Error Rate:** {report.statistics.error_rate_per_day:.2f} per day ({report.statistics.error_rate_per_hour:.2f} per hour)",
            f"",
            f"## By Severity",
            f""
        ]

        for severity, count in sorted(report.statistics.by_severity.items()):
            lines.append(f"- **{severity}:** {count}")

        lines.extend([
            f"",
            f"## By Category",
            f""
        ])

        for category, count in sorted(report.statistics.by_category.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{category}:** {count}")

        if report.statistics.top_errors:
            lines.extend([
                f"",
                f"## Top Errors",
                f""
            ])
            for i, error in enumerate(report.statistics.top_errors, 1):
                lines.extend([
                    f"### {i}. {error.message[:80]}",
                    f"",
                    f"- **Count:** {error.count}",
                    f"- **Severity:** {error.severity}",
                    f"- **Category:** {error.category}",
                    f"- **First Seen:** {error.first_seen.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"- **Last Seen:** {error.last_seen.strftime('%Y-%m-%d %H:%M:%S')}",
                    f""
                ])

        return "\n".join(lines)

    def _format_trend_markdown(self, report: TrendReport) -> str:
        """Format trend report as Markdown."""
        lines = [
            f"# Error Trend Report",
            f"",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"**Time Range:** Last {report.time_range_days} days",
            f"**Granularity:** {report.granularity}",
            f"",
            f"## Time Series",
            f"",
            f"| Period | Total | Error | Warning | Info |",
            f"|--------|-------|-------|---------|------|"
        ]

        for tp in report.time_series:
            error_count = tp.by_severity.get('error', 0)
            warning_count = tp.by_severity.get('warning', 0)
            info_count = tp.by_severity.get('info', 0)
            lines.append(f"| {tp.period} | {tp.count} | {error_count} | {warning_count} | {info_count} |")

        return "\n".join(lines)

    def _format_top_errors_markdown(self, report: TopErrorsReport) -> str:
        """Format top errors report as Markdown."""
        lines = [
            f"# Top {report.limit} Errors Report",
            f"",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f""
        ]

        for i, error in enumerate(report.top_errors, 1):
            lines.extend([
                f"## {i}. {error.message[:80]}",
                f"",
                f"- **Count:** {error.count}",
                f"- **Severity:** {error.severity}",
                f"- **Category:** {error.category}",
                f"- **First Seen:** {error.first_seen.strftime('%Y-%m-%d %H:%M:%S')}",
                f"- **Last Seen:** {error.last_seen.strftime('%Y-%m-%d %H:%M:%S')}",
                f""
            ])

        return "\n".join(lines)

    def _format_resolution_markdown(self, report: ResolutionReport) -> str:
        """Format resolution report as Markdown."""
        lines = [
            f"# Error Resolution Report",
            f"",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        ]

        if report.time_range_days:
            lines.append(f"**Time Range:** Last {report.time_range_days} days")

        lines.extend([
            f"",
            f"## Resolution Metrics",
            f"",
            f"- **Total Errors:** {report.metrics.total_errors}",
            f"- **Acknowledged:** {report.metrics.acknowledged_count}",
            f"- **Unacknowledged:** {report.metrics.unacknowledged_count}",
            f"- **Acknowledgment Rate:** {report.metrics.acknowledgment_rate:.1f}%"
        ])

        if report.metrics.avg_time_to_acknowledge is not None:
            avg_hours = report.metrics.avg_time_to_acknowledge / 3600
            lines.append(f"- **Avg Time to Acknowledge:** {avg_hours:.1f} hours")

        return "\n".join(lines)

    async def _get_top_errors(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10
    ) -> List[TopError]:
        """
        Get top N most frequent errors.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of errors to return

        Returns:
            List of TopError objects
        """
        # Build query
        query = """
            SELECT
                message,
                category,
                severity,
                COUNT(*) as count,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen,
                context
            FROM messages
        """

        conditions = []
        params = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY message, category, severity ORDER BY count DESC LIMIT ?"
        params.append(limit)

        async with self.error_manager.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()

            top_errors = []
            for row in rows:
                # Parse context if available
                sample_context = None
                if row["context"]:
                    try:
                        sample_context = json.loads(row["context"])
                    except json.JSONDecodeError:
                        pass

                top_errors.append(TopError(
                    message=row["message"],
                    count=row["count"],
                    category=row["category"],
                    severity=row["severity"],
                    first_seen=datetime.fromisoformat(row["first_seen"]),
                    last_seen=datetime.fromisoformat(row["last_seen"]),
                    sample_context=sample_context
                ))

            return top_errors

    async def _calculate_avg_acknowledgment_time(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[float]:
        """
        Calculate average time to acknowledgment in seconds.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Average acknowledgment time in seconds, or None if no data
        """
        query = """
            SELECT
                AVG(
                    CAST(
                        (julianday(acknowledged_at) - julianday(timestamp)) * 86400
                        AS REAL
                    )
                ) as avg_seconds
            FROM messages
            WHERE acknowledged = 1
            AND acknowledged_at IS NOT NULL
        """

        conditions = []
        params = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())

        if conditions:
            query += " AND " + " AND ".join(conditions)

        async with self.error_manager.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, tuple(params))
            row = cursor.fetchone()

            if row and row["avg_seconds"] is not None:
                return float(row["avg_seconds"])

            return None
