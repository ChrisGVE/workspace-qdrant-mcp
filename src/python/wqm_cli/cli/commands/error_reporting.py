"""Error reporting and statistics CLI commands.

This module provides commands to view error statistics, trends, and generate
comprehensive reports from the workspace-qdrant-mcp daemon.

Usage:
    wqm errors stats                        # Show summary statistics
    wqm errors stats --days=30              # Last 30 days statistics
    wqm errors trends --days=7              # Show error trends
    wqm errors trends --granularity=hourly  # Hourly trends
    wqm errors top --limit=20               # Top 20 errors
    wqm errors report --format=json         # Generate full JSON report
    wqm errors report --format=markdown     # Generate Markdown report
"""

import asyncio
import json
from typing import Optional, Literal

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.bar import Bar
from rich.panel import Panel
from rich.text import Text

from common.core.error_statistics import (
    ErrorReportGenerator,
    SummaryReport,
    TrendReport,
    TopErrorsReport,
    ResolutionReport
)

from ..utils import (
    create_command_app,
    error_message,
    handle_async,
    success_message,
    warning_message,
)

# Create the errors app
errors_app = create_command_app(
    name="errors",
    help_text="""Error statistics and reporting.

View comprehensive error statistics, trends, and generate reports
for monitoring daemon health and error patterns.

Examples:
    wqm errors stats                        # Summary statistics
    wqm errors stats --days=30              # Last 30 days
    wqm errors trends --days=7              # Error trends
    wqm errors trends --granularity=hourly  # Hourly breakdown
    wqm errors top --limit=20               # Top 20 errors
    wqm errors report --format=json         # JSON report
    wqm errors resolution                   # Acknowledgment metrics""",
    no_args_is_help=True,
)


def get_severity_color(severity: str) -> str:
    """
    Get Rich color for severity level.

    Args:
        severity: Severity string ('error', 'warning', 'info')

    Returns:
        Rich color name
    """
    severity_colors = {
        'error': 'red',
        'warning': 'yellow',
        'info': 'blue'
    }
    return severity_colors.get(severity.lower(), 'white')


@errors_app.command("stats")
def show_statistics(
    days: int = typer.Option(
        7,
        "--days",
        help="Number of days to analyze",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json",
    ),
):
    """Show error summary statistics.

    Displays aggregate error statistics including totals by severity and category,
    error rates, and top errors.
    """
    handle_async(_show_statistics(days, format))


async def _show_statistics(days: int, format: str) -> None:
    """Implementation of show_statistics command."""
    try:
        generator = ErrorReportGenerator()
        await generator.initialize()

        # Generate summary report
        report = await generator.generate_summary_report(days=days)

        if format.lower() == "json":
            print(generator.export_report(report, format='json'))
        else:
            _display_summary_table(report)

        await generator.close()

    except Exception as e:
        error_message(f"Failed to generate statistics: {e}")
        logger.error("Error generating statistics", error=str(e), exc_info=True)
        raise typer.Exit(1)


def _display_summary_table(report: SummaryReport) -> None:
    """
    Display summary report as Rich table.

    Args:
        report: Summary report to display
    """
    console = Console()
    stats = report.statistics

    # Overview panel
    overview_text = Text()
    overview_text.append(f"Total Errors: ", style="bold")
    overview_text.append(f"{stats.total_count}\n")
    overview_text.append(f"Unacknowledged: ", style="bold")
    overview_text.append(f"{stats.unacknowledged_count}\n", style="red")
    overview_text.append(f"Error Rate: ", style="bold")
    overview_text.append(f"{stats.error_rate_per_day:.2f} per day ({stats.error_rate_per_hour:.2f} per hour)\n")

    if stats.last_error_at:
        overview_text.append(f"Last Error: ", style="bold")
        overview_text.append(f"{stats.last_error_at.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")

    console.print(Panel(
        overview_text,
        title=f"[bold]Error Statistics - Last {report.time_range_days} Days[/bold]",
        border_style="blue"
    ))

    # Severity breakdown table
    severity_table = Table(title="By Severity", show_header=True)
    severity_table.add_column("Severity", style="bold")
    severity_table.add_column("Count", justify="right")
    severity_table.add_column("Percentage", justify="right")
    severity_table.add_column("Bar")

    # Sort by severity order: error, warning, info
    severity_order = ['error', 'warning', 'info']
    for severity in severity_order:
        count = stats.by_severity.get(severity, 0)
        if count > 0:
            percentage = (count / stats.total_count) * 100 if stats.total_count > 0 else 0
            bar_width = int(percentage / 2)  # Scale to 50 chars max
            bar = "█" * bar_width

            severity_table.add_row(
                severity.upper(),
                str(count),
                f"{percentage:.1f}%",
                f"[{get_severity_color(severity)}]{bar}[/]"
            )

    console.print(severity_table)
    console.print()

    # Category breakdown table
    category_table = Table(title="By Category (Top 10)", show_header=True)
    category_table.add_column("Category", style="bold")
    category_table.add_column("Count", justify="right")
    category_table.add_column("Percentage", justify="right")

    # Sort by count descending
    sorted_categories = sorted(
        stats.by_category.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for category, count in sorted_categories:
        percentage = (count / stats.total_count) * 100 if stats.total_count > 0 else 0
        category_table.add_row(
            category,
            str(count),
            f"{percentage:.1f}%"
        )

    console.print(category_table)

    # Top errors if available
    if stats.top_errors:
        console.print()
        top_errors_table = Table(title="Top Errors", show_header=True)
        top_errors_table.add_column("#", style="cyan", width=3)
        top_errors_table.add_column("Message", style="white", no_wrap=False)
        top_errors_table.add_column("Count", justify="right", style="yellow")
        top_errors_table.add_column("Severity", style="magenta")

        for i, error in enumerate(stats.top_errors[:5], 1):
            # Truncate message for display
            message = error.message[:80] + "..." if len(error.message) > 80 else error.message
            top_errors_table.add_row(
                str(i),
                message,
                str(error.count),
                error.severity
            )

        console.print(top_errors_table)


@errors_app.command("trends")
def show_trends(
    days: int = typer.Option(
        7,
        "--days",
        help="Number of days to analyze",
    ),
    granularity: str = typer.Option(
        "daily",
        "--granularity",
        help="Time granularity: hourly, daily, or weekly",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json",
    ),
):
    """Show error trends over time.

    Displays time series analysis of errors with configurable granularity.
    """
    # Validate granularity
    valid_granularities = ['hourly', 'daily', 'weekly']
    if granularity.lower() not in valid_granularities:
        error_message(f"Invalid granularity: {granularity}. Valid options: {', '.join(valid_granularities)}")
        raise typer.Exit(1)

    handle_async(_show_trends(days, granularity.lower(), format))


async def _show_trends(days: int, granularity: str, format: str) -> None:
    """Implementation of show_trends command."""
    try:
        generator = ErrorReportGenerator()
        await generator.initialize()

        # Generate trend report
        report = await generator.generate_trend_report(
            days=days,
            granularity=granularity  # type: ignore
        )

        if format.lower() == "json":
            print(generator.export_report(report, format='json'))
        else:
            _display_trend_table(report)

        await generator.close()

    except Exception as e:
        error_message(f"Failed to generate trends: {e}")
        logger.error("Error generating trends", error=str(e), exc_info=True)
        raise typer.Exit(1)


def _display_trend_table(report: TrendReport) -> None:
    """
    Display trend report as Rich table.

    Args:
        report: Trend report to display
    """
    console = Console()

    table = Table(
        title=f"Error Trends - {report.granularity.capitalize()} over {report.time_range_days} days",
        show_header=True
    )
    table.add_column("Period", style="cyan")
    table.add_column("Total", justify="right", style="bold")
    table.add_column("Error", justify="right", style="red")
    table.add_column("Warning", justify="right", style="yellow")
    table.add_column("Info", justify="right", style="blue")
    table.add_column("Trend")

    # Find max count for scaling
    max_count = max((tp.count for tp in report.time_series), default=0)

    for tp in report.time_series:
        error_count = tp.by_severity.get('error', 0)
        warning_count = tp.by_severity.get('warning', 0)
        info_count = tp.by_severity.get('info', 0)

        # Create mini bar chart
        if max_count > 0:
            bar_width = int((tp.count / max_count) * 20)
            bar = "█" * bar_width
        else:
            bar = ""

        table.add_row(
            tp.period,
            str(tp.count),
            str(error_count),
            str(warning_count),
            str(info_count),
            bar
        )

    console.print(table)


@errors_app.command("top")
def show_top_errors(
    limit: int = typer.Option(
        10,
        "--limit",
        help="Number of top errors to show",
    ),
    days: Optional[int] = typer.Option(
        None,
        "--days",
        help="Number of days to analyze (default: all time)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json",
    ),
):
    """Show most frequent errors.

    Displays the top N most frequently occurring errors with details.
    """
    handle_async(_show_top_errors(limit, days, format))


async def _show_top_errors(limit: int, days: Optional[int], format: str) -> None:
    """Implementation of show_top_errors command."""
    try:
        generator = ErrorReportGenerator()
        await generator.initialize()

        # Generate top errors report
        report = await generator.generate_top_errors_report(
            limit=limit,
            days=days
        )

        if format.lower() == "json":
            print(generator.export_report(report, format='json'))
        else:
            _display_top_errors_table(report)

        await generator.close()

    except Exception as e:
        error_message(f"Failed to generate top errors report: {e}")
        logger.error("Error generating top errors", error=str(e), exc_info=True)
        raise typer.Exit(1)


def _display_top_errors_table(report: TopErrorsReport) -> None:
    """
    Display top errors report as Rich table.

    Args:
        report: Top errors report to display
    """
    console = Console()

    if not report.top_errors:
        console.print("[yellow]No errors found[/yellow]")
        return

    table = Table(
        title=f"Top {report.limit} Errors",
        show_header=True,
        header_style="bold"
    )
    table.add_column("Rank", style="cyan", width=4)
    table.add_column("Message", style="white", no_wrap=False, max_width=60)
    table.add_column("Count", justify="right", style="yellow")
    table.add_column("Severity", style="magenta")
    table.add_column("Category", style="blue")
    table.add_column("First", style="green")
    table.add_column("Last", style="green")

    for i, error in enumerate(report.top_errors, 1):
        # Truncate message
        message = error.message[:60] + "..." if len(error.message) > 60 else error.message

        # Format dates
        first_seen = error.first_seen.strftime("%m-%d %H:%M")
        last_seen = error.last_seen.strftime("%m-%d %H:%M")

        table.add_row(
            str(i),
            message,
            str(error.count),
            error.severity,
            error.category,
            first_seen,
            last_seen
        )

    console.print(table)


@errors_app.command("resolution")
def show_resolution_metrics(
    days: Optional[int] = typer.Option(
        None,
        "--days",
        help="Number of days to analyze (default: all time)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json",
    ),
):
    """Show error resolution and acknowledgment metrics.

    Displays statistics about error acknowledgment rates and resolution times.
    """
    handle_async(_show_resolution_metrics(days, format))


async def _show_resolution_metrics(days: Optional[int], format: str) -> None:
    """Implementation of show_resolution_metrics command."""
    try:
        generator = ErrorReportGenerator()
        await generator.initialize()

        # Generate resolution report
        report = await generator.generate_resolution_report(days=days)

        if format.lower() == "json":
            print(generator.export_report(report, format='json'))
        else:
            _display_resolution_table(report)

        await generator.close()

    except Exception as e:
        error_message(f"Failed to generate resolution report: {e}")
        logger.error("Error generating resolution report", error=str(e), exc_info=True)
        raise typer.Exit(1)


def _display_resolution_table(report: ResolutionReport) -> None:
    """
    Display resolution report as Rich table.

    Args:
        report: Resolution report to display
    """
    console = Console()
    metrics = report.metrics

    # Resolution metrics panel
    metrics_text = Text()
    metrics_text.append("Total Errors: ", style="bold")
    metrics_text.append(f"{metrics.total_errors}\n")
    metrics_text.append("Acknowledged: ", style="bold green")
    metrics_text.append(f"{metrics.acknowledged_count}\n")
    metrics_text.append("Unacknowledged: ", style="bold red")
    metrics_text.append(f"{metrics.unacknowledged_count}\n")
    metrics_text.append("Acknowledgment Rate: ", style="bold")

    # Color code acknowledgment rate
    rate_color = "green" if metrics.acknowledgment_rate >= 80 else "yellow" if metrics.acknowledgment_rate >= 50 else "red"
    metrics_text.append(f"{metrics.acknowledgment_rate:.1f}%\n", style=rate_color)

    if metrics.avg_time_to_acknowledge is not None:
        avg_hours = metrics.avg_time_to_acknowledge / 3600
        metrics_text.append("Avg Time to Acknowledge: ", style="bold")
        metrics_text.append(f"{avg_hours:.1f} hours\n")

    title = "Resolution Metrics"
    if report.time_range_days:
        title += f" - Last {report.time_range_days} Days"

    console.print(Panel(
        metrics_text,
        title=f"[bold]{title}[/bold]",
        border_style="blue"
    ))

    # Progress bar for acknowledgment rate
    if metrics.total_errors > 0:
        console.print()
        console.print("[bold]Acknowledgment Progress:[/bold]")

        # Calculate bar components
        acknowledged_width = int((metrics.acknowledged_count / metrics.total_errors) * 50)
        unacknowledged_width = 50 - acknowledged_width

        bar = (
            f"[green]{'█' * acknowledged_width}[/green]"
            f"[red]{'█' * unacknowledged_width}[/red]"
        )

        console.print(bar)
        console.print(
            f"[green]Acknowledged: {metrics.acknowledged_count}[/green]  "
            f"[red]Unacknowledged: {metrics.unacknowledged_count}[/red]"
        )


@errors_app.command("report")
def generate_full_report(
    days: int = typer.Option(
        7,
        "--days",
        help="Number of days to analyze",
    ),
    format: str = typer.Option(
        "json",
        "--format",
        help="Output format: json or markdown",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: print to stdout)",
    ),
):
    """Generate comprehensive error report.

    Generates a complete report including statistics, trends, top errors,
    and resolution metrics in JSON or Markdown format.
    """
    # Validate format
    valid_formats = ['json', 'markdown']
    if format.lower() not in valid_formats:
        error_message(f"Invalid format: {format}. Valid options: {', '.join(valid_formats)}")
        raise typer.Exit(1)

    handle_async(_generate_full_report(days, format.lower(), output))


async def _generate_full_report(days: int, format: str, output: Optional[str]) -> None:
    """Implementation of generate_full_report command."""
    try:
        generator = ErrorReportGenerator()
        await generator.initialize()

        # Generate all reports
        summary = await generator.generate_summary_report(days=days)
        trends = await generator.generate_trend_report(days=days, granularity='daily')
        top_errors = await generator.generate_top_errors_report(limit=20, days=days)
        resolution = await generator.generate_resolution_report(days=days)

        # Combine into comprehensive report
        if format == 'json':
            comprehensive_report = {
                "generated_at": summary.generated_at.isoformat(),
                "time_range_days": days,
                "summary": json.loads(generator.export_report(summary, format='json')),
                "trends": json.loads(generator.export_report(trends, format='json')),
                "top_errors": json.loads(generator.export_report(top_errors, format='json')),
                "resolution": json.loads(generator.export_report(resolution, format='json'))
            }
            report_content = json.dumps(comprehensive_report, indent=2)
        else:  # markdown
            report_parts = [
                generator.export_report(summary, format='markdown'),
                "\n---\n\n",
                generator.export_report(trends, format='markdown'),
                "\n---\n\n",
                generator.export_report(top_errors, format='markdown'),
                "\n---\n\n",
                generator.export_report(resolution, format='markdown')
            ]
            report_content = "".join(report_parts)

        # Output to file or stdout
        if output:
            with open(output, 'w') as f:
                f.write(report_content)
            success_message(f"Report written to {output}")
        else:
            print(report_content)

        await generator.close()

    except Exception as e:
        error_message(f"Failed to generate report: {e}")
        logger.error("Error generating report", error=str(e), exc_info=True)
        raise typer.Exit(1)
