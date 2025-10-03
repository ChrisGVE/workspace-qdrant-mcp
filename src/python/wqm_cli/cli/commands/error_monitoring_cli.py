"""Error monitoring CLI commands.

This module provides commands to view health status, display current metrics,
and start monitoring with webhook alerts.

Usage:
    wqm errors health                         # Show error-based health status
    wqm errors health --format=json           # Health status as JSON
    wqm errors metrics                        # Display current error metrics
    wqm errors metrics --format=json          # Metrics as JSON
    wqm errors monitor --webhook=URL          # Start monitoring with webhook alerts
"""

import asyncio
import json
import signal
from datetime import datetime
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from common.core.error_message_manager import ErrorMessageManager
from common.core.error_monitoring import (
    ErrorMetricsCollector,
    HealthCheckManager,
    HealthStatus,
    LoggingHook,
    WebhookHook,
)

from ..utils import (
    create_command_app,
    error_message,
    handle_async,
    success_message,
    warning_message,
)

# Create the monitoring commands app (to be registered under errors_app)
monitoring_app = create_command_app(
    name="monitoring",
    help_text="""Error monitoring commands.

View health status, display metrics, and start monitoring with webhook alerts.

Examples:
    wqm errors health                         # Health status
    wqm errors metrics                        # Current metrics
    wqm errors monitor --webhook=URL          # Start monitoring""",
    no_args_is_help=False,
)

console = Console()


@monitoring_app.command("health")
def show_health_status(
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table or json",
    ),
):
    """Show error-based health status.

    Displays overall health status based on error rates, acknowledgment
    rates, and custom health checks.
    """
    handle_async(_show_health_status(format))


async def _show_health_status(format: str) -> None:
    """Implementation of show_health_status command."""
    try:
        # Initialize error manager
        error_manager = ErrorMessageManager()
        await error_manager.initialize()

        # Initialize health check manager
        health_manager = HealthCheckManager(error_manager)
        await health_manager.initialize()

        # Get health status
        status = await health_manager.get_health_status()

        if format.lower() == "json":
            # JSON output
            print(json.dumps(status.to_dict(), indent=2))
        else:
            # Table output
            _display_health_status_table(status)

        # Cleanup
        await health_manager.close()
        await error_manager.close()

    except Exception as e:
        error_message(f"Failed to get health status: {e}")
        logger.error("Error getting health status", error=str(e), exc_info=True)
        raise typer.Exit(1)


def _display_health_status_table(status) -> None:
    """Display health status as formatted table."""
    # Determine status color
    status_colors = {
        HealthStatus.HEALTHY: "green",
        HealthStatus.DEGRADED: "yellow",
        HealthStatus.UNHEALTHY: "red"
    }
    status_color = status_colors.get(status.status, "white")

    # Create main status panel
    status_text = Text(status.status.value.upper(), style=f"bold {status_color}")
    panel = Panel(status_text, title="Overall Health Status", expand=False)
    console.print(panel)
    console.print()

    # Create checks table
    table = Table(title="Health Checks", show_header=True, header_style="bold cyan")
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")

    for check_name, check_status in status.checks.items():
        check_color = status_colors.get(check_status, "white")
        table.add_row(
            check_name,
            Text(check_status.value, style=check_color)
        )

    console.print(table)
    console.print()

    # Display details if available
    if status.details:
        console.print("[bold]Details:[/bold]")
        for key, value in status.details.items():
            if key == "error_stats":
                # Format error stats nicely
                if isinstance(value, dict):
                    console.print(f"  [cyan]{key}:[/cyan]")
                    for stat_key, stat_value in value.items():
                        console.print(f"    {stat_key}: {stat_value}")
            else:
                console.print(f"  [cyan]{key}:[/cyan] {value}")


@monitoring_app.command("metrics")
def show_metrics(
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table or json",
    ),
):
    """Display current error metrics.

    Shows error counts, error rates, acknowledgment metrics, and other
    monitoring statistics.
    """
    handle_async(_show_metrics(format))


async def _show_metrics(format: str) -> None:
    """Implementation of show_metrics command."""
    try:
        # Initialize error manager
        error_manager = ErrorMessageManager()
        await error_manager.initialize()

        # Initialize metrics collector
        collector = ErrorMetricsCollector()
        collector.register_hook(LoggingHook())

        # Get current metrics
        metrics = collector.get_current_metrics()

        # Get error stats for additional context
        stats = await error_manager.get_error_stats()

        # Combine metrics
        combined_metrics = {
            "current_metrics": metrics,
            "error_statistics": stats.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

        if format.lower() == "json":
            # JSON output
            print(json.dumps(combined_metrics, indent=2))
        else:
            # Table output
            _display_metrics_table(metrics, stats)

        # Cleanup
        await collector.close()
        await error_manager.close()

    except Exception as e:
        error_message(f"Failed to get metrics: {e}")
        logger.error("Error getting metrics", error=str(e), exc_info=True)
        raise typer.Exit(1)


def _display_metrics_table(metrics: dict, stats) -> None:
    """Display metrics as formatted table."""
    # Error totals table
    if metrics.get("error_total"):
        table = Table(title="Error Counts", show_header=True, header_style="bold cyan")
        table.add_column("Severity_Category", style="cyan")
        table.add_column("Count", justify="right", style="bold")

        for key, count in metrics["error_total"].items():
            table.add_row(key, str(count))

        console.print(table)
        console.print()

    # Error rate
    if "error_rate" in metrics:
        console.print(f"[bold]Error Rate:[/bold] {metrics['error_rate']:.2f} errors/minute")
        console.print()

    # Acknowledgment metrics
    if metrics.get("acknowledgment_times"):
        ack_metrics = metrics["acknowledgment_times"]
        table = Table(
            title="Acknowledgment Metrics",
            show_header=True,
            header_style="bold cyan"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="bold")

        table.add_row("Count", str(ack_metrics.get("count", 0)))
        table.add_row("Average (seconds)", f"{ack_metrics.get('avg', 0):.2f}")
        table.add_row("Min (seconds)", f"{ack_metrics.get('min', 0):.2f}")
        table.add_row("Max (seconds)", f"{ack_metrics.get('max', 0):.2f}")

        console.print(table)
        console.print()

    # Error statistics
    table = Table(
        title="Error Statistics",
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", justify="right", style="bold")

    table.add_row("Total Errors", str(stats.total_count))
    table.add_row("Unacknowledged", str(stats.unacknowledged_count))

    if stats.last_error_at:
        table.add_row("Last Error", stats.last_error_at.strftime("%Y-%m-%d %H:%M:%S"))

    console.print(table)


@monitoring_app.command("monitor")
def start_monitoring(
    webhook: str = typer.Option(
        ...,
        "--webhook",
        "-w",
        help="Webhook URL for alerts",
    ),
    interval: int = typer.Option(
        60,
        "--interval",
        "-i",
        help="Monitoring interval in seconds",
    ),
):
    """Start monitoring with webhook alerts.

    Continuously monitors error metrics and sends alerts to the configured
    webhook endpoint when issues are detected.

    The monitor runs until interrupted (Ctrl+C).
    """
    handle_async(_start_monitoring(webhook, interval))


async def _start_monitoring(webhook: str, interval: int) -> None:
    """Implementation of start_monitoring command."""
    try:
        # Initialize error manager
        error_manager = ErrorMessageManager()
        await error_manager.initialize()

        # Initialize metrics collector
        collector = ErrorMetricsCollector()
        collector.register_hook(LoggingHook())
        collector.register_hook(WebhookHook(webhook))

        # Initialize health check manager
        health_manager = HealthCheckManager(error_manager)
        await health_manager.initialize()

        console.print(f"[green]Starting error monitoring...[/green]")
        console.print(f"[cyan]Webhook:[/cyan] {webhook}")
        console.print(f"[cyan]Interval:[/cyan] {interval} seconds")
        console.print("[yellow]Press Ctrl+C to stop[/yellow]")
        console.print()

        # Setup signal handler for graceful shutdown
        stop_monitoring = False

        def signal_handler(sig, frame):
            nonlocal stop_monitoring
            stop_monitoring = True
            console.print("\n[yellow]Stopping monitoring...[/yellow]")

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        iteration = 0

        # Monitoring loop
        while not stop_monitoring:
            iteration += 1
            console.print(f"[bold]Iteration {iteration}:[/bold] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Get health status
            status = await health_manager.get_health_status()

            # Display status
            status_colors = {
                HealthStatus.HEALTHY: "green",
                HealthStatus.DEGRADED: "yellow",
                HealthStatus.UNHEALTHY: "red"
            }
            status_color = status_colors.get(status.status, "white")
            console.print(f"  Health: [{status_color}]{status.status.value}[/{status_color}]")

            # Emit health status to hooks
            for hook in collector.hooks:
                try:
                    await hook.emit_health_status(status)
                except Exception as e:
                    logger.error(f"Failed to emit health status to hook: {e}")

            # Get and display error rate
            stats = await error_manager.get_error_stats()
            # Calculate error rate (simplified - errors in last minute)
            error_rate = stats.total_count / max(interval / 60, 1)
            console.print(f"  Error Rate: {error_rate:.2f} errors/min")

            # Emit error rate metric
            await collector.emit_error_rate(error_rate, f"{interval}s")

            console.print()

            # Wait for next iteration
            await asyncio.sleep(interval)

        # Cleanup
        console.print("[green]Monitoring stopped[/green]")
        await health_manager.close()
        await collector.close()
        await error_manager.close()

    except Exception as e:
        error_message(f"Monitoring failed: {e}")
        logger.error("Error in monitoring loop", error=str(e), exc_info=True)
        raise typer.Exit(1)
