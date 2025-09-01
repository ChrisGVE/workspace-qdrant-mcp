"""
Observability CLI tools for workspace-qdrant-mcp.

Provides command-line tools for health checking, metrics monitoring, and system
diagnostics. Useful for CI/CD pipelines, deployment validation, and troubleshooting.

Commands:
    health - Check system health status
    metrics - Display current metrics  
    diagnostics - Full system diagnostics
    monitor - Continuous monitoring mode

Example:
    ```bash
    # Check health status
    wqm observability health
    
    # Get metrics in JSON format
    wqm observability metrics --format json
    
    # Run continuous monitoring
    wqm observability monitor --interval 30
    
    # Full diagnostics with troubleshooting
    wqm observability diagnostics --verbose
    ```
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich.json import JSON
from rich.live import Live
from rich.layout import Layout

from ..observability import (
    get_logger,
    configure_logging,
    health_checker_instance,
    metrics_instance,
    HealthStatus
)
from ..core.config import Config
from ..core.client import QdrantWorkspaceClient

logger = get_logger(__name__)
console = Console()

# CLI app for observability commands
observability_app = typer.Typer(
    name="observability",
    help="Observability and monitoring tools",
    no_args_is_help=True
)


def create_health_table(health_data: Dict[str, Any]) -> Table:
    """Create a Rich table for health status display."""
    table = Table(title="System Health Status")
    table.add_column("Component", style="cyan", min_width=20)
    table.add_column("Status", min_width=12)
    table.add_column("Message", style="dim", min_width=30)
    table.add_column("Response Time", style="magenta", min_width=12)
    
    # Overall status row
    overall_status = health_data.get("status", "unknown")
    status_color = {
        "healthy": "green",
        "degraded": "yellow", 
        "unhealthy": "red",
        "unknown": "dim"
    }.get(overall_status, "dim")
    
    table.add_row(
        "OVERALL",
        Text(overall_status.upper(), style=f"bold {status_color}"),
        health_data.get("message", ""),
        ""
    )
    
    table.add_section()
    
    # Component status rows
    components = health_data.get("components", {})
    for name, comp in components.items():
        comp_status = comp.get("status", "unknown")
        comp_color = {
            "healthy": "green",
            "degraded": "yellow",
            "unhealthy": "red", 
            "unknown": "dim"
        }.get(comp_status, "dim")
        
        response_time = comp.get("response_time")
        response_str = f"{response_time:.3f}s" if response_time else ""
        
        table.add_row(
            name.replace("_", " ").title(),
            Text(comp_status, style=comp_color),
            comp.get("message", ""),
            response_str
        )
    
    return table


def create_metrics_table(metrics_data: Dict[str, Any]) -> Table:
    """Create a Rich table for metrics display."""
    table = Table(title="System Metrics Summary")
    table.add_column("Metric Type", style="cyan", min_width=15)
    table.add_column("Name", style="white", min_width=25)
    table.add_column("Value", style="green", min_width=15)
    table.add_column("Description", style="dim", min_width=30)
    
    # Counters
    counters = metrics_data.get("counters", {})
    for name, data in counters.items():
        if name.endswith("_total"):
            display_name = name.replace("_total", "").replace("_", " ").title()
        else:
            display_name = name.replace("_", " ").title()
        
        table.add_row(
            "Counter",
            display_name,
            str(data.get("value", 0)),
            data.get("description", "")
        )
    
    # Gauges
    gauges = metrics_data.get("gauges", {})
    for name, data in gauges.items():
        display_name = name.replace("_", " ").title()
        value = data.get("value", 0)
        
        # Format common gauge types
        if "bytes" in name:
            if value > 1024*1024*1024:
                value_str = f"{value/(1024*1024*1024):.1f} GB"
            elif value > 1024*1024:
                value_str = f"{value/(1024*1024):.1f} MB"
            elif value > 1024:
                value_str = f"{value/1024:.1f} KB"
            else:
                value_str = f"{value:.0f} B"
        elif "percent" in name:
            value_str = f"{value:.1f}%"
        else:
            value_str = str(value)
        
        table.add_row(
            "Gauge",
            display_name,
            value_str,
            data.get("description", "")
        )
    
    # Key histograms
    histograms = metrics_data.get("histograms", {})
    for name, data in histograms.items():
        if data.get("count", 0) > 0:  # Only show histograms with data
            display_name = name.replace("_seconds", "").replace("_", " ").title()
            avg = data.get("average", 0)
            count = data.get("count", 0)
            
            if "duration" in name or "seconds" in name:
                value_str = f"{avg:.3f}s (n={count})"
            else:
                value_str = f"{avg:.2f} (n={count})"
            
            table.add_row(
                "Histogram",
                display_name,
                value_str,
                data.get("description", "")
            )
    
    return table


@observability_app.command()
def health(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed component information"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
    timeout: int = typer.Option(10, "--timeout", "-t", help="Health check timeout in seconds")
) -> None:
    """Check system health status."""
    
    async def check_health():
        try:
            # Configure minimal logging for CLI
            configure_logging(level="WARNING", json_format=False, console_output=False)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Checking system health...", total=None)
                
                health_data = await asyncio.wait_for(
                    health_checker_instance.get_health_status(),
                    timeout=timeout
                )
                
                progress.update(task, completed=True)
            
            if json_output:
                console.print_json(json.dumps(health_data, indent=2))
            else:
                # Display formatted health status
                table = create_health_table(health_data)
                console.print(table)
                
                # Show detailed information if requested
                if detailed:
                    console.print("\n[bold]Detailed Component Information:[/bold]")
                    components = health_data.get("components", {})
                    for name, comp in components.items():
                        if comp.get("details"):
                            panel_title = f"{name.replace('_', ' ').title()} Details"
                            panel_content = JSON(json.dumps(comp["details"], indent=2))
                            console.print(Panel(panel_content, title=panel_title, border_style="dim"))
                
                # Overall status message
                status = health_data.get("status", "unknown")
                if status == "healthy":
                    console.print("\n✅ [green]System is healthy and operational[/green]")
                elif status == "degraded":
                    console.print("\n⚠️  [yellow]System is degraded but operational[/yellow]")
                else:
                    console.print("\n❌ [red]System is unhealthy[/red]")
                    # Set error exit code for CI/CD
                    raise typer.Exit(1)
        
        except asyncio.TimeoutError:
            console.print(f"❌ [red]Health check timed out after {timeout} seconds[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"❌ [red]Health check failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(check_health())


@observability_app.command()
def metrics(
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, prometheus"),
    filter_type: Optional[str] = typer.Option(None, "--type", help="Filter by metric type: counter, gauge, histogram"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Write output to file")
) -> None:
    """Display current system metrics."""
    
    try:
        # Configure minimal logging for CLI
        configure_logging(level="WARNING", json_format=False, console_output=False)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Collecting metrics...", total=None)
            
            # Update system metrics before display
            metrics_instance.update_system_metrics()
            metrics_data = metrics_instance.get_metrics_summary()
            
            progress.update(task, completed=True)
        
        # Filter by type if requested
        if filter_type:
            if filter_type == "counter":
                metrics_data = {"counters": metrics_data.get("counters", {})}
            elif filter_type == "gauge":
                metrics_data = {"gauges": metrics_data.get("gauges", {})}
            elif filter_type == "histogram":
                metrics_data = {"histograms": metrics_data.get("histograms", {})}
        
        # Generate output
        if format == "json":
            output_content = json.dumps(metrics_data, indent=2)
            if output_file:
                output_file.write_text(output_content)
                console.print(f"✅ Metrics saved to {output_file}")
            else:
                console.print_json(output_content)
        
        elif format == "prometheus":
            output_content = metrics_instance.export_prometheus_format()
            if output_file:
                output_file.write_text(output_content)
                console.print(f"✅ Prometheus metrics saved to {output_file}")
            else:
                console.print(output_content)
        
        else:  # table format
            if output_file:
                console.print("❌ [red]Table format cannot be saved to file. Use --format json or prometheus[/red]")
                raise typer.Exit(1)
            
            table = create_metrics_table(metrics_data)
            console.print(table)
            
            # Show summary stats
            counter_count = len(metrics_data.get("counters", {}))
            gauge_count = len(metrics_data.get("gauges", {}))
            histogram_count = len(metrics_data.get("histograms", {}))
            
            console.print(f"\n📊 Total metrics: {counter_count} counters, {gauge_count} gauges, {histogram_count} histograms")
    
    except Exception as e:
        console.print(f"❌ [red]Failed to collect metrics: {e}[/red]")
        raise typer.Exit(1)


@observability_app.command()
def diagnostics(
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Write diagnostics to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Include detailed diagnostic information")
) -> None:
    """Generate comprehensive system diagnostics."""
    
    async def run_diagnostics():
        try:
            # Configure minimal logging for CLI
            configure_logging(level="WARNING", json_format=False, console_output=False)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                diagnostics_task = progress.add_task("Generating diagnostics...", total=None)
                
                diagnostics_data = await health_checker_instance.get_detailed_diagnostics()
                
                progress.update(diagnostics_task, completed=True)
            
            if output_file:
                # Save detailed diagnostics to file
                diagnostics_json = json.dumps(diagnostics_data, indent=2, default=str)
                output_file.write_text(diagnostics_json)
                console.print(f"✅ [green]Diagnostics saved to {output_file}[/green]")
            else:
                # Display formatted diagnostics
                console.print("[bold]System Diagnostics Report[/bold]")
                console.print(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
                console.print()
                
                # Health status summary
                health_status = diagnostics_data.get("health_status", {})
                health_table = create_health_table(health_status)
                console.print(health_table)
                console.print()
                
                # System information
                system_info = diagnostics_data.get("system_info", {})
                if system_info and not system_info.get("error"):
                    console.print("[bold]System Information:[/bold]")
                    
                    sys_table = Table(show_header=False, box=None)
                    sys_table.add_column("Property", style="cyan", min_width=20)
                    sys_table.add_column("Value", style="white")
                    
                    sys_table.add_row("Process ID", str(system_info.get("process_id", "N/A")))
                    sys_table.add_row("CPU Usage", f"{system_info.get('cpu_percent', 0):.1f}%")
                    
                    memory_info = system_info.get("memory_info", {})
                    if memory_info:
                        rss_mb = memory_info.get("rss", 0) / 1024 / 1024
                        sys_table.add_row("Memory Usage", f"{rss_mb:.1f} MB")
                    
                    sys_table.add_row("Thread Count", str(system_info.get("num_threads", "N/A")))
                    
                    console.print(sys_table)
                    console.print()
                
                # Configuration status
                check_history = diagnostics_data.get("check_history", {})
                if check_history:
                    console.print("[bold]Health Check Configuration:[/bold]")
                    
                    check_table = Table()
                    check_table.add_column("Check", style="cyan")
                    check_table.add_column("Enabled", style="green")
                    check_table.add_column("Critical", style="yellow")
                    check_table.add_column("Failures", style="red")
                    
                    for check_name, check_info in check_history.items():
                        check_table.add_row(
                            check_name.replace("_", " ").title(),
                            "✓" if check_info.get("enabled") else "✗",
                            "✓" if check_info.get("critical") else "✗",
                            str(check_info.get("consecutive_failures", 0))
                        )
                    
                    console.print(check_table)
                    console.print()
                
                # Verbose information
                if verbose:
                    console.print("[bold]Verbose Diagnostics:[/bold]")
                    verbose_content = JSON(json.dumps(diagnostics_data, indent=2, default=str))
                    console.print(Panel(verbose_content, title="Full Diagnostic Data", border_style="dim"))
                
        except Exception as e:
            console.print(f"❌ [red]Diagnostics failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_diagnostics())


@observability_app.command()
def monitor(
    interval: int = typer.Option(30, "--interval", "-i", help="Monitoring interval in seconds"),
    duration: Optional[int] = typer.Option(None, "--duration", "-d", help="Total monitoring duration in seconds"),
    alert_threshold: str = typer.Option("unhealthy", "--alert", help="Alert on health status: degraded, unhealthy")
) -> None:
    """Run continuous monitoring mode."""
    
    async def continuous_monitor():
        try:
            # Configure minimal logging for CLI
            configure_logging(level="WARNING", json_format=False, console_output=False)
            
            start_time = time.time()
            iteration = 0
            
            console.print(f"🔍 [cyan]Starting continuous monitoring (interval: {interval}s)[/cyan]")
            if duration:
                console.print(f"⏱️  [dim]Will run for {duration} seconds[/dim]")
            console.print("Press Ctrl+C to stop\n")
            
            def create_monitor_layout(health_data: Dict[str, Any], metrics_data: Dict[str, Any]) -> Layout:
                """Create live monitoring layout."""
                layout = Layout()
                layout.split_column(
                    Layout(name="header", size=3),
                    Layout(name="main"),
                    Layout(name="footer", size=2)
                )
                
                # Header
                current_time = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
                status = health_data.get("status", "unknown")
                status_color = {
                    "healthy": "green",
                    "degraded": "yellow",
                    "unhealthy": "red",
                    "unknown": "dim"
                }.get(status, "dim")
                
                header_text = Text()
                header_text.append("System Monitor - ", style="bold")
                header_text.append(status.upper(), style=f"bold {status_color}")
                header_text.append(f" - {current_time}", style="dim")
                
                layout["header"].update(Panel(header_text, border_style="blue"))
                
                # Main content
                layout["main"].split_row(
                    Layout(name="health", ratio=1),
                    Layout(name="metrics", ratio=1)
                )
                
                health_table = create_health_table(health_data)
                layout["main"]["health"].update(Panel(health_table, title="Health Status", border_style="green"))
                
                # Key metrics only for live view
                key_metrics = {
                    "counters": {},
                    "gauges": {},
                    "histograms": {}
                }
                
                # Select most important metrics
                counters = metrics_data.get("counters", {})
                for name in ["requests_total", "operations_total", "search_queries_total"]:
                    if name in counters:
                        key_metrics["counters"][name] = counters[name]
                
                gauges = metrics_data.get("gauges", {})
                for name in ["memory_usage_bytes", "cpu_usage_percent", "active_connections"]:
                    if name in gauges:
                        key_metrics["gauges"][name] = gauges[name]
                
                metrics_table = create_metrics_table(key_metrics)
                layout["main"]["metrics"].update(Panel(metrics_table, title="Key Metrics", border_style="yellow"))
                
                # Footer
                runtime = time.time() - start_time
                footer_text = f"Runtime: {runtime:.0f}s | Iteration: {iteration} | Next check in: {interval}s"
                layout["footer"].update(Panel(footer_text, style="dim", border_style="dim"))
                
                return layout
            
            with Live(console=console, refresh_per_second=1) as live:
                while True:
                    try:
                        # Check duration limit
                        if duration and (time.time() - start_time) >= duration:
                            break
                        
                        iteration += 1
                        
                        # Get health and metrics data
                        health_data = await health_checker_instance.get_health_status()
                        metrics_instance.update_system_metrics()
                        metrics_data = metrics_instance.get_metrics_summary()
                        
                        # Update live display
                        layout = create_monitor_layout(health_data, metrics_data)
                        live.update(layout)
                        
                        # Check for alerts
                        status = health_data.get("status", "unknown")
                        if (alert_threshold == "degraded" and status in ["degraded", "unhealthy"]) or \
                           (alert_threshold == "unhealthy" and status == "unhealthy"):
                            
                            # Print alert to stderr (won't interfere with live display)
                            alert_message = f"🚨 ALERT: System status is {status.upper()} at {time.strftime('%H:%M:%S')}"
                            console.print(f"\n{alert_message}", style="bold red", err=True)
                        
                        # Sleep until next check
                        await asyncio.sleep(interval)
                        
                    except KeyboardInterrupt:
                        break
                
            runtime = time.time() - start_time
            console.print(f"\n✅ [green]Monitoring completed. Runtime: {runtime:.1f}s, Iterations: {iteration}[/green]")
            
        except Exception as e:
            console.print(f"\n❌ [red]Monitoring failed: {e}[/red]")
            raise typer.Exit(1)
    
    try:
        asyncio.run(continuous_monitor())
    except KeyboardInterrupt:
        console.print("\n👋 [yellow]Monitoring stopped by user[/yellow]")


# Export the observability CLI app
__all__ = ["observability_app"]