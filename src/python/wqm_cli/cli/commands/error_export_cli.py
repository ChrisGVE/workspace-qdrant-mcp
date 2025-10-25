"""Error export CLI commands.

This module provides commands to export error messages and create debug bundles
for troubleshooting and support.

Usage:
    wqm errors export --format=csv --output=errors.csv
    wqm errors export --format=json --severity=error --output=critical.json
    wqm errors export --days=30 --format=csv --output=monthly.csv
    wqm errors debug-bundle --error-id=123 --output=debug.tar.gz
    wqm errors debug-bundle --last-n=10 --output=last_errors.tar.gz
"""

from datetime import datetime, timedelta
from pathlib import Path

import typer
from common.core.error_categorization import ErrorCategory, ErrorSeverity
from common.core.error_export import DebugBundleGenerator, ErrorExporter
from common.core.error_filtering import ErrorFilter
from common.core.error_message_manager import ErrorMessageManager
from loguru import logger

from ..utils import (
    create_command_app,
    error_message,
    handle_async,
    success_message,
)

# Create the export commands app (to be registered under errors_app)
export_app = create_command_app(
    name="export",
    help_text="""Export error messages and create debug bundles.

Export errors to CSV or JSON formats with filtering options, or create
comprehensive debug bundles for troubleshooting.

Examples:
    wqm errors export --format=csv --output=errors.csv
    wqm errors export --format=json --severity=error --output=critical.json
    wqm errors debug-bundle --error-id=123 --output=debug.tar.gz""",
    no_args_is_help=False,
)


@export_app.command("export")
def export_errors(
    format: str = typer.Option(
        "csv",
        "--format",
        "-f",
        help="Output format: csv or json",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file path",
    ),
    severity: str | None = typer.Option(
        None,
        "--severity",
        "-s",
        help="Filter by severity (error, warning, info)",
    ),
    category: str | None = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category (e.g., file_corrupt, network)",
    ),
    days: int | None = typer.Option(
        None,
        "--days",
        "-d",
        help="Export errors from last N days",
    ),
    start_date: str | None = typer.Option(
        None,
        "--start-date",
        help="Start date (ISO format: YYYY-MM-DD)",
    ),
    end_date: str | None = typer.Option(
        None,
        "--end-date",
        help="End date (ISO format: YYYY-MM-DD)",
    ),
    acknowledged: bool | None = typer.Option(
        None,
        "--acknowledged/--unacknowledged",
        help="Filter by acknowledgment status",
    ),
    limit: int = typer.Option(
        1000,
        "--limit",
        "-l",
        help="Maximum number of errors to export",
    ),
):
    """Export error messages to CSV or JSON format.

    Exports errors with optional filtering by severity, category, date range,
    and acknowledgment status. Use --format to choose output format.
    """
    handle_async(_export_errors(
        format, output, severity, category, days, start_date, end_date,
        acknowledged, limit
    ))


async def _export_errors(
    format: str,
    output: str,
    severity: str | None,
    category: str | None,
    days: int | None,
    start_date_str: str | None,
    end_date_str: str | None,
    acknowledged: bool | None,
    limit: int,
) -> None:
    """Implementation of export_errors command."""
    try:
        # Validate format
        if format.lower() not in ['csv', 'json']:
            error_message(f"Invalid format: {format}. Use 'csv' or 'json'")
            raise typer.Exit(1)

        # Initialize error manager and exporter
        error_manager = ErrorMessageManager()
        await error_manager.initialize()

        exporter = ErrorExporter(error_manager)
        await exporter.initialize()

        # Build filter criteria

        # Parse severity
        severity_levels = None
        if severity:
            try:
                severity_levels = [ErrorSeverity.from_string(severity)]
            except ValueError as e:
                error_message(str(e))
                await exporter.close()
                await error_manager.close()
                raise typer.Exit(1)

        # Parse category
        categories = None
        if category:
            try:
                categories = [ErrorCategory.from_string(category)]
            except ValueError as e:
                error_message(str(e))
                await exporter.close()
                await error_manager.close()
                raise typer.Exit(1)

        # Parse date range
        date_range = None
        if days is not None:
            # Last N days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            date_range = (start_date, end_date)
        elif start_date_str or end_date_str:
            # Explicit date range
            try:
                start_date = datetime.fromisoformat(start_date_str) if start_date_str else datetime.min
                end_date = datetime.fromisoformat(end_date_str) if end_date_str else datetime.now()
                date_range = (start_date, end_date)
            except ValueError as e:
                error_message(f"Invalid date format: {e}. Use YYYY-MM-DD")
                await exporter.close()
                await error_manager.close()
                raise typer.Exit(1)

        # Build filter
        filter = ErrorFilter(
            severity_levels=severity_levels,
            categories=categories,
            date_range=date_range,
            acknowledged_only=acknowledged if acknowledged is True else False,
            unacknowledged_only=acknowledged if acknowledged is False else False
        )

        # Export
        try:
            success = await exporter.export_filtered(filter, format, output, limit=limit)

            if success:
                success_message(f"Successfully exported errors to {output}")
            else:
                error_message(f"Failed to export errors to {output}")
                raise typer.Exit(1)

        except ValueError as e:
            error_message(str(e))
            await exporter.close()
            await error_manager.close()
            raise typer.Exit(1)

        # Cleanup
        await exporter.close()
        await error_manager.close()

    except Exception as e:
        error_message(f"Export failed: {e}")
        logger.error("Error exporting errors", error=str(e), exc_info=True)
        raise typer.Exit(1)


@export_app.command("debug-bundle")
def create_debug_bundle(
    error_id: int | None = typer.Option(
        None,
        "--error-id",
        "-e",
        help="Specific error ID to include in bundle",
    ),
    last_n: int | None = typer.Option(
        None,
        "--last-n",
        "-n",
        help="Include last N errors in bundle",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path for .tar.gz bundle file",
    ),
    severity: str | None = typer.Option(
        None,
        "--severity",
        "-s",
        help="Filter by severity when using --last-n (error, warning, info)",
    ),
):
    """Create comprehensive debug bundle for troubleshooting.

    Creates a debug bundle containing error details, system information,
    related queue context, and log excerpts. Use --error-id for specific
    errors or --last-n for recent errors.
    """
    handle_async(_create_debug_bundle(error_id, last_n, output, severity))


async def _create_debug_bundle(
    error_id: int | None,
    last_n: int | None,
    output: str,
    severity: str | None,
) -> None:
    """Implementation of create_debug_bundle command."""
    try:
        # Validate inputs
        if error_id is None and last_n is None:
            error_message("Specify either --error-id or --last-n")
            raise typer.Exit(1)

        if error_id is not None and last_n is not None:
            error_message("Cannot specify both --error-id and --last-n")
            raise typer.Exit(1)

        # Initialize error manager and bundle generator
        error_manager = ErrorMessageManager()
        await error_manager.initialize()

        bundle_gen = DebugBundleGenerator(error_manager)
        await bundle_gen.initialize()

        # Collect error IDs
        error_ids = []

        if error_id is not None:
            # Single error ID
            error_ids = [str(error_id)]

            # Verify error exists
            error = await error_manager.get_error_by_id(error_id)
            if not error:
                error_message(f"Error ID {error_id} not found")
                await bundle_gen.close()
                await error_manager.close()
                raise typer.Exit(1)

        elif last_n is not None:
            # Get last N errors
            severity_filter = None
            if severity:
                try:
                    severity_filter = ErrorSeverity.from_string(severity).value
                except ValueError as e:
                    error_message(str(e))
                    await bundle_gen.close()
                    await error_manager.close()
                    raise typer.Exit(1)

            errors = await error_manager.get_errors(
                severity=severity_filter,
                limit=last_n
            )

            if not errors:
                error_message("No errors found matching criteria")
                await bundle_gen.close()
                await error_manager.close()
                raise typer.Exit(1)

            error_ids = [str(err.id) for err in errors]
            print(f"Including {len(error_ids)} errors in debug bundle")

        # Create temporary directory for bundle
        temp_dir = Path("/tmp") / f"debug_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create bundle
        try:
            bundle = await bundle_gen.create_debug_bundle(
                error_ids=error_ids,
                output_path=str(temp_dir)
            )

            # Create archive
            archive_path = await bundle_gen.bundle_to_archive(bundle, output)

            success_message(f"Debug bundle created: {archive_path}")
            print(f"Bundle ID: {bundle.bundle_id}")
            print(f"Errors included: {len(error_ids)}")

            # Cleanup temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            error_message(f"Failed to create debug bundle: {e}")
            logger.error("Error creating debug bundle", error=str(e), exc_info=True)
            await bundle_gen.close()
            await error_manager.close()
            raise typer.Exit(1)

        # Cleanup
        await bundle_gen.close()
        await error_manager.close()

    except Exception as e:
        error_message(f"Debug bundle creation failed: {e}")
        logger.error("Error in debug bundle creation", error=str(e), exc_info=True)
        raise typer.Exit(1)
