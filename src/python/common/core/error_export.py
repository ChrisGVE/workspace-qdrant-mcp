"""
Error Export and Debug Bundle Generation

Provides export capabilities for error messages and comprehensive debug bundle
generation for troubleshooting and support. Supports CSV and JSON export formats,
and creates detailed debug bundles with system information, error context, and logs.

Features:
    - CSV export with proper field escaping
    - JSON export with full error details
    - Filtered exports by date range, severity, category
    - Debug bundle generation with system info
    - Error context gathering from related queue items
    - Log extraction around error timestamps
    - Compressed archive creation (.tar.gz)

Example:
    ```python
    from workspace_qdrant_mcp.core.error_export import ErrorExporter, DebugBundleGenerator
    from workspace_qdrant_mcp.core.error_filtering import ErrorFilter
    from workspace_qdrant_mcp.core.error_categorization import ErrorSeverity

    # Initialize exporter
    exporter = ErrorExporter(error_manager)
    await exporter.initialize()

    # Export to CSV
    await exporter.export_to_csv(errors, "errors.csv")

    # Export filtered errors
    filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
    await exporter.export_filtered(filter, "csv", "critical_errors.csv")

    # Create debug bundle
    bundle_gen = DebugBundleGenerator(error_manager)
    await bundle_gen.initialize()
    bundle = await bundle_gen.create_debug_bundle(
        error_ids=["123", "124"],
        output_path="/tmp/debug_bundle"
    )
    archive_path = await bundle_gen.bundle_to_archive(bundle, "debug.tar.gz")
    ```
"""

import csv
import json
import platform
import sqlite3
import sys
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .error_filtering import ErrorFilter, ErrorFilterManager
from .error_message_manager import ErrorMessage, ErrorMessageManager


@dataclass
class DebugBundle:
    """
    Debug bundle containing comprehensive error information.

    Attributes:
        bundle_id: Unique identifier for this bundle
        created_at: When the bundle was created
        error_ids: List of error IDs included in bundle
        error_details: Full error information
        system_info: System and environment information
        context_data: Related queue items and context
        log_excerpts: Relevant log lines around error times
        metadata: Additional bundle metadata
        output_path: Directory path where bundle files are stored
    """
    bundle_id: str
    created_at: datetime
    error_ids: list[str]
    error_details: list[dict[str, Any]]
    system_info: dict[str, Any]
    context_data: dict[str, Any]
    log_excerpts: dict[str, str]  # error_id -> log text
    metadata: dict[str, Any] = field(default_factory=dict)
    output_path: Path | None = None


class ErrorExporter:
    """
    Error message exporter with multiple format support.

    Provides methods to export error messages to CSV and JSON formats
    with filtering capabilities and proper data formatting.
    """

    def __init__(self, error_manager: ErrorMessageManager):
        """
        Initialize error exporter.

        Args:
            error_manager: ErrorMessageManager instance for data access
        """
        self.error_manager = error_manager
        self.filter_manager = ErrorFilterManager(error_manager)
        self._initialized = False

    async def initialize(self):
        """Initialize the error exporter."""
        if self._initialized:
            return

        # Ensure managers are initialized
        if not self.error_manager._initialized:
            await self.error_manager.initialize()

        await self.filter_manager.initialize()

        self._initialized = True
        logger.info("Error exporter initialized")

    async def close(self):
        """Close the error exporter."""
        if not self._initialized:
            return

        await self.filter_manager.close()
        self._initialized = False
        logger.info("Error exporter closed")

    async def export_to_csv(
        self,
        errors: list[ErrorMessage],
        filepath: str
    ) -> bool:
        """
        Export error messages to CSV format.

        CSV format includes proper escaping for fields containing commas,
        newlines, and quotes. Uses UTF-8 encoding.

        Args:
            errors: List of ErrorMessage instances to export
            filepath: Output file path

        Returns:
            True if export successful, False otherwise

        Raises:
            ValueError: If errors list is empty
            IOError: If file cannot be written
        """
        if not self._initialized:
            raise RuntimeError("Exporter not initialized. Call initialize() first.")

        if not errors:
            raise ValueError("Cannot export empty error list")

        try:
            output_path = Path(filepath)

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'id',
                    'timestamp',
                    'severity',
                    'category',
                    'message',
                    'file_path',
                    'collection',
                    'tenant_id',
                    'acknowledged',
                    'retry_count'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()

                for error in errors:
                    # Extract context fields safely
                    context = error.context or {}
                    file_path = context.get('file_path', '')
                    collection = context.get('collection', '')
                    tenant_id = context.get('tenant_id', '')

                    writer.writerow({
                        'id': error.id,
                        'timestamp': error.timestamp.isoformat(),
                        'severity': error.severity.value,
                        'category': error.category.value,
                        'message': error.message,
                        'file_path': file_path,
                        'collection': collection,
                        'tenant_id': tenant_id,
                        'acknowledged': error.acknowledged,
                        'retry_count': error.retry_count
                    })

            logger.info(f"Exported {len(errors)} errors to CSV: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}", exc_info=True)
            return False

    async def export_to_json(
        self,
        errors: list[ErrorMessage],
        filepath: str
    ) -> bool:
        """
        Export error messages to JSON format.

        JSON output includes full error details with proper formatting.

        Args:
            errors: List of ErrorMessage instances to export
            filepath: Output file path

        Returns:
            True if export successful, False otherwise

        Raises:
            ValueError: If errors list is empty
            IOError: If file cannot be written
        """
        if not self._initialized:
            raise RuntimeError("Exporter not initialized. Call initialize() first.")

        if not errors:
            raise ValueError("Cannot export empty error list")

        try:
            output_path = Path(filepath)

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert errors to dictionaries
            error_dicts = [error.to_dict() for error in errors]

            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(error_dicts, jsonfile, indent=2, default=str)

            logger.info(f"Exported {len(errors)} errors to JSON: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}", exc_info=True)
            return False

    async def export_filtered(
        self,
        filter: ErrorFilter,
        format: str,
        filepath: str,
        limit: int = 1000
    ) -> bool:
        """
        Export errors matching filter criteria.

        Args:
            filter: ErrorFilter with criteria to apply
            format: Output format ('csv' or 'json')
            filepath: Output file path
            limit: Maximum number of errors to export

        Returns:
            True if export successful, False otherwise

        Raises:
            ValueError: If format is invalid or no errors match filter
        """
        if not self._initialized:
            raise RuntimeError("Exporter not initialized. Call initialize() first.")

        # Validate format
        format_lower = format.lower()
        if format_lower not in ['csv', 'json']:
            raise ValueError(f"Invalid format: {format}. Use 'csv' or 'json'")

        # Filter errors
        result = await self.filter_manager.filter_errors(filter, limit=limit)

        if not result.errors:
            raise ValueError("No errors match the specified filter criteria")

        # Export based on format
        if format_lower == 'csv':
            return await self.export_to_csv(result.errors, filepath)
        else:
            return await self.export_to_json(result.errors, filepath)

    async def export_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str,
        filepath: str,
        limit: int = 1000
    ) -> bool:
        """
        Export errors within a specific date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            format: Output format ('csv' or 'json')
            filepath: Output file path
            limit: Maximum number of errors to export

        Returns:
            True if export successful, False otherwise

        Raises:
            ValueError: If date range is invalid or format is invalid
        """
        if not self._initialized:
            raise RuntimeError("Exporter not initialized. Call initialize() first.")

        if start_date > end_date:
            raise ValueError("start_date must be before or equal to end_date")

        # Create filter with date range
        filter = ErrorFilter(date_range=(start_date, end_date))

        return await self.export_filtered(filter, format, filepath, limit=limit)


class DebugBundleGenerator:
    """
    Debug bundle generator for comprehensive error troubleshooting.

    Creates debug bundles containing error details, system information,
    related context, and log excerpts for support and diagnostics.
    """

    def __init__(self, error_manager: ErrorMessageManager):
        """
        Initialize debug bundle generator.

        Args:
            error_manager: ErrorMessageManager instance for data access
        """
        self.error_manager = error_manager
        self._initialized = False
        self._db_path: str | None = None

    async def initialize(self):
        """Initialize the debug bundle generator."""
        if self._initialized:
            return

        # Ensure error manager is initialized
        if not self.error_manager._initialized:
            await self.error_manager.initialize()

        # Get database path for context queries
        self._db_path = self.error_manager.connection_pool.db_path

        self._initialized = True
        logger.info("Debug bundle generator initialized")

    async def close(self):
        """Close the debug bundle generator."""
        if not self._initialized:
            return

        self._initialized = False
        logger.info("Debug bundle generator closed")

    def include_system_info(self) -> dict[str, Any]:
        """
        Collect system information for debug bundle.

        Returns:
            Dictionary containing OS, Python version, package versions
        """
        try:
            # Get Python version info
            python_version_short = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

            # Get platform information
            system_info = {
                "os": platform.system(),
                "os_version": platform.version(),
                "os_release": platform.release(),
                "platform": platform.platform(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": python_version_short,
                "python_implementation": platform.python_implementation(),
                "python_compiler": platform.python_compiler(),
            }

            # Try to get package versions
            try:
                import importlib.metadata

                packages = [
                    "workspace-qdrant-mcp",
                    "qdrant-client",
                    "fastembed",
                    "loguru",
                    "typer",
                ]

                package_versions = {}
                for pkg in packages:
                    try:
                        version = importlib.metadata.version(pkg)
                        package_versions[pkg] = version
                    except importlib.metadata.PackageNotFoundError:
                        package_versions[pkg] = "not installed"

                system_info["package_versions"] = package_versions

            except Exception as e:
                logger.warning(f"Could not get package versions: {e}")
                system_info["package_versions"] = {}

            logger.debug("Collected system information for debug bundle")
            return system_info

        except Exception as e:
            logger.error(f"Error collecting system info: {e}", exc_info=True)
            return {"error": str(e)}

    async def include_error_context(self, error_id: str) -> dict[str, Any]:
        """
        Include error context with related queue items.

        Args:
            error_id: Error message ID

        Returns:
            Dictionary with full error details and related queue items
        """
        if not self._initialized:
            raise RuntimeError("Generator not initialized. Call initialize() first.")

        try:
            # Get error details
            error = await self.error_manager.get_error_by_id(int(error_id))

            if not error:
                return {"error": f"Error ID {error_id} not found"}

            context = {
                "error": error.to_dict(),
                "related_queue_items": []
            }

            # Query related queue items if we have file_path in context
            if error.context and 'file_path' in error.context:
                file_path = error.context['file_path']

                try:
                    async with self.error_manager.connection_pool.get_connection_async() as conn:
                        # Query ingestion_queue for related items
                        cursor = conn.execute(
                            """
                            SELECT
                                file_absolute_path,
                                collection_name,
                                tenant_id,
                                operation,
                                priority,
                                retry_count,
                                queued_timestamp
                            FROM ingestion_queue
                            WHERE file_absolute_path = ?
                            ORDER BY queued_timestamp DESC
                            LIMIT 10
                            """,
                            (file_path,)
                        )

                        rows = cursor.fetchall()
                        context["related_queue_items"] = [dict(row) for row in rows]

                except sqlite3.Error as e:
                    logger.warning(f"Could not query queue items for error {error_id}: {e}")
                    context["related_queue_items_error"] = str(e)

            return context

        except Exception as e:
            logger.error(f"Error including context for {error_id}: {e}", exc_info=True)
            return {"error": str(e)}

    async def include_logs(
        self,
        error_id: str,
        lines: int = 100
    ) -> str:
        """
        Extract log lines around error timestamp.

        Args:
            error_id: Error message ID
            lines: Number of log lines to extract (before and after)

        Returns:
            String containing relevant log excerpts
        """
        if not self._initialized:
            raise RuntimeError("Generator not initialized. Call initialize() first.")

        try:
            # Get error to find timestamp
            error = await self.error_manager.get_error_by_id(int(error_id))

            if not error:
                return f"Error ID {error_id} not found"

            # Try to find log file
            # Default log location
            log_paths = [
                Path.home() / ".config" / "workspace-qdrant" / "logs" / "daemon.log",
                Path.home() / ".local" / "share" / "workspace-qdrant" / "logs" / "daemon.log",
                Path("/var/log/workspace-qdrant/daemon.log"),
            ]

            log_content = []
            log_found = False

            for log_path in log_paths:
                if log_path.exists():
                    log_found = True
                    try:
                        # Read log file and find lines around timestamp
                        # This is a simplified version - in production would use more sophisticated log parsing
                        with open(log_path, encoding='utf-8') as f:
                            all_lines = f.readlines()

                        # For now, just include last N lines
                        # In a real implementation, we'd parse timestamps and find relevant section
                        relevant_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

                        log_content.append(f"=== Log file: {log_path} ===\n")
                        log_content.append(f"Error timestamp: {error.timestamp.isoformat()}\n")
                        log_content.append(f"Showing last {len(relevant_lines)} lines:\n\n")
                        log_content.extend(relevant_lines)

                        break  # Found log file, stop searching

                    except Exception as e:
                        log_content.append(f"Error reading log file {log_path}: {e}\n")
                        continue

            if not log_found:
                log_content.append(
                    "No log files found. Searched:\n" +
                    "\n".join(f"  - {p}" for p in log_paths)
                )

            return "".join(log_content)

        except Exception as e:
            logger.error(f"Error extracting logs for {error_id}: {e}", exc_info=True)
            return f"Error extracting logs: {e}"

    async def create_debug_bundle(
        self,
        error_ids: list[str],
        output_path: str
    ) -> DebugBundle:
        """
        Create a comprehensive debug bundle for specified errors.

        Args:
            error_ids: List of error IDs to include
            output_path: Directory path for bundle output

        Returns:
            DebugBundle instance with all collected information

        Raises:
            ValueError: If error_ids is empty
            RuntimeError: If generator not initialized
        """
        if not self._initialized:
            raise RuntimeError("Generator not initialized. Call initialize() first.")

        if not error_ids:
            raise ValueError("error_ids cannot be empty")

        try:
            bundle_id = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            created_at = datetime.now()

            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Collect system info
            system_info = self.include_system_info()

            # Collect error details and context
            error_details = []
            context_data = {}
            log_excerpts = {}

            for error_id in error_ids:
                # Error details and context
                context = await self.include_error_context(error_id)
                error_details.append(context.get("error", {}))
                context_data[error_id] = context

                # Log excerpts
                logs = await self.include_logs(error_id, lines=100)
                log_excerpts[error_id] = logs

            # Create bundle metadata
            metadata = {
                "bundle_id": bundle_id,
                "created_at": created_at.isoformat(),
                "error_count": len(error_ids),
                "generator_version": "1.0.0",
            }

            bundle = DebugBundle(
                bundle_id=bundle_id,
                created_at=created_at,
                error_ids=error_ids,
                error_details=error_details,
                system_info=system_info,
                context_data=context_data,
                log_excerpts=log_excerpts,
                metadata=metadata,
                output_path=output_dir
            )

            # Write bundle files to output directory
            await self._write_bundle_files(bundle)

            logger.info(
                f"Created debug bundle {bundle_id} with {len(error_ids)} errors "
                f"at {output_path}"
            )

            return bundle

        except Exception as e:
            logger.error(f"Error creating debug bundle: {e}", exc_info=True)
            raise

    async def _write_bundle_files(self, bundle: DebugBundle):
        """
        Write bundle contents to individual files.

        Args:
            bundle: DebugBundle to write
        """
        if not bundle.output_path:
            raise ValueError("Bundle output_path not set")

        # Write error_details.json
        with open(bundle.output_path / "error_details.json", 'w', encoding='utf-8') as f:
            json.dump(bundle.error_details, f, indent=2, default=str)

        # Write system_info.json
        with open(bundle.output_path / "system_info.json", 'w', encoding='utf-8') as f:
            json.dump(bundle.system_info, f, indent=2, default=str)

        # Write context.json
        with open(bundle.output_path / "context.json", 'w', encoding='utf-8') as f:
            json.dump(bundle.context_data, f, indent=2, default=str)

        # Write logs.txt (concatenate all log excerpts)
        with open(bundle.output_path / "logs.txt", 'w', encoding='utf-8') as f:
            for error_id, logs in bundle.log_excerpts.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"Error ID: {error_id}\n")
                f.write(f"{'='*80}\n\n")
                f.write(logs)
                f.write("\n\n")

        # Write metadata.json
        with open(bundle.output_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(bundle.metadata, f, indent=2, default=str)

    async def bundle_to_archive(
        self,
        bundle: DebugBundle,
        output_path: str
    ) -> str:
        """
        Create compressed archive from debug bundle.

        Args:
            bundle: DebugBundle to archive
            output_path: Output path for .tar.gz file

        Returns:
            Path to created archive file

        Raises:
            ValueError: If bundle has no output_path
            IOError: If archive creation fails
        """
        if not bundle.output_path:
            raise ValueError("Bundle has no output_path set")

        try:
            archive_path = Path(output_path)

            # Ensure parent directory exists
            archive_path.parent.mkdir(parents=True, exist_ok=True)

            # Create tar.gz archive
            with tarfile.open(archive_path, 'w:gz') as tar:
                # Add all files from bundle directory
                tar.add(
                    bundle.output_path,
                    arcname=bundle.bundle_id,
                    recursive=True
                )

            logger.info(f"Created debug bundle archive: {archive_path}")
            return str(archive_path)

        except Exception as e:
            logger.error(f"Error creating archive: {e}", exc_info=True)
            raise OSError(f"Failed to create archive: {e}") from e
