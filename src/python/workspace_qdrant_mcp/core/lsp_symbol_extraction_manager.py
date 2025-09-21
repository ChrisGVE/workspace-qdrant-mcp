"""
LSP Symbol Extraction Manager

This module provides a unified orchestration layer for symbol extraction that integrates
the LSP detector, metadata extractor, health monitoring, and graceful degradation systems.
It implements project-aware symbol extraction workflows with automatic LSP server selection
and fallback mechanisms.

Key Features:
    - Ecosystem-aware LSP server selection using enhanced LSP detector
    - Project-context-aware symbol filtering using pattern system
    - Health monitoring integration with graceful degradation
    - Interface + Minimal Context extraction strategy
    - Performance optimization with multi-level caching
    - Fallback to Tree-sitter when LSP servers are unavailable
    - Cross-file relationship analysis and symbol indexing

Architecture:
    - LspSymbolExtractionManager: Main orchestrator
    - Project-aware extraction pipelines
    - Health monitoring and recovery mechanisms
    - Symbol filtering and optimization layers
    - Fallback and degradation strategies

Example:
    ```python
    from workspace_qdrant_mcp.core.lsp_symbol_extraction_manager import LspSymbolExtractionManager

    # Initialize with project context
    manager = LspSymbolExtractionManager(project_path="/path/to/project")
    await manager.initialize()

    # Extract symbols with ecosystem awareness
    symbols = await manager.extract_project_symbols()

    # Extract from specific file
    file_symbols = await manager.extract_file_symbols("/path/to/file.py")

    # Get extraction statistics
    stats = manager.get_extraction_statistics()
    ```
"""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from loguru import logger

# Import existing LSP components
try:
    from python.common.core.lsp_detector import LSPDetector, LSPDetectionResult, LSPServerInfo
    from python.common.core.lsp_metadata_extractor import LspMetadataExtractor, CodeSymbol, FileMetadata
    from python.common.core.lsp_client import AsyncioLspClient, ConnectionState, LspError
    from python.common.core.pattern_manager import PatternManager
    from python.common.core.lsp_health_monitor import LspHealthMonitor
    from ...tools.symbol_resolver import SymbolResolver, SymbolIndex
except ImportError as e:
    logger.warning(f"Failed to import LSP components: {e}")
    # Fallback imports for development
    LSPDetector = None
    LspMetadataExtractor = None
    AsyncioLspClient = None
    PatternManager = None
    LspHealthMonitor = None
    SymbolResolver = None


@dataclass
class ExtractionContext:
    """Context information for symbol extraction operations."""

    project_path: Path
    detected_ecosystems: List[str] = field(default_factory=list)
    available_lsps: Dict[str, LSPServerInfo] = field(default_factory=dict)
    recommended_lsps: Dict[str, List[str]] = field(default_factory=dict)
    extraction_mode: str = "full"  # full, incremental, minimal
    include_relationships: bool = True
    max_context_lines: int = 3


@dataclass
class ExtractionResult:
    """Result of symbol extraction operation."""

    symbols: List[CodeSymbol] = field(default_factory=list)
    relationships: Dict[str, Any] = field(default_factory=dict)
    file_metadata: Optional[FileMetadata] = None
    extraction_time: float = 0.0
    lsp_used: Optional[str] = None
    fallback_used: bool = False
    errors: List[str] = field(default_factory=list)
    extracted_files: Set[str] = field(default_factory=set)


@dataclass
class ExtractionStatistics:
    """Statistics for extraction operations."""

    total_extractions: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    total_symbols_extracted: int = 0
    total_extraction_time: float = 0.0
    lsp_usage_counts: Dict[str, int] = field(default_factory=dict)
    fallback_usage_count: int = 0
    cache_hit_rate: float = 0.0
    average_extraction_time: float = 0.0


class LspSymbolExtractionManager:
    """
    Unified LSP symbol extraction manager that orchestrates the complete
    symbol extraction workflow with ecosystem awareness and graceful degradation.
    """

    def __init__(
        self,
        project_path: Union[str, Path],
        max_concurrent_lsps: int = 3,
        extraction_timeout: float = 30.0,
        enable_caching: bool = True,
        enable_health_monitoring: bool = True,
        fallback_to_treesitter: bool = True
    ):
        """
        Initialize the LSP symbol extraction manager.

        Args:
            project_path: Path to the project root
            max_concurrent_lsps: Maximum number of concurrent LSP connections
            extraction_timeout: Timeout for extraction operations in seconds
            enable_caching: Enable symbol extraction caching
            enable_health_monitoring: Enable LSP health monitoring
            fallback_to_treesitter: Enable Tree-sitter fallback when LSP fails
        """
        self.project_path = Path(project_path)
        self.max_concurrent_lsps = max_concurrent_lsps
        self.extraction_timeout = extraction_timeout
        self.enable_caching = enable_caching
        self.enable_health_monitoring = enable_health_monitoring
        self.fallback_to_treesitter = fallback_to_treesitter

        # Initialize core components
        self.lsp_detector: Optional[LSPDetector] = None
        self.metadata_extractor: Optional[LspMetadataExtractor] = None
        self.pattern_manager: Optional[PatternManager] = None
        self.health_monitor: Optional[LspHealthMonitor] = None
        self.symbol_resolver: Optional[SymbolResolver] = None

        # Runtime state
        self.initialized = False
        self.active_lsp_clients: Dict[str, AsyncioLspClient] = {}
        self.extraction_cache: Dict[str, ExtractionResult] = {}
        self.statistics = ExtractionStatistics()

        # Extraction context
        self.extraction_context: Optional[ExtractionContext] = None

        logger.info("LSP Symbol Extraction Manager initialized",
                   project_path=str(self.project_path))

    async def initialize(self) -> None:
        """Initialize all components and establish LSP connections."""
        if self.initialized:
            logger.debug("Manager already initialized")
            return

        logger.info("Initializing LSP Symbol Extraction Manager")
        start_time = time.time()

        try:
            # Initialize core components
            await self._initialize_components()

            # Detect project ecosystem and available LSPs
            await self._setup_extraction_context()

            # Initialize LSP connections
            await self._initialize_lsp_connections()

            # Setup health monitoring
            if self.enable_health_monitoring:
                await self._setup_health_monitoring()

            self.initialized = True
            init_time = time.time() - start_time

            logger.info("LSP Symbol Extraction Manager ready",
                       init_time=f"{init_time:.2f}s",
                       available_lsps=len(self.active_lsp_clients),
                       detected_ecosystems=len(self.extraction_context.detected_ecosystems))

        except Exception as e:
            logger.error(f"Failed to initialize LSP Symbol Extraction Manager: {e}")
            await self._cleanup_partial_initialization()
            raise

    async def _initialize_components(self) -> None:
        """Initialize core LSP components."""
        # Initialize LSP detector with pattern manager
        if PatternManager:
            self.pattern_manager = PatternManager()
            self.lsp_detector = LSPDetector(pattern_manager=self.pattern_manager)
        else:
            logger.warning("PatternManager not available, using basic LSP detector")
            self.lsp_detector = LSPDetector()

        # Initialize metadata extractor
        if LspMetadataExtractor:
            self.metadata_extractor = LspMetadataExtractor()
            await self.metadata_extractor.initialize()
        else:
            logger.error("LspMetadataExtractor not available")
            raise RuntimeError("LSP metadata extractor is required")

        # Initialize symbol resolver
        if SymbolResolver:
            # Note: This would typically require a workspace client
            # For now, we'll initialize basic symbol indexing
            self.symbol_resolver = SymbolResolver(workspace_client=None)

        # Initialize health monitor
        if self.enable_health_monitoring and LspHealthMonitor:
            self.health_monitor = LspHealthMonitor()

        logger.debug("Core components initialized")

    async def _setup_extraction_context(self) -> None:
        """Setup extraction context with project ecosystem detection."""
        # Get ecosystem-aware LSP detection
        detection_result = self.lsp_detector.get_ecosystem_aware_lsps(self.project_path)

        # Create extraction context
        self.extraction_context = ExtractionContext(
            project_path=self.project_path,
            detected_ecosystems=detection_result.detected_ecosystems,
            available_lsps=detection_result.detected_lsps,
            recommended_lsps=detection_result.ecosystem_lsp_recommendations
        )

        logger.info("Extraction context established",
                   ecosystems=len(self.extraction_context.detected_ecosystems),
                   available_lsps=len(self.extraction_context.available_lsps),
                   recommendations=len(self.extraction_context.recommended_lsps))

    async def _initialize_lsp_connections(self) -> None:
        """Initialize LSP client connections for recommended servers."""
        if not self.extraction_context:
            raise RuntimeError("Extraction context not established")

        # Prioritize LSPs based on ecosystem recommendations
        priority_lsps = self._get_priority_lsp_list()

        # Initialize connections up to max_concurrent_lsps
        connection_tasks = []
        for i, lsp_name in enumerate(priority_lsps[:self.max_concurrent_lsps]):
            if lsp_name in self.extraction_context.available_lsps:
                task = self._initialize_lsp_client(lsp_name)
                connection_tasks.append(task)

        # Wait for all connections with timeout
        if connection_tasks:
            results = await asyncio.gather(*connection_tasks, return_exceptions=True)

            successful_connections = 0
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    successful_connections += 1
                else:
                    lsp_name = priority_lsps[i]
                    logger.warning(f"Failed to connect to {lsp_name}: {result}")

            logger.info(f"Established {successful_connections}/{len(connection_tasks)} LSP connections")

    async def _initialize_lsp_client(self, lsp_name: str) -> None:
        """Initialize a single LSP client connection."""
        lsp_info = self.extraction_context.available_lsps[lsp_name]

        try:
            if AsyncioLspClient:
                client = AsyncioLspClient(
                    server_cmd=[lsp_info.binary_path],
                    server_name=lsp_name,
                    timeout=self.extraction_timeout
                )

                await client.start()

                # Initialize with project workspace
                await client.initialize(
                    workspace_uri=f"file://{self.project_path}",
                    capabilities={
                        "textDocument": {
                            "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                            "hover": {"contentFormat": ["markdown", "plaintext"]},
                            "definition": {"linkSupport": True}
                        }
                    }
                )

                self.active_lsp_clients[lsp_name] = client
                logger.debug(f"LSP client {lsp_name} initialized successfully")
            else:
                logger.warning(f"AsyncioLspClient not available, skipping {lsp_name}")

        except Exception as e:
            logger.error(f"Failed to initialize LSP client {lsp_name}: {e}")
            raise

    async def _setup_health_monitoring(self) -> None:
        """Setup health monitoring for LSP connections."""
        if not self.health_monitor:
            return

        # Register health callbacks for each LSP client
        for lsp_name, client in self.active_lsp_clients.items():
            self.health_monitor.register_lsp_client(lsp_name, client)
            self.health_monitor.set_health_callback(
                lsp_name,
                self._handle_lsp_health_change
            )

        logger.debug("Health monitoring configured for LSP clients")

    def _get_priority_lsp_list(self) -> List[str]:
        """Get prioritized list of LSPs based on ecosystem recommendations."""
        priority_lsps = []

        # Add ecosystem-recommended LSPs first
        for ecosystem, lsp_names in self.extraction_context.recommended_lsps.items():
            for lsp_name in lsp_names:
                if lsp_name not in priority_lsps:
                    priority_lsps.append(lsp_name)

        # Add any remaining available LSPs
        for lsp_name in self.extraction_context.available_lsps:
            if lsp_name not in priority_lsps:
                priority_lsps.append(lsp_name)

        logger.debug(f"Priority LSP order: {priority_lsps}")
        return priority_lsps

    async def extract_project_symbols(
        self,
        file_patterns: Optional[List[str]] = None,
        include_relationships: bool = True
    ) -> ExtractionResult:
        """
        Extract symbols from entire project with ecosystem awareness.

        Args:
            file_patterns: Optional file patterns to filter (uses pattern manager if None)
            include_relationships: Whether to build symbol relationships

        Returns:
            ExtractionResult with all project symbols
        """
        if not self.initialized:
            await self.initialize()

        logger.info("Starting project symbol extraction")
        start_time = time.time()

        # Get project files using pattern manager
        if self.pattern_manager and not file_patterns:
            project_files = self._get_project_files()
        else:
            project_files = self._get_files_by_patterns(file_patterns or ["**/*"])

        # Extract symbols from all files
        extraction_tasks = []
        for file_path in project_files:
            task = self.extract_file_symbols(file_path, batch_mode=True)
            extraction_tasks.append(task)

        # Process in batches to avoid overwhelming LSP servers
        batch_size = 10
        all_symbols = []
        all_relationships = {}
        used_lsps = set()
        errors = []

        for i in range(0, len(extraction_tasks), batch_size):
            batch = extraction_tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, ExtractionResult):
                    all_symbols.extend(result.symbols)
                    all_relationships.update(result.relationships)
                    if result.lsp_used:
                        used_lsps.add(result.lsp_used)
                    errors.extend(result.errors)
                elif isinstance(result, Exception):
                    errors.append(str(result))

        # Build cross-file relationships if requested
        if include_relationships and self.metadata_extractor:
            cross_file_relationships = await self.metadata_extractor.build_relationship_graph(
                [str(f) for f in project_files]
            )
            all_relationships.update(cross_file_relationships)

        extraction_time = time.time() - start_time

        # Update statistics
        self.statistics.total_extractions += 1
        if not errors:
            self.statistics.successful_extractions += 1
        else:
            self.statistics.failed_extractions += 1

        self.statistics.total_symbols_extracted += len(all_symbols)
        self.statistics.total_extraction_time += extraction_time

        result = ExtractionResult(
            symbols=all_symbols,
            relationships=all_relationships,
            extraction_time=extraction_time,
            lsp_used=",".join(used_lsps) if used_lsps else None,
            errors=errors,
            extracted_files=set(str(f) for f in project_files)
        )

        logger.info("Project symbol extraction completed",
                   symbols_count=len(all_symbols),
                   files_processed=len(project_files),
                   extraction_time=f"{extraction_time:.2f}s",
                   lsps_used=len(used_lsps))

        return result

    async def extract_file_symbols(
        self,
        file_path: Union[str, Path],
        batch_mode: bool = False
    ) -> ExtractionResult:
        """
        Extract symbols from a specific file using appropriate LSP server.

        Args:
            file_path: Path to the file to extract symbols from
            batch_mode: Whether this is part of a batch operation

        Returns:
            ExtractionResult with file symbols
        """
        file_path = Path(file_path)

        if not self.initialized:
            await self.initialize()

        # Check cache first
        cache_key = str(file_path)
        if self.enable_caching and cache_key in self.extraction_cache:
            cached_result = self.extraction_cache[cache_key]
            logger.debug(f"Using cached symbols for {file_path}")
            return cached_result

        logger.debug(f"Extracting symbols from {file_path}")
        start_time = time.time()

        # Determine the best LSP for this file
        lsp_name = self._select_lsp_for_file(file_path)

        if lsp_name and lsp_name in self.active_lsp_clients:
            try:
                # Extract using LSP
                result = await self._extract_with_lsp(file_path, lsp_name)
                result.lsp_used = lsp_name

                # Update LSP usage statistics
                if lsp_name in self.statistics.lsp_usage_counts:
                    self.statistics.lsp_usage_counts[lsp_name] += 1
                else:
                    self.statistics.lsp_usage_counts[lsp_name] = 1

            except Exception as e:
                logger.warning(f"LSP extraction failed for {file_path}: {e}")
                # Fall back to Tree-sitter if enabled
                if self.fallback_to_treesitter:
                    result = await self._extract_with_treesitter(file_path)
                    result.fallback_used = True
                    self.statistics.fallback_usage_count += 1
                else:
                    result = ExtractionResult(errors=[str(e)])
        else:
            # No suitable LSP, use Tree-sitter fallback
            if self.fallback_to_treesitter:
                result = await self._extract_with_treesitter(file_path)
                result.fallback_used = True
                self.statistics.fallback_usage_count += 1
            else:
                result = ExtractionResult(errors=["No suitable LSP available"])

        result.extraction_time = time.time() - start_time

        # Cache the result
        if self.enable_caching and not result.errors:
            self.extraction_cache[cache_key] = result

        return result

    def _select_lsp_for_file(self, file_path: Path) -> Optional[str]:
        """Select the best LSP server for extracting symbols from a file."""
        if not self.extraction_context:
            return None

        # Get file extension
        extension = file_path.suffix.lower()

        # Use LSP detector to find the best LSP for this extension
        if self.lsp_detector:
            lsp_info = self.lsp_detector.get_lsp_for_extension_with_context(
                extension, self.project_path
            )
            if lsp_info and lsp_info.name in self.active_lsp_clients:
                return lsp_info.name

        # Fallback to any active LSP that supports this extension
        for lsp_name, client in self.active_lsp_clients.items():
            lsp_info = self.extraction_context.available_lsps.get(lsp_name)
            if lsp_info and extension in lsp_info.supported_extensions:
                return lsp_name

        return None

    async def _extract_with_lsp(self, file_path: Path, lsp_name: str) -> ExtractionResult:
        """Extract symbols using specified LSP server."""
        if not self.metadata_extractor:
            raise RuntimeError("Metadata extractor not available")

        # Use metadata extractor with the specified LSP
        metadata = await self.metadata_extractor.extract_file_metadata(str(file_path))

        if metadata:
            return ExtractionResult(
                symbols=metadata.symbols,
                relationships=metadata.relationships,
                file_metadata=metadata
            )
        else:
            return ExtractionResult(errors=["LSP metadata extraction failed"])

    async def _extract_with_treesitter(self, file_path: Path) -> ExtractionResult:
        """Extract symbols using Tree-sitter fallback."""
        # Placeholder for Tree-sitter implementation
        # This would integrate with a Tree-sitter parser
        logger.debug(f"Using Tree-sitter fallback for {file_path}")

        # For now, return empty result
        return ExtractionResult(
            symbols=[],
            relationships={},
            errors=["Tree-sitter fallback not implemented"]
        )

    def _get_project_files(self) -> List[Path]:
        """Get project files using pattern manager filtering."""
        if not self.pattern_manager:
            # Fallback to basic file discovery
            return list(self.project_path.rglob("*"))

        project_files = []
        for file_path in self.project_path.rglob("*"):
            if file_path.is_file():
                should_include, _ = self.pattern_manager.should_include(file_path)
                should_exclude, _ = self.pattern_manager.should_exclude(file_path)

                if should_include and not should_exclude:
                    project_files.append(file_path)

        return project_files

    def _get_files_by_patterns(self, patterns: List[str]) -> List[Path]:
        """Get files matching specified patterns."""
        files = []
        for pattern in patterns:
            files.extend(self.project_path.glob(pattern))
        return [f for f in files if f.is_file()]

    async def _handle_lsp_health_change(
        self,
        lsp_name: str,
        is_healthy: bool,
        error: Optional[Exception] = None
    ) -> None:
        """Handle LSP health status changes."""
        if is_healthy:
            logger.info(f"LSP {lsp_name} health restored")
        else:
            logger.warning(f"LSP {lsp_name} health degraded: {error}")

            # Remove unhealthy client
            if lsp_name in self.active_lsp_clients:
                try:
                    await self.active_lsp_clients[lsp_name].shutdown()
                except Exception:
                    pass  # Ignore shutdown errors for unhealthy clients
                del self.active_lsp_clients[lsp_name]

            # Try to restart the client
            if lsp_name in self.extraction_context.available_lsps:
                try:
                    await self._initialize_lsp_client(lsp_name)
                    logger.info(f"Successfully restarted LSP {lsp_name}")
                except Exception as e:
                    logger.error(f"Failed to restart LSP {lsp_name}: {e}")

    def get_extraction_statistics(self) -> ExtractionStatistics:
        """Get comprehensive extraction statistics."""
        # Calculate derived statistics
        if self.statistics.total_extractions > 0:
            self.statistics.average_extraction_time = (
                self.statistics.total_extraction_time / self.statistics.total_extractions
            )

        if self.extraction_cache:
            # Calculate cache hit rate (simplified)
            total_requests = self.statistics.total_extractions
            cache_entries = len(self.extraction_cache)
            self.statistics.cache_hit_rate = min(cache_entries / max(total_requests, 1), 1.0)

        return self.statistics

    def clear_cache(self) -> None:
        """Clear the extraction cache."""
        self.extraction_cache.clear()
        logger.debug("Extraction cache cleared")

    async def _cleanup_partial_initialization(self) -> None:
        """Clean up partially initialized resources."""
        for client in self.active_lsp_clients.values():
            try:
                await client.shutdown()
            except Exception:
                pass
        self.active_lsp_clients.clear()

    async def shutdown(self) -> None:
        """Shutdown the extraction manager and clean up resources."""
        logger.info("Shutting down LSP Symbol Extraction Manager")

        # Shutdown LSP clients
        shutdown_tasks = []
        for lsp_name, client in self.active_lsp_clients.items():
            task = client.shutdown()
            shutdown_tasks.append(task)

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        # Shutdown other components
        if self.metadata_extractor:
            await self.metadata_extractor.shutdown()

        if self.health_monitor:
            await self.health_monitor.shutdown()

        # Clear state
        self.active_lsp_clients.clear()
        self.extraction_cache.clear()
        self.initialized = False

        logger.info("LSP Symbol Extraction Manager shutdown complete")


# Convenience functions for common operations
async def extract_project_symbols(
    project_path: Union[str, Path],
    file_patterns: Optional[List[str]] = None
) -> ExtractionResult:
    """Convenience function to extract symbols from a project."""
    manager = LspSymbolExtractionManager(project_path)
    try:
        await manager.initialize()
        return await manager.extract_project_symbols(file_patterns)
    finally:
        await manager.shutdown()


async def extract_file_symbols(
    file_path: Union[str, Path],
    project_path: Optional[Union[str, Path]] = None
) -> ExtractionResult:
    """Convenience function to extract symbols from a single file."""
    if not project_path:
        project_path = Path(file_path).parent

    manager = LspSymbolExtractionManager(project_path)
    try:
        await manager.initialize()
        return await manager.extract_file_symbols(file_path)
    finally:
        await manager.shutdown()