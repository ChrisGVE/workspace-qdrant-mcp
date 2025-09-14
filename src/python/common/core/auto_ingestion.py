"""
Automatic file ingestion system for project initialization.

This module provides automatic file watching and ingestion capabilities that activate
during MCP server startup. It detects project types, sets up appropriate file watches,
and handles initial bulk ingestion with progress tracking and rate limiting.

Key Features:
    - Automatic project type detection and watch configuration
    - Configurable file patterns for different project types
    - Progress tracking for initial bulk ingestion operations
    - Rate limiting to prevent system overload during large imports
    - Persistent state management across server restarts
    - Integration with existing WatchToolsManager infrastructure

Example:
    ```python
    from workspace_qdrant_mcp.core.auto_ingestion import AutoIngestionManager

    # Initialize and setup automatic ingestion
    manager = AutoIngestionManager(workspace_client, watch_manager)
    await manager.setup_project_watches()
    ```
"""

import asyncio
from common.logging import get_logger
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..core.client import QdrantWorkspaceClient
# from ..core.collection_naming import (
#     build_project_collection_name,
#     normalize_collection_name_component
# )
# TODO: These functions don't exist - fix imports after Task 175 integration

# Temporary implementations for compatibility
def build_project_collection_name(project: str, suffix: str) -> str:
    """Temporary implementation - build project collection name"""
    return f"{project.replace('-', '_')}-{suffix}"

def normalize_collection_name_component(name: str) -> str:
    """Temporary implementation - normalize collection name component"""
    return name.replace('-', '_').replace(' ', '_')
from ..core.config import AutoIngestionConfig
from workspace_qdrant_mcp.tools.watch_management import WatchToolsManager
from ..utils.project_detection import ProjectDetector

logger = get_logger(__name__)


class ProjectPatterns:
    """Project-specific file patterns for automatic ingestion."""

    # Common documentation and text files
    COMMON_DOC_PATTERNS = [
        "*.md",
        "*.txt",
        "*.rst",
        "*.pdf",
        "*.epub",
        "*.docx",
        "*.odt",
        "*.rtf",
    ]

    # Source code patterns by language/framework
    SOURCE_PATTERNS = {
        "python": ["*.py", "*.pyx", "*.pyi"],
        "javascript": ["*.js", "*.jsx", "*.mjs"],
        "typescript": ["*.ts", "*.tsx", "*.d.ts"],
        "web": ["*.html", "*.css", "*.scss", "*.sass", "*.vue"],
        "rust": ["*.rs"],
        "go": ["*.go"],
        "java": ["*.java", "*.kt", "*.scala"],
        "cpp": ["*.cpp", "*.cxx", "*.cc", "*.c", "*.h", "*.hpp"],
        "csharp": ["*.cs", "*.fs", "*.vb"],
        "ruby": ["*.rb", "*.erb"],
        "php": ["*.php", "*.phtml"],
        "yaml": ["*.yml", "*.yaml"],
        "json": ["*.json", "*.jsonc"],
        "xml": ["*.xml", "*.xsd", "*.xsl"],
        "config": ["*.ini", "*.cfg", "*.conf", "*.toml", "*.properties"],
        "shell": ["*.sh", "*.bash", "*.zsh", "*.fish"],
        "sql": ["*.sql", "*.ddl", "*.dml"],
        "docker": ["Dockerfile*", "*.dockerfile", "docker-compose*.yml"],
    }

    # Universal ignore patterns
    IGNORE_PATTERNS = [
        # Version control
        ".git/*",
        ".svn/*",
        ".hg/*",
        ".bzr/*",
        # Package managers and dependencies
        "node_modules/*",
        "__pycache__/*",
        ".pyc",
        "vendor/*",
        "target/*",
        "build/*",
        "dist/*",
        "out/*",
        # IDE and editor files
        ".vscode/*",
        ".idea/*",
        "*.swp",
        "*.swo",
        "*~",
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
        # Temporary and log files
        "*.tmp",
        "*.temp",
        "*.log",
        "*.pid",
        "*.lock",
        "*.cache",
        "*.bak",
        "*.backup",
        # Build artifacts
        "*.o",
        "*.obj",
        "*.exe",
        "*.dll",
        "*.so",
        "*.dylib",
        "*.class",
        "*.jar",
        "*.war",
        "*.ear",
        # Large media files (optional - could be made configurable)
        "*.mp4",
        "*.avi",
        "*.mkv",
        "*.mov",
        "*.wmv",
        "*.mp3",
        "*.wav",
        "*.flac",
        "*.aac",
        "*.ogg",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.gif",
        "*.bmp",
        "*.tiff",
        # Archives
        "*.zip",
        "*.tar",
        "*.gz",
        "*.bz2",
        "*.xz",
        "*.7z",
        "*.rar",
    ]

    @classmethod
    def get_patterns_for_project(
        cls, project_info: Dict[str, Any], config: AutoIngestionConfig
    ) -> List[str]:
        """
        Get file patterns appropriate for the detected project.

        Args:
            project_info: Project detection results
            config: Auto-ingestion configuration

        Returns:
            List of file patterns to include
        """
        patterns = []

        # Always include common documentation if enabled
        if config.include_common_files:
            patterns.extend(cls.COMMON_DOC_PATTERNS)

        # Include source patterns if enabled
        if config.include_source_files:
            project_path = Path(project_info.get("path", "."))
            detected_languages = cls._detect_project_languages(project_path)

            for language in detected_languages:
                if language in cls.SOURCE_PATTERNS:
                    patterns.extend(cls.SOURCE_PATTERNS[language])

        return list(set(patterns))  # Remove duplicates

    @classmethod
    def _detect_project_languages(cls, project_path: Path) -> Set[str]:
        """Detect programming languages used in the project."""
        languages = set()

        # Check for language indicators in the project
        indicators = {
            "python": [
                "requirements.txt",
                "setup.py",
                "pyproject.toml",
                "Pipfile",
                "*.py",
            ],
            "javascript": ["package.json", "package-lock.json", "*.js", "*.jsx"],
            "typescript": ["tsconfig.json", "*.ts", "*.tsx"],
            "rust": ["Cargo.toml", "Cargo.lock", "*.rs"],
            "go": ["go.mod", "go.sum", "*.go"],
            "java": ["pom.xml", "build.gradle", "*.java"],
            "cpp": ["CMakeLists.txt", "Makefile", "*.cpp", "*.c"],
            "csharp": ["*.csproj", "*.sln", "*.cs"],
            "ruby": ["Gemfile", "Gemfile.lock", "*.rb"],
            "php": ["composer.json", "*.php"],
            "docker": ["Dockerfile", "docker-compose.yml"],
        }

        try:
            # Sample files to check (don't scan entire tree for performance)
            files_to_check = []
            for item in project_path.rglob("*"):
                if (
                    item.is_file() and len(files_to_check) < 100
                ):  # Limit for performance
                    files_to_check.append(item)
                elif len(files_to_check) >= 100:
                    break

            # Check against indicators
            for language, patterns in indicators.items():
                for pattern in patterns:
                    if any(item.match(pattern) for item in files_to_check):
                        languages.add(language)
                        break

        except Exception as e:
            logger.warning(f"Error detecting project languages: {e}")

        return languages


class IngestionProgressTracker:
    """Track progress of bulk file ingestion operations."""

    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.start_time = None
        self.current_batch = 0
        self.total_batches = 0
        self.current_file = None
        self.errors = []

    def start(self, total_files: int, batch_size: int):
        """Start tracking progress."""
        self.total_files = total_files
        self.total_batches = (total_files + batch_size - 1) // batch_size
        self.start_time = datetime.now(timezone.utc)
        self.processed_files = 0
        self.failed_files = 0
        self.current_batch = 0
        self.errors = []

        logger.info(
            f"Starting bulk ingestion: {total_files} files in {self.total_batches} batches"
        )

    def start_batch(self, batch_num: int, batch_files: List[str]):
        """Start processing a new batch."""
        self.current_batch = batch_num
        logger.info(
            f"Processing batch {batch_num}/{self.total_batches} ({len(batch_files)} files)"
        )

    def start_file(self, file_path: str):
        """Start processing a specific file."""
        self.current_file = file_path

    def file_completed(self, file_path: str, success: bool, error: str = None):
        """Mark a file as completed."""
        if success:
            self.processed_files += 1
        else:
            self.failed_files += 1
            if error:
                self.errors.append({"file": file_path, "error": error})

        # Log progress every 10 files or on failure
        if self.processed_files % 10 == 0 or not success:
            self.log_progress()

    def log_progress(self):
        """Log current progress."""
        if self.start_time:
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            files_per_sec = self.processed_files / elapsed if elapsed > 0 else 0

            logger.info(
                f"Ingestion progress: {self.processed_files}/{self.total_files} files "
                f"({self.failed_files} failed) - {files_per_sec:.1f} files/sec"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        elapsed = None
        if self.start_time:
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "success_rate": self.processed_files / max(1, self.total_files),
            "elapsed_seconds": elapsed,
            "files_per_second": self.processed_files / max(1, elapsed or 1),
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "errors": self.errors[-10:],  # Last 10 errors only
        }


class AutoIngestionManager:
    """Manages automatic file ingestion for project initialization."""

    def __init__(
        self,
        workspace_client: QdrantWorkspaceClient,
        watch_manager: WatchToolsManager,
        config: Optional[AutoIngestionConfig] = None,
    ):
        """
        Initialize the auto-ingestion manager.

        Args:
            workspace_client: Workspace client for database operations
            watch_manager: Watch tools manager for folder watching
            config: Configuration settings (defaults to environment-based)
        """
        self.workspace_client = workspace_client
        self.watch_manager = watch_manager
        
        # Handle case where config is passed as dict instead of AutoIngestionConfig object
        if isinstance(config, dict):
            self.config = AutoIngestionConfig(**config)
        else:
            self.config = config or AutoIngestionConfig()
            
        self.project_detector = ProjectDetector()
        self.progress_tracker = IngestionProgressTracker()
        self._rate_limit_semaphore = asyncio.Semaphore(self.config.max_files_per_batch)

    async def setup_project_watches(self, project_path: str = ".") -> Dict[str, Any]:
        """
        Set up automatic file watching for the current project.

        Args:
            project_path: Path to the project directory

        Returns:
            Dict with setup results and created watches
        """
        if not self.config.enabled:
            return {
                "success": True,
                "message": "Auto-ingestion disabled by configuration",
                "watches_created": [],
            }

        try:
            logger.info("Setting up automatic file ingestion for project")

            # Detect project information
            project_info = await self._detect_project_info(project_path)
            logger.info(f"Detected project: {project_info['main_project']}")

            # Get available collections
            collections = self.workspace_client.list_collections()
            if not collections:
                logger.warning("No collections available - skipping watch setup")
                return {
                    "success": False,
                    "error": "No collections available for ingestion",
                    "watches_created": [],
                }

            # Select primary collection for the main project
            primary_collection, needs_creation = self._select_primary_collection(
                project_info, collections
            )
            
            # Create the primary collection if it doesn't exist
            if needs_creation:
                logger.info(f"Creating target collection for auto-ingestion: {primary_collection}")
                try:
                    await self._create_collection_for_auto_ingestion(primary_collection)
                    # Update collections list to include the newly created collection
                    collections.append(primary_collection)
                except Exception as e:
                    logger.error(f"Failed to create target collection '{primary_collection}': {e}")
                    return {
                        "success": False,
                        "error": f"Failed to create target collection: {str(e)}",
                        "watches_created": [],
                    }

            results = []

            # Create watch for main project if enabled
            if self.config.auto_create_watches:
                main_watch_result = await self._create_project_watch(
                    project_info, primary_collection, project_path
                )
                results.append(main_watch_result)

                # Create watches for subprojects if they exist
                for subproject in project_info.get("subprojects", []):
                    subproject_collection = (
                        f"{project_info['main_project']}.{subproject}"
                    )
                    if subproject_collection in collections:
                        subproject_watch_result = await self._create_subproject_watch(
                            project_info, subproject, subproject_collection
                        )
                        results.append(subproject_watch_result)

            # Perform initial bulk ingestion if any watches were created successfully
            successful_watches = [r for r in results if r.get("success")]
            if successful_watches:
                bulk_ingestion_result = await self._perform_initial_bulk_ingestion(
                    project_path, primary_collection
                )

                return {
                    "success": True,
                    "message": f"Auto-ingestion setup completed for {project_info['main_project']}",
                    "project_info": project_info,
                    "primary_collection": primary_collection,
                    "watches_created": results,
                    "bulk_ingestion": bulk_ingestion_result,
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create any watches",
                    "project_info": project_info,
                    "watches_created": results,
                }

        except Exception as e:
            logger.error(f"Failed to setup project watches: {e}")
            return {
                "success": False,
                "error": f"Setup failed: {str(e)}",
                "watches_created": [],
            }

    async def _detect_project_info(self, project_path: str) -> Dict[str, Any]:
        """Detect project information asynchronously."""
        # Run project detection in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        project_info = await loop.run_in_executor(
            None, self.project_detector.get_project_info, project_path
        )
        return project_info

    def _select_primary_collection(
        self, project_info: Dict[str, Any], collections: List[str]
    ) -> tuple[str, bool]:
        """Select the primary collection for the project based on configuration.
        
        Returns:
            tuple: (collection_name, needs_creation) where needs_creation indicates
                  if the collection needs to be created
        """
        main_project = project_info["main_project"]
        target_suffix = self.config.target_collection_suffix
        
        # Validate configuration and provide better error messages
        if not target_suffix:
            logger.info(
                "auto_ingestion.target_collection_suffix not configured - using intelligent fallback selection. "
                "Will prefer existing project collections, then common collections like 'scratchbook', "
                "or create a default collection if needed."
            )
        
        # First preference: exact match for configured target suffix
        if target_suffix:
            target_collection = build_project_collection_name(main_project, target_suffix)
            if target_collection in collections:
                logger.info(f"Selected existing target collection for auto-ingestion: {target_collection}")
                return target_collection, False
            else:
                # Target collection doesn't exist - check if other project collections exist first
                logger.info(f"Target collection '{target_collection}' does not exist")
                logger.info(f"Available collections: {collections}")
                
                # Check if any project-related collections exist before creating new one
                normalized_project = normalize_collection_name_component(main_project)
                project_collections = [c for c in collections if c.startswith(f"{normalized_project}-") or c.startswith(f"{normalized_project}.")]
                if project_collections:
                    selected = project_collections[0]
                    logger.info(f"Selected existing project collection for auto-ingestion: {selected}")
                    return selected, False
                
                # No project collections exist, so create the target collection
                return target_collection, True

        # Second preference: exact project name match (legacy behavior)
        if main_project in collections:
            logger.info(f"Selected project collection for auto-ingestion: {main_project}")
            return main_project, False

        # Third preference: any collection that starts with the project name
        normalized_project = normalize_collection_name_component(main_project)
        matching = [c for c in collections if c.startswith(f"{normalized_project}.") or c.startswith(f"{normalized_project}-")]
        if matching:
            selected = matching[0]
            logger.info(f"Selected first matching project collection for auto-ingestion: {selected}")
            return selected, False

        # Fourth preference: check for common standalone collections
        common_collections = ["scratchbook", "notes", "documents", "reference"]
        for common in common_collections:
            if common in collections:
                logger.info(f"Selected common collection for auto-ingestion: {common}")
                return common, False

        # Final fallback: create a default collection if no collections exist
        if not collections:
            default_collection = f"{main_project}-scratchbook" if not target_suffix else f"{main_project}-{target_suffix}"
            logger.warning(f"No collections available. Will create default collection: {default_collection}")
            return default_collection, True
        else:
            # Use first available collection as absolute fallback
            fallback = collections[0]
            logger.warning(
                f"No target collection found for suffix '{target_suffix}' and project '{main_project}'. "
                f"Falling back to existing collection: {fallback}"
            )
            return fallback, False

    async def _create_project_watch(
        self, project_info: Dict[str, Any], collection: str, project_path: str
    ) -> Dict[str, Any]:
        """Create a watch for the main project."""
        patterns = ProjectPatterns.get_patterns_for_project(project_info, self.config)

        watch_id = (
            f"auto-{project_info['main_project']}-{int(datetime.now().timestamp())}"
        )

        logger.info(
            f"Creating automatic watch for project: {project_info['main_project']}"
        )
        logger.debug(f"Watch patterns: {patterns}")

        result = await self.watch_manager.add_watch_folder(
            path=str(Path(project_path).resolve()),
            collection=collection,
            patterns=patterns,
            ignore_patterns=ProjectPatterns.IGNORE_PATTERNS,
            auto_ingest=True,
            recursive=True,
            recursive_depth=self.config.recursive_depth,
            debounce_seconds=self.config.debounce_seconds,
            watch_id=watch_id,
        )

        if result.get("success"):
            logger.info(f"Successfully created project watch: {watch_id}")
        else:
            logger.error(f"Failed to create project watch: {result.get('error')}")

        return result

    async def _create_subproject_watch(
        self, project_info: Dict[str, Any], subproject: str, collection: str
    ) -> Dict[str, Any]:
        """Create a watch for a subproject."""
        # Find subproject path
        subproject_path = None
        for detailed_submodule in project_info.get("detailed_submodules", []):
            if detailed_submodule.get("project_name") == subproject:
                subproject_path = detailed_submodule.get("local_path")
                break

        if not subproject_path or not Path(subproject_path).exists():
            return {
                "success": False,
                "error": f"Subproject path not found or doesn't exist: {subproject}",
                "subproject": subproject,
            }

        patterns = ProjectPatterns.get_patterns_for_project(project_info, self.config)
        watch_id = f"auto-{project_info['main_project']}-{subproject}-{int(datetime.now().timestamp())}"

        logger.info(f"Creating automatic watch for subproject: {subproject}")

        result = await self.watch_manager.add_watch_folder(
            path=str(Path(subproject_path).resolve()),
            collection=collection,
            patterns=patterns,
            ignore_patterns=ProjectPatterns.IGNORE_PATTERNS,
            auto_ingest=True,
            recursive=True,
            recursive_depth=self.config.recursive_depth,
            debounce_seconds=self.config.debounce_seconds,
            watch_id=watch_id,
        )

        if result.get("success"):
            logger.info(f"Successfully created subproject watch: {watch_id}")
        else:
            logger.error(f"Failed to create subproject watch: {result.get('error')}")

        return result

    async def _perform_initial_bulk_ingestion(
        self, project_path: str, collection: str
    ) -> Dict[str, Any]:
        """Perform initial bulk ingestion of existing files."""
        try:
            logger.info("Starting initial bulk ingestion of existing files")

            # Find all matching files
            project_info = {
                "path": project_path
            }  # Minimal project info for pattern detection
            patterns = ProjectPatterns.get_patterns_for_project(
                project_info, self.config
            )

            files_to_ingest = []
            project_path_obj = Path(project_path).resolve()

            for pattern in patterns:
                matching_files = list(project_path_obj.rglob(pattern))
                files_to_ingest.extend(matching_files)

            # Filter out ignored patterns and size limits
            files_to_ingest = self._filter_files_for_ingestion(files_to_ingest)

            if not files_to_ingest:
                logger.info("No files found for initial bulk ingestion")
                return {
                    "success": True,
                    "message": "No files to ingest",
                    "files_processed": 0,
                }

            # Start progress tracking
            self.progress_tracker.start(
                len(files_to_ingest), self.config.max_files_per_batch
            )

            # Process files in batches with rate limiting
            results = []
            for i in range(0, len(files_to_ingest), self.config.max_files_per_batch):
                batch = files_to_ingest[i : i + self.config.max_files_per_batch]
                batch_num = (i // self.config.max_files_per_batch) + 1

                self.progress_tracker.start_batch(batch_num, [str(f) for f in batch])

                # Process batch with rate limiting
                batch_tasks = []
                for file_path in batch:
                    task = self._ingest_single_file(file_path, collection)
                    batch_tasks.append(task)

                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

                # Track results
                for j, result in enumerate(batch_results):
                    file_path = batch[j]
                    if isinstance(result, Exception):
                        self.progress_tracker.file_completed(
                            str(file_path), False, str(result)
                        )
                    else:
                        success = result.get("success", False)
                        error = result.get("error") if not success else None
                        self.progress_tracker.file_completed(
                            str(file_path), success, error
                        )

                results.extend(batch_results)

                # Rate limiting delay between batches
                if batch_num < self.progress_tracker.total_batches:
                    await asyncio.sleep(self.config.batch_delay_seconds)

            summary = self.progress_tracker.get_summary()
            logger.info(
                f"Bulk ingestion completed: {summary['processed_files']}/{summary['total_files']} files processed "
                f"({summary['failed_files']} failed) in {summary['elapsed_seconds']:.1f}s"
            )

            return {
                "success": True,
                "message": "Initial bulk ingestion completed",
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"Initial bulk ingestion failed: {e}")
            return {
                "success": False,
                "error": f"Bulk ingestion failed: {str(e)}",
                "summary": self.progress_tracker.get_summary(),
            }

    def _filter_files_for_ingestion(self, files: List[Path]) -> List[Path]:
        """Filter files based on ignore patterns and size limits."""
        filtered_files = []
        max_size_bytes = self.config.max_file_size_mb * 1024 * 1024

        for file_path in files:
            try:
                # Check if file should be ignored
                if self._should_ignore_file(file_path):
                    continue

                # Check file size
                if file_path.stat().st_size > max_size_bytes:
                    logger.debug(
                        f"Skipping large file: {file_path} ({file_path.stat().st_size / 1024 / 1024:.1f}MB)"
                    )
                    continue

                # Check if file is readable
                if not file_path.is_file() or not os.access(file_path, os.R_OK):
                    continue

                filtered_files.append(file_path)

            except (OSError, PermissionError) as e:
                logger.debug(f"Skipping inaccessible file {file_path}: {e}")
                continue

        return filtered_files

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if a file should be ignored based on ignore patterns."""
        # Convert to relative path for pattern matching
        try:
            file_str = str(file_path)

            for pattern in ProjectPatterns.IGNORE_PATTERNS:
                # Simple glob-like matching
                if pattern.endswith("/*"):
                    # Directory pattern
                    dir_pattern = pattern[:-2]  # Remove /*
                    if f"/{dir_pattern}/" in file_str or file_str.endswith(
                        f"/{dir_pattern}"
                    ):
                        return True
                elif "*" in pattern:
                    # Wildcard pattern - use simple matching
                    import fnmatch

                    if fnmatch.fnmatch(file_path.name, pattern):
                        return True
                else:
                    # Exact pattern
                    if pattern in file_str:
                        return True

        except Exception as e:
            logger.debug(f"Error checking ignore patterns for {file_path}: {e}")

        return False

    async def _create_collection_for_auto_ingestion(self, collection_name: str) -> None:
        """Create a collection specifically for auto-ingestion.
        
        This method creates a collection with the appropriate configuration
        for auto-ingestion, including dense and sparse vector support.
        
        Args:
            collection_name: Name of the collection to create
            
        Raises:
            Exception: If collection creation fails
        """
        try:
            # Import here to avoid circular dependencies
            from ..core.collections import CollectionConfig, WorkspaceCollectionManager
            from qdrant_client import QdrantClient
            from ..core.config import Config
            
            # Get the workspace client from the project
            # We need to create a collection config for auto-ingestion
            config = CollectionConfig(
                name=collection_name,
                description=f"Auto-created collection for auto-ingestion: {collection_name}",
                collection_type="scratchbook",  # Default type for auto-ingestion
                vector_size=self._get_vector_size(),
                enable_sparse_vectors=True,  # Enable sparse vectors for better search
            )
            
            # Create a temporary collection manager to handle the creation
            from qdrant_client import QdrantClient
            from ..core.config import Config
            
            # Get client configuration
            from .ssl_config import suppress_qdrant_ssl_warnings
            full_config = Config()
            with suppress_qdrant_ssl_warnings():
                client = QdrantClient(**full_config.qdrant_client_config)
            collection_manager = WorkspaceCollectionManager(client, full_config)
            
            # Use the collection manager's method to ensure proper creation
            collection_manager._ensure_collection_exists(config)
            
            logger.info(f"Successfully created collection for auto-ingestion: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}' for auto-ingestion: {e}")
            raise
    
    def _get_vector_size(self) -> int:
        """Get the vector dimension size for the embedding model.
        
        Returns:
            int: Vector dimension size (384 for all-MiniLM-L6-v2)
        """
        # This should match the embedding service configuration
        model_sizes = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-m3": 1024,
        }
        
        # Default to 384 if model not found
        return model_sizes.get(
            getattr(self.config, 'model', 'sentence-transformers/all-MiniLM-L6-v2'), 
            384
        )

    async def _ingest_single_file(
        self, file_path: Path, collection: str
    ) -> Dict[str, Any]:
        """Ingest a single file with rate limiting."""
        async with self._rate_limit_semaphore:
            try:
                self.progress_tracker.start_file(str(file_path))

                # Read file content
                content = file_path.read_text(encoding="utf-8", errors="ignore")

                # Add document using existing infrastructure
                from ..tools.documents import add_document

                result = await add_document(
                    self.workspace_client,
                    content=content,
                    collection=collection,
                    metadata={
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "file_extension": file_path.suffix,
                        "file_size": file_path.stat().st_size,
                        "ingestion_source": "auto_bulk_ingestion",
                        "ingestion_time": datetime.now(timezone.utc).isoformat(),
                    },
                    document_id=f"auto-bulk-{hash(str(file_path))}",
                    chunk_text=True,
                )

                return result

            except Exception as e:
                logger.debug(f"Failed to ingest file {file_path}: {e}")
                return {"success": False, "error": str(e), "file_path": str(file_path)}
