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
    - Single collection per project with metadata-based file type differentiation

Collection Architecture (Task 374.6):
    - ONE collection per project: _{project_id} format
    - project_id generated from project root path using calculate_tenant_id()
    - All file types (code, docs, tests, config, etc.) go to same collection
    - Files differentiated by metadata fields:
        * file_type: "code", "test", "docs", "config", "data", "build", "other"
        * branch: Current git branch (from get_current_branch())
        * project_id: Unique project identifier (from calculate_tenant_id())
    - No collection type suffixes (e.g., NO {project}-code, {project}-docs)

Metadata Enrichment:
    Note: Metadata enrichment (project_id, branch, file_type) happens at actual
    ingestion points (memory.py, client.py) when documents are stored to Qdrant.
    This module only sets up watches and determines target collection names.

Example:
    ```python
    from workspace_qdrant_mcp.core.auto_ingestion import AutoIngestionManager

    # Initialize and setup automatic ingestion
    manager = AutoIngestionManager(workspace_client, watch_manager)
    await manager.setup_project_watches()
    ```
"""

import asyncio
from loguru import logger
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..core.client import QdrantWorkspaceClient
from ..core.config import AutoIngestionConfig
from ..utils.project_detection import ProjectDetector, calculate_tenant_id
from ..utils.git_utils import get_current_branch
from ..utils.file_type_classifier import determine_file_type

# Import WatchToolsManager only when needed to prevent circular imports
# This will be imported in the constructor when actually used

# Import PatternManager for intelligent pattern management
try:
    from .pattern_manager import PatternManager
    PATTERN_MANAGER_AVAILABLE = True
except ImportError:
    PATTERN_MANAGER_AVAILABLE = False
    logger.debug("PatternManager not available - using hardcoded patterns")

# logger imported from loguru


def build_project_collection_name(project_root: Path) -> str:
    """
    Build project collection name using single-collection-per-project format.

    This implements Task 374.6 requirements:
    - Uses calculate_tenant_id() from project_detection module
    - Format: _{project_id} where project_id comes from calculate_tenant_id()
    - calculate_tenant_id() uses git remote URL if available, else path hash
    - Examples: _github_com_user_repo OR _path_abc123def456789a

    Args:
        project_root: Path to the project root directory

    Returns:
        Collection name in format _{project_id}

    Examples:
        >>> build_project_collection_name(Path("/path/to/repo"))
        '_github_com_user_repo'  # if git remote exists

        >>> build_project_collection_name(Path("/path/to/local"))
        '_path_abc123def456789a'  # if no git remote
    """
    project_id = calculate_tenant_id(project_root)
    return f"_{project_id}"


# Legacy function kept for backwards compatibility but deprecated
def normalize_collection_name_component(name: str) -> str:
    """Normalize collection name component (DEPRECATED - kept for backwards compatibility)."""
    return name.replace('-', '_').replace(' ', '_')


class ProjectPatterns:
    """Project-specific file patterns for automatic ingestion.

    This class uses PatternManager when available, with fallback patterns.
    """

    @classmethod
    def get_common_doc_patterns(cls) -> List[str]:
        """Get common documentation file patterns."""
        if PATTERN_MANAGER_AVAILABLE:
            try:
                pattern_manager = PatternManager()
                # TODO: In future, get these from PatternManager's document patterns
                # Return standard patterns
                return [
                    "*.md", "*.txt", "*.rst", "*.pdf", "*.epub",
                    "*.docx", "*.odt", "*.rtf"
                ]
            except Exception as e:
                logger.debug(f"Failed to use PatternManager for doc patterns: {e}")

        # Fallback hardcoded patterns
        return [
            "*.md", "*.txt", "*.rst", "*.pdf", "*.epub",
            "*.docx", "*.odt", "*.rtf",
        ]

    @classmethod
    def get_source_patterns_for_language(cls, language: str) -> List[str]:
        """Get source file patterns for a specific language."""
        if PATTERN_MANAGER_AVAILABLE:
            try:
                pattern_manager = PatternManager()
                # TODO: In future, get language-specific patterns from PatternManager
                # Return standard patterns from fallback
                pass
            except Exception as e:
                logger.debug(f"Failed to use PatternManager for language patterns: {e}")

        # Fallback hardcoded patterns by language
        source_patterns = {
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
        return source_patterns.get(language, [])

    @classmethod
    def get_all_source_patterns(cls) -> Dict[str, List[str]]:
        """Get all source patterns by language."""
        if PATTERN_MANAGER_AVAILABLE:
            try:
                pattern_manager = PatternManager()
                # TODO: In future, get all language patterns from PatternManager
                # For now, build from individual language queries
                pass
            except Exception as e:
                logger.debug(f"Failed to use PatternManager for all patterns: {e}")

        # Fallback: build from individual language calls
        languages = [
            "python", "javascript", "typescript", "web", "rust", "go",
            "java", "cpp", "csharp", "ruby", "php", "yaml", "json",
            "xml", "config", "shell", "sql", "docker"
        ]
        return {lang: cls.get_source_patterns_for_language(lang) for lang in languages}

    # Standard properties
    @property
    def COMMON_DOC_PATTERNS(self) -> List[str]:
        """Get common documentation patterns."""
        return self.get_common_doc_patterns()

    @property
    def SOURCE_PATTERNS(self) -> Dict[str, List[str]]:
        """Get all source patterns."""
        return self.get_all_source_patterns()

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
        # Media files (usually not needed for code search)
        "*.mp3",
        "*.mp4",
        "*.avi",
        "*.mov",
        "*.wmv",
        "*.flv",
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.ico",
        "*.svg",
        "*.webp",
        "*.psd",
        # Compressed archives
        "*.zip",
        "*.tar",
        "*.gz",
        "*.bz2",
        "*.7z",
        "*.rar",
        # Environment and secret files
        ".env",
        ".env.local",
        ".env.*",
        "*.pem",
        "*.key",
        "*.crt",
        "*.pfx",
        # Test coverage and reports
        "coverage/*",
        ".coverage",
        "*.cover",
        "htmlcov/*",
        ".pytest_cache/*",
        ".tox/*",
    ]

    @staticmethod
    def get_patterns_for_project(
        project_info: Dict[str, Any], config: AutoIngestionConfig
    ) -> List[str]:
        """Get file patterns for a project based on its type and configuration.

        Args:
            project_info: Project information from ProjectDetector
            config: Auto-ingestion configuration

        Returns:
            List of file patterns to watch
        """
        patterns = []

        # Always include common documentation patterns
        patterns.extend(ProjectPatterns.get_common_doc_patterns())

        # Add source patterns if configured
        if config.ingest_source_code:
            # Get detected ecosystems/languages from project
            ecosystems = project_info.get("detected_ecosystems", [])

            if ecosystems:
                # Add patterns for detected languages
                for lang in ecosystems:
                    lang_patterns = ProjectPatterns.get_source_patterns_for_language(lang)
                    patterns.extend(lang_patterns)
            else:
                # No specific languages detected - add all common source patterns
                all_patterns = ProjectPatterns.get_all_source_patterns()
                for lang_patterns in all_patterns.values():
                    patterns.extend(lang_patterns)

        # Add config patterns if configured
        if config.ingest_config_files:
            patterns.extend(ProjectPatterns.get_source_patterns_for_language("config"))
            patterns.extend(ProjectPatterns.get_source_patterns_for_language("yaml"))
            patterns.extend(ProjectPatterns.get_source_patterns_for_language("json"))
            patterns.extend(ProjectPatterns.get_source_patterns_for_language("docker"))

        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        for pattern in patterns:
            if pattern not in seen:
                seen.add(pattern)
                unique_patterns.append(pattern)

        return unique_patterns


class AutoIngestionManager:
    """Manager for automatic file ingestion on project initialization.

    This manager sets up automatic file watching and handles initial bulk ingestion
    when the MCP server starts. It integrates with the existing WatchToolsManager
    to provide seamless project-wide file indexing.

    Uses single-collection-per-project architecture (Task 374.6):
    - All files from a project go to single _{project_id} collection
    - Collection name generated using calculate_tenant_id()
    - Files differentiated by metadata:
        * file_type: Determined by determine_file_type()
        * branch: Determined by get_current_branch()
        * project_id: Determined by calculate_tenant_id()
    - Metadata enrichment happens at ingestion points (memory.py, client.py)
    """

    def __init__(
        self,
        workspace_client: QdrantWorkspaceClient,
        watch_manager: Any,  # WatchToolsManager type hint causes circular import
        config: Optional[AutoIngestionConfig] = None,
    ):
        """Initialize the auto-ingestion manager.

        Args:
            workspace_client: Qdrant workspace client for database operations
            watch_manager: Watch tools manager for file watching
            config: Auto-ingestion configuration (uses defaults if not provided)
        """
        self.workspace_client = workspace_client
        self.watch_manager = watch_manager
        self.config = config or AutoIngestionConfig()
        self.project_detector = ProjectDetector()

        logger.debug(
            "AutoIngestionManager initialized with config: enabled=%s, ingest_source=%s, ingest_config=%s",
            self.config.enabled,
            self.config.ingest_source_code,
            self.config.ingest_config_files,
        )

    async def setup_project_watches(self, project_path: Optional[str] = None) -> Dict[str, Any]:
        """Set up automatic file watches for the current or specified project.

        This method implements Task 374.6 single-collection-per-project architecture:
        1. Detects the project structure (main project and subprojects)
        2. Determines the target collection using _{project_id} format (calculate_tenant_id())
        3. Creates or ensures the collection exists
        4. Sets up file watches with appropriate patterns
        5. Optionally performs initial bulk ingestion

        Collection Routing (Task 374.6):
        - ALL file types go to SAME _{project_id} collection
        - No separate collections for docs, code, tests, etc.
        - Files differentiated by metadata fields (added at ingestion time):
            * file_type: "code", "test", "docs", "config", "data", "build", "other"
            * branch: Current git branch name
            * project_id: Unique project identifier

        Args:
            project_path: Path to the project root (uses current directory if not specified)

        Returns:
            Dictionary with setup results including success status and watch details
        """
        if not self.config.enabled:
            logger.info("Auto-ingestion is disabled in configuration")
            return {
                "success": True,
                "message": "Auto-ingestion is disabled",
                "watches_created": 0,
            }

        # Detect project information
        working_path = project_path or os.getcwd()
        project_info = self.project_detector.get_project_info(working_path)

        if not project_info or not project_info.get("is_git_repo"):
            logger.info("Not in a Git repository - skipping auto-ingestion setup")
            return {
                "success": False,
                "message": "Not in a Git repository",
                "watches_created": 0,
            }

        logger.info(f"Detected project: {project_info['main_project']}")

        # Get or create target collection using new single-collection format
        git_root = project_info.get("git_root")
        if not git_root:
            logger.error("No git root found in project info")
            return {
                "success": False,
                "message": "No git root found",
                "watches_created": 0,
            }

        project_root_path = Path(git_root)

        # Task 374.6: Use calculate_tenant_id() for collection name
        target_collection = build_project_collection_name(project_root_path)

        logger.info(
            f"Using single project collection: {target_collection} "
            f"(all file types will use this collection with metadata-based routing)"
        )

        # Get list of existing collections
        try:
            collections = await self.workspace_client.list_collections()
            collection_names = [c["name"] for c in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return {
                "success": False,
                "message": f"Failed to list collections: {e}",
                "watches_created": 0,
            }

        # Create collection if it doesn't exist
        needs_creation = target_collection not in collection_names

        if needs_creation:
            try:
                logger.info(f"Creating new project collection: {target_collection}")
                await self.workspace_client.create_collection(target_collection)
                logger.info(f"Successfully created collection: {target_collection}")
            except Exception as e:
                logger.error(f"Failed to create collection {target_collection}: {e}")
                return {
                    "success": False,
                    "message": f"Failed to create collection: {e}",
                    "watches_created": 0,
                }

        # Create watch for the main project
        watch_result = await self._create_project_watch(
            project_info, target_collection, git_root
        )

        # Set up watches for subprojects if they exist
        subproject_watches = []
        if project_info.get("subprojects"):
            for subproject in project_info["subprojects"]:
                # Subprojects use the same collection as main project
                # Differentiated by metadata fields (project_id, branch, file_type)
                try:
                    subproject_result = await self._create_subproject_watch(
                        project_info, subproject, target_collection, git_root
                    )
                    subproject_watches.append(subproject_result)
                except Exception as e:
                    logger.error(f"Failed to create watch for subproject {subproject}: {e}")

        return {
            "success": watch_result.get("success", False),
            "message": f"Auto-ingestion setup completed for {project_info['main_project']}",
            "collection": target_collection,
            "collection_created": needs_creation,
            "main_watch": watch_result,
            "subproject_watches": subproject_watches,
            "watches_created": 1 + len(subproject_watches),
            "routing_strategy": "metadata-based",  # Task 374.6
            "metadata_fields": ["file_type", "branch", "project_id"],  # Task 374.6
        }

    async def _create_project_watch(
        self, project_info: Dict[str, Any], collection: str, project_path: str
    ) -> Dict[str, Any]:
        """Create a watch for the main project.

        Note: Metadata enrichment (file_type, branch, project_id) happens at actual
        ingestion points (memory.py, client.py) when files are stored to Qdrant.
        """
        patterns = ProjectPatterns.get_patterns_for_project(project_info, self.config)

        watch_id = (
            f"auto-{project_info['main_project']}-{int(datetime.now().timestamp())}"
        )

        logger.info(
            f"Creating automatic watch for project: {project_info['main_project']}"
        )
        logger.debug(f"Watch patterns: {patterns}")
        logger.debug(
            f"Files will be stored to single collection '{collection}' "
            f"with metadata-based differentiation (file_type, branch, project_id)"
        )

        result = await self.watch_manager.add_watch_folder(
            path=str(Path(project_path).resolve()),
            collection=collection,
            patterns=patterns,
            ignore_patterns=ProjectPatterns.IGNORE_PATTERNS,
            auto_ingest=True,
            recursive=True,
            recursive_depth=-1,  # Allow unlimited recursive depth for comprehensive ingestion
            debounce_seconds=self.config.debounce_seconds,
            watch_id=watch_id,
        )

        if result.get("success"):
            logger.info(f"Successfully created project watch: {watch_id}")
        else:
            logger.error(f"Failed to create project watch: {result.get('error')}")

        return result

    async def _create_subproject_watch(
        self,
        project_info: Dict[str, Any],
        subproject: str,
        collection: str,
        base_path: str,
    ) -> Dict[str, Any]:
        """Create a watch for a subproject.

        Subprojects share the same collection as the main project (Task 374.6).
        They are differentiated by metadata fields (project_id, branch, file_type).
        Metadata enrichment happens at ingestion points (memory.py, client.py).
        """
        patterns = ProjectPatterns.get_patterns_for_project(project_info, self.config)

        # Subproject path is relative to base project path
        subproject_path = str(Path(base_path) / subproject)

        watch_id = f"auto-{project_info['main_project']}-{subproject}-{int(datetime.now().timestamp())}"

        logger.info(f"Creating watch for subproject: {subproject}")
        logger.debug(
            f"Subproject will use same collection '{collection}' as main project "
            f"with metadata-based differentiation"
        )

        result = await self.watch_manager.add_watch_folder(
            path=subproject_path,
            collection=collection,
            patterns=patterns,
            ignore_patterns=ProjectPatterns.IGNORE_PATTERNS,
            auto_ingest=True,
            recursive=True,
            recursive_depth=-1,
            debounce_seconds=self.config.debounce_seconds,
            watch_id=watch_id,
        )

        if result.get("success"):
            logger.info(f"Successfully created subproject watch: {watch_id}")
        else:
            logger.error(f"Failed to create subproject watch: {result.get('error')}")

        return result
