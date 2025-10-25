"""
Collection Manager Integration with Collision Detection System.

This module provides integration between the collision detection system
and the existing collection management infrastructure. It demonstrates
how to use the collision detection system for safe collection creation
and management operations.

Key Features:
    - Safe collection creation with collision prevention
    - Integration with existing naming validation
    - Automatic suggestion generation for conflicts
    - Performance monitoring and statistics
    - Error handling and recovery mechanisms

Integration Points:
    - QdrantWorkspaceClient: Main client integration
    - Collection creation workflows
    - Naming validation integration
    - Conflict resolution strategies

Example Usage:
    ```python
    # Initialize collision-aware collection manager
    manager = CollisionAwareCollectionManager(qdrant_client)
    await manager.initialize()

    # Safe collection creation
    try:
        result = await manager.create_collection_safely(
            "my-project-docs",
            CollectionCategory.PROJECT
        )
        print(f"Created collection: {result.collection_name}")
    except CollectionCollisionError as e:
        print(f"Collision detected: {e.collision_result.collision_reason}")
        print(f"Suggestions: {e.collision_result.suggested_alternatives}")
    ```
"""

from dataclasses import dataclass
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import CollectionInfo

from .collection_naming_validation import CollectionNamingValidator, NamingConfiguration
from .collision_detection import CollisionDetector, CollisionResult, CollisionSeverity
from .metadata_schema import CollectionCategory, MultiTenantMetadataSchema


class CollectionCollisionError(Exception):
    """Exception raised when a collection collision is detected."""

    def __init__(self, message: str, collision_result: CollisionResult):
        """
        Initialize collision error.

        Args:
            message: Error message
            collision_result: Detailed collision analysis result
        """
        super().__init__(message)
        self.collision_result = collision_result


@dataclass
class CollectionCreationResult:
    """Result of collection creation operation."""

    success: bool
    collection_name: str
    category: CollectionCategory
    metadata: MultiTenantMetadataSchema | None = None
    collision_result: CollisionResult | None = None
    creation_time_ms: float = 0.0
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class CollisionAwareCollectionManager:
    """
    Collection manager with integrated collision detection.

    This class provides a high-level interface for safe collection
    management operations that automatically detect and prevent
    naming collisions while providing intelligent alternatives.
    """

    def __init__(self, qdrant_client: QdrantClient,
                 naming_config: NamingConfiguration | None = None):
        """
        Initialize collision-aware collection manager.

        Args:
            qdrant_client: Qdrant client for database operations
            naming_config: Optional naming configuration
        """
        self.qdrant_client = qdrant_client
        self.naming_config = naming_config or NamingConfiguration()

        # Initialize components
        self.naming_validator = CollectionNamingValidator(self.naming_config)
        self.collision_detector = CollisionDetector(qdrant_client, self.naming_validator)

        # State tracking
        self._initialized = False
        self._statistics = {
            'collections_created': 0,
            'collisions_detected': 0,
            'suggestions_generated': 0,
            'creation_attempts': 0
        }

        logger.info("Initialized collision-aware collection manager")

    async def initialize(self):
        """Initialize the collection manager and collision detection system."""
        if self._initialized:
            return

        logger.info("Initializing collision-aware collection manager...")

        try:
            # Initialize collision detector
            await self.collision_detector.initialize()

            self._initialized = True
            logger.info("Collection manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize collection manager: {e}")
            raise

    async def create_collection_safely(self, collection_name: str,
                                     category: CollectionCategory,
                                     project_context: str | None = None,
                                     auto_resolve_conflicts: bool = False) -> CollectionCreationResult:
        """
        Create a collection with comprehensive collision detection.

        Args:
            collection_name: Name of the collection to create
            category: Category of the collection
            project_context: Optional project context
            auto_resolve_conflicts: Whether to automatically resolve minor conflicts

        Returns:
            CollectionCreationResult with detailed outcome

        Raises:
            CollectionCollisionError: If collision is detected and cannot be resolved
        """
        import time
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        self._statistics['creation_attempts'] += 1

        logger.info(f"Creating collection safely: {collection_name} (category: {category})")

        try:
            # Check for collisions first
            collision_result = await self.collision_detector.check_collection_collision(
                collection_name, category, project_context
            )

            # Handle collision detection results
            if collision_result.has_collision:
                self._statistics['collisions_detected'] += 1

                if collision_result.severity == CollisionSeverity.BLOCKING:
                    # Hard collision - cannot proceed
                    logger.error(f"Blocking collision detected for {collection_name}: {collision_result.collision_reason}")
                    raise CollectionCollisionError(
                        f"Cannot create collection '{collection_name}': {collision_result.collision_reason}",
                        collision_result
                    )

                elif collision_result.severity == CollisionSeverity.WARNING and not auto_resolve_conflicts:
                    # Soft collision - warn but allow if auto-resolve is disabled
                    logger.warning(f"Warning collision detected for {collection_name}: {collision_result.collision_reason}")
                    if len(collision_result.suggested_alternatives) > 0:
                        self._statistics['suggestions_generated'] += 1

                    raise CollectionCollisionError(
                        f"Collection '{collection_name}' has conflicts: {collision_result.collision_reason}",
                        collision_result
                    )

            # Use collision protection during creation
            async with self.collision_detector.create_collection_guard(collection_name):

                # Create the collection in Qdrant
                # Note: This is a simplified example - real implementation would
                # include vector configuration, index parameters, etc.
                self._create_qdrant_collection(collection_name, category)

                # Register successful creation
                await self.collision_detector.register_collection_creation(
                    collection_name, category, project_context
                )

                # Generate metadata
                metadata = self._generate_collection_metadata(collection_name, category)

                self._statistics['collections_created'] += 1
                creation_time = (time.time() - start_time) * 1000

                logger.info(f"Successfully created collection: {collection_name}")

                result = CollectionCreationResult(
                    success=True,
                    collection_name=collection_name,
                    category=category,
                    metadata=metadata,
                    collision_result=collision_result,
                    creation_time_ms=creation_time
                )

                # Add warnings if soft collisions were detected
                if collision_result.has_collision and collision_result.severity == CollisionSeverity.WARNING:
                    result.warnings.append(f"Warning: {collision_result.collision_reason}")

                return result

        except CollectionCollisionError:
            # Re-raise collision errors as-is
            raise

        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            creation_time = (time.time() - start_time) * 1000

            return CollectionCreationResult(
                success=False,
                collection_name=collection_name,
                category=category,
                creation_time_ms=creation_time,
                warnings=[f"Creation failed: {str(e)}"]
            )

    async def suggest_alternative_names(self, collection_name: str,
                                      category: CollectionCategory,
                                      count: int = 5) -> list[str]:
        """
        Generate alternative collection names for a given name.

        Args:
            collection_name: Original collection name
            category: Collection category
            count: Maximum number of suggestions

        Returns:
            List of suggested alternative names
        """
        if not self._initialized:
            await self.initialize()

        logger.debug(f"Generating alternative names for: {collection_name}")

        suggestions = await self.collision_detector.suggestion_engine.generate_suggestions(
            collection_name, category, count
        )

        self._statistics['suggestions_generated'] += len(suggestions)

        logger.debug(f"Generated {len(suggestions)} suggestions: {suggestions}")
        return suggestions

    async def validate_collection_name(self, collection_name: str,
                                     category: CollectionCategory | None = None) -> CollisionResult:
        """
        Validate a collection name without creating it.

        Args:
            collection_name: Name to validate
            category: Optional collection category

        Returns:
            CollisionResult with validation details
        """
        if not self._initialized:
            await self.initialize()

        return await self.collision_detector.check_collection_collision(
            collection_name, category
        )

    async def list_collections_by_category(self, category: CollectionCategory) -> list[str]:
        """
        List all collections in a specific category.

        Args:
            category: Collection category to filter by

        Returns:
            List of collection names in the category
        """
        if not self._initialized:
            await self.initialize()

        category_collections = self.collision_detector.registry.get_by_category(category)
        return list(category_collections)

    async def get_collection_statistics(self) -> dict[str, Any]:
        """Get comprehensive collection management statistics."""
        if not self._initialized:
            await self.initialize()

        collision_stats = await self.collision_detector.get_collision_statistics()

        return {
            'manager_statistics': self._statistics.copy(),
            'collision_detection': collision_stats,
            'naming_configuration': {
                'memory_collection_name': self.naming_config.memory_collection_name,
                'valid_project_suffixes': list(self.naming_config.valid_project_suffixes) if self.naming_config.valid_project_suffixes else [],
                'strict_validation': self.naming_config.strict_validation,
                'generate_suggestions': self.naming_config.generate_suggestions
            }
        }

    async def remove_collection_safely(self, collection_name: str) -> bool:
        """
        Remove a collection safely with registry cleanup.

        Args:
            collection_name: Name of collection to remove

        Returns:
            True if successfully removed
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Removing collection safely: {collection_name}")

        try:
            # Remove from Qdrant
            self.qdrant_client.delete_collection(collection_name)

            # Remove from collision registry
            await self.collision_detector.remove_collection_registration(collection_name)

            logger.info(f"Successfully removed collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove collection {collection_name}: {e}")
            return False

    async def shutdown(self):
        """Shutdown the collection manager and clean up resources."""
        if self.collision_detector:
            await self.collision_detector.shutdown()

        self._initialized = False
        logger.info("Collection manager shut down")

    def _create_qdrant_collection(self, collection_name: str, category: CollectionCategory) -> CollectionInfo:
        """
        Create a Qdrant collection with appropriate configuration.

        This is a simplified implementation for demonstration.
        Real implementation would include vector dimensions, distance metrics, etc.
        """
        # Example simplified collection creation
        # In real implementation, this would use proper Qdrant collection configuration
        from qdrant_client.http.models import Distance, VectorParams

        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        # Return mock collection info
        return CollectionInfo(
            name=collection_name,
            status="green",
            vectors_count=0,
            segments_count=1
        )

    def _generate_collection_metadata(self, collection_name: str,
                                    category: CollectionCategory) -> MultiTenantMetadataSchema:
        """Generate metadata schema for the collection."""
        # This would integrate with the metadata schema from subtask 249.1
        if category == CollectionCategory.SYSTEM:
            return MultiTenantMetadataSchema.create_for_system(
                collection_name=collection_name,
                collection_type="system_collection"
            )
        elif category == CollectionCategory.LIBRARY:
            return MultiTenantMetadataSchema.create_for_library(
                collection_name=collection_name,
                collection_type="library_collection"
            )
        elif category == CollectionCategory.PROJECT:
            parts = collection_name.split('-')
            project_name = '-'.join(parts[:-1]) if len(parts) > 1 else collection_name
            collection_type = parts[-1] if len(parts) > 1 else "default"
            return MultiTenantMetadataSchema.create_for_project(
                project_name=project_name,
                collection_type=collection_type
            )
        elif category == CollectionCategory.GLOBAL:
            return MultiTenantMetadataSchema.create_for_global(
                collection_name=collection_name,
                collection_type="global"
            )
        else:
            raise ValueError(f"Unknown collection category: {category}")


# Convenience functions for integration

async def create_collection_with_collision_detection(qdrant_client: QdrantClient,
                                                   collection_name: str,
                                                   category: CollectionCategory,
                                                   naming_config: NamingConfiguration | None = None) -> CollectionCreationResult:
    """
    Convenience function to create a collection with collision detection.

    Args:
        qdrant_client: Qdrant client instance
        collection_name: Name of collection to create
        category: Collection category
        naming_config: Optional naming configuration

    Returns:
        CollectionCreationResult with outcome details
    """
    manager = CollisionAwareCollectionManager(qdrant_client, naming_config)
    try:
        await manager.initialize()
        return await manager.create_collection_safely(collection_name, category)
    finally:
        await manager.shutdown()


async def validate_collection_name_with_collision_detection(qdrant_client: QdrantClient,
                                                          collection_name: str,
                                                          category: CollectionCategory | None = None) -> CollisionResult:
    """
    Convenience function to validate a collection name.

    Args:
        qdrant_client: Qdrant client instance
        collection_name: Name to validate
        category: Optional collection category

    Returns:
        CollisionResult with validation details
    """
    manager = CollisionAwareCollectionManager(qdrant_client)
    try:
        await manager.initialize()
        return await manager.validate_collection_name(collection_name, category)
    finally:
        await manager.shutdown()


# Export public classes and functions
__all__ = [
    'CollisionAwareCollectionManager',
    'CollectionCollisionError',
    'CollectionCreationResult',
    'create_collection_with_collision_detection',
    'validate_collection_name_with_collision_detection'
]
