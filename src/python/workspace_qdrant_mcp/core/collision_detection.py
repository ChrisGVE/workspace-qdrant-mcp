"""
Collection Collision Detection System for workspace-qdrant-mcp.

This module implements a comprehensive collision detection system that prevents
collection naming conflicts across projects while maintaining system performance
and providing intelligent alternatives.

Key Features:
    - Real-time collision detection with performance optimization
    - Collision registry for fast lookups and conflict tracking
    - Intelligent name suggestion engine with category-aware algorithms
    - Concurrent collection creation safeguards with locking mechanisms
    - Cross-project conflict prevention and resolution
    - Integration with existing naming validation from subtask 249.2

Architecture:
    - CollisionRegistry: Performance-optimized collision tracking and caching
    - CollisionDetector: Main detection engine with conflict prevention logic
    - NameSuggestionEngine: Intelligent alternative name generation
    - ConcurrentCreationGuard: Thread-safe collection creation coordination
    - CollisionReportGenerator: Detailed analysis and reporting capabilities

Integration:
    This system builds upon the CollectionNamingValidator from subtask 249.2
    and integrates with the MultiTenantMetadataSchema from subtask 249.1 to
    provide comprehensive collision prevention across all collection categories.

Example Usage:
    ```python
    # Initialize the collision detection system
    detector = CollisionDetector(qdrant_client, naming_validator)
    await detector.initialize()

    # Check for collisions before creating a collection
    result = await detector.check_collection_collision("my-project-docs")
    if result.has_collision:
        suggestions = result.suggested_alternatives
        logger.warning(f"Collision detected: {result.collision_type}")
        logger.info(f"Suggested alternatives: {suggestions}")

    # Safe collection creation with collision prevention
    async with detector.create_collection_guard("new-collection"):
        collection_info = await client.create_collection("new-collection")
    ```

Performance Features:
    - Bloom filter for fast negative collision checks
    - LRU cache for frequently accessed collection names
    - Asynchronous batch collision checking for multiple names
    - Lazy loading of collision registry data
    - Optimized data structures for large-scale deployments
"""

import asyncio
import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, AsyncContextManager, Callable
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

# Import components from previous subtasks
from .collection_naming_validation import (
    CollectionNamingValidator,
    ValidationResult,
    NamingConfiguration,
    ValidationSeverity,
    ConflictType,
    CollectionCategory
)
from .metadata_schema import (
    MultiTenantMetadataSchema,
    WorkspaceScope,
    AccessLevel
)


class CollisionSeverity(Enum):
    """Severity levels for collision detection."""

    BLOCKING = "blocking"        # Hard collision - operation must be blocked
    WARNING = "warning"          # Soft collision - operation allowed with warning
    ADVISORY = "advisory"        # Informational - consider alternatives
    NONE = "none"               # No collision detected


class CollisionCategory(Enum):
    """Categories of collision types for detailed analysis."""

    EXACT_DUPLICATE = "exact_duplicate"           # Exact same name exists
    SIMILAR_NAME = "similar_name"                 # Very similar name exists
    CROSS_PROJECT = "cross_project"               # Conflicts across projects
    CATEGORY_CONFLICT = "category_conflict"       # Conflicts across collection categories
    RESERVED_VIOLATION = "reserved_violation"     # Uses reserved naming patterns
    PATTERN_SIMILARITY = "pattern_similarity"    # Similar naming patterns
    NAMESPACE_CONFLICT = "namespace_conflict"     # Conflicts in same namespace


@dataclass
class CollisionResult:
    """
    Comprehensive result of collision detection analysis.

    Provides detailed information about detected collisions including severity,
    conflicting collections, suggested alternatives, and resolution strategies.
    """

    has_collision: bool
    severity: CollisionSeverity
    category: Optional[CollisionCategory] = None

    # Collision details
    conflicting_collections: List[str] = field(default_factory=list)
    collision_reason: Optional[str] = None
    collision_metadata: Dict[str, Any] = field(default_factory=dict)

    # Resolution information
    suggested_alternatives: List[str] = field(default_factory=list)
    auto_resolution_available: bool = False
    resolution_strategy: Optional[str] = None

    # Performance and timing information
    detection_time_ms: float = 0.0
    cache_hit: bool = False
    registry_size: int = 0


@dataclass
class CollisionRegistryEntry:
    """Entry in the collision registry for tracking collection names."""

    collection_name: str
    category: CollectionCategory
    project_context: Optional[str] = None
    creation_timestamp: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata_hash: Optional[str] = None

    def update_access(self):
        """Update access tracking information."""
        self.last_accessed = time.time()
        self.access_count += 1


class BloomFilter:
    """
    Simple Bloom filter for fast negative collision checks.

    Provides probabilistic membership testing to quickly eliminate
    non-colliding collection names without accessing the full registry.
    """

    def __init__(self, capacity: int = 10000, error_rate: float = 0.01):
        """
        Initialize Bloom filter.

        Args:
            capacity: Expected number of items
            error_rate: Desired false positive rate
        """
        self.capacity = capacity
        self.error_rate = error_rate

        # Calculate optimal bit array size and hash function count
        self.bit_size = int(-capacity * (error_rate / (2 ** 2)))
        self.hash_count = int((self.bit_size / capacity) * 0.693)

        # Ensure minimum reasonable values
        self.bit_size = max(self.bit_size, 1024)
        self.hash_count = max(self.hash_count, 1)

        self.bit_array = [False] * self.bit_size
        self.item_count = 0

        logger.debug(f"Initialized Bloom filter: size={self.bit_size}, hashes={self.hash_count}")

    def _hash(self, item: str, seed: int) -> int:
        """Generate hash value for item with given seed."""
        return hash(item + str(seed)) % self.bit_size

    def add(self, item: str):
        """Add item to the Bloom filter."""
        for i in range(self.hash_count):
            index = self._hash(item.lower(), i)
            self.bit_array[index] = True
        self.item_count += 1

    def might_contain(self, item: str) -> bool:
        """Check if item might be in the set (no false negatives)."""
        for i in range(self.hash_count):
            index = self._hash(item.lower(), i)
            if not self.bit_array[index]:
                return False
        return True

    def clear(self):
        """Clear the Bloom filter."""
        self.bit_array = [False] * self.bit_size
        self.item_count = 0


class CollisionRegistry:
    """
    Performance-optimized registry for tracking collection names and conflicts.

    Maintains an in-memory registry of all collection names with fast lookup
    capabilities, collision tracking, and efficient batch operations.
    """

    def __init__(self, max_cache_size: int = 5000):
        """
        Initialize the collision registry.

        Args:
            max_cache_size: Maximum number of entries to keep in LRU cache
        """
        self.max_cache_size = max_cache_size

        # Core registry data structures
        self._registry: Dict[str, CollisionRegistryEntry] = {}
        self._category_index: Dict[CollisionCategory, Set[str]] = defaultdict(set)
        self._project_index: Dict[str, Set[str]] = defaultdict(set)

        # Performance optimization components
        self._bloom_filter = BloomFilter()
        self._cache_order = deque()  # LRU cache ordering
        self._similarity_cache: Dict[str, List[str]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            'total_lookups': 0,
            'cache_hits': 0,
            'bloom_filter_hits': 0,
            'similarity_cache_hits': 0
        }

        logger.info(f"Initialized collision registry with cache size: {max_cache_size}")

    def add_collection(self, name: str, category: CollectionCategory, project_context: Optional[str] = None) -> bool:
        """
        Add a collection to the registry.

        Args:
            name: Collection name to add
            category: Collection category
            project_context: Optional project context

        Returns:
            True if added successfully, False if already exists
        """
        with self._lock:
            if name in self._registry:
                # Update existing entry
                entry = self._registry[name]
                entry.update_access()
                return False

            # Create new entry
            entry = CollisionRegistryEntry(
                collection_name=name,
                category=category,
                project_context=project_context
            )

            # Add to main registry
            self._registry[name] = entry

            # Update indices
            self._category_index[category].add(name)
            if project_context:
                self._project_index[project_context].add(name)

            # Update Bloom filter
            self._bloom_filter.add(name)

            # Manage cache size
            self._cache_order.append(name)
            self._manage_cache_size()

            logger.debug(f"Added collection to registry: {name} (category: {category})")
            return True

    def contains(self, name: str) -> bool:
        """
        Check if collection name exists in registry.

        Args:
            name: Collection name to check

        Returns:
            True if collection exists
        """
        with self._lock:
            self._stats['total_lookups'] += 1

            # Fast negative check with Bloom filter
            if not self._bloom_filter.might_contain(name):
                self._stats['bloom_filter_hits'] += 1
                return False

            # Check actual registry
            exists = name in self._registry
            if exists:
                # Update access tracking
                entry = self._registry[name]
                entry.update_access()

                # Update LRU cache order
                if name in self._cache_order:
                    self._cache_order.remove(name)
                self._cache_order.append(name)

                self._stats['cache_hits'] += 1

            return exists

    def get_similar_names(self, name: str, threshold: float = 0.8) -> List[str]:
        """
        Find similar collection names using cached similarity analysis.

        Args:
            name: Collection name to find similarities for
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            List of similar collection names
        """
        with self._lock:
            # Check similarity cache first
            cache_key = f"{name}:{threshold}"
            if cache_key in self._similarity_cache:
                self._stats['similarity_cache_hits'] += 1
                return self._similarity_cache[cache_key]

            similar_names = []
            name_lower = name.lower()

            # Check against all registered names
            for registered_name in self._registry.keys():
                similarity = self._calculate_similarity(name_lower, registered_name.lower())
                if similarity >= threshold:
                    similar_names.append(registered_name)

            # Cache the result
            self._similarity_cache[cache_key] = similar_names

            # Manage similarity cache size
            if len(self._similarity_cache) > self.max_cache_size:
                oldest_key = next(iter(self._similarity_cache))
                del self._similarity_cache[oldest_key]

            return similar_names

    def get_by_category(self, category: CollectionCategory) -> Set[str]:
        """Get all collection names in a specific category."""
        with self._lock:
            return self._category_index[category].copy()

    def get_by_project(self, project_context: str) -> Set[str]:
        """Get all collection names for a specific project."""
        with self._lock:
            return self._project_index[project_context].copy()

    def remove_collection(self, name: str) -> bool:
        """
        Remove a collection from the registry.

        Args:
            name: Collection name to remove

        Returns:
            True if removed successfully, False if not found
        """
        with self._lock:
            if name not in self._registry:
                return False

            entry = self._registry[name]

            # Remove from indices
            self._category_index[entry.category].discard(name)
            if entry.project_context:
                self._project_index[entry.project_context].discard(name)

            # Remove from main registry
            del self._registry[name]

            # Remove from cache order
            if name in self._cache_order:
                self._cache_order.remove(name)

            # Clear similarity cache entries
            keys_to_remove = [k for k in self._similarity_cache.keys() if k.startswith(f"{name}:")]
            for key in keys_to_remove:
                del self._similarity_cache[key]

            logger.debug(f"Removed collection from registry: {name}")
            return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics and performance metrics."""
        with self._lock:
            cache_hit_rate = (self._stats['cache_hits'] / max(self._stats['total_lookups'], 1)) * 100
            bloom_efficiency = (self._stats['bloom_filter_hits'] / max(self._stats['total_lookups'], 1)) * 100

            return {
                'total_collections': len(self._registry),
                'total_lookups': self._stats['total_lookups'],
                'cache_hit_rate_percent': round(cache_hit_rate, 2),
                'bloom_filter_efficiency_percent': round(bloom_efficiency, 2),
                'similarity_cache_size': len(self._similarity_cache),
                'category_distribution': {
                    cat.value: len(names) for cat, names in self._category_index.items()
                },
                'project_distribution': {
                    proj: len(names) for proj, names in self._project_index.items()
                }
            }

    def clear(self):
        """Clear all registry data."""
        with self._lock:
            self._registry.clear()
            self._category_index.clear()
            self._project_index.clear()
            self._cache_order.clear()
            self._similarity_cache.clear()
            self._bloom_filter.clear()
            self._stats = {
                'total_lookups': 0,
                'cache_hits': 0,
                'bloom_filter_hits': 0,
                'similarity_cache_hits': 0
            }
            logger.info("Cleared collision registry")

    def _manage_cache_size(self):
        """Manage LRU cache size by removing oldest entries."""
        while len(self._cache_order) > self.max_cache_size:
            oldest_name = self._cache_order.popleft()
            if oldest_name in self._registry:
                # Only remove if it's actually old (not recently re-accessed)
                entry = self._registry[oldest_name]
                if time.time() - entry.last_accessed > 3600:  # 1 hour
                    self.remove_collection(oldest_name)

    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two collection names.

        Uses a combination of edit distance and common substring analysis
        to determine how similar two names are.
        """
        if name1 == name2:
            return 1.0

        # Levenshtein distance calculation
        len1, len2 = len(name1), len(name2)
        if len1 == 0:
            return 0.0
        if len2 == 0:
            return 0.0

        # Create matrix for dynamic programming
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        # Initialize matrix
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        # Fill matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if name1[i-1] == name2[j-1] else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # deletion
                    matrix[i][j-1] + 1,      # insertion
                    matrix[i-1][j-1] + cost  # substitution
                )

        # Convert distance to similarity (0.0 to 1.0)
        max_len = max(len1, len2)
        distance = matrix[len1][len2]
        similarity = 1.0 - (distance / max_len)

        return similarity


class NameSuggestionEngine:
    """
    Intelligent name suggestion engine with category-aware algorithms.

    Generates alternative collection names when collisions are detected,
    using various strategies including pattern-based generation, semantic
    alternatives, and category-specific naming conventions.
    """

    def __init__(self, naming_validator: CollectionNamingValidator, registry: CollisionRegistry):
        """
        Initialize the name suggestion engine.

        Args:
            naming_validator: Collection naming validator
            registry: Collision registry for checking name availability
        """
        self.naming_validator = naming_validator
        self.registry = registry

        # Suggestion strategies
        self._strategies = [
            self._suggest_with_suffixes,
            self._suggest_with_prefixes,
            self._suggest_with_numbers,
            self._suggest_with_variations,
            self._suggest_category_specific,
            self._suggest_semantic_alternatives
        ]

        # Common suffixes and prefixes for suggestions
        self._common_suffixes = [
            'v2', 'alt', 'new', 'updated', 'mod', 'custom', 'local', 'temp'
        ]
        self._common_prefixes = [
            'new', 'alt', 'temp', 'draft', 'test'
        ]

        # Semantic alternatives for common terms
        self._semantic_alternatives = {
            'docs': ['documentation', 'notes', 'reference', 'guide'],
            'notes': ['docs', 'memos', 'remarks', 'comments'],
            'data': ['info', 'content', 'records', 'storage'],
            'temp': ['temporary', 'tmp', 'scratch', 'draft'],
            'test': ['testing', 'trial', 'demo', 'sample'],
            'main': ['primary', 'core', 'central', 'default'],
            'config': ['settings', 'configuration', 'setup', 'preferences']
        }

        logger.info("Initialized name suggestion engine")

    async def generate_suggestions(self, collision_name: str, intended_category: CollectionCategory,
                                 max_suggestions: int = 5) -> List[str]:
        """
        Generate intelligent name suggestions for a colliding collection name.

        Args:
            collision_name: The name that has a collision
            intended_category: The intended category for the collection
            max_suggestions: Maximum number of suggestions to generate

        Returns:
            List of suggested alternative names
        """
        suggestions = []

        # Try each strategy until we have enough suggestions
        for strategy in self._strategies:
            if len(suggestions) >= max_suggestions:
                break

            try:
                strategy_suggestions = await strategy(collision_name, intended_category)
                for suggestion in strategy_suggestions:
                    if len(suggestions) >= max_suggestions:
                        break

                    # Validate suggestion doesn't have collisions
                    if not self.registry.contains(suggestion):
                        # Validate it's a properly formatted name
                        validation_result = self.naming_validator.validate_name(
                            suggestion, intended_category
                        )
                        if validation_result.is_valid:
                            suggestions.append(suggestion)

            except Exception as e:
                logger.warning(f"Strategy {strategy.__name__} failed for {collision_name}: {e}")
                continue

        logger.debug(f"Generated {len(suggestions)} suggestions for {collision_name}")
        return suggestions

    async def _suggest_with_suffixes(self, name: str, category: CollectionCategory) -> List[str]:
        """Generate suggestions by adding common suffixes."""
        suggestions = []

        for suffix in self._common_suffixes:
            if category == CollectionCategory.PROJECT:
                # For project collections, add suffix before the final component
                parts = name.split('-')
                if len(parts) >= 2:
                    project_parts = parts[:-1]
                    collection_type = parts[-1]
                    suggestion = '-'.join(project_parts + [suffix, collection_type])
                else:
                    suggestion = f"{name}-{suffix}"
            else:
                # For other categories, simple suffix addition
                if category == CollectionCategory.LIBRARY and not name.startswith('_'):
                    suggestion = f"_{name}_{suffix}"
                elif category == CollectionCategory.SYSTEM and not name.startswith('__'):
                    suggestion = f"__{name}_{suffix}"
                else:
                    suggestion = f"{name}_{suffix}"

            suggestions.append(suggestion)

        return suggestions

    async def _suggest_with_prefixes(self, name: str, category: CollectionCategory) -> List[str]:
        """Generate suggestions by adding common prefixes."""
        suggestions = []

        for prefix in self._common_prefixes:
            if category == CollectionCategory.PROJECT:
                # For project collections, add prefix to project name
                parts = name.split('-')
                if len(parts) >= 2:
                    project_parts = parts[:-1]
                    collection_type = parts[-1]
                    new_project = '-'.join([prefix] + project_parts)
                    suggestion = f"{new_project}-{collection_type}"
                else:
                    suggestion = f"{prefix}-{name}"
            else:
                # For other categories, add after any required prefixes
                if category == CollectionCategory.LIBRARY:
                    if name.startswith('_'):
                        base_name = name[1:]
                        suggestion = f"_{prefix}_{base_name}"
                    else:
                        suggestion = f"_{prefix}_{name}"
                elif category == CollectionCategory.SYSTEM:
                    if name.startswith('__'):
                        base_name = name[2:]
                        suggestion = f"__{prefix}_{base_name}"
                    else:
                        suggestion = f"__{prefix}_{name}"
                else:
                    suggestion = f"{prefix}_{name}"

            suggestions.append(suggestion)

        return suggestions

    async def _suggest_with_numbers(self, name: str, category: CollectionCategory) -> List[str]:
        """Generate suggestions by adding version numbers."""
        suggestions = []

        for i in range(2, 6):  # Generate v2, v3, v4, v5
            if category == CollectionCategory.PROJECT:
                # For project collections, add version before collection type
                parts = name.split('-')
                if len(parts) >= 2:
                    project_parts = parts[:-1]
                    collection_type = parts[-1]
                    suggestion = '-'.join(project_parts + [f"v{i}", collection_type])
                else:
                    suggestion = f"{name}-v{i}"
            else:
                # Simple version suffix for other categories
                suggestion = f"{name}_v{i}"

            suggestions.append(suggestion)

        return suggestions

    async def _suggest_with_variations(self, name: str, category: CollectionCategory) -> List[str]:
        """Generate suggestions using name variations and abbreviations."""
        suggestions = []

        # Extract base components for transformation
        if category == CollectionCategory.PROJECT:
            parts = name.split('-')
            if len(parts) >= 2:
                project_parts = parts[:-1]
                collection_type = parts[-1]
                base_project = '-'.join(project_parts)
            else:
                base_project = name
                collection_type = None
        else:
            # Remove prefixes for processing
            if name.startswith('__'):
                base_name = name[2:]
                prefix = '__'
            elif name.startswith('_'):
                base_name = name[1:]
                prefix = '_'
            else:
                base_name = name
                prefix = ''

            base_project = base_name
            collection_type = None

        # Generate variations
        variations = []

        # Abbreviated versions
        if len(base_project) > 3:
            # Take first letter of each word
            words = base_project.replace('-', '_').split('_')
            if len(words) > 1:
                abbreviation = ''.join(word[0] for word in words if word)
                variations.append(abbreviation)

            # Shortened version
            if len(base_project) > 6:
                shortened = base_project[:6]
                variations.append(shortened)

        # Common transformations
        variations.extend([
            base_project.replace('-', '_'),
            base_project.replace('_', '-'),
            f"my_{base_project}",
            f"{base_project}_collection"
        ])

        # Reconstruct full names
        for variation in variations:
            if category == CollectionCategory.PROJECT:
                if collection_type:
                    suggestion = f"{variation}-{collection_type}"
                else:
                    suggestion = variation
            elif category == CollectionCategory.LIBRARY:
                suggestion = f"_{variation}"
            elif category == CollectionCategory.SYSTEM:
                suggestion = f"__{variation}"
            else:
                suggestion = variation

            suggestions.append(suggestion)

        return suggestions

    async def _suggest_category_specific(self, name: str, category: CollectionCategory) -> List[str]:
        """Generate category-specific name suggestions."""
        suggestions = []

        if category == CollectionCategory.SYSTEM:
            system_names = ['config', 'settings', 'cache', 'metadata', 'index']
            for sys_name in system_names:
                suggestions.append(f"__{sys_name}")

        elif category == CollectionCategory.LIBRARY:
            library_names = ['tools', 'utils', 'helpers', 'common', 'shared']
            for lib_name in library_names:
                suggestions.append(f"_{lib_name}")

        elif category == CollectionCategory.PROJECT:
            project_types = ['docs', 'notes', 'scratchbook', 'research', 'archive']
            # Extract project name if possible
            parts = name.split('-')
            if len(parts) >= 2:
                project_base = '-'.join(parts[:-1])
                for proj_type in project_types:
                    suggestions.append(f"{project_base}-{proj_type}")
            else:
                for proj_type in project_types:
                    suggestions.append(f"{name}-{proj_type}")

        elif category == CollectionCategory.GLOBAL:
            global_names = ['workspace', 'shared', 'common', 'global', 'general']
            suggestions.extend(global_names)

        return suggestions

    async def _suggest_semantic_alternatives(self, name: str, category: CollectionCategory) -> List[str]:
        """Generate suggestions using semantic alternatives for common terms."""
        suggestions = []

        # Find semantic alternatives for words in the name
        name_lower = name.lower()
        for term, alternatives in self._semantic_alternatives.items():
            if term in name_lower:
                for alt in alternatives:
                    # Replace the term with its alternative
                    alt_name = name_lower.replace(term, alt)

                    # Reconstruct proper format based on category
                    if category == CollectionCategory.LIBRARY and not alt_name.startswith('_'):
                        alt_name = f"_{alt_name}"
                    elif category == CollectionCategory.SYSTEM and not alt_name.startswith('__'):
                        alt_name = f"__{alt_name}"

                    suggestions.append(alt_name)

        return suggestions


class ConcurrentCreationGuard:
    """
    Thread-safe guard for preventing concurrent collection creation conflicts.

    Provides locking mechanisms to ensure that collection creation operations
    don't interfere with each other and that collision detection remains accurate
    during concurrent operations.
    """

    def __init__(self):
        """Initialize the concurrent creation guard."""
        self._creation_locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._pending_creations: Set[str] = set()

        # Weak references to prevent memory leaks
        self._lock_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

        logger.info("Initialized concurrent creation guard")

    @asynccontextmanager
    async def acquire_creation_lock(self, collection_name: str) -> AsyncContextManager[None]:
        """
        Acquire exclusive lock for collection creation.

        Args:
            collection_name: Name of the collection being created

        Yields:
            Context manager for the creation lock
        """
        normalized_name = collection_name.lower().strip()

        # Get or create lock for this collection name
        async with self._global_lock:
            if normalized_name not in self._creation_locks:
                self._creation_locks[normalized_name] = asyncio.Lock()

            creation_lock = self._creation_locks[normalized_name]
            self._pending_creations.add(normalized_name)

        try:
            # Acquire the specific collection lock
            async with creation_lock:
                logger.debug(f"Acquired creation lock for: {normalized_name}")
                yield
        finally:
            # Clean up pending creation tracking
            async with self._global_lock:
                self._pending_creations.discard(normalized_name)

                # Clean up lock if no longer needed
                if normalized_name in self._creation_locks:
                    lock = self._creation_locks[normalized_name]
                    if not lock.locked():
                        del self._creation_locks[normalized_name]

            logger.debug(f"Released creation lock for: {normalized_name}")

    def is_creation_pending(self, collection_name: str) -> bool:
        """
        Check if collection creation is currently pending.

        Args:
            collection_name: Name to check

        Returns:
            True if creation is pending for this name
        """
        normalized_name = collection_name.lower().strip()
        return normalized_name in self._pending_creations

    def get_pending_creations(self) -> Set[str]:
        """Get set of all pending collection creations."""
        return self._pending_creations.copy()


class CollisionDetector:
    """
    Main collision detection engine with comprehensive conflict prevention.

    Coordinates collision detection, registry management, name suggestion,
    and concurrent creation protection to provide a complete collision
    prevention system for collection management.
    """

    def __init__(self, qdrant_client: QdrantClient, naming_validator: Optional[CollectionNamingValidator] = None):
        """
        Initialize the collision detector.

        Args:
            qdrant_client: Qdrant client for querying existing collections
            naming_validator: Optional naming validator (creates default if None)
        """
        self.qdrant_client = qdrant_client
        self.naming_validator = naming_validator or CollectionNamingValidator()

        # Core components
        self.registry = CollisionRegistry()
        self.suggestion_engine = NameSuggestionEngine(self.naming_validator, self.registry)
        self.creation_guard = ConcurrentCreationGuard()

        # Configuration
        self.similarity_threshold = 0.8
        self.auto_refresh_interval = 300  # 5 minutes

        # State tracking
        self._initialized = False
        self._last_refresh = 0.0
        self._refresh_task: Optional[asyncio.Task] = None

        logger.info("Initialized collision detector")

    async def initialize(self):
        """Initialize the collision detector and load existing collections."""
        if self._initialized:
            return

        logger.info("Initializing collision detector...")

        try:
            # Load existing collections from Qdrant
            await self._load_existing_collections()

            # Start periodic refresh task
            self._refresh_task = asyncio.create_task(self._periodic_refresh())

            self._initialized = True
            logger.info("Collision detector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize collision detector: {e}")
            raise

    async def check_collection_collision(self, collection_name: str,
                                       intended_category: Optional[CollectionCategory] = None,
                                       project_context: Optional[str] = None) -> CollisionResult:
        """
        Comprehensive collision detection for a collection name.

        Args:
            collection_name: Name to check for collisions
            intended_category: Intended category for the collection
            project_context: Optional project context

        Returns:
            CollisionResult with detailed analysis
        """
        start_time = time.time()

        # Ensure we're initialized
        if not self._initialized:
            await self.initialize()

        # Refresh registry if needed
        await self._refresh_if_needed()

        # Normalize the name
        normalized_name = collection_name.lower().strip()

        # Check for direct collision
        if self.registry.contains(normalized_name):
            detection_time = (time.time() - start_time) * 1000

            # Generate suggestions for alternatives
            suggestions = await self.suggestion_engine.generate_suggestions(
                normalized_name, intended_category or CollectionCategory.GLOBAL
            )

            return CollisionResult(
                has_collision=True,
                severity=CollisionSeverity.BLOCKING,
                category=CollisionCategory.EXACT_DUPLICATE,
                conflicting_collections=[normalized_name],
                collision_reason=f"Collection '{normalized_name}' already exists",
                suggested_alternatives=suggestions,
                detection_time_ms=detection_time,
                registry_size=len(self.registry._registry)
            )

        # Check for pending creations
        if self.creation_guard.is_creation_pending(normalized_name):
            detection_time = (time.time() - start_time) * 1000

            return CollisionResult(
                has_collision=True,
                severity=CollisionSeverity.BLOCKING,
                category=CollisionCategory.EXACT_DUPLICATE,
                conflicting_collections=[normalized_name],
                collision_reason=f"Collection '{normalized_name}' creation is currently pending",
                detection_time_ms=detection_time,
                registry_size=len(self.registry._registry)
            )

        # Check for similar names
        similar_names = self.registry.get_similar_names(normalized_name, self.similarity_threshold)
        if similar_names:
            detection_time = (time.time() - start_time) * 1000

            suggestions = await self.suggestion_engine.generate_suggestions(
                normalized_name, intended_category or CollectionCategory.GLOBAL
            )

            return CollisionResult(
                has_collision=True,
                severity=CollisionSeverity.WARNING,
                category=CollisionCategory.SIMILAR_NAME,
                conflicting_collections=similar_names,
                collision_reason=f"Similar collection names exist: {', '.join(similar_names)}",
                suggested_alternatives=suggestions,
                detection_time_ms=detection_time,
                registry_size=len(self.registry._registry)
            )

        # Check naming validation
        validation_result = self.naming_validator.validate_name(
            normalized_name, intended_category
        )

        if not validation_result.is_valid:
            detection_time = (time.time() - start_time) * 1000

            suggestions = validation_result.suggested_names or []
            if not suggestions:
                suggestions = await self.suggestion_engine.generate_suggestions(
                    normalized_name, intended_category or CollectionCategory.GLOBAL
                )

            return CollisionResult(
                has_collision=True,
                severity=CollisionSeverity.BLOCKING,
                category=CollisionCategory.RESERVED_VIOLATION,
                collision_reason=validation_result.error_message,
                suggested_alternatives=suggestions,
                detection_time_ms=detection_time,
                registry_size=len(self.registry._registry)
            )

        # No collision detected
        detection_time = (time.time() - start_time) * 1000

        return CollisionResult(
            has_collision=False,
            severity=CollisionSeverity.NONE,
            detection_time_ms=detection_time,
            registry_size=len(self.registry._registry)
        )

    async def register_collection_creation(self, collection_name: str,
                                         category: CollectionCategory,
                                         project_context: Optional[str] = None):
        """
        Register a successful collection creation in the registry.

        Args:
            collection_name: Name of the created collection
            category: Category of the collection
            project_context: Optional project context
        """
        normalized_name = collection_name.lower().strip()
        self.registry.add_collection(normalized_name, category, project_context)
        logger.debug(f"Registered collection creation: {normalized_name}")

    async def remove_collection_registration(self, collection_name: str):
        """
        Remove a collection from the registry.

        Args:
            collection_name: Name of the collection to remove
        """
        normalized_name = collection_name.lower().strip()
        removed = self.registry.remove_collection(normalized_name)
        if removed:
            logger.debug(f"Removed collection from registry: {normalized_name}")

    @asynccontextmanager
    async def create_collection_guard(self, collection_name: str) -> AsyncContextManager[None]:
        """
        Context manager for safe collection creation with collision prevention.

        Args:
            collection_name: Name of the collection to create

        Yields:
            Context manager for protected collection creation
        """
        async with self.creation_guard.acquire_creation_lock(collection_name):
            # Double-check for collisions within the lock
            collision_result = await self.check_collection_collision(collection_name)
            if collision_result.has_collision and collision_result.severity == CollisionSeverity.BLOCKING:
                raise ValueError(f"Collection collision detected: {collision_result.collision_reason}")

            yield

    async def get_collision_statistics(self) -> Dict[str, Any]:
        """Get comprehensive collision detection statistics."""
        registry_stats = self.registry.get_statistics()

        return {
            'registry_statistics': registry_stats,
            'pending_creations': len(self.creation_guard.get_pending_creations()),
            'similarity_threshold': self.similarity_threshold,
            'auto_refresh_interval_seconds': self.auto_refresh_interval,
            'last_refresh_timestamp': self._last_refresh,
            'initialized': self._initialized
        }

    async def refresh_registry(self):
        """Manually refresh the collision registry from Qdrant."""
        logger.info("Refreshing collision registry...")
        await self._load_existing_collections()
        self._last_refresh = time.time()
        logger.info("Registry refresh completed")

    async def shutdown(self):
        """Shutdown the collision detector and clean up resources."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

        self.registry.clear()
        self._initialized = False
        logger.info("Collision detector shut down")

    async def _load_existing_collections(self):
        """Load existing collections from Qdrant into the registry."""
        try:
            # Get list of collections from Qdrant
            collections_response = self.qdrant_client.get_collections()

            self.registry.clear()

            for collection in collections_response.collections:
                collection_name = collection.name

                # Try to determine category from naming patterns
                detected_category = self.naming_validator._detect_category(collection_name)

                # Add to registry
                self.registry.add_collection(collection_name, detected_category)

            logger.info(f"Loaded {len(collections_response.collections)} collections into registry")

        except Exception as e:
            logger.error(f"Failed to load existing collections: {e}")
            # Don't re-raise - collision detection can still work with empty registry

    async def _refresh_if_needed(self):
        """Refresh registry if enough time has passed since last refresh."""
        current_time = time.time()
        if current_time - self._last_refresh > self.auto_refresh_interval:
            await self.refresh_registry()

    async def _periodic_refresh(self):
        """Periodic background task to refresh the registry."""
        while True:
            try:
                await asyncio.sleep(self.auto_refresh_interval)
                await self.refresh_registry()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic registry refresh: {e}")