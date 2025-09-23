"""
Project Collection Naming Validator for Multi-Tenant Support

This module provides validation utilities for ensuring proper collection naming
conventions in multi-tenant environments. It validates that collection names
follow the expected patterns for project isolation and prevents naming conflicts.

Key Features:
    - Validates project-specific collection naming patterns
    - Ensures proper tenant isolation through naming conventions
    - Detects potential collection name conflicts
    - Supports both project-scoped and global collection patterns

Collection Naming Conventions:
    - Project collections: {project_name}-{collection_type}
    - Global collections: {collection_type} (no project prefix)
    - Scratchbook: {project_name}-scratchbook
    - Special collections: docs, reference, standards (global)

Example:
    ```python
    from project_collection_validator import ProjectCollectionValidator

    validator = ProjectCollectionValidator()

    # Validate project collection
    is_valid = validator.validate_collection_name("myproject-notes", "myproject")

    # Check naming conflicts
    conflicts = validator.check_naming_conflicts(["proj-docs", "proj-notes"])
    ```
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from loguru import logger


@dataclass
class CollectionNamingRule:
    """Defines a collection naming rule for validation."""

    pattern: str              # Regex pattern for the collection name
    description: str          # Human-readable description
    scope: str               # 'project' or 'global'
    required_suffix: bool    # Whether a project suffix is required
    allowed_types: Set[str]  # Allowed collection types for this pattern


class ProjectCollectionValidator:
    """
    Validator for project collection naming conventions in multi-tenant environments.

    This validator ensures that collection names follow consistent patterns that
    support proper tenant isolation and avoid naming conflicts between projects.

    Naming Rules:
        1. Project collections must include project prefix: {project}-{type}
        2. Global collections have no project prefix: {type}
        3. Special global collections: docs, reference, standards
        4. Scratchbook follows pattern: {project}-scratchbook
        5. No conflicting patterns between project and global collections
    """

    # Standard collection types
    PROJECT_COLLECTION_TYPES = {
        'notes', 'docs', 'code', 'research', 'scratchbook',
        'knowledge', 'context', 'memory', 'assets', 'logs'
    }

    GLOBAL_COLLECTION_TYPES = {
        'docs', 'reference', 'standards', 'shared', 'templates'
    }

    # Reserved collection names that cannot be used for projects
    RESERVED_NAMES = {
        'system', 'admin', 'config', 'metadata', 'internal',
        'qdrant', 'vector', 'index', 'default', 'temp'
    }

    def __init__(self):
        """Initialize the collection naming validator."""
        self.naming_rules = self._initialize_naming_rules()

    def _initialize_naming_rules(self) -> List[CollectionNamingRule]:
        """Initialize the collection naming rules."""
        return [
            CollectionNamingRule(
                pattern=r'^[a-z0-9][a-z0-9\-\_]*[a-z0-9]$',
                description="Basic naming pattern: lowercase, alphanumeric, hyphens, underscores",
                scope='both',
                required_suffix=False,
                allowed_types=set()
            ),
            CollectionNamingRule(
                pattern=r'^([a-z0-9][a-z0-9\-\_]*)-([a-z]+)$',
                description="Project collection pattern: project-type",
                scope='project',
                required_suffix=True,
                allowed_types=self.PROJECT_COLLECTION_TYPES
            ),
            CollectionNamingRule(
                pattern=r'^(docs|reference|standards|shared|templates)$',
                description="Global collection pattern: reserved global types",
                scope='global',
                required_suffix=False,
                allowed_types=self.GLOBAL_COLLECTION_TYPES
            )
        ]

    def validate_collection_name(
        self,
        collection_name: str,
        project_name: Optional[str] = None,
        collection_type: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Validate a collection name against naming conventions.

        Args:
            collection_name: The collection name to validate
            project_name: Expected project name (for project collections)
            collection_type: Expected collection type

        Returns:
            Dict with validation results: {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'detected_pattern': str,
                'suggestions': List[str]
            }
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'detected_pattern': None,
            'suggestions': []
        }

        # Basic validation
        if not collection_name:
            result['valid'] = False
            result['errors'].append("Collection name cannot be empty")
            return result

        if collection_name.lower() != collection_name:
            result['warnings'].append("Collection name should be lowercase")

        # Check for reserved names
        if collection_name.lower() in self.RESERVED_NAMES:
            result['valid'] = False
            result['errors'].append(f"'{collection_name}' is a reserved collection name")

        # Check length constraints
        if len(collection_name) < 2:
            result['valid'] = False
            result['errors'].append("Collection name must be at least 2 characters")

        if len(collection_name) > 64:
            result['valid'] = False
            result['errors'].append("Collection name must be 64 characters or less")

        # Pattern matching
        detected_pattern = self._detect_naming_pattern(collection_name)
        result['detected_pattern'] = detected_pattern

        if detected_pattern == 'project':
            self._validate_project_collection(collection_name, project_name, collection_type, result)
        elif detected_pattern == 'global':
            self._validate_global_collection(collection_name, collection_type, result)
        else:
            result['valid'] = False
            result['errors'].append("Collection name does not match any valid naming pattern")
            result['suggestions'].append("Use format: 'project-type' for project collections or 'type' for global collections")

        return result

    def _detect_naming_pattern(self, collection_name: str) -> Optional[str]:
        """Detect which naming pattern a collection name follows."""
        # Check for project pattern (contains hyphen)
        if '-' in collection_name and not collection_name.startswith('-') and not collection_name.endswith('-'):
            parts = collection_name.split('-')
            if len(parts) >= 2:
                return 'project'

        # Check for global pattern (single word, no hyphens for project separation)
        if collection_name in self.GLOBAL_COLLECTION_TYPES:
            return 'global'

        # Check if it's a simple name that could be global
        if re.match(r'^[a-z0-9]+$', collection_name):
            return 'global'

        return None

    def _validate_project_collection(
        self,
        collection_name: str,
        project_name: Optional[str],
        collection_type: Optional[str],
        result: Dict
    ):
        """Validate a project-scoped collection name."""
        parts = collection_name.split('-')
        detected_project = parts[0]
        detected_type = '-'.join(parts[1:])  # Support multi-part types

        # Validate project name match
        if project_name and detected_project != project_name:
            result['valid'] = False
            result['errors'].append(
                f"Collection project '{detected_project}' does not match expected project '{project_name}'"
            )

        # Validate collection type
        if collection_type and detected_type != collection_type:
            result['warnings'].append(
                f"Collection type '{detected_type}' does not match expected type '{collection_type}'"
            )

        # Check if type is allowed for project collections
        if detected_type in self.GLOBAL_COLLECTION_TYPES and detected_type not in {'docs', 'notes'}:
            result['warnings'].append(
                f"Collection type '{detected_type}' is typically used for global collections"
            )

        logger.debug(f"Validated project collection: {collection_name} -> project={detected_project}, type={detected_type}")

    def _validate_global_collection(
        self,
        collection_name: str,
        collection_type: Optional[str],
        result: Dict
    ):
        """Validate a global collection name."""
        # Check if it's a known global type
        if collection_name not in self.GLOBAL_COLLECTION_TYPES:
            if collection_type and collection_type in self.GLOBAL_COLLECTION_TYPES:
                result['suggestions'].append(f"Consider using standard global name: '{collection_type}'")
            else:
                result['warnings'].append(f"'{collection_name}' is not a standard global collection type")

        logger.debug(f"Validated global collection: {collection_name}")

    def check_naming_conflicts(self, collection_names: List[str]) -> List[Dict[str, any]]:
        """
        Check for naming conflicts between collections.

        Args:
            collection_names: List of collection names to check

        Returns:
            List of conflict descriptions
        """
        conflicts = []
        seen_patterns = {}

        for name in collection_names:
            pattern = self._detect_naming_pattern(name)
            if pattern == 'project':
                parts = name.split('-')
                project = parts[0]
                type_part = '-'.join(parts[1:])

                key = f"{project}:{type_part}"
                if key in seen_patterns:
                    conflicts.append({
                        'type': 'duplicate_project_collection',
                        'collections': [seen_patterns[key], name],
                        'description': f"Duplicate project collection type '{type_part}' for project '{project}'"
                    })
                else:
                    seen_patterns[key] = name

            elif pattern == 'global':
                key = f"global:{name}"
                if key in seen_patterns:
                    conflicts.append({
                        'type': 'duplicate_global_collection',
                        'collections': [seen_patterns[key], name],
                        'description': f"Duplicate global collection '{name}'"
                    })
                else:
                    seen_patterns[key] = name

        return conflicts

    def suggest_collection_name(
        self,
        project_name: str,
        collection_type: str,
        scope: str = 'project'
    ) -> str:
        """
        Suggest a proper collection name following conventions.

        Args:
            project_name: Name of the project
            collection_type: Type of collection (notes, docs, etc.)
            scope: 'project' or 'global'

        Returns:
            Suggested collection name
        """
        # Normalize inputs
        project_name = project_name.lower().replace('_', '-')
        collection_type = collection_type.lower().replace('_', '-')

        if scope == 'global' or collection_type in self.GLOBAL_COLLECTION_TYPES:
            return collection_type
        else:
            return f"{project_name}-{collection_type}"

    def get_project_collections_pattern(self, project_name: str) -> str:
        """Get a regex pattern for matching all collections for a project."""
        escaped_project = re.escape(project_name)
        return f"^{escaped_project}-[a-z0-9][a-z0-9\\-]*$"

    def extract_project_from_collection(self, collection_name: str) -> Optional[str]:
        """Extract the project name from a collection name if it follows project pattern."""
        if self._detect_naming_pattern(collection_name) == 'project':
            return collection_name.split('-')[0]
        return None

    def extract_type_from_collection(self, collection_name: str) -> Optional[str]:
        """Extract the collection type from a collection name."""
        pattern = self._detect_naming_pattern(collection_name)

        if pattern == 'project':
            parts = collection_name.split('-')
            return '-'.join(parts[1:])
        elif pattern == 'global':
            return collection_name

        return None