"""
Comprehensive documentation for the multi-tenant metadata schema.

This module provides complete field specifications, examples, constraints,
and usage patterns for the multi-tenant metadata schema. It serves as the
authoritative reference for understanding and implementing metadata-based
project isolation in workspace-qdrant-mcp.

Key Features:
    - Complete field specifications with types, constraints, and examples
    - Usage patterns for different collection types and scenarios
    - Migration guidance for existing collections
    - Best practices and performance considerations
    - Integration examples with existing collection management

Documentation Categories:
    - **Field Reference**: Complete field specifications
    - **Usage Patterns**: Common scenarios and examples
    - **Migration Guide**: Backward compatibility and migration
    - **Performance Guide**: Optimization and indexing recommendations
    - **Integration Guide**: Working with existing systems

Example:
    ```python
    from schema_documentation import SchemaDocumentation, get_field_specification

    # Get documentation for a specific field
    field_doc = get_field_specification("project_id")
    print(f"Description: {field_doc.description}")
    print(f"Type: {field_doc.type_info}")
    print(f"Constraints: {field_doc.constraints}")

    # Get usage examples
    docs = SchemaDocumentation()
    examples = docs.get_usage_examples("project_collection")
    ```
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum

try:
    from .metadata_schema import (
        MultiTenantMetadataSchema,
        CollectionCategory,
        WorkspaceScope,
        AccessLevel,
        METADATA_SCHEMA_VERSION
    )
except ImportError:
    # Fallback for development
    pass


class FieldType(Enum):
    """Field type classifications for documentation."""

    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ENUM = "enum"
    LIST = "list"
    OPTIONAL_STRING = "optional_string"
    TIMESTAMP = "timestamp"


@dataclass
class FieldSpecification:
    """Complete specification for a metadata field."""

    name: str
    description: str
    type_info: str
    required: bool
    indexed: bool
    constraints: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    default_value: Optional[Any] = None
    validation_rules: List[str] = field(default_factory=list)
    related_fields: List[str] = field(default_factory=list)
    migration_notes: Optional[str] = None
    performance_notes: Optional[str] = None


@dataclass
class UsagePattern:
    """Documentation for common usage patterns."""

    name: str
    description: str
    scenario: str
    example_code: str
    metadata_example: Dict[str, Any]
    notes: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)


class SchemaDocumentation:
    """
    Comprehensive documentation system for the multi-tenant metadata schema.

    This class provides complete reference documentation, usage examples,
    migration guidance, and best practices for working with the metadata schema.
    """

    def __init__(self):
        """Initialize schema documentation with complete field specifications."""
        self._field_specifications = self._build_field_specifications()
        self._usage_patterns = self._build_usage_patterns()
        self._migration_scenarios = self._build_migration_scenarios()

    def _build_field_specifications(self) -> Dict[str, FieldSpecification]:
        """Build complete field specifications for all metadata fields."""
        specs = {}

        # === Core Tenant Isolation Fields ===
        specs['project_id'] = FieldSpecification(
            name='project_id',
            description='Stable 12-character identifier for efficient project-based filtering',
            type_info='string (exactly 12 hexadecimal characters)',
            required=True,
            indexed=True,
            constraints=[
                'Must be exactly 12 characters long',
                'Must contain only hexadecimal characters (a-f, 0-9)',
                'Must be deterministically generated from project_name',
                'Must be stable across application restarts'
            ],
            examples=[
                'a1b2c3d4e5f6',
                'workspace12ab',
                'myproject123'
            ],
            validation_rules=[
                'Matches regex: ^[a-f0-9]{12}$',
                'Generated via SHA256(project_name)[:12]'
            ],
            related_fields=['project_name', 'tenant_namespace'],
            performance_notes='Primary filtering field - always indexed for optimal query performance'
        )

        specs['project_name'] = FieldSpecification(
            name='project_name',
            description='Human-readable project name for display and organization',
            type_info='string (max 128 characters)',
            required=True,
            indexed=True,
            constraints=[
                'Maximum length: 128 characters',
                'Allowed characters: letters, numbers, underscore, hyphen',
                'Automatically normalized to lowercase with underscores',
                'Must be unique within workspace context'
            ],
            examples=[
                'workspace_qdrant_mcp',
                'my_project',
                'enterprise_system'
            ],
            validation_rules=[
                'Matches regex: ^[a-zA-Z0-9_-]+$',
                'Length <= 128 characters',
                'Normalized to lowercase'
            ],
            related_fields=['project_id', 'tenant_namespace'],
            migration_notes='For migrated collections, extracted from collection name pattern'
        )

        specs['tenant_namespace'] = FieldSpecification(
            name='tenant_namespace',
            description='Hierarchical namespace for precise tenant isolation',
            type_info='string (format: project_name.collection_type)',
            required=True,
            indexed=True,
            constraints=[
                'Format: {project_name}.{collection_type}',
                'Maximum length: 192 characters',
                'Must be consistent with project_name and collection_type',
                'Used for hierarchical filtering'
            ],
            examples=[
                'workspace_qdrant_mcp.docs',
                'my_project.notes',
                'system.memory_collection'
            ],
            validation_rules=[
                'Matches regex: ^[a-zA-Z0-9_-]+\\.[a-zA-Z0-9_]+$',
                'Must equal project_name.collection_type'
            ],
            related_fields=['project_name', 'collection_type'],
            performance_notes='Enables efficient hierarchical filtering and tenant isolation'
        )

        # === Collection Classification Fields ===
        specs['collection_type'] = FieldSpecification(
            name='collection_type',
            description='Workspace collection type for functional classification',
            type_info='string (predefined workspace types)',
            required=True,
            indexed=True,
            constraints=[
                'Maximum length: 64 characters',
                'Allowed characters: letters, numbers, underscore',
                'Should use standard workspace types when possible',
                'Determines collection behavior and search scope'
            ],
            examples=[
                'docs',
                'notes',
                'scratchbook',
                'memory_collection',
                'code_collection'
            ],
            validation_rules=[
                'Matches regex: ^[a-zA-Z0-9_]+$',
                'Length <= 64 characters',
                'Preferably from standard types list'
            ],
            related_fields=['collection_category', 'workspace_scope', 'tenant_namespace']
        )

        specs['collection_category'] = FieldSpecification(
            name='collection_category',
            description='System-level classification for access control and behavior',
            type_info='CollectionCategory enum (system, library, project, global)',
            required=True,
            indexed=True,
            constraints=[
                'Must be valid CollectionCategory enum value',
                'Determines access control rules',
                'Affects naming pattern validation',
                'Influences search behavior'
            ],
            examples=[
                'CollectionCategory.SYSTEM',
                'CollectionCategory.LIBRARY',
                'CollectionCategory.PROJECT',
                'CollectionCategory.GLOBAL'
            ],
            validation_rules=[
                'Must be CollectionCategory enum instance',
                'Must be consistent with naming patterns'
            ],
            related_fields=['workspace_scope', 'access_level', 'naming_pattern']
        )

        specs['workspace_scope'] = FieldSpecification(
            name='workspace_scope',
            description='Accessibility scope for search and collaboration',
            type_info='WorkspaceScope enum (project, shared, global, library)',
            required=True,
            indexed=True,
            constraints=[
                'Must be valid WorkspaceScope enum value',
                'Determines search inclusion rules',
                'Affects cross-project accessibility',
                'Should align with collection_category'
            ],
            examples=[
                'WorkspaceScope.PROJECT',
                'WorkspaceScope.SHARED',
                'WorkspaceScope.GLOBAL',
                'WorkspaceScope.LIBRARY'
            ],
            validation_rules=[
                'Must be WorkspaceScope enum instance',
                'Should be consistent with collection_category'
            ],
            related_fields=['collection_category', 'access_level']
        )

        # === Access Control Fields ===
        specs['access_level'] = FieldSpecification(
            name='access_level',
            description='Access control level for security and permissions',
            type_info='AccessLevel enum (public, private, shared, readonly)',
            required=True,
            indexed=True,
            constraints=[
                'Must be valid AccessLevel enum value',
                'Determines who can access the collection',
                'Affects search result inclusion',
                'Should align with collection_category'
            ],
            examples=[
                'AccessLevel.PRIVATE',
                'AccessLevel.SHARED',
                'AccessLevel.PUBLIC',
                'AccessLevel.READONLY'
            ],
            validation_rules=[
                'Must be AccessLevel enum instance'
            ],
            related_fields=['collection_category', 'workspace_scope', 'mcp_readonly']
        )

        specs['mcp_readonly'] = FieldSpecification(
            name='mcp_readonly',
            description='Flag indicating if collection is read-only via MCP server',
            type_info='boolean',
            required=True,
            indexed=True,
            constraints=[
                'Must be boolean value (true/false)',
                'True for library collections (required)',
                'Determines MCP write permissions',
                'Affects tool operation availability'
            ],
            examples=[
                'true (for library collections)',
                'false (for project collections)'
            ],
            validation_rules=[
                'Must be boolean type',
                'Must be true for library collections'
            ],
            related_fields=['collection_category', 'cli_writable', 'access_level']
        )

        specs['cli_writable'] = FieldSpecification(
            name='cli_writable',
            description='Flag indicating if collection is writable via CLI',
            type_info='boolean',
            required=True,
            indexed=True,
            constraints=[
                'Must be boolean value (true/false)',
                'Usually true except for special cases',
                'Independent of MCP write permissions',
                'Affects CLI tool availability'
            ],
            examples=[
                'true (standard)',
                'false (rare special cases)'
            ],
            validation_rules=[
                'Must be boolean type'
            ],
            related_fields=['mcp_readonly', 'collection_category']
        )

        specs['created_by'] = FieldSpecification(
            name='created_by',
            description='Origin or creator of the collection',
            type_info='string (max 64 characters)',
            required=True,
            indexed=True,
            constraints=[
                'Maximum length: 64 characters',
                'Should use standard values when possible',
                'Tracks collection origin for auditing',
                'Helps with migration tracking'
            ],
            examples=[
                'system',
                'user',
                'cli',
                'migration',
                'admin'
            ],
            validation_rules=[
                'Length <= 64 characters',
                'Preferably from standard values'
            ],
            related_fields=['migration_source'],
            migration_notes='Set to "migration" for collections migrated from existing systems'
        )

        # === Reserved Naming and Compatibility ===
        specs['naming_pattern'] = FieldSpecification(
            name='naming_pattern',
            description='Original naming convention used for the collection',
            type_info='string (predefined patterns)',
            required=True,
            indexed=True,
            constraints=[
                'Must be from predefined pattern list',
                'Indicates original naming convention',
                'Used for backward compatibility',
                'Should match collection_category'
            ],
            examples=[
                'metadata_based',
                'system_prefix',
                'library_prefix',
                'project_pattern',
                'global_collection'
            ],
            validation_rules=[
                'Must be from valid naming patterns list',
                'Should be consistent with collection_category'
            ],
            related_fields=['collection_category', 'is_reserved_name']
        )

        specs['is_reserved_name'] = FieldSpecification(
            name='is_reserved_name',
            description='Flag indicating if collection uses reserved naming (system/library)',
            type_info='boolean',
            required=True,
            indexed=True,
            constraints=[
                'Must be boolean value',
                'True for system and library collections',
                'False for project and global collections',
                'Used for naming validation'
            ],
            examples=[
                'true (for system/library)',
                'false (for project/global)'
            ],
            validation_rules=[
                'Must be boolean type',
                'Must match collection_category expectations'
            ],
            related_fields=['collection_category', 'naming_pattern']
        )

        # === Migration and Compatibility ===
        specs['migration_source'] = FieldSpecification(
            name='migration_source',
            description='How the collection was created or migrated',
            type_info='string (predefined sources)',
            required=True,
            indexed=False,
            constraints=[
                'Should use standard migration source values',
                'Tracks collection history',
                'Helps with debugging migration issues',
                'Used for backward compatibility'
            ],
            examples=[
                'metadata_based',
                'suffix_based',
                'manual',
                'auto_create',
                'cli',
                'migration'
            ],
            validation_rules=[
                'Preferably from standard migration sources'
            ],
            related_fields=['legacy_collection_name', 'compatibility_version']
        )

        specs['legacy_collection_name'] = FieldSpecification(
            name='legacy_collection_name',
            description='Original collection name before migration (if applicable)',
            type_info='optional string',
            required=False,
            indexed=False,
            constraints=[
                'Should be set for migrated collections',
                'Preserves migration history',
                'Helps with debugging and rollback',
                'Not required for new collections'
            ],
            examples=[
                'my-project-docs',
                '__user_preferences',
                '_library_code'
            ],
            validation_rules=[
                'Should match original naming patterns'
            ],
            related_fields=['migration_source', 'original_name_pattern'],
            migration_notes='Always set this for collections migrated from existing systems'
        )

        # === Temporal and Organizational ===
        specs['created_at'] = FieldSpecification(
            name='created_at',
            description='ISO timestamp of collection creation',
            type_info='string (ISO 8601 timestamp)',
            required=True,
            indexed=False,
            constraints=[
                'Must be valid ISO 8601 timestamp',
                'Should include timezone information',
                'Automatically set on creation',
                'Used for auditing and debugging'
            ],
            examples=[
                '2024-01-01T12:00:00Z',
                '2024-01-01T12:00:00.123456Z'
            ],
            validation_rules=[
                'Must be valid ISO timestamp string'
            ],
            related_fields=['updated_at', 'version']
        )

        specs['updated_at'] = FieldSpecification(
            name='updated_at',
            description='ISO timestamp of last metadata update',
            type_info='string (ISO 8601 timestamp)',
            required=True,
            indexed=False,
            constraints=[
                'Must be valid ISO 8601 timestamp',
                'Automatically updated on changes',
                'Should be >= created_at',
                'Used for change tracking'
            ],
            examples=[
                '2024-01-01T12:30:00Z',
                '2024-01-01T12:30:00.789012Z'
            ],
            validation_rules=[
                'Must be valid ISO timestamp string',
                'Should be >= created_at'
            ],
            related_fields=['created_at', 'version']
        )

        specs['version'] = FieldSpecification(
            name='version',
            description='Metadata version number for change tracking',
            type_info='integer (starting from 1)',
            required=True,
            indexed=False,
            constraints=[
                'Must be positive integer',
                'Starts at 1 for new collections',
                'Incremented on each update',
                'Used for optimistic locking'
            ],
            examples=[
                '1',
                '5',
                '42'
            ],
            validation_rules=[
                'Must be integer >= 1'
            ],
            related_fields=['updated_at', 'compatibility_version']
        )

        specs['tags'] = FieldSpecification(
            name='tags',
            description='List of organizational tags for collection categorization',
            type_info='list of strings',
            required=False,
            indexed=False,
            constraints=[
                'Must be list of strings',
                'Each tag should be descriptive',
                'Used for organization and search',
                'Can be empty list'
            ],
            examples=[
                '["documentation", "user-facing"]',
                '["internal", "development"]',
                '[]'
            ],
            validation_rules=[
                'Must be list type',
                'All elements must be strings'
            ],
            related_fields=['category', 'priority']
        )

        specs['category'] = FieldSpecification(
            name='category',
            description='General category for high-level organization',
            type_info='string',
            required=False,
            indexed=False,
            constraints=[
                'Should be descriptive',
                'Used for high-level organization',
                'Can be any string value',
                'Defaults to "general"'
            ],
            examples=[
                'documentation',
                'development',
                'system',
                'user-data'
            ],
            default_value='general',
            validation_rules=[
                'Must be string type'
            ],
            related_fields=['tags', 'collection_type']
        )

        specs['priority'] = FieldSpecification(
            name='priority',
            description='Priority level for collection importance (1-5 scale)',
            type_info='integer (1=lowest, 5=highest)',
            required=False,
            indexed=False,
            constraints=[
                'Must be integer between 1 and 5',
                '1 = lowest priority',
                '5 = highest priority',
                'Used for resource allocation and search ranking'
            ],
            examples=[
                '1 (lowest)',
                '3 (medium)',
                '5 (highest)'
            ],
            default_value=3,
            validation_rules=[
                'Must be integer',
                'Must be between 1 and 5 inclusive'
            ],
            related_fields=['category', 'collection_category']
        )

        return specs

    def _build_usage_patterns(self) -> Dict[str, UsagePattern]:
        """Build documentation for common usage patterns."""
        patterns = {}

        patterns['project_collection'] = UsagePattern(
            name='Project Collection',
            description='Standard project-scoped collection for user content',
            scenario='Creating a collection for project-specific documentation',
            example_code="""
# Create metadata for project collection
metadata = MultiTenantMetadataSchema.create_for_project(
    project_name="my_awesome_project",
    collection_type="docs",
    created_by="user",
    access_level=AccessLevel.PRIVATE,
    tags=["documentation", "user-guide"],
    category="documentation",
    priority=4
)

# Convert to Qdrant payload for storage
qdrant_payload = metadata.to_qdrant_payload()
""",
            metadata_example={
                "project_id": "a1b2c3d4e5f6",
                "project_name": "my_awesome_project",
                "tenant_namespace": "my_awesome_project.docs",
                "collection_type": "docs",
                "collection_category": "project",
                "workspace_scope": "project",
                "access_level": "private",
                "mcp_readonly": False,
                "cli_writable": True,
                "created_by": "user"
            },
            notes=[
                "Project collections are isolated by project_id",
                "Use tenant_namespace for hierarchical filtering",
                "MCP server has read-write access by default"
            ],
            best_practices=[
                "Use descriptive collection_type names",
                "Set appropriate access_level for security",
                "Include relevant tags for organization",
                "Set priority based on collection importance"
            ]
        )

        patterns['system_collection'] = UsagePattern(
            name='System Collection',
            description='System-level collection with CLI-only write access',
            scenario='Creating a system collection for user preferences',
            example_code="""
# Create metadata for system collection
metadata = MultiTenantMetadataSchema.create_for_system(
    collection_name="__user_preferences",
    collection_type="memory_collection",
    created_by="system"
)

# System collections are automatically configured with:
# - mcp_readonly=False (CLI can write)
# - access_level=AccessLevel.PRIVATE
# - workspace_scope=WorkspaceScope.GLOBAL
""",
            metadata_example={
                "project_id": "systemhash12",
                "project_name": "system",
                "tenant_namespace": "system.memory_collection",
                "collection_type": "memory_collection",
                "collection_category": "system",
                "workspace_scope": "global",
                "access_level": "private",
                "mcp_readonly": False,
                "cli_writable": True,
                "is_reserved_name": True,
                "naming_pattern": "system_prefix",
                "original_name_pattern": "__user_preferences"
            },
            notes=[
                "System collections use __ prefix",
                "Not globally searchable by default",
                "CLI has write access, MCP typically read-only",
                "Used for system configuration and preferences"
            ],
            best_practices=[
                "Use meaningful names after __ prefix",
                "Document purpose clearly",
                "Limit to actual system data",
                "Consider security implications"
            ]
        )

        patterns['library_collection'] = UsagePattern(
            name='Library Collection',
            description='Library collection with MCP read-only access',
            scenario='Creating a library collection for shared code references',
            example_code="""
# Create metadata for library collection
metadata = MultiTenantMetadataSchema.create_for_library(
    collection_name="_code_references",
    collection_type="code_collection",
    created_by="cli"
)

# Library collections are automatically configured with:
# - mcp_readonly=True (MCP cannot write)
# - access_level=AccessLevel.SHARED
# - workspace_scope=WorkspaceScope.LIBRARY
""",
            metadata_example={
                "project_id": "libraryhash",
                "project_name": "library",
                "tenant_namespace": "library.code_collection",
                "collection_type": "code_collection",
                "collection_category": "library",
                "workspace_scope": "library",
                "access_level": "shared",
                "mcp_readonly": True,
                "cli_writable": True,
                "is_reserved_name": True,
                "naming_pattern": "library_prefix",
                "original_name_pattern": "_code_references"
            },
            notes=[
                "Library collections use _ prefix (not __)",
                "Globally searchable by default",
                "CLI has write access, MCP is read-only",
                "Shared across projects"
            ],
            best_practices=[
                "Use for shared reference data",
                "Populate via CLI or daemon",
                "Document access patterns",
                "Consider versioning strategies"
            ]
        )

        patterns['global_collection'] = UsagePattern(
            name='Global Collection',
            description='Global collection available system-wide',
            scenario='Creating a global collection for system-wide algorithms',
            example_code="""
# Create metadata for global collection
metadata = MultiTenantMetadataSchema.create_for_global(
    collection_name="algorithms",
    collection_type="global",
    created_by="system"
)

# Global collections are automatically configured with:
# - access_level=AccessLevel.PUBLIC
# - workspace_scope=WorkspaceScope.GLOBAL
""",
            metadata_example={
                "project_id": "globalhash12",
                "project_name": "global",
                "tenant_namespace": "global.global",
                "collection_type": "global",
                "collection_category": "global",
                "workspace_scope": "global",
                "access_level": "public",
                "mcp_readonly": False,
                "cli_writable": True,
                "naming_pattern": "global_collection"
            },
            notes=[
                "Global collections have no prefix",
                "Available to all projects",
                "Publicly accessible",
                "System-wide scope"
            ],
            best_practices=[
                "Use for truly global data",
                "Document access patterns",
                "Consider performance implications",
                "Implement proper governance"
            ]
        )

        patterns['migrated_collection'] = UsagePattern(
            name='Migrated Collection',
            description='Collection migrated from existing suffix-based naming',
            scenario='Migrating existing project-docs collection to metadata schema',
            example_code="""
# Migrate existing collection to metadata schema
original_name = "my-project-docs"
project_name, collection_type = original_name.split('-', 1)

metadata = MultiTenantMetadataSchema.create_for_project(
    project_name=project_name,
    collection_type=collection_type,
    created_by="migration"
)

# Set migration tracking fields
metadata.migration_source = "suffix_based"
metadata.legacy_collection_name = original_name
metadata.original_name_pattern = original_name
""",
            metadata_example={
                "project_id": "myprojecthash",
                "project_name": "my_project",
                "tenant_namespace": "my_project.docs",
                "collection_type": "docs",
                "collection_category": "project",
                "workspace_scope": "project",
                "migration_source": "suffix_based",
                "legacy_collection_name": "my-project-docs",
                "original_name_pattern": "my-project-docs",
                "created_by": "migration"
            },
            notes=[
                "Preserves original collection name",
                "Tracks migration source",
                "Maintains backward compatibility",
                "Enables rollback if needed"
            ],
            best_practices=[
                "Always set legacy_collection_name",
                "Use migration_source appropriately",
                "Test filtering after migration",
                "Document migration process"
            ]
        )

        return patterns

    def _build_migration_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Build documentation for migration scenarios."""
        scenarios = {}

        scenarios['suffix_to_metadata'] = {
            'name': 'Suffix-based to Metadata Migration',
            'description': 'Migrating from suffix-based collection naming to metadata-based filtering',
            'before': {
                'collections': ['my-project-docs', 'my-project-notes', 'other-project-docs'],
                'filtering': 'Based on collection name patterns'
            },
            'after': {
                'collections': ['docs', 'notes'],  # Shared collections
                'filtering': 'Based on metadata fields (project_id, tenant_namespace)'
            },
            'migration_steps': [
                '1. Scan existing collections to identify patterns',
                '2. Create metadata for each collection',
                '3. Add metadata to existing collections without renaming',
                '4. Update filtering logic to use metadata',
                '5. Test compatibility with existing code',
                '6. Document migration for team'
            ],
            'compatibility': 'Full backward compatibility maintained'
        }

        return scenarios

    def get_field_specification(self, field_name: str) -> Optional[FieldSpecification]:
        """Get complete specification for a specific field."""
        return self._field_specifications.get(field_name)

    def get_all_fields(self) -> Dict[str, FieldSpecification]:
        """Get specifications for all fields."""
        return self._field_specifications.copy()

    def get_required_fields(self) -> List[FieldSpecification]:
        """Get specifications for all required fields."""
        return [spec for spec in self._field_specifications.values() if spec.required]

    def get_indexed_fields(self) -> List[FieldSpecification]:
        """Get specifications for all indexed fields."""
        return [spec for spec in self._field_specifications.values() if spec.indexed]

    def get_usage_examples(self, pattern_name: str) -> Optional[UsagePattern]:
        """Get usage examples for a specific pattern."""
        return self._usage_patterns.get(pattern_name)

    def get_all_usage_patterns(self) -> Dict[str, UsagePattern]:
        """Get all usage patterns."""
        return self._usage_patterns.copy()

    def get_migration_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Get all migration scenarios."""
        return self._migration_scenarios.copy()

    def generate_field_reference(self) -> str:
        """Generate markdown field reference documentation."""
        markdown = "# Multi-Tenant Metadata Schema Field Reference\n\n"

        # Group fields by category
        categories = {
            'Core Tenant Isolation': ['project_id', 'project_name', 'tenant_namespace'],
            'Collection Classification': ['collection_type', 'collection_category', 'workspace_scope'],
            'Access Control': ['access_level', 'mcp_readonly', 'cli_writable', 'created_by'],
            'Reserved Naming': ['naming_pattern', 'is_reserved_name', 'original_name_pattern'],
            'Migration Support': ['migration_source', 'legacy_collection_name', 'compatibility_version'],
            'Temporal & Organizational': ['created_at', 'updated_at', 'version', 'tags', 'category', 'priority']
        }

        for category, field_names in categories.items():
            markdown += f"## {category}\n\n"

            for field_name in field_names:
                if field_name in self._field_specifications:
                    spec = self._field_specifications[field_name]
                    markdown += f"### {spec.name}\n\n"
                    markdown += f"**Description:** {spec.description}\n\n"
                    markdown += f"**Type:** {spec.type_info}\n\n"
                    markdown += f"**Required:** {'Yes' if spec.required else 'No'}\n\n"
                    markdown += f"**Indexed:** {'Yes' if spec.indexed else 'No'}\n\n"

                    if spec.constraints:
                        markdown += "**Constraints:**\n"
                        for constraint in spec.constraints:
                            markdown += f"- {constraint}\n"
                        markdown += "\n"

                    if spec.examples:
                        markdown += "**Examples:**\n"
                        for example in spec.examples:
                            markdown += f"- `{example}`\n"
                        markdown += "\n"

                    if spec.validation_rules:
                        markdown += "**Validation Rules:**\n"
                        for rule in spec.validation_rules:
                            markdown += f"- {rule}\n"
                        markdown += "\n"

                    if spec.related_fields:
                        markdown += f"**Related Fields:** {', '.join(spec.related_fields)}\n\n"

                    if spec.performance_notes:
                        markdown += f"**Performance Notes:** {spec.performance_notes}\n\n"

                    if spec.migration_notes:
                        markdown += f"**Migration Notes:** {spec.migration_notes}\n\n"

                    markdown += "---\n\n"

        return markdown

    def generate_usage_guide(self) -> str:
        """Generate markdown usage guide documentation."""
        markdown = "# Multi-Tenant Metadata Schema Usage Guide\n\n"

        markdown += "## Overview\n\n"
        markdown += "This guide provides practical examples for using the multi-tenant metadata schema "
        markdown += "in different scenarios. Each pattern includes code examples, metadata examples, "
        markdown += "and best practices.\n\n"

        for pattern_name, pattern in self._usage_patterns.items():
            markdown += f"## {pattern.name}\n\n"
            markdown += f"**Scenario:** {pattern.scenario}\n\n"
            markdown += f"{pattern.description}\n\n"

            markdown += "### Code Example\n\n"
            markdown += f"```python{pattern.example_code}\n```\n\n"

            markdown += "### Metadata Example\n\n"
            markdown += "```json\n"
            import json
            markdown += json.dumps(pattern.metadata_example, indent=2)
            markdown += "\n```\n\n"

            if pattern.notes:
                markdown += "### Notes\n\n"
                for note in pattern.notes:
                    markdown += f"- {note}\n"
                markdown += "\n"

            if pattern.best_practices:
                markdown += "### Best Practices\n\n"
                for practice in pattern.best_practices:
                    markdown += f"- {practice}\n"
                markdown += "\n"

            markdown += "---\n\n"

        return markdown


# Module-level convenience functions
def get_field_specification(field_name: str) -> Optional[FieldSpecification]:
    """Get specification for a specific field."""
    docs = SchemaDocumentation()
    return docs.get_field_specification(field_name)


def get_usage_pattern(pattern_name: str) -> Optional[UsagePattern]:
    """Get usage pattern by name."""
    docs = SchemaDocumentation()
    return docs.get_usage_examples(pattern_name)


def generate_documentation() -> Dict[str, str]:
    """Generate complete documentation set."""
    docs = SchemaDocumentation()
    return {
        'field_reference': docs.generate_field_reference(),
        'usage_guide': docs.generate_usage_guide()
    }


# Export all public classes and functions
__all__ = [
    'SchemaDocumentation',
    'FieldSpecification',
    'UsagePattern',
    'FieldType',
    'get_field_specification',
    'get_usage_pattern',
    'generate_documentation'
]