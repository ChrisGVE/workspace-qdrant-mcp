"""
LLM Collection Management Access Control System for workspace-qdrant-mcp.

This module provides comprehensive access control for LLM operations on Qdrant collections,
preventing unauthorized creation, deletion, or modification of protected collections.
It enforces naming restrictions based on the collection type hierarchy and integrates
with the existing collection type system and configuration.

Key security features:
- Prevents LLM from creating/deleting protected system and library collections
- Enforces naming restrictions for global collections and memory collections
- Validates against forbidden patterns and existing collections
- Provides clear error messages for access violations
- Integrates seamlessly with existing collection type classification

Collection Protection Rules:
- SYSTEM collections (__*): LLM cannot create or delete
- LIBRARY collections (_*): LLM cannot create or delete  
- GLOBAL collections: LLM cannot create collections with reserved names
- MEMORY collections: LLM cannot delete, special creation rules apply
- PROJECT collections: LLM can create/delete if properly formatted

Access Control Matrix:
                | Create | Delete | Write |
SYSTEM (__*)    |   ❌   |   ❌   |   ❌  |
LIBRARY (_*)    |   ❌   |   ❌   |   ❌  |
GLOBAL          |   ❌   |   ❌   |   ✅  |
PROJECT         |   ✅   |   ✅   |   ✅  |
MEMORY          |   ⚠️   |   ❌   |   ✅  |

Legend: ✅ = Allowed, ❌ = Forbidden, ⚠️ = Conditional
"""

import re
from typing import Dict, List, Optional, Set, Union, Any
from dataclasses import dataclass
from enum import Enum

try:
    from .collection_types import (
        CollectionType, 
        CollectionInfo, 
        CollectionTypeClassifier,
        SYSTEM_PREFIX,
        LIBRARY_PREFIX, 
        GLOBAL_COLLECTIONS
    )
    from .collection_naming import CollectionNamingManager, CollectionType as NamingCollectionType
    from .config import Config
except ImportError:
    # For direct imports when not used as a package
    from collection_types import (
        CollectionType, 
        CollectionInfo, 
        CollectionTypeClassifier,
        SYSTEM_PREFIX,
        LIBRARY_PREFIX, 
        GLOBAL_COLLECTIONS
    )
    from collection_naming import CollectionNamingManager, CollectionType as NamingCollectionType
    from config import Config


class AccessViolationType(Enum):
    """Types of access control violations."""
    
    FORBIDDEN_SYSTEM_CREATION = "forbidden_system_creation"
    FORBIDDEN_LIBRARY_CREATION = "forbidden_library_creation"
    FORBIDDEN_GLOBAL_CREATION = "forbidden_global_creation"
    FORBIDDEN_SYSTEM_DELETION = "forbidden_system_deletion"
    FORBIDDEN_LIBRARY_DELETION = "forbidden_library_deletion"
    FORBIDDEN_MEMORY_DELETION = "forbidden_memory_deletion"
    FORBIDDEN_SYSTEM_WRITE = "forbidden_system_write"
    FORBIDDEN_LIBRARY_WRITE = "forbidden_library_write"
    COLLECTION_ALREADY_EXISTS = "collection_already_exists"
    INVALID_COLLECTION_NAME = "invalid_collection_name"
    RESERVED_NAME_CONFLICT = "reserved_name_conflict"


@dataclass
class AccessViolation:
    """Information about an access control violation."""
    
    violation_type: AccessViolationType
    collection_name: str
    operation: str
    message: str
    suggested_alternatives: Optional[List[str]] = None


class LLMAccessControlError(Exception):
    """Exception raised when LLM access control is violated."""
    
    def __init__(self, violation: AccessViolation):
        self.violation = violation
        super().__init__(violation.message)


class LLMAccessController:
    """
    Access control system for LLM operations on collections.
    
    This class provides comprehensive validation for LLM operations,
    ensuring that LLMs cannot perform unauthorized actions on protected
    collections while providing helpful error messages and alternatives.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the LLM access controller.
        
        Args:
            config: Optional configuration object for context
        """
        self.config = config
        self.classifier = CollectionTypeClassifier()
        
        # Cache forbidden patterns for performance
        self._forbidden_patterns = self._build_forbidden_patterns()
        
        # Set of existing collections (would be populated by calling code)
        self._existing_collections: Set[str] = set()
    
    def set_existing_collections(self, collections: List[str]) -> None:
        """
        Update the set of existing collections for validation.
        
        Args:
            collections: List of currently existing collection names
        """
        self._existing_collections = set(collections)
    
    def can_llm_create_collection(self, name: str) -> bool:
        """
        Check if LLM is allowed to create a collection with the given name.
        
        Args:
            name: The collection name to validate
            
        Returns:
            bool: True if LLM can create the collection, False otherwise
        """
        try:
            self.validate_llm_collection_access("create", name)
            return True
        except LLMAccessControlError:
            return False
    
    def can_llm_delete_collection(self, name: str) -> bool:
        """
        Check if LLM is allowed to delete a collection with the given name.
        
        Args:
            name: The collection name to validate
            
        Returns:
            bool: True if LLM can delete the collection, False otherwise
        """
        try:
            self.validate_llm_collection_access("delete", name)
            return True
        except LLMAccessControlError:
            return False
    
    def can_llm_write_to_collection(self, name: str) -> bool:
        """
        Check if LLM is allowed to write to a collection with the given name.
        
        Args:
            name: The collection name to validate
            
        Returns:
            bool: True if LLM can write to the collection, False otherwise
        """
        try:
            self.validate_llm_collection_access("write", name)
            return True
        except LLMAccessControlError:
            return False
    
    def validate_llm_collection_access(self, operation: str, name: str) -> None:
        """
        Unified validation function that raises exceptions for access violations.
        
        Args:
            operation: The operation to validate ('create', 'delete', 'write')
            name: The collection name to validate
            
        Raises:
            LLMAccessControlError: If the operation violates access control rules
        """
        if not isinstance(name, str) or not name.strip():
            violation = AccessViolation(
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME,
                collection_name=name,
                operation=operation,
                message=f"Invalid collection name: '{name}' - collection names cannot be empty"
            )
            raise LLMAccessControlError(violation)
        
        name = name.strip()
        operation = operation.lower()
        
        # Validate collection name format - use the naming manager
        naming_manager = CollectionNamingManager()
        validation_result = naming_manager.validate_collection_name(name)
        if not validation_result.is_valid:
            violation = AccessViolation(
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME,
                collection_name=name,
                operation=operation,
                message=f"Invalid collection name format: '{name}' - {validation_result.error_message}"
            )
            raise LLMAccessControlError(violation)
        
        # Get collection information
        collection_info = self.classifier.get_collection_info(name)
        
        # Apply operation-specific validation
        if operation == "create":
            self._validate_create_access(name, collection_info)
        elif operation == "delete":
            self._validate_delete_access(name, collection_info)
        elif operation == "write":
            self._validate_write_access(name, collection_info)
        else:
            violation = AccessViolation(
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME,
                collection_name=name,
                operation=operation,
                message=f"Invalid operation: '{operation}' - must be 'create', 'delete', or 'write'"
            )
            raise LLMAccessControlError(violation)
    
    def _validate_create_access(self, name: str, info: CollectionInfo) -> None:
        """Validate LLM create access for a collection."""
        
        # Check if collection already exists
        if name in self._existing_collections:
            violation = AccessViolation(
                violation_type=AccessViolationType.COLLECTION_ALREADY_EXISTS,
                collection_name=name,
                operation="create",
                message=f"Collection '{name}' already exists - cannot create duplicate collection"
            )
            raise LLMAccessControlError(violation)
        
        # System collections - forbidden
        if info.type == CollectionType.SYSTEM:
            alternatives = self._suggest_alternatives_for_system(name)
            violation = AccessViolation(
                violation_type=AccessViolationType.FORBIDDEN_SYSTEM_CREATION,
                collection_name=name,
                operation="create",
                message=f"LLM cannot create system collection '{name}' - system collections (prefix '__') are managed by CLI only",
                suggested_alternatives=alternatives
            )
            raise LLMAccessControlError(violation)
        
        # Library collections - forbidden
        if info.type == CollectionType.LIBRARY:
            alternatives = self._suggest_alternatives_for_library(name)
            violation = AccessViolation(
                violation_type=AccessViolationType.FORBIDDEN_LIBRARY_CREATION,
                collection_name=name,
                operation="create",
                message=f"LLM cannot create library collection '{name}' - library collections (prefix '_') are managed by CLI only",
                suggested_alternatives=alternatives
            )
            raise LLMAccessControlError(violation)
        
        # Global collections - forbidden (reserved names)
        if info.type == CollectionType.GLOBAL:
            alternatives = self._suggest_alternatives_for_global(name)
            violation = AccessViolation(
                violation_type=AccessViolationType.FORBIDDEN_GLOBAL_CREATION,
                collection_name=name,
                operation="create",
                message=f"LLM cannot create collection '{name}' - this is a reserved global collection name",
                suggested_alternatives=alternatives
            )
            raise LLMAccessControlError(violation)
        
        # Project collections - allowed but validate format
        if info.type == CollectionType.PROJECT:
            # Additional validation for project collections
            if info.project_name and info.suffix:
                # Check for reserved suffixes that might conflict
                reserved_suffixes = ["memory", "system", "config", "admin"]
                if info.suffix.lower() in reserved_suffixes:
                    alternatives = [f"{info.project_name}-docs", f"{info.project_name}-data", f"{info.project_name}-content"]
                    violation = AccessViolation(
                        violation_type=AccessViolationType.RESERVED_NAME_CONFLICT,
                        collection_name=name,
                        operation="create",
                        message=f"Collection suffix '{info.suffix}' is reserved - use a different suffix",
                        suggested_alternatives=alternatives
                    )
                    raise LLMAccessControlError(violation)
        
        # Unknown collections - forbidden for safety
        if info.type == CollectionType.UNKNOWN:
            violation = AccessViolation(
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME,
                collection_name=name,
                operation="create",
                message=f"Collection name '{name}' does not follow recognized patterns - use 'project-suffix' format",
                suggested_alternatives=[f"{name.replace('_', '-')}-docs", f"myproject-{name.replace('_', '')}"]
            )
            raise LLMAccessControlError(violation)
    
    def _validate_delete_access(self, name: str, info: CollectionInfo) -> None:
        """Validate LLM delete access for a collection."""
        
        # System collections - forbidden
        if info.type == CollectionType.SYSTEM:
            violation = AccessViolation(
                violation_type=AccessViolationType.FORBIDDEN_SYSTEM_DELETION,
                collection_name=name,
                operation="delete",
                message=f"LLM cannot delete system collection '{name}' - system collections are protected"
            )
            raise LLMAccessControlError(violation)
        
        # Library collections - forbidden
        if info.type == CollectionType.LIBRARY:
            violation = AccessViolation(
                violation_type=AccessViolationType.FORBIDDEN_LIBRARY_DELETION,
                collection_name=name,
                operation="delete",
                message=f"LLM cannot delete library collection '{name}' - library collections are protected"
            )
            raise LLMAccessControlError(violation)
        
        # Memory collections - forbidden (regardless of type)
        if info.is_memory_collection:
            violation = AccessViolation(
                violation_type=AccessViolationType.FORBIDDEN_MEMORY_DELETION,
                collection_name=name,
                operation="delete",
                message=f"LLM cannot delete memory collection '{name}' - memory collections contain persistent state"
            )
            raise LLMAccessControlError(violation)
        
        # Global collections - forbidden (these are system-wide)
        if info.type == CollectionType.GLOBAL:
            violation = AccessViolation(
                violation_type=AccessViolationType.FORBIDDEN_LIBRARY_DELETION,
                collection_name=name,
                operation="delete",
                message=f"LLM cannot delete global collection '{name}' - global collections are system-wide resources"
            )
            raise LLMAccessControlError(violation)
        
        # Project collections - allowed (except memory ones, handled above)
        # Unknown collections - forbidden for safety
        if info.type == CollectionType.UNKNOWN:
            violation = AccessViolation(
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME,
                collection_name=name,
                operation="delete",
                message=f"Cannot delete collection '{name}' - unrecognized collection type"
            )
            raise LLMAccessControlError(violation)
    
    def _validate_write_access(self, name: str, info: CollectionInfo) -> None:
        """Validate LLM write access for a collection."""
        
        # System collections - forbidden (CLI-only writable)
        if info.type == CollectionType.SYSTEM:
            violation = AccessViolation(
                violation_type=AccessViolationType.FORBIDDEN_SYSTEM_WRITE,
                collection_name=name,
                operation="write",
                message=f"LLM cannot write to system collection '{name}' - system collections are CLI-writable only"
            )
            raise LLMAccessControlError(violation)
        
        # Library collections - forbidden (MCP read-only)
        if info.type == CollectionType.LIBRARY:
            violation = AccessViolation(
                violation_type=AccessViolationType.FORBIDDEN_LIBRARY_WRITE,
                collection_name=name,
                operation="write",
                message=f"LLM cannot write to library collection '{name}' - library collections are read-only"
            )
            raise LLMAccessControlError(violation)
        
        # Global, Project collections - allowed
        # Memory collections - allowed (data storage)
        # Unknown collections - forbidden
        if info.type == CollectionType.UNKNOWN:
            violation = AccessViolation(
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME,
                collection_name=name,
                operation="write",
                message=f"Cannot write to collection '{name}' - unrecognized collection type"
            )
            raise LLMAccessControlError(violation)
    
    def get_forbidden_collection_patterns(self) -> List[str]:
        """
        Get list of patterns that LLM cannot use for collection names.
        
        Returns:
            List[str]: Forbidden patterns with explanations
        """
        return [
            f"{SYSTEM_PREFIX}* (system collections - CLI managed)",
            f"{LIBRARY_PREFIX}* (library collections - CLI managed)", 
            f"Global names: {', '.join(GLOBAL_COLLECTIONS)}",
            "*-memory (memory collections - special handling)",
            "Invalid formats (must use 'project-suffix' pattern)"
        ]
    
    def suggest_collection_name(self, intended_name: str, purpose: str = "general") -> List[str]:
        """
        Suggest valid collection names based on intended name and purpose.
        
        Args:
            intended_name: The name the user wanted to use
            purpose: The purpose of the collection ("documents", "code", "data", etc.)
            
        Returns:
            List[str]: Suggested valid collection names
        """
        # Normalize the intended name
        normalized_base = re.sub(r'[^a-z0-9]', '', intended_name.lower()) or "myproject"
        
        # Generate suggestions based on purpose
        purpose_suffixes = {
            "documents": ["docs", "documents", "content"],
            "code": ["code", "source", "implementation"],
            "data": ["data", "dataset", "information"],
            "memory": ["state", "context", "workspace"],  # Avoid "memory" suffix
            "general": ["docs", "data", "content"]
        }
        
        suffixes = purpose_suffixes.get(purpose.lower(), purpose_suffixes["general"])
        
        suggestions = []
        for suffix in suffixes:
            suggestion = f"{normalized_base}-{suffix}"
            if suggestion not in self._existing_collections:
                suggestions.append(suggestion)
        
        # Fallback suggestions if none generated
        if not suggestions:
            for i in range(1, 4):
                fallback = f"{normalized_base}-collection{i}"
                if fallback not in self._existing_collections:
                    suggestions.append(fallback)
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _build_forbidden_patterns(self) -> Dict[str, str]:
        """Build a dictionary of forbidden patterns with reasons."""
        patterns = {}
        
        # System patterns
        patterns[f"^{re.escape(SYSTEM_PREFIX)}"] = "System collections are CLI-managed only"
        
        # Library patterns  
        patterns[f"^{re.escape(LIBRARY_PREFIX)}(?!{re.escape(SYSTEM_PREFIX)})"] = "Library collections are CLI-managed only"
        
        # Global collection names
        for global_name in GLOBAL_COLLECTIONS:
            patterns[f"^{re.escape(global_name)}$"] = f"'{global_name}' is a reserved global collection name"
        
        return patterns
    
    def _suggest_alternatives_for_system(self, name: str) -> List[str]:
        """Suggest alternatives for blocked system collection names."""
        if name.startswith(SYSTEM_PREFIX):
            base_name = name[len(SYSTEM_PREFIX):]
            return self.suggest_collection_name(base_name, "general")
        return self.suggest_collection_name(name, "general")
    
    def _suggest_alternatives_for_library(self, name: str) -> List[str]:
        """Suggest alternatives for blocked library collection names."""
        if name.startswith(LIBRARY_PREFIX):
            base_name = name[len(LIBRARY_PREFIX):]
            return self.suggest_collection_name(base_name, "general")
        return self.suggest_collection_name(name, "general")
    
    def _suggest_alternatives_for_global(self, name: str) -> List[str]:
        """Suggest alternatives for blocked global collection names."""
        purpose_mapping = {
            "algorithms": "code",
            "codebase": "code", 
            "context": "data",
            "documents": "documents",
            "knowledge": "data",
            "memory": "memory",
            "projects": "general",
            "workspace": "general"
        }
        purpose = purpose_mapping.get(name.lower(), "general")
        return self.suggest_collection_name(name, purpose)


# Module-level convenience functions

def can_llm_create_collection(name: str, config: Optional[Config] = None) -> bool:
    """
    Check if LLM can create a collection with the given name.
    
    Args:
        name: Collection name to validate
        config: Optional configuration context
        
    Returns:
        bool: True if creation is allowed
    """
    controller = LLMAccessController(config)
    return controller.can_llm_create_collection(name)


def can_llm_delete_collection(name: str, config: Optional[Config] = None) -> bool:
    """
    Check if LLM can delete a collection with the given name.
    
    Args:
        name: Collection name to validate
        config: Optional configuration context
        
    Returns:
        bool: True if deletion is allowed
    """
    controller = LLMAccessController(config)
    return controller.can_llm_delete_collection(name)


def validate_llm_collection_access(operation: str, name: str, config: Optional[Config] = None) -> None:
    """
    Validate LLM access for collection operations.
    
    Args:
        operation: Operation to validate ('create', 'delete', 'write')
        name: Collection name to validate
        config: Optional configuration context
        
    Raises:
        LLMAccessControlError: If operation violates access control
    """
    controller = LLMAccessController(config)
    controller.validate_llm_collection_access(operation, name)


def get_forbidden_collection_patterns(config: Optional[Config] = None) -> List[str]:
    """
    Get list of forbidden collection patterns for LLM.
    
    Args:
        config: Optional configuration context
        
    Returns:
        List[str]: Forbidden patterns with explanations
    """
    controller = LLMAccessController(config)
    return controller.get_forbidden_collection_patterns()


# Export public interface
__all__ = [
    # Classes
    'AccessViolationType',
    'AccessViolation', 
    'LLMAccessControlError',
    'LLMAccessController',
    
    # Convenience functions
    'can_llm_create_collection',
    'can_llm_delete_collection',
    'validate_llm_collection_access',
    'get_forbidden_collection_patterns'
]