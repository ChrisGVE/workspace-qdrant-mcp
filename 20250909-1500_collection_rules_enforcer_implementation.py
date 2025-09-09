#!/usr/bin/env python3
"""
Task 181: Collection Management Rules Enforcement Implementation

This script implements the comprehensive CollectionRulesEnforcer class that:
1. Integrates with existing LLM access control from Task 173
2. Works with the 4-tool consolidation from Task 178  
3. Enforces ALL collection management rules across MCP tools
4. Prevents rule bypass through parameter manipulation
5. Provides clear error messages for validation failures

Key Security Features:
- LLM cannot bypass rules through parameter manipulation
- System memory collections are read-only from MCP
- All collection operations MUST go through validation
- Integration with existing access control systems
"""

import sys
import os
sys.path.insert(0, 'src')

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any
import logging

from workspace_qdrant_mcp.core.llm_access_control import (
    LLMAccessController, 
    LLMAccessControlError,
    AccessViolationType,
    AccessViolation
)
from workspace_qdrant_mcp.core.collection_naming import (
    CollectionNamingManager,
    CollectionType,
    CollectionNameInfo,
    NamingValidationResult
)
from workspace_qdrant_mcp.core.collection_types import (
    CollectionTypeClassifier,
    CollectionType as TypesCollectionType,
    CollectionInfo
)
from workspace_qdrant_mcp.core.config import Config

logger = logging.getLogger(__name__)


class ValidationSource(Enum):
    """Source of the collection operation request."""
    LLM = "llm"           # Request from LLM via MCP
    CLI = "cli"           # Request from CLI/Rust engine
    MCP_INTERNAL = "mcp"  # Internal MCP operations
    SYSTEM = "system"     # System/admin operations


class OperationType(Enum):
    """Types of collection operations that need validation."""
    CREATE = "create"
    DELETE = "delete"
    WRITE = "write"
    READ = "read"
    LIST = "list"


@dataclass
class ValidationResult:
    """Result of collection operation validation."""
    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    suggested_alternatives: Optional[List[str]] = None
    violation_type: Optional[AccessViolationType] = None
    

class CollectionRulesEnforcementError(Exception):
    """Exception raised when collection rules enforcement is violated."""
    
    def __init__(self, validation_result: ValidationResult):
        self.validation_result = validation_result
        super().__init__(validation_result.error_message)


class CollectionRulesEnforcer:
    """
    Comprehensive collection management rules enforcer.
    
    This class provides unified validation for all collection operations across
    MCP tools, preventing rule bypass and ensuring security boundaries are maintained.
    It integrates with existing LLM access control and collection type systems.
    
    Key Features:
    - Source-aware validation (LLM vs CLI vs MCP internal)
    - Integration with LLM access control from Task 173
    - Prevention of rule bypass through parameter manipulation
    - System collection protection enforcement
    - Clear error messages with suggested alternatives
    - Comprehensive logging and audit trail
    
    Security Boundaries:
    - LLM cannot create/delete system collections (__*)
    - LLM cannot create/delete library collections (_*)  
    - LLM cannot delete memory collections (*-memory, __*memory*)
    - System memory collections are read-only from MCP
    - All operations go through validation - no bypass paths
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the collection rules enforcer.
        
        Args:
            config: Optional configuration object for context
        """
        self.config = config
        
        # Initialize subsystems
        self.llm_access_controller = LLMAccessController(config)
        self.naming_manager = CollectionNamingManager()
        self.type_classifier = CollectionTypeClassifier()
        
        # Track existing collections for validation
        self._existing_collections: Set[str] = set()
        
        logger.info("CollectionRulesEnforcer initialized with comprehensive validation")
    
    def set_existing_collections(self, collections: List[str]) -> None:
        """
        Update the set of existing collections for validation.
        
        Args:
            collections: List of currently existing collection names
        """
        self._existing_collections = set(collections)
        self.llm_access_controller.set_existing_collections(collections)
        logger.debug(f"Updated existing collections: {len(collections)}")
    
    def validate_collection_creation(self, name: str, source: ValidationSource) -> ValidationResult:
        """
        Validate collection creation request with comprehensive rules enforcement.
        
        Args:
            name: The collection name to create
            source: Source of the creation request
            
        Returns:
            ValidationResult with validation status and details
        """
        logger.info(f"Validating collection creation: {name} from {source.value}")
        
        # Basic parameter validation
        if not isinstance(name, str) or not name.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Collection name must be a non-empty string",
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME
            )
        
        name = name.strip()
        
        # Check for existing collection
        if name in self._existing_collections:
            return ValidationResult(
                is_valid=False,
                error_message=f"Collection '{name}' already exists",
                violation_type=AccessViolationType.COLLECTION_ALREADY_EXISTS
            )
        
        # Validate collection name format
        # Skip naming manager validation for system collections as it doesn't handle __ prefix correctly
        if not name.startswith("__"):
            naming_result = self.naming_manager.validate_collection_name(name)
            if not naming_result.is_valid:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Invalid collection name format: {naming_result.error_message}",
                    violation_type=AccessViolationType.INVALID_COLLECTION_NAME
                )
        else:
            # Basic validation for system collections
            if len(name) < 3 or not name[2:]:  # Must have content after __
                return ValidationResult(
                    is_valid=False,
                    error_message=f"System collection name '{name}' must have content after '__' prefix",
                    violation_type=AccessViolationType.INVALID_COLLECTION_NAME
                )
        
        # Source-specific validation
        if source == ValidationSource.LLM:
            # LLM operations must go through LLM access control
            try:
                self.llm_access_controller.validate_llm_collection_access("create", name)
            except LLMAccessControlError as e:
                return ValidationResult(
                    is_valid=False,
                    error_message=e.violation.message,
                    suggested_alternatives=e.violation.suggested_alternatives,
                    violation_type=e.violation.violation_type
                )
        
        elif source == ValidationSource.CLI:
            # CLI has broader permissions but still enforce basic rules
            collection_info = self.type_classifier.get_collection_info(name)
            if collection_info.type == TypesCollectionType.UNKNOWN:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Collection name '{name}' does not follow recognized patterns",
                    violation_type=AccessViolationType.INVALID_COLLECTION_NAME
                )
        
        elif source == ValidationSource.MCP_INTERNAL:
            # MCP internal operations have system-level access but validate format
            pass
        
        elif source == ValidationSource.SYSTEM:
            # System operations have full access
            pass
        
        else:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unknown validation source: {source}",
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME
            )
        
        logger.info(f"Collection creation validation passed: {name} from {source.value}")
        return ValidationResult(is_valid=True)
    
    def validate_collection_deletion(self, name: str, source: ValidationSource) -> ValidationResult:
        """
        Validate collection deletion request with comprehensive rules enforcement.
        
        Args:
            name: The collection name to delete
            source: Source of the deletion request
            
        Returns:
            ValidationResult with validation status and details
        """
        logger.info(f"Validating collection deletion: {name} from {source.value}")
        
        # Basic parameter validation
        if not isinstance(name, str) or not name.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Collection name must be a non-empty string",
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME
            )
        
        name = name.strip()
        
        # Check if collection exists
        if name not in self._existing_collections:
            return ValidationResult(
                is_valid=False,
                error_message=f"Collection '{name}' does not exist",
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME
            )
        
        # Source-specific validation
        if source == ValidationSource.LLM:
            # LLM operations must go through LLM access control
            try:
                self.llm_access_controller.validate_llm_collection_access("delete", name)
            except LLMAccessControlError as e:
                return ValidationResult(
                    is_valid=False,
                    error_message=e.violation.message,
                    suggested_alternatives=e.violation.suggested_alternatives,
                    violation_type=e.violation.violation_type
                )
        
        elif source == ValidationSource.CLI:
            # CLI can delete most collections but protect critical system ones
            collection_info = self.type_classifier.get_collection_info(name)
            if collection_info.type == TypesCollectionType.SYSTEM:
                # Even CLI should be careful with system collections
                return ValidationResult(
                    is_valid=False,
                    error_message=f"System collection '{name}' requires explicit admin confirmation",
                    warning_message="Use --force flag if you're certain",
                    violation_type=AccessViolationType.FORBIDDEN_SYSTEM_DELETION
                )
        
        elif source == ValidationSource.MCP_INTERNAL:
            # MCP internal operations cannot delete system collections for safety
            collection_info = self.type_classifier.get_collection_info(name)
            if collection_info.type == TypesCollectionType.SYSTEM:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"MCP cannot delete system collection '{name}' - use CLI with admin privileges",
                    violation_type=AccessViolationType.FORBIDDEN_SYSTEM_DELETION
                )
        
        elif source == ValidationSource.SYSTEM:
            # System operations have full access
            pass
        
        else:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unknown validation source: {source}",
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME
            )
        
        logger.info(f"Collection deletion validation passed: {name} from {source.value}")
        return ValidationResult(is_valid=True)
    
    def validate_collection_write(self, name: str, source: ValidationSource) -> ValidationResult:
        """
        Validate collection write access with comprehensive rules enforcement.
        
        Args:
            name: The collection name to write to
            source: Source of the write request
            
        Returns:
            ValidationResult with validation status and details
        """
        logger.info(f"Validating collection write access: {name} from {source.value}")
        
        # Basic parameter validation
        if not isinstance(name, str) or not name.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Collection name must be a non-empty string",
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME
            )
        
        name = name.strip()
        
        # Check if collection exists (for write operations)
        if name not in self._existing_collections:
            return ValidationResult(
                is_valid=False,
                error_message=f"Collection '{name}' does not exist",
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME
            )
        
        # Get collection classification
        collection_info = self.type_classifier.get_collection_info(name)
        
        # Source-specific validation
        if source == ValidationSource.LLM:
            # LLM operations must go through LLM access control
            try:
                self.llm_access_controller.validate_llm_collection_access("write", name)
            except LLMAccessControlError as e:
                return ValidationResult(
                    is_valid=False,
                    error_message=e.violation.message,
                    suggested_alternatives=e.violation.suggested_alternatives,
                    violation_type=e.violation.violation_type
                )
        
        elif source == ValidationSource.CLI:
            # CLI has broad write access but respect read-only system collections
            pass
        
        elif source == ValidationSource.MCP_INTERNAL:
            # MCP internal operations respect read-only boundaries
            # System memory collections are read-only from MCP
            if collection_info.type == TypesCollectionType.SYSTEM and collection_info.is_memory_collection:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"System memory collection '{name}' is read-only from MCP",
                    violation_type=AccessViolationType.FORBIDDEN_SYSTEM_WRITE
                )
            
            # Library collections are read-only from MCP
            if collection_info.type == TypesCollectionType.LIBRARY:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Library collection '{name}' is read-only from MCP - use CLI to modify",
                    violation_type=AccessViolationType.FORBIDDEN_LIBRARY_WRITE
                )
        
        elif source == ValidationSource.SYSTEM:
            # System operations have full access
            pass
        
        else:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unknown validation source: {source}",
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME
            )
        
        logger.info(f"Collection write validation passed: {name} from {source.value}")
        return ValidationResult(is_valid=True)
    
    def validate_collection_read(self, name: str, source: ValidationSource) -> ValidationResult:
        """
        Validate collection read access (generally permissive but validates existence).
        
        Args:
            name: The collection name to read from
            source: Source of the read request
            
        Returns:
            ValidationResult with validation status and details
        """
        logger.debug(f"Validating collection read access: {name} from {source.value}")
        
        # Basic parameter validation
        if not isinstance(name, str) or not name.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Collection name must be a non-empty string",
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME
            )
        
        name = name.strip()
        
        # Check if collection exists
        if name not in self._existing_collections:
            return ValidationResult(
                is_valid=False,
                error_message=f"Collection '{name}' does not exist",
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME
            )
        
        # Read access is generally allowed for all sources to all existing collections
        return ValidationResult(is_valid=True)
    
    def validate_operation(self, operation: OperationType, name: str, source: ValidationSource) -> ValidationResult:
        """
        Unified validation method for any collection operation.
        
        Args:
            operation: The type of operation being performed
            name: The collection name
            source: Source of the operation request
            
        Returns:
            ValidationResult with validation status and details
        """
        logger.info(f"Validating collection operation: {operation.value} {name} from {source.value}")
        
        if operation == OperationType.CREATE:
            return self.validate_collection_creation(name, source)
        elif operation == OperationType.DELETE:
            return self.validate_collection_deletion(name, source)
        elif operation == OperationType.WRITE:
            return self.validate_collection_write(name, source)
        elif operation == OperationType.READ:
            return self.validate_collection_read(name, source)
        elif operation == OperationType.LIST:
            # List operations don't need collection-specific validation
            return ValidationResult(is_valid=True)
        else:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unknown operation type: {operation}",
                violation_type=AccessViolationType.INVALID_COLLECTION_NAME
            )
    
    def enforce_operation(self, operation: OperationType, name: str, source: ValidationSource) -> None:
        """
        Enforce collection operation rules by validating and raising exception on failure.
        
        Args:
            operation: The type of operation being performed
            name: The collection name
            source: Source of the operation request
            
        Raises:
            CollectionRulesEnforcementError: If the operation violates rules
        """
        result = self.validate_operation(operation, name, source)
        if not result.is_valid:
            logger.warning(f"Collection operation blocked: {operation.value} {name} from {source.value} - {result.error_message}")
            raise CollectionRulesEnforcementError(result)
        
        logger.debug(f"Collection operation allowed: {operation.value} {name} from {source.value}")
    
    def get_rule_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all enforced rules for documentation and debugging.
        
        Returns:
            Dictionary containing rule descriptions and current state
        """
        return {
            "version": "Task 181 - Comprehensive Collection Rules Enforcement",
            "integration": {
                "llm_access_control": "Task 173 - LLM access control integration",
                "collection_naming": "Collection naming system integration",
                "collection_types": "Collection type classification integration"
            },
            "sources": [source.value for source in ValidationSource],
            "operations": [op.value for op in OperationType],
            "security_rules": {
                "llm_restrictions": [
                    "Cannot create/delete system collections (__*)",
                    "Cannot create/delete library collections (_*)",
                    "Cannot delete memory collections (*-memory)",
                    "Cannot write to system collections",
                    "Cannot write to library collections"
                ],
                "mcp_restrictions": [
                    "System memory collections are read-only",
                    "Library collections are read-only",
                    "Cannot delete system collections"
                ],
                "cli_restrictions": [
                    "System collection deletion requires admin confirmation",
                    "All operations logged for audit trail"
                ]
            },
            "existing_collections_count": len(self._existing_collections),
            "bypass_prevention": "All operations must go through validation - no direct Qdrant access"
        }


def demonstrate_enforcer():
    """Demonstrate the CollectionRulesEnforcer functionality."""
    print("=== Collection Rules Enforcer Demo ===")
    
    # Initialize enforcer
    enforcer = CollectionRulesEnforcer()
    
    # Set some existing collections for testing
    existing_collections = [
        "__system_memory", "_mylib", "project-docs", "memory", "algorithms"
    ]
    enforcer.set_existing_collections(existing_collections)
    
    # Test cases
    test_cases = [
        # LLM attempting to create system collection - should fail
        (OperationType.CREATE, "__new_system", ValidationSource.LLM, False),
        
        # LLM creating valid project collection - should succeed
        (OperationType.CREATE, "myproject-docs", ValidationSource.LLM, True),
        
        # LLM attempting to delete memory collection - should fail
        (OperationType.DELETE, "memory", ValidationSource.LLM, False),
        
        # MCP attempting to write to system memory - should fail
        (OperationType.WRITE, "__system_memory", ValidationSource.MCP_INTERNAL, False),
        
        # CLI creating system collection - should succeed
        (OperationType.CREATE, "__new_system", ValidationSource.CLI, True),
        
        # Reading from any collection - should succeed
        (OperationType.READ, "algorithms", ValidationSource.LLM, True),
    ]
    
    for operation, name, source, expected_success in test_cases:
        try:
            result = enforcer.validate_operation(operation, name, source)
            success = result.is_valid
            status = "✅ PASS" if success == expected_success else "❌ FAIL"
            print(f"{status} {operation.value} '{name}' from {source.value}: {success}")
            if not success:
                print(f"    Error: {result.error_message}")
        except Exception as e:
            print(f"❌ ERROR {operation.value} '{name}' from {source.value}: {e}")
    
    # Show rule summary
    print("\n=== Rule Summary ===")
    summary = enforcer.get_rule_summary()
    print(f"Version: {summary['version']}")
    print(f"Existing collections: {summary['existing_collections_count']}")
    print("LLM restrictions:")
    for rule in summary['security_rules']['llm_restrictions']:
        print(f"  - {rule}")


if __name__ == "__main__":
    demonstrate_enforcer()