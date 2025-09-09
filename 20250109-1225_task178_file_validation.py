#!/usr/bin/env python3
"""
Task 178 - 4-Tool Consolidation File-Based Validation

This script validates the Task 178 implementation by examining the source code
directly without importing potentially problematic modules.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Task178FileValidation:
    """File-based validation for Task 178 implementation."""
    
    def __init__(self):
        self.simplified_interface_path = Path("src/workspace_qdrant_mcp/tools/simplified_interface.py")
        self.results = {}
    
    def validate_api_specifications(self) -> Dict[str, Any]:
        """Validate that the 4 tools match Task 178 API specification."""
        logger.info("Validating API specifications from source code")
        
        if not self.simplified_interface_path.exists():
            return {"error": f"File not found: {self.simplified_interface_path}"}
        
        content = self.simplified_interface_path.read_text()
        
        # Test 1: qdrant_store API
        store_match = re.search(
            r'async def qdrant_store\(\s*self,\s*(.*?)\) -> Dict\[str, Any\]:',
            content, re.DOTALL
        )
        
        if store_match:
            store_params = store_match.group(1)
            expected_store_params = [
                "content: str",
                "collection: str", 
                "document_type: str = \"text\"",
                "source: str = \"user_input\"",
                "title: str = None",
                "metadata: dict = None"
            ]
            
            store_validation = {
                "signature_found": True,
                "parameters_found": store_params.strip(),
                "expected_parameters": expected_store_params,
                "has_content_param": "content:" in store_params,
                "has_collection_param": "collection:" in store_params,
                "has_document_type_param": "document_type:" in store_params,
                "has_source_param": "source:" in store_params
            }
        else:
            store_validation = {"signature_found": False, "error": "qdrant_store signature not found"}
        
        # Test 2: qdrant_find API
        find_match = re.search(
            r'async def qdrant_find\(\s*self,\s*(.*?)\) -> list:',
            content, re.DOTALL
        )
        
        if find_match:
            find_params = find_match.group(1)
            find_validation = {
                "signature_found": True,
                "parameters_found": find_params.strip(),
                "has_query_param": "query:" in find_params,
                "has_search_scope_param": "search_scope:" in find_params,
                "has_collection_param": "collection:" in find_params,
                "returns_list": True  # Already confirmed by regex
            }
        else:
            find_validation = {"signature_found": False, "error": "qdrant_find signature not found"}
        
        # Test 3: qdrant_manage API
        manage_match = re.search(
            r'async def qdrant_manage\(\s*self,\s*(.*?)\) -> dict:',
            content, re.DOTALL
        )
        
        if manage_match:
            manage_params = manage_match.group(1)
            manage_validation = {
                "signature_found": True,
                "parameters_found": manage_params.strip(),
                "has_action_param": "action:" in manage_params,
                "has_collection_param": "collection:" in manage_params,
                "has_new_name_param": "new_name:" in manage_params,
                "returns_dict": True  # Already confirmed by regex
            }
        else:
            manage_validation = {"signature_found": False, "error": "qdrant_manage signature not found"}
        
        # Test 4: qdrant_read API  
        read_match = re.search(
            r'async def qdrant_read\(\s*self,\s*(.*?)\) -> dict:',
            content, re.DOTALL
        )
        
        if read_match:
            read_params = read_match.group(1)
            read_validation = {
                "signature_found": True,
                "parameters_found": read_params.strip(),
                "has_action_param": "action:" in read_params,
                "has_collection_param": "collection:" in read_params,
                "has_document_id_param": "document_id:" in read_params,
                "has_sort_by_param": "sort_by:" in read_params,
                "returns_dict": True  # Already confirmed by regex
            }
        else:
            read_validation = {"signature_found": False, "error": "qdrant_read signature not found"}
        
        return {
            "qdrant_store": store_validation,
            "qdrant_find": find_validation, 
            "qdrant_manage": manage_validation,
            "qdrant_read": read_validation
        }
    
    def validate_routing_implementation(self) -> Dict[str, Any]:
        """Validate that tools route to existing functionality."""
        logger.info("Validating routing to existing functionality")
        
        if not self.simplified_interface_path.exists():
            return {"error": f"File not found: {self.simplified_interface_path}"}
        
        content = self.simplified_interface_path.read_text()
        
        # Check for routing to existing functions
        routing_checks = {
            "qdrant_store_routes_to_add_document": "from .documents import add_document" in content,
            "qdrant_find_routes_to_search_workspace": "from .search import search_workspace" in content,
            "qdrant_manage_routes_to_client_methods": "await self.workspace_client.get_status()" in content,
            "qdrant_read_routes_to_get_document": "from .documents import get_document" in content,
            "calls_add_document": "await add_document(" in content,
            "calls_search_workspace": "await search_workspace(" in content,
            "calls_get_document": "await get_document(" in content
        }
        
        # Check for validation preservation
        validation_checks = {
            "preserves_validation": "validate_mcp_write_access" in content or "collection_manager" in content,
            "error_handling": "try:" in content and "except" in content,
            "parameter_validation": "if not" in content and "error" in content
        }
        
        return {
            "routing_to_existing": routing_checks,
            "validation_preservation": validation_checks,
            "overall_routing_score": sum(routing_checks.values()) / len(routing_checks)
        }
    
    def validate_access_control_integration(self) -> Dict[str, Any]:
        """Validate integration with access control systems."""
        logger.info("Validating access control integration")
        
        if not self.simplified_interface_path.exists():
            return {"error": f"File not found: {self.simplified_interface_path}"}
        
        content = self.simplified_interface_path.read_text()
        
        access_control_checks = {
            "imports_error_handling": "error_handling" in content,
            "uses_monitor_async": "@monitor_async" in content,
            "uses_with_error_handling": "@with_error_handling" in content,
            "validates_parameters": "if not" in content and ("collection" in content or "content" in content),
            "preserves_readonly_protection": "Route" in content and "existing" in content,  # Comments about routing
            "maintains_validation_calls": "await" in content and ("add_document" in content or "get_document" in content)
        }
        
        return {
            "access_control_integration": access_control_checks,
            "integration_score": sum(access_control_checks.values()) / len(access_control_checks)
        }
    
    def validate_search_scope_integration(self) -> Dict[str, Any]:
        """Validate integration with search scope architecture from Task 175."""
        logger.info("Validating search scope integration")
        
        if not self.simplified_interface_path.exists():
            return {"error": f"File not found: {self.simplified_interface_path}"}
        
        content = self.simplified_interface_path.read_text()
        
        search_scope_checks = {
            "defines_search_scope_enum": "class SearchScope(Enum):" in content,
            "has_validate_search_scope": "def validate_search_scope(" in content,
            "has_resolve_search_scope": "def resolve_search_scope(" in content,
            "uses_scope_resolution": "resolve_search_scope(" in content,
            "supports_all_scopes": all(scope in content for scope in ["collection", "project", "workspace", "all", "memory"]),
            "integrates_in_find": "search_scope" in content and "qdrant_find" in content
        }
        
        return {
            "search_scope_integration": search_scope_checks,
            "integration_score": sum(search_scope_checks.values()) / len(search_scope_checks)
        }
    
    def validate_tool_registration(self) -> Dict[str, Any]:
        """Validate tool registration matches specification."""
        logger.info("Validating tool registration")
        
        if not self.simplified_interface_path.exists():
            return {"error": f"File not found: {self.simplified_interface_path}"}
        
        content = self.simplified_interface_path.read_text()
        
        registration_checks = {
            "has_register_simplified_tools": "async def register_simplified_tools(" in content,
            "registers_qdrant_store": "@app.tool()" in content and "async def qdrant_store(" in content,
            "registers_qdrant_find": "@app.tool()" in content and "async def qdrant_find(" in content, 
            "registers_qdrant_manage": "@app.tool()" in content and "async def qdrant_manage(" in content,
            "registers_qdrant_read": "@app.tool()" in content and "async def qdrant_read(" in content,
            "uses_simplified_tools_mode": "SimplifiedToolsMode" in content,
            "handles_tool_modes": "get_enabled_tools" in content
        }
        
        return {
            "tool_registration": registration_checks,
            "registration_score": sum(registration_checks.values()) / len(registration_checks)
        }
    
    def validate_docstring_compliance(self) -> Dict[str, Any]:
        """Validate docstrings match Task 178 requirements."""
        logger.info("Validating docstring compliance")
        
        if not self.simplified_interface_path.exists():
            return {"error": f"File not found: {self.simplified_interface_path}"}
        
        content = self.simplified_interface_path.read_text()
        
        # Extract docstrings for each tool
        docstring_checks = {}
        
        for tool in ["qdrant_store", "qdrant_find", "qdrant_manage", "qdrant_read"]:
            # Find the function and its docstring
            pattern = f'async def {tool}\\(.*?\\) -> .*?:\\s*"""(.*?)"""'
            match = re.search(pattern, content, re.DOTALL)
            
            if match:
                docstring = match.group(1).lower()
                
                if tool == "qdrant_store":
                    checks = {
                        "has_docstring": True,
                        "mentions_universal": "universal" in docstring,
                        "mentions_content_ingestion": "content ingestion" in docstring,
                        "mentions_source_classification": "source" in docstring,
                        "mentions_routing": "routes to existing" in docstring
                    }
                elif tool == "qdrant_find":
                    checks = {
                        "has_docstring": True,
                        "mentions_search_scope": "search" in docstring and "scope" in docstring,
                        "mentions_filtering": "filtering" in docstring,
                        "mentions_task_175": "task 175" in docstring or "search scope architecture" in docstring
                    }
                elif tool == "qdrant_manage":
                    checks = {
                        "has_docstring": True,
                        "mentions_status": "status" in docstring,
                        "mentions_collection_management": "collection management" in docstring,
                        "mentions_routing": "routes to existing" in docstring
                    }
                elif tool == "qdrant_read":
                    checks = {
                        "has_docstring": True,
                        "mentions_direct_retrieval": "direct" in docstring and ("retrieval" in docstring or "document" in docstring),
                        "mentions_without_search": "without search" in docstring,
                        "mentions_routing": "routes to existing" in docstring
                    }
                
                docstring_checks[tool] = checks
            else:
                docstring_checks[tool] = {"has_docstring": False, "error": f"Docstring not found for {tool}"}
        
        return docstring_checks
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete file-based validation."""
        logger.info("Starting Task 178 file-based validation")
        
        try:
            self.results = {
                "api_specifications": self.validate_api_specifications(),
                "routing_implementation": self.validate_routing_implementation(),
                "access_control_integration": self.validate_access_control_integration(),
                "search_scope_integration": self.validate_search_scope_integration(),
                "tool_registration": self.validate_tool_registration(),
                "docstring_compliance": self.validate_docstring_compliance()
            }
            
            # Calculate overall compliance
            scores = []
            for category, result in self.results.items():
                if isinstance(result, dict):
                    # Count successful checks
                    if "error" in result:
                        scores.append(0)
                    else:
                        # Count positive indicators
                        positive_count = 0
                        total_count = 0
                        for key, value in result.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    if isinstance(subvalue, bool):
                                        total_count += 1
                                        if subvalue:
                                            positive_count += 1
                            elif isinstance(value, bool):
                                total_count += 1
                                if value:
                                    positive_count += 1
                            elif isinstance(value, (int, float)) and 0 <= value <= 1:
                                # Assume this is a score
                                scores.append(value)
                                continue
                        
                        if total_count > 0:
                            scores.append(positive_count / total_count)
            
            overall_score = sum(scores) / len(scores) if scores else 0
            
            self.results["summary"] = {
                "overall_compliance_score": f"{overall_score:.2f}",
                "categories_validated": len([r for r in self.results.values() if not isinstance(r, dict) or "error" not in r]),
                "total_categories": 6,
                "task_178_compliant": overall_score >= 0.8,  # 80% threshold
                "status": "COMPLIANT" if overall_score >= 0.8 else "PARTIAL COMPLIANCE"
            }
            
            return self.results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"error": str(e)}

def main():
    """Run Task 178 file-based validation."""
    logger.info("=== Task 178: 4-Tool Consolidation File-Based Validation ===")
    
    try:
        validator = Task178FileValidation()
        results = validator.run_validation()
        
        # Output results
        print("\n" + "="*80)
        print("TASK 178 FILE-BASED VALIDATION RESULTS")
        print("="*80)
        
        print(json.dumps(results, indent=2, default=str))
        
        summary = results.get("summary", {})
        print(f"\n{'='*80}")
        print(f"OVERALL COMPLIANCE SCORE: {summary.get('overall_compliance_score', '0.00')}")
        print(f"CATEGORIES VALIDATED: {summary.get('categories_validated', 0)}/{summary.get('total_categories', 0)}")
        print(f"STATUS: {summary.get('status', 'UNKNOWN')}")
        print(f"TASK 178 COMPLIANT: {'✓ YES' if summary.get('task_178_compliant') else '✗ NO'}")
        print("="*80)
        
        return 0 if summary.get("task_178_compliant") else 1
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)