# Enterprise Role-Based Access Control (RBAC)

**Version**: 1.0  
**Target Audience**: Enterprise Security Teams, DevOps Engineers, System Administrators  
**Prerequisites**: workspace-qdrant-mcp v1.0+, Enterprise Authentication Framework

## Overview

This document provides comprehensive guidance for implementing Role-Based Access Control (RBAC) in workspace-qdrant-mcp. It covers permission models, collection-level access control, user workspace separation, and enterprise authorization patterns.

## Table of Contents

- [RBAC Architecture](#rbac-architecture)
- [Permission Model](#permission-model)
- [Role Definitions](#role-definitions)
- [Collection-Level Access Control](#collection-level-access-control)
- [User Workspace Separation](#user-workspace-separation)
- [Dynamic Permission Management](#dynamic-permission-management)
- [Authorization Middleware](#authorization-middleware)
- [Policy Engines](#policy-engines)
- [Audit and Compliance](#audit-and-compliance)
- [Implementation Examples](#implementation-examples)

## RBAC Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     Users       │    │      Roles       │    │   Permissions   │
│                 │────│                  │────│                 │
│ - User ID       │    │ - Role Name      │    │ - Resource      │
│ - Email         │    │ - Description    │    │ - Action        │
│ - Department    │    │ - Permissions    │    │ - Scope         │
│ - Groups        │    │ - Hierarchy      │    │ - Conditions    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                ┌──────────────────────────────┐
                │     Authorization Engine     │
                │                              │
                │ - Policy Evaluation          │
                │ - Context-Aware Decisions    │
                │ - Hierarchical Role Support  │
                │ - Dynamic Permission Updates │
                └──────────────────────────────┘
```

### Database Schema Design

```sql
-- Users table (can integrate with existing user management)
CREATE TABLE rbac_users (
    user_id VARCHAR(64) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    department VARCHAR(100),
    organization_unit VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Roles table
CREATE TABLE rbac_roles (
    role_id VARCHAR(64) PRIMARY KEY,
    role_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    role_type VARCHAR(50) DEFAULT 'custom', -- system, custom, inherited
    parent_role_id VARCHAR(64) REFERENCES rbac_roles(role_id),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Permissions table
CREATE TABLE rbac_permissions (
    permission_id VARCHAR(64) PRIMARY KEY,
    resource_type VARCHAR(100) NOT NULL, -- collection, workspace, system
    resource_name VARCHAR(255), -- specific collection name or *
    action VARCHAR(50) NOT NULL, -- read, write, delete, admin
    scope VARCHAR(100) DEFAULT 'specific', -- global, project, specific
    conditions JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Role-Permission assignments
CREATE TABLE rbac_role_permissions (
    role_id VARCHAR(64) REFERENCES rbac_roles(role_id) ON DELETE CASCADE,
    permission_id VARCHAR(64) REFERENCES rbac_permissions(permission_id) ON DELETE CASCADE,
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    granted_by VARCHAR(64) REFERENCES rbac_users(user_id),
    PRIMARY KEY (role_id, permission_id)
);

-- User-Role assignments
CREATE TABLE rbac_user_roles (
    user_id VARCHAR(64) REFERENCES rbac_users(user_id) ON DELETE CASCADE,
    role_id VARCHAR(64) REFERENCES rbac_roles(role_id) ON DELETE CASCADE,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    assigned_by VARCHAR(64) REFERENCES rbac_users(user_id),
    expires_at TIMESTAMP WITH TIME ZONE, -- For temporary access
    is_active BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (user_id, role_id)
);

-- Collection access control
CREATE TABLE rbac_collection_access (
    collection_name VARCHAR(255) NOT NULL,
    user_id VARCHAR(64) REFERENCES rbac_users(user_id),
    role_id VARCHAR(64) REFERENCES rbac_roles(role_id),
    access_level VARCHAR(50) NOT NULL, -- read, write, admin, deny
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    CHECK (user_id IS NOT NULL OR role_id IS NOT NULL)
);

-- Indexes for performance
CREATE INDEX idx_rbac_user_roles_user_id ON rbac_user_roles(user_id);
CREATE INDEX idx_rbac_user_roles_role_id ON rbac_user_roles(role_id);
CREATE INDEX idx_rbac_collection_access_collection ON rbac_collection_access(collection_name);
CREATE INDEX idx_rbac_collection_access_user ON rbac_collection_access(user_id);
```

## Permission Model

### Hierarchical Permission Structure

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from datetime import datetime

class ResourceType(Enum):
    COLLECTION = "collection"
    WORKSPACE = "workspace" 
    SYSTEM = "system"
    SCRATCHBOOK = "scratchbook"

class Action(Enum):
    READ = "read"
    WRITE = "write" 
    DELETE = "delete"
    ADMIN = "admin"
    CREATE = "create"
    SEARCH = "search"
    INGEST = "ingest"

class Scope(Enum):
    GLOBAL = "global"         # System-wide access
    PROJECT = "project"       # Project-specific access
    COLLECTION = "collection" # Single collection access
    USER = "user"            # User's own resources only

@dataclass
class Permission:
    permission_id: str
    resource_type: ResourceType
    resource_name: Optional[str]  # None means all resources of type
    action: Action
    scope: Scope
    conditions: Dict = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}
    
    def matches(self, resource_type: ResourceType, resource_name: str, action: Action) -> bool:
        """Check if this permission matches the requested access"""
        # Check resource type
        if self.resource_type != resource_type:
            return False
        
        # Check action (with hierarchy: admin > write > read)
        action_hierarchy = {
            Action.READ: [Action.READ],
            Action.WRITE: [Action.READ, Action.WRITE],
            Action.DELETE: [Action.READ, Action.WRITE, Action.DELETE],
            Action.ADMIN: [Action.READ, Action.WRITE, Action.DELETE, Action.ADMIN, Action.CREATE, Action.INGEST]
        }
        
        if action not in action_hierarchy.get(self.action, []):
            return False
        
        # Check resource name (wildcard support)
        if self.resource_name is None or self.resource_name == "*":
            return True
        
        if self.resource_name == resource_name:
            return True
        
        # Pattern matching for collections (e.g., "project-*")
        if "*" in self.resource_name:
            import fnmatch
            return fnmatch.fnmatch(resource_name, self.resource_name)
        
        return False

@dataclass 
class Role:
    role_id: str
    role_name: str
    description: str
    permissions: List[Permission]
    parent_role: Optional['Role'] = None
    role_type: str = "custom"
    is_active: bool = True
    
    def get_all_permissions(self) -> Set[Permission]:
        """Get all permissions including inherited from parent roles"""
        all_permissions = set(self.permissions)
        
        if self.parent_role:
            all_permissions.update(self.parent_role.get_all_permissions())
        
        return all_permissions
    
    def has_permission(self, resource_type: ResourceType, resource_name: str, action: Action) -> bool:
        """Check if this role has the specified permission"""
        for permission in self.get_all_permissions():
            if permission.matches(resource_type, resource_name, action):
                return True
        return False
```

### Permission Evaluation Engine

```python
from typing import List, Set, Dict, Any
import logging

class PermissionEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_user_access(self, user_id: str, resource_type: ResourceType, 
                           resource_name: str, action: Action, 
                           context: Dict[str, Any] = None) -> bool:
        """Evaluate if user has access to perform action on resource"""
        try:
            # Get user's roles
            user_roles = self.get_user_roles(user_id)
            
            # Check direct permissions
            if self.check_direct_permissions(user_id, resource_type, resource_name, action):
                self.logger.info(f"Access granted via direct permission: {user_id}")
                return True
            
            # Check role-based permissions
            for role in user_roles:
                if role.has_permission(resource_type, resource_name, action):
                    # Check contextual conditions
                    if self.evaluate_conditions(role, resource_name, context or {}):
                        self.logger.info(f"Access granted via role '{role.role_name}': {user_id}")
                        return True
            
            # Check collection-specific access
            if self.check_collection_access(user_id, resource_name, action):
                self.logger.info(f"Access granted via collection ACL: {user_id}")
                return True
            
            self.logger.warning(f"Access denied: {user_id} -> {resource_type}:{resource_name}:{action}")
            return False
            
        except Exception as e:
            self.logger.error(f"Permission evaluation error: {e}")
            return False
    
    def evaluate_conditions(self, role: Role, resource_name: str, context: Dict[str, Any]) -> bool:
        """Evaluate contextual conditions for permission"""
        for permission in role.permissions:
            if not permission.conditions:
                continue
            
            # Time-based conditions
            if 'time_restrictions' in permission.conditions:
                if not self.check_time_restrictions(permission.conditions['time_restrictions']):
                    return False
            
            # IP-based conditions
            if 'allowed_ips' in permission.conditions:
                client_ip = context.get('client_ip')
                if not self.check_ip_restrictions(client_ip, permission.conditions['allowed_ips']):
                    return False
            
            # Department-based conditions
            if 'allowed_departments' in permission.conditions:
                user_dept = context.get('user_department')
                if user_dept not in permission.conditions['allowed_departments']:
                    return False
            
            # Resource pattern conditions
            if 'resource_patterns' in permission.conditions:
                if not self.check_resource_patterns(resource_name, permission.conditions['resource_patterns']):
                    return False
        
        return True
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user (direct + role-based)"""
        all_permissions = set()
        
        # Direct permissions
        direct_perms = self.get_direct_user_permissions(user_id)
        all_permissions.update(direct_perms)
        
        # Role-based permissions  
        user_roles = self.get_user_roles(user_id)
        for role in user_roles:
            all_permissions.update(role.get_all_permissions())
        
        return all_permissions
```

## Role Definitions

### System Roles

```python
# Predefined system roles for workspace-qdrant-mcp

SYSTEM_ROLES = {
    "system_admin": Role(
        role_id="system_admin",
        role_name="System Administrator", 
        description="Full system access including user management and system configuration",
        permissions=[
            Permission("sys_admin_all", ResourceType.SYSTEM, "*", Action.ADMIN, Scope.GLOBAL),
            Permission("ws_admin_all", ResourceType.WORKSPACE, "*", Action.ADMIN, Scope.GLOBAL),
            Permission("col_admin_all", ResourceType.COLLECTION, "*", Action.ADMIN, Scope.GLOBAL)
        ],
        role_type="system"
    ),
    
    "workspace_admin": Role(
        role_id="workspace_admin",
        role_name="Workspace Administrator",
        description="Administrative access to workspace and collections",
        permissions=[
            Permission("ws_admin", ResourceType.WORKSPACE, "*", Action.ADMIN, Scope.PROJECT),
            Permission("col_admin", ResourceType.COLLECTION, "*", Action.ADMIN, Scope.PROJECT),
            Permission("scratch_admin", ResourceType.SCRATCHBOOK, "*", Action.ADMIN, Scope.PROJECT)
        ],
        role_type="system"
    ),
    
    "data_scientist": Role(
        role_id="data_scientist", 
        role_name="Data Scientist",
        description="Read/write access to collections with search and analysis capabilities",
        permissions=[
            Permission("col_read", ResourceType.COLLECTION, "*", Action.READ, Scope.PROJECT),
            Permission("col_search", ResourceType.COLLECTION, "*", Action.SEARCH, Scope.PROJECT),
            Permission("col_write", ResourceType.COLLECTION, "*", Action.WRITE, Scope.PROJECT),
            Permission("scratch_write", ResourceType.SCRATCHBOOK, "*", Action.WRITE, Scope.USER)
        ],
        role_type="system"
    ),
    
    "analyst": Role(
        role_id="analyst",
        role_name="Data Analyst", 
        description="Read-only access with search capabilities",
        permissions=[
            Permission("col_read", ResourceType.COLLECTION, "*", Action.READ, Scope.PROJECT),
            Permission("col_search", ResourceType.COLLECTION, "*", Action.SEARCH, Scope.PROJECT),
            Permission("scratch_read", ResourceType.SCRATCHBOOK, "*", Action.READ, Scope.USER)
        ],
        role_type="system"
    ),
    
    "viewer": Role(
        role_id="viewer",
        role_name="Viewer",
        description="Read-only access to assigned collections",
        permissions=[
            Permission("col_read", ResourceType.COLLECTION, "*", Action.READ, Scope.COLLECTION),
        ],
        role_type="system"
    ),
    
    "guest": Role(
        role_id="guest", 
        role_name="Guest User",
        description="Limited read access with time restrictions",
        permissions=[
            Permission("col_read_limited", ResourceType.COLLECTION, "public-*", Action.READ, Scope.COLLECTION,
                      conditions={'time_restrictions': {'max_session_hours': 2}})
        ],
        role_type="system"
    )
}

# Role hierarchy (inheritance)
SYSTEM_ROLES["data_scientist"].parent_role = SYSTEM_ROLES["analyst"]
SYSTEM_ROLES["analyst"].parent_role = SYSTEM_ROLES["viewer"]
```

### Custom Role Creation

```python
class RoleManager:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def create_custom_role(self, role_name: str, description: str, 
                          permissions: List[Dict], parent_role_id: str = None) -> Role:
        """Create a custom role with specified permissions"""
        role_id = f"custom_{role_name.lower().replace(' ', '_')}"
        
        # Convert permission dictionaries to Permission objects
        perm_objects = []
        for perm_data in permissions:
            perm_obj = Permission(
                permission_id=f"{role_id}_{len(perm_objects)}",
                resource_type=ResourceType(perm_data['resource_type']),
                resource_name=perm_data.get('resource_name'),
                action=Action(perm_data['action']),
                scope=Scope(perm_data['scope']),
                conditions=perm_data.get('conditions', {})
            )
            perm_objects.append(perm_obj)
        
        # Create role
        role = Role(
            role_id=role_id,
            role_name=role_name,
            description=description,
            permissions=perm_objects,
            parent_role=self.get_role(parent_role_id) if parent_role_id else None,
            role_type="custom"
        )
        
        # Persist to database
        self.save_role(role)
        
        return role
    
    def create_department_role(self, department: str, access_level: str) -> Role:
        """Create department-specific role"""
        role_name = f"{department} {access_level.title()}"
        role_id = f"dept_{department.lower()}_{access_level}"
        
        # Define permissions based on access level
        if access_level == "admin":
            permissions = [
                Permission(f"{role_id}_admin", ResourceType.COLLECTION, f"{department}-*", Action.ADMIN, Scope.PROJECT)
            ]
        elif access_level == "editor":
            permissions = [
                Permission(f"{role_id}_write", ResourceType.COLLECTION, f"{department}-*", Action.WRITE, Scope.PROJECT),
                Permission(f"{role_id}_read", ResourceType.COLLECTION, f"{department}-*", Action.READ, Scope.PROJECT)
            ]
        else:  # viewer
            permissions = [
                Permission(f"{role_id}_read", ResourceType.COLLECTION, f"{department}-*", Action.READ, Scope.PROJECT)
            ]
        
        role = Role(
            role_id=role_id,
            role_name=role_name,
            description=f"{access_level.title()} access for {department} department collections",
            permissions=permissions,
            role_type="department"
        )
        
        self.save_role(role)
        return role
```

## Collection-Level Access Control

### Collection Access Matrix

```python
from enum import Enum
import fnmatch

class AccessLevel(Enum):
    DENY = "deny"
    READ = "read" 
    WRITE = "write"
    ADMIN = "admin"

class CollectionACL:
    def __init__(self):
        self.access_rules = []
    
    def add_rule(self, collection_pattern: str, user_or_role: str, 
                 access_level: AccessLevel, is_role: bool = False):
        """Add access control rule"""
        rule = {
            'collection_pattern': collection_pattern,
            'principal': user_or_role,
            'is_role': is_role,
            'access_level': access_level,
            'created_at': datetime.utcnow()
        }
        self.access_rules.append(rule)
    
    def check_collection_access(self, collection_name: str, user_id: str, 
                               user_roles: List[str], requested_action: Action) -> bool:
        """Check if user has access to collection"""
        effective_access = AccessLevel.DENY
        
        # Check direct user rules
        for rule in self.access_rules:
            if not rule['is_role'] and rule['principal'] == user_id:
                if fnmatch.fnmatch(collection_name, rule['collection_pattern']):
                    effective_access = max(effective_access, rule['access_level'], 
                                         key=lambda x: list(AccessLevel).index(x))
        
        # Check role-based rules
        for rule in self.access_rules:
            if rule['is_role'] and rule['principal'] in user_roles:
                if fnmatch.fnmatch(collection_name, rule['collection_pattern']):
                    effective_access = max(effective_access, rule['access_level'],
                                         key=lambda x: list(AccessLevel).index(x))
        
        # Map access level to actions
        allowed_actions = {
            AccessLevel.DENY: [],
            AccessLevel.READ: [Action.READ, Action.SEARCH],
            AccessLevel.WRITE: [Action.READ, Action.SEARCH, Action.WRITE, Action.INGEST],
            AccessLevel.ADMIN: [Action.READ, Action.SEARCH, Action.WRITE, Action.INGEST, Action.DELETE, Action.ADMIN]
        }
        
        return requested_action in allowed_actions.get(effective_access, [])

# Example usage
collection_acl = CollectionACL()

# Department-based access
collection_acl.add_rule("finance-*", "finance_team", AccessLevel.WRITE, is_role=True)
collection_acl.add_rule("hr-*", "hr_team", AccessLevel.ADMIN, is_role=True)
collection_acl.add_rule("public-*", "all_employees", AccessLevel.READ, is_role=True)

# Individual user access
collection_acl.add_rule("sensitive-legal-*", "legal_counsel_user", AccessLevel.ADMIN, is_role=False)
collection_acl.add_rule("*-scratchbook", "owner_user_id", AccessLevel.ADMIN, is_role=False)
```

### Collection Isolation Strategies

```python
class CollectionIsolationManager:
    def __init__(self, isolation_strategy: str = "namespace"):
        self.strategy = isolation_strategy
    
    def get_collection_name(self, base_name: str, tenant_id: str, 
                           project_id: str = None) -> str:
        """Generate tenant-isolated collection name"""
        if self.strategy == "namespace":
            # Strategy 1: Namespace-based isolation
            return f"{tenant_id}_{base_name}"
        
        elif self.strategy == "hierarchical":
            # Strategy 2: Hierarchical isolation  
            if project_id:
                return f"{tenant_id}_{project_id}_{base_name}"
            return f"{tenant_id}_global_{base_name}"
        
        elif self.strategy == "database":
            # Strategy 3: Database-level isolation (separate Qdrant instances)
            return base_name  # Same name, different database
        
        else:
            raise ValueError(f"Unknown isolation strategy: {self.strategy}")
    
    def get_allowed_collections(self, user_id: str, tenant_id: str) -> List[str]:
        """Get list of collections user can access"""
        # Get user's roles and permissions
        user_roles = self.get_user_roles(user_id)
        allowed_patterns = []
        
        for role in user_roles:
            for permission in role.get_all_permissions():
                if permission.resource_type == ResourceType.COLLECTION:
                    # Apply tenant isolation to permission patterns
                    if permission.resource_name:
                        isolated_pattern = self.get_collection_name(
                            permission.resource_name, tenant_id
                        )
                        allowed_patterns.append(isolated_pattern)
        
        # Get actual collections and filter by patterns
        all_collections = self.list_tenant_collections(tenant_id)
        allowed_collections = []
        
        for collection in all_collections:
            for pattern in allowed_patterns:
                if fnmatch.fnmatch(collection, pattern):
                    allowed_collections.append(collection)
                    break
        
        return list(set(allowed_collections))
```

## User Workspace Separation

### Workspace Boundaries

```python
class WorkspaceManager:
    def __init__(self, db_connection, qdrant_client):
        self.db = db_connection
        self.qdrant = qdrant_client
    
    def create_user_workspace(self, user_id: str, workspace_config: Dict) -> str:
        """Create isolated workspace for user"""
        workspace_id = f"ws_{user_id}_{int(datetime.now().timestamp())}"
        
        workspace = {
            'workspace_id': workspace_id,
            'owner_id': user_id,
            'name': workspace_config['name'],
            'description': workspace_config.get('description', ''),
            'isolation_level': workspace_config.get('isolation_level', 'standard'),
            'resource_limits': workspace_config.get('resource_limits', {}),
            'created_at': datetime.utcnow()
        }
        
        # Create workspace in database
        self.save_workspace(workspace)
        
        # Create default collections for workspace
        self.create_default_collections(workspace_id, user_id)
        
        # Set up resource quotas
        self.apply_resource_limits(workspace_id, workspace['resource_limits'])
        
        return workspace_id
    
    def create_default_collections(self, workspace_id: str, owner_id: str):
        """Create default collections for new workspace"""
        default_collections = [
            f"{workspace_id}-documents",
            f"{workspace_id}-scratchbook", 
            f"{workspace_id}-references"
        ]
        
        for collection_name in default_collections:
            # Create Qdrant collection
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            
            # Set collection ACL (owner has admin access)
            self.set_collection_access(collection_name, owner_id, AccessLevel.ADMIN)
    
    def get_user_workspaces(self, user_id: str) -> List[Dict]:
        """Get all workspaces user has access to"""
        query = """
        SELECT w.* FROM workspaces w
        LEFT JOIN workspace_access wa ON w.workspace_id = wa.workspace_id  
        WHERE w.owner_id = ? OR wa.user_id = ?
        """
        return self.db.execute(query, [user_id, user_id]).fetchall()
    
    def share_workspace(self, workspace_id: str, target_user_id: str, 
                       access_level: AccessLevel, granted_by: str):
        """Share workspace with another user"""
        # Check if granter has admin access to workspace
        if not self.check_workspace_access(granted_by, workspace_id, Action.ADMIN):
            raise PermissionError("Insufficient privileges to share workspace")
        
        # Create workspace access record
        access_record = {
            'workspace_id': workspace_id,
            'user_id': target_user_id,
            'access_level': access_level.value,
            'granted_by': granted_by,
            'granted_at': datetime.utcnow()
        }
        
        self.save_workspace_access(access_record)
        
        # Update collection access for workspace collections
        workspace_collections = self.get_workspace_collections(workspace_id)
        for collection_name in workspace_collections:
            self.set_collection_access(collection_name, target_user_id, access_level)
```

### Resource Quotas and Limits

```python
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ResourceQuota:
    max_collections: int = 10
    max_documents_per_collection: int = 100000
    max_storage_mb: int = 1000
    max_queries_per_hour: int = 10000
    max_concurrent_operations: int = 10

class ResourceManager:
    def __init__(self, db_connection, redis_client):
        self.db = db_connection
        self.redis = redis_client
    
    def apply_quota(self, workspace_id: str, quota: ResourceQuota):
        """Apply resource quotas to workspace"""
        quota_data = {
            'workspace_id': workspace_id,
            'max_collections': quota.max_collections,
            'max_documents_per_collection': quota.max_documents_per_collection,
            'max_storage_mb': quota.max_storage_mb,
            'max_queries_per_hour': quota.max_queries_per_hour,
            'max_concurrent_operations': quota.max_concurrent_operations,
            'created_at': datetime.utcnow()
        }
        
        self.save_quota(quota_data)
    
    def check_quota_compliance(self, workspace_id: str, operation: str, 
                              amount: int = 1) -> bool:
        """Check if operation would exceed quota"""
        quota = self.get_workspace_quota(workspace_id)
        if not quota:
            return True  # No quota set
        
        current_usage = self.get_current_usage(workspace_id)
        
        if operation == "create_collection":
            return current_usage['collection_count'] + amount <= quota['max_collections']
        
        elif operation == "add_documents":
            # Check per-collection limit
            return amount <= quota['max_documents_per_collection']
        
        elif operation == "query":
            # Check hourly query limit
            current_queries = self.get_hourly_query_count(workspace_id)
            return current_queries + amount <= quota['max_queries_per_hour']
        
        return True
    
    def get_usage_metrics(self, workspace_id: str) -> Dict:
        """Get detailed usage metrics for workspace"""
        collections = self.get_workspace_collections(workspace_id)
        
        total_documents = 0
        total_storage_mb = 0
        
        for collection_name in collections:
            collection_info = self.qdrant.get_collection(collection_name)
            total_documents += collection_info.points_count
            total_storage_mb += collection_info.disk_usage_bytes / (1024 * 1024)
        
        return {
            'workspace_id': workspace_id,
            'collection_count': len(collections),
            'total_documents': total_documents,
            'storage_used_mb': round(total_storage_mb, 2),
            'queries_last_hour': self.get_hourly_query_count(workspace_id),
            'last_updated': datetime.utcnow().isoformat()
        }
```

## Dynamic Permission Management

### Permission Updates and Propagation

```python
class DynamicPermissionManager:
    def __init__(self, db_connection, cache_client):
        self.db = db_connection
        self.cache = cache_client
        
    def update_user_permissions(self, user_id: str, permission_changes: List[Dict]):
        """Update user permissions with change tracking"""
        for change in permission_changes:
            if change['action'] == 'grant':
                self.grant_permission(
                    user_id=user_id,
                    resource_type=change['resource_type'],
                    resource_name=change['resource_name'],
                    action=change['permission_action'],
                    granted_by=change['granted_by']
                )
            elif change['action'] == 'revoke':
                self.revoke_permission(
                    user_id=user_id,
                    resource_type=change['resource_type'], 
                    resource_name=change['resource_name'],
                    action=change['permission_action'],
                    revoked_by=change['revoked_by']
                )
        
        # Invalidate user's permission cache
        self.invalidate_user_cache(user_id)
        
        # Log permission changes
        self.log_permission_changes(user_id, permission_changes)
    
    def grant_temporary_access(self, user_id: str, collection_name: str, 
                              access_level: AccessLevel, duration_hours: int,
                              granted_by: str) -> str:
        """Grant temporary access that expires automatically"""
        temp_access_id = f"temp_{user_id}_{int(datetime.now().timestamp())}"
        expires_at = datetime.utcnow() + timedelta(hours=duration_hours)
        
        temp_access = {
            'temp_access_id': temp_access_id,
            'user_id': user_id,
            'collection_name': collection_name,
            'access_level': access_level.value,
            'granted_by': granted_by,
            'granted_at': datetime.utcnow(),
            'expires_at': expires_at
        }
        
        self.save_temporary_access(temp_access)
        
        # Schedule cleanup task
        self.schedule_access_cleanup(temp_access_id, expires_at)
        
        return temp_access_id
    
    def bulk_permission_update(self, updates: List[Dict]):
        """Process multiple permission updates efficiently"""
        affected_users = set()
        
        try:
            # Start database transaction
            self.db.begin()
            
            for update in updates:
                user_id = update['user_id']
                affected_users.add(user_id)
                
                if update['operation'] == 'role_assignment':
                    self.assign_role(user_id, update['role_id'], update['assigned_by'])
                elif update['operation'] == 'collection_access':
                    self.set_collection_access(
                        update['collection_name'], 
                        user_id, 
                        AccessLevel(update['access_level'])
                    )
            
            self.db.commit()
            
            # Invalidate caches for all affected users
            for user_id in affected_users:
                self.invalidate_user_cache(user_id)
                
        except Exception as e:
            self.db.rollback()
            raise e
```

## Authorization Middleware

### FastAPI Authorization Middleware

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time

class RBACMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, permission_evaluator: PermissionEvaluator):
        super().__init__(app)
        self.permission_evaluator = permission_evaluator
        self.protected_endpoints = {
            '/api/search': (ResourceType.COLLECTION, Action.SEARCH),
            '/api/store': (ResourceType.COLLECTION, Action.WRITE),
            '/api/delete': (ResourceType.COLLECTION, Action.DELETE),
            '/api/collections': (ResourceType.WORKSPACE, Action.READ),
            '/api/admin': (ResourceType.SYSTEM, Action.ADMIN)
        }
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Skip authorization for public endpoints
        if request.url.path in ['/health', '/metrics', '/docs']:
            return await call_next(request)
        
        # Extract user information from request
        user_info = await self.extract_user_info(request)
        if not user_info:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Check endpoint permissions
        if request.url.path in self.protected_endpoints:
            resource_type, required_action = self.protected_endpoints[request.url.path]
            resource_name = self.extract_resource_name(request)
            
            # Evaluate permission
            context = {
                'client_ip': request.client.host,
                'user_agent': request.headers.get('user-agent'),
                'user_department': user_info.get('department'),
                'request_time': datetime.utcnow()
            }
            
            has_permission = self.permission_evaluator.evaluate_user_access(
                user_id=user_info['user_id'],
                resource_type=resource_type,
                resource_name=resource_name,
                action=required_action,
                context=context
            )
            
            if not has_permission:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Add user context to request
        request.state.user = user_info
        
        # Process request
        response = await call_next(request)
        
        # Log access
        duration = time.time() - start_time
        self.log_access(user_info['user_id'], request.url.path, response.status_code, duration)
        
        return response
    
    def extract_resource_name(self, request: Request) -> str:
        """Extract resource name from request"""
        # Check query parameters
        if 'collection' in request.query_params:
            return request.query_params['collection']
        
        # Check path parameters
        path_parts = request.url.path.split('/')
        if 'collections' in path_parts:
            idx = path_parts.index('collections')
            if idx + 1 < len(path_parts):
                return path_parts[idx + 1]
        
        # Default to wildcard for workspace-level operations
        return "*"
```

### Collection-Level Authorization Decorator

```python
from functools import wraps
from typing import Callable

def require_collection_permission(action: Action):
    """Decorator to enforce collection-level permissions"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract collection name from arguments
            collection_name = kwargs.get('collection_name') or kwargs.get('collection')
            if not collection_name:
                raise ValueError("Collection name required for permission check")
            
            # Get current user from request context
            request = kwargs.get('request')
            if not request or not hasattr(request.state, 'user'):
                raise HTTPException(status_code=401, detail="User context not found")
            
            user_info = request.state.user
            
            # Check permission
            permission_evaluator = get_permission_evaluator()  # Get from DI container
            has_permission = permission_evaluator.evaluate_user_access(
                user_id=user_info['user_id'],
                resource_type=ResourceType.COLLECTION,
                resource_name=collection_name,
                action=action
            )
            
            if not has_permission:
                raise HTTPException(
                    status_code=403, 
                    detail=f"Permission denied: {action.value} on collection '{collection_name}'"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Usage examples
@app.post("/api/collections/{collection_name}/documents")
@require_collection_permission(Action.WRITE)
async def add_documents(collection_name: str, documents: List[Dict], 
                       request: Request):
    """Add documents to collection"""
    return await document_service.add_documents(collection_name, documents)

@app.get("/api/collections/{collection_name}/search")
@require_collection_permission(Action.SEARCH)
async def search_collection(collection_name: str, query: str, 
                           request: Request):
    """Search within specific collection"""
    return await search_service.search(collection_name, query)
```

This completes the comprehensive RBAC documentation with enterprise-grade permission models, role hierarchies, and authorization middleware implementations.