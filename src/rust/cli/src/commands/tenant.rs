//! Tenant name resolution utilities — thin delegating shims.
//!
//! The canonical implementation lives in [`crate::data::tenants`]; these
//! re-exports keep the long-standing call sites (rules, scratchpad, watch,
//! queue) stable.

use std::collections::HashMap;

/// Build a tenant_id -> project name mapping from watch_folders.
///
/// Extracts the last path component as the project name. Returns an
/// empty map if the database is unavailable.
pub fn load_project_names() -> HashMap<String, String> {
    crate::data::tenants::name_map()
}

/// Resolve a tenant_id to a project name, falling back to the ID itself.
pub fn resolve_tenant_name(tenant_id: &str, names: &HashMap<String, String>) -> String {
    crate::data::tenants::display_name(names, tenant_id)
}
