//! Tenant-id ⇄ project-name resolution for CLI args and outputs.
//!
//! CLI UX rule: outputs display a project's NAME (the watch-folder path
//! basename) rather than its raw tenant id unless both are explicitly part
//! of the output, and name arguments accept partial input resolved by
//! prefix/substring match — ambiguity is an error listing candidates, never
//! a silent pick.
//!
//! All lookups read `watch_folders` via the shared read-only connection.

use std::collections::HashMap;

use anyhow::{bail, Result};
use rusqlite::Connection;

use super::db::connect_readonly;

/// One registered watch folder, as relevant for name resolution.
pub struct TenantEntry {
    pub tenant_id: String,
    /// Path basename — the project's display name.
    pub name: String,
    pub path: String,
}

impl TenantEntry {
    /// Render for candidate listings: `name (tenant_id) — path`.
    fn describe(&self) -> String {
        format!("{} ({}) — {}", self.name, self.tenant_id, self.path)
    }
}

/// Load all registered tenants (projects AND libraries) from `watch_folders`.
pub fn load_tenants(conn: &Connection) -> Result<Vec<TenantEntry>> {
    let mut stmt = conn.prepare("SELECT tenant_id, path FROM watch_folders ORDER BY path")?;
    let rows = stmt.query_map([], |row| {
        let tenant_id: String = row.get(0)?;
        let path: String = row.get(1)?;
        Ok((tenant_id, path))
    })?;

    let mut out = Vec::new();
    for row in rows {
        let (tenant_id, path) = row?;
        let name = path
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or(&path)
            .to_string();
        out.push(TenantEntry {
            tenant_id,
            name,
            path,
        });
    }
    Ok(out)
}

/// Resolve a user-supplied tenant argument to a tenant id.
///
/// `input` may be a full tenant id, a project name (path basename), or a
/// partial form of either. Resolution order (first unique tier wins):
///
/// 1. exact tenant id
/// 2. exact name
/// 3. unique tenant-id prefix
/// 4. unique name prefix
/// 5. unique name substring
///
/// An ambiguous tier is an error listing the candidates; nothing is ever
/// picked silently. Unknown input lists the registered projects.
pub fn resolve_tenant(input: &str) -> Result<String> {
    let conn = connect_readonly()?;
    resolve_tenant_in(&conn, input)
}

/// [`resolve_tenant`] against an existing connection (testable form).
pub fn resolve_tenant_in(conn: &Connection, input: &str) -> Result<String> {
    let entries = load_tenants(conn)?;
    resolve_tenant_entries(&entries, input)
}

/// Pure resolution over a loaded entry list.
fn resolve_tenant_entries(entries: &[TenantEntry], input: &str) -> Result<String> {
    if input.is_empty() {
        bail!("tenant argument must not be empty");
    }

    // 1. Exact tenant id.
    if let Some(e) = entries.iter().find(|e| e.tenant_id == input) {
        return Ok(e.tenant_id.clone());
    }

    // 2. Exact name. Multiple clones can share a basename — ambiguity error.
    let tier: Vec<_> = entries.iter().filter(|e| e.name == input).collect();
    match tier.len() {
        1 => return Ok(tier[0].tenant_id.clone()),
        n if n > 1 => bail!(ambiguous_message(input, &tier)),
        _ => {}
    }

    // 3. Unique tenant-id prefix.
    let tier: Vec<_> = entries
        .iter()
        .filter(|e| e.tenant_id.starts_with(input))
        .collect();
    match tier.len() {
        1 => return Ok(tier[0].tenant_id.clone()),
        n if n > 1 => bail!(ambiguous_message(input, &tier)),
        _ => {}
    }

    // 4. Unique name prefix.
    let tier: Vec<_> = entries
        .iter()
        .filter(|e| e.name.starts_with(input))
        .collect();
    match tier.len() {
        1 => return Ok(tier[0].tenant_id.clone()),
        n if n > 1 => bail!(ambiguous_message(input, &tier)),
        _ => {}
    }

    // 5. Unique name substring.
    let tier: Vec<_> = entries.iter().filter(|e| e.name.contains(input)).collect();
    match tier.len() {
        1 => return Ok(tier[0].tenant_id.clone()),
        n if n > 1 => bail!(ambiguous_message(input, &tier)),
        _ => {}
    }

    let known: Vec<String> = entries.iter().map(|e| e.describe()).collect();
    bail!(
        "no project matches '{}'. Registered projects:\n  {}",
        input,
        known.join("\n  ")
    )
}

fn ambiguous_message(input: &str, candidates: &[&TenantEntry]) -> String {
    let listing: Vec<String> = candidates.iter().map(|e| e.describe()).collect();
    format!(
        "'{}' is ambiguous — matches {} projects:\n  {}\nUse the full name or tenant id.",
        input,
        candidates.len(),
        listing.join("\n  ")
    )
}

/// tenant_id → display name map for rendering outputs.
///
/// Best-effort: an unreadable database yields an empty map so callers fall
/// back to showing raw ids.
pub fn name_map() -> HashMap<String, String> {
    let Ok(conn) = connect_readonly() else {
        return HashMap::new();
    };
    name_map_in(&conn)
}

/// [`name_map`] against an existing connection.
pub fn name_map_in(conn: &Connection) -> HashMap<String, String> {
    load_tenants(conn)
        .map(|entries| entries.into_iter().map(|e| (e.tenant_id, e.name)).collect())
        .unwrap_or_default()
}

/// Display form for a tenant id: the project name when known, the raw id
/// otherwise.
pub fn display_name(map: &HashMap<String, String>, tenant_id: &str) -> String {
    map.get(tenant_id)
        .cloned()
        .unwrap_or_else(|| tenant_id.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries() -> Vec<TenantEntry> {
        vec![
            TenantEntry {
                tenant_id: "7166665a005b".into(),
                name: "main-docker".into(),
                path: "/Users/x/.config/main-docker".into(),
            },
            TenantEntry {
                tenant_id: "4ed81466dec7".into(),
                name: "workspace-qdrant-mcp".into(),
                path: "/Users/x/dev/projects/mcp/workspace-qdrant-mcp".into(),
            },
            TenantEntry {
                tenant_id: "aaaa00000001".into(),
                name: "tool".into(),
                path: "/Users/x/dev/a/tool".into(),
            },
            TenantEntry {
                tenant_id: "bbbb00000002".into(),
                name: "tool".into(),
                path: "/Users/x/dev/b/tool".into(),
            },
        ]
    }

    #[test]
    fn exact_tenant_id_wins() {
        assert_eq!(
            resolve_tenant_entries(&entries(), "7166665a005b").unwrap(),
            "7166665a005b"
        );
    }

    #[test]
    fn exact_name_resolves() {
        assert_eq!(
            resolve_tenant_entries(&entries(), "main-docker").unwrap(),
            "7166665a005b"
        );
    }

    #[test]
    fn exact_name_duplicate_is_ambiguous() {
        let err = resolve_tenant_entries(&entries(), "tool").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("ambiguous"), "{msg}");
        assert!(msg.contains("/Users/x/dev/a/tool"), "{msg}");
        assert!(msg.contains("/Users/x/dev/b/tool"), "{msg}");
    }

    #[test]
    fn tenant_id_prefix_resolves() {
        assert_eq!(
            resolve_tenant_entries(&entries(), "7166").unwrap(),
            "7166665a005b"
        );
    }

    #[test]
    fn name_prefix_resolves() {
        assert_eq!(
            resolve_tenant_entries(&entries(), "main-").unwrap(),
            "7166665a005b"
        );
    }

    #[test]
    fn name_substring_resolves() {
        assert_eq!(
            resolve_tenant_entries(&entries(), "qdrant").unwrap(),
            "4ed81466dec7"
        );
    }

    #[test]
    fn ambiguous_prefix_lists_candidates() {
        // "to" prefixes both "tool" entries.
        let err = resolve_tenant_entries(&entries(), "to").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("ambiguous"), "{msg}");
        assert!(msg.contains("aaaa00000001"), "{msg}");
        assert!(msg.contains("bbbb00000002"), "{msg}");
    }

    #[test]
    fn unknown_input_lists_registered() {
        let err = resolve_tenant_entries(&entries(), "nope-nothing").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no project matches"), "{msg}");
        assert!(msg.contains("main-docker"), "{msg}");
    }

    #[test]
    fn empty_input_rejected() {
        assert!(resolve_tenant_entries(&entries(), "").is_err());
    }

    #[test]
    fn display_name_falls_back_to_id() {
        let mut map = HashMap::new();
        map.insert("t1".to_string(), "proj".to_string());
        assert_eq!(display_name(&map, "t1"), "proj");
        assert_eq!(display_name(&map, "unknown"), "unknown");
    }
}
