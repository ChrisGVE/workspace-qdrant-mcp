//! Per-entity queue breakdown query.

use crate::commands::watch::helpers::{build_full_tenant_name_map, prefixed_display_name};
use crate::data::db::connect_readonly;

/// Get per-entity queue breakdown with collection-aware display names.
///
/// Returns `(display_name, pending, in_progress, failed)` tuples sorted
/// alphabetically by display name. Display names use collection prefixes
/// (`prj:`, `lib:`, `rls:`, `scp:`) when the queue contains items from
/// multiple collection types.
pub(super) fn get_per_entity_queue(
    conn: &rusqlite::Connection,
    tenant_names: &std::collections::HashMap<String, String>,
) -> Vec<(String, usize, usize, usize)> {
    let mut result: std::collections::HashMap<(String, String), (usize, usize, usize)> =
        std::collections::HashMap::new();

    let Ok(mut stmt) = conn.prepare(
        "SELECT collection, tenant_id, status, COUNT(*) FROM unified_queue \
         WHERE status IN ('pending', 'in_progress', 'failed') \
         GROUP BY collection, tenant_id, status",
    ) else {
        return Vec::new();
    };

    let Ok(rows) = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, usize>(3)?,
        ))
    }) else {
        return Vec::new();
    };

    // Track which collections appear to decide if prefixes are needed
    let mut collections_seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for row in rows.flatten() {
        let (collection, tenant_id, status, count) = row;
        collections_seen.insert(collection.clone());
        let entry = result.entry((collection, tenant_id)).or_insert((0, 0, 0));
        match status.as_str() {
            "pending" => entry.0 += count,
            "in_progress" => entry.1 += count,
            "failed" => entry.2 += count,
            _ => {}
        }
    }

    let use_prefixes = collections_seen.len() > 1;

    let mut sorted: Vec<(String, usize, usize, usize)> = result
        .into_iter()
        .map(|((collection, tenant_id), (p, i, f))| {
            let display = if use_prefixes {
                prefixed_display_name(&collection, &tenant_id, tenant_names)
            } else {
                tenant_names.get(&tenant_id).cloned().unwrap_or(tenant_id)
            };
            (display, p, i, f)
        })
        .collect();
    sorted.sort_by_key(|row| row.0.to_lowercase());
    sorted
}

/// Load per-entity queue data for verbose display.
///
/// Returns `None` if the DB is unavailable or there are no queue entries.
pub(super) fn load_entity_queue_data() -> Option<Vec<(String, usize, usize, usize)>> {
    connect_readonly().ok().and_then(|conn| {
        let tenant_names = build_full_tenant_name_map(&conn);
        let per_entity = get_per_entity_queue(&conn, &tenant_names);
        if per_entity.is_empty() {
            None
        } else {
            Some(per_entity)
        }
    })
}
