//! Field-level migration verification: diff two graph backends node-by-node.
//!
//! DATA-05: count comparison is insufficient — this module performs a full
//! content diff so that a migrated backend can be validated field-by-field
//! before the old backend is decommissioned.

use std::collections::HashMap;

use crate::graph::schema::GraphDbResult;
use crate::graph::{GraphEdge, GraphNode, GraphStore};

// ─── Public types ────────────────────────────────────────────────────────────

/// Kind of divergence found between source and destination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DivergenceKind {
    /// Primary key (node_id / edge_id) exists in `src` but not in `dst`.
    MissingInDst,
    /// Primary key exists in `dst` but not in `src`.
    MissingInSrc,
    /// Both backends have the record but a specific field differs.
    ValueMismatch,
}

/// A single divergence between source and destination backends.
#[derive(Debug, Clone)]
pub struct FieldDivergence {
    /// Nature of the divergence.
    pub kind: DivergenceKind,
    /// The primary key of the record (node_id or edge_id).
    pub primary_key: String,
    /// Field name that differs, or `"<record>"` for missing-record divergences.
    pub field: String,
    /// Value in the source backend (`None` when record is missing in src).
    pub src_value: Option<String>,
    /// Value in the destination backend (`None` when record is missing in dst).
    pub dst_value: Option<String>,
}

// ─── Main entry point ────────────────────────────────────────────────────────

/// Compare graph contents between two backends for a single tenant.
///
/// Exports nodes and edges from both `src` and `dst` using
/// [`GraphStore::export_nodes_for_tenant`] / [`GraphStore::export_edges_for_tenant`],
/// then compares field-by-field. Returns every divergence found; an empty vec
/// means the migration is clean for this tenant.
///
/// **DATA-05**: count comparison alone is forbidden — this function always
/// performs a full content diff.
pub async fn diff_graph_contents(
    src: &dyn GraphStore,
    dst: &dyn GraphStore,
    tenant_id: &str,
) -> GraphDbResult<Vec<FieldDivergence>> {
    let (src_nodes, dst_nodes) = tokio::try_join!(
        src.export_nodes_for_tenant(tenant_id),
        dst.export_nodes_for_tenant(tenant_id),
    )?;
    let (src_edges, dst_edges) = tokio::try_join!(
        src.export_edges_for_tenant(tenant_id),
        dst.export_edges_for_tenant(tenant_id),
    )?;

    let mut divergences = Vec::new();
    diff_nodes(&src_nodes, &dst_nodes, &mut divergences);
    diff_edges(&src_edges, &dst_edges, &mut divergences);
    Ok(divergences)
}

// ─── Node comparison ─────────────────────────────────────────────────────────

fn diff_nodes(src: &[GraphNode], dst: &[GraphNode], out: &mut Vec<FieldDivergence>) {
    let src_map: HashMap<&str, &GraphNode> = src.iter().map(|n| (n.node_id.as_str(), n)).collect();
    let dst_map: HashMap<&str, &GraphNode> = dst.iter().map(|n| (n.node_id.as_str(), n)).collect();

    // Records present in src but missing or diverged in dst.
    for (id, sn) in &src_map {
        match dst_map.get(id) {
            None => out.push(FieldDivergence {
                kind: DivergenceKind::MissingInDst,
                primary_key: id.to_string(),
                field: "<record>".to_string(),
                src_value: Some(sn.symbol_name.clone()),
                dst_value: None,
            }),
            Some(dn) => compare_node_fields(id, sn, dn, out),
        }
    }

    // Records present in dst but missing in src.
    for id in dst_map.keys() {
        if !src_map.contains_key(id) {
            let dn = dst_map[id];
            out.push(FieldDivergence {
                kind: DivergenceKind::MissingInSrc,
                primary_key: id.to_string(),
                field: "<record>".to_string(),
                src_value: None,
                dst_value: Some(dn.symbol_name.clone()),
            });
        }
    }
}

fn compare_node_fields(id: &str, s: &GraphNode, d: &GraphNode, out: &mut Vec<FieldDivergence>) {
    // Every field that both backends faithfully round-trip is compared (DATA-05:
    // a content diff, not a count). `node_id` is the key and is omitted.
    let checks: &[(&str, String, String)] = &[
        ("tenant_id", s.tenant_id.clone(), d.tenant_id.clone()),
        ("symbol_name", s.symbol_name.clone(), d.symbol_name.clone()),
        (
            "symbol_type",
            s.symbol_type.as_str().to_string(),
            d.symbol_type.as_str().to_string(),
        ),
        ("file_path", s.file_path.clone(), d.file_path.clone()),
        ("start_line", opt_num(s.start_line), opt_num(d.start_line)),
        ("end_line", opt_num(s.end_line), opt_num(d.end_line)),
        ("signature", opt_str(&s.signature), opt_str(&d.signature)),
        ("language", opt_str(&s.language), opt_str(&d.language)),
        ("branches", s.branches.clone(), d.branches.clone()),
        (
            "qdrant_point_id",
            opt_str(&s.qdrant_point_id),
            opt_str(&d.qdrant_point_id),
        ),
        (
            "point_id_state",
            s.point_id_state.clone(),
            d.point_id_state.clone(),
        ),
    ];

    for (field, sv, dv) in checks {
        if sv != dv {
            out.push(FieldDivergence {
                kind: DivergenceKind::ValueMismatch,
                primary_key: id.to_string(),
                field: field.to_string(),
                src_value: Some(sv.clone()),
                dst_value: Some(dv.clone()),
            });
        }
    }
}

// ─── Edge comparison ─────────────────────────────────────────────────────────

fn diff_edges(src: &[GraphEdge], dst: &[GraphEdge], out: &mut Vec<FieldDivergence>) {
    let src_map: HashMap<&str, &GraphEdge> = src.iter().map(|e| (e.edge_id.as_str(), e)).collect();
    let dst_map: HashMap<&str, &GraphEdge> = dst.iter().map(|e| (e.edge_id.as_str(), e)).collect();

    for (id, se) in &src_map {
        match dst_map.get(id) {
            None => out.push(FieldDivergence {
                kind: DivergenceKind::MissingInDst,
                primary_key: id.to_string(),
                field: "<record>".to_string(),
                src_value: Some(se.edge_type.as_str().to_string()),
                dst_value: None,
            }),
            Some(de) => compare_edge_fields(id, se, de, out),
        }
    }

    for id in dst_map.keys() {
        if !src_map.contains_key(id) {
            let de = dst_map[id];
            out.push(FieldDivergence {
                kind: DivergenceKind::MissingInSrc,
                primary_key: id.to_string(),
                field: "<record>".to_string(),
                src_value: None,
                dst_value: Some(de.edge_type.as_str().to_string()),
            });
        }
    }
}

fn compare_edge_fields(id: &str, s: &GraphEdge, d: &GraphEdge, out: &mut Vec<FieldDivergence>) {
    let checks: &[(&str, String, String)] = &[
        ("tenant_id", s.tenant_id.clone(), d.tenant_id.clone()),
        (
            "source_node_id",
            s.source_node_id.clone(),
            d.source_node_id.clone(),
        ),
        (
            "target_node_id",
            s.target_node_id.clone(),
            d.target_node_id.clone(),
        ),
        (
            "edge_type",
            s.edge_type.as_str().to_string(),
            d.edge_type.as_str().to_string(),
        ),
        ("source_file", s.source_file.clone(), d.source_file.clone()),
        // Canonical float formatting so equal weights never spuriously mismatch.
        (
            "weight",
            format!("{:.6}", s.weight),
            format!("{:.6}", d.weight),
        ),
        (
            "metadata_json",
            opt_str(&s.metadata_json),
            opt_str(&d.metadata_json),
        ),
        ("branch", opt_str(&s.branch), opt_str(&d.branch)),
    ];

    for (field, sv, dv) in checks {
        if sv != dv {
            out.push(FieldDivergence {
                kind: DivergenceKind::ValueMismatch,
                primary_key: id.to_string(),
                field: field.to_string(),
                src_value: Some(sv.clone()),
                dst_value: Some(dv.clone()),
            });
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn opt_str(v: &Option<String>) -> String {
    v.as_deref().unwrap_or("").to_string()
}

fn opt_num(v: Option<u32>) -> String {
    v.map(|n| n.to_string()).unwrap_or_default()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::sqlite_store::SqliteGraphStore;
    use crate::graph::{EdgeType, GraphEdge, GraphNode, GraphStore, NodeType};
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    async fn test_store() -> SqliteGraphStore {
        let opts = SqliteConnectOptions::new()
            .filename(":memory:")
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(opts)
            .await
            .unwrap();

        // Minimal schema matching the current graph schema (including v5 columns).
        sqlx::query(
            "CREATE TABLE graph_nodes (
                node_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                signature TEXT,
                language TEXT,
                branches TEXT NOT NULL DEFAULT '[\"main\"]',
                qdrant_point_id TEXT,
                point_id_state TEXT NOT NULL DEFAULT 'none',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE graph_edges (
                edge_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                source_file TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata_json TEXT,
                branch TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                FOREIGN KEY (source_node_id) REFERENCES graph_nodes(node_id),
                FOREIGN KEY (target_node_id) REFERENCES graph_nodes(node_id)
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        SqliteGraphStore::new(pool)
    }

    const TENANT: &str = "t-verify";

    /// Identical contents → empty divergence list.
    #[tokio::test]
    async fn test_diff_identical_is_empty() {
        let src = test_store().await;
        let dst = test_store().await;

        let a = GraphNode::new(TENANT, "a.rs", "func_a", NodeType::Function);
        let b = GraphNode::new(TENANT, "b.rs", "struct_b", NodeType::Struct);
        src.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();
        dst.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();

        let edge = GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs");
        src.insert_edge(&edge).await.unwrap();
        dst.insert_edge(&edge).await.unwrap();

        let divs = diff_graph_contents(&src, &dst, TENANT).await.unwrap();
        assert!(
            divs.is_empty(),
            "identical backends should produce no divergences, got: {:?}",
            divs.iter().map(|d| &d.field).collect::<Vec<_>>()
        );
    }

    /// Mutating one node field produces exactly one ValueMismatch.
    #[tokio::test]
    async fn test_diff_single_field_mutation_produces_one_divergence() {
        let src = test_store().await;
        let dst = test_store().await;

        let node = GraphNode::new(TENANT, "a.rs", "func_a", NodeType::Function);
        src.upsert_node(&node).await.unwrap();

        // dst gets the same node but with a different signature
        let mut node_mutated = node.clone();
        node_mutated.signature = Some("fn func_a(x: i32) -> i32".to_string());
        dst.upsert_node(&node_mutated).await.unwrap();

        let divs = diff_graph_contents(&src, &dst, TENANT).await.unwrap();
        assert_eq!(
            divs.len(),
            1,
            "expected exactly 1 divergence, got {}: {:?}",
            divs.len(),
            divs.iter().map(|d| &d.field).collect::<Vec<_>>()
        );
        assert_eq!(divs[0].kind, DivergenceKind::ValueMismatch);
        assert_eq!(divs[0].field, "signature");
        assert_eq!(divs[0].primary_key, node.node_id);
        assert_eq!(divs[0].src_value, Some("".to_string())); // src has None → ""
        assert_eq!(
            divs[0].dst_value,
            Some("fn func_a(x: i32) -> i32".to_string())
        );
    }

    /// Node present in src but absent in dst → MissingInDst.
    #[tokio::test]
    async fn test_diff_missing_in_dst() {
        let src = test_store().await;
        let dst = test_store().await;

        let node = GraphNode::new(TENANT, "a.rs", "func_a", NodeType::Function);
        src.upsert_node(&node).await.unwrap();
        // dst is empty

        let divs = diff_graph_contents(&src, &dst, TENANT).await.unwrap();
        assert_eq!(divs.len(), 1);
        assert_eq!(divs[0].kind, DivergenceKind::MissingInDst);
        assert_eq!(divs[0].primary_key, node.node_id);
    }

    /// Node present in dst but absent in src → MissingInSrc.
    #[tokio::test]
    async fn test_diff_missing_in_src() {
        let src = test_store().await;
        let dst = test_store().await;

        let node = GraphNode::new(TENANT, "a.rs", "func_a", NodeType::Function);
        dst.upsert_node(&node).await.unwrap();
        // src is empty

        let divs = diff_graph_contents(&src, &dst, TENANT).await.unwrap();
        assert_eq!(divs.len(), 1);
        assert_eq!(divs[0].kind, DivergenceKind::MissingInSrc);
        assert_eq!(divs[0].primary_key, node.node_id);
    }

    /// Mutating qdrant_point_id produces a ValueMismatch on that field.
    #[tokio::test]
    async fn test_diff_qdrant_point_id_mismatch() {
        let src = test_store().await;
        let dst = test_store().await;

        let node = GraphNode::new(TENANT, "a.rs", "func_a", NodeType::Function)
            .with_qdrant_point_id("pid-original".to_string());
        src.upsert_node(&node).await.unwrap();

        let mut mutated = node.clone();
        mutated.qdrant_point_id = Some("pid-different".to_string());
        dst.upsert_node(&mutated).await.unwrap();

        let divs = diff_graph_contents(&src, &dst, TENANT).await.unwrap();
        assert_eq!(divs.len(), 1);
        assert_eq!(divs[0].kind, DivergenceKind::ValueMismatch);
        assert_eq!(divs[0].field, "qdrant_point_id");
    }

    /// Mutating an edge field produces exactly one ValueMismatch.
    #[tokio::test]
    async fn test_diff_edge_field_mutation() {
        let src = test_store().await;
        let dst = test_store().await;

        let a = GraphNode::new(TENANT, "a.rs", "func_a", NodeType::Function);
        let b = GraphNode::new(TENANT, "b.rs", "struct_b", NodeType::Struct);
        for store in [&src, &dst] {
            store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();
        }

        let edge = GraphEdge::new(TENANT, &a.node_id, &b.node_id, EdgeType::Calls, "a.rs");
        src.insert_edge(&edge).await.unwrap();

        let mut edge_mutated = edge.clone();
        edge_mutated.branch = Some("feature-x".to_string());
        dst.insert_edge(&edge_mutated).await.unwrap();

        let divs = diff_graph_contents(&src, &dst, TENANT).await.unwrap();
        assert_eq!(divs.len(), 1);
        assert_eq!(divs[0].kind, DivergenceKind::ValueMismatch);
        assert_eq!(divs[0].field, "branch");
    }
}
