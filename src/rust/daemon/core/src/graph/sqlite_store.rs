//! SQLite-backed graph store using recursive CTEs for traversal.

use async_trait::async_trait;
use sqlx::{Row, SqlitePool};
use tracing::debug;
use wqm_common::timestamps::now_utc;

use super::{
    compute_edge_id, EdgeType, GraphDbResult, GraphEdge, GraphNode, GraphStats, GraphStore,
    ImpactNode, ImpactReport, TraversalNode,
};

/// SQLite-backed implementation of `GraphStore`.
///
/// Uses a dedicated `graph.db` with WAL mode. Recursive CTEs handle
/// multi-hop traversal without requiring a graph database engine.
#[derive(Clone)]
pub struct SqliteGraphStore {
    pool: SqlitePool,
}

impl SqliteGraphStore {
    /// Create a new store from an existing connection pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Get a reference to the pool (for advanced queries in tests).
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

#[async_trait]
impl GraphStore for SqliteGraphStore {
    async fn upsert_node(&self, node: &GraphNode) -> GraphDbResult<()> {
        let now = now_utc();
        sqlx::query(
            "INSERT INTO graph_nodes (node_id, tenant_id, symbol_name, symbol_type,
                file_path, start_line, end_line, signature, language,
                created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?10)
            ON CONFLICT(node_id) DO UPDATE SET
                symbol_name = excluded.symbol_name,
                symbol_type = excluded.symbol_type,
                file_path = CASE WHEN excluded.file_path = '' THEN graph_nodes.file_path
                                 ELSE excluded.file_path END,
                start_line = COALESCE(excluded.start_line, graph_nodes.start_line),
                end_line = COALESCE(excluded.end_line, graph_nodes.end_line),
                signature = COALESCE(excluded.signature, graph_nodes.signature),
                language = COALESCE(excluded.language, graph_nodes.language),
                updated_at = ?10",
        )
        .bind(&node.node_id)
        .bind(&node.tenant_id)
        .bind(&node.symbol_name)
        .bind(node.symbol_type.as_str())
        .bind(&node.file_path)
        .bind(node.start_line.map(|v| v as i64))
        .bind(node.end_line.map(|v| v as i64))
        .bind(&node.signature)
        .bind(&node.language)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn upsert_nodes(&self, nodes: &[GraphNode]) -> GraphDbResult<()> {
        if nodes.is_empty() {
            return Ok(());
        }
        let now = now_utc();
        let mut tx = self.pool.begin().await?;

        for node in nodes {
            sqlx::query(
                "INSERT INTO graph_nodes (node_id, tenant_id, symbol_name, symbol_type,
                    file_path, start_line, end_line, signature, language,
                    created_at, updated_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?10)
                ON CONFLICT(node_id) DO UPDATE SET
                    symbol_name = excluded.symbol_name,
                    symbol_type = excluded.symbol_type,
                    file_path = CASE WHEN excluded.file_path = '' THEN graph_nodes.file_path
                                     ELSE excluded.file_path END,
                    start_line = COALESCE(excluded.start_line, graph_nodes.start_line),
                    end_line = COALESCE(excluded.end_line, graph_nodes.end_line),
                    signature = COALESCE(excluded.signature, graph_nodes.signature),
                    language = COALESCE(excluded.language, graph_nodes.language),
                    updated_at = ?10",
            )
            .bind(&node.node_id)
            .bind(&node.tenant_id)
            .bind(&node.symbol_name)
            .bind(node.symbol_type.as_str())
            .bind(&node.file_path)
            .bind(node.start_line.map(|v| v as i64))
            .bind(node.end_line.map(|v| v as i64))
            .bind(&node.signature)
            .bind(&node.language)
            .bind(&now)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        debug!("Upserted {} graph nodes", nodes.len());
        Ok(())
    }

    async fn insert_edge(&self, edge: &GraphEdge) -> GraphDbResult<()> {
        let now = now_utc();
        sqlx::query(
            "INSERT OR IGNORE INTO graph_edges
                (edge_id, tenant_id, source_node_id, target_node_id, edge_type,
                 source_file, weight, metadata_json, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )
        .bind(&edge.edge_id)
        .bind(&edge.tenant_id)
        .bind(&edge.source_node_id)
        .bind(&edge.target_node_id)
        .bind(edge.edge_type.as_str())
        .bind(&edge.source_file)
        .bind(edge.weight)
        .bind(&edge.metadata_json)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn insert_edges(&self, edges: &[GraphEdge]) -> GraphDbResult<()> {
        if edges.is_empty() {
            return Ok(());
        }
        let now = now_utc();
        let mut tx = self.pool.begin().await?;

        for edge in edges {
            sqlx::query(
                "INSERT OR IGNORE INTO graph_edges
                    (edge_id, tenant_id, source_node_id, target_node_id, edge_type,
                     source_file, weight, metadata_json, created_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )
            .bind(&edge.edge_id)
            .bind(&edge.tenant_id)
            .bind(&edge.source_node_id)
            .bind(&edge.target_node_id)
            .bind(edge.edge_type.as_str())
            .bind(&edge.source_file)
            .bind(edge.weight)
            .bind(&edge.metadata_json)
            .bind(&now)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        debug!("Inserted {} graph edges", edges.len());
        Ok(())
    }

    async fn delete_edges_by_file(&self, tenant_id: &str, file_path: &str) -> GraphDbResult<u64> {
        let result =
            sqlx::query("DELETE FROM graph_edges WHERE tenant_id = ?1 AND source_file = ?2")
                .bind(tenant_id)
                .bind(file_path)
                .execute(&self.pool)
                .await?;

        let count = result.rows_affected();
        debug!(
            "Deleted {} graph edges for file {} in tenant {}",
            count, file_path, tenant_id
        );
        Ok(count)
    }

    async fn delete_tenant(&self, tenant_id: &str) -> GraphDbResult<u64> {
        let mut tx = self.pool.begin().await?;

        let edge_result = sqlx::query("DELETE FROM graph_edges WHERE tenant_id = ?1")
            .bind(tenant_id)
            .execute(&mut *tx)
            .await?;

        let node_result = sqlx::query("DELETE FROM graph_nodes WHERE tenant_id = ?1")
            .bind(tenant_id)
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;

        let total = edge_result.rows_affected() + node_result.rows_affected();
        debug!(
            "Deleted {} items (edges + nodes) for tenant {}",
            total, tenant_id
        );
        Ok(total)
    }

    async fn query_related(
        &self,
        tenant_id: &str,
        node_id: &str,
        max_hops: u32,
        edge_types: Option<&[EdgeType]>,
    ) -> GraphDbResult<Vec<TraversalNode>> {
        // Build edge type filter clause
        let type_filter = match edge_types {
            Some(types) if !types.is_empty() => {
                let placeholders: Vec<String> =
                    types.iter().map(|t| format!("'{}'", t.as_str())).collect();
                format!("AND e.edge_type IN ({})", placeholders.join(", "))
            }
            _ => String::new(),
        };

        let query = format!(
            "WITH RECURSIVE graph_traverse AS (
                SELECT e.target_node_id AS node_id, e.edge_type, 1 AS depth,
                       e.source_node_id || ' -> ' || e.target_node_id AS path
                FROM graph_edges e
                WHERE e.source_node_id = ?1 AND e.tenant_id = ?2 AND ?3 >= 1 {type_filter}

                UNION ALL

                SELECT e.target_node_id, e.edge_type, gt.depth + 1,
                       gt.path || ' -> ' || e.target_node_id
                FROM graph_edges e
                INNER JOIN graph_traverse gt ON e.source_node_id = gt.node_id
                WHERE gt.depth < ?3 AND e.tenant_id = ?2 {type_filter}
            )
            SELECT DISTINCT gt.node_id, gt.edge_type, gt.depth, gt.path,
                   n.symbol_name, n.symbol_type, n.file_path
            FROM graph_traverse gt
            JOIN graph_nodes n ON gt.node_id = n.node_id
            ORDER BY gt.depth, n.symbol_name"
        );

        let rows = sqlx::query(&query)
            .bind(node_id)
            .bind(tenant_id)
            .bind(max_hops as i64)
            .fetch_all(&self.pool)
            .await?;

        let results = rows
            .iter()
            .map(|row| TraversalNode {
                node_id: row.get("node_id"),
                symbol_name: row.get("symbol_name"),
                symbol_type: row.get("symbol_type"),
                file_path: row.get("file_path"),
                edge_type: row.get("edge_type"),
                depth: row.get::<i64, _>("depth") as u32,
                path: row.get("path"),
            })
            .collect();

        Ok(results)
    }

    async fn impact_analysis(
        &self,
        tenant_id: &str,
        symbol_name: &str,
        file_path: Option<&str>,
    ) -> GraphDbResult<ImpactReport> {
        let target_nodes = self
            .find_target_nodes(tenant_id, symbol_name, file_path)
            .await?;

        if target_nodes.is_empty() {
            return Ok(ImpactReport {
                symbol_name: symbol_name.to_string(),
                impacted_nodes: vec![],
                total_impacted: 0,
            });
        }

        let mut all_impacted = Vec::new();
        for target_id in &target_nodes {
            let impacted = self.reverse_traverse(tenant_id, target_id).await?;
            all_impacted.extend(impacted);
        }

        all_impacted.sort_by_key(|n| n.distance);
        let mut seen = std::collections::HashSet::new();
        all_impacted.retain(|n| seen.insert(n.node_id.clone()));

        let total = all_impacted.len() as u32;
        Ok(ImpactReport {
            symbol_name: symbol_name.to_string(),
            impacted_nodes: all_impacted,
            total_impacted: total,
        })
    }

    async fn stats(&self, tenant_id: Option<&str>) -> GraphDbResult<GraphStats> {
        let (node_rows, edge_rows) = match tenant_id {
            Some(tid) => {
                let nodes = sqlx::query(
                    "SELECT symbol_type, COUNT(*) as cnt FROM graph_nodes
                     WHERE tenant_id = ?1 GROUP BY symbol_type",
                )
                .bind(tid)
                .fetch_all(&self.pool)
                .await?;
                let edges = sqlx::query(
                    "SELECT edge_type, COUNT(*) as cnt FROM graph_edges
                     WHERE tenant_id = ?1 GROUP BY edge_type",
                )
                .bind(tid)
                .fetch_all(&self.pool)
                .await?;
                (nodes, edges)
            }
            None => {
                let nodes = sqlx::query(
                    "SELECT symbol_type, COUNT(*) as cnt FROM graph_nodes
                     GROUP BY symbol_type",
                )
                .fetch_all(&self.pool)
                .await?;
                let edges = sqlx::query(
                    "SELECT edge_type, COUNT(*) as cnt FROM graph_edges
                     GROUP BY edge_type",
                )
                .fetch_all(&self.pool)
                .await?;
                (nodes, edges)
            }
        };

        let mut stats = GraphStats::default();
        for row in &node_rows {
            let stype: String = row.get("symbol_type");
            let cnt: i64 = row.get("cnt");
            stats.total_nodes += cnt as u64;
            stats.nodes_by_type.insert(stype, cnt as u64);
        }
        for row in &edge_rows {
            let etype: String = row.get("edge_type");
            let cnt: i64 = row.get("cnt");
            stats.total_edges += cnt as u64;
            stats.edges_by_type.insert(etype, cnt as u64);
        }

        Ok(stats)
    }

    async fn prune_orphans(&self, tenant_id: &str) -> GraphDbResult<u64> {
        let result = sqlx::query(
            "DELETE FROM graph_nodes
             WHERE tenant_id = ?1
               AND node_id NOT IN (
                   SELECT source_node_id FROM graph_edges WHERE tenant_id = ?1
                   UNION
                   SELECT target_node_id FROM graph_edges WHERE tenant_id = ?1
               )",
        )
        .bind(tenant_id)
        .execute(&self.pool)
        .await?;

        let count = result.rows_affected();
        debug!("Pruned {} orphaned nodes for tenant {}", count, tenant_id);
        Ok(count)
    }

    async fn resolve_stub_edges(&self, tenant_id: &str) -> GraphDbResult<u64> {
        use std::collections::HashMap;

        // 1. All dangling edges: target is a stub node (empty file_path).
        let dangling = sqlx::query(
            "SELECT e.edge_id, e.source_node_id, e.edge_type, e.source_file,
                    e.weight, e.metadata_json, t.symbol_name AS target_name
             FROM graph_edges e
             JOIN graph_nodes t ON e.target_node_id = t.node_id
             WHERE e.tenant_id = ?1 AND (t.file_path IS NULL OR t.file_path = '')",
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await?;

        if dangling.is_empty() {
            return Ok(0);
        }

        // 2. Real candidate nodes (resolved file_path, not file-typed), indexed
        //    by symbol_name -> [(node_id, file_path)].
        let real_rows = sqlx::query(
            "SELECT node_id, symbol_name, file_path FROM graph_nodes
             WHERE tenant_id = ?1 AND file_path <> '' AND symbol_type <> 'file'",
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await?;

        let mut by_name: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for r in &real_rows {
            let name: String = r.get("symbol_name");
            let nid: String = r.get("node_id");
            let fp: String = r.get("file_path");
            by_name.entry(name).or_default().push((nid, fp));
        }

        // 3. Repoint each resolvable dangling edge to its real node.
        let now = now_utc();
        let mut repointed: u64 = 0;
        let mut tx = self.pool.begin().await?;
        for d in &dangling {
            let target_name: String = d.get("target_name");
            let Some(candidates) = by_name.get(&target_name) else {
                continue; // external/stdlib — no project node with this name.
            };
            let source_file: String = d.get("source_file");
            // Prefer a definition in the caller's own file; else require a
            // unique tenant-wide match. Ambiguous (>1 file) names are skipped.
            let chosen: Option<&String> = candidates
                .iter()
                .find(|(_, fp)| *fp == source_file)
                .map(|(nid, _)| nid)
                .or_else(|| {
                    if candidates.len() == 1 {
                        Some(&candidates[0].0)
                    } else {
                        None
                    }
                });
            let Some(new_target) = chosen else {
                continue;
            };
            let source_node_id: String = d.get("source_node_id");
            // Skip self-loops (e.g. direct recursion) — they don't add
            // relationship signal and skew centrality measures.
            if &source_node_id == new_target {
                continue;
            }
            let edge_type_str: String = d.get("edge_type");
            let Some(edge_type) = EdgeType::from_str(&edge_type_str) else {
                continue;
            };
            let old_edge_id: String = d.get("edge_id");
            let weight: f64 = d.get("weight");
            let metadata_json: Option<String> = d.get("metadata_json");
            let new_edge_id = compute_edge_id(&source_node_id, new_target, edge_type);

            sqlx::query(
                "INSERT OR IGNORE INTO graph_edges
                    (edge_id, tenant_id, source_node_id, target_node_id, edge_type,
                     source_file, weight, metadata_json, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )
            .bind(&new_edge_id)
            .bind(tenant_id)
            .bind(&source_node_id)
            .bind(new_target)
            .bind(edge_type.as_str())
            .bind(&source_file)
            .bind(weight)
            .bind(&metadata_json)
            .bind(&now)
            .execute(&mut *tx)
            .await?;

            sqlx::query("DELETE FROM graph_edges WHERE edge_id = ?1 AND tenant_id = ?2")
                .bind(&old_edge_id)
                .bind(tenant_id)
                .execute(&mut *tx)
                .await?;
            repointed += 1;
        }
        tx.commit().await?;

        // 4. Drop stub nodes that no longer have any edges.
        sqlx::query(
            "DELETE FROM graph_nodes
             WHERE tenant_id = ?1 AND file_path = ''
               AND node_id NOT IN (
                   SELECT source_node_id FROM graph_edges WHERE tenant_id = ?1
                   UNION
                   SELECT target_node_id FROM graph_edges WHERE tenant_id = ?1
               )",
        )
        .bind(tenant_id)
        .execute(&self.pool)
        .await?;

        debug!(
            "Resolved {} stub edges for tenant {} ({} dangling examined)",
            repointed,
            tenant_id,
            dangling.len()
        );
        Ok(repointed)
    }
}

impl SqliteGraphStore {
    async fn find_target_nodes(
        &self,
        tenant_id: &str,
        symbol_name: &str,
        file_path: Option<&str>,
    ) -> GraphDbResult<Vec<String>> {
        if let Some(fp) = file_path {
            let rows = sqlx::query(
                "SELECT node_id FROM graph_nodes
                 WHERE tenant_id = ?1 AND symbol_name = ?2 AND file_path = ?3",
            )
            .bind(tenant_id)
            .bind(symbol_name)
            .bind(fp)
            .fetch_all(&self.pool)
            .await?;
            Ok(rows.iter().map(|r| r.get("node_id")).collect())
        } else {
            let rows = sqlx::query(
                "SELECT node_id FROM graph_nodes
                 WHERE tenant_id = ?1 AND symbol_name = ?2",
            )
            .bind(tenant_id)
            .bind(symbol_name)
            .fetch_all(&self.pool)
            .await?;
            Ok(rows.iter().map(|r| r.get("node_id")).collect())
        }
    }

    async fn reverse_traverse(
        &self,
        tenant_id: &str,
        target_id: &str,
    ) -> GraphDbResult<Vec<ImpactNode>> {
        let rows = sqlx::query(
            "WITH RECURSIVE reverse_traverse AS (
                SELECT e.source_node_id AS node_id, e.edge_type, 1 AS depth
                FROM graph_edges e
                WHERE e.target_node_id = ?1 AND e.tenant_id = ?2

                UNION ALL

                SELECT e.source_node_id, e.edge_type, rt.depth + 1
                FROM graph_edges e
                INNER JOIN reverse_traverse rt ON e.target_node_id = rt.node_id
                WHERE rt.depth < 3 AND e.tenant_id = ?2
            )
            SELECT DISTINCT rt.node_id, rt.edge_type, rt.depth,
                   n.symbol_name, n.file_path
            FROM reverse_traverse rt
            JOIN graph_nodes n ON rt.node_id = n.node_id
            ORDER BY rt.depth, n.symbol_name",
        )
        .bind(target_id)
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|row| {
                let depth: i64 = row.get("depth");
                let edge_type_str: String = row.get("edge_type");
                let impact_type = match (depth, edge_type_str.as_str()) {
                    (1, "CALLS") => "direct_caller",
                    (1, "USES_TYPE") => "type_user",
                    (1, _) => "direct_reference",
                    (_, "CALLS") => "indirect_caller",
                    _ => "indirect_reference",
                };
                ImpactNode {
                    node_id: row.get("node_id"),
                    symbol_name: row.get("symbol_name"),
                    file_path: row.get("file_path"),
                    impact_type: impact_type.to_string(),
                    distance: depth as u32,
                }
            })
            .collect())
    }
}
