//! SQLite-backed graph store using recursive CTEs for traversal.

use crate::config::GraphRagConfig;
use async_trait::async_trait;
use sqlx::{Row, SqlitePool};
use tracing::debug;
use wqm_common::timestamps::now_utc;

use super::{
    compute_edge_id, is_cross_branch, AdjacencyExport, EdgeType, GraphDbResult, GraphEdge,
    GraphNode, GraphStats, GraphStore, ImpactNode, ImpactReport, SymbolRow, TraversalNode,
};

use super::cross_boundary::{apply_fan_out_caps, tenant_relaxation_set, CROSS_BOUNDARY_MAX_HOPS};

/// SQLite-backed implementation of `GraphStore`.
///
/// Uses a dedicated `graph.db` with WAL mode. Recursive CTEs handle
/// multi-hop traversal without requiring a graph database engine.
#[derive(Clone)]
pub struct SqliteGraphStore {
    pool: SqlitePool,
    /// Cross-boundary fan-out caps and fusion settings.
    graph_rag: GraphRagConfig,
}

impl SqliteGraphStore {
    /// Create a new store from an existing connection pool, using default
    /// cross-boundary graph-RAG caps.
    pub fn new(pool: SqlitePool) -> Self {
        Self {
            pool,
            graph_rag: GraphRagConfig::default(),
        }
    }

    /// Override the cross-boundary graph-RAG configuration (fan-out caps).
    pub fn with_graph_rag_config(mut self, config: GraphRagConfig) -> Self {
        self.graph_rag = config;
        self
    }

    /// Get a reference to the pool (for advanced queries in tests).
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

// ── SQL fragment helpers ────────────────────────────────────────────

/// SQL clause filtering nodes by branch membership in their JSON array.
fn node_branch_clause(alias: &str, idx: usize) -> String {
    format!(" AND EXISTS (SELECT 1 FROM json_each({alias}.branches) WHERE value = ?{idx})")
}

/// SQL clause filtering edges by branch (NULL branch = global, always included).
fn edge_branch_clause(alias: &str, idx: usize) -> String {
    format!(" AND ({alias}.branch = ?{idx} OR {alias}.branch IS NULL)")
}

#[async_trait]
impl GraphStore for SqliteGraphStore {
    async fn upsert_node(&self, node: &GraphNode) -> GraphDbResult<()> {
        let now = now_utc();
        sqlx::query(
            "INSERT INTO graph_nodes (node_id, tenant_id, symbol_name, symbol_type,
                file_path, start_line, end_line, signature, language,
                branches, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?11)
            ON CONFLICT(node_id) DO UPDATE SET
                symbol_name = excluded.symbol_name,
                symbol_type = excluded.symbol_type,
                file_path = CASE WHEN excluded.file_path = '' THEN graph_nodes.file_path
                                 ELSE excluded.file_path END,
                start_line = COALESCE(excluded.start_line, graph_nodes.start_line),
                end_line = COALESCE(excluded.end_line, graph_nodes.end_line),
                signature = COALESCE(excluded.signature, graph_nodes.signature),
                language = COALESCE(excluded.language, graph_nodes.language),
                branches = excluded.branches,
                updated_at = ?11",
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
        .bind(&node.branches)
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
                    branches, created_at, updated_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?11)
                ON CONFLICT(node_id) DO UPDATE SET
                    symbol_name = excluded.symbol_name,
                    symbol_type = excluded.symbol_type,
                    file_path = CASE WHEN excluded.file_path = '' THEN graph_nodes.file_path
                                     ELSE excluded.file_path END,
                    start_line = COALESCE(excluded.start_line, graph_nodes.start_line),
                    end_line = COALESCE(excluded.end_line, graph_nodes.end_line),
                    signature = COALESCE(excluded.signature, graph_nodes.signature),
                    language = COALESCE(excluded.language, graph_nodes.language),
                    branches = excluded.branches,
                    updated_at = ?11",
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
            .bind(&node.branches)
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
                 source_file, weight, metadata_json, branch, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        )
        .bind(&edge.edge_id)
        .bind(&edge.tenant_id)
        .bind(&edge.source_node_id)
        .bind(&edge.target_node_id)
        .bind(edge.edge_type.as_str())
        .bind(&edge.source_file)
        .bind(edge.weight)
        .bind(&edge.metadata_json)
        .bind(&edge.branch)
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
                     source_file, weight, metadata_json, branch, created_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )
            .bind(&edge.edge_id)
            .bind(&edge.tenant_id)
            .bind(&edge.source_node_id)
            .bind(&edge.target_node_id)
            .bind(edge.edge_type.as_str())
            .bind(&edge.source_file)
            .bind(edge.weight)
            .bind(&edge.metadata_json)
            .bind(&edge.branch)
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
        branch: Option<&str>,
    ) -> GraphDbResult<Vec<TraversalNode>> {
        let type_filter = match edge_types {
            Some(types) if !types.is_empty() => {
                let vals: Vec<String> = types.iter().map(|t| format!("'{}'", t.as_str())).collect();
                format!("AND e.edge_type IN ({})", vals.join(", "))
            }
            _ => String::new(),
        };
        let scoped = !is_cross_branch(branch);
        let edge_br = if scoped {
            edge_branch_clause("e", 4)
        } else {
            String::new()
        };
        let node_br = if scoped {
            node_branch_clause("n", 4)
        } else {
            String::new()
        };

        let sql = format!(
            "WITH RECURSIVE graph_traverse AS (
                SELECT e.target_node_id AS node_id, e.edge_type, 1 AS depth,
                       e.source_node_id || ' -> ' || e.target_node_id AS path
                FROM graph_edges e
                WHERE e.source_node_id = ?1 AND e.tenant_id = ?2 AND ?3 >= 1
                      {type_filter}{edge_br}
                UNION ALL
                SELECT e.target_node_id, e.edge_type, gt.depth + 1,
                       gt.path || ' -> ' || e.target_node_id
                FROM graph_edges e
                INNER JOIN graph_traverse gt ON e.source_node_id = gt.node_id
                WHERE gt.depth < ?3 AND e.tenant_id = ?2
                      {type_filter}{edge_br}
            )
            SELECT DISTINCT gt.node_id, gt.edge_type, gt.depth, gt.path,
                   n.symbol_name, n.symbol_type, n.file_path
            FROM graph_traverse gt
            JOIN graph_nodes n ON gt.node_id = n.node_id
            WHERE 1=1{node_br}
            ORDER BY gt.depth, n.symbol_name"
        );
        let mut qb = sqlx::query(&sql)
            .bind(node_id)
            .bind(tenant_id)
            .bind(max_hops as i64);
        if scoped {
            qb = qb.bind(branch.unwrap_or_default());
        }
        let rows = qb.fetch_all(&self.pool).await?;
        Ok(rows
            .iter()
            .map(|row| TraversalNode {
                node_id: row.get("node_id"),
                symbol_name: row.get("symbol_name"),
                symbol_type: row.get("symbol_type"),
                file_path: row.get("file_path"),
                edge_type: row.get("edge_type"),
                depth: row.get::<i64, _>("depth") as u32,
                path: row.get("path"),
                tenant_id: tenant_id.to_string(),
                edge_confidence: 1.0,
            })
            .collect())
    }

    async fn impact_analysis(
        &self,
        tenant_id: &str,
        symbol_name: &str,
        file_path: Option<&str>,
        branch: Option<&str>,
    ) -> GraphDbResult<ImpactReport> {
        let targets = self
            .find_target_nodes(tenant_id, symbol_name, file_path, branch)
            .await?;
        if targets.is_empty() {
            return Ok(ImpactReport {
                symbol_name: symbol_name.to_string(),
                impacted_nodes: vec![],
                total_impacted: 0,
            });
        }
        let mut all = Vec::new();
        for tid in &targets {
            all.extend(self.reverse_traverse(tenant_id, tid, branch).await?);
        }
        all.sort_by_key(|n| n.distance);
        let mut seen = std::collections::HashSet::new();
        all.retain(|n| seen.insert(n.node_id.clone()));
        let total = all.len() as u32;
        Ok(ImpactReport {
            symbol_name: symbol_name.to_string(),
            impacted_nodes: all,
            total_impacted: total,
        })
    }

    async fn stats(
        &self,
        tenant_id: Option<&str>,
        branch: Option<&str>,
    ) -> GraphDbResult<GraphStats> {
        let scoped = !is_cross_branch(branch);
        let (node_rows, edge_rows) = match (tenant_id, scoped) {
            (Some(tid), true) => {
                let b = branch.unwrap_or_default();
                let nodes = sqlx::query(
                    "SELECT symbol_type, COUNT(*) as cnt FROM graph_nodes
                     WHERE tenant_id = ?1
                       AND EXISTS (SELECT 1 FROM json_each(branches) WHERE value = ?2)
                     GROUP BY symbol_type",
                )
                .bind(tid)
                .bind(b)
                .fetch_all(&self.pool)
                .await?;
                let edges = sqlx::query(
                    "SELECT edge_type, COUNT(*) as cnt FROM graph_edges
                     WHERE tenant_id = ?1 AND (branch = ?2 OR branch IS NULL)
                     GROUP BY edge_type",
                )
                .bind(tid)
                .bind(b)
                .fetch_all(&self.pool)
                .await?;
                (nodes, edges)
            }
            (Some(tid), false) => {
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
            (None, true) => {
                let b = branch.unwrap_or_default();
                let nodes = sqlx::query(
                    "SELECT symbol_type, COUNT(*) as cnt FROM graph_nodes
                     WHERE EXISTS (SELECT 1 FROM json_each(branches) WHERE value = ?1)
                     GROUP BY symbol_type",
                )
                .bind(b)
                .fetch_all(&self.pool)
                .await?;
                let edges = sqlx::query(
                    "SELECT edge_type, COUNT(*) as cnt FROM graph_edges
                     WHERE (branch = ?1 OR branch IS NULL) GROUP BY edge_type",
                )
                .bind(b)
                .fetch_all(&self.pool)
                .await?;
                (nodes, edges)
            }
            (None, false) => {
                let nodes = sqlx::query(
                    "SELECT symbol_type, COUNT(*) as cnt FROM graph_nodes GROUP BY symbol_type",
                )
                .fetch_all(&self.pool)
                .await?;
                let edges = sqlx::query(
                    "SELECT edge_type, COUNT(*) as cnt FROM graph_edges GROUP BY edge_type",
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
            "DELETE FROM graph_nodes WHERE tenant_id = ?1
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

    async fn find_path(
        &self,
        tenant_id: &str,
        source_id: &str,
        target_id: &str,
        max_depth: u32,
        edge_types: Option<&[EdgeType]>,
        branch: Option<&str>,
    ) -> GraphDbResult<Option<Vec<TraversalNode>>> {
        let scoped = !is_cross_branch(branch);
        let num_et = edge_types.map_or(0, |t| t.len());
        let edge_filter = if num_et > 0 {
            let ph: Vec<String> = (0..num_et).map(|i| format!("?{}", i + 4)).collect();
            format!("AND e.edge_type IN ({})", ph.join(","))
        } else {
            String::new()
        };
        let branch_slot = num_et + 4;
        let edge_br = if scoped {
            edge_branch_clause("e", branch_slot)
        } else {
            String::new()
        };
        let target_slot = if scoped { branch_slot + 1 } else { branch_slot };

        let sql = format!(
            r#"
            WITH RECURSIVE bfs(node_id, depth, path) AS (
                SELECT ?2, 0, ?2
                UNION ALL
                SELECT e.target_node_id, bfs.depth + 1,
                       bfs.path || ',' || e.target_node_id
                FROM bfs
                JOIN graph_edges e ON e.source_node_id = bfs.node_id
                                  AND e.tenant_id = ?1
                                  {edge_filter}{edge_br}
                WHERE bfs.depth < ?3
                  AND INSTR(bfs.path, e.target_node_id) = 0
            )
            SELECT bfs.node_id, bfs.depth, bfs.path,
                   n.symbol_name, n.symbol_type, n.file_path
            FROM bfs
            JOIN graph_nodes n ON n.node_id = bfs.node_id AND n.tenant_id = ?1
            WHERE bfs.node_id = ?{target_slot}
            ORDER BY bfs.depth ASC LIMIT 1
            "#
        );
        let mut query = sqlx::query(&sql)
            .bind(tenant_id)
            .bind(source_id)
            .bind(max_depth as i64);
        if let Some(types) = edge_types {
            for et in types {
                query = query.bind(et.as_str());
            }
        }
        if scoped {
            query = query.bind(branch.unwrap_or_default());
        }
        query = query.bind(target_id);

        let row = query.fetch_optional(&self.pool).await?;
        match row {
            None => Ok(None),
            Some(row) => {
                use sqlx::Row;
                let path_str: String = row.get("path");
                let node_ids: Vec<&str> = path_str.split(',').collect();
                let mut path_nodes = Vec::with_capacity(node_ids.len());
                for (hop, nid) in node_ids.iter().enumerate() {
                    let nr = sqlx::query(
                        "SELECT symbol_name, symbol_type, file_path
                         FROM graph_nodes WHERE node_id = ?1 AND tenant_id = ?2",
                    )
                    .bind(nid)
                    .bind(tenant_id)
                    .fetch_optional(&self.pool)
                    .await?;
                    if let Some(nr) = nr {
                        path_nodes.push(TraversalNode {
                            node_id: nid.to_string(),
                            symbol_name: nr.get("symbol_name"),
                            symbol_type: nr.get("symbol_type"),
                            file_path: nr.get("file_path"),
                            edge_type: String::new(),
                            depth: hop as u32,
                            path: String::new(),
                            tenant_id: tenant_id.to_string(),
                            edge_confidence: 1.0,
                        });
                    }
                }
                Ok(Some(path_nodes))
            }
        }
    }

    async fn reingest_file(
        &self,
        tenant_id: &str,
        file_path: &str,
        nodes: &[GraphNode],
        edges: &[GraphEdge],
    ) -> GraphDbResult<()> {
        let now = now_utc();
        let mut tx = self.pool.begin().await?;
        let deleted =
            sqlx::query("DELETE FROM graph_edges WHERE tenant_id = ?1 AND source_file = ?2")
                .bind(tenant_id)
                .bind(file_path)
                .execute(&mut *tx)
                .await?;
        debug!(
            "reingest_file: deleted {} old edges for {} in tenant {}",
            deleted.rows_affected(),
            file_path,
            tenant_id
        );
        // Delete file-owned narrative nodes so re-ingestion does not leave
        // orphans when a heading or comment shifts (re-keying its node id).
        // library_section is scoped per-file (tenant_id == library_name for the
        // libraries collection), so it is cleaned alongside the other narrative
        // node types; concept_node (global) and code-graph nodes are
        // deliberately excluded by the type filter.
        let deleted_nodes = sqlx::query(
            "DELETE FROM graph_nodes
             WHERE tenant_id = ?1 AND file_path = ?2
               AND symbol_type IN ('document_section', 'library_section', 'code_comment', 'docstring')",
        )
        .bind(tenant_id)
        .bind(file_path)
        .execute(&mut *tx)
        .await?;
        debug!(
            "reingest_file: deleted {} old narrative nodes for {} in tenant {}",
            deleted_nodes.rows_affected(),
            file_path,
            tenant_id
        );
        for node in nodes {
            sqlx::query(
                "INSERT INTO graph_nodes (node_id, tenant_id, symbol_name, symbol_type,
                    file_path, start_line, end_line, signature, language,
                    branches, created_at, updated_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?11)
                ON CONFLICT(node_id) DO UPDATE SET
                    symbol_name = excluded.symbol_name,
                    symbol_type = excluded.symbol_type,
                    file_path = CASE WHEN excluded.file_path = '' THEN graph_nodes.file_path ELSE excluded.file_path END,
                    start_line = COALESCE(excluded.start_line, graph_nodes.start_line),
                    end_line = COALESCE(excluded.end_line, graph_nodes.end_line),
                    signature = COALESCE(excluded.signature, graph_nodes.signature),
                    language = COALESCE(excluded.language, graph_nodes.language),
                    branches = excluded.branches,
                    updated_at = ?11",
            )
            .bind(&node.node_id).bind(&node.tenant_id).bind(&node.symbol_name)
            .bind(node.symbol_type.as_str()).bind(&node.file_path)
            .bind(node.start_line.map(|v| v as i64)).bind(node.end_line.map(|v| v as i64))
            .bind(&node.signature).bind(&node.language).bind(&node.branches).bind(&now)
            .execute(&mut *tx).await?;
        }
        for edge in edges {
            sqlx::query(
                "INSERT OR IGNORE INTO graph_edges
                    (edge_id, tenant_id, source_node_id, target_node_id, edge_type,
                     source_file, weight, metadata_json, branch, created_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )
            .bind(&edge.edge_id)
            .bind(&edge.tenant_id)
            .bind(&edge.source_node_id)
            .bind(&edge.target_node_id)
            .bind(edge.edge_type.as_str())
            .bind(&edge.source_file)
            .bind(edge.weight)
            .bind(&edge.metadata_json)
            .bind(&edge.branch)
            .bind(&now)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        debug!(
            "reingest_file: committed {} nodes + {} edges for {} in tenant {}",
            nodes.len(),
            edges.len(),
            file_path,
            tenant_id
        );
        Ok(())
    }

    async fn query_code_symbols(&self, tenant_id: &str) -> GraphDbResult<Vec<SymbolRow>> {
        // Structural (code) node types only — exclude narrative + concept nodes.
        let rows: Vec<(String, String, String)> = sqlx::query_as(
            "SELECT symbol_name, node_id, file_path
             FROM graph_nodes
             WHERE tenant_id = ?1
               AND symbol_type IN ('function', 'async_function', 'class', 'method',
                   'struct', 'trait', 'interface', 'enum', 'impl', 'module',
                   'constant', 'type_alias', 'macro')
               AND symbol_name != ''",
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|(symbol_name, node_id, file_path)| SymbolRow {
                symbol_name,
                node_id,
                file_path,
            })
            .collect())
    }

    async fn delete_narrative_nodes_by_file(
        &self,
        tenant_id: &str,
        file_path: &str,
    ) -> GraphDbResult<u64> {
        let result = sqlx::query(
            "DELETE FROM graph_nodes
             WHERE tenant_id = ?1 AND file_path = ?2
               AND symbol_type IN ('document_section', 'library_section', 'code_comment', 'docstring')",
        )
        .bind(tenant_id)
        .bind(file_path)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    async fn query_edges_by_type(&self, edge_type: EdgeType) -> GraphDbResult<Vec<GraphEdge>> {
        let rows: Vec<(
            String,
            String,
            String,
            String,
            String,
            String,
            f64,
            Option<String>,
            Option<String>,
        )> = sqlx::query_as(
            "SELECT edge_id, tenant_id, source_node_id, target_node_id, edge_type,
                        source_file, weight, metadata_json, branch
                 FROM graph_edges
                 WHERE edge_type = ?1",
        )
        .bind(edge_type.as_str())
        .fetch_all(&self.pool)
        .await?;

        let edges = rows
            .into_iter()
            .filter_map(|(eid, tid, src, tgt, etype, sf, w, meta, br)| {
                let et = EdgeType::from_str(&etype)?;
                Some(GraphEdge {
                    edge_id: eid,
                    tenant_id: tid,
                    source_node_id: src,
                    target_node_id: tgt,
                    edge_type: et,
                    source_file: sf,
                    weight: w,
                    metadata_json: meta,
                    branch: br,
                })
            })
            .collect();
        Ok(edges)
    }

    async fn query_cross_boundary(
        &self,
        source_tenant: &str,
        source_node_id: &str,
        edge_types: &[EdgeType],
        max_hops: u32,
        library_tenants: &[String],
    ) -> GraphDbResult<Vec<TraversalNode>> {
        if edge_types.is_empty() || max_hops == 0 {
            return Ok(Vec::new());
        }
        // Clamp to the supported recursion budget (1..=3).
        let hops = max_hops.clamp(1, CROSS_BOUNDARY_MAX_HOPS) as i64;

        // Edge-type IN-list. Edge types come from a fixed enum (as_str is a
        // static literal), so inlining is injection-safe.
        let type_list = edge_types
            .iter()
            .map(|et| format!("'{}'", et.as_str()))
            .collect::<Vec<_>>()
            .join(",");

        // Tenant relaxation set: source_tenant ∪ {"__global__"} ∪ library_tenants.
        // ConceptNodes live under "__global__", so it must always be included or
        // code→concept→code traversal returns nothing.
        let tenants = tenant_relaxation_set(source_tenant, library_tenants);
        // Bound parameters ?1=source_node_id, ?2=hops, ?3..=one per tenant, and
        // a final placeholder for source_tenant used by the seed-ownership guard.
        let tenant_placeholders = (0..tenants.len())
            .map(|i| format!("?{}", i + 3))
            .collect::<Vec<_>>()
            .join(",");
        // Seed-ownership guard: traversal may only START from a node inside the
        // relaxation set (source_tenant ∪ __global__ ∪ library_tenants). Without
        // it a caller could seed from a foreign node id and reach
        // __global__ / library nodes, bypassing tenant scoping. Concept and
        // library seeds remain valid because those tenants are in the set.

        // Per-edge-type base confidence (Feature E weight × per-type base is
        // applied in Rust; the CTE carries weight × base per reaching edge).
        let confidence_case = "CASE e.edge_type                 WHEN 'EXPLAINS' THEN 0.6                 WHEN 'COVERS_TOPIC' THEN 0.6                 WHEN 'IMPLEMENTS_CONCEPT' THEN 0.7                 ELSE 1.0 END";

        // Bidirectional recursive traversal. Each recursive member follows an
        // edge of the allowed set in one direction, joins the reached node, and
        // applies the tenant guard so we never hop through a foreign tenant.
        // `conf` carries the reaching edge's (weight × per-type base); `MAX` of
        // it is taken per reached node after grouping.
        let sql = format!(
            "WITH RECURSIVE traverse(node_id, depth, path, edge_type, conf) AS (
                SELECT ?1, 0, ?1, '', 1.0
                WHERE EXISTS (
                    SELECT 1 FROM graph_nodes
                    WHERE node_id = ?1 AND tenant_id IN ({tenant_placeholders})
                )
                UNION ALL
                SELECT n.node_id, t.depth + 1,
                       t.path || ' -> ' || n.node_id,
                       e.edge_type,
                       COALESCE(e.weight, 1.0) * ({confidence_case})
                FROM traverse t
                CROSS JOIN graph_edges e ON e.source_node_id = t.node_id
                JOIN graph_nodes n ON n.node_id = e.target_node_id
                WHERE t.depth < ?2
                  AND e.edge_type IN ({type_list})
                  AND INSTR(t.path, n.node_id) = 0
                  AND (n.tenant_id IN ({tenant_placeholders}))
                UNION ALL
                SELECT n.node_id, t.depth + 1,
                       t.path || ' -> ' || n.node_id,
                       e.edge_type,
                       COALESCE(e.weight, 1.0) * ({confidence_case})
                FROM traverse t
                CROSS JOIN graph_edges e ON e.target_node_id = t.node_id
                JOIN graph_nodes n ON n.node_id = e.source_node_id
                WHERE t.depth < ?2
                  AND e.edge_type IN ({type_list})
                  AND INSTR(t.path, n.node_id) = 0
                  AND (n.tenant_id IN ({tenant_placeholders}))
            ),
            ranked AS (
                SELECT t.node_id, t.depth, t.path, t.edge_type, t.conf,
                       ROW_NUMBER() OVER (
                           PARTITION BY t.node_id
                           ORDER BY t.depth ASC, t.conf DESC, t.path ASC
                       ) AS rn
                FROM traverse t
                WHERE t.depth > 0
            )
            SELECT n.node_id, n.symbol_name, n.symbol_type, n.file_path,
                   n.tenant_id,
                   r.depth AS depth,
                   r.edge_type AS edge_type,
                   r.conf AS edge_confidence,
                   r.path AS path
            FROM ranked r
            JOIN graph_nodes n ON n.node_id = r.node_id
            WHERE r.rn = 1
            ORDER BY r.depth, n.symbol_name"
        );

        let mut q = sqlx::query_as::<
            _,
            (
                String,
                String,
                String,
                String,
                String,
                i64,
                String,
                f64,
                String,
            ),
        >(&sql)
        .bind(source_node_id)
        .bind(hops);
        for tenant in &tenants {
            q = q.bind(tenant);
        }
        let rows = q.fetch_all(&self.pool).await?;

        let results = rows
            .into_iter()
            .map(
                |(node_id, name, stype, fpath, tenant_id, depth, etype, conf, path)| {
                    TraversalNode {
                        node_id,
                        symbol_name: name,
                        symbol_type: stype,
                        file_path: fpath,
                        edge_type: etype,
                        depth: depth as u32,
                        path,
                        tenant_id,
                        edge_confidence: conf,
                    }
                },
            )
            .collect();

        Ok(apply_fan_out_caps(results, &self.graph_rag))
    }

    async fn resolve_stub_edges(&self, tenant_id: &str) -> GraphDbResult<u64> {
        use std::collections::HashMap;

        // 1. All dangling edges: target is a stub node (empty file_path).
        //    Carry e.branch so the repointed edge keeps the original branch
        //    scope (NULL = global); the JOIN is tenant-scoped on both sides so
        //    a node_id can never resolve against another tenant's node.
        let dangling = sqlx::query(
            "SELECT e.edge_id, e.source_node_id, e.edge_type, e.source_file,
                    e.weight, e.metadata_json, e.branch, t.symbol_name AS target_name
             FROM graph_edges e
             JOIN graph_nodes t
                  ON e.target_node_id = t.node_id AND t.tenant_id = e.tenant_id
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
            let branch: Option<String> = d.get("branch");
            let new_edge_id = compute_edge_id(&source_node_id, new_target, edge_type);

            // `new_target` is a real node read above under the same write lock,
            // so its FK cannot vanish mid-call; an INSERT that affects 0 rows
            // therefore means the repointed edge already exists (PK conflict),
            // not an FK failure. Either way the repoint target is present, so
            // removing the old dangling edge is safe. We only delete the old
            // edge after confirming the repoint target edge exists (inserted or
            // already there) so a silent INSERT failure can never drop an edge.
            let inserted = sqlx::query(
                "INSERT OR IGNORE INTO graph_edges
                    (edge_id, tenant_id, source_node_id, target_node_id, edge_type,
                     source_file, weight, metadata_json, branch, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )
            .bind(&new_edge_id)
            .bind(tenant_id)
            .bind(&source_node_id)
            .bind(new_target)
            .bind(edge_type.as_str())
            .bind(&source_file)
            .bind(weight)
            .bind(&metadata_json)
            .bind(&branch)
            .bind(&now)
            .execute(&mut *tx)
            .await?;

            if inserted.rows_affected() == 0 {
                // Repoint target already present: confirm before deleting so we
                // never drop the old edge on an unexpected no-op.
                let exists: i64 = sqlx::query_scalar(
                    "SELECT COUNT(*) FROM graph_edges WHERE edge_id = ?1 AND tenant_id = ?2",
                )
                .bind(&new_edge_id)
                .bind(tenant_id)
                .fetch_one(&mut *tx)
                .await?;
                if exists == 0 {
                    // Should be unreachable given the invariant above; skip
                    // rather than risk dropping the edge.
                    continue;
                }
            }

            sqlx::query("DELETE FROM graph_edges WHERE edge_id = ?1 AND tenant_id = ?2")
                .bind(&old_edge_id)
                .bind(tenant_id)
                .execute(&mut *tx)
                .await?;
            repointed += 1;
        }

        // 4. Drop stub nodes that no longer have any edges. Done INSIDE the
        //    transaction so the NOT IN subquery sees this pass's repoints and a
        //    concurrent writer cannot insert an edge onto a stub between the
        //    repoint commit and the prune.
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
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        debug!(
            "Resolved {} stub edges for tenant {} ({} dangling examined)",
            repointed,
            tenant_id,
            dangling.len()
        );
        Ok(repointed)
    }

    async fn graph_tenants(&self) -> GraphDbResult<Vec<String>> {
        let tenants: Vec<String> = sqlx::query_scalar("SELECT DISTINCT tenant_id FROM graph_edges")
            .fetch_all(&self.pool)
            .await?;
        Ok(tenants)
    }

    async fn resolve_all_stub_edges(&self) -> GraphDbResult<u64> {
        let mut total: u64 = 0;
        for tenant in self.graph_tenants().await? {
            total += self.resolve_stub_edges(&tenant).await?;
        }
        Ok(total)
    }

    async fn export_adjacency(
        &self,
        tenant_id: &str,
        edge_types: Option<&[EdgeType]>,
    ) -> GraphDbResult<AdjacencyExport> {
        use std::collections::HashMap;

        // 1. Load all nodes for the tenant, sorted deterministically (DOM-01).
        let node_rows =
            sqlx::query("SELECT node_id FROM graph_nodes WHERE tenant_id = ?1 ORDER BY node_id")
                .bind(tenant_id)
                .fetch_all(&self.pool)
                .await?;

        let node_ids: Vec<String> = node_rows.iter().map(|r| r.get::<String, _>(0)).collect();

        // Build a name→index map so edge lookups are O(1).
        let index_map: HashMap<&str, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();

        // 2. Load edges with an optional type filter.
        //    Bind types as individual parameters (no string interpolation of
        //    values, unlike the old load_adjacency_graph which used format!).
        let num_et = edge_types.map_or(0, |t| t.len());
        let edge_rows = if num_et > 0 {
            // Build a query with bound placeholders for each edge type value.
            let placeholders: Vec<String> = (0..num_et).map(|i| format!("?{}", i + 2)).collect();
            let sql = format!(
                "SELECT source_node_id, target_node_id, weight \
                 FROM graph_edges \
                 WHERE tenant_id = ?1 AND edge_type IN ({}) \
                 ORDER BY source_node_id, target_node_id",
                placeholders.join(", ")
            );
            let mut q = sqlx::query(&sql).bind(tenant_id);
            for et in edge_types.unwrap_or(&[]) {
                q = q.bind(et.as_str());
            }
            q.fetch_all(&self.pool).await?
        } else {
            sqlx::query(
                "SELECT source_node_id, target_node_id, weight \
                 FROM graph_edges \
                 WHERE tenant_id = ?1 \
                 ORDER BY source_node_id, target_node_id",
            )
            .bind(tenant_id)
            .fetch_all(&self.pool)
            .await?
        };

        // 3. Convert to indexed edges; skip orphan endpoints.
        let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(edge_rows.len());
        for row in &edge_rows {
            let src: String = row.get(0);
            let tgt: String = row.get(1);
            let weight: f64 = row.get(2);
            let (Some(&si), Some(&ti)) = (index_map.get(src.as_str()), index_map.get(tgt.as_str()))
            else {
                // Orphan edge: at least one endpoint is absent from the node list.
                continue;
            };
            edges.push((si, ti, weight));
        }

        Ok(AdjacencyExport { node_ids, edges })
    }
}

impl SqliteGraphStore {
    async fn find_target_nodes(
        &self,
        tenant_id: &str,
        symbol_name: &str,
        file_path: Option<&str>,
        branch: Option<&str>,
    ) -> GraphDbResult<Vec<String>> {
        let scoped = !is_cross_branch(branch);
        match (file_path, scoped) {
            (Some(fp), true) => {
                let rows = sqlx::query(
                    "SELECT node_id FROM graph_nodes
                     WHERE tenant_id = ?1 AND symbol_name = ?2 AND file_path = ?3
                       AND EXISTS (SELECT 1 FROM json_each(branches) WHERE value = ?4)",
                )
                .bind(tenant_id)
                .bind(symbol_name)
                .bind(fp)
                .bind(branch.unwrap_or_default())
                .fetch_all(&self.pool)
                .await?;
                Ok(rows.iter().map(|r| r.get("node_id")).collect())
            }
            (Some(fp), false) => {
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
            }
            (None, true) => {
                let rows = sqlx::query(
                    "SELECT node_id FROM graph_nodes
                     WHERE tenant_id = ?1 AND symbol_name = ?2
                       AND EXISTS (SELECT 1 FROM json_each(branches) WHERE value = ?3)",
                )
                .bind(tenant_id)
                .bind(symbol_name)
                .bind(branch.unwrap_or_default())
                .fetch_all(&self.pool)
                .await?;
                Ok(rows.iter().map(|r| r.get("node_id")).collect())
            }
            (None, false) => {
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
    }

    async fn reverse_traverse(
        &self,
        tenant_id: &str,
        target_id: &str,
        branch: Option<&str>,
    ) -> GraphDbResult<Vec<ImpactNode>> {
        let scoped = !is_cross_branch(branch);
        let edge_br = if scoped {
            edge_branch_clause("e", 3)
        } else {
            String::new()
        };
        let node_br = if scoped {
            node_branch_clause("n", 3)
        } else {
            String::new()
        };
        let sql = format!(
            "WITH RECURSIVE reverse_traverse AS (
                SELECT e.source_node_id AS node_id, e.edge_type, 1 AS depth
                FROM graph_edges e
                WHERE e.target_node_id = ?1 AND e.tenant_id = ?2{edge_br}
                UNION ALL
                SELECT e.source_node_id, e.edge_type, rt.depth + 1
                FROM graph_edges e
                INNER JOIN reverse_traverse rt ON e.target_node_id = rt.node_id
                WHERE rt.depth < 3 AND e.tenant_id = ?2{edge_br}
            )
            SELECT DISTINCT rt.node_id, rt.edge_type, rt.depth,
                   n.symbol_name, n.file_path
            FROM reverse_traverse rt
            JOIN graph_nodes n ON rt.node_id = n.node_id
            WHERE 1=1{node_br}
            ORDER BY rt.depth, n.symbol_name"
        );
        let mut qb = sqlx::query(&sql).bind(target_id).bind(tenant_id);
        if scoped {
            qb = qb.bind(branch.unwrap_or_default());
        }
        let rows = qb.fetch_all(&self.pool).await?;
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
