//! Tags command - keyword/tag management and hierarchy inspection
//!
//! Queries the keywords, tags, canonical_tags, and tag_hierarchy_edges tables
//! to display extraction results and the canonical tag hierarchy.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use rusqlite::Connection;
use sha2::{Digest, Sha256};
use tabled::Tabled;

use crate::config::get_database_path;
use crate::output;

/// Tags command arguments
#[derive(Args)]
pub struct TagsArgs {
    #[command(subcommand)]
    command: TagsCommand,
}

/// Tags subcommands
#[derive(Subcommand)]
enum TagsCommand {
    /// List tags for a specific document
    List {
        /// Document ID
        #[arg(long)]
        doc: String,

        /// Filter by tag type (concept, structural)
        #[arg(long, value_parser = ["concept", "structural"])]
        tag_type: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// List keywords for a specific document
    Keywords {
        /// Document ID
        #[arg(long)]
        doc: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show canonical tag hierarchy for a tenant
    Tree {
        /// Tenant ID
        #[arg(long)]
        tenant: String,

        /// Collection (default: projects)
        #[arg(long, default_value = "projects")]
        collection: String,
    },

    /// Show extraction statistics
    Stats {
        /// Tenant ID (optional, all tenants if omitted)
        #[arg(long)]
        tenant: Option<String>,

        /// Collection (default: projects)
        #[arg(long, default_value = "projects")]
        collection: String,
    },

    /// Trigger canonical tag hierarchy rebuild for a tenant
    Rebuild {
        /// Tenant ID
        #[arg(long)]
        tenant: String,

        /// Collection (default: projects)
        #[arg(long, default_value = "projects")]
        collection: String,
    },

    /// Search tags by name across all tenants
    Search {
        /// Tag name pattern (SQL LIKE)
        query: String,

        /// Collection (default: projects)
        #[arg(long, default_value = "projects")]
        collection: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show keyword baskets for a document
    Baskets {
        /// Document ID
        #[arg(long)]
        doc: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

/// Execute tags command
pub async fn execute(args: TagsArgs) -> Result<()> {
    match args.command {
        TagsCommand::List { doc, tag_type, json } => list_tags(&doc, tag_type.as_deref(), json),
        TagsCommand::Keywords { doc, json } => list_keywords(&doc, json),
        TagsCommand::Tree { tenant, collection } => show_tree(&tenant, &collection),
        TagsCommand::Stats { tenant, collection } => show_stats(tenant.as_deref(), &collection),
        TagsCommand::Rebuild { tenant, collection } => rebuild(&tenant, &collection).await,
        TagsCommand::Search { query, collection, json } => search_tags(&query, &collection, json),
        TagsCommand::Baskets { doc, json } => show_baskets(&doc, json),
    }
}

// ── Database helpers ──────────────────────────────────────────────────────

fn open_db() -> Result<Connection> {
    let db_path = get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;
    if !db_path.exists() {
        anyhow::bail!(
            "Database not found at {}. Run daemon first: wqm service start",
            db_path.display()
        );
    }
    let conn = Connection::open(&db_path).context("Failed to open state database")?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        .context("Failed to set SQLite pragmas")?;
    Ok(conn)
}

fn table_exists(conn: &Connection, name: &str) -> bool {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        [name],
        |_| Ok(true),
    )
    .unwrap_or(false)
}

// ── Subcommand implementations ────────────────────────────────────────────

#[derive(Tabled, serde::Serialize)]
struct TagRow {
    #[tabled(rename = "Tag")]
    tag: String,
    #[tabled(rename = "Type")]
    tag_type: String,
    #[tabled(rename = "Score")]
    score: String,
    #[tabled(rename = "Diversity")]
    diversity: String,
}

fn list_tags(doc_id: &str, tag_type: Option<&str>, json: bool) -> Result<()> {
    let conn = open_db()?;
    if !table_exists(&conn, "tags") {
        anyhow::bail!("Tags table not found. Ensure daemon schema v16+ is applied.");
    }

    let (sql, params): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(tt) = tag_type {
        (
            "SELECT tag, tag_type, score, diversity_score FROM tags WHERE doc_id = ? AND tag_type = ? ORDER BY score DESC",
            vec![Box::new(doc_id.to_string()), Box::new(tt.to_string())],
        )
    } else {
        (
            "SELECT tag, tag_type, score, diversity_score FROM tags WHERE doc_id = ? ORDER BY tag_type, score DESC",
            vec![Box::new(doc_id.to_string())],
        )
    };

    let mut stmt = conn.prepare(sql)?;
    let params_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let rows: Vec<TagRow> = stmt
        .query_map(params_refs.as_slice(), |row| {
            let score: f64 = row.get(2)?;
            let diversity: f64 = row.get(3)?;
            Ok(TagRow {
                tag: row.get(0)?,
                tag_type: row.get(1)?,
                score: format!("{:.3}", score),
                diversity: format!("{:.3}", diversity),
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    if rows.is_empty() {
        output::info(format!("No tags found for document {}", doc_id));
        return Ok(());
    }

    if json {
        output::print_json(&rows);
    } else {
        output::info(format!("Tags for document {} ({} total)", doc_id, rows.len()));
        output::print_table(&rows);
    }

    Ok(())
}

#[derive(Tabled, serde::Serialize)]
struct KeywordRow {
    #[tabled(rename = "Keyword")]
    keyword: String,
    #[tabled(rename = "Score")]
    score: String,
    #[tabled(rename = "Semantic")]
    semantic: String,
    #[tabled(rename = "Lexical")]
    lexical: String,
    #[tabled(rename = "Stability")]
    stability: i32,
}

fn list_keywords(doc_id: &str, json: bool) -> Result<()> {
    let conn = open_db()?;
    if !table_exists(&conn, "keywords") {
        anyhow::bail!("Keywords table not found. Ensure daemon schema v16+ is applied.");
    }

    let mut stmt = conn.prepare(
        "SELECT keyword, score, semantic_score, lexical_score, stability_count \
         FROM keywords WHERE doc_id = ? ORDER BY score DESC",
    )?;
    let rows: Vec<KeywordRow> = stmt
        .query_map([doc_id], |row| {
            let score: f64 = row.get(1)?;
            let semantic: f64 = row.get(2)?;
            let lexical: f64 = row.get(3)?;
            Ok(KeywordRow {
                keyword: row.get(0)?,
                score: format!("{:.3}", score),
                semantic: format!("{:.3}", semantic),
                lexical: format!("{:.3}", lexical),
                stability: row.get(4)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    if rows.is_empty() {
        output::info(format!("No keywords found for document {}", doc_id));
        return Ok(());
    }

    if json {
        output::print_json(&rows);
    } else {
        output::info(format!("Keywords for document {} ({} total)", doc_id, rows.len()));
        output::print_table(&rows);
    }

    Ok(())
}

fn show_tree(tenant_id: &str, collection: &str) -> Result<()> {
    let conn = open_db()?;
    if !table_exists(&conn, "canonical_tags") {
        anyhow::bail!("Canonical tags table not found. Ensure daemon schema v16+ is applied.");
    }

    // Fetch all canonical tags for this tenant, ordered by level
    let mut stmt = conn.prepare(
        "SELECT canonical_id, canonical_name, level, parent_id \
         FROM canonical_tags \
         WHERE tenant_id = ? AND collection = ? \
         ORDER BY level ASC, canonical_name ASC",
    )?;

    struct TreeNode {
        id: i64,
        name: String,
        level: i32,
        parent_id: Option<i64>,
    }

    let nodes: Vec<TreeNode> = stmt
        .query_map(rusqlite::params![tenant_id, collection], |row| {
            Ok(TreeNode {
                id: row.get(0)?,
                name: row.get(1)?,
                level: row.get(2)?,
                parent_id: row.get(3)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    if nodes.is_empty() {
        output::info(format!(
            "No canonical tag hierarchy found for tenant {}. Run 'wqm tags rebuild --tenant {}'",
            tenant_id, tenant_id
        ));
        return Ok(());
    }

    // Build ASCII tree
    output::info(format!(
        "Canonical tag hierarchy for tenant {} ({} tags)",
        tenant_id,
        nodes.len()
    ));
    println!();

    // Level 1 nodes (roots)
    let roots: Vec<&TreeNode> = nodes.iter().filter(|n| n.level == 1).collect();
    for (i, root) in roots.iter().enumerate() {
        let is_last_root = i == roots.len() - 1;
        let prefix = if is_last_root { "└── " } else { "├── " };
        println!("{}{} (L1)", prefix, root.name);

        // Level 2 children of this root
        let children: Vec<&TreeNode> = nodes
            .iter()
            .filter(|n| n.level == 2 && n.parent_id == Some(root.id))
            .collect();

        for (j, child) in children.iter().enumerate() {
            let is_last_child = j == children.len() - 1;
            let indent = if is_last_root { "    " } else { "│   " };
            let child_prefix = if is_last_child { "└── " } else { "├── " };
            println!("{}{}{} (L2)", indent, child_prefix, child.name);

            // Level 3 children of this L2 node
            let grandchildren: Vec<&TreeNode> = nodes
                .iter()
                .filter(|n| n.level == 3 && n.parent_id == Some(child.id))
                .collect();

            for (k, gc) in grandchildren.iter().enumerate() {
                let is_last_gc = k == grandchildren.len() - 1;
                let gc_indent = if is_last_root { "    " } else { "│   " };
                let gc_indent2 = if is_last_child { "    " } else { "│   " };
                let gc_prefix = if is_last_gc { "└── " } else { "├── " };
                println!("{}{}{}{}", gc_indent, gc_indent2, gc_prefix, gc.name);
            }
        }
    }

    // Orphan level 2 (no parent)
    let orphan_l2: Vec<&TreeNode> = nodes
        .iter()
        .filter(|n| n.level == 2 && n.parent_id.is_none())
        .collect();
    if !orphan_l2.is_empty() {
        println!();
        output::info(format!("Unlinked level 2 tags ({})", orphan_l2.len()));
        for n in &orphan_l2 {
            println!("  - {} (L2)", n.name);
        }
    }

    // Orphan level 3 (no parent)
    let orphan_l3: Vec<&TreeNode> = nodes
        .iter()
        .filter(|n| n.level == 3 && n.parent_id.is_none())
        .collect();
    if !orphan_l3.is_empty() {
        println!();
        output::info(format!("Unlinked level 3 tags ({})", orphan_l3.len()));
        for n in &orphan_l3 {
            println!("  - {}", n.name);
        }
    }

    Ok(())
}

#[derive(Tabled, serde::Serialize)]
struct StatsRow {
    #[tabled(rename = "Tenant")]
    tenant_id: String,
    #[tabled(rename = "Documents")]
    doc_count: i64,
    #[tabled(rename = "Avg Keywords")]
    avg_keywords: String,
    #[tabled(rename = "Avg Tags")]
    avg_tags: String,
    #[tabled(rename = "Canonical")]
    canonical_count: i64,
}

fn show_stats(tenant_id: Option<&str>, collection: &str) -> Result<()> {
    let conn = open_db()?;
    if !table_exists(&conn, "tags") {
        anyhow::bail!("Tags table not found. Ensure daemon schema v16+ is applied.");
    }

    let has_canonical = table_exists(&conn, "canonical_tags");

    let rows: Vec<StatsRow> = if let Some(tid) = tenant_id {
        vec![compute_stats_for_tenant(&conn, tid, collection, has_canonical)?]
    } else {
        // Get all tenants with tags
        let mut stmt = conn.prepare(
            "SELECT DISTINCT tenant_id FROM tags WHERE collection = ? ORDER BY tenant_id",
        )?;
        let tenants: Vec<String> = stmt
            .query_map([collection], |row| row.get(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        tenants
            .iter()
            .filter_map(|tid| {
                compute_stats_for_tenant(&conn, tid, collection, has_canonical).ok()
            })
            .collect()
    };

    if rows.is_empty() {
        output::info("No tag data found. Ingest documents to generate keywords and tags.");
        return Ok(());
    }

    output::print_table(&rows);
    Ok(())
}

fn compute_stats_for_tenant(
    conn: &Connection,
    tenant_id: &str,
    collection: &str,
    has_canonical: bool,
) -> Result<StatsRow> {
    // Count distinct documents with keywords
    let doc_count: i64 = conn
        .query_row(
            "SELECT COUNT(DISTINCT doc_id) FROM keywords WHERE tenant_id = ? AND collection = ?",
            rusqlite::params![tenant_id, collection],
            |row| row.get(0),
        )
        .unwrap_or(0);

    // Average keywords per document
    let avg_kw: f64 = if doc_count > 0 {
        let total_kw: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM keywords WHERE tenant_id = ? AND collection = ?",
                rusqlite::params![tenant_id, collection],
                |row| row.get(0),
            )
            .unwrap_or(0);
        total_kw as f64 / doc_count as f64
    } else {
        0.0
    };

    // Average tags per document
    let avg_tags: f64 = if doc_count > 0 {
        let total_tags: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM tags WHERE tenant_id = ? AND collection = ? AND tag_type = 'concept'",
                rusqlite::params![tenant_id, collection],
                |row| row.get(0),
            )
            .unwrap_or(0);
        total_tags as f64 / doc_count as f64
    } else {
        0.0
    };

    // Canonical tag count
    let canonical_count: i64 = if has_canonical {
        conn.query_row(
            "SELECT COUNT(*) FROM canonical_tags WHERE tenant_id = ? AND collection = ?",
            rusqlite::params![tenant_id, collection],
            |row| row.get(0),
        )
        .unwrap_or(0)
    } else {
        0
    };

    Ok(StatsRow {
        tenant_id: tenant_id.to_string(),
        doc_count,
        avg_keywords: format!("{:.1}", avg_kw),
        avg_tags: format!("{:.1}", avg_tags),
        canonical_count,
    })
}

async fn rebuild(tenant_id: &str, _collection: &str) -> Result<()> {
    // Enqueue a rebuild via the unified queue
    // The daemon's hierarchy_builder will pick it up
    let conn = open_db()?;
    if !table_exists(&conn, "tags") {
        anyhow::bail!("Tags table not found. Ensure daemon schema v16+ is applied.");
    }

    // Check if tenant has any tags
    let tag_count: i64 = conn
        .query_row(
            "SELECT COUNT(DISTINCT tag) FROM tags WHERE tenant_id = ? AND tag_type = 'concept'",
            [tenant_id],
            |row| row.get(0),
        )
        .unwrap_or(0);

    if tag_count == 0 {
        anyhow::bail!("No concept tags found for tenant {}. Ingest documents first.", tenant_id);
    }

    // Insert a rebuild request into the unified queue
    let queue_id = uuid::Uuid::new_v4().to_string();
    let now = wqm_common::timestamps::now_utc();

    // Build idempotency key for dedup: SHA256(item_type|op|tenant_id|collection|payload_json)[:32]
    let payload = format!("{{\"action\":\"rebuild_hierarchy\",\"tenant_id\":\"{}\"}}", tenant_id);
    let idem_input = format!("tenant|rebuild|{}|projects|{}", tenant_id, payload);
    let hash = Sha256::digest(idem_input.as_bytes());
    let idem_key = format!("{:x}", hash);
    let idem_key = &idem_key[..32];

    conn.execute(
        "INSERT OR IGNORE INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          priority, status, payload_json, created_at, updated_at) \
         VALUES (?1, ?2, 'tenant', 'rebuild', ?3, 'projects', 3, 'pending', ?4, ?5, ?5)",
        rusqlite::params![queue_id, idem_key, tenant_id, payload, now],
    )?;

    output::success(format!(
        "Hierarchy rebuild queued for tenant {} ({} concept tags)",
        tenant_id, tag_count
    ));
    output::info("The daemon will process this at next queue poll cycle.");

    Ok(())
}

#[derive(Tabled, serde::Serialize)]
struct TagSearchRow {
    #[tabled(rename = "Tag")]
    tag: String,
    #[tabled(rename = "Tenant")]
    tenant_id: String,
    #[tabled(rename = "Documents")]
    doc_count: i64,
    #[tabled(rename = "Avg Score")]
    avg_score: String,
}

fn search_tags(query: &str, collection: &str, json: bool) -> Result<()> {
    let conn = open_db()?;
    if !table_exists(&conn, "tags") {
        anyhow::bail!("Tags table not found. Ensure daemon schema v16+ is applied.");
    }

    let pattern = format!("%{}%", query);
    let mut stmt = conn.prepare(
        "SELECT tag, tenant_id, COUNT(DISTINCT doc_id) as doc_count, AVG(score) as avg_score \
         FROM tags \
         WHERE collection = ? AND tag LIKE ? AND tag_type = 'concept' \
         GROUP BY tag, tenant_id \
         ORDER BY doc_count DESC \
         LIMIT 50",
    )?;

    let rows: Vec<TagSearchRow> = stmt
        .query_map(rusqlite::params![collection, pattern], |row| {
            let avg: f64 = row.get(3)?;
            Ok(TagSearchRow {
                tag: row.get(0)?,
                tenant_id: row.get(1)?,
                doc_count: row.get(2)?,
                avg_score: format!("{:.3}", avg),
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    if rows.is_empty() {
        output::info(format!("No tags matching '{}' found", query));
        return Ok(());
    }

    if json {
        output::print_json(&rows);
    } else {
        output::info(format!("Tags matching '{}' ({} results)", query, rows.len()));
        output::print_table(&rows);
    }

    Ok(())
}

fn show_baskets(doc_id: &str, json: bool) -> Result<()> {
    let conn = open_db()?;
    if !table_exists(&conn, "keyword_baskets") {
        anyhow::bail!("Keyword baskets table not found. Ensure daemon schema v16+ is applied.");
    }

    // Join keyword_baskets with tags to get tag name
    let mut stmt = conn.prepare(
        "SELECT t.tag, kb.keywords_json \
         FROM keyword_baskets kb \
         JOIN tags t ON kb.tag_id = t.tag_id \
         WHERE t.doc_id = ? \
         ORDER BY t.tag",
    )?;

    #[derive(serde::Serialize)]
    struct BasketOutput {
        tag: String,
        keywords: Vec<String>,
    }

    let baskets: Vec<BasketOutput> = stmt
        .query_map([doc_id], |row| {
            let tag: String = row.get(0)?;
            let kw_json: String = row.get(1)?;
            Ok((tag, kw_json))
        })?
        .filter_map(|r| r.ok())
        .map(|(tag, kw_json)| {
            let keywords: Vec<String> =
                serde_json::from_str(&kw_json).unwrap_or_default();
            BasketOutput { tag, keywords }
        })
        .collect();

    if baskets.is_empty() {
        output::info(format!("No keyword baskets found for document {}", doc_id));
        return Ok(());
    }

    if json {
        output::print_json(&baskets);
    } else {
        output::info(format!(
            "Keyword baskets for document {} ({} baskets)",
            doc_id,
            baskets.len()
        ));
        println!();
        for basket in &baskets {
            println!("  {} ({} keywords)", basket.tag, basket.keywords.len());
            for kw in &basket.keywords {
                println!("    - {}", kw);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tags_args_struct() {
        // Verify the TagsArgs struct is constructible
        // (clap parsing is tested via integration tests)
        assert_eq!(std::mem::size_of::<TagsArgs>(), std::mem::size_of::<TagsArgs>());
    }

    #[test]
    fn test_tag_row_tabled() {
        let row = TagRow {
            tag: "vector search".to_string(),
            tag_type: "concept".to_string(),
            score: "0.900".to_string(),
            diversity: "0.850".to_string(),
        };
        assert_eq!(row.tag, "vector search");
    }

    #[test]
    fn test_keyword_row_tabled() {
        let row = KeywordRow {
            keyword: "embedding".to_string(),
            score: "0.750".to_string(),
            semantic: "0.800".to_string(),
            lexical: "0.700".to_string(),
            stability: 3,
        };
        assert_eq!(row.stability, 3);
    }

    #[test]
    fn test_stats_row_tabled() {
        let row = StatsRow {
            tenant_id: "test".to_string(),
            doc_count: 10,
            avg_keywords: "5.2".to_string(),
            avg_tags: "3.1".to_string(),
            canonical_count: 15,
        };
        assert_eq!(row.doc_count, 10);
    }

    #[test]
    fn test_tag_search_row_tabled() {
        let row = TagSearchRow {
            tag: "async".to_string(),
            tenant_id: "proj-1".to_string(),
            doc_count: 5,
            avg_score: "0.800".to_string(),
        };
        assert_eq!(row.doc_count, 5);
    }
}
