//! Search and baskets subcommand handlers: `tags search` and `tags baskets`.

use anyhow::Result;
use tabled::Tabled;

use crate::output;
use super::db::{open_db, table_exists};

#[derive(Tabled, serde::Serialize)]
pub(super) struct TagSearchRow {
    #[tabled(rename = "Tag")]
    pub tag: String,
    #[tabled(rename = "Tenant")]
    pub tenant_id: String,
    #[tabled(rename = "Documents")]
    pub doc_count: i64,
    #[tabled(rename = "Avg Score")]
    pub avg_score: String,
}

pub(super) fn search_tags(
    query: &str,
    collection: &str,
    json: bool,
    script: bool,
    no_headers: bool,
) -> Result<()> {
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
    } else if script {
        output::print_script(&rows, !no_headers);
    } else {
        output::info(format!("Tags matching '{}' ({} results)", query, rows.len()));
        output::print_table(&rows);
    }

    Ok(())
}

#[derive(serde::Serialize)]
struct BasketOutput {
    tag: String,
    keywords: Vec<String>,
}

pub(super) fn show_baskets(
    doc_id: &str,
    json: bool,
    script: bool,
    no_headers: bool,
) -> Result<()> {
    let conn = open_db()?;
    if !table_exists(&conn, "keyword_baskets") {
        anyhow::bail!("Keyword baskets table not found. Ensure daemon schema v16+ is applied.");
    }

    let mut stmt = conn.prepare(
        "SELECT t.tag, kb.keywords_json \
         FROM keyword_baskets kb \
         JOIN tags t ON kb.tag_id = t.tag_id \
         WHERE t.doc_id = ? \
         ORDER BY t.tag",
    )?;

    let baskets: Vec<BasketOutput> = stmt
        .query_map([doc_id], |row| {
            let tag: String = row.get(0)?;
            let kw_json: String = row.get(1)?;
            Ok((tag, kw_json))
        })?
        .filter_map(|r| r.ok())
        .map(|(tag, kw_json)| {
            let keywords: Vec<String> = serde_json::from_str(&kw_json).unwrap_or_default();
            BasketOutput { tag, keywords }
        })
        .collect();

    if baskets.is_empty() {
        output::info(format!("No keyword baskets found for document {}", doc_id));
        return Ok(());
    }

    if json {
        output::print_json(&baskets);
    } else if script {
        #[derive(Tabled)]
        struct BasketRow {
            #[tabled(rename = "Tag")]
            tag: String,
            #[tabled(rename = "Keywords")]
            keywords: String,
            #[tabled(rename = "Count")]
            count: usize,
        }
        let rows: Vec<BasketRow> = baskets
            .iter()
            .map(|b| BasketRow {
                tag: b.tag.clone(),
                keywords: b.keywords.join(","),
                count: b.keywords.len(),
            })
            .collect();
        output::print_script(&rows, !no_headers);
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
