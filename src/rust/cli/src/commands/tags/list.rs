//! List subcommand handlers: `tags list` and `tags keywords`.

use anyhow::Result;
use tabled::Tabled;

use crate::output;
use super::db::{open_db, table_exists};

#[derive(Tabled, serde::Serialize)]
pub(super) struct TagRow {
    #[tabled(rename = "Tag")]
    pub tag: String,
    #[tabled(rename = "Type")]
    pub tag_type: String,
    #[tabled(rename = "Score")]
    pub score: String,
    #[tabled(rename = "Diversity")]
    pub diversity: String,
}

pub(super) fn list_tags(
    doc_id: &str,
    tag_type: Option<&str>,
    json: bool,
    script: bool,
    no_headers: bool,
) -> Result<()> {
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
    let params_refs: Vec<&dyn rusqlite::types::ToSql> =
        params.iter().map(|p| p.as_ref()).collect();
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
    } else if script {
        output::print_script(&rows, !no_headers);
    } else {
        output::info(format!("Tags for document {} ({} total)", doc_id, rows.len()));
        output::print_table(&rows);
    }

    Ok(())
}

#[derive(Tabled, serde::Serialize)]
pub(super) struct KeywordRow {
    #[tabled(rename = "Keyword")]
    pub keyword: String,
    #[tabled(rename = "Score")]
    pub score: String,
    #[tabled(rename = "Semantic")]
    pub semantic: String,
    #[tabled(rename = "Lexical")]
    pub lexical: String,
    #[tabled(rename = "Stability")]
    pub stability: i32,
}

pub(super) fn list_keywords(
    doc_id: &str,
    json: bool,
    script: bool,
    no_headers: bool,
) -> Result<()> {
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
    } else if script {
        output::print_script(&rows, !no_headers);
    } else {
        output::info(format!(
            "Keywords for document {} ({} total)",
            doc_id,
            rows.len()
        ));
        output::print_table(&rows);
    }

    Ok(())
}
