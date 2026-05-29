//! Tag and keyword-basket queries — reads from `tags`, `keyword_baskets`,
//! and `canonical_tags`.
//!
//! SQL is verbatim from `tag-queries.ts`.

use rusqlite::{params, Connection};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct MatchingTag {
    pub tag_id: i64,
    pub tag: String,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KeywordBasket {
    pub tag_id: i64,
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TagSummary {
    pub tag: String,
    pub doc_count: i64,
    pub avg_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalTagEntry {
    pub name: String,
    pub level: i64,
    pub parent_name: Option<String>,
    pub child_count: i64,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Tokenise a query string the same way as `tokenizeQuery` in `tag-queries.ts`.
///
/// Lowercases, splits on whitespace, strips non-`[a-z0-9_-]` chars, and
/// keeps only tokens with length ≥ 3.
fn tokenize_query(query: &str) -> Vec<String> {
    query
        .to_lowercase()
        .split_whitespace()
        .map(|t| {
            t.chars()
                .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
                .collect::<String>()
        })
        .filter(|t| t.len() >= 3)
        .collect()
}

fn is_no_such_table(e: &rusqlite::Error) -> bool {
    e.to_string().contains("no such table")
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Find tags matching query terms.
///
/// SQL verbatim from `tag-queries.ts:58-73`:
/// ```sql
/// SELECT DISTINCT t.tag_id, t.tag, t.score
/// FROM tags t
/// WHERE t.collection = ?
///   AND t.tag_type = 'concept'
///   AND (<LIKE conditions>)
///   [AND t.tenant_id = ?]
/// ORDER BY t.score DESC
/// LIMIT 10
/// ```
pub fn get_matching_tags(
    conn: Option<&Connection>,
    query: &str,
    collection: &str,
    tenant_id: Option<&str>,
) -> Vec<MatchingTag> {
    let Some(conn) = conn else {
        return Vec::new();
    };
    let tokens = tokenize_query(query);
    if tokens.is_empty() {
        return Vec::new();
    }

    let like_conditions: Vec<String> = tokens
        .iter()
        .map(|_| "LOWER(t.tag) LIKE ?".to_string())
        .collect();
    let tenant_clause = if tenant_id.is_some() {
        " AND t.tenant_id = ?"
    } else {
        ""
    };
    let sql = format!(
        "SELECT DISTINCT t.tag_id, t.tag, t.score \
         FROM tags t \
         WHERE t.collection = ? \
           AND t.tag_type = 'concept' \
           AND ({like}) \
           {tenant_clause} \
         ORDER BY t.score DESC \
         LIMIT 10",
        like = like_conditions.join(" OR "),
    );

    let result: Result<Vec<MatchingTag>, rusqlite::Error> = (|| {
        let mut stmt = conn.prepare(&sql)?;
        // Build params: collection, %token% for each token, optional tenant_id
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
            vec![Box::new(collection.to_string())];
        for t in &tokens {
            param_values.push(Box::new(format!("%{t}%")));
        }
        if let Some(tid) = tenant_id {
            param_values.push(Box::new(tid.to_string()));
        }
        let refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|b| b.as_ref()).collect();
        let rows: Vec<MatchingTag> = stmt
            .query_map(refs.as_slice(), |row| {
                Ok(MatchingTag {
                    tag_id: row.get(0)?,
                    tag: row.get(1)?,
                    score: row.get(2)?,
                })
            })?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    })();

    match result {
        Ok(v) => v,
        Err(e) if is_no_such_table(&e) => Vec::new(),
        Err(e) => {
            tracing::warn!("get_matching_tags failed: {e}");
            Vec::new()
        }
    }
}

/// Retrieve keyword baskets for a set of tag IDs.
///
/// SQL verbatim from `tag-queries.ts:96-99`:
/// ```sql
/// SELECT kb.tag_id, kb.keywords_json
/// FROM keyword_baskets kb
/// WHERE kb.tag_id IN (…)
/// ```
pub fn get_keyword_baskets_for_tags(
    conn: Option<&Connection>,
    tag_ids: &[i64],
) -> Vec<KeywordBasket> {
    let Some(conn) = conn else {
        return Vec::new();
    };
    if tag_ids.is_empty() {
        return Vec::new();
    }

    let placeholders: Vec<&str> = tag_ids.iter().map(|_| "?").collect();
    let sql = format!(
        "SELECT kb.tag_id, kb.keywords_json \
         FROM keyword_baskets kb \
         WHERE kb.tag_id IN ({})",
        placeholders.join(",")
    );

    let result: Result<Vec<KeywordBasket>, rusqlite::Error> = (|| {
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::types::ToSql> = tag_ids
            .iter()
            .map(|id| id as &dyn rusqlite::types::ToSql)
            .collect();
        let pairs: Vec<(i64, String)> = stmt
            .query_map(params.as_slice(), |row| {
                let tag_id: i64 = row.get(0)?;
                let keywords_json: String = row.get(1)?;
                Ok((tag_id, keywords_json))
            })?
            .collect::<Result<_, _>>()?;
        let rows: Vec<KeywordBasket> = pairs
            .into_iter()
            .map(|(tag_id, json)| {
                let keywords: Vec<String> = serde_json::from_str(&json).unwrap_or_default();
                KeywordBasket { tag_id, keywords }
            })
            .collect();
        Ok(rows)
    })();

    match result {
        Ok(v) => v,
        Err(e) if is_no_such_table(&e) => Vec::new(),
        Err(e) => {
            tracing::warn!("get_keyword_baskets_for_tags failed: {e}");
            Vec::new()
        }
    }
}

/// List concept tags for a collection.
///
/// SQL verbatim from `tag-queries.ts:136-153`:
/// ```sql
/// SELECT t.tag,
///        COUNT(DISTINCT t.doc_id) as doc_count,
///        ROUND(AVG(t.score), 4) as avg_score
/// FROM tags t
/// WHERE t.collection = ?
///   AND t.tag_type = 'concept'
///   [AND t.tenant_id = ?]
/// GROUP BY t.tag
/// ORDER BY doc_count DESC, avg_score DESC
/// LIMIT ?
/// ```
pub fn list_tags(
    conn: Option<&Connection>,
    collection: &str,
    tenant_id: Option<&str>,
    limit: usize,
) -> Vec<TagSummary> {
    let Some(conn) = conn else {
        return Vec::new();
    };
    let tenant_clause = if tenant_id.is_some() {
        " AND t.tenant_id = ?"
    } else {
        ""
    };
    let sql = format!(
        "SELECT t.tag, \
                COUNT(DISTINCT t.doc_id) as doc_count, \
                ROUND(AVG(t.score), 4) as avg_score \
         FROM tags t \
         WHERE t.collection = ? \
           AND t.tag_type = 'concept' \
           {tenant_clause} \
         GROUP BY t.tag \
         ORDER BY doc_count DESC, avg_score DESC \
         LIMIT ?",
    );

    let result: Result<Vec<TagSummary>, rusqlite::Error> = (|| {
        let mut stmt = conn.prepare(&sql)?;
        let limit_i64 = limit as i64;
        let rows: Vec<TagSummary> = if let Some(tid) = tenant_id {
            stmt.query_map(params![collection, tid, limit_i64], |row| {
                Ok(TagSummary {
                    tag: row.get(0)?,
                    doc_count: row.get(1)?,
                    avg_score: row.get(2)?,
                })
            })?
            .collect::<Result<_, _>>()?
        } else {
            stmt.query_map(params![collection, limit_i64], |row| {
                Ok(TagSummary {
                    tag: row.get(0)?,
                    doc_count: row.get(1)?,
                    avg_score: row.get(2)?,
                })
            })?
            .collect::<Result<_, _>>()?
        };
        Ok(rows)
    })();

    match result {
        Ok(v) => v,
        Err(e) if is_no_such_table(&e) => Vec::new(),
        Err(e) => {
            tracing::warn!("list_tags failed: {e}");
            Vec::new()
        }
    }
}

/// Get the canonical tag hierarchy for a collection.
///
/// SQL verbatim from `tag-queries.ts:176-195`.
pub fn get_tag_hierarchy(
    conn: Option<&Connection>,
    collection: &str,
    tenant_id: Option<&str>,
) -> Vec<CanonicalTagEntry> {
    let Some(conn) = conn else {
        return Vec::new();
    };
    let tenant_clause = if tenant_id.is_some() {
        " AND ct.tenant_id = ?"
    } else {
        ""
    };
    let sql = format!(
        "SELECT ct.canonical_name, \
                ct.level, \
                parent.canonical_name as parent_name, \
                (SELECT COUNT(*) FROM canonical_tags child \
                 WHERE child.parent_id = ct.canonical_id) as child_count \
         FROM canonical_tags ct \
         LEFT JOIN canonical_tags parent ON ct.parent_id = parent.canonical_id \
         WHERE ct.collection = ? \
           {tenant_clause} \
         ORDER BY ct.level ASC, ct.canonical_name ASC",
    );

    let result: Result<Vec<CanonicalTagEntry>, rusqlite::Error> = (|| {
        let mut stmt = conn.prepare(&sql)?;
        let rows: Vec<CanonicalTagEntry> = if let Some(tid) = tenant_id {
            stmt.query_map(params![collection, tid], |row| {
                Ok(CanonicalTagEntry {
                    name: row.get(0)?,
                    level: row.get(1)?,
                    parent_name: row.get(2)?,
                    child_count: row.get(3)?,
                })
            })?
            .collect::<Result<_, _>>()?
        } else {
            stmt.query_map(params![collection], |row| {
                Ok(CanonicalTagEntry {
                    name: row.get(0)?,
                    level: row.get(1)?,
                    parent_name: row.get(2)?,
                    child_count: row.get(3)?,
                })
            })?
            .collect::<Result<_, _>>()?
        };
        Ok(rows)
    })();

    match result {
        Ok(v) => v,
        Err(e) if is_no_such_table(&e) => Vec::new(),
        Err(e) => {
            tracing::warn!("get_tag_hierarchy failed: {e}");
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "tag_queries_tests.rs"]
mod tests;
