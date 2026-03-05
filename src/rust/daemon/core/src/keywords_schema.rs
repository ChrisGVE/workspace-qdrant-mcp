//! Schema for keyword/tag extraction tables.
//!
//! Tables created in schema v16:
//! - `keywords`: per-document keyword records with scores
//! - `tags`: per-document tag records with diversity scoring
//! - `keyword_baskets`: keyword-to-tag assignments
//! - `canonical_tags`: deduplicated cross-document tag graph
//! - `tag_hierarchy_edges`: parent-child relationships between canonical tags

/// SQL to create the keywords table
pub const CREATE_KEYWORDS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS keywords (
    keyword_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    keyword TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    semantic_score REAL NOT NULL DEFAULT 0.0,
    lexical_score REAL NOT NULL DEFAULT 0.0,
    stability_count INTEGER NOT NULL DEFAULT 0,
    collection TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"#;

/// SQL to create the tags table
pub const CREATE_TAGS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    tag_type TEXT NOT NULL DEFAULT 'concept' CHECK (tag_type IN ('concept', 'structural')),
    score REAL NOT NULL DEFAULT 0.0,
    diversity_score REAL NOT NULL DEFAULT 0.0,
    basket_id INTEGER,
    collection TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"#;

/// SQL to create the keyword_baskets table
pub const CREATE_KEYWORD_BASKETS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS keyword_baskets (
    basket_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_id INTEGER NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
    keywords_json TEXT NOT NULL DEFAULT '[]',
    tenant_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"#;

/// SQL to create the canonical_tags table
pub const CREATE_CANONICAL_TAGS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS canonical_tags (
    canonical_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL,
    centroid_vector_id TEXT,
    level INTEGER NOT NULL DEFAULT 3,
    parent_id INTEGER REFERENCES canonical_tags(canonical_id) ON DELETE SET NULL,
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"#;

/// SQL to create the tag_hierarchy_edges table
pub const CREATE_TAG_HIERARCHY_EDGES_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS tag_hierarchy_edges (
    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_tag_id INTEGER NOT NULL REFERENCES canonical_tags(canonical_id) ON DELETE CASCADE,
    child_tag_id INTEGER NOT NULL REFERENCES canonical_tags(canonical_id) ON DELETE CASCADE,
    similarity_score REAL NOT NULL DEFAULT 0.0,
    tenant_id TEXT NOT NULL,
    UNIQUE(parent_tag_id, child_tag_id)
)
"#;

/// Indexes for keyword/tag tables
pub const CREATE_KEYWORDS_INDEXES_SQL: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_keywords_tenant_collection ON keywords(tenant_id, collection)",
    "CREATE INDEX IF NOT EXISTS idx_keywords_doc ON keywords(doc_id, collection)",
    "CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON keywords(keyword, tenant_id)",
];

pub const CREATE_TAGS_INDEXES_SQL: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_tags_tenant_collection ON tags(tenant_id, collection)",
    "CREATE INDEX IF NOT EXISTS idx_tags_doc ON tags(doc_id, collection)",
    "CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag, tenant_id)",
];

pub const CREATE_KEYWORD_BASKETS_INDEXES_SQL: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_keyword_baskets_tag ON keyword_baskets(tag_id)",
    "CREATE INDEX IF NOT EXISTS idx_keyword_baskets_tenant ON keyword_baskets(tenant_id)",
];

pub const CREATE_CANONICAL_TAGS_INDEXES_SQL: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_canonical_tags_tenant ON canonical_tags(tenant_id, collection)",
    "CREATE INDEX IF NOT EXISTS idx_canonical_tags_parent ON canonical_tags(parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_canonical_tags_name ON canonical_tags(canonical_name, tenant_id)",
];

pub const CREATE_TAG_HIERARCHY_EDGES_INDEXES_SQL: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_hierarchy_edges_parent ON tag_hierarchy_edges(parent_tag_id)",
    "CREATE INDEX IF NOT EXISTS idx_hierarchy_edges_child ON tag_hierarchy_edges(child_tag_id)",
    "CREATE INDEX IF NOT EXISTS idx_hierarchy_edges_tenant ON tag_hierarchy_edges(tenant_id)",
];

/// Keyword record
#[derive(Debug, Clone)]
pub struct KeywordRecord {
    pub keyword_id: i64,
    pub doc_id: String,
    pub keyword: String,
    pub score: f64,
    pub semantic_score: f64,
    pub lexical_score: f64,
    pub stability_count: i32,
    pub collection: String,
    pub tenant_id: String,
    pub created_at: String,
}

/// Tag record
#[derive(Debug, Clone)]
pub struct TagRecord {
    pub tag_id: i64,
    pub doc_id: String,
    pub tag: String,
    pub tag_type: String,
    pub score: f64,
    pub diversity_score: f64,
    pub basket_id: Option<i64>,
    pub collection: String,
    pub tenant_id: String,
    pub created_at: String,
}

/// Keyword basket record
#[derive(Debug, Clone)]
pub struct KeywordBasket {
    pub basket_id: i64,
    pub tag_id: i64,
    pub keywords_json: String,
    pub tenant_id: String,
    pub created_at: String,
}

/// Canonical tag record
#[derive(Debug, Clone)]
pub struct CanonicalTag {
    pub canonical_id: i64,
    pub canonical_name: String,
    pub centroid_vector_id: Option<String>,
    pub level: i32,
    pub parent_id: Option<i64>,
    pub tenant_id: String,
    pub collection: String,
    pub created_at: String,
}

/// Hierarchy edge record
#[derive(Debug, Clone)]
pub struct HierarchyEdge {
    pub edge_id: i64,
    pub parent_tag_id: i64,
    pub child_tag_id: i64,
    pub similarity_score: f64,
    pub tenant_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keywords_sql_is_valid() {
        assert!(CREATE_KEYWORDS_SQL.contains("CREATE TABLE"));
        assert!(CREATE_KEYWORDS_SQL.contains("keywords"));
        assert!(CREATE_KEYWORDS_SQL.contains("doc_id TEXT NOT NULL"));
        assert!(CREATE_KEYWORDS_SQL.contains("tenant_id TEXT NOT NULL"));
    }

    #[test]
    fn test_tags_sql_is_valid() {
        assert!(CREATE_TAGS_SQL.contains("CREATE TABLE"));
        assert!(CREATE_TAGS_SQL.contains("tags"));
        assert!(CREATE_TAGS_SQL.contains("tag_type TEXT NOT NULL"));
        assert!(CREATE_TAGS_SQL.contains("CHECK (tag_type IN"));
    }

    #[test]
    fn test_keyword_baskets_sql_has_fk() {
        assert!(CREATE_KEYWORD_BASKETS_SQL.contains("REFERENCES tags(tag_id)"));
        assert!(CREATE_KEYWORD_BASKETS_SQL.contains("ON DELETE CASCADE"));
    }

    #[test]
    fn test_canonical_tags_sql_has_self_ref() {
        assert!(CREATE_CANONICAL_TAGS_SQL.contains("REFERENCES canonical_tags(canonical_id)"));
        assert!(CREATE_CANONICAL_TAGS_SQL.contains("ON DELETE SET NULL"));
    }

    #[test]
    fn test_hierarchy_edges_sql_has_unique() {
        assert!(CREATE_TAG_HIERARCHY_EDGES_SQL.contains("UNIQUE(parent_tag_id, child_tag_id)"));
    }

    #[test]
    fn test_indexes_are_idempotent() {
        for index_sql in CREATE_KEYWORDS_INDEXES_SQL {
            assert!(
                index_sql.contains("IF NOT EXISTS"),
                "Missing IF NOT EXISTS in: {}",
                index_sql
            );
        }
        for index_sql in CREATE_TAGS_INDEXES_SQL {
            assert!(
                index_sql.contains("IF NOT EXISTS"),
                "Missing IF NOT EXISTS in: {}",
                index_sql
            );
        }
        for index_sql in CREATE_KEYWORD_BASKETS_INDEXES_SQL {
            assert!(
                index_sql.contains("IF NOT EXISTS"),
                "Missing IF NOT EXISTS in: {}",
                index_sql
            );
        }
        for index_sql in CREATE_CANONICAL_TAGS_INDEXES_SQL {
            assert!(
                index_sql.contains("IF NOT EXISTS"),
                "Missing IF NOT EXISTS in: {}",
                index_sql
            );
        }
        for index_sql in CREATE_TAG_HIERARCHY_EDGES_INDEXES_SQL {
            assert!(
                index_sql.contains("IF NOT EXISTS"),
                "Missing IF NOT EXISTS in: {}",
                index_sql
            );
        }
    }

    #[test]
    fn test_keywords_index_count() {
        assert_eq!(CREATE_KEYWORDS_INDEXES_SQL.len(), 3);
    }

    #[test]
    fn test_tags_index_count() {
        assert_eq!(CREATE_TAGS_INDEXES_SQL.len(), 3);
    }

    #[test]
    fn test_keyword_baskets_index_count() {
        assert_eq!(CREATE_KEYWORD_BASKETS_INDEXES_SQL.len(), 2);
    }

    #[test]
    fn test_canonical_tags_index_count() {
        assert_eq!(CREATE_CANONICAL_TAGS_INDEXES_SQL.len(), 3);
    }

    #[test]
    fn test_hierarchy_edges_index_count() {
        assert_eq!(CREATE_TAG_HIERARCHY_EDGES_INDEXES_SQL.len(), 3);
    }
}
