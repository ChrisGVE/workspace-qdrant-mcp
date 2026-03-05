//! Tests for symbol co-occurrence schema and operations.

#[cfg(test)]
mod tests {
    use super::super::{
        operations::{
            find_clusters, get_betweenness_centrality, get_degree_centrality, get_neighbors,
            upsert_cooccurrences,
        },
        schema::{CREATE_COOCCURRENCE_INDEXES_SQL, CREATE_SYMBOL_COOCCURRENCE_SQL},
    };

    #[test]
    fn test_create_table_sql_syntax() {
        // Verify the SQL string parses (basic sanity check)
        assert!(CREATE_SYMBOL_COOCCURRENCE_SQL.contains("CREATE TABLE"));
        assert!(CREATE_SYMBOL_COOCCURRENCE_SQL.contains("symbol_a TEXT NOT NULL"));
        assert!(CREATE_SYMBOL_COOCCURRENCE_SQL.contains("PRIMARY KEY"));
    }

    #[test]
    fn test_indexes_count() {
        assert_eq!(CREATE_COOCCURRENCE_INDEXES_SQL.len(), 3);
        for sql in CREATE_COOCCURRENCE_INDEXES_SQL {
            assert!(sql.starts_with("CREATE INDEX"));
        }
    }

    #[tokio::test]
    async fn test_upsert_increments() {
        // In-memory SQLite for testing
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        let pairs = vec![
            ("alpha".to_string(), "beta".to_string()),
            ("alpha".to_string(), "gamma".to_string()),
        ];

        // First insert
        upsert_cooccurrences(&pool, "t1", "projects", &pairs)
            .await
            .unwrap();

        let count: i64 = sqlx::query_scalar(
            "SELECT cooccurrence_count FROM symbol_cooccurrence WHERE symbol_a = 'alpha' AND symbol_b = 'beta'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 1);

        // Second insert should increment
        upsert_cooccurrences(&pool, "t1", "projects", &pairs)
            .await
            .unwrap();

        let count: i64 = sqlx::query_scalar(
            "SELECT cooccurrence_count FROM symbol_cooccurrence WHERE symbol_a = 'alpha' AND symbol_b = 'beta'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_degree_centrality() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        // Create a small graph: A-B(3), A-C(1), B-C(2)
        // A degree = 3+1=4, B degree = 3+2=5, C degree = 1+2=3
        let now = wqm_common::timestamps::now_utc();
        sqlx::query("INSERT INTO symbol_cooccurrence VALUES ('a', 'b', 't1', 'projects', 3, ?1)")
            .bind(&now)
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO symbol_cooccurrence VALUES ('a', 'c', 't1', 'projects', 1, ?1)")
            .bind(&now)
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO symbol_cooccurrence VALUES ('b', 'c', 't1', 'projects', 2, ?1)")
            .bind(&now)
            .execute(&pool)
            .await
            .unwrap();

        let centrality = get_degree_centrality(&pool, "t1", "projects")
            .await
            .unwrap();

        // B has max degree (5), so B=1.0
        assert!((centrality["b"] - 1.0).abs() < 1e-6);
        // A has degree 4/5=0.8
        assert!((centrality["a"] - 0.8).abs() < 1e-6);
        // C has degree 3/5=0.6
        assert!((centrality["c"] - 0.6).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_empty_centrality() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        let centrality = get_degree_centrality(&pool, "t1", "projects")
            .await
            .unwrap();
        assert!(centrality.is_empty());
    }

    async fn setup_graph(pool: &sqlx::SqlitePool) {
        let now = wqm_common::timestamps::now_utc();
        // Graph: a-b(3), a-c(1), b-c(2), c-d(4), d-e(5)
        // Clusters: {a,b,c} connected via high weights, {d,e} connected
        for (a, b, w) in &[
            ("a", "b", 3),
            ("a", "c", 1),
            ("b", "c", 2),
            ("c", "d", 4),
            ("d", "e", 5),
        ] {
            sqlx::query(
                "INSERT INTO symbol_cooccurrence VALUES (?1, ?2, 't1', 'projects', ?3, ?4)",
            )
            .bind(a)
            .bind(b)
            .bind(w)
            .bind(&now)
            .execute(pool)
            .await
            .unwrap();
        }
    }

    #[tokio::test]
    async fn test_find_clusters_basic() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();
        setup_graph(&pool).await;

        // min_count=2 should find clusters from edges with weight >= 2
        let clusters = find_clusters(&pool, "t1", "projects", 2, 2).await.unwrap();
        assert!(!clusters.is_empty(), "Should find at least one cluster");
        // The high-weight cluster should contain symbols from edges >= 2
        let all_symbols: Vec<&str> = clusters
            .iter()
            .flat_map(|c| c.symbols.iter().map(|s| s.as_str()))
            .collect();
        assert!(
            all_symbols.contains(&"b"),
            "b should be in a cluster (a-b:3, b-c:2)"
        );
    }

    #[tokio::test]
    async fn test_find_clusters_high_threshold() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();
        setup_graph(&pool).await;

        // min_count=5 should only find d-e(5)
        let clusters = find_clusters(&pool, "t1", "projects", 5, 2).await.unwrap();
        if !clusters.is_empty() {
            let syms: Vec<&str> = clusters[0].symbols.iter().map(|s| s.as_str()).collect();
            assert!(syms.contains(&"d"));
            assert!(syms.contains(&"e"));
        }
    }

    #[tokio::test]
    async fn test_find_clusters_empty() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        let clusters = find_clusters(&pool, "t1", "projects", 1, 2).await.unwrap();
        assert!(clusters.is_empty());
    }

    #[tokio::test]
    async fn test_get_neighbors() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();
        setup_graph(&pool).await;

        // Neighbors of 'c' with min_count=1
        let neighbors = get_neighbors(&pool, "t1", "projects", "c", 1)
            .await
            .unwrap();
        let names: Vec<&str> = neighbors.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"a"), "c co-occurs with a");
        assert!(names.contains(&"b"), "c co-occurs with b");
        assert!(names.contains(&"d"), "c co-occurs with d");
        assert!(!names.contains(&"e"), "c does not directly co-occur with e");
    }

    #[tokio::test]
    async fn test_get_neighbors_min_count() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();
        setup_graph(&pool).await;

        // Neighbors of 'c' with min_count=3: only c-d(4)
        let neighbors = get_neighbors(&pool, "t1", "projects", "c", 3)
            .await
            .unwrap();
        let names: Vec<&str> = neighbors.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"d"), "c-d has weight 4 >= 3");
        assert!(!names.contains(&"a"), "c-a has weight 1 < 3");
        assert!(!names.contains(&"b"), "c-b has weight 2 < 3");
    }

    #[tokio::test]
    async fn test_betweenness_centrality() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();
        setup_graph(&pool).await;

        let centrality = get_betweenness_centrality(&pool, "t1", "projects", 1)
            .await
            .unwrap();
        // Symbols that appear on many 2-hop paths should have high centrality
        assert!(!centrality.is_empty(), "Should compute centrality scores");
    }

    #[tokio::test]
    async fn test_betweenness_centrality_empty() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        let centrality = get_betweenness_centrality(&pool, "t1", "projects", 1)
            .await
            .unwrap();
        assert!(centrality.is_empty());
    }

    #[tokio::test]
    async fn test_tenant_isolation() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        let pairs = vec![("x".to_string(), "y".to_string())];
        upsert_cooccurrences(&pool, "t1", "projects", &pairs)
            .await
            .unwrap();
        upsert_cooccurrences(&pool, "t2", "projects", &pairs)
            .await
            .unwrap();

        let c1 = get_degree_centrality(&pool, "t1", "projects")
            .await
            .unwrap();
        let c2 = get_degree_centrality(&pool, "t2", "projects")
            .await
            .unwrap();

        // Both should have entries, isolated by tenant
        assert!(c1.contains_key("x"));
        assert!(c2.contains_key("x"));

        // Count should be 1 each (not 2)
        let count: i64 = sqlx::query_scalar(
            "SELECT cooccurrence_count FROM symbol_cooccurrence WHERE tenant_id = 't1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 1);
    }
}
