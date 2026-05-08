//! Tests for tracking-related WriteActor commands:
//! LogSearchEvent.

use crate::write_actor::commands::*;

use super::common::setup_test_db;

// ── LogSearchEvent tests ─────────────────────────────────────────────

#[tokio::test]
async fn log_search_event_inserts_record() {
    let (pool, handle) = setup_test_db().await;

    handle
        .log_search_event(LogSearchEventData {
            id: "evt-1".into(),
            session_id: Some("sess-1".into()),
            project_id: Some("proj-1".into()),
            actor: "claude".into(),
            tool: "search".into(),
            op: "semantic".into(),
            query_text: Some("find auth module".into()),
            filters: None,
            top_k: Some(10),
            result_count: Some(5),
            latency_ms: Some(42),
            top_result_refs: None,
            outcome: Some("success".into()),
            parent_event_id: None,
        })
        .await
        .unwrap();

    let count =
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM search_events WHERE id = 'evt-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 1);

    let actor =
        sqlx::query_scalar::<_, String>("SELECT actor FROM search_events WHERE id = 'evt-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(actor, "claude");
}
