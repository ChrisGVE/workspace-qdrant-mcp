use super::super::*;
use super::{create_test_pool, setup_tables};

#[tokio::test]
async fn test_batch_insert_large_chunk_count() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool, "w1", "src/large.rs", Some("main"), Some("code"), Some("rust"),
        "2025-01-01T00:00:00Z", "hash1", 250, Some("tree_sitter"), ProcessingStatus::Done,
        ProcessingStatus::Done, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    // Generate 250 chunks (spans 3 batches: 100 + 100 + 50)
    let chunks: Vec<_> = (0..250)
        .map(|i| {
            (
                format!("point-{}", i),
                i as i32,
                format!("hash-{}", i),
                Some(ChunkType::Function),
                Some(format!("func_{}", i)),
                Some(i as i32 * 10),
                Some(i as i32 * 10 + 9),
            )
        })
        .collect();

    insert_qdrant_chunks(&pool, file_id, &chunks)
        .await
        .expect("Batch insert failed");

    let point_ids = get_chunk_point_ids(&pool, file_id).await.unwrap();
    assert_eq!(point_ids.len(), 250, "All 250 chunks should be inserted");
    assert!(point_ids.contains(&"point-0".to_string()));
    assert!(point_ids.contains(&"point-124".to_string()));
    assert!(point_ids.contains(&"point-249".to_string()));
}

#[tokio::test]
async fn test_batch_insert_boundary_sizes() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    for count in [1usize, 99, 100, 101] {
        let path = format!("src/boundary_{}.rs", count);
        let file_id = insert_tracked_file(
            &pool, "w1", &path, Some("main"), None, None, "2025-01-01T00:00:00Z", "h1",
            count as i32, None, ProcessingStatus::None, ProcessingStatus::None, None, None, false,
            None, None, None,
        )
        .await
        .unwrap();

        let chunks: Vec<_> = (0..count)
            .map(|i| {
                (
                    format!("p-{}-{}", count, i),
                    i as i32,
                    format!("c-{}", i),
                    None,
                    None,
                    None,
                    None,
                )
            })
            .collect();

        insert_qdrant_chunks(&pool, file_id, &chunks)
            .await
            .unwrap_or_else(|e| panic!("Failed for count={}: {}", count, e));

        let ids = get_chunk_point_ids(&pool, file_id).await.unwrap();
        assert_eq!(
            ids.len(),
            count,
            "Expected {} chunks, got {}",
            count,
            ids.len()
        );
    }
}

#[tokio::test]
async fn test_batch_insert_empty_chunks() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool, "w1", "src/empty.rs", Some("main"), None, None, "2025-01-01T00:00:00Z", "h1", 0,
        None, ProcessingStatus::None, ProcessingStatus::None, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    insert_qdrant_chunks(&pool, file_id, &[])
        .await
        .expect("Empty insert should succeed");

    let ids = get_chunk_point_ids(&pool, file_id).await.unwrap();
    assert_eq!(ids.len(), 0);
}

#[tokio::test]
async fn test_batch_insert_tx_large_count() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mut tx = pool.begin().await.unwrap();
    let file_id = insert_tracked_file_tx(
        &mut tx, "w1", "src/tx_large.rs", Some("main"), Some("code"), Some("rust"),
        "2025-01-01T00:00:00Z", "hash1", 150, Some("tree_sitter"), ProcessingStatus::Done,
        ProcessingStatus::Done, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    let chunks: Vec<_> = (0..150)
        .map(|i| {
            (
                format!("tp-{}", i),
                i as i32,
                format!("th-{}", i),
                Some(ChunkType::Method),
                None,
                Some(i as i32),
                Some(i as i32 + 5),
            )
        })
        .collect();

    insert_qdrant_chunks_tx(&mut tx, file_id, &chunks)
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let ids = get_chunk_point_ids(&pool, file_id).await.unwrap();
    assert_eq!(ids.len(), 150, "All 150 tx chunks should be inserted");
}
