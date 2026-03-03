//! Task payload execution: dispatch and concrete implementations

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::super::{PriorityError, TaskContext, TaskPayload, TaskResultData};

/// Execute the actual task payload
pub(crate) async fn execute_task_payload(
    payload: TaskPayload,
    context: &TaskContext,
    ingestion_engine: &Option<Arc<crate::IngestionEngine>>,
) -> Result<TaskResultData, PriorityError> {
    match payload {
        TaskPayload::ProcessDocument { file_path, collection, branch } => {
            execute_process_document(
                &file_path, &collection, &branch, context, ingestion_engine,
            ).await
        }
        TaskPayload::WatchDirectory { path, recursive } => {
            tracing::info!("Watching directory: {:?}, recursive: {}", path, recursive);
            Ok(TaskResultData::FileWatching {
                files_processed: 0,
                errors: vec![],
                checkpoint_id: context.checkpoint_id.clone(),
            })
        }
        TaskPayload::ExecuteQuery { query, collection, limit } => {
            tracing::info!(
                "Executing query: '{}' on collection: '{}' with limit: {}",
                query, collection, limit
            );
            Ok(TaskResultData::QueryExecution {
                results: vec![],
                total_results: 0,
                checkpoint_id: context.checkpoint_id.clone(),
            })
        }
        TaskPayload::Generic { operation, parameters } => {
            execute_generic_task(&operation, &parameters, context).await
        }
    }
}

/// Execute a ProcessDocument task payload
async fn execute_process_document(
    file_path: &std::path::Path,
    collection: &str,
    branch: &str,
    context: &TaskContext,
    ingestion_engine: &Option<Arc<crate::IngestionEngine>>,
) -> Result<TaskResultData, PriorityError> {
    if let Some(engine) = ingestion_engine {
        tracing::info!(
            file = %file_path.display(),
            collection = %collection,
            "Processing document with ingestion engine"
        );

        let result = engine
            .process_document(file_path, collection, branch)
            .await
            .map_err(|e| {
                tracing::error!(
                    file = %file_path.display(),
                    collection = %collection,
                    error = %e,
                    "Document processing failed"
                );
                PriorityError::ExecutionFailed {
                    reason: format!("Document processing failed: {}", e),
                }
            })?;

        tracing::info!(
            document_id = %result.document_id,
            collection = %result.collection,
            chunks_created = result.chunks_created.unwrap_or(0),
            processing_time_ms = result.processing_time_ms,
            "Document processed successfully"
        );

        Ok(TaskResultData::DocumentProcessing {
            document_id: result.document_id,
            collection: result.collection,
            chunks_created: result.chunks_created.unwrap_or(0),
            checkpoint_id: context.checkpoint_id.clone(),
        })
    } else {
        tracing::info!(
            "Processing document (stub): {:?} for collection: {}",
            file_path, collection
        );
        tokio::time::sleep(Duration::from_millis(100)).await;

        Ok(TaskResultData::DocumentProcessing {
            document_id: context.task_id.to_string(),
            collection: collection.to_string(),
            chunks_created: 1,
            checkpoint_id: context.checkpoint_id.clone(),
        })
    }
}

/// Execute a Generic task payload
async fn execute_generic_task(
    operation: &str,
    parameters: &HashMap<String, serde_json::Value>,
    context: &TaskContext,
) -> Result<TaskResultData, PriorityError> {
    tracing::info!(
        "Executing generic operation: '{}' with {} parameters",
        operation, parameters.len()
    );

    let sleep_duration = if operation.starts_with("long_") {
        Duration::from_millis(2000)
    } else {
        Duration::from_millis(100)
    };
    tokio::time::sleep(sleep_duration).await;

    Ok(TaskResultData::Generic {
        message: format!("Completed operation: {}", operation),
        data: serde_json::json!(parameters),
        checkpoint_id: context.checkpoint_id.clone(),
    })
}
