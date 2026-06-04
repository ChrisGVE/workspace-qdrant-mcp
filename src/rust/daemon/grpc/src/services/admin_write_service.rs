//! AdminWriteService gRPC implementation
//!
//! Delegates all state.db mutations to the WriteActor via WriteActorHandle.
//! TriggerReembed bypasses the WriteActor channel because the reembed
//! pipeline manages its own queue + Qdrant lifecycle (PRD §6.6).

use std::sync::Arc;
use std::time::Duration;

use sqlx::SqlitePool;
use tonic::{Request, Response, Status};
use workspace_qdrant_core::storage::StorageClient;
use workspace_qdrant_core::write_actor::{
    RebalanceIdfData, RenameTenantAdminData, WriteActorHandle,
};

use crate::proto::{
    admin_write_service_server::AdminWriteService, PerCollectionRebalance, RebalanceIdfRequest,
    RebalanceIdfResponse, RenameTenantAdminRequest, RenameTenantAdminResponse,
    TriggerReembedRequest, TriggerReembedResponse,
};
use crate::services::reembed::{execute_reembed, ReembedContext, StorageClientRecreator};

/// Drain-to-quiescence timeout per PRD §6.6.
const REEMBED_DRAIN_TIMEOUT: Duration = Duration::from_secs(60);
/// Polling cadence while waiting for in_progress items to drain.
const REEMBED_DRAIN_POLL: Duration = Duration::from_millis(500);

pub struct AdminWriteServiceImpl {
    write_actor: WriteActorHandle,
    reembed_ctx: Option<Arc<ReembedContext>>,
    /// Qdrant client + SQLite pool for the full RebalanceIdf engine (WI-f1).
    rebalance_ctx: Option<(Arc<StorageClient>, SqlitePool)>,
}

impl AdminWriteServiceImpl {
    pub fn new(write_actor: WriteActorHandle) -> Self {
        Self {
            write_actor,
            reembed_ctx: None,
            rebalance_ctx: None,
        }
    }

    /// Inject the dependencies required by `TriggerReembed`. Without this
    /// call the RPC returns `failed_precondition`.
    pub fn with_reembed_context(mut self, ctx: Arc<ReembedContext>) -> Self {
        self.reembed_ctx = Some(ctx);
        self
    }

    /// Inject the Qdrant client + SQLite pool the `RebalanceIdf` engine needs.
    /// Without this the RPC returns `failed_precondition`.
    pub fn with_rebalance_context(mut self, storage: Arc<StorageClient>, pool: SqlitePool) -> Self {
        self.rebalance_ctx = Some((storage, pool));
        self
    }
}

fn to_status(err: String) -> Status {
    if err.contains("must not be empty") {
        Status::invalid_argument(err)
    } else {
        Status::internal(err)
    }
}

#[tonic::async_trait]
impl AdminWriteService for AdminWriteServiceImpl {
    async fn rename_tenant_admin(
        &self,
        request: Request<RenameTenantAdminRequest>,
    ) -> Result<Response<RenameTenantAdminResponse>, Status> {
        let req = request.into_inner();
        let result = self
            .write_actor
            .rename_tenant_admin(RenameTenantAdminData {
                old_tenant_id: req.old_tenant_id,
                new_tenant_id: req.new_tenant_id,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(RenameTenantAdminResponse {
            success: result.success,
            total_rows_updated: result.total_rows_updated,
            message: result.message,
        }))
    }

    async fn rebalance_idf(
        &self,
        request: Request<RebalanceIdfRequest>,
    ) -> Result<Response<RebalanceIdfResponse>, Status> {
        let req = request.into_inner();
        let (storage, pool) = self.rebalance_ctx.clone().ok_or_else(|| {
            Status::failed_precondition(
                "RebalanceIdf not available: storage client + db pool not wired",
            )
        })?;

        // Engine: reads corpus_statistics/sparse_vocabulary, corrects Qdrant
        // sparse vectors. No state.db writes here (single-writer = WriteActor).
        let report = workspace_qdrant_core::idf_rebalance::rebalance_idf(
            &pool,
            &storage,
            req.collection,
            req.dry_run,
            req.min_growth_pct,
        )
        .await
        .map_err(|e| Status::internal(format!("rebalance failed: {e}")))?;

        // Persist last_corrected_n for corrected collections via the WriteActor.
        for c in &report.per_collection {
            if c.persist_n {
                self.write_actor
                    .rebalance_idf(RebalanceIdfData {
                        collection: c.collection.clone(),
                        last_corrected_n: c.current_n as i64,
                    })
                    .await
                    .map_err(to_status)?;
            }
        }

        let per_collection = report
            .per_collection
            .iter()
            .map(|c| PerCollectionRebalance {
                collection: c.collection.clone(),
                current_n: c.current_n,
                updated: c.updated,
                skipped_reason: c.skipped_reason.clone(),
            })
            .collect();

        let message = if report.dry_run {
            format!(
                "Dry-run: {} point(s) would be corrected",
                report.total_updated
            )
        } else {
            format!("Corrected {} point(s)", report.total_updated)
        };

        Ok(Response::new(RebalanceIdfResponse {
            success: true,
            dry_run: report.dry_run,
            total_updated: report.total_updated,
            per_collection,
            message,
        }))
    }

    async fn trigger_reembed(
        &self,
        request: Request<TriggerReembedRequest>,
    ) -> Result<Response<TriggerReembedResponse>, Status> {
        let req = request.into_inner();
        if !req.confirm {
            return Err(Status::failed_precondition(
                "TriggerReembed requires confirm=true",
            ));
        }

        let ctx = self.reembed_ctx.clone().ok_or_else(|| {
            Status::failed_precondition(
                "TriggerReembed not available: reembed context not wired \
                     (embedding settings, provider, storage client, db pool, \
                     pause flag must all be present)",
            )
        })?;

        let recreator = StorageClientRecreator {
            storage: Arc::clone(&ctx.storage_client),
        };
        let resp =
            execute_reembed(&ctx, &recreator, REEMBED_DRAIN_TIMEOUT, REEMBED_DRAIN_POLL).await?;
        Ok(Response::new(resp))
    }
}
