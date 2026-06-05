//! AdminWriteService gRPC implementation
//!
//! Delegates all state.db mutations to the WriteActor via WriteActorHandle.
//! TriggerReembed bypasses the WriteActor channel because the reembed
//! pipeline manages its own queue + Qdrant lifecycle (PRD §6.6).

use std::sync::Arc;
use std::time::Duration;

use tonic::{Request, Response, Status};
use tracing::{error, info};
use workspace_qdrant_core::write_actor::{
    RebalanceIdfData, ReembedTenantData, RenameTenantAdminData, WriteActorHandle,
};

use crate::proto::{
    admin_write_service_server::AdminWriteService, ReapplyIgnoreRulesResponse, RebalanceIdfRequest,
    RebalanceIdfResponse, ReembedTenantRequest, ReembedTenantResponse, RenameTenantAdminRequest,
    RenameTenantAdminResponse, TriggerReembedRequest, TriggerReembedResponse,
};
use crate::services::reembed::{execute_reembed, ReembedContext, StorageClientRecreator};

/// Drain-to-quiescence timeout per PRD §6.6.
const REEMBED_DRAIN_TIMEOUT: Duration = Duration::from_secs(60);
/// Polling cadence while waiting for in_progress items to drain.
const REEMBED_DRAIN_POLL: Duration = Duration::from_millis(500);

pub struct AdminWriteServiceImpl {
    write_actor: WriteActorHandle,
    reembed_ctx: Option<Arc<ReembedContext>>,
}

impl AdminWriteServiceImpl {
    pub fn new(write_actor: WriteActorHandle) -> Self {
        Self {
            write_actor,
            reembed_ctx: None,
        }
    }

    /// Inject the dependencies required by `TriggerReembed`. Without this
    /// call the RPC returns `failed_precondition`.
    pub fn with_reembed_context(mut self, ctx: Arc<ReembedContext>) -> Self {
        self.reembed_ctx = Some(ctx);
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
        let result = self
            .write_actor
            .rebalance_idf(RebalanceIdfData {
                collection: req.collection,
                last_corrected_n: req.last_corrected_n,
            })
            .await
            .map_err(to_status)?;

        Ok(Response::new(RebalanceIdfResponse {
            success: result.success,
            message: result.message,
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

        // Run the reembed DETACHED and return an immediate ack. The reembed is a
        // long, multi-step operation (drain-to-quiescence up to 60s, recreate the
        // 4 canonical collections, re-enqueue every watch folder). Done inline,
        // the RPC outlived the client's gRPC deadline, and on the client cancel
        // tonic dropped this handler's future mid-flight — sometimes after the
        // pause flag was set but BEFORE the folder scans were enqueued, leaving
        // the whole call a silent no-op. Spawning detaches it from the RPC
        // lifetime so the client deadline can no longer abort it; the actual
        // re-embedding then always runs via the queue.
        tokio::spawn(async move {
            let recreator = StorageClientRecreator {
                storage: Arc::clone(&ctx.storage_client),
            };
            match execute_reembed(&ctx, &recreator, REEMBED_DRAIN_TIMEOUT, REEMBED_DRAIN_POLL).await
            {
                Ok(resp) => info!("TriggerReembed complete: {}", resp.message),
                Err(e) => error!("TriggerReembed failed in background: {}", e),
            }
        });

        Ok(Response::new(TriggerReembedResponse {
            files_enqueued: 0,
            rules_enqueued: 0,
            scratchpad_enqueued: 0,
            message: "reembed started in background: the 4 canonical collections will be \
                      recreated and every watch folder re-enqueued. Track progress with \
                      `wqm queue stats` or the Grafana queue panels."
                .to_string(),
        }))
    }

    async fn reapply_ignore_rules(
        &self,
        _request: Request<()>,
    ) -> Result<Response<ReapplyIgnoreRulesResponse>, Status> {
        let result = self
            .write_actor
            .reapply_ignore_rules()
            .await
            .map_err(to_status)?;
        Ok(Response::new(ReapplyIgnoreRulesResponse {
            projects_processed: result.projects_processed,
            stale_deleted: result.stale_deleted,
            missing_added: result.missing_added,
        }))
    }

    async fn reembed_tenant(
        &self,
        request: Request<ReembedTenantRequest>,
    ) -> Result<Response<ReembedTenantResponse>, Status> {
        let req = request.into_inner();
        let result = self
            .write_actor
            .reembed_tenant(ReembedTenantData {
                tenant_id: req.tenant_id,
            })
            .await
            .map_err(to_status)?;
        Ok(Response::new(ReembedTenantResponse {
            files_enqueued: result.files_enqueued,
            message: result.message,
        }))
    }
}
