//! Phase 7: Graceful shutdown with signal handling and cleanup.
//!
//! Listens for SIGTERM / SIGINT / Ctrl-C, then tears down components in
//! reverse startup order: watchers, adaptive resources, queue processor,
//! LSP manager, and background tasks.

use std::sync::Arc;

use tokio::signal;
use tokio::sync::RwLock;
use tracing::{error, info};

use workspace_qdrant_core::{LanguageServerManager, UnifiedQueueProcessor, WatchManager};

use crate::background::BackgroundHandles;

/// Block until a termination signal is received.
pub async fn wait_for_signal() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(unix)]
    {
        let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())?;
        let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())?;

        tokio::select! {
            _ = sigterm.recv() => {
                info!("Received SIGTERM, initiating graceful shutdown");
            }
            _ = sigint.recv() => {
                info!("Received SIGINT, initiating graceful shutdown");
            }
            _ = signal::ctrl_c() => {
                info!("Received Ctrl+C, initiating graceful shutdown");
            }
        }
    }

    #[cfg(not(unix))]
    {
        signal::ctrl_c().await?;
        info!("Received Ctrl+C, initiating graceful shutdown");
    }

    Ok(())
}

/// Stop file watchers to prevent new queue items from being created.
pub async fn stop_watchers(watch_manager: &WatchManager) {
    info!("Stopping file watchers...");
    if let Err(e) = watch_manager.stop_all_watches().await {
        error!("Error stopping file watchers: {}", e);
    } else {
        info!("File watchers stopped");
    }
}

/// Stop the unified queue processor gracefully.
pub async fn stop_queue_processor(uqp: &mut UnifiedQueueProcessor) {
    info!("Stopping unified queue processor...");
    if let Err(e) = uqp.stop().await {
        error!("Error stopping unified queue processor: {}", e);
    } else {
        info!("Unified queue processor stopped");
    }
}

/// Shutdown LSP manager and all running language servers.
pub async fn stop_lsp(lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>) {
    if let Some(lsp) = lsp_manager {
        info!("Shutting down LSP manager...");
        let manager = lsp.write().await;
        if let Err(e) = manager.shutdown().await {
            error!("Error shutting down LSP manager: {}", e);
        } else {
            info!("LSP manager shutdown complete");
        }
    }
}

/// Abort all background task handles and cancellation tokens.
pub fn abort_background_tasks(
    handles: BackgroundHandles,
    adaptive_shutdown_token: tokio_util::sync::CancellationToken,
    hierarchy_cancel: tokio_util::sync::CancellationToken,
) {
    adaptive_shutdown_token.cancel();
    hierarchy_cancel.cancel();

    handles.uptime_handle.abort();
    handles.pause_poll_handle.abort();
    handles.metrics_collect_handle.abort();
    handles.metrics_maint_handle.abort();

    if let Some(grpc) = handles.grpc_handle {
        grpc.abort();
        info!("gRPC server stopped");
    }
    if let Some(metrics) = handles.metrics_handle {
        metrics.abort();
        info!("Metrics server stopped");
    }

    info!("memexd daemon shutdown complete");
}
