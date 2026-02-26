//! Restart and health-check logic
//!
//! Implements exponential backoff restart, health-driven restart,
//! and periodic health checking for LSP server instances.

use std::time::{Duration, Instant};

use tracing::info;

use crate::lsp::LspResult;

use super::process::ServerInstance;
use super::{HealthMetrics, ServerStatus};

impl ServerInstance {
    /// Perform health check on the server
    pub async fn health_check(&self) -> LspResult<HealthMetrics> {
        let start_time = Instant::now();

        // Check if process is still running
        let process_healthy = {
            let mut process_guard = self.process.lock().await;
            match process_guard.as_mut() {
                Some(child) => match child.try_wait() {
                    Ok(Some(_)) => false, // Process has exited
                    Ok(None) => true,     // Process is still running
                    Err(_) => false,      // Error checking process
                },
                None => false, // No process
            }
        };

        let mut metrics = self.health_metrics.write().await;

        if !process_healthy {
            metrics.status = ServerStatus::Failed;
            metrics.consecutive_failures += 1;
            return Ok(metrics.clone());
        }

        // Try to send a simple request to check responsiveness
        let rpc_healthy = match tokio::time::timeout(
            Duration::from_secs(5),
            self.rpc_client
                .send_request("shutdown", serde_json::json!(null)),
        )
        .await
        {
            Ok(Ok(_)) => {
                // Server responded, re-initialize to keep it running
                let _ = self.reinitialize_after_health_check().await;
                true
            }
            Ok(Err(_)) => false, // RPC error
            Err(_) => false,     // Timeout
        };

        let response_time = start_time.elapsed();
        metrics.response_time_ms = response_time.as_millis() as u64;

        if rpc_healthy {
            metrics.status = ServerStatus::Running;
            metrics.last_healthy = chrono::Utc::now();
            metrics.consecutive_failures = 0;
        } else {
            metrics.status = if metrics.consecutive_failures > 3 {
                ServerStatus::Failed
            } else {
                ServerStatus::Degraded
            };
            metrics.consecutive_failures += 1;
        }

        // Update average response time (exponential moving average)
        let alpha = 0.1;
        metrics.avg_response_time_ms =
            alpha * (metrics.response_time_ms as f64) + (1.0 - alpha) * metrics.avg_response_time_ms;

        Ok(metrics.clone())
    }

    /// Restart the server instance
    pub async fn restart(&mut self) -> LspResult<()> {
        info!("Restarting LSP server: {}", self.metadata.name);

        // Update restart policy
        self.restart_policy.current_attempts += 1;
        self.restart_policy.last_restart = Some(Instant::now());

        // Stop current process
        self.stop_process().await?;

        // Wait for restart delay with exponential backoff
        let delay = self.calculate_restart_delay();
        tokio::time::sleep(delay).await;

        // Start new process
        self.start().await?;

        info!("LSP server {} restarted successfully", self.metadata.name);
        Ok(())
    }

    /// Calculate restart delay with exponential backoff
    fn calculate_restart_delay(&self) -> Duration {
        let base_delay_secs = self.restart_policy.base_delay.as_secs_f64();
        let multiplier = self.restart_policy.backoff_multiplier;
        let attempts = self.restart_policy.current_attempts as f64;

        let delay_secs = base_delay_secs * multiplier.powf(attempts - 1.0);
        let max_delay_secs = self.restart_policy.max_delay.as_secs_f64();

        Duration::from_secs_f64(delay_secs.min(max_delay_secs))
    }

    /// Check server health and restart if needed
    ///
    /// Returns true if server needed restart, false otherwise.
    pub async fn check_and_restart_if_needed(&mut self) -> LspResult<bool> {
        // Check if process is alive
        if self.is_alive().await {
            return Ok(false); // No restart needed
        }

        // Process died - check restart policy
        if !self.restart_policy.enabled {
            info!(
                "LSP server {} crashed but restart is disabled",
                self.metadata.name
            );
            let mut metrics = self.health_metrics.write().await;
            metrics.status = ServerStatus::Failed;
            return Ok(false);
        }

        if self.restart_policy.current_attempts >= self.restart_policy.max_attempts {
            info!(
                "LSP server {} crashed and exceeded max restart attempts ({})",
                self.metadata.name, self.restart_policy.max_attempts
            );
            let mut metrics = self.health_metrics.write().await;
            metrics.status = ServerStatus::Failed;
            return Ok(false);
        }

        // Check if we should reset restart counter based on time window
        if let Some(last) = self.restart_policy.last_restart {
            if last.elapsed() > self.restart_policy.reset_window {
                self.restart_policy.current_attempts = 0;
            }
        }

        // Attempt restart
        let attempt_number = self.restart_policy.current_attempts + 1;
        info!(
            "LSP server {} crashed, attempting restart (attempt {} of {})",
            self.metadata.name, attempt_number, self.restart_policy.max_attempts
        );

        // Update metrics
        {
            let mut metrics = self.health_metrics.write().await;
            metrics.status = ServerStatus::Failed;
            metrics.consecutive_failures += 1;
        }

        // Perform restart with backoff
        self.restart().await?;

        info!(
            "LSP server {} restarted successfully",
            self.metadata.name
        );

        Ok(true)
    }

    /// Reset restart attempts counter (e.g., after extended stable period)
    pub fn reset_restart_attempts(&mut self) {
        self.restart_policy.current_attempts = 0;
        self.restart_policy.last_restart = None;
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::lsp::detection::ServerCapabilities;
    use crate::lsp::{DetectedServer, Language, LspConfig};

    use super::*;

    #[tokio::test]
    async fn test_server_instance_restart_policy_getter() {
        let detected = DetectedServer {
            name: "test-server".to_string(),
            path: PathBuf::from("/usr/bin/test"),
            languages: vec![Language::Rust],
            version: Some("1.0".to_string()),
            capabilities: ServerCapabilities::default(),
            priority: 1,
        };

        let instance = ServerInstance::new(detected, LspConfig::default()).await.unwrap();
        let policy = instance.restart_policy();

        assert!(policy.enabled);
        assert_eq!(policy.max_attempts, 5);
    }

    #[tokio::test]
    async fn test_server_instance_reset_restart_attempts() {
        let detected = DetectedServer {
            name: "test-server".to_string(),
            path: PathBuf::from("/usr/bin/test"),
            languages: vec![Language::Rust],
            version: Some("1.0".to_string()),
            capabilities: ServerCapabilities::default(),
            priority: 1,
        };

        let mut instance = ServerInstance::new(detected, LspConfig::default()).await.unwrap();

        // Simulate some restart attempts
        instance.restart_policy.current_attempts = 3;
        instance.restart_policy.last_restart = Some(Instant::now());

        // Reset
        instance.reset_restart_attempts();

        assert_eq!(instance.restart_policy.current_attempts, 0);
        assert!(instance.restart_policy.last_restart.is_none());
    }
}
