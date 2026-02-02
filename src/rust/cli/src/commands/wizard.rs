//! Wizard command - guided setup wizards
//!
//! Phase 3 LOW priority command for guided setup.
//! Helps users configure the system interactively.

use anyhow::Result;
use clap::{Args, Subcommand};

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

/// Wizard command arguments
#[derive(Args)]
pub struct WizardArgs {
    #[command(subcommand)]
    command: WizardCommand,
}

/// Wizard subcommands
#[derive(Subcommand)]
enum WizardCommand {
    /// Initial setup wizard
    Setup,

    /// Diagnose and fix common issues
    Diagnose,

    /// Optimize configuration for your system
    Optimize,
}

/// Execute wizard command
pub async fn execute(args: WizardArgs) -> Result<()> {
    match args.command {
        WizardCommand::Setup => run_setup_wizard().await,
        WizardCommand::Diagnose => run_diagnose_wizard().await,
        WizardCommand::Optimize => run_optimize_wizard().await,
    }
}

async fn run_setup_wizard() -> Result<()> {
    output::section("Setup Wizard");

    output::info("Welcome to the Workspace Qdrant MCP setup wizard!");
    output::info("This will help you configure the system for first use.");
    output::separator();

    // Step 1: Check prerequisites
    output::info("Step 1: Checking prerequisites...");

    // Check Qdrant
    let qdrant_ok = check_qdrant().await;
    output::status_line(
        "  Qdrant Server",
        if qdrant_ok {
            ServiceStatus::Healthy
        } else {
            ServiceStatus::Unhealthy
        },
    );

    if !qdrant_ok {
        output::warning("Qdrant is not running. Start it with:");
        output::info("  docker run -p 6333:6333 qdrant/qdrant");
        output::separator();
    }

    // Check daemon
    let daemon_ok = DaemonClient::connect_default().await.is_ok();
    output::status_line(
        "  Daemon (memexd)",
        if daemon_ok {
            ServiceStatus::Healthy
        } else {
            ServiceStatus::Unhealthy
        },
    );

    if !daemon_ok {
        output::info("  Install and start the daemon:");
        output::info("    wqm service install");
        output::info("    wqm service start");
    }

    output::separator();

    // Step 2: Configuration
    output::info("Step 2: Configuration files");
    output::info("  Config location: ~/.config/workspace-qdrant/");
    output::info("  Data location: ~/.local/share/workspace-qdrant/");

    output::separator();

    // Step 3: Environment
    output::info("Step 3: Environment variables (optional)");
    output::info("  QDRANT_URL        - Qdrant server URL (default: http://localhost:6333)");
    output::info("  WQM_DAEMON_ADDR   - Daemon address (default: http://127.0.0.1:50051)");
    output::info("  FASTEMBED_MODEL   - Embedding model (default: all-MiniLM-L6-v2)");

    output::separator();

    // Step 4: Next steps
    output::info("Step 4: Next steps");
    output::info("  1. Start Qdrant (if not running)");
    output::info("  2. Start daemon: wqm service start");
    output::info("  3. Register a project: wqm project register");
    output::info("  4. Check status: wqm status");

    Ok(())
}

async fn check_qdrant() -> bool {
    // Try to connect to Qdrant
    std::process::Command::new("curl")
        .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:6333"])
        .output()
        .map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout) == "200")
        .unwrap_or(false)
}

async fn run_diagnose_wizard() -> Result<()> {
    output::section("Diagnostic Wizard");

    output::info("Running system diagnostics...");
    output::separator();

    // Check 1: Qdrant connectivity
    output::info("1. Qdrant Server");
    let qdrant_ok = check_qdrant().await;
    output::status_line(
        "   Connection",
        if qdrant_ok {
            ServiceStatus::Healthy
        } else {
            ServiceStatus::Unhealthy
        },
    );

    if !qdrant_ok {
        output::warning("   Issue: Cannot connect to Qdrant");
        output::info("   Fix: Ensure Qdrant is running on localhost:6333");
    }

    output::separator();

    // Check 2: Daemon
    output::info("2. Daemon (memexd)");
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("   Connection", ServiceStatus::Healthy);

            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();
                    let status = ServiceStatus::from_proto(health.status);
                    output::status_line("   Health", status);

                    for comp in health.components {
                        let comp_status = ServiceStatus::from_proto(comp.status);
                        output::status_line(&format!("   {}", comp.component_name), comp_status);
                    }
                }
                Err(e) => {
                    output::warning(format!("   Could not get health: {}", e));
                }
            }
        }
        Err(_) => {
            output::status_line("   Connection", ServiceStatus::Unhealthy);
            output::warning("   Issue: Daemon not running");
            output::info("   Fix: wqm service start");
        }
    }

    output::separator();

    // Check 3: Database
    output::info("3. SQLite Database");
    let db_path = dirs::data_local_dir()
        .map(|p| p.join("workspace-qdrant/state.db"))
        .unwrap_or_default();

    if db_path.exists() {
        output::status_line("   Database File", ServiceStatus::Healthy);
        output::kv("   Path", &db_path.display().to_string());
    } else {
        output::status_line("   Database File", ServiceStatus::Unknown);
        output::info("   Note: Database will be created on first use");
    }

    output::separator();
    output::info("Diagnostics complete.");

    Ok(())
}

async fn run_optimize_wizard() -> Result<()> {
    output::section("Optimization Wizard");

    output::info("Analyzing your system configuration...");
    output::separator();

    // Check current status
    let daemon_connected = DaemonClient::connect_default().await.is_ok();

    if !daemon_connected {
        output::warning("Cannot analyze without daemon connection.");
        output::info("Start daemon first: wqm service start");
        return Ok(());
    }

    output::info("Recommendations:");
    output::separator();

    // Recommendation 1: Embedding model
    output::info("1. Embedding Model");
    output::info("   Current: all-MiniLM-L6-v2 (default)");
    output::info("   For better quality, consider: BAAI/bge-small-en-v1.5");
    output::info("   Set via: FASTEMBED_MODEL environment variable");

    output::separator();

    // Recommendation 2: Qdrant indexing
    output::info("2. Qdrant Indexing");
    output::info("   For large collections (>10k docs), enable HNSW indexing");
    output::info("   This is automatic for collections created by the daemon");

    output::separator();

    // Recommendation 3: Watch folders
    output::info("3. Watch Configuration");
    output::info("   Set appropriate file patterns to reduce unnecessary indexing");
    output::info("   Example: wqm library watch docs /path -p \"*.md\" -p \"*.txt\"");

    output::separator();

    // Recommendation 4: Memory/CPU
    output::info("4. Resource Usage");
    output::info("   The daemon is optimized for low resource usage");
    output::info("   Monitor with: wqm status performance");

    output::separator();
    output::info("Run 'wqm help quick' for common usage patterns.");

    Ok(())
}
