//! Workspace Qdrant Daemon
//!
//! High-performance Rust daemon for workspace document processing and vector search.
//! Provides gRPC services for document processing, search operations, memory management,
//! and system administration.

use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;
use tracing::{info, warn};

mod grpc;
mod daemon;
mod config;
mod error;

// Include generated protobuf code
pub mod proto {
    tonic::include_proto!("workspace_daemon");
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// gRPC server address
    #[arg(short, long, default_value = "127.0.0.1:50051")]
    address: SocketAddr,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<std::path::PathBuf>,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Enable metrics collection
    #[arg(long)]
    enable_metrics: bool,

    /// Daemon mode (run in background)
    #[arg(short, long)]
    daemon: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing
    init_tracing(&args.log_level)?;

    info!("Starting Workspace Qdrant Daemon v{}", env!("CARGO_PKG_VERSION"));
    info!("gRPC server will listen on: {}", args.address);

    // Load configuration
    let config = config::DaemonConfig::load(args.config.as_deref())?;
    info!("Configuration loaded successfully");

    // Initialize daemon
    let mut daemon = daemon::WorkspaceDaemon::new(config).await?;

    // Start daemon services
    daemon.start().await?;

    // Start gRPC server
    let grpc_server = grpc::server::GrpcServer::new(daemon, args.address);

    if args.daemon {
        info!("Running in daemon mode");
        grpc_server.serve_daemon().await?;
    } else {
        info!("Running in foreground mode");
        grpc_server.serve().await?;
    }

    Ok(())
}

fn init_tracing(level: &str) -> Result<()> {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(level))
        .map_err(|e| anyhow::anyhow!("Invalid log level: {}", e))?;

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(filter)
        .init();

    Ok(())
}
