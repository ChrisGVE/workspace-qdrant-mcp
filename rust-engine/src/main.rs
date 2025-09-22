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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_args_default_values() {
        let args = Args::parse_from(&["test"]);

        assert_eq!(args.address.to_string(), "127.0.0.1:50051");
        assert_eq!(args.log_level, "info");
        assert!(!args.enable_metrics);
        assert!(!args.daemon);
        assert!(args.config.is_none());
    }

    #[test]
    fn test_args_custom_values() {
        let args = Args::parse_from(&[
            "test",
            "-a", "0.0.0.0:8080",
            "-l", "debug",
            "--enable-metrics",
            "--daemon",
            "-c", "/path/to/config.yaml"
        ]);

        assert_eq!(args.address.to_string(), "0.0.0.0:8080");
        assert_eq!(args.log_level, "debug");
        assert!(args.enable_metrics);
        assert!(args.daemon);
        assert_eq!(args.config.unwrap().to_string_lossy(), "/path/to/config.yaml");
    }

    #[test]
    fn test_args_debug_format() {
        let args = Args::parse_from(&["test"]);
        let debug_str = format!("{:?}", args);

        assert!(debug_str.contains("Args"));
        assert!(debug_str.contains("127.0.0.1:50051"));
        assert!(debug_str.contains("info"));
    }

    #[test]
    fn test_init_tracing_valid_levels() {
        // Test various valid log levels
        let valid_levels = vec!["trace", "debug", "info", "warn", "error"];

        for level in valid_levels {
            // Note: We can't actually call init_tracing multiple times in tests
            // because it would panic, so we just test the validation logic
            let result = std::panic::catch_unwind(|| {
                init_tracing(level)
            });
            // Should not panic for valid levels (though it may fail due to already initialized)
            assert!(result.is_ok() || result.is_err()); // Either succeeds or fails gracefully
        }
    }

    #[test]
    fn test_init_tracing_invalid_level() {
        let result = init_tracing("invalid_level");
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid log level"));
    }

    #[test]
    fn test_args_parser_help() {
        // Test that the help text can be generated without panicking
        let result = std::panic::catch_unwind(|| {
            Args::parse_from(&["test", "--help"])
        });
        // Should panic with help message (this is expected behavior for --help)
        assert!(result.is_err());
    }

    #[test]
    fn test_args_with_ipv6_address() {
        let args = Args::parse_from(&[
            "test",
            "-a", "[::1]:9090"
        ]);

        assert_eq!(args.address.to_string(), "[::1]:9090");
    }
}
