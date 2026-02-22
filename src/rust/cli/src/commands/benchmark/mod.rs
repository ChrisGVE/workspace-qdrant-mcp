//! Benchmark commands — sparse vector and search evaluation tools.

mod search;
mod sparse;
pub mod stats;

use anyhow::Result;
use clap::{Args, Subcommand};

/// Benchmark command arguments
#[derive(Args)]
pub struct BenchmarkArgs {
    #[command(subcommand)]
    command: BenchmarkCommand,
}

/// Benchmark subcommands
#[derive(Subcommand)]
enum BenchmarkCommand {
    /// Compare BM25 vs SPLADE++ sparse vectors
    Sparse {
        /// Collection to sample from (default: projects)
        #[arg(long, default_value = "projects")]
        collection: String,

        /// Number of documents to sample
        #[arg(long, default_value_t = 100)]
        sample_size: usize,

        /// Number of queries to evaluate
        #[arg(long, default_value_t = 20)]
        query_count: usize,

        /// Output JSON report to file
        #[arg(long)]
        output: Option<String>,
    },

    /// Compare FTS5 search DB vs ripgrep (rg)
    Search {
        /// Tenant ID to scope FTS5 queries (auto-detected if omitted)
        #[arg(long)]
        tenant_id: Option<String>,

        /// Warmup iterations before measuring (default: 2)
        #[arg(long, default_value_t = 2)]
        warmup: usize,

        /// Measurement iterations per query (default: 10)
        #[arg(long, default_value_t = 10)]
        iterations: usize,

        /// High-iteration stress mode (50 iterations, 5 warmup)
        #[arg(long)]
        stress: bool,

        /// Output JSON report to file
        #[arg(long)]
        output: Option<String>,
    },
}

/// Execute benchmark command
pub async fn execute(args: BenchmarkArgs) -> Result<()> {
    match args.command {
        BenchmarkCommand::Sparse {
            collection,
            sample_size,
            query_count,
            output: output_file,
        } => sparse::execute(&collection, sample_size, query_count, output_file).await,

        BenchmarkCommand::Search {
            tenant_id,
            warmup,
            iterations,
            stress,
            output: output_file,
        } => {
            let (warmup, iterations) = if stress {
                (5, 50)
            } else {
                (warmup, iterations)
            };
            search::execute(tenant_id, warmup, iterations, output_file).await
        }
    }
}
