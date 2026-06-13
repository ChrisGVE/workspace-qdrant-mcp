//! Semantic-search benchmark wrapper for the TS harness.
//!
//! This command shells out to the existing `benchmark-semantic-search.ts`
//! runner so the CLI can expose a first-class benchmark for embedding/model
//! changes without duplicating the harness logic in Rust.

use anyhow::{bail, Context, Result};
use clap::Args;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Benchmark the semantic-search quality harness used for embedding/model swaps.
#[derive(Args, Debug, Clone)]
pub struct SemanticBenchmarkArgs {
    /// Workspace root used to normalize file paths and locate the TS runner.
    #[arg(long)]
    pub workspace_root: Option<PathBuf>,

    /// Path to the benchmark YAML dataset.
    #[arg(long)]
    pub dataset: Option<PathBuf>,

    /// Project tenant id (skips auto-detection in the runner).
    #[arg(long)]
    pub project_id: Option<String>,

    /// Override the Qdrant endpoint.
    #[arg(long)]
    pub qdrant_url: Option<String>,

    /// Override the Qdrant API key.
    #[arg(long)]
    pub qdrant_api_key: Option<String>,

    /// Override the daemon gRPC host.
    #[arg(long)]
    pub daemon_host: Option<String>,

    /// Override the daemon gRPC port.
    #[arg(long)]
    pub daemon_port: Option<u16>,

    /// Override the SQLite database path.
    #[arg(long)]
    pub database_path: Option<PathBuf>,

    /// Search scope to evaluate (project, global, or all).
    #[arg(long)]
    pub scope: Option<String>,

    /// Override the collection name.
    #[arg(long)]
    pub collection: Option<String>,

    /// Search limit per query.
    #[arg(long, default_value_t = 10)]
    pub limit: usize,

    /// Evaluation cutoff.
    #[arg(long, default_value_t = 10)]
    pub topk: usize,

    /// Warmup runs per mode.
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,

    /// Measured runs per mode.
    #[arg(long, default_value_t = 1)]
    pub iterations: usize,

    /// Include libraries in project-scope search.
    #[arg(long = "include-libraries")]
    pub include_libraries: bool,

    /// Run only the selected query id (repeatable).
    #[arg(long = "query-id")]
    pub query_ids: Vec<String>,

    /// Write the report as JSON.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

/// Sweep runtime semantic-search knobs without redeploying the daemon.
#[derive(Args, Debug, Clone)]
pub struct SemanticSweepBenchmarkArgs {
    /// Workspace root used to normalize file paths and locate the TS runner.
    #[arg(long)]
    pub workspace_root: Option<PathBuf>,

    /// Path to the benchmark YAML dataset.
    #[arg(long)]
    pub dataset: Option<PathBuf>,

    /// Project tenant id (skips auto-detection in the runner).
    #[arg(long)]
    pub project_id: Option<String>,

    /// Override the Qdrant endpoint.
    #[arg(long)]
    pub qdrant_url: Option<String>,

    /// Override the Qdrant API key.
    #[arg(long)]
    pub qdrant_api_key: Option<String>,

    /// Override the daemon gRPC host.
    #[arg(long)]
    pub daemon_host: Option<String>,

    /// Override the daemon gRPC port.
    #[arg(long)]
    pub daemon_port: Option<u16>,

    /// Override the SQLite database path.
    #[arg(long)]
    pub database_path: Option<PathBuf>,

    /// Search scope to evaluate (project, global, or all).
    #[arg(long)]
    pub scope: Option<String>,

    /// Override the collection name.
    #[arg(long)]
    pub collection: Option<String>,

    /// Search limit per query.
    #[arg(long, default_value_t = 10)]
    pub limit: usize,

    /// Evaluation cutoff.
    #[arg(long, default_value_t = 10)]
    pub topk: usize,

    /// Warmup runs per mode.
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,

    /// Measured runs per mode.
    #[arg(long, default_value_t = 1)]
    pub iterations: usize,

    /// Include libraries in project-scope search.
    #[arg(long = "include-libraries")]
    pub include_libraries: bool,

    /// Run only the selected query id (repeatable).
    #[arg(long = "query-id")]
    pub query_ids: Vec<String>,

    /// Default rerank weights to test, comma-separated.
    #[arg(long)]
    pub weights: Option<String>,

    /// Custom scenario spec (repeatable), e.g. off:rerank=false or weak:rerank=true,weight=0.25.
    #[arg(long = "scenario")]
    pub scenarios: Vec<String>,

    /// Write the combined sweep report as JSON.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub async fn execute(args: SemanticBenchmarkArgs) -> Result<()> {
    let workspace_root = resolve_workspace_root(args.workspace_root.clone())?;
    run_ts_benchmark(
        &workspace_root,
        "scripts/benchmark-semantic-search.ts",
        build_forwarded_args(&args, &workspace_root)?,
        "Semantic benchmark",
    )
}

pub async fn execute_sweep(args: SemanticSweepBenchmarkArgs) -> Result<()> {
    let workspace_root = resolve_workspace_root(args.workspace_root.clone())?;
    run_ts_benchmark(
        &workspace_root,
        "scripts/benchmark-semantic-search-sweep.ts",
        build_sweep_forwarded_args(&args, &workspace_root)?,
        "Semantic benchmark sweep",
    )
}

fn run_ts_benchmark(
    workspace_root: &Path,
    script_relative_path: &str,
    forwarded_args: Vec<OsString>,
    label: &str,
) -> Result<()> {
    let package_dir = workspace_root.join("src/typescript/mcp-server");
    let script_path = package_dir.join(script_relative_path);

    if !script_path.exists() {
        bail!("{} script not found at {}", label, script_path.display());
    }

    let tsx = resolve_tsx_binary(&package_dir).unwrap_or_else(|| PathBuf::from("tsx"));
    let mut cmd = Command::new(tsx);
    let temp_dir = temp_dir_for_tsx();
    cmd.current_dir(&package_dir)
        .env("WQM_REPO_DIR", workspace_root)
        .env("TMPDIR", &temp_dir)
        .env("TMP", &temp_dir)
        .env("TEMP", &temp_dir)
        .arg(script_path.as_os_str())
        .args(forwarded_args);

    let status = cmd
        .status()
        .with_context(|| format!("Failed to execute the {} runner", label))?;

    if !status.success() {
        bail!("{} exited with status {status}", label);
    }

    Ok(())
}

fn build_forwarded_args(
    args: &SemanticBenchmarkArgs,
    workspace_root: &Path,
) -> Result<Vec<OsString>> {
    let mut forwarded = vec![OsString::from(format!(
        "--workspace-root={}",
        workspace_root.display()
    ))];

    if let Some(dataset) = args.dataset.as_ref() {
        forwarded.push(OsString::from(format!(
            "--dataset={}",
            resolve_path_argument(dataset)?.display()
        )));
    }
    if let Some(project_id) = args.project_id.as_ref() {
        forwarded.push(OsString::from(format!("--project-id={project_id}")));
    }
    if let Some(qdrant_url) = args.qdrant_url.as_ref() {
        forwarded.push(OsString::from(format!("--qdrant-url={qdrant_url}")));
    }
    if let Some(qdrant_api_key) = args.qdrant_api_key.as_ref() {
        forwarded.push(OsString::from(format!("--qdrant-api-key={qdrant_api_key}")));
    }
    if let Some(daemon_host) = args.daemon_host.as_ref() {
        forwarded.push(OsString::from(format!("--daemon-host={daemon_host}")));
    }
    if let Some(daemon_port) = args.daemon_port {
        forwarded.push(OsString::from(format!("--daemon-port={daemon_port}")));
    }
    if let Some(database_path) = args.database_path.as_ref() {
        forwarded.push(OsString::from(format!(
            "--database-path={}",
            resolve_path_argument(database_path)?.display()
        )));
    }
    if let Some(scope) = args.scope.as_ref() {
        forwarded.push(OsString::from(format!("--scope={scope}")));
    }
    if let Some(collection) = args.collection.as_ref() {
        forwarded.push(OsString::from(format!("--collection={collection}")));
    }
    forwarded.push(OsString::from(format!("--limit={}", args.limit)));
    forwarded.push(OsString::from(format!("--topk={}", args.topk)));
    forwarded.push(OsString::from(format!("--warmup={}", args.warmup)));
    forwarded.push(OsString::from(format!("--iterations={}", args.iterations)));
    if args.include_libraries {
        forwarded.push(OsString::from("--include-libraries"));
    }
    for query_id in &args.query_ids {
        forwarded.push(OsString::from(format!("--query-id={query_id}")));
    }
    if let Some(output) = args.output.as_ref() {
        forwarded.push(OsString::from(format!("--output={}", resolve_path_argument(output)?.display())));
    }

    Ok(forwarded)
}

fn build_sweep_forwarded_args(
    args: &SemanticSweepBenchmarkArgs,
    workspace_root: &Path,
) -> Result<Vec<OsString>> {
    let mut forwarded = vec![OsString::from(format!(
        "--workspace-root={}",
        workspace_root.display()
    ))];

    if let Some(dataset) = args.dataset.as_ref() {
        forwarded.push(OsString::from(format!(
            "--dataset={}",
            resolve_path_argument(dataset)?.display()
        )));
    }
    if let Some(project_id) = args.project_id.as_ref() {
        forwarded.push(OsString::from(format!("--project-id={project_id}")));
    }
    if let Some(qdrant_url) = args.qdrant_url.as_ref() {
        forwarded.push(OsString::from(format!("--qdrant-url={qdrant_url}")));
    }
    if let Some(qdrant_api_key) = args.qdrant_api_key.as_ref() {
        forwarded.push(OsString::from(format!(
            "--qdrant-api-key={qdrant_api_key}"
        )));
    }
    if let Some(daemon_host) = args.daemon_host.as_ref() {
        forwarded.push(OsString::from(format!("--daemon-host={daemon_host}")));
    }
    if let Some(daemon_port) = args.daemon_port {
        forwarded.push(OsString::from(format!("--daemon-port={daemon_port}")));
    }
    if let Some(database_path) = args.database_path.as_ref() {
        forwarded.push(OsString::from(format!(
            "--database-path={}",
            resolve_path_argument(database_path)?.display()
        )));
    }
    if let Some(scope) = args.scope.as_ref() {
        forwarded.push(OsString::from(format!("--scope={scope}")));
    }
    if let Some(collection) = args.collection.as_ref() {
        forwarded.push(OsString::from(format!("--collection={collection}")));
    }
    forwarded.push(OsString::from(format!("--limit={}", args.limit)));
    forwarded.push(OsString::from(format!("--topk={}", args.topk)));
    forwarded.push(OsString::from(format!("--warmup={}", args.warmup)));
    forwarded.push(OsString::from(format!("--iterations={}", args.iterations)));
    if args.include_libraries {
        forwarded.push(OsString::from("--include-libraries"));
    }
    for query_id in &args.query_ids {
        forwarded.push(OsString::from(format!("--query-id={query_id}")));
    }
    if let Some(weights) = args.weights.as_ref() {
        forwarded.push(OsString::from(format!("--weights={weights}")));
    }
    for scenario in &args.scenarios {
        forwarded.push(OsString::from(format!("--scenario={scenario}")));
    }
    if let Some(output) = args.output.as_ref() {
        forwarded.push(OsString::from(format!(
            "--output={}",
            resolve_path_argument(output)?.display()
        )));
    }

    Ok(forwarded)
}

fn resolve_path_argument(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }

    Ok(std::env::current_dir()
        .context("Failed to read current directory")?
        .join(path))
}

fn resolve_workspace_root(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return normalize_path(path, "workspace root");
    }

    if let Some(env_root) = std::env::var_os("WQM_REPO_DIR") {
        return normalize_path(PathBuf::from(env_root), "WQM_REPO_DIR");
    }

    if let Some(found) = detect_workspace_root_from(
        std::env::current_dir().context("Failed to read current directory")?,
    ) {
        return Ok(found);
    }

    bail!(
        "Unable to resolve the workspace root. Pass --workspace-root or set WQM_REPO_DIR."
    )
}

fn detect_workspace_root_from(start: PathBuf) -> Option<PathBuf> {
    let mut dir = start;
    loop {
        if dir.join("src/typescript/mcp-server/package.json").is_file() {
            return Some(dir);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

fn normalize_path(path: PathBuf, label: &str) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .context("Failed to read current directory")?
            .join(path)
    };

    if !absolute.is_dir() {
        bail!("{} must be a directory: {}", label, absolute.display());
    }

    Ok(absolute)
}

fn temp_dir_for_tsx() -> PathBuf {
    if cfg!(unix) {
        PathBuf::from("/tmp")
    } else {
        std::env::temp_dir()
    }
}

fn resolve_tsx_binary(package_dir: &Path) -> Option<PathBuf> {
    let candidates = if cfg!(windows) {
        vec![
            package_dir.join("node_modules/.bin/tsx.cmd"),
            package_dir.join("node_modules/.bin/tsx"),
        ]
    } else {
        vec![package_dir.join("node_modules/.bin/tsx")]
    };

    candidates.into_iter().find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn build_forwarded_args_includes_repeatable_query_ids() {
        let args = SemanticBenchmarkArgs {
            workspace_root: None,
            dataset: Some(PathBuf::from("/work/data/bench.yaml")),
            project_id: Some("tenant-123".to_string()),
            qdrant_url: Some("http://localhost:6333".to_string()),
            qdrant_api_key: None,
            daemon_host: Some("127.0.0.1".to_string()),
            daemon_port: Some(50051),
            database_path: Some(PathBuf::from("/work/tmp/bench.db")),
            scope: Some("project".to_string()),
            collection: Some("projects".to_string()),
            limit: 7,
            topk: 11,
            warmup: 2,
            iterations: 3,
            include_libraries: true,
            query_ids: vec!["q1".to_string(), "q2".to_string()],
            output: Some(PathBuf::from("/work/tmp/report.json")),
        };

        let root = PathBuf::from("/repo");
        let forwarded = build_forwarded_args(&args, &root).expect("should build args");
        let strings: Vec<String> = forwarded
            .into_iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();

        assert_eq!(strings[0], "--workspace-root=/repo");
        assert!(strings.contains(&"--dataset=/work/data/bench.yaml".to_string()));
        assert!(strings.contains(&"--project-id=tenant-123".to_string()));
        assert!(strings.contains(&"--qdrant-url=http://localhost:6333".to_string()));
        assert!(strings.contains(&"--daemon-host=127.0.0.1".to_string()));
        assert!(strings.contains(&"--daemon-port=50051".to_string()));
        assert!(strings.contains(&"--database-path=/work/tmp/bench.db".to_string()));
        assert!(strings.contains(&"--scope=project".to_string()));
        assert!(strings.contains(&"--collection=projects".to_string()));
        assert!(strings.contains(&"--limit=7".to_string()));
        assert!(strings.contains(&"--topk=11".to_string()));
        assert!(strings.contains(&"--warmup=2".to_string()));
        assert!(strings.contains(&"--iterations=3".to_string()));
        assert!(strings.contains(&"--include-libraries".to_string()));
        assert!(strings.contains(&"--query-id=q1".to_string()));
        assert!(strings.contains(&"--query-id=q2".to_string()));
        assert!(strings.contains(&"--output=/work/tmp/report.json".to_string()));
    }

    #[test]
    fn build_sweep_forwarded_args_includes_weights_and_scenarios() {
        let args = SemanticSweepBenchmarkArgs {
            workspace_root: None,
            dataset: None,
            project_id: Some("tenant-123".to_string()),
            qdrant_url: Some("http://qdrant:6333".to_string()),
            qdrant_api_key: Some("secret".to_string()),
            daemon_host: Some("localhost".to_string()),
            daemon_port: Some(50051),
            database_path: Some(PathBuf::from("/work/tmp/bench.db")),
            scope: Some("project".to_string()),
            collection: None,
            limit: 10,
            topk: 10,
            warmup: 1,
            iterations: 2,
            include_libraries: false,
            query_ids: vec!["embedding-provider".to_string()],
            weights: Some("0,0.25,0.5,1".to_string()),
            scenarios: vec![
                "off:rerank=false".to_string(),
                "weak:rerank=true,weight=0.25".to_string(),
            ],
            output: Some(PathBuf::from("/work/tmp/sweep.json")),
        };

        let root = PathBuf::from("/repo");
        let forwarded = build_sweep_forwarded_args(&args, &root).expect("should build args");
        let strings: Vec<String> = forwarded
            .into_iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();

        assert_eq!(strings[0], "--workspace-root=/repo");
        assert!(strings.contains(&"--project-id=tenant-123".to_string()));
        assert!(strings.contains(&"--qdrant-url=http://qdrant:6333".to_string()));
        assert!(strings.contains(&"--qdrant-api-key=secret".to_string()));
        assert!(strings.contains(&"--daemon-host=localhost".to_string()));
        assert_eq!(
            strings
                .iter()
                .filter(|arg| arg.as_str() == "--daemon-port=50051")
                .count(),
            1
        );
        assert!(strings.contains(&"--database-path=/work/tmp/bench.db".to_string()));
        assert!(strings.contains(&"--query-id=embedding-provider".to_string()));
        assert!(strings.contains(&"--weights=0,0.25,0.5,1".to_string()));
        assert!(strings.contains(&"--scenario=off:rerank=false".to_string()));
        assert!(strings.contains(&"--scenario=weak:rerank=true,weight=0.25".to_string()));
        assert!(strings.contains(&"--output=/work/tmp/sweep.json".to_string()));
    }

    #[test]
    fn detect_workspace_root_finds_repo_root_from_nested_dir() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let base = std::env::temp_dir().join(format!("wqm-semantic-bench-{unique}"));
        let repo = base.join("repo");
        let nested = repo.join("a/b/c");
        std::fs::create_dir_all(repo.join("src/typescript/mcp-server")).unwrap();
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(
            repo.join("src/typescript/mcp-server/package.json"),
            "{}",
        )
        .unwrap();

        let found = detect_workspace_root_from(nested).expect("should find the repo root");
        assert_eq!(found, repo);

        std::fs::remove_dir_all(base).unwrap();
    }
}
