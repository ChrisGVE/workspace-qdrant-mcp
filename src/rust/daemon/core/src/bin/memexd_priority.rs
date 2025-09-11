//! memexd - Memory eXchange Daemon with Priority-based Resource Management
//!
//! Pure daemon architecture with priority queues for optimal resource utilization.
//! High-priority tasks (MCP operations, current project) get immediate attention,
//! while low-priority tasks (background ingestion) are throttled when needed.

use clap::{Arg, Command};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::signal;
use tokio::sync::{Mutex, RwLock};
use tokio::time::{Duration, interval, Instant};
use tracing::{debug, error, info, warn};
use workspace_qdrant_core::{
    ProcessingEngine, config::Config, 
    LoggingConfig, initialize_logging,
};

/// Command-line arguments for memexd daemon
#[derive(Debug, Clone)]
struct DaemonArgs {
    /// Path to configuration file
    config_file: Option<PathBuf>,
    /// Port for IPC communication
    port: Option<u16>,
    /// Logging level
    log_level: String,
    /// PID file path
    pid_file: PathBuf,
    /// Run in foreground (don't daemonize)
    foreground: bool,
}

impl Default for DaemonArgs {
    fn default() -> Self {
        Self {
            config_file: None,
            port: None,
            log_level: "info".to_string(),
            pid_file: PathBuf::from("/tmp/memexd.pid"),
            foreground: false,
        }
    }
}

/// Task priority levels for resource management
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Low priority: background folder ingestion, batch processing
    Low = 0,
    /// High priority: MCP server interactions, current project operations
    High = 1,
}

/// A task in the priority queue system
#[derive(Debug)]
pub struct PriorityTask {
    id: String,
    priority: Priority,
    payload: TaskPayload,
    created_at: Instant,
}

/// Types of tasks that can be queued
#[derive(Debug, Clone)]
pub enum TaskType {
    DocumentIngestion,
    Search,
    FolderWatch,
    HealthCheck,
    Maintenance,
}

/// Source of the task for priority determination
#[derive(Debug, Clone)]
pub enum TaskSource {
    McpServer,
    CurrentProject,
    BackgroundWatch,
    Maintenance,
}

/// Task payload containing the actual work to be performed
#[derive(Debug, Clone)]
pub enum TaskPayload {
    IngestDocument { path: String, collection: String },
    SearchQuery { query: String, collections: Vec<String> },
    WatchFolder { path: String },
    HealthCheck,
    Maintenance { task: String },
}

/// Resource manager for priority-based task scheduling
pub struct ResourceManager {
    high_priority_queue: Arc<Mutex<Vec<PriorityTask>>>,
    low_priority_queue: Arc<Mutex<Vec<PriorityTask>>>,
    active_tasks: Arc<RwLock<HashMap<String, PriorityTask>>>,
    max_concurrent_high: usize,
    max_concurrent_low: usize,
    mcp_active: Arc<RwLock<bool>>,
    stats: Arc<RwLock<ResourceStats>>,
    task_counter: AtomicU64,
}

/// Resource usage statistics
#[derive(Debug, Default, Clone)]
pub struct ResourceStats {
    tasks_processed: u64,
    high_priority_processed: u64,
    low_priority_processed: u64,
    tasks_throttled: u64,
    tasks_queued_total: u64,
    average_high_priority_time_ms: f64,
    average_low_priority_time_ms: f64,
    queue_wait_times_ms: Vec<f64>,
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceManager {
    pub fn new() -> Self {
        let high_priority_max = env::var("MEMEXD_HIGH_PRIORITY_QUEUE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);
        
        let low_priority_max = env::var("MEMEXD_LOW_PRIORITY_QUEUE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);
        
        let max_concurrent_high = env::var("MEMEXD_MAX_CONCURRENT_HIGH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
            
        let max_concurrent_low = env::var("MEMEXD_MAX_CONCURRENT_LOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        
        info!(
            "Resource manager configuration: high_priority_max={}, low_priority_max={}, concurrent_high={}, concurrent_low={}",
            high_priority_max, low_priority_max, max_concurrent_high, max_concurrent_low
        );
        
        Self {
            high_priority_queue: Arc::new(Mutex::new(Vec::with_capacity(high_priority_max))),
            low_priority_queue: Arc::new(Mutex::new(Vec::with_capacity(low_priority_max))),
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            max_concurrent_high,
            max_concurrent_low,
            mcp_active: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(ResourceStats::default())),
            task_counter: AtomicU64::new(0),
        }
    }
    
    /// Add a task to the appropriate priority queue
    pub async fn enqueue_task(&self, payload: TaskPayload, source: TaskSource) -> String {
        let task_id = format!("task_{}", self.task_counter.fetch_add(1, Ordering::SeqCst));
        
        // Determine priority based on source
        let priority = match source {
            TaskSource::McpServer | TaskSource::CurrentProject => Priority::High,
            TaskSource::BackgroundWatch | TaskSource::Maintenance => Priority::Low,
        };
        
        let task = PriorityTask {
            id: task_id.clone(),
            priority,
            payload,
            created_at: Instant::now(),
        };
        
        match priority {
            Priority::High => {
                let mut queue = self.high_priority_queue.lock().await;
                queue.push(task);
                debug!("Enqueued high-priority task: {}", task_id);
            }
            Priority::Low => {
                let mut queue = self.low_priority_queue.lock().await;
                queue.push(task);
                debug!("Enqueued low-priority task: {}", task_id);
            }
        }
        
        self.stats.write().await.tasks_queued_total += 1;
        task_id
    }
    
    /// Set MCP server activity status for throttling decisions
    pub async fn set_mcp_active(&self, active: bool) {
        let mut mcp_active = self.mcp_active.write().await;
        if *mcp_active != active {
            *mcp_active = active;
            info!("MCP activity status changed: {}", if active { "ACTIVE" } else { "INACTIVE" });
            
            if active {
                info!("Low-priority tasks will be throttled while MCP is active");
            } else {
                info!("Low-priority tasks can now resume processing");
            }
        }
    }
    
    /// Get current queue sizes and active task count
    pub async fn get_status(&self) -> (usize, usize, usize) {
        let high_queue_size = self.high_priority_queue.lock().await.len();
        let low_queue_size = self.low_priority_queue.lock().await.len();
        let active_count = self.active_tasks.read().await.len();
        
        (high_queue_size, low_queue_size, active_count)
    }
    
    /// Get comprehensive statistics
    pub async fn get_stats(&self) -> ResourceStats {
        self.stats.read().await.clone()
    }
}

/// Parse command-line arguments
fn parse_args() -> DaemonArgs {
    let matches = Command::new("memexd")
        .version("0.2.0")
        .author("Christian C. Berclaz <christian.berclaz@mac.com>")
        .about("Memory eXchange Daemon - Pure daemon with priority-based resource management")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .value_name("PORT")
                .help("IPC communication port")
                .value_parser(clap::value_parser!(u16)),
        )
        .arg(
            Arg::new("log-level")
                .short('l')
                .long("log-level")
                .value_name("LEVEL")
                .help("Logging level (error, warn, info, debug, trace)")
                .default_value("info")
                .value_parser(["error", "warn", "info", "debug", "trace"]),
        )
        .arg(
            Arg::new("pid-file")
                .long("pid-file")
                .value_name("FILE")
                .help("PID file path")
                .default_value("/tmp/memexd.pid")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("foreground")
                .short('f')
                .long("foreground")
                .help("Run in foreground (don't daemonize)")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    DaemonArgs {
        config_file: matches.get_one::<PathBuf>("config").cloned(),
        port: matches.get_one::<u16>("port").copied(),
        log_level: matches.get_one::<String>("log-level").unwrap().clone(),
        pid_file: matches.get_one::<PathBuf>("pid-file").unwrap().clone(),
        foreground: matches.get_flag("foreground"),
    }
}

/// Initialize comprehensive logging based on the specified level
fn init_logging(log_level: &str, foreground: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = if foreground {
        LoggingConfig::development()
    } else {
        LoggingConfig::production()
    };
    
    // Parse log level
    use tracing::Level;
    config.level = match log_level.to_lowercase().as_str() {
        "error" => Level::ERROR,
        "warn" => Level::WARN,
        "info" => Level::INFO,
        "debug" => Level::DEBUG,
        "trace" => Level::TRACE,
        _ => Level::INFO,
    };
    
    // Configure based on daemon mode
    if !foreground {
        // For daemon mode, disable file logging to let launchd handle redirection
        // The plist redirects stdout/stderr to user-writable log files
        config.json_format = false; // Keep readable format for launchd logs
        config.file_logging = false; // Let launchd handle file logging
        config.log_file_path = None; // No direct file logging
    }
    
    initialize_logging(config).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

/// Create PID file with current process ID
fn create_pid_file(pid_file: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let pid = process::id();
    fs::write(pid_file, pid.to_string())?;
    info!("Created PID file at {} with PID {}", pid_file.display(), pid);
    Ok(())
}

/// Remove PID file
fn remove_pid_file(pid_file: &Path) {
    if let Err(e) = fs::remove_file(pid_file) {
        warn!("Failed to remove PID file {}: {}", pid_file.display(), e);
    } else {
        info!("Removed PID file {}", pid_file.display());
    }
}

/// Check if another instance is already running
fn check_existing_instance(pid_file: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if pid_file.exists() {
        let pid_content = fs::read_to_string(pid_file)?;
        let pid: u32 = pid_content.trim().parse()?;
        
        // Check if process with this PID is still running
        #[cfg(unix)]
        {
            use std::process::Command;
            let output = Command::new("ps")
                .args(["-p", &pid.to_string()])
                .output()?;
            
            if output.status.success() && !output.stdout.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if stdout.lines().count() > 1 { // Header + process line
                    return Err(format!(
                        "Another memexd instance is already running with PID {}. \
                         Use 'kill {}' to stop it or remove stale PID file at {}",
                        pid, pid, pid_file.display()
                    ).into());
                }
            }
        }
        
        // PID file exists but process is not running - remove stale file
        warn!("Found stale PID file {}, removing it", pid_file.display());
        fs::remove_file(pid_file)?;
    }
    Ok(())
}

/// Load configuration from file or use defaults
fn load_config(args: &DaemonArgs) -> Result<Config, Box<dyn std::error::Error>> {
    match &args.config_file {
        Some(config_path) => {
            info!("Loading configuration from {}", config_path.display());
            let config_content = fs::read_to_string(config_path)?;
            let config: Config = toml::from_str(&config_content)?;
            
            if args.port.is_some() {
                info!("Port override specified: {}, but will be handled by IPC layer", args.port.unwrap());
            }
            
            Ok(config)
        }
        None => {
            info!("Using default configuration");
            let config = Config::default();
            
            if args.port.is_some() {
                info!("Port override specified: {}, but will be handled by IPC layer", args.port.unwrap());
            }
            
            Ok(config)
        }
    }
}

/// Set up signal handlers for graceful shutdown
async fn setup_signal_handlers() -> Result<(), Box<dyn std::error::Error>> {
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
        // On non-Unix systems (Windows), only handle Ctrl+C
        signal::ctrl_c().await?;
        info!("Received Ctrl+C, initiating graceful shutdown");
    }
    
    Ok(())
}

/// Main task processor that handles priority-based scheduling
async fn run_task_processor(resource_manager: Arc<ResourceManager>) {
    let mut interval = interval(Duration::from_millis(100)); // Check every 100ms
    
    loop {
        interval.tick().await;
        
        // Check if MCP is active to determine throttling
        let mcp_active = *resource_manager.mcp_active.read().await;
        
        // Process high-priority tasks first
        let active_count = resource_manager.active_tasks.read().await.len();
        let high_priority_slots = resource_manager.max_concurrent_high.saturating_sub(active_count);
        
        if high_priority_slots > 0 {
            let mut high_queue = resource_manager.high_priority_queue.lock().await;
            for _ in 0..high_priority_slots.min(high_queue.len()) {
                if let Some(task) = high_queue.pop() {
                    let task_id = task.id.clone();
                    resource_manager.active_tasks.write().await.insert(task_id.clone(), task);
                    
                    let rm_clone = resource_manager.clone();
                    tokio::spawn(async move {
                        execute_task(&rm_clone, task_id).await;
                    });
                }
            }
        }
        
        // Process low-priority tasks only if MCP is not active and we have capacity
        if !mcp_active {
            let active_count = resource_manager.active_tasks.read().await.len();
            let low_priority_slots = resource_manager.max_concurrent_low.saturating_sub(active_count);
            
            if low_priority_slots > 0 {
                let mut low_queue = resource_manager.low_priority_queue.lock().await;
                for _ in 0..low_priority_slots.min(low_queue.len()) {
                    if let Some(task) = low_queue.pop() {
                        let task_id = task.id.clone();
                        resource_manager.active_tasks.write().await.insert(task_id.clone(), task);
                        
                        let rm_clone = resource_manager.clone();
                        tokio::spawn(async move {
                            execute_task(&rm_clone, task_id).await;
                        });
                    }
                }
            }
        } else {
            // Throttle low-priority tasks when MCP is active
            let low_queue_size = resource_manager.low_priority_queue.lock().await.len();
            if low_queue_size > 0 {
                debug!("Throttling {} low-priority tasks while MCP is active", low_queue_size);
                resource_manager.stats.write().await.tasks_throttled += low_queue_size as u64;
            }
        }
    }
}

/// Execute a single task and update statistics
async fn execute_task(resource_manager: &Arc<ResourceManager>, task_id: String) {
    let start_time = Instant::now();
    
    let task = {
        let mut active_tasks = resource_manager.active_tasks.write().await;
        active_tasks.remove(&task_id)
    };
    
    if let Some(task) = task {
        let queue_wait_time = start_time.duration_since(task.created_at);
        debug!("Executing task: {} (priority: {:?}, waited: {:.2}ms)", 
               task.id, task.priority, queue_wait_time.as_secs_f64() * 1000.0);
        
        // Simulate task execution - in real implementation, this would call the actual processors
        match &task.payload {
            TaskPayload::IngestDocument { path, collection } => {
                debug!("Processing document: {} -> {}", path, collection);
                // TODO: Call actual document ingestion
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            TaskPayload::SearchQuery { query, collections } => {
                debug!("Executing search: {} across {:?}", query, collections);
                // TODO: Call actual search
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            TaskPayload::WatchFolder { path } => {
                debug!("Processing folder watch: {}", path);
                // TODO: Call actual folder processing
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            TaskPayload::HealthCheck => {
                debug!("Performing health check");
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            TaskPayload::Maintenance { task: maintenance_task } => {
                debug!("Performing maintenance: {}", maintenance_task);
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
        }
        
        let duration = start_time.elapsed();
        
        // Update statistics
        let mut stats = resource_manager.stats.write().await;
        stats.tasks_processed += 1;
        stats.queue_wait_times_ms.push(queue_wait_time.as_secs_f64() * 1000.0);
        
        // Keep only last 1000 wait times for memory efficiency
        if stats.queue_wait_times_ms.len() > 1000 {
            stats.queue_wait_times_ms.drain(0..500);
        }
        
        match task.priority {
            Priority::High => {
                stats.high_priority_processed += 1;
                stats.average_high_priority_time_ms = 
                    (stats.average_high_priority_time_ms * (stats.high_priority_processed - 1) as f64 + 
                     duration.as_secs_f64() * 1000.0) / stats.high_priority_processed as f64;
            }
            Priority::Low => {
                stats.low_priority_processed += 1;
                stats.average_low_priority_time_ms = 
                    (stats.average_low_priority_time_ms * (stats.low_priority_processed - 1) as f64 + 
                     duration.as_secs_f64() * 1000.0) / stats.low_priority_processed as f64;
            }
        }
        
        debug!("Task completed: {} in {:.2}ms", task.id, duration.as_secs_f64() * 1000.0);
    }
}

/// Statistics reporter that logs resource usage periodically
async fn run_stats_reporter(resource_manager: Arc<ResourceManager>) {
    let mut interval = interval(Duration::from_secs(60)); // Report every minute
    
    loop {
        interval.tick().await;
        
        let stats = resource_manager.stats.read().await;
        let (high_queue_size, low_queue_size, active_count) = resource_manager.get_status().await;
        let mcp_active = *resource_manager.mcp_active.read().await;
        
        info!(
            "Resource stats: processed={} (high={}, low={}), queued={} (high={}, low={}), active={}, mcp_active={}, throttled={}",
            stats.tasks_processed,
            stats.high_priority_processed,
            stats.low_priority_processed,
            high_queue_size + low_queue_size,
            high_queue_size,
            low_queue_size,
            active_count,
            mcp_active,
            stats.tasks_throttled
        );
        
        if stats.high_priority_processed > 0 {
            info!("Avg high-priority task time: {:.2}ms", stats.average_high_priority_time_ms);
        }
        if stats.low_priority_processed > 0 {
            info!("Avg low-priority task time: {:.2}ms", stats.average_low_priority_time_ms);
        }
        
        // Report average queue wait time
        if !stats.queue_wait_times_ms.is_empty() {
            let avg_wait_time = stats.queue_wait_times_ms.iter().sum::<f64>() / stats.queue_wait_times_ms.len() as f64;
            info!("Avg queue wait time: {:.2}ms", avg_wait_time);
        }
    }
}

/// Health monitoring task that enqueues periodic health checks
async fn run_health_monitor(resource_manager: Arc<ResourceManager>) {
    let mut interval = interval(Duration::from_secs(30)); // Health check every 30 seconds
    
    loop {
        interval.tick().await;
        
        resource_manager.enqueue_task(
            TaskPayload::HealthCheck,
            TaskSource::Maintenance,
        ).await;
    }
}

/// Main daemon loop with pure daemon architecture
async fn run_daemon(config: Config, args: DaemonArgs) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting memexd daemon with pure daemon architecture (version 0.2.0)");
    
    // Check for existing instances
    check_existing_instance(&args.pid_file)?;
    
    // Create PID file
    create_pid_file(&args.pid_file)?;
    
    // Ensure PID file is cleaned up on exit
    let pid_file_cleanup = args.pid_file.clone();
    let _cleanup_guard = scopeguard::guard((), move |_| {
        remove_pid_file(&pid_file_cleanup);
    });
    
    // Initialize priority-based resource manager
    info!("Initializing priority-based resource manager");
    let resource_manager = Arc::new(ResourceManager::new());
    
    // Initialize the processing engine
    info!("Initializing ProcessingEngine with configuration");
    let mut engine = ProcessingEngine::with_config(config);
    
    // Start the engine with IPC support
    info!("Starting ProcessingEngine with IPC support");
    let _ipc_client = engine.start_with_ipc().await.map_err(|e| {
        error!("Failed to start processing engine: {}", e);
        e
    })?;
    
    info!("ProcessingEngine started successfully");
    info!("IPC client available for Python integration");
    
    // Start resource manager tasks
    let resource_manager_clone = resource_manager.clone();
    let task_processor = tokio::spawn(async move {
        run_task_processor(resource_manager_clone).await;
    });
    
    let resource_manager_clone = resource_manager.clone();
    let stats_reporter = tokio::spawn(async move {
        run_stats_reporter(resource_manager_clone).await;
    });
    
    let resource_manager_clone = resource_manager.clone();
    let health_monitor = tokio::spawn(async move {
        run_health_monitor(resource_manager_clone).await;
    });
    
    // Enqueue some example tasks to demonstrate the system
    info!("Enqueueing example tasks to demonstrate priority system");
    for i in 0..5 {
        resource_manager.enqueue_task(
            TaskPayload::IngestDocument {
                path: format!("/example/doc_{}.pdf", i),
                collection: "example".to_string(),
            },
            if i < 2 { TaskSource::McpServer } else { TaskSource::BackgroundWatch },
        ).await;
    }
    
    // Set up graceful shutdown handling
    let shutdown_future = setup_signal_handlers();
    
    // Main daemon loop
    info!("memexd daemon is running with priority-based resource management");
    info!("  High-priority queue: MCP operations, current project tasks");
    info!("  Low-priority queue: Background folder ingestion, maintenance");
    info!("  Resource throttling: Active for low-priority when MCP is busy");
    info!("Send SIGTERM or SIGINT to stop.");
    
    // Simulate MCP activity for demonstration
    let resource_manager_clone = resource_manager.clone();
    let mcp_simulator = tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(30));
        let mut mcp_active = false;
        loop {
            interval.tick().await;
            mcp_active = !mcp_active;
            resource_manager_clone.set_mcp_active(mcp_active).await;
        }
    });
    
    // Wait for shutdown signal
    if let Err(e) = shutdown_future.await {
        error!("Error in signal handling: {}", e);
    }
    
    // Graceful shutdown
    info!("Shutting down resource manager...");
    task_processor.abort();
    stats_reporter.abort();
    health_monitor.abort();
    mcp_simulator.abort();
    
    info!("Shutting down ProcessingEngine...");
    if let Err(e) = engine.shutdown().await {
        error!("Error during engine shutdown: {}", e);
    }
    
    // Print final statistics
    let final_stats = resource_manager.get_stats().await;
    info!(
        "Final resource statistics: processed={} (high={}, low={}), throttled={}",
        final_stats.tasks_processed,
        final_stats.high_priority_processed,
        final_stats.low_priority_processed,
        final_stats.tasks_throttled
    );
    
    info!("memexd daemon shutdown complete");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    
    // Initialize comprehensive logging early
    init_logging(&args.log_level, args.foreground)?;
    
    info!("memexd daemon starting with pure daemon architecture and priority-based resource management");
    info!("Command-line arguments: {:?}", args);
    
    // Check if priority mode is enabled
    let priority_mode = env::var("MEMEXD_PRIORITY_MODE").unwrap_or_else(|_| "enabled".to_string());
    info!("Priority-based resource management: {}", priority_mode);
    
    // Log resource management configuration
    if let Ok(resource_throttle) = env::var("MEMEXD_RESOURCE_THROTTLE") {
        info!("Resource throttling: {}", resource_throttle);
    }
    
    // Load configuration
    let config = load_config(&args).map_err(|e| {
        error!("Failed to load configuration: {}", e);
        e
    })?;
    
    // Run the daemon with pure daemon architecture
    if let Err(e) = run_daemon(config, args).await {
        error!("Pure daemon failed: {}", e);
        process::exit(1);
    }
    
    Ok(())
}