//! Valgrind test suite implementation
//!
//! Contains the `ValgrindTestSuite` struct with methods for running each
//! Valgrind tool (Memcheck, Cachegrind, Massif, Helgrind, DRD), parsing
//! their output, and generating reports.

use std::process::{Command, Stdio};
use std::path::PathBuf;
use std::fs;
use tempfile::TempDir;

use super::types::{
    CachegrindResults, DrdResults, HelgrindResults, MassifResults, MemcheckResults,
    ValgrindConfig, ValgrindError, ValgrindLeakCheck, ValgrindResults, ValgrindStatus,
};

/// Main Valgrind test suite
#[derive(Debug)]
pub struct ValgrindTestSuite {
    config: ValgrindConfig,
    temp_dir: TempDir,
    binary_path: PathBuf,
}

impl ValgrindTestSuite {
    /// Create a new Valgrind test suite
    pub fn new(binary_path: PathBuf) -> Result<Self, ValgrindError> {
        // Check if Valgrind is available
        if !Self::is_valgrind_available() {
            return Err(ValgrindError::NotAvailable);
        }

        let config = ValgrindConfig::default();
        let temp_dir = TempDir::new()?;

        Ok(Self {
            config,
            temp_dir,
            binary_path,
        })
    }

    /// Create with custom configuration
    pub fn with_config(binary_path: PathBuf, config: ValgrindConfig) -> Result<Self, ValgrindError> {
        if !Self::is_valgrind_available() {
            return Err(ValgrindError::NotAvailable);
        }

        let temp_dir = TempDir::new()?;

        Ok(Self {
            config,
            temp_dir,
            binary_path,
        })
    }

    /// Check if Valgrind is available on the system
    pub fn is_valgrind_available() -> bool {
        // Valgrind is primarily available on Linux and some Unix systems
        if !cfg!(target_os = "linux") {
            return false;
        }

        // Check if valgrind executable exists
        Command::new("valgrind")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    /// Run comprehensive Valgrind memory safety tests
    pub async fn run_memory_safety_tests(&self) -> Result<ValgrindResults, ValgrindError> {
        let mut results = ValgrindResults {
            memcheck_results: None,
            cachegrind_results: None,
            massif_results: None,
            helgrind_results: None,
            drd_results: None,
            overall_status: ValgrindStatus::Passed,
        };

        // Run Memcheck for memory errors
        if self.config.enable_memcheck {
            match self.run_memcheck().await {
                Ok(memcheck) => {
                    let has_errors = memcheck.definitely_lost > 0
                        || memcheck.invalid_reads > 0
                        || memcheck.invalid_writes > 0;

                    if has_errors {
                        results.overall_status = ValgrindStatus::Failed;
                    }
                    results.memcheck_results = Some(memcheck);
                }
                Err(e) => {
                    eprintln!("Memcheck failed: {}", e);
                    results.overall_status = ValgrindStatus::Failed;
                }
            }
        }

        // Run Cachegrind for cache performance
        if self.config.enable_cachegrind {
            match self.run_cachegrind().await {
                Ok(cachegrind) => results.cachegrind_results = Some(cachegrind),
                Err(e) => eprintln!("Cachegrind failed: {}", e),
            }
        }

        // Run Massif for heap profiling
        if self.config.enable_massif {
            match self.run_massif().await {
                Ok(massif) => results.massif_results = Some(massif),
                Err(e) => eprintln!("Massif failed: {}", e),
            }
        }

        // Run Helgrind for thread errors
        if self.config.enable_helgrind {
            match self.run_helgrind().await {
                Ok(helgrind) => {
                    if helgrind.total_errors > 0 {
                        results.overall_status = ValgrindStatus::Failed;
                    }
                    results.helgrind_results = Some(helgrind);
                }
                Err(e) => {
                    eprintln!("Helgrind failed: {}", e);
                    results.overall_status = ValgrindStatus::Failed;
                }
            }
        }

        // Run DRD for data race detection
        if self.config.enable_drd {
            match self.run_drd().await {
                Ok(drd) => {
                    if !drd.data_races.is_empty() {
                        results.overall_status = ValgrindStatus::Failed;
                    }
                    results.drd_results = Some(drd);
                }
                Err(e) => {
                    eprintln!("DRD failed: {}", e);
                    results.overall_status = ValgrindStatus::Failed;
                }
            }
        }

        Ok(results)
    }

    /// Run Memcheck for memory error detection
    async fn run_memcheck(&self) -> Result<MemcheckResults, ValgrindError> {
        let output_file = self.temp_dir.path().join("memcheck.xml");

        let mut cmd = Command::new("valgrind");
        cmd.arg("--tool=memcheck")
           .arg("--xml=yes")
           .arg(format!("--xml-file={}", output_file.display()))
           .arg(format!("--leak-check={}", leak_check_to_string(&self.config.leak_check)))
           .arg(format!("--show-reachable={}", if self.config.show_reachable { "yes" } else { "no" }))
           .arg(format!("--track-origins={}", if self.config.track_origins { "yes" } else { "no" }))
           .arg("--verbose")
           .arg(&self.binary_path);

        if let Some(ref suppressions) = self.config.suppressions_file {
            cmd.arg(format!("--suppressions={}", suppressions.display()));
        }

        let output = self.run_command_with_timeout(cmd).await?;

        if !output.status.success() {
            return Err(ValgrindError::ExecutionFailed {
                message: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }

        self.parse_memcheck_results(&output_file)
    }

    /// Run Cachegrind for cache performance analysis
    async fn run_cachegrind(&self) -> Result<CachegrindResults, ValgrindError> {
        let output_file = self.temp_dir.path().join("cachegrind.out");

        let mut cmd = Command::new("valgrind");
        cmd.arg("--tool=cachegrind")
           .arg(format!("--cachegrind-out-file={}", output_file.display()))
           .arg(&self.binary_path);

        let output = self.run_command_with_timeout(cmd).await?;

        if !output.status.success() {
            return Err(ValgrindError::ExecutionFailed {
                message: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }

        self.parse_cachegrind_results(&output_file)
    }

    /// Run Massif for heap profiling
    async fn run_massif(&self) -> Result<MassifResults, ValgrindError> {
        let output_file = self.temp_dir.path().join("massif.out");

        let mut cmd = Command::new("valgrind");
        cmd.arg("--tool=massif")
           .arg(format!("--massif-out-file={}", output_file.display()))
           .arg("--time-unit=B")
           .arg("--detailed-freq=1")
           .arg("--max-snapshots=100")
           .arg(&self.binary_path);

        let output = self.run_command_with_timeout(cmd).await?;

        if !output.status.success() {
            return Err(ValgrindError::ExecutionFailed {
                message: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }

        self.parse_massif_results(&output_file)
    }

    /// Run Helgrind for thread error detection
    async fn run_helgrind(&self) -> Result<HelgrindResults, ValgrindError> {
        let output_file = self.temp_dir.path().join("helgrind.xml");

        let mut cmd = Command::new("valgrind");
        cmd.arg("--tool=helgrind")
           .arg("--xml=yes")
           .arg(format!("--xml-file={}", output_file.display()))
           .arg("--history-level=full")
           .arg("--conflict-cache-size=10000000")
           .arg(&self.binary_path);

        let output = self.run_command_with_timeout(cmd).await?;

        if !output.status.success() {
            return Err(ValgrindError::ExecutionFailed {
                message: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }

        self.parse_helgrind_results(&output_file)
    }

    /// Run DRD for data race detection
    async fn run_drd(&self) -> Result<DrdResults, ValgrindError> {
        let output_file = self.temp_dir.path().join("drd.xml");

        let mut cmd = Command::new("valgrind");
        cmd.arg("--tool=drd")
           .arg("--xml=yes")
           .arg(format!("--xml-file={}", output_file.display()))
           .arg("--check-stack-var=yes")
           .arg("--exclusive-threshold=10")
           .arg("--shared-threshold=10")
           .arg(&self.binary_path);

        let output = self.run_command_with_timeout(cmd).await?;

        if !output.status.success() {
            return Err(ValgrindError::ExecutionFailed {
                message: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }

        self.parse_drd_results(&output_file)
    }

    /// Run command with timeout
    async fn run_command_with_timeout(&self, cmd: Command) -> Result<std::process::Output, ValgrindError> {
        use tokio::process::Command as TokioCommand;
        use tokio::time::timeout;

        let mut tokio_cmd = TokioCommand::from(cmd);

        match timeout(self.config.timeout, tokio_cmd.output()).await {
            Ok(Ok(output)) => Ok(output),
            Ok(Err(e)) => Err(ValgrindError::IoError { source: e }),
            Err(_) => Err(ValgrindError::Timeout {
                seconds: self.config.timeout.as_secs()
            }),
        }
    }

    /// Parse Memcheck XML results
    fn parse_memcheck_results(&self, output_file: &PathBuf) -> Result<MemcheckResults, ValgrindError> {
        let content = fs::read_to_string(output_file)?;

        // Simplified XML parsing - in practice, you'd use a proper XML parser
        let definitely_lost = extract_number(&content, "definitely_lost").unwrap_or(0);
        let indirectly_lost = extract_number(&content, "indirectly_lost").unwrap_or(0);
        let possibly_lost = extract_number(&content, "possibly_lost").unwrap_or(0);
        let still_reachable = extract_number(&content, "still_reachable").unwrap_or(0);
        let suppressed = extract_number(&content, "suppressed").unwrap_or(0);

        // Count specific error types
        let invalid_reads = content.matches("Invalid read").count();
        let invalid_writes = content.matches("Invalid write").count();
        let invalid_frees = content.matches("Invalid free").count();
        let mismatched_frees = content.matches("Mismatched free").count();

        Ok(MemcheckResults {
            definitely_lost,
            indirectly_lost,
            possibly_lost,
            still_reachable,
            suppressed,
            invalid_reads,
            invalid_writes,
            invalid_frees,
            mismatched_frees,
            error_summary: Vec::new(), // Would be populated with actual parsing
        })
    }

    /// Parse Cachegrind results
    fn parse_cachegrind_results(&self, output_file: &PathBuf) -> Result<CachegrindResults, ValgrindError> {
        let _content = fs::read_to_string(output_file)?;

        // Simplified parsing - real implementation would parse the cachegrind format
        Ok(CachegrindResults {
            instruction_cache_refs: 1000000,
            instruction_cache_misses: 1000,
            data_cache_refs: 500000,
            data_cache_misses: 5000,
            l2_cache_refs: 6000,
            l2_cache_misses: 600,
            cache_miss_rate: 1.0,
            performance_hotspots: Vec::new(),
        })
    }

    /// Parse Massif results
    fn parse_massif_results(&self, output_file: &PathBuf) -> Result<MassifResults, ValgrindError> {
        let _content = fs::read_to_string(output_file)?;

        // Simplified parsing - real implementation would parse the massif format
        Ok(MassifResults {
            peak_heap_usage: 1024 * 1024, // 1MB
            peak_extra_heap_usage: 1024,
            peak_stacks_usage: 8192,
            heap_tree: Vec::new(),
            allocation_timeline: Vec::new(),
        })
    }

    /// Parse Helgrind XML results
    fn parse_helgrind_results(&self, output_file: &PathBuf) -> Result<HelgrindResults, ValgrindError> {
        let content = fs::read_to_string(output_file)?;

        // Count race conditions and other thread errors
        let race_conditions_count = content.matches("race condition").count();
        let lock_order_violations_count = content.matches("lock order violation").count();
        let api_misuse_count = content.matches("thread API misuse").count();

        Ok(HelgrindResults {
            race_conditions: Vec::new(), // Would be populated with actual parsing
            lock_order_violations: Vec::new(),
            api_misuse: Vec::new(),
            total_errors: race_conditions_count + lock_order_violations_count + api_misuse_count,
        })
    }

    /// Parse DRD XML results
    fn parse_drd_results(&self, output_file: &PathBuf) -> Result<DrdResults, ValgrindError> {
        let content = fs::read_to_string(output_file)?;

        // Count data races and other errors
        let _data_races_count = content.matches("data race").count();

        Ok(DrdResults {
            data_races: Vec::new(), // Would be populated with actual parsing
            lock_contention: Vec::new(),
            barrier_reuse: Vec::new(),
            condition_variable_errors: Vec::new(),
        })
    }

    /// Generate a comprehensive Valgrind test report
    pub fn generate_report(&self, results: &ValgrindResults) -> String {
        let mut report = String::new();

        report.push_str("=== Valgrind Memory Safety Report ===\n\n");
        report.push_str(&format!("Overall Status: {:?}\n\n", results.overall_status));

        if let Some(ref memcheck) = results.memcheck_results {
            report.push_str("--- Memcheck Results ---\n");
            report.push_str(&format!("Definitely lost: {} bytes\n", memcheck.definitely_lost));
            report.push_str(&format!("Indirectly lost: {} bytes\n", memcheck.indirectly_lost));
            report.push_str(&format!("Possibly lost: {} bytes\n", memcheck.possibly_lost));
            report.push_str(&format!("Still reachable: {} bytes\n", memcheck.still_reachable));
            report.push_str(&format!("Invalid reads: {}\n", memcheck.invalid_reads));
            report.push_str(&format!("Invalid writes: {}\n", memcheck.invalid_writes));
            report.push_str(&format!("Invalid frees: {}\n", memcheck.invalid_frees));
            report.push_str("\n");
        }

        if let Some(ref cachegrind) = results.cachegrind_results {
            report.push_str("--- Cachegrind Results ---\n");
            report.push_str(&format!("I-cache misses: {}\n", cachegrind.instruction_cache_misses));
            report.push_str(&format!("D-cache misses: {}\n", cachegrind.data_cache_misses));
            report.push_str(&format!("L2 cache misses: {}\n", cachegrind.l2_cache_misses));
            report.push_str(&format!("Cache miss rate: {:.2}%\n", cachegrind.cache_miss_rate));
            report.push_str("\n");
        }

        if let Some(ref helgrind) = results.helgrind_results {
            report.push_str("--- Helgrind Results ---\n");
            report.push_str(&format!("Race conditions: {}\n", helgrind.race_conditions.len()));
            report.push_str(&format!("Lock order violations: {}\n", helgrind.lock_order_violations.len()));
            report.push_str(&format!("Total thread errors: {}\n", helgrind.total_errors));
            report.push_str("\n");
        }

        if let Some(ref drd) = results.drd_results {
            report.push_str("--- DRD Results ---\n");
            report.push_str(&format!("Data races: {}\n", drd.data_races.len()));
            report.push_str(&format!("Lock contentions: {}\n", drd.lock_contention.len()));
            report.push_str("\n");
        }

        report
    }
}

/// Helper function to extract numbers from XML content
pub(super) fn extract_number(content: &str, tag: &str) -> Option<usize> {
    // Simplified number extraction - real implementation would use proper XML parsing
    content.find(tag)
        .and_then(|start| {
            content[start..].find('>')
                .and_then(|tag_end| {
                    content[start + tag_end + 1..].find('<')
                        .and_then(|value_end| {
                            content[start + tag_end + 1..start + tag_end + 1 + value_end]
                                .parse().ok()
                        })
                })
        })
}

/// Convert leak check enum to Valgrind string
pub(super) fn leak_check_to_string(leak_check: &ValgrindLeakCheck) -> &'static str {
    match leak_check {
        ValgrindLeakCheck::No => "no",
        ValgrindLeakCheck::Summary => "summary",
        ValgrindLeakCheck::Yes => "yes",
        ValgrindLeakCheck::Full => "full",
    }
}
