//! Project detection helpers for ingest commands

/// Detect tenant_id from current working directory
pub fn detect_tenant_id() -> String {
    // Try to get project root from git
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
    {
        if output.status.success() {
            if let Ok(path) = String::from_utf8(output.stdout) {
                let path = path.trim();
                // Normalize to a tenant_id
                return path
                    .replace(['/', '\\', ' '], "_")
                    .trim_start_matches('_')
                    .to_string();
            }
        }
    }

    // Fallback to current directory
    std::env::current_dir()
        .map(|p| {
            p.to_string_lossy()
                .replace(['/', '\\', ' '], "_")
                .trim_start_matches('_')
                .to_string()
        })
        .unwrap_or_else(|_| "default".to_string())
}

/// Detect current git branch
pub fn detect_branch() -> String {
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
    {
        if output.status.success() {
            if let Ok(branch) = String::from_utf8(output.stdout) {
                return branch.trim().to_string();
            }
        }
    }
    "main".to_string()
}
