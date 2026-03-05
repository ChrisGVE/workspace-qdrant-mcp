//! Branch management subcommands

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::output;

pub(super) async fn branch_list() -> Result<()> {
    output::section("Git Branches");

    let output_result = std::process::Command::new("git")
        .args(["branch", "-a"])
        .output();

    match output_result {
        Ok(out) if out.status.success() => {
            let branches = String::from_utf8_lossy(&out.stdout);
            for line in branches.lines() {
                if line.contains("* ") {
                    output::info(format!("{} (current)", line));
                } else {
                    println!("{}", line);
                }
            }
        }
        Ok(_) => {
            output::warning("Not a git repository");
        }
        Err(e) => {
            output::error(format!("Failed to list branches: {}", e));
        }
    }

    Ok(())
}

pub(super) async fn branch_info() -> Result<()> {
    output::section("Current Branch");

    let branch = std::process::Command::new("git")
        .args(["branch", "--show-current"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        });

    match branch {
        Some(b) => {
            output::kv("Branch", &b);

            // Get last commit info
            if let Ok(out) = std::process::Command::new("git")
                .args(["log", "-1", "--format=%h %s"])
                .output()
            {
                if out.status.success() {
                    let commit = String::from_utf8_lossy(&out.stdout);
                    output::kv("Last Commit", commit.trim());
                }
            }
        }
        None => {
            output::warning("Not a git repository or no branch checked out");
        }
    }

    Ok(())
}

pub(super) async fn branch_switch(branch: &str) -> Result<()> {
    output::section(format!("Switch Branch: {}", branch));

    output::info("Branch switching affects which content gets indexed.");
    output::info(format!("Documents will be tagged with branch='{}'", branch));
    output::separator();

    // Git checkout
    let status = std::process::Command::new("git")
        .args(["checkout", branch])
        .status();

    match status {
        Ok(s) if s.success() => {
            output::success(format!("Switched to branch '{}'", branch));

            // Signal daemon to re-index
            if let Ok(mut client) = DaemonClient::connect_default().await {
                let request = crate::grpc::proto::RefreshSignalRequest {
                    queue_type: crate::grpc::proto::QueueType::WatchedProjects as i32,
                    lsp_languages: vec![],
                    grammar_languages: vec![],
                };

                if client.system().send_refresh_signal(request).await.is_ok() {
                    output::info("Daemon notified to update index for new branch");
                }
            }
        }
        Ok(_) => {
            output::error("Failed to switch branch");
        }
        Err(e) => {
            output::error(format!("Git error: {}", e));
        }
    }

    Ok(())
}
