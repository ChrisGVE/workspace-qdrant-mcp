//! Man command - man page generation and installation
//!
//! Generates roff-format man pages from clap command definitions.
//! Usage: wqm man install | wqm man generate [--output-dir <dir>]

use anyhow::{Context, Result};
use clap::{Args, Command, Subcommand};
use clap_mangen::Man;
use std::path::PathBuf;

use crate::output;

/// Man command arguments
#[derive(Args)]
pub struct ManArgs {
    #[command(subcommand)]
    command: ManCommand,
}

/// Man subcommands
#[derive(Subcommand)]
enum ManCommand {
    /// Generate man pages to a directory (or stdout for top-level page)
    Generate {
        /// Output directory (generates all pages; omit for stdout)
        #[arg(long)]
        output_dir: Option<PathBuf>,
    },

    /// Install man pages to ~/.local/share/man/man1/
    Install,
}

/// Execute man command
pub async fn execute(args: ManArgs, cmd: &mut Command) -> Result<()> {
    match args.command {
        ManCommand::Generate { output_dir } => generate(cmd, output_dir).await,
        ManCommand::Install => install(cmd).await,
    }
}

/// Generate man pages
async fn generate(cmd: &mut Command, output_dir: Option<PathBuf>) -> Result<()> {
    match output_dir {
        Some(dir) => {
            std::fs::create_dir_all(&dir)
                .with_context(|| format!("Failed to create output directory: {}", dir.display()))?;
            let count = write_man_pages(cmd, &dir)?;
            output::success(format!("Generated {} man pages in {}", count, dir.display()));
        }
        None => {
            // Write top-level page to stdout
            let man = Man::new(cmd.clone());
            let mut buf = Vec::new();
            man.render(&mut buf)?;
            std::io::Write::write_all(&mut std::io::stdout(), &buf)?;
        }
    }
    Ok(())
}

/// Install man pages to user-local man directory
async fn install(cmd: &mut Command) -> Result<()> {
    let man_dir = man_install_dir()?;
    std::fs::create_dir_all(&man_dir)
        .with_context(|| format!("Failed to create man directory: {}", man_dir.display()))?;

    let count = write_man_pages(cmd, &man_dir)?;
    output::success(format!("Installed {} man pages to {}", count, man_dir.display()));
    output::info("View with: man wqm");
    Ok(())
}

/// Install man pages non-interactively (called from service install)
pub fn install_man_pages(cmd: &mut Command) -> Result<()> {
    let man_dir = man_install_dir()?;
    std::fs::create_dir_all(&man_dir)
        .with_context(|| format!("Failed to create man directory: {}", man_dir.display()))?;
    let count = write_man_pages(cmd, &man_dir)?;
    output::kv("Man pages", format!("{} pages installed to {}", count, man_dir.display()));
    Ok(())
}

/// Get the user-local man page installation directory
fn man_install_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".local/share/man/man1"))
}

/// Write all man pages (top-level + subcommands) to a directory.
/// Returns the number of pages written.
fn write_man_pages(cmd: &mut Command, dir: &std::path::Path) -> Result<usize> {
    let mut count = 0;
    let name = cmd.get_name().to_string();

    // Top-level man page
    let man = Man::new(cmd.clone());
    let mut buf = Vec::new();
    man.render(&mut buf)?;
    std::fs::write(dir.join(format!("{}.1", name)), &buf)
        .with_context(|| format!("Failed to write {}.1", name))?;
    count += 1;

    // Subcommand man pages
    for sub in cmd.get_subcommands() {
        let sub_name = sub.get_name().to_string();
        let sub_man = Man::new(sub.clone());
        let mut buf = Vec::new();
        sub_man.render(&mut buf)?;
        std::fs::write(dir.join(format!("{}-{}.1", name, sub_name)), &buf)
            .with_context(|| format!("Failed to write {}-{}.1", name, sub_name))?;
        count += 1;
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_command() -> Command {
        Command::new("wqm")
            .about("Workspace Qdrant MCP CLI")
            .subcommand(Command::new("search").about("Search collections"))
            .subcommand(Command::new("status").about("System status"))
    }

    #[test]
    fn test_man_page_generation() {
        let cmd = test_command();
        let man = Man::new(cmd.clone());
        let mut buf = Vec::new();
        man.render(&mut buf).unwrap();
        let content = String::from_utf8(buf).unwrap();
        // Verify roff format header
        assert!(content.contains(".TH"), "Man page should contain .TH header");
        assert!(content.contains("wqm"), "Man page should reference command name");
    }

    #[test]
    fn test_write_man_pages() {
        let dir = tempfile::tempdir().unwrap();
        let mut cmd = test_command();
        let count = write_man_pages(&mut cmd, dir.path()).unwrap();
        // Top-level + 2 subcommands = 3
        assert_eq!(count, 3);
        assert!(dir.path().join("wqm.1").exists());
        assert!(dir.path().join("wqm-search.1").exists());
        assert!(dir.path().join("wqm-status.1").exists());
    }

    #[test]
    fn test_man_install_dir() {
        let dir = man_install_dir().unwrap();
        assert!(dir.to_str().unwrap().contains(".local/share/man/man1"));
    }

    #[test]
    fn test_man_page_contains_about() {
        let cmd = test_command();
        let man = Man::new(cmd.clone());
        let mut buf = Vec::new();
        man.render(&mut buf).unwrap();
        let content = String::from_utf8(buf).unwrap();
        assert!(content.contains("Workspace Qdrant MCP CLI"));
    }
}
