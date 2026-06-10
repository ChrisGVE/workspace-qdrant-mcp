//! External `$EDITOR` integration for TUI amend actions (#122).
//!
//! Located at: `src/rust/cli/src/tui/editor.rs`
//!
//! Multi-line editing follows the established TUI-tool pattern (lazygit,
//! k9s, gh): suspend the TUI, hand the terminal to the user's editor on a
//! temp file, and resume — rather than re-implementing a text editor inside
//! the TUI. The run loop (`app.rs`) owns the suspend/resume choreography
//! (event-poller pause, raw-mode teardown, terminal re-init); this module
//! only runs the editor process.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};

/// What the user wants to amend; captured at request time (pattern law:
/// confirm/act on data captured when the action was requested, never on
/// `items[selected]` at completion time).
#[derive(Debug, Clone)]
pub enum EditorRequest {
    /// Amend a scratchpad entry (full row state captured).
    Scratchpad {
        tenant_id: String,
        title: String,
        tags_json: String,
        content: String,
    },
    /// Amend a rule's content (label-addressed).
    Rule { label: String, content: String },
}

impl EditorRequest {
    /// The text placed in the editor buffer.
    pub fn initial_content(&self) -> &str {
        match self {
            EditorRequest::Scratchpad { content, .. } => content,
            EditorRequest::Rule { content, .. } => content,
        }
    }
}

/// Resolve the editor command: `$VISUAL`, then `$EDITOR`, then `vi`.
fn editor_command() -> String {
    std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "vi".to_string())
}

/// Temp file path for the edit buffer (OS temp dir, pid-scoped — the TUI
/// runs one editor at a time).
fn edit_buffer_path() -> PathBuf {
    std::env::temp_dir().join(format!("wqm-edit-{}.md", std::process::id()))
}

/// Run the user's editor on `initial` and return the edited text, or `None`
/// when the content is unchanged. The caller must have released the
/// terminal (raw mode off, alternate screen left, event poller paused)
/// before calling this.
pub fn edit_text(initial: &str) -> Result<Option<String>> {
    let path = edit_buffer_path();
    {
        let mut f = std::fs::File::create(&path).context("create edit buffer")?;
        f.write_all(initial.as_bytes())
            .context("write edit buffer")?;
    }

    let cmdline = editor_command();
    // Support multi-word editor settings ("code --wait").
    let mut parts = cmdline.split_whitespace();
    let program = parts.next().unwrap_or("vi");
    let status = std::process::Command::new(program)
        .args(parts)
        .arg(&path)
        .status();

    let result = match status {
        Ok(s) if s.success() => {
            let edited = std::fs::read_to_string(&path).context("read edit buffer")?;
            let edited = edited.trim_end().to_string();
            if edited == initial.trim_end() || edited.is_empty() {
                Ok(None)
            } else {
                Ok(Some(edited))
            }
        }
        Ok(s) => Err(anyhow::anyhow!("editor exited with {}", s)),
        Err(e) => Err(anyhow::anyhow!("failed to launch '{}': {}", cmdline, e)),
    };

    let _ = std::fs::remove_file(&path);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_request_initial_content() {
        let r = EditorRequest::Rule {
            label: "l".into(),
            content: "rule body".into(),
        };
        assert_eq!(r.initial_content(), "rule body");
        let s = EditorRequest::Scratchpad {
            tenant_id: "t".into(),
            title: "x".into(),
            tags_json: "[]".into(),
            content: "note body".into(),
        };
        assert_eq!(s.initial_content(), "note body");
    }

    #[test]
    fn edit_buffer_path_is_pid_scoped_temp() {
        let p = edit_buffer_path();
        assert!(p.starts_with(std::env::temp_dir()));
        assert!(p
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains(&std::process::id().to_string()));
    }
}
