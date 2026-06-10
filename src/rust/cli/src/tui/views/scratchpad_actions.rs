//! Act-on-object machinery for the scratchpad browser (#122):
//! delete (typed confirm), reassign (target prompt), amend (external editor).
//!
//! Located at: `src/rust/cli/src/tui/views/scratchpad_actions.rs`
//!
//! Pattern law: every action captures the FULL target row at request time —
//! a periodic refresh can rebuild/reorder `items` while a modal is open, so
//! the action must never read `items[selected]` at confirm time.
//!
//! Neighbors: `scratchpad.rs` (browser state + draw), `confirm.rs` (modals),
//! `../editor.rs` (external-editor amend), `../commands.rs` (enqueue ops).

use std::collections::HashMap;

use wqm_common::constants::TENANT_GLOBAL;

use super::confirm::{TargetPrompt, TypedConfirm};
use super::scratchpad::ScratchpadBrowser;
use super::scratchpad_data::ScratchpadRow;
use crate::tui::editor::EditorRequest;

impl ScratchpadBrowser {
    // ── Delete (typed confirm) ───────────────────────────────────────────

    /// Whether the typed-name delete confirmation modal is open.
    pub fn delete_confirm_open(&self) -> bool {
        self.delete_confirm.is_some()
    }

    /// Open a typed-name delete confirmation for the selected entry,
    /// capturing the full row.
    pub fn request_delete(&mut self) {
        if let Some(entry) = self.items().get(self.selected_index()).cloned() {
            let name = confirm_name(&entry);
            self.delete_confirm = Some((TypedConfirm::new(name), entry));
        }
    }

    /// Mutable reference to the pending delete confirm (for key input).
    pub fn delete_confirm_mut(&mut self) -> Option<&mut TypedConfirm> {
        self.delete_confirm.as_mut().map(|(tc, _)| tc)
    }

    /// Return the row captured when the confirm opened, consuming the modal.
    /// `None` when not open or the input does not match yet.
    pub fn take_delete_if_confirmed(&mut self) -> Option<ScratchpadRow> {
        if self
            .delete_confirm
            .as_ref()
            .is_some_and(|(c, _)| c.matches())
        {
            self.delete_confirm.take().map(|(_, row)| row)
        } else {
            if let Some((ref mut c, _)) = self.delete_confirm {
                c.mark_rejected();
            }
            None
        }
    }

    /// Cancel and close the delete confirmation modal.
    pub fn cancel_delete(&mut self) {
        self.delete_confirm = None;
    }

    // ── Reassign (target prompt) ─────────────────────────────────────────

    /// Whether the reassign-target prompt is open.
    pub fn reassign_open(&self) -> bool {
        self.reassign_prompt.is_some()
    }

    /// Open the reassign prompt for the selected entry, capturing the row.
    pub fn request_reassign(&mut self) {
        if let Some(entry) = self.items().get(self.selected_index()).cloned() {
            self.reassign_prompt = Some((TargetPrompt::default(), entry));
        }
    }

    /// Mutable reference to the reassign prompt (for key input).
    pub fn reassign_prompt_mut(&mut self) -> Option<&mut TargetPrompt> {
        self.reassign_prompt.as_mut().map(|(p, _)| p)
    }

    /// Confirm the reassign: returns the captured row and the resolved
    /// target tenant id, consuming the prompt.
    pub fn take_reassign(
        &mut self,
        names: &HashMap<String, String>,
    ) -> Option<(ScratchpadRow, String)> {
        let (prompt, row) = self.reassign_prompt.take()?;
        let target = resolve_target_tenant(&prompt.input, names);
        Some((row, target))
    }

    /// Cancel and close the reassign prompt.
    pub fn cancel_reassign(&mut self) {
        self.reassign_prompt = None;
    }

    // ── Amend (external editor) ──────────────────────────────────────────

    /// Build an external-editor request for the selected entry.
    pub fn request_edit(&self) -> Option<EditorRequest> {
        self.items()
            .get(self.selected_index())
            .map(|entry| EditorRequest::Scratchpad {
                tenant_id: entry.tenant_id.clone(),
                title: entry.title.clone(),
                tags_json: entry.tags.clone(),
                content: entry.content.clone(),
            })
    }
}

/// Name the user must type in the destructive confirmation: the title, or
/// the first content line for untitled entries, truncated to 40 chars.
pub(super) fn confirm_name(entry: &ScratchpadRow) -> String {
    let base = if entry.title.is_empty() || entry.title == "(untitled)" {
        entry.content.lines().next().unwrap_or("")
    } else {
        &entry.title
    };
    base.chars().take(40).collect()
}

/// Resolve the reassign-prompt input to a tenant id: empty → global,
/// a known project name → its tenant id, anything else → used verbatim
/// (assumed to be a tenant id).
pub fn resolve_target_tenant(input: &str, names: &HashMap<String, String>) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() || trimmed == TENANT_GLOBAL {
        return TENANT_GLOBAL.to_string();
    }
    for (tenant_id, name) in names {
        if name == trimmed {
            return tenant_id.clone();
        }
    }
    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(title: &str, content: &str) -> ScratchpadRow {
        ScratchpadRow {
            title: title.to_string(),
            content: content.to_string(),
            tags: "[]".to_string(),
            tenant_id: "t1".to_string(),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    #[test]
    fn confirm_name_prefers_title() {
        assert_eq!(confirm_name(&row("my note", "body")), "my note");
    }

    #[test]
    fn confirm_name_falls_back_to_first_content_line() {
        assert_eq!(
            confirm_name(&row("(untitled)", "first line\nsecond")),
            "first line"
        );
    }

    #[test]
    fn confirm_name_truncates_to_40_chars() {
        let long = "x".repeat(80);
        assert_eq!(confirm_name(&row(&long, "")).chars().count(), 40);
    }

    #[test]
    fn resolve_target_empty_is_global() {
        let names = HashMap::new();
        assert_eq!(resolve_target_tenant("", &names), TENANT_GLOBAL);
        assert_eq!(resolve_target_tenant("  ", &names), TENANT_GLOBAL);
    }

    #[test]
    fn resolve_target_project_name_maps_to_tenant() {
        let mut names = HashMap::new();
        names.insert("abc123".to_string(), "my-project".to_string());
        assert_eq!(resolve_target_tenant("my-project", &names), "abc123");
    }

    #[test]
    fn resolve_target_unknown_input_used_verbatim() {
        let names = HashMap::new();
        assert_eq!(
            resolve_target_tenant("raw-tenant-id", &names),
            "raw-tenant-id"
        );
    }
}
