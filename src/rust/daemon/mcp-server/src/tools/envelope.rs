//! MCP `CallToolResult` envelope helpers.
//!
//! The TypeScript dispatcher (`tool-dispatcher.ts dispatchToolCall`) wraps every
//! tool result as a single unstructured text content block:
//!
//! - Success: `{ content: [{ type: "text", text: <pretty JSON> }] }`, no `isError`.
//! - Error:   `{ content: [{ type: "text", text: "Error: <msg>" }], isError: true }`.
//! - Unknown tool: `{ content: [{ type: "text", text: "Unknown tool: <name>" }], isError: true }`.
//!
//! These three helpers replicate that contract byte-for-byte.

use rmcp::model::{CallToolResult, Content};

/// Wrap a serializable value as a success `CallToolResult`.
///
/// Serializes `value` using `serde_json::to_string_pretty` (2-space indent) and
/// wraps it in a single `text` content block with no `isError` flag set.
///
/// In practice this never fails — the tool result structs used here are all
/// serializable. If serialization ever does fail the error itself is surfaced
/// as an error envelope rather than a panic.
pub fn ok_text<T: serde::Serialize>(value: &T) -> CallToolResult {
    let text = serde_json::to_string_pretty(value)
        .unwrap_or_else(|e| format!("Error: failed to serialize result: {e}"));
    // CallToolResult is #[non_exhaustive]; use Default + mutation to construct.
    // is_error stays None (absent) — matching TS success contract.
    let mut r = CallToolResult::default();
    r.content = vec![Content::text(text)];
    r
}

/// Wrap an error message as a `CallToolResult` with `isError: true`.
///
/// Text is `"Error: <message>"`, matching the TS dispatcher catch branch.
pub fn error_text(message: &str) -> CallToolResult {
    let mut r = CallToolResult::default();
    r.content = vec![Content::text(format!("Error: {message}"))];
    r.is_error = Some(true);
    r
}

/// Wrap an unknown-tool name as a `CallToolResult` with `isError: true`.
///
/// Text is `"Unknown tool: <name>"`, matching the TS dispatcher default branch.
pub fn unknown_tool(name: &str) -> CallToolResult {
    let mut r = CallToolResult::default();
    r.content = vec![Content::text(format!("Unknown tool: {name}"))];
    r.is_error = Some(true);
    r
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use serde::Serialize;

    use super::*;
    use crate::tools::envelope::{error_text, ok_text, unknown_tool};

    // ── helper ─────────────────────────────────────────────────────────────────

    /// Extract the text from the first content item of a `CallToolResult`.
    fn extract_text(result: &CallToolResult) -> &str {
        let item = result.content.first().expect("content must not be empty");
        // Content = Annotated<RawContent>; .raw is the RawContent; .as_text() gives &RawTextContent
        item.raw
            .as_text()
            .expect("first content must be text")
            .text
            .as_str()
    }

    // ── ok_text ────────────────────────────────────────────────────────────────

    #[derive(Serialize)]
    struct Simple {
        success: bool,
        value: i32,
    }

    #[test]
    fn ok_text_has_no_is_error() {
        let result = ok_text(&Simple {
            success: true,
            value: 42,
        });
        assert!(result.is_error.is_none(), "ok_text must NOT set isError");
    }

    #[test]
    fn ok_text_has_single_text_content() {
        let result = ok_text(&Simple {
            success: true,
            value: 1,
        });
        assert_eq!(
            result.content.len(),
            1,
            "must have exactly one content item"
        );
    }

    #[test]
    fn ok_text_content_is_pretty_json() {
        let result = ok_text(&Simple {
            success: true,
            value: 99,
        });
        let text = extract_text(&result);
        let expected = serde_json::to_string_pretty(&Simple {
            success: true,
            value: 99,
        })
        .unwrap();
        assert_eq!(text, expected);
    }

    #[test]
    fn ok_text_empty_struct() {
        #[derive(Serialize)]
        struct Empty {}
        let result = ok_text(&Empty {});
        let text = extract_text(&result);
        assert_eq!(text, "{}");
        assert!(result.is_error.is_none());
    }

    // ── error_text ─────────────────────────────────────────────────────────────

    #[test]
    fn error_text_sets_is_error_true() {
        let result = error_text("something went wrong");
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn error_text_has_single_content() {
        let result = error_text("oops");
        assert_eq!(result.content.len(), 1);
    }

    #[test]
    fn error_text_content_prefix() {
        let result = error_text("daemon offline");
        let text = extract_text(&result);
        assert_eq!(text, "Error: daemon offline");
    }

    #[test]
    fn error_text_empty_message() {
        let result = error_text("");
        let text = extract_text(&result);
        assert_eq!(text, "Error: ");
    }

    #[test]
    fn error_text_preserves_colons_in_message() {
        let result = error_text("Failed to fetch: timeout");
        let text = extract_text(&result);
        assert_eq!(text, "Error: Failed to fetch: timeout");
    }

    // ── unknown_tool ───────────────────────────────────────────────────────────

    #[test]
    fn unknown_tool_sets_is_error_true() {
        let result = unknown_tool("nonexistent");
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn unknown_tool_has_single_content() {
        let result = unknown_tool("foo");
        assert_eq!(result.content.len(), 1);
    }

    #[test]
    fn unknown_tool_content_text() {
        let result = unknown_tool("my_tool");
        let text = extract_text(&result);
        assert_eq!(text, "Unknown tool: my_tool");
    }

    #[test]
    fn unknown_tool_empty_name() {
        let result = unknown_tool("");
        let text = extract_text(&result);
        assert_eq!(text, "Unknown tool: ");
    }
}
