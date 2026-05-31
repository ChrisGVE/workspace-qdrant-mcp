//! MCP initialize `instructions` string.
//!
//! The constant in this module must remain a byte-for-byte copy of the string
//! produced by the TypeScript server (server.ts, constructor).  MCP clients
//! receive this text during the `initialize` handshake and may surface it to
//! language models; any deviation breaks drop-in parity (CC-4 / RISK-18).
//!
//! Source: `src/typescript/mcp-server/src/server.ts`, lines 93–100.
//! The TS code joins an array of sentences with a single space `' '`.
//! The resulting flat string is captured here verbatim.

/// Verbatim `instructions` field sent in the MCP `initialize` response.
///
/// Byte-for-byte identical to the TypeScript server's instructions string
/// (server.ts constructor, `.join(' ')` of the six-element sentence array).
/// Length: 554 bytes (all ASCII except the em dash U+2014 in sentence 5).
pub const INSTRUCTIONS: &str = concat!(
    "This server provides access to the user's indexed codebase and knowledge libraries. ",
    "ALWAYS use the `search` tool before answering questions about the user's code, ",
    "project structure, or library documentation. ",
    "Use the `rules` tool to check for behavioral preferences before starting work. ",
    "Use `retrieve` to access specific documents when you know the document ID. ",
    "Use `list` to browse project file/folder structure \u{2014} start with format \"summary\" to get an overview. ",
    "Collections: projects (indexed code), libraries (reference docs), rules (behavioral rules).",
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert byte-for-byte match with the TypeScript server's instructions string.
    ///
    /// The expected value was produced by running:
    ///   node -e "console.log([...].join(' '))"
    /// against the exact sentence array in server.ts lines 93-100.
    #[test]
    fn instructions_matches_typescript_verbatim() {
        let expected = concat!(
            "This server provides access to the user's indexed codebase and knowledge libraries.",
            " ALWAYS use the `search` tool before answering questions about the user's code,",
            " project structure, or library documentation.",
            " Use the `rules` tool to check for behavioral preferences before starting work.",
            " Use `retrieve` to access specific documents when you know the document ID.",
            " Use `list` to browse project file/folder structure \u{2014} start with format \"summary\" to get an overview.",
            " Collections: projects (indexed code), libraries (reference docs), rules (behavioral rules).",
        );
        assert_eq!(
            INSTRUCTIONS, expected,
            "INSTRUCTIONS must be byte-identical to the TypeScript server string"
        );
    }

    #[test]
    fn instructions_length_matches_typescript() {
        // The TS string produced by .join(' ') over the six sentences is 554 UTF-16
        // code-units (JavaScript .length) = 556 UTF-8 bytes (Rust .len()).
        // The em dash U+2014 is 1 UTF-16 unit but 3 UTF-8 bytes, causing the +2 delta.
        assert_eq!(
            INSTRUCTIONS.len(),
            556,
            "Expected 556 UTF-8 bytes; got {}. Check for whitespace/character drift.",
            INSTRUCTIONS.len()
        );
    }

    #[test]
    fn instructions_is_valid_utf8() {
        // Confirm the string is valid UTF-8 (it contains one em dash U+2014).
        let _: &str = INSTRUCTIONS;
        assert!(INSTRUCTIONS.contains('\u{2014}'), "em dash must be present");
    }

    #[test]
    fn instructions_does_not_start_or_end_with_whitespace() {
        assert!(
            !INSTRUCTIONS.starts_with(char::is_whitespace),
            "INSTRUCTIONS must not start with whitespace"
        );
        assert!(
            !INSTRUCTIONS.ends_with(char::is_whitespace),
            "INSTRUCTIONS must not end with whitespace"
        );
    }
}
