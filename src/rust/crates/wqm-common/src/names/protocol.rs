//! The supported MCP protocol revisions (N8).
//!
//! MCP-SURFACE.md §5.1 makes this a constant with one owner -- N8, "like the
//! collection map" -- because AS-F003 was a server that *echoed* whatever version
//! it was sent, including `"2024-10-07"`, which has never existed. A closed list
//! held in one place is what makes echoing impossible: [`negotiate`] can only
//! return a member of it.

/// The revisions this server serves, preferred first (§5.1's table).
///
/// - `2026-07-28` -- preferred; carries SEP-2106, under which the surface's
///   composition/conditional/reference schema keywords are normative.
/// - `2025-11-25`, `2025-06-18` -- supported; predate SEP-2106, and server-side
///   re-validation covers the gap.
///
/// If the preferred revision slips, the *list* is unchanged and only its first
/// member moves: the design degrades along this column rather than breaking.
pub const SUPPORTED_PROTOCOLS: [&str; 3] = ["2026-07-28", "2025-11-25", "2025-06-18"];

/// The revision served when the client asks for nothing in particular.
pub const PREFERRED_PROTOCOL: &str = SUPPORTED_PROTOCOLS[0];

/// What a handshake settled on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Negotiated {
    /// What the client asked for, verbatim, so `status.handshake` can report it.
    /// `None` when the client named no version.
    pub requested: Option<String>,
    /// What this server will actually speak. Always a member of
    /// [`SUPPORTED_PROTOCOLS`].
    pub served: &'static str,
    /// Whether the client was served something other than what it asked for --
    /// the condition §4.4's `protocol_downgraded` notice discloses on the first
    /// tool call of the session.
    pub downgraded: bool,
}

/// Settle on a protocol revision. **Never echoes**: an unrecognised request is
/// served the preferred revision and flagged as a downgrade, so the client is
/// told rather than left believing a version that does not exist was accepted.
pub fn negotiate(requested: Option<&str>) -> Negotiated {
    match requested {
        None => Negotiated {
            requested: None,
            served: PREFERRED_PROTOCOL,
            // Asking for nothing is not being downgraded from anything.
            downgraded: false,
        },
        Some(asked) => {
            let served = SUPPORTED_PROTOCOLS
                .into_iter()
                .find(|supported| *supported == asked);
            Negotiated {
                requested: Some(asked.to_owned()),
                served: served.unwrap_or(PREFERRED_PROTOCOL),
                downgraded: served.is_none(),
            }
        }
    }
}
