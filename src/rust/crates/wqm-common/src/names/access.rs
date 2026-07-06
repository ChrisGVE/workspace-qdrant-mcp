//! Operation-class and consumer vocabularies (N8) that the N51 access-capability
//! registry grants over.
//!
//! N51 (F-06) owns the per-consumer grant *table*; N8 owns the two closed name sets
//! it is indexed by. [`OpClass`] is the engine's operation-class axis (ARCH rev08
//! §3.1 N51); [`Consumer`] is the closed set of four surfaces (ARCH rev08 §6.2:
//! "the consumer set is closed -- 4 surfaces").

/// An engine operation class. `Read` folds in query; `ProxiedRead` is a pure-read
/// op that crosses the daemon socket by design; `Schedule` is a compute- or
/// write-inducing background-work request (ARCH rev08 §3.1 N51, R2/alpha).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpClass {
    /// A read (query folded in): served in-process via the N50 kernel facade.
    Read,
    /// A pure-read op that crosses the daemon socket by design.
    ProxiedRead,
    /// A compute- or write-inducing background-work request.
    Schedule,
    /// Create a stored item.
    Create,
    /// Update a stored item.
    Update,
    /// Delete a stored item.
    Delete,
}

impl OpClass {
    /// Every operation class, in declaration order.
    pub const ALL: [OpClass; 6] = [
        OpClass::Read,
        OpClass::ProxiedRead,
        OpClass::Schedule,
        OpClass::Create,
        OpClass::Update,
        OpClass::Delete,
    ];

    /// The canonical name of this operation class.
    pub const fn name(self) -> &'static str {
        match self {
            OpClass::Read => "read",
            OpClass::ProxiedRead => "proxied_read",
            OpClass::Schedule => "schedule",
            OpClass::Create => "create",
            OpClass::Update => "update",
            OpClass::Delete => "delete",
        }
    }
}

/// A surface that consumes the engine. The set is closed (ARCH rev08 §6.2); N51
/// declares what each consumer may do, keyed by this discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Consumer {
    /// S1 -- the `memexd` daemon (the only writer).
    Daemon,
    /// S2 -- the MCP server.
    Mcp,
    /// S3 -- the `wqm` CLI.
    Cli,
    /// S4 -- the terminal UI.
    Tui,
}

impl Consumer {
    /// Every consumer surface, in declaration order.
    pub const ALL: [Consumer; 4] = [
        Consumer::Daemon,
        Consumer::Mcp,
        Consumer::Cli,
        Consumer::Tui,
    ];

    /// The canonical name of this consumer surface.
    pub const fn name(self) -> &'static str {
        match self {
            Consumer::Daemon => "daemon",
            Consumer::Mcp => "mcp",
            Consumer::Cli => "cli",
            Consumer::Tui => "tui",
        }
    }
}
