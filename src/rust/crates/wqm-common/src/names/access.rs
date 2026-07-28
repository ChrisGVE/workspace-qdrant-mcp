//! Operation-class and consumer vocabularies (N8) that the N51 access-capability
//! registry grants over.
//!
//! N51 (`P04-GT025`) owns the per-consumer grant *table*; N8 owns the two name sets
//! it is indexed by. [`OpClass`] is the engine's operation-class axis (ARCH rev15
//! §3.1 N51); [`Consumer`] is the closed consumer set (ARCH rev15 §7, the access-
//! grants row: "The consumer set is closed (4 surfaces + the restore binary)").

/// An engine operation class. `Read` folds in query; `ProxiedRead` is a pure-read
/// op that crosses the daemon socket by design; `Schedule` is a compute- or
/// write-inducing background-work request (ARCH rev15 §3.1 N51, R2/alpha).
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

/// A consumer of the engine -- the axis N51's grant table is keyed by. The set is
/// closed: **four surfaces plus the restore binary** (ARCH rev15 §7, access-grants
/// row). [`Consumer::Restore`] is deliberately not a *surface*: the restore binary
/// links no serving surface and has no client seam (ARCH rev15 §3.4). It is here
/// because this enum's contract is "the closed set N51 is indexed by", and rev11
/// gave the restore binary its own grant row -- C/U/D under the exclusive storage
/// lock, everything else denied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Consumer {
    /// S1 -- the `memexd` daemon (the only *live* writer).
    Daemon,
    /// S2 -- the MCP server.
    Mcp,
    /// S3 -- the `wqm` CLI.
    Cli,
    /// S4 -- the terminal UI.
    Tui,
    /// The offline `wqm-restore` maintenance binary. Not a surface; runs only
    /// while the daemon is down, holding the same exclusive storage lock S1 holds
    /// when live (the one-live-writer relaxation, ARCH rev15 §8.3 / N14 row).
    Restore,
}

impl Consumer {
    /// Every consumer, in declaration order.
    pub const ALL: [Consumer; 5] = [
        Consumer::Daemon,
        Consumer::Mcp,
        Consumer::Cli,
        Consumer::Tui,
        Consumer::Restore,
    ];

    /// The canonical name of this consumer.
    pub const fn name(self) -> &'static str {
        match self {
            Consumer::Daemon => "daemon",
            Consumer::Mcp => "mcp",
            Consumer::Cli => "cli",
            Consumer::Tui => "tui",
            Consumer::Restore => "restore",
        }
    }

    /// Whether this consumer is a *serving surface*. False for
    /// [`Consumer::Restore`], which links no serving surface and exposes no
    /// socket -- the distinction the widened set must not blur.
    pub const fn is_surface(self) -> bool {
        !matches!(self, Consumer::Restore)
    }
}
