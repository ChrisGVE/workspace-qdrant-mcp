//! XDG-compliant path resolution and canonical-path abstraction for
//! workspace-qdrant.
//!
//! This module has two distinct concerns:
//!
//! 1. **Process-local directory layout** — XDG-compliant resolution of config,
//!    data, cache, and log directories. The free functions [`get_config_dir`],
//!    [`get_data_dir`], etc. are the single source of truth for filesystem
//!    layout. Every component (daemon, CLI, MCP server) MUST use these
//!    functions instead of hardcoding directory names.
//!
//! 2. **Canonical path abstraction** — newtypes [`CanonicalPath`],
//!    [`LocalPath`], and [`MountMap`] enforce a type-system discipline that
//!    keeps stored/transmitted paths in a single deployment-independent form
//!    and translates to process-local paths only at the filesystem I/O
//!    boundary. See `docs/specs/16-path-abstraction.md` for the full design.
//!
//! XDG layout (macOS, no env overrides):
//!   Config: ~/.config/workspace-qdrant/           (config.yaml, cli-config.toml)
//!   Data:   ~/.local/share/workspace-qdrant/      (state.db, search.db, graph.db)
//!   Cache:  ~/.cache/workspace-qdrant/            (grammars/, models/)
//!   Logs:   ~/Library/Logs/workspace-qdrant/
//!
//! Environment overrides (highest priority):
//!   WQM_CONFIG_PATH  — explicit config file path
//!   WQM_CONFIG_DIR   — config directory
//!   WQM_DATA_DIR     — data directory
//!   WQM_CACHE_DIR    — cache directory
//!   WQM_DATABASE_PATH — explicit database file path
//!   WQM_LOG_DIR      — log directory
//!
//! XDG variables ($XDG_CONFIG_HOME, $XDG_DATA_HOME, $XDG_CACHE_HOME) are
//! respected on all platforms.

mod boundary;
mod canonical;
mod discovery;
mod error;
mod local;
mod mount_map;
mod normalize;
mod relative;
mod resolve;

#[cfg(test)]
mod tests;

pub use boundary::is_within_boundary;
pub use canonical::CanonicalPath;
pub use discovery::{find_config_file, get_config_search_paths};
pub use error::PathError;
pub use local::LocalPath;
pub use mount_map::{mount_section_hash, MountEntry, MountMap};
pub use normalize::canonicalize_host_path;
pub use relative::{RelativePath, RelativePathError};
pub use resolve::{
    get_cache_dir, get_canonical_log_dir, get_config_dir, get_data_dir, get_database_path,
    get_database_path_checked, get_grammar_cache_dir, get_model_cache_dir, ConfigPathError,
};
