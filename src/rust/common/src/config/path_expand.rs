//! TS-faithful tilde-only path expansion.
//!
//! Mirrors the TypeScript MCP server's `expandPath` (config.ts:34-39):
//!
//! ```typescript
//! function expandPath(path: string): string {
//!   if (path.startsWith('~')) {
//!     return join(homedir(), path.slice(1));
//!   }
//!   return path;
//! }
//! ```
//!
//! Only a leading `~` is expanded; `$VAR` / `${VAR}` are left verbatim (unlike
//! the full env-expand mode in [`crate::env_expand::expand_path`], used by the
//! daemon and CLI). Node `path.join` normalizes the result, so this reimplements
//! `path.posix.join` / `path.posix.normalize` for byte-for-byte parity.

/// Expand a leading `~` using the real home directory; non-tilde paths are
/// returned verbatim. `$VAR` is NOT expanded (TS-faithful).
pub fn expand_path_ts(path: &str) -> String {
    let home = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("~"))
        .to_string_lossy()
        .into_owned();
    expand_path_ts_with_home(path, &home)
}

/// Home-injectable core of [`expand_path_ts`] (lets parity suites pin `home`).
///
/// Returns `path` unchanged unless it starts with `~`; otherwise the
/// POSIX-normalized join of `home` with `path[1..]`.
pub fn expand_path_ts_with_home(path: &str, home: &str) -> String {
    match path.strip_prefix('~') {
        Some(slice) => node_posix_join(home, slice),
        None => path.to_owned(),
    }
}

/// Faithful re-implementation of Node `path.posix.join(a, b)`.
fn node_posix_join(a: &str, b: &str) -> String {
    let joined = match (a.is_empty(), b.is_empty()) {
        (true, true) => return ".".to_string(),
        (false, true) => a.to_string(),
        (true, false) => b.to_string(),
        (false, false) => format!("{a}/{b}"),
    };
    node_posix_normalize(&joined)
}

/// Faithful re-implementation of Node `path.posix.normalize(p)`.
fn node_posix_normalize(p: &str) -> String {
    if p.is_empty() {
        return ".".to_string();
    }
    let is_absolute = p.starts_with('/');
    let trailing_sep = p.ends_with('/');

    let mut out: Vec<&str> = Vec::new();
    for seg in p.split('/') {
        match seg {
            "" | "." => {}
            ".." => match out.last() {
                Some(&last) if last != ".." => {
                    out.pop();
                }
                _ => {
                    if !is_absolute {
                        out.push("..");
                    }
                }
            },
            other => out.push(other),
        }
    }

    let mut body = out.join("/");
    if body.is_empty() {
        return if is_absolute {
            "/".to_string()
        } else {
            ".".to_string()
        };
    }
    if trailing_sep {
        body.push('/');
    }
    if is_absolute {
        format!("/{body}")
    } else {
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_posix_normalize_collapses_and_resolves() {
        assert_eq!(node_posix_normalize("/a//b"), "/a/b");
        assert_eq!(node_posix_normalize("/a/./b"), "/a/b");
        assert_eq!(node_posix_normalize("/a/../b"), "/b");
        assert_eq!(node_posix_normalize("/a/b/.."), "/a");
        assert_eq!(node_posix_normalize("/a/b/../.."), "/");
        assert_eq!(node_posix_normalize("/../../x"), "/x");
        assert_eq!(node_posix_normalize("../x"), "../x");
        assert_eq!(node_posix_normalize("a/../../x"), "../x");
        assert_eq!(node_posix_normalize("/a/b/"), "/a/b/");
        assert_eq!(node_posix_normalize("/a//"), "/a/");
        assert_eq!(node_posix_normalize(""), ".");
        assert_eq!(node_posix_normalize("a/.."), ".");
    }

    #[test]
    fn node_posix_join_matches_node() {
        assert_eq!(node_posix_join("/home/u", ""), "/home/u");
        assert_eq!(node_posix_join("/home/u", "/x"), "/home/u/x");
        assert_eq!(node_posix_join("/home/u", "a/../b"), "/home/u/b");
        assert_eq!(node_posix_join("/home/u", "x/"), "/home/u/x/");
        assert_eq!(node_posix_join("", ""), ".");
    }

    #[test]
    fn expand_path_ts_with_home_normalizes_like_node() {
        let home = "/home/testuser";
        assert_eq!(expand_path_ts_with_home("~", home), "/home/testuser");
        assert_eq!(expand_path_ts_with_home("~/", home), "/home/testuser/");
        assert_eq!(
            expand_path_ts_with_home("~/a/../b", home),
            "/home/testuser/b"
        );
        assert_eq!(expand_path_ts_with_home("~//x", home), "/home/testuser/x");
        assert_eq!(
            expand_path_ts_with_home("~/trailing/..", home),
            "/home/testuser"
        );
        // Non-tilde paths returned verbatim; $VAR NOT expanded; literal ~ mid-string
        // is left alone (only a LEADING ~ triggers expansion).
        assert_eq!(expand_path_ts_with_home("/abs/p", home), "/abs/p");
        assert_eq!(expand_path_ts_with_home("rel/p", home), "rel/p");
        assert_eq!(expand_path_ts_with_home("$HOME/x", home), "$HOME/x");
        assert_eq!(expand_path_ts_with_home("/data/~/x", home), "/data/~/x");
    }
}
