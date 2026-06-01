//! Tilde/path expansion, mirroring TS `expandPath` (config.ts:34-39).
//!
//! TS uses Node `path.join(home, path.slice(1))`, which **normalizes** the
//! joined path (collapses repeated `/`, resolves `.`/`..`, keeps at most one
//! trailing slash). To stay byte-for-byte identical, this module reimplements
//! `path.posix.join` / `path.posix.normalize` faithfully.

/// Expand a leading `~` in a path, mirroring the TypeScript `expandPath`:
///
/// ```typescript
/// function expandPath(path: string): string {
///   if (path.startsWith('~')) {
///     return join(homedir(), path.slice(1));
///   }
///   return path;
/// }
/// ```
///
/// Delegates to [`expand_path_ts_with_home`] with the real home directory.
/// `$VAR` / `${VAR}` are NOT expanded (TS leaves them verbatim).
pub fn expand_path_ts(path: &str) -> String {
    let home = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("~"))
        .to_string_lossy()
        .into_owned();
    expand_path_ts_with_home(path, &home)
}

/// Home-injectable core of [`expand_path_ts`].
///
/// `pub` so the TS↔Rust parity corpus suite (`tests/parity_corpus.rs`) can
/// drive it with a fixed home that matches the capture harness, asserting
/// byte-for-byte parity with Node `path.join(home, path.slice(1))`.
///
/// Returns `path` unchanged unless it starts with `~`; otherwise returns the
/// POSIX-normalized join of `home` with `path[1..]`.
pub fn expand_path_ts_with_home(path: &str, home: &str) -> String {
    match path.strip_prefix('~') {
        Some(slice) => node_posix_join(home, slice),
        None => path.to_owned(),
    }
}

/// Faithful re-implementation of Node `path.posix.join(a, b)`.
///
/// Node joins the non-empty arguments with `/`, then runs `path.posix.normalize`
/// on the result. An all-empty join returns `"."`.
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
///
/// - Preserves a leading `/` (absolute).
/// - Collapses repeated separators, drops `.`, resolves `..` (popping the
///   previous real segment; leading `..` are dropped on absolute paths and
///   kept on relative ones).
/// - Preserves a single trailing `/` when the input ended with a separator and
///   the result is more than just the root.
/// - An empty relative result becomes `.`.
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
                    // Keep ".." only on relative paths; drop at absolute root.
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

    // node_posix_join / node_posix_normalize — faithful to Node path.join,
    // which normalizes the joined path (collapse //, resolve ./.., keep one
    // trailing slash). Values captured from Node `path.posix.normalize/join`.

    #[test]
    fn node_posix_normalize_collapses_and_resolves() {
        assert_eq!(node_posix_normalize("/a//b"), "/a/b");
        assert_eq!(node_posix_normalize("/a/./b"), "/a/b");
        assert_eq!(node_posix_normalize("/a/../b"), "/b");
        assert_eq!(node_posix_normalize("/a/b/.."), "/a");
        assert_eq!(node_posix_normalize("/a/b/../.."), "/");
        // ".." past absolute root is dropped (clamped to root).
        assert_eq!(node_posix_normalize("/../../x"), "/x");
        // Relative ".." is preserved.
        assert_eq!(node_posix_normalize("../x"), "../x");
        assert_eq!(node_posix_normalize("a/../../x"), "../x");
        // Trailing slash preserved when result is more than root.
        assert_eq!(node_posix_normalize("/a/b/"), "/a/b/");
        assert_eq!(node_posix_normalize("/a//"), "/a/");
        // Empty / dot-only relative inputs become ".".
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
        // Non-tilde paths are returned verbatim.
        assert_eq!(expand_path_ts_with_home("/abs/p", home), "/abs/p");
        assert_eq!(expand_path_ts_with_home("rel/p", home), "rel/p");
        assert_eq!(expand_path_ts_with_home("$HOME/x", home), "$HOME/x");
    }
}
