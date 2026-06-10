//! Git remote URL sanitization (#126).
//!
//! Remote URLs read from `.git/config` may carry credentials in the URL
//! userinfo component (e.g. `https://x-access-token:ghp_xxx@github.com/...`).
//! Those credentials must never reach persistent storage, logs, display
//! surfaces, or the tenant-id hash. `sanitize_git_remote_url` is called at
//! every boundary where a remote URL enters the system, so downstream
//! consumers only ever see the credential-free form.

/// Strip the userinfo component from a git remote URL.
///
/// - Scheme-ful URLs (`https://`, `http://`, `ssh://`, `git://`): the entire
///   `user[:password]@` part is removed — usernames in these forms are
///   frequently tokens themselves (`https://ghp_xxx@github.com/...`).
/// - scp-like URLs (`git@github.com:org/repo.git`): the username is kept
///   (it is the standard, non-secret SSH convention and removing it changes
///   the URL's meaning), but any `:password` inside the userinfo is dropped.
/// - URLs without userinfo are returned unchanged.
pub fn sanitize_git_remote_url(url: &str) -> String {
    if let Some(scheme_end) = url.find("://") {
        let after = &url[scheme_end + 3..];
        let authority_end = after.find('/').unwrap_or(after.len());
        if let Some(at) = after[..authority_end].rfind('@') {
            return format!("{}{}", &url[..scheme_end + 3], &after[at + 1..]);
        }
        return url.to_string();
    }

    // scp-like form: userinfo ends at the first '@' that precedes the path.
    let path_start = url.find('/').unwrap_or(url.len());
    if let Some(at) = url[..path_start].rfind('@') {
        let userinfo = &url[..at];
        if let Some(colon) = userinfo.find(':') {
            return format!("{}{}", &userinfo[..colon], &url[at..]);
        }
    }
    url.to_string()
}

#[cfg(test)]
mod tests {
    use super::sanitize_git_remote_url;

    #[test]
    fn strips_user_and_token_from_https() {
        assert_eq!(
            sanitize_git_remote_url("https://x-access-token:ghp_secret123@github.com/org/repo.git"),
            "https://github.com/org/repo.git"
        );
    }

    #[test]
    fn strips_token_only_userinfo() {
        assert_eq!(
            sanitize_git_remote_url("https://ghp_secret123@github.com/org/repo.git"),
            "https://github.com/org/repo.git"
        );
    }

    #[test]
    fn strips_userinfo_from_ssh_scheme() {
        assert_eq!(
            sanitize_git_remote_url("ssh://git@github.com/org/repo.git"),
            "ssh://github.com/org/repo.git"
        );
    }

    #[test]
    fn keeps_scp_like_username() {
        assert_eq!(
            sanitize_git_remote_url("git@github.com:org/repo.git"),
            "git@github.com:org/repo.git"
        );
    }

    #[test]
    fn drops_password_in_scp_like_userinfo() {
        assert_eq!(
            sanitize_git_remote_url("git:hunter2@github.com:org/repo.git"),
            "git@github.com:org/repo.git"
        );
    }

    #[test]
    fn clean_urls_unchanged() {
        assert_eq!(
            sanitize_git_remote_url("https://github.com/org/repo.git"),
            "https://github.com/org/repo.git"
        );
        assert_eq!(
            sanitize_git_remote_url("http://gitlab.example.com/org/repo"),
            "http://gitlab.example.com/org/repo"
        );
    }

    #[test]
    fn at_sign_in_path_not_treated_as_userinfo() {
        assert_eq!(
            sanitize_git_remote_url("https://github.com/org/repo@v2.git"),
            "https://github.com/org/repo@v2.git"
        );
    }

    #[test]
    fn preserves_port_after_userinfo_strip() {
        assert_eq!(
            sanitize_git_remote_url("https://user:pass@github.com:8443/org/repo.git"),
            "https://github.com:8443/org/repo.git"
        );
    }
}
