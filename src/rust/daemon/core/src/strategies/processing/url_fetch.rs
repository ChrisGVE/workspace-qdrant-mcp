//! Secure URL fetcher with SSRF protection, redirect re-validation,
//! content-type allowlist, and streamed body cap.
//!
//! This module is the only place in the daemon that should make outbound
//! HTTP requests for user-provided URLs. The policy is enforced at every
//! redirect hop using `redirect::Policy::custom`.
//!
//! DNS rebinding is mitigated by resolving the hostname once at the policy
//! boundary, validating the resolved IP, then pinning the resolved IP via
//! `reqwest::ClientBuilder::resolve_to_addrs` so the actual TCP connection
//! uses the validated address, not whatever the DNS resolver returns later.

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use futures::StreamExt;
use reqwest::redirect::Policy;
use reqwest::Client;
use tokio::net::lookup_host;
use tracing::{debug, warn};

use crate::config::UrlIngestionConfig;
use crate::strategies::processing::url_security::{
    check_ip_policy, check_literal_ip_host, extract_host, parse_and_validate_scheme, UrlPolicyError,
};

/// Output of a secure URL fetch.
#[derive(Debug)]
pub struct FetchedDocument {
    /// Final URL after redirect chain (may equal the original URL).
    pub final_url: String,
    /// Response Content-Type header (lower-cased, no parameters stripped).
    pub content_type: String,
    /// Body as UTF-8 string. May be truncated to `max_body_bytes`.
    pub body: String,
    /// True iff the body was truncated due to size cap.
    pub truncated: bool,
}

/// Top-level fetch error covering both policy and network failures.
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("policy violation: {0}")]
    Policy(#[from] UrlPolicyError),
    #[error("HTTP {0} for {1}")]
    HttpStatus(u16, String),
    #[error("content-type rejected: {0}")]
    ContentTypeRejected(String),
    #[error("network: {0}")]
    Network(String),
    #[error("body decode failed: {0}")]
    Decode(String),
}

/// Resolve a hostname to an IP address with SSRF validation.
///
/// Returns the first allowed IP. Returns `BlockedAddress` if every resolved
/// address is in the denylist. Returns `NoResolvedAddresses` if no addresses
/// could be resolved. If the host is a literal IP, returns it without DNS.
pub async fn resolve_with_policy(
    host: &str,
    port: u16,
    allow_private: bool,
) -> Result<IpAddr, UrlPolicyError> {
    // Literal IP shortcut
    if let Ok(addr) = host.parse::<IpAddr>() {
        check_ip_policy(addr, allow_private)?;
        return Ok(addr);
    }
    let host_port = format!("{host}:{port}");
    let iter = lookup_host(host_port.as_str())
        .await
        .map_err(|e| UrlPolicyError::DnsResolution(host.to_string(), e.to_string()))?;
    let mut last_block: Option<UrlPolicyError> = None;
    let mut seen_any = false;
    for sock in iter {
        seen_any = true;
        let ip = sock.ip();
        match check_ip_policy(ip, allow_private) {
            Ok(()) => return Ok(ip),
            Err(e) => last_block = Some(e),
        }
    }
    if !seen_any {
        return Err(UrlPolicyError::NoResolvedAddresses(host.to_string()));
    }
    Err(last_block.expect("at least one address yielded an error"))
}

/// Fetch a URL with full SSRF protection, redirect re-validation,
/// content-type allowlist, and body-size cap.
pub async fn fetch_url_secured(
    raw_url: &str,
    cfg: &UrlIngestionConfig,
) -> Result<FetchedDocument, FetchError> {
    // 1. Parse + scheme check
    let url = parse_and_validate_scheme(raw_url)?;
    // 2. Reject metadata hostnames
    let host = extract_host(&url)?;
    // 3. Literal-IP host check (before DNS for literals)
    if let Some(literal) = url.host() {
        // url::Host has Domain/Ipv4/Ipv6
        let owned = literal.to_owned();
        check_literal_ip_host(&owned, cfg.allow_private_networks)?;
    }
    // 4. Resolve host → IP, validate against denylist
    let port = url
        .port_or_known_default()
        .ok_or_else(|| UrlPolicyError::InvalidUrl("URL has no port".to_string()))?;
    let resolved_ip = resolve_with_policy(&host, port, cfg.allow_private_networks).await?;
    let socket = SocketAddr::new(resolved_ip, port);

    // 5. Build reqwest client with timeouts + per-hop SSRF redirect policy
    let max_redirects = cfg.max_redirects;
    let allow_private = cfg.allow_private_networks;
    let redirect_policy = Policy::custom(move |attempt| {
        // Compute the decision without moving `attempt`, then perform the
        // final action call (which moves `attempt`) at the end.
        let previous_len = attempt.previous().len();
        let err_msg: Option<String> = if previous_len >= max_redirects {
            Some(format!(
                "exceeded max redirects ({previous_len} >= {max_redirects})"
            ))
        } else {
            let next = attempt.url();
            let scheme = next.scheme().to_string();
            if !matches!(scheme.as_str(), "http" | "https") {
                Some(format!("redirect to non-http scheme: {scheme}"))
            } else if let Some(h) = next.host_str().map(|s| s.to_ascii_lowercase()) {
                if super::url_security::METADATA_HOSTS.iter().any(|m| h == *m) {
                    Some(format!("redirect to blocked metadata host: {h}"))
                } else if let Some(host) = next.host() {
                    let owned = host.to_owned();
                    check_literal_ip_host(&owned, allow_private)
                        .err()
                        .map(|e| e.to_string())
                } else {
                    None
                }
            } else {
                None
            }
        };
        // Note: redirect-target hostnames re-resolve via the system resolver
        // (resolve_to_addrs only applies to the originally requested host).
        // Literal-IP + metadata-hostname checks above mitigate the common
        // SSRF redirect attack; set `max_redirects: 0` for full lock-down.
        match err_msg {
            Some(msg) => attempt.error(msg),
            None => attempt.follow(),
        }
    });

    let client = Client::builder()
        .connect_timeout(Duration::from_secs(cfg.connect_timeout_secs))
        .timeout(Duration::from_secs(cfg.read_timeout_secs))
        .redirect(redirect_policy)
        .resolve_to_addrs(&host, &[socket])
        .build()
        .map_err(|e| FetchError::Network(format!("client build: {e}")))?;

    debug!(
        "secure URL fetch: url={} resolved={} max_redirects={} cap={}B",
        raw_url, socket, cfg.max_redirects, cfg.max_body_bytes
    );

    // 6. Send request
    let response = client
        .get(url.as_str())
        .header(reqwest::header::USER_AGENT, "workspace-qdrant/0.1")
        .send()
        .await
        .map_err(|e| FetchError::Network(e.to_string()))?;

    let status = response.status();
    let final_url = response.url().to_string();
    if !status.is_success() {
        return Err(FetchError::HttpStatus(status.as_u16(), final_url));
    }

    // 7. Validate content-type
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("text/html")
        .to_string();
    if !cfg.is_content_type_allowed(&content_type) {
        return Err(FetchError::ContentTypeRejected(content_type));
    }

    // 8. Stream body with size cap
    let mut buf: Vec<u8> = Vec::new();
    let mut truncated = false;
    let cap = cfg.max_body_bytes as usize;
    let mut stream = response.bytes_stream();
    while let Some(next) = stream.next().await {
        let chunk = next.map_err(|e| FetchError::Network(format!("stream: {e}")))?;
        let remaining = cap.saturating_sub(buf.len());
        if remaining == 0 {
            truncated = true;
            break;
        }
        if chunk.len() > remaining {
            buf.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            break;
        } else {
            buf.extend_from_slice(&chunk);
        }
    }
    if truncated {
        warn!(
            "URL body truncated at {} bytes (cap={})",
            buf.len(),
            cfg.max_body_bytes
        );
    }

    // 9. Decode to UTF-8 lossily so we never fail on byte-misaligned truncation
    let body = String::from_utf8_lossy(&buf).into_owned();

    Ok(FetchedDocument {
        final_url,
        content_type,
        body,
        truncated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn dev_cfg(server_addr: &str) -> UrlIngestionConfig {
        // Allow private networks for wiremock (127.0.0.1)
        let mut c = UrlIngestionConfig::default();
        c.allow_private_networks = true;
        // Override defaults are fine; just ensure private allowed for tests
        let _ = server_addr;
        c
    }

    #[tokio::test]
    async fn test_fetch_simple_html_succeeds() {
        let server = MockServer::start().await;
        // `set_body_raw` sets both body and content-type atomically; `set_body_string`
        // forces content-type to text/plain regardless of `insert_header` ordering.
        Mock::given(method("GET"))
            .and(path("/page"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                b"<html><body><h1>Hello</h1></body></html>".to_vec(),
                "text/html; charset=utf-8",
            ))
            .mount(&server)
            .await;
        let url = format!("{}/page", server.uri());
        let cfg = dev_cfg(&url);
        let out = fetch_url_secured(&url, &cfg).await.unwrap();
        assert!(out.body.contains("Hello"));
        assert!(out.content_type.starts_with("text/html"));
        assert!(!out.truncated);
    }

    #[tokio::test]
    async fn test_fetch_loopback_rejected_by_default() {
        // Use a routable-looking but loopback host; the policy must reject it
        // even though wiremock is on 127.0.0.1.
        let server = MockServer::start().await;
        let url = format!("{}/page", server.uri());
        let cfg = UrlIngestionConfig::default(); // private NOT allowed
        let err = fetch_url_secured(&url, &cfg).await.unwrap_err();
        match err {
            FetchError::Policy(UrlPolicyError::BlockedAddress(_, _)) => {}
            other => panic!("expected BlockedAddress, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_fetch_oversize_body_truncated() {
        let server = MockServer::start().await;
        let payload = "A".repeat(2 * 1024); // 2 KiB
        Mock::given(method("GET"))
            .and(path("/big"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/plain")
                    .set_body_string(payload),
            )
            .mount(&server)
            .await;
        let url = format!("{}/big", server.uri());
        let mut cfg = dev_cfg(&url);
        cfg.max_body_bytes = 512;
        let out = fetch_url_secured(&url, &cfg).await.unwrap();
        assert_eq!(out.body.len(), 512);
        assert!(out.truncated);
    }

    #[tokio::test]
    async fn test_fetch_disallowed_content_type_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/bin"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/octet-stream")
                    .set_body_bytes(vec![0u8; 16]),
            )
            .mount(&server)
            .await;
        let url = format!("{}/bin", server.uri());
        let cfg = dev_cfg(&url);
        let err = fetch_url_secured(&url, &cfg).await.unwrap_err();
        assert!(matches!(err, FetchError::ContentTypeRejected(_)));
    }

    #[tokio::test]
    async fn test_fetch_image_content_type_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/img"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "image/png")
                    .set_body_bytes(vec![0u8; 16]),
            )
            .mount(&server)
            .await;
        let url = format!("{}/img", server.uri());
        let cfg = dev_cfg(&url);
        let err = fetch_url_secured(&url, &cfg).await.unwrap_err();
        assert!(matches!(err, FetchError::ContentTypeRejected(_)));
    }

    #[tokio::test]
    async fn test_fetch_404_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/missing"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        let url = format!("{}/missing", server.uri());
        let cfg = dev_cfg(&url);
        let err = fetch_url_secured(&url, &cfg).await.unwrap_err();
        assert!(matches!(err, FetchError::HttpStatus(404, _)));
    }

    #[tokio::test]
    async fn test_fetch_5xx_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/oops"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let url = format!("{}/oops", server.uri());
        let cfg = dev_cfg(&url);
        let err = fetch_url_secured(&url, &cfg).await.unwrap_err();
        assert!(matches!(err, FetchError::HttpStatus(500, _)));
    }

    #[tokio::test]
    async fn test_fetch_file_scheme_rejected() {
        let cfg = UrlIngestionConfig::default();
        let err = fetch_url_secured("file:///etc/passwd", &cfg)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            FetchError::Policy(UrlPolicyError::InvalidScheme(_))
        ));
    }

    #[tokio::test]
    async fn test_fetch_metadata_host_rejected_pre_dns() {
        let cfg = UrlIngestionConfig::default();
        let err = fetch_url_secured("http://metadata.google.internal/computeMetadata/v1/", &cfg)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            FetchError::Policy(UrlPolicyError::BlockedMetadataHost(_))
        ));
    }

    #[tokio::test]
    async fn test_fetch_literal_aws_metadata_rejected() {
        let cfg = UrlIngestionConfig::default();
        let err = fetch_url_secured("http://169.254.169.254/latest/meta-data/", &cfg)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            FetchError::Policy(UrlPolicyError::BlockedAddress(_, _))
        ));
    }

    // NOTE: A direct test for "redirect to RFC1918 rejected" would require
    // `allow_private_networks=false` while still allowing wiremock on loopback —
    // a mode the current single-flag policy does not express. Redirect-policy
    // rejection is exercised by `test_fetch_redirect_to_metadata_rejected` (which
    // is independent of the `allow_private` flag) and `test_fetch_redirect_chain_capped`.
    // The redirect-hop SSRF check itself is unit-covered in `url_security` tests.

    #[tokio::test]
    async fn test_fetch_redirect_to_metadata_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/r2"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("location", "http://metadata.google.internal/"),
            )
            .mount(&server)
            .await;
        let url = format!("{}/r2", server.uri());
        let cfg = dev_cfg(&url);
        let err = fetch_url_secured(&url, &cfg).await.unwrap_err();
        assert!(matches!(err, FetchError::Network(_)));
    }

    #[tokio::test]
    async fn test_fetch_redirect_chain_capped() {
        let server = MockServer::start().await;
        // /a -> /b -> /c -> /d -> /e -> /f (5 hops); cap is 3.
        for (from, to) in [("/a", "/b"), ("/b", "/c"), ("/c", "/d"), ("/d", "/e")] {
            Mock::given(method("GET"))
                .and(path(from))
                .respond_with(
                    ResponseTemplate::new(302)
                        .insert_header("location", format!("{}{}", server.uri(), to)),
                )
                .mount(&server)
                .await;
        }
        let url = format!("{}/a", server.uri());
        let mut cfg = dev_cfg(&url);
        cfg.max_redirects = 2;
        let err = fetch_url_secured(&url, &cfg).await.unwrap_err();
        assert!(matches!(err, FetchError::Network(_)));
    }

    #[tokio::test]
    async fn test_fetch_json_content_type_allowed() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(r#"{"key":"value"}"#),
            )
            .mount(&server)
            .await;
        let url = format!("{}/api", server.uri());
        let cfg = dev_cfg(&url);
        let out = fetch_url_secured(&url, &cfg).await.unwrap();
        assert!(out.body.contains("value"));
    }
}
