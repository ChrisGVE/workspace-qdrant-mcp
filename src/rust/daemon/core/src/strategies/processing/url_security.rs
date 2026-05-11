//! URL fetch security policy: SSRF denylist, host validation, scheme checks.
//!
//! This module is the policy boundary for URL ingestion. It rejects
//! requests targeting private, loopback, link-local, ULA, IPv4-mapped, and
//! known cloud-metadata addresses. The policy is applied at three points:
//!
//! 1. Before the initial request (resolved host).
//! 2. Per redirect hop (resolved redirect target host).
//! 3. After resolution, the *resolved IP* is passed to the HTTP client to
//!    defeat DNS rebinding (the same IP that passed policy is the IP
//!    actually contacted).
//!
//! Defaults are restrictive. The `UrlIngestionConfig` exposes
//! `allow_private_networks` for explicit opt-in.
//!
//! IPv4 CIDRs blocked by default:
//!   * 127.0.0.0/8        loopback
//!   * 169.254.0.0/16     link-local + AWS metadata (169.254.169.254)
//!   * 10.0.0.0/8         RFC1918
//!   * 172.16.0.0/12      RFC1918
//!   * 192.168.0.0/16     RFC1918
//!   * 0.0.0.0/8          "this network"
//!   * 100.64.0.0/10      shared address space
//!   * 192.0.0.0/24       IETF protocol assignments
//!   * 192.0.2.0/24       TEST-NET-1
//!   * 198.18.0.0/15      benchmarking
//!   * 198.51.100.0/24    TEST-NET-2
//!   * 203.0.113.0/24     TEST-NET-3
//!   * 224.0.0.0/4        multicast
//!   * 240.0.0.0/4        reserved
//!   * 255.255.255.255/32 broadcast
//!
//! IPv6 ranges blocked by default:
//!   * ::1/128            loopback
//!   * fe80::/10          link-local
//!   * fc00::/7           ULA
//!   * ::                 unspecified
//!   * ::ffff:0:0/96      IPv4-mapped (validated against IPv4 policy)
//!   * 64:ff9b::/96       IPv4/IPv6 translation
//!   * 64:ff9b:1::/48     local IPv4/IPv6 translation
//!   * 2001:db8::/32      documentation
//!   * 100::/64           discard
//!   * 2001:10::/28       ORCHID (deprecated)
//!   * ff00::/8           multicast

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use thiserror::Error;
use url::Host;
use url::Url;

/// Host strings that should always be rejected even when the resolved IP
/// would otherwise pass (cloud metadata endpoints commonly use a friendly
/// hostname that resolves to a private link-local address; the hostname
/// check is a belt-and-suspenders defence).
pub(crate) const METADATA_HOSTS: &[&str] = &[
    "metadata.google.internal",
    "metadata",
    "instance-data",
    "instance-data.ec2.internal",
];

/// Errors returned by the SSRF policy.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum UrlPolicyError {
    #[error("URL scheme must be http or https, got: {0}")]
    InvalidScheme(String),
    #[error("URL must have a host")]
    MissingHost,
    #[error("URL parse failed: {0}")]
    InvalidUrl(String),
    #[error("hostname matches blocked metadata service: {0}")]
    BlockedMetadataHost(String),
    #[error("IP {0} is in a blocked range ({1})")]
    BlockedAddress(IpAddr, &'static str),
    #[error("host resolved to zero addresses: {0}")]
    NoResolvedAddresses(String),
    #[error("DNS resolution failed for {0}: {1}")]
    DnsResolution(String, String),
    #[error("redirect blocked: {0}")]
    RedirectBlocked(String),
}

/// Classify an IPv4 address against the SSRF denylist, returning the name of
/// the matching range, or `None` if the address is public.
pub(crate) fn classify_ipv4(addr: Ipv4Addr) -> Option<&'static str> {
    let octets = addr.octets();
    let [a, b, _, _] = octets;
    if a == 0 {
        return Some("this-network 0.0.0.0/8");
    }
    if a == 127 {
        return Some("loopback 127.0.0.0/8");
    }
    if a == 10 {
        return Some("rfc1918 10.0.0.0/8");
    }
    if a == 172 && (16..=31).contains(&b) {
        return Some("rfc1918 172.16.0.0/12");
    }
    if a == 192 && b == 168 {
        return Some("rfc1918 192.168.0.0/16");
    }
    if a == 169 && b == 254 {
        return Some("link-local 169.254.0.0/16 (incl. cloud metadata)");
    }
    if a == 100 && (64..=127).contains(&b) {
        return Some("shared-address 100.64.0.0/10");
    }
    if a == 192 && b == 0 && octets[2] == 0 {
        return Some("ietf-protocol 192.0.0.0/24");
    }
    if a == 192 && b == 0 && octets[2] == 2 {
        return Some("test-net-1 192.0.2.0/24");
    }
    if a == 198 && (b == 18 || b == 19) {
        return Some("benchmark 198.18.0.0/15");
    }
    if a == 198 && b == 51 && octets[2] == 100 {
        return Some("test-net-2 198.51.100.0/24");
    }
    if a == 203 && b == 0 && octets[2] == 113 {
        return Some("test-net-3 203.0.113.0/24");
    }
    if (224..=239).contains(&a) {
        return Some("multicast 224.0.0.0/4");
    }
    if a >= 240 {
        return Some("reserved 240.0.0.0/4");
    }
    None
}

/// Classify an IPv6 address against the SSRF denylist.
pub(crate) fn classify_ipv6(addr: Ipv6Addr) -> Option<&'static str> {
    if addr.is_unspecified() {
        return Some("unspecified ::");
    }
    if addr.is_loopback() {
        return Some("loopback ::1");
    }
    // IPv4-mapped (::ffff:0:0/96) — defer to IPv4 policy
    if let Some(v4) = addr.to_ipv4_mapped() {
        return classify_ipv4(v4).or(Some("ipv4-mapped (passed v4 check)"));
    }
    let segments = addr.segments();
    // Link-local fe80::/10
    if segments[0] & 0xffc0 == 0xfe80 {
        return Some("link-local fe80::/10");
    }
    // Unique-local fc00::/7
    if segments[0] & 0xfe00 == 0xfc00 {
        return Some("unique-local fc00::/7");
    }
    // IPv4-translated 64:ff9b::/96
    if segments[0] == 0x0064 && segments[1] == 0xff9b && segments[2..6] == [0, 0, 0, 0] {
        return Some("ipv4-ipv6 translation 64:ff9b::/96");
    }
    // 64:ff9b:1::/48
    if segments[0] == 0x0064 && segments[1] == 0xff9b && segments[2] == 0x0001 {
        return Some("local ipv4-ipv6 translation 64:ff9b:1::/48");
    }
    // 2001:db8::/32 documentation
    if segments[0] == 0x2001 && segments[1] == 0x0db8 {
        return Some("documentation 2001:db8::/32");
    }
    // 100::/64 discard
    if segments[0] == 0x0100 && segments[1..4] == [0, 0, 0] {
        return Some("discard 100::/64");
    }
    // ORCHID 2001:10::/28
    if segments[0] == 0x2001 && (segments[1] & 0xfff0) == 0x0010 {
        return Some("orchid 2001:10::/28");
    }
    // Multicast ff00::/8
    if segments[0] & 0xff00 == 0xff00 {
        return Some("multicast ff00::/8");
    }
    None
}

/// Classify any IP address. Returns the violated range name, or `None` if
/// public.
pub fn classify_ip(addr: IpAddr) -> Option<&'static str> {
    match addr {
        IpAddr::V4(v4) => classify_ipv4(v4),
        IpAddr::V6(v6) => classify_ipv6(v6),
    }
}

/// Validate URL scheme. Only `http` and `https` are accepted.
pub fn validate_scheme(url: &Url) -> Result<(), UrlPolicyError> {
    match url.scheme() {
        "http" | "https" => Ok(()),
        other => Err(UrlPolicyError::InvalidScheme(other.to_string())),
    }
}

/// Parse and validate a URL string, returning a parsed `Url` on success.
pub fn parse_and_validate_scheme(raw: &str) -> Result<Url, UrlPolicyError> {
    let url = Url::parse(raw).map_err(|e| UrlPolicyError::InvalidUrl(e.to_string()))?;
    validate_scheme(&url)?;
    if url.host().is_none() {
        return Err(UrlPolicyError::MissingHost);
    }
    Ok(url)
}

/// Reject a host string that matches a known metadata service hostname.
pub fn reject_metadata_hostname(host: &str) -> Result<(), UrlPolicyError> {
    let lower = host.trim().to_ascii_lowercase();
    if METADATA_HOSTS.iter().any(|m| lower == *m) {
        return Err(UrlPolicyError::BlockedMetadataHost(lower));
    }
    Ok(())
}

/// Extract host string from a Url, rejecting metadata hostnames.
pub fn extract_host(url: &Url) -> Result<String, UrlPolicyError> {
    let host = url.host().ok_or(UrlPolicyError::MissingHost)?.to_string();
    reject_metadata_hostname(&host)?;
    Ok(host)
}

/// Apply per-IP SSRF policy. When `allow_private` is true, the address is
/// accepted unconditionally; otherwise the address is checked against the
/// denylist.
pub fn check_ip_policy(addr: IpAddr, allow_private: bool) -> Result<(), UrlPolicyError> {
    if allow_private {
        return Ok(());
    }
    match classify_ip(addr) {
        None => Ok(()),
        Some(range) => Err(UrlPolicyError::BlockedAddress(addr, range)),
    }
}

/// Apply policy to a literal-IP host (Url::host() returns Ipv4/Ipv6).
pub fn check_literal_ip_host(
    host: &Host<String>,
    allow_private: bool,
) -> Result<(), UrlPolicyError> {
    match host {
        Host::Ipv4(v4) => check_ip_policy(IpAddr::V4(*v4), allow_private),
        Host::Ipv6(v6) => check_ip_policy(IpAddr::V6(*v6), allow_private),
        Host::Domain(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};
    use std::str::FromStr;

    // ── IPv4 classification ────────────────────────────────────────────

    #[test]
    fn test_ipv4_loopback_blocked() {
        assert!(classify_ipv4(Ipv4Addr::new(127, 0, 0, 1)).is_some());
        assert!(classify_ipv4(Ipv4Addr::new(127, 255, 255, 254)).is_some());
    }

    #[test]
    fn test_ipv4_rfc1918_blocked() {
        assert!(classify_ipv4(Ipv4Addr::new(10, 0, 0, 1)).is_some());
        assert!(classify_ipv4(Ipv4Addr::new(10, 255, 255, 255)).is_some());
        assert!(classify_ipv4(Ipv4Addr::new(172, 16, 0, 1)).is_some());
        assert!(classify_ipv4(Ipv4Addr::new(172, 31, 255, 254)).is_some());
        assert!(classify_ipv4(Ipv4Addr::new(192, 168, 1, 1)).is_some());
    }

    #[test]
    fn test_ipv4_rfc1918_boundary_172_15_passes() {
        // 172.15.x.x is NOT in 172.16.0.0/12
        assert!(classify_ipv4(Ipv4Addr::new(172, 15, 0, 1)).is_none());
        // 172.32.x.x is NOT in 172.16.0.0/12
        assert!(classify_ipv4(Ipv4Addr::new(172, 32, 0, 1)).is_none());
    }

    #[test]
    fn test_ipv4_link_local_blocked() {
        assert!(classify_ipv4(Ipv4Addr::new(169, 254, 0, 1)).is_some());
        // AWS metadata
        assert!(classify_ipv4(Ipv4Addr::new(169, 254, 169, 254)).is_some());
    }

    #[test]
    fn test_ipv4_shared_address_blocked() {
        assert!(classify_ipv4(Ipv4Addr::new(100, 64, 0, 1)).is_some());
        assert!(classify_ipv4(Ipv4Addr::new(100, 127, 255, 254)).is_some());
        // 100.63 not in shared
        assert!(classify_ipv4(Ipv4Addr::new(100, 63, 0, 1)).is_none());
        // 100.128 not in shared
        assert!(classify_ipv4(Ipv4Addr::new(100, 128, 0, 1)).is_none());
    }

    #[test]
    fn test_ipv4_test_nets_blocked() {
        assert!(classify_ipv4(Ipv4Addr::new(192, 0, 2, 1)).is_some());
        assert!(classify_ipv4(Ipv4Addr::new(198, 51, 100, 1)).is_some());
        assert!(classify_ipv4(Ipv4Addr::new(203, 0, 113, 1)).is_some());
    }

    #[test]
    fn test_ipv4_benchmark_blocked() {
        assert!(classify_ipv4(Ipv4Addr::new(198, 18, 0, 1)).is_some());
        assert!(classify_ipv4(Ipv4Addr::new(198, 19, 255, 254)).is_some());
        assert!(classify_ipv4(Ipv4Addr::new(198, 20, 0, 1)).is_none());
    }

    #[test]
    fn test_ipv4_multicast_blocked() {
        assert!(classify_ipv4(Ipv4Addr::new(224, 0, 0, 1)).is_some());
        assert!(classify_ipv4(Ipv4Addr::new(239, 255, 255, 254)).is_some());
    }

    #[test]
    fn test_ipv4_reserved_blocked() {
        assert!(classify_ipv4(Ipv4Addr::new(240, 0, 0, 1)).is_some());
        assert!(classify_ipv4(Ipv4Addr::new(255, 255, 255, 255)).is_some());
    }

    #[test]
    fn test_ipv4_public_passes() {
        assert!(classify_ipv4(Ipv4Addr::new(8, 8, 8, 8)).is_none());
        assert!(classify_ipv4(Ipv4Addr::new(1, 1, 1, 1)).is_none());
        assert!(classify_ipv4(Ipv4Addr::new(140, 82, 121, 4)).is_none()); // github
    }

    #[test]
    fn test_ipv4_zero_network_blocked() {
        assert!(classify_ipv4(Ipv4Addr::new(0, 0, 0, 0)).is_some());
        assert!(classify_ipv4(Ipv4Addr::new(0, 1, 2, 3)).is_some());
    }

    // ── IPv6 classification ────────────────────────────────────────────

    #[test]
    fn test_ipv6_loopback_blocked() {
        assert!(classify_ipv6(Ipv6Addr::from_str("::1").unwrap()).is_some());
    }

    #[test]
    fn test_ipv6_unspecified_blocked() {
        assert!(classify_ipv6(Ipv6Addr::UNSPECIFIED).is_some());
    }

    #[test]
    fn test_ipv6_link_local_blocked() {
        assert!(classify_ipv6(Ipv6Addr::from_str("fe80::1").unwrap()).is_some());
        assert!(classify_ipv6(Ipv6Addr::from_str("febf::1").unwrap()).is_some());
        // fec0 is OUTSIDE fe80::/10 (deprecated site-local but not in our list)
        assert!(classify_ipv6(Ipv6Addr::from_str("fec0::1").unwrap()).is_none());
    }

    #[test]
    fn test_ipv6_ula_blocked() {
        assert!(classify_ipv6(Ipv6Addr::from_str("fc00::1").unwrap()).is_some());
        assert!(classify_ipv6(Ipv6Addr::from_str("fd00::1").unwrap()).is_some());
        assert!(classify_ipv6(Ipv6Addr::from_str("fdff::1").unwrap()).is_some());
    }

    #[test]
    fn test_ipv6_ipv4_mapped_loopback_blocked() {
        // ::ffff:127.0.0.1
        let mapped = Ipv6Addr::from_str("::ffff:127.0.0.1").unwrap();
        assert!(classify_ipv6(mapped).is_some());
    }

    #[test]
    fn test_ipv6_ipv4_mapped_private_blocked() {
        let mapped = Ipv6Addr::from_str("::ffff:10.0.0.1").unwrap();
        assert!(classify_ipv6(mapped).is_some());
        let mapped = Ipv6Addr::from_str("::ffff:169.254.169.254").unwrap();
        assert!(classify_ipv6(mapped).is_some());
    }

    #[test]
    fn test_ipv6_ipv4_mapped_public_blocked_anyway() {
        // We block IPv4-mapped IPv6 even if the inner v4 is public, because
        // accepting IPv4-mapped addresses is unusual and a smuggling vector.
        let mapped = Ipv6Addr::from_str("::ffff:8.8.8.8").unwrap();
        assert!(classify_ipv6(mapped).is_some());
    }

    #[test]
    fn test_ipv6_translation_blocked() {
        assert!(classify_ipv6(Ipv6Addr::from_str("64:ff9b::1").unwrap()).is_some());
        assert!(classify_ipv6(Ipv6Addr::from_str("64:ff9b:1::1").unwrap()).is_some());
    }

    #[test]
    fn test_ipv6_multicast_blocked() {
        assert!(classify_ipv6(Ipv6Addr::from_str("ff00::1").unwrap()).is_some());
        assert!(classify_ipv6(Ipv6Addr::from_str("ff02::1").unwrap()).is_some());
    }

    #[test]
    fn test_ipv6_documentation_blocked() {
        assert!(classify_ipv6(Ipv6Addr::from_str("2001:db8::1").unwrap()).is_some());
    }

    #[test]
    fn test_ipv6_public_passes() {
        assert!(classify_ipv6(Ipv6Addr::from_str("2606:4700:4700::1111").unwrap()).is_none());
        assert!(classify_ipv6(Ipv6Addr::from_str("2001:4860:4860::8888").unwrap()).is_none());
    }

    // ── Scheme + URL validation ───────────────────────────────────────

    #[test]
    fn test_parse_https_url_ok() {
        let u = parse_and_validate_scheme("https://example.com/page").unwrap();
        assert_eq!(u.scheme(), "https");
    }

    #[test]
    fn test_parse_http_url_ok() {
        let u = parse_and_validate_scheme("http://example.com/page").unwrap();
        assert_eq!(u.scheme(), "http");
    }

    #[test]
    fn test_parse_file_url_rejected() {
        let e = parse_and_validate_scheme("file:///etc/passwd").unwrap_err();
        assert!(matches!(e, UrlPolicyError::InvalidScheme(_)));
    }

    #[test]
    fn test_parse_javascript_url_rejected() {
        let e = parse_and_validate_scheme("javascript:alert(1)").unwrap_err();
        assert!(matches!(e, UrlPolicyError::InvalidScheme(_)));
    }

    #[test]
    fn test_parse_ftp_url_rejected() {
        let e = parse_and_validate_scheme("ftp://example.com/x").unwrap_err();
        assert!(matches!(e, UrlPolicyError::InvalidScheme(_)));
    }

    #[test]
    fn test_parse_data_url_rejected() {
        let e = parse_and_validate_scheme("data:text/plain,hello").unwrap_err();
        // data: URLs parse without a host; either MissingHost OR InvalidScheme is fine
        match e {
            UrlPolicyError::InvalidScheme(_) | UrlPolicyError::MissingHost => {}
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn test_parse_malformed_url_rejected() {
        let e = parse_and_validate_scheme("not a url").unwrap_err();
        assert!(matches!(e, UrlPolicyError::InvalidUrl(_)));
    }

    // ── Metadata hostname rejection ───────────────────────────────────

    #[test]
    fn test_reject_gcp_metadata_host() {
        let e = reject_metadata_hostname("metadata.google.internal").unwrap_err();
        assert!(matches!(e, UrlPolicyError::BlockedMetadataHost(_)));
    }

    #[test]
    fn test_reject_metadata_case_insensitive() {
        let e = reject_metadata_hostname("Metadata.Google.Internal").unwrap_err();
        assert!(matches!(e, UrlPolicyError::BlockedMetadataHost(_)));
    }

    #[test]
    fn test_reject_short_metadata_host() {
        assert!(reject_metadata_hostname("metadata").is_err());
        assert!(reject_metadata_hostname("instance-data").is_err());
    }

    #[test]
    fn test_normal_host_accepted_by_hostname_check() {
        assert!(reject_metadata_hostname("example.com").is_ok());
        assert!(reject_metadata_hostname("github.com").is_ok());
    }

    // ── Literal-IP host policy ────────────────────────────────────────

    #[test]
    fn test_literal_ipv4_localhost_blocked_in_url() {
        let url = Url::parse("http://127.0.0.1/").unwrap();
        let host = url.host().unwrap().to_owned();
        let e = check_literal_ip_host(&host, false).unwrap_err();
        assert!(matches!(e, UrlPolicyError::BlockedAddress(_, _)));
    }

    #[test]
    fn test_literal_ipv6_loopback_blocked_in_url() {
        let url = Url::parse("http://[::1]/").unwrap();
        let host = url.host().unwrap().to_owned();
        let e = check_literal_ip_host(&host, false).unwrap_err();
        assert!(matches!(e, UrlPolicyError::BlockedAddress(_, _)));
    }

    #[test]
    fn test_literal_ipv4_private_blocked() {
        let url = Url::parse("http://192.168.1.1/").unwrap();
        let host = url.host().unwrap().to_owned();
        let e = check_literal_ip_host(&host, false).unwrap_err();
        assert!(matches!(e, UrlPolicyError::BlockedAddress(_, _)));
    }

    #[test]
    fn test_literal_ipv4_metadata_blocked() {
        let url = Url::parse("http://169.254.169.254/latest/meta-data/").unwrap();
        let host = url.host().unwrap().to_owned();
        let e = check_literal_ip_host(&host, false).unwrap_err();
        assert!(matches!(e, UrlPolicyError::BlockedAddress(_, _)));
    }

    #[test]
    fn test_literal_ipv4_public_allowed() {
        let url = Url::parse("http://8.8.8.8/").unwrap();
        let host = url.host().unwrap().to_owned();
        assert!(check_literal_ip_host(&host, false).is_ok());
    }

    #[test]
    fn test_allow_private_override_passes_loopback() {
        let url = Url::parse("http://127.0.0.1/").unwrap();
        let host = url.host().unwrap().to_owned();
        assert!(check_literal_ip_host(&host, true).is_ok());
    }
}
