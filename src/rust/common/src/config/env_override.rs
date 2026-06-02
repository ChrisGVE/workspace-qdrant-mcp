//! Declarative environment-variable override engine + gRPC-endpoint parsing.
//!
//! Components describe their overrides as a list of [`EnvOverride`] specs (each
//! a precedence-ordered set of env-var names plus a setter closure); the shared
//! [`apply_env_overrides`] loop reads the environment and applies them. Multi-
//! variable precedence (e.g. endpoint-then-port-fallback) is expressed by
//! ordering the spec list and the `vars` within each spec.
//!
//! The gRPC-endpoint helpers ([`parse_grpc_endpoint`], [`parse_int_prefix`])
//! are shared because both clients resolve `host:port` from env strings with
//! identical (TypeScript-faithful) semantics.

use super::discovery::EnvGetter;
use crate::constants::DEFAULT_GRPC_PORT;

/// One declarative env-override: the first present var in `vars` (in order)
/// supplies the value passed to `apply`.
pub struct EnvOverride<C> {
    /// Candidate env-var names in precedence order (first present wins).
    pub vars: Vec<String>,
    /// Mutates the config with the resolved value.
    pub apply: Box<dyn Fn(&mut C, String)>,
}

impl<C> EnvOverride<C> {
    /// Convenience constructor for a single-variable override.
    pub fn single(var: impl Into<String>, apply: impl Fn(&mut C, String) + 'static) -> Self {
        Self {
            vars: vec![var.into()],
            apply: Box::new(apply),
        }
    }

    /// Convenience constructor for a multi-variable (precedence-ordered) override.
    pub fn any(
        vars: impl IntoIterator<Item = impl Into<String>>,
        apply: impl Fn(&mut C, String) + 'static,
    ) -> Self {
        Self {
            vars: vars.into_iter().map(Into::into).collect(),
            apply: Box::new(apply),
        }
    }
}

/// Apply each spec in order: for a spec, the first present env var triggers its
/// setter; later specs still run, so order the list when one override must take
/// effect only if an earlier one was absent.
pub fn apply_env_overrides<C>(config: &mut C, env: &EnvGetter, specs: &[EnvOverride<C>]) {
    for spec in specs {
        for var in &spec.vars {
            if let Some(value) = env(var) {
                (spec.apply)(config, value);
                break;
            }
        }
    }
}

/// Parsed `host:port` gRPC endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrpcEndpoint {
    /// Host portion (scheme stripped).
    pub host: String,
    /// Port portion ([`DEFAULT_GRPC_PORT`] when absent/invalid).
    pub port: u16,
}

/// Parse a gRPC endpoint string into host + port.
///
/// Accepts `http(s)://host:port`, `host:port`, and `host` (port defaults).
/// Invalid or out-of-range ports fall back to [`DEFAULT_GRPC_PORT`]; an empty
/// string yields `host = ""` with the default port. Mirrors the TypeScript
/// `parseGrpcEndpoint`.
pub fn parse_grpc_endpoint(endpoint: &str) -> GrpcEndpoint {
    let without_scheme = strip_scheme(endpoint);
    match without_scheme.rfind(':') {
        None => GrpcEndpoint {
            host: without_scheme.to_owned(),
            port: DEFAULT_GRPC_PORT,
        },
        Some(idx) => GrpcEndpoint {
            host: without_scheme[..idx].to_owned(),
            port: parse_port(&without_scheme[idx + 1..]),
        },
    }
}

fn strip_scheme(s: &str) -> &str {
    s.strip_prefix("https://")
        .or_else(|| s.strip_prefix("http://"))
        .unwrap_or(s)
}

fn parse_port(s: &str) -> u16 {
    match s.parse::<i64>() {
        Ok(n) if n > 0 && n <= 65535 => n as u16,
        _ => DEFAULT_GRPC_PORT,
    }
}

/// Parse the leading integer in a string, mirroring JS `parseInt(s, 10)`:
/// skip leading ASCII whitespace, take an optional `+`/`-` sign, consume the
/// leading decimal-digit run, ignore the rest. Returns `None` when no digit run
/// is found (JS `NaN`).
pub fn parse_int_prefix(s: &str) -> Option<i64> {
    let s = s.trim_start_matches(|c: char| c.is_ascii_whitespace());
    if s.is_empty() {
        return None;
    }
    let (s, negative) = if let Some(rest) = s.strip_prefix('-') {
        (rest, true)
    } else if let Some(rest) = s.strip_prefix('+') {
        (rest, false)
    } else {
        (s, false)
    };
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let n: i64 = digits.parse().ok()?;
    Some(if negative { -n } else { n })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env_from<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        let map: HashMap<&str, &str> = pairs.iter().copied().collect();
        move |key: &str| map.get(key).map(|v| v.to_string())
    }

    #[derive(Default, Debug, PartialEq)]
    struct Cfg {
        url: String,
        host: String,
        port: u16,
    }

    #[test]
    fn single_var_override_applies() {
        let specs = vec![EnvOverride::single("URL", |c: &mut Cfg, v| c.url = v)];
        let mut cfg = Cfg::default();
        apply_env_overrides(&mut cfg, &env_from(&[("URL", "http://x")]), &specs);
        assert_eq!(cfg.url, "http://x");
    }

    #[test]
    fn absent_var_leaves_default() {
        let specs = vec![EnvOverride::single("URL", |c: &mut Cfg, v| c.url = v)];
        let mut cfg = Cfg {
            url: "default".into(),
            ..Cfg::default()
        };
        apply_env_overrides(&mut cfg, &env_from(&[]), &specs);
        assert_eq!(cfg.url, "default");
    }

    #[test]
    fn multi_var_first_present_wins() {
        let specs = vec![EnvOverride::any(["PRIMARY", "ALIAS"], |c: &mut Cfg, v| {
            let ep = parse_grpc_endpoint(&v);
            c.host = ep.host;
            c.port = ep.port;
        })];
        let mut cfg = Cfg::default();
        // ALIAS present, PRIMARY absent → ALIAS used.
        apply_env_overrides(&mut cfg, &env_from(&[("ALIAS", "alias-host:2222")]), &specs);
        assert_eq!(cfg.host, "alias-host");
        assert_eq!(cfg.port, 2222);
    }

    #[test]
    fn precedence_env_over_file_over_default() {
        // Spec order encodes precedence: endpoint spec first, port-fallback spec
        // second only fires when its var is present. With endpoint present, the
        // fallback's blind set would still run — so the fallback guards on the
        // higher-priority var being absent, demonstrating the intended pattern.
        let specs = vec![
            EnvOverride::any(["ENDPOINT"], |c: &mut Cfg, v| {
                let ep = parse_grpc_endpoint(&v);
                c.host = ep.host;
                c.port = ep.port;
            }),
            EnvOverride::single("PORT_ONLY", |c: &mut Cfg, v| {
                if let Some(n) = parse_int_prefix(&v) {
                    if (0..=65535).contains(&n) {
                        c.port = n as u16;
                    }
                }
            }),
        ];
        // File/default baseline.
        let mut cfg = Cfg {
            host: "file-host".into(),
            port: 1000,
            ..Cfg::default()
        };
        // Only PORT_ONLY set (no ENDPOINT) → fallback applies over default.
        apply_env_overrides(&mut cfg, &env_from(&[("PORT_ONLY", "3333")]), &specs);
        assert_eq!(cfg.host, "file-host");
        assert_eq!(cfg.port, 3333);
    }

    #[test]
    fn parse_grpc_endpoint_cases() {
        assert_eq!(
            parse_grpc_endpoint("http://myhost:9090"),
            GrpcEndpoint {
                host: "myhost".into(),
                port: 9090
            }
        );
        assert_eq!(parse_grpc_endpoint("onlyhost").port, DEFAULT_GRPC_PORT);
        assert_eq!(parse_grpc_endpoint("srv:7777").port, 7777);
        assert_eq!(parse_grpc_endpoint("host:notaport").port, DEFAULT_GRPC_PORT);
        assert_eq!(parse_grpc_endpoint("").host, "");
        assert_eq!(parse_grpc_endpoint("http://myhost").port, DEFAULT_GRPC_PORT);
    }

    #[test]
    fn parse_int_prefix_cases() {
        assert_eq!(parse_int_prefix("8080abc"), Some(8080));
        assert_eq!(parse_int_prefix("  8080"), Some(8080));
        assert_eq!(parse_int_prefix("-5x"), Some(-5));
        assert_eq!(parse_int_prefix("+7"), Some(7));
        assert_eq!(parse_int_prefix("abc"), None);
        assert_eq!(parse_int_prefix(""), None);
    }
}
