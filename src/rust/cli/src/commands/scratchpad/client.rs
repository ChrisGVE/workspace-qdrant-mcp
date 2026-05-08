//! Qdrant HTTP client and tenant resolution helpers

use anyhow::{Context, Result};

pub(super) fn qdrant_url() -> String {
    crate::config::resolve_qdrant_url()
}

pub(super) fn build_qdrant_client() -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder().timeout(std::time::Duration::from_secs(10));

    if let Some(api_key) = crate::config::resolve_qdrant_api_key() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "api-key",
            reqwest::header::HeaderValue::from_str(&api_key).context("Invalid QDRANT_API_KEY")?,
        );
        builder = builder.default_headers(headers);
    }

    builder.build().context("Failed to create HTTP client")
}

pub(super) fn resolve_tenant_id(project: Option<&str>) -> Result<String> {
    match project {
        None => Ok(wqm_common::constants::TENANT_GLOBAL.to_string()),
        Some(p) => {
            let path = std::path::Path::new(p);
            if path.exists() {
                Ok(wqm_common::project_id::calculate_tenant_id(path))
            } else {
                // Assume it's a project ID directly
                Ok(p.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_tenant_id_default() {
        assert_eq!(resolve_tenant_id(None).unwrap(), "global");
    }

    #[test]
    fn test_resolve_tenant_id_direct_id() {
        assert_eq!(
            resolve_tenant_id(Some("proj_abc123")).unwrap(),
            "proj_abc123"
        );
    }
}
