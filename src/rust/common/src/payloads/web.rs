//! Payloads for web content: single URLs and multi-page website crawls

use serde::{Deserialize, Serialize};

pub(super) fn default_crawl_depth() -> u32 {
    2
}

pub(super) fn default_max_pages() -> u32 {
    50
}

/// Payload for URL fetch and ingestion items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrlPayload {
    /// The URL to fetch
    pub url: String,
    /// Whether to crawl linked pages (same domain only)
    #[serde(default)]
    pub crawl: bool,
    /// Maximum crawl depth (0 = single page, default: 2)
    #[serde(default = "default_crawl_depth")]
    pub max_depth: u32,
    /// Maximum pages to crawl (default: 50)
    #[serde(default = "default_max_pages")]
    pub max_pages: u32,
    /// Content type hint from HTTP HEAD (populated by fetcher)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Library name (when storing to libraries collection)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub library_name: Option<String>,
    /// Title extracted from the page
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// Payload for website items (multi-page crawl)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebsitePayload {
    /// Root URL of the website
    pub url: String,
    /// Maximum crawl depth from root (default: 2)
    #[serde(default = "default_crawl_depth")]
    pub max_depth: u32,
    /// Maximum pages to crawl (default: 50)
    #[serde(default = "default_max_pages")]
    pub max_pages: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_payload_full_serde() {
        let payload = UrlPayload {
            url: "https://example.com/docs".to_string(),
            crawl: true,
            max_depth: 3,
            max_pages: 100,
            content_type: Some("text/html".to_string()),
            library_name: Some("example-docs".to_string()),
            title: Some("Example Documentation".to_string()),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("https://example.com/docs"));
        assert!(json.contains("\"crawl\":true"));
        assert!(json.contains("\"max_depth\":3"));

        let back: UrlPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.url, "https://example.com/docs");
        assert!(back.crawl);
        assert_eq!(back.max_depth, 3);
        assert_eq!(back.library_name, Some("example-docs".to_string()));
    }

    #[test]
    fn test_url_payload_minimal_serde() {
        let json = r#"{"url":"https://example.com"}"#;
        let payload: UrlPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.url, "https://example.com");
        assert!(!payload.crawl);
        assert_eq!(payload.max_depth, 2);
        assert_eq!(payload.max_pages, 50);
        assert_eq!(payload.content_type, None);
        assert_eq!(payload.library_name, None);
    }

    #[test]
    fn test_website_payload_serde() {
        let payload = WebsitePayload {
            url: "https://docs.rs/tokio".to_string(),
            max_depth: 3,
            max_pages: 100,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("https://docs.rs/tokio"));

        let back: WebsitePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.url, "https://docs.rs/tokio");
        assert_eq!(back.max_depth, 3);
        assert_eq!(back.max_pages, 100);
    }

    #[test]
    fn test_website_payload_defaults() {
        let json = r#"{"url":"https://example.com"}"#;
        let payload: WebsitePayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.max_depth, 2);
        assert_eq!(payload.max_pages, 50);
    }
}
