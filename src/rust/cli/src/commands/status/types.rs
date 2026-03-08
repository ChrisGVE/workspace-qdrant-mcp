//! JSON-serializable types and display helpers for status commands.

use serde::Serialize;

use crate::output::ServiceStatus;

/// JSON-serializable system status
#[derive(Serialize)]
pub struct SystemStatusJson {
    pub connected: bool,
    pub status: String,
    pub collections: i32,
    pub documents: i32,
    pub active_projects: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending_operations: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idle_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_max_embeddings: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_inter_item_delay_ms: Option<i64>,
}

/// JSON-serializable health status
#[derive(Serialize)]
pub struct HealthStatusJson {
    pub connected: bool,
    pub health: String,
    pub components: Vec<HealthComponentJson>,
}

/// JSON-serializable health component
#[derive(Serialize)]
pub struct HealthComponentJson {
    pub name: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

pub fn status_label(s: ServiceStatus) -> &'static str {
    match s {
        ServiceStatus::Healthy => "healthy",
        ServiceStatus::Degraded => "degraded",
        ServiceStatus::Unhealthy => "unhealthy",
        ServiceStatus::Active => "active",
        ServiceStatus::Inactive => "inactive",
        ServiceStatus::Unknown => "unknown",
    }
}

pub fn format_bytes(bytes: i64) -> String {
    const KB: i64 = 1024;
    const MB: i64 = KB * 1024;
    const GB: i64 = MB * 1024;

    if bytes < KB {
        format!("{} B", bytes)
    } else if bytes < MB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else if bytes < GB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes_small() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1023), "1023 B");
    }

    #[test]
    fn test_format_bytes_kb() {
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
    }

    #[test]
    fn test_format_bytes_mb() {
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 + 512 * 1024), "1.5 MB");
    }

    #[test]
    fn test_format_bytes_gb() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_status_label() {
        assert_eq!(status_label(ServiceStatus::Healthy), "healthy");
        assert_eq!(status_label(ServiceStatus::Degraded), "degraded");
        assert_eq!(status_label(ServiceStatus::Unhealthy), "unhealthy");
        assert_eq!(status_label(ServiceStatus::Unknown), "unknown");
    }

    #[test]
    fn test_system_status_json_serialization() {
        let json_out = SystemStatusJson {
            connected: true,
            status: "healthy".to_string(),
            collections: 3,
            documents: 100,
            active_projects: vec!["project-a".to_string()],
            pending_operations: Some(5),
            resource_mode: Some("normal".to_string()),
            idle_seconds: Some(42.5),
            current_max_embeddings: Some(1),
            current_inter_item_delay_ms: Some(100),
        };
        let serialized = serde_json::to_string(&json_out).unwrap();
        assert!(serialized.contains("\"connected\":true"));
        assert!(serialized.contains("\"resource_mode\":\"normal\""));
        assert!(serialized.contains("\"idle_seconds\":42.5"));
        assert!(serialized.contains("\"current_max_embeddings\":1"));
        assert!(serialized.contains("\"current_inter_item_delay_ms\":100"));
    }

    #[test]
    fn test_system_status_json_omits_none_fields() {
        let json_out = SystemStatusJson {
            connected: false,
            status: "unhealthy".to_string(),
            collections: 0,
            documents: 0,
            active_projects: Vec::new(),
            pending_operations: None,
            resource_mode: None,
            idle_seconds: None,
            current_max_embeddings: None,
            current_inter_item_delay_ms: None,
        };
        let serialized = serde_json::to_string(&json_out).unwrap();
        assert!(!serialized.contains("resource_mode"));
        assert!(!serialized.contains("idle_seconds"));
        assert!(!serialized.contains("current_max_embeddings"));
        assert!(!serialized.contains("pending_operations"));
    }

    #[test]
    fn test_health_status_json_serialization() {
        let json_out = HealthStatusJson {
            connected: true,
            health: "healthy".to_string(),
            components: vec![
                HealthComponentJson {
                    name: "qdrant".to_string(),
                    status: "healthy".to_string(),
                    message: None,
                },
                HealthComponentJson {
                    name: "sqlite".to_string(),
                    status: "degraded".to_string(),
                    message: Some("high latency".to_string()),
                },
            ],
        };
        let serialized = serde_json::to_string(&json_out).unwrap();
        assert!(serialized.contains("\"health\":\"healthy\""));
        assert!(serialized.contains("\"qdrant\""));
        assert!(serialized.contains("\"high latency\""));
        // Component with message: None should omit message field
        let value: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        let components = value["components"].as_array().unwrap();
        assert!(components[0].get("message").is_none());
        assert!(components[1].get("message").is_some());
    }
}
