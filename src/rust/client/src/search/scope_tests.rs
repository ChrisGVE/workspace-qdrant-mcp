//! Tests for the SQLite-free scope helpers: decay-map building and relevance
//! decay. Project isolation is the tenant-id filter alone — the former
//! `base_point` worktree-isolation path was removed (#115).

use std::collections::HashMap;

use serde_json::Value;

use super::*;
use crate::qdrant::fusion::{SearchType, TaggedResult};
use crate::workspace_daemon::{ResolveSearchScopeResponse, TenantDecay};

fn tagged(id: &str, score: f64, tenant: Option<&str>) -> TaggedResult {
    let mut payload: HashMap<String, Value> = HashMap::new();
    if let Some(t) = tenant {
        payload.insert("tenant_id".to_string(), Value::String(t.to_string()));
    }
    TaggedResult {
        id: id.to_string(),
        score,
        collection: "projects".to_string(),
        payload,
        search_type: SearchType::Semantic,
    }
}

// ── scope_filter_from_response ──────────────────────────────────────────────

#[test]
fn scope_filter_group_returns_tenant_ids_and_decay() {
    let resp = ResolveSearchScopeResponse {
        tenant_ids: vec!["a".into(), "b".into()],
        filter_by_tenant: true,
        decay_map: vec![
            TenantDecay {
                tenant_id: "a".into(),
                multiplier: 1.0,
            },
            TenantDecay {
                tenant_id: "b".into(),
                multiplier: 0.7,
            },
        ],
    };
    let (ids, decay) = scope_filter_from_response(&resp);
    assert_eq!(ids, Some(vec!["a".to_string(), "b".to_string()]));
    let decay = decay.unwrap();
    assert_eq!(decay.get("a"), Some(&1.0));
    // 0.7 originates as an f32 proto field widened to f64, so compare with tolerance.
    assert!((decay.get("b").copied().unwrap() - 0.7).abs() < 1e-6);
}

#[test]
fn scope_filter_all_returns_no_tenant_filter_but_decay() {
    // scope=all: filter_by_tenant=false → no group filter, decay still present.
    let resp = ResolveSearchScopeResponse {
        tenant_ids: vec!["a".into()],
        filter_by_tenant: false,
        decay_map: vec![TenantDecay {
            tenant_id: "a".into(),
            multiplier: 1.0,
        }],
    };
    let (ids, decay) = scope_filter_from_response(&resp);
    assert_eq!(ids, None);
    assert!(decay.is_some());
}

#[test]
fn scope_filter_empty_decay_is_none() {
    let resp = ResolveSearchScopeResponse {
        tenant_ids: vec![],
        filter_by_tenant: true,
        decay_map: vec![],
    };
    let (ids, decay) = scope_filter_from_response(&resp);
    assert_eq!(ids, Some(vec![]));
    assert!(decay.is_none());
}

// ── apply_relevance_decay ───────────────────────────────────────────────────

#[test]
fn relevance_decay_multiplies_and_resorts() {
    let mut results = vec![
        tagged("other", 0.9, Some("other_tenant")), // *0.4 = 0.36
        tagged("current", 0.8, Some("cur")),        // *1.0 = 0.80
        tagged("untenanted", 0.5, None),            // unchanged 0.50
    ];
    let mut decay = HashMap::new();
    decay.insert("cur".to_string(), 1.0);
    // "other_tenant" absent → default 0.4.
    apply_relevance_decay(&mut results, &decay);

    // Re-sorted by decayed score desc: current(0.80) > untenanted(0.50) > other(0.36).
    assert_eq!(results[0].id, "current");
    assert!((results[0].score - 0.80).abs() < 1e-9);
    assert_eq!(results[1].id, "untenanted");
    assert!((results[1].score - 0.50).abs() < 1e-9);
    assert_eq!(results[2].id, "other");
    assert!((results[2].score - 0.36).abs() < 1e-9);
}
