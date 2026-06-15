//! Bounded model-label interner.
//!
//! The telemetry sample (`MetricSample`) is `Copy` and carries its model label as
//! a `&'static str` (arch §1.6 — stable labels are baked in, never re-resolved per
//! emit). But the embedder-batch emit site (`EmbeddingGenerator::generate_embeddings_batch`)
//! receives its model label as a runtime `&str` argument that originates from
//! several call sites — most are string literals, but the tagging path passes
//! `provider_label()`, which is not `'static`. To preserve that label **exactly**
//! through the `Copy` buffer (zero telemetry regression), it is interned here to a
//! process-lifetime `&'static str`.
//!
//! The label set is closed in practice (a handful of provider/model names), so the
//! one-time leak per distinct label is bounded — never an unbounded leak. The emit
//! site is per-batch (coarse), not the per-chunk hot path, so the single lock +
//! set lookup is off the budget the arch §9 hot-path proof covers.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

fn pool() -> &'static Mutex<HashSet<&'static str>> {
    static POOL: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Intern a model label to a process-lifetime `&'static str`. On a repeat label
/// the already-interned pointer is returned (steady state: one lock + a set hit,
/// no allocation). On a new label the string is leaked once and remembered.
pub fn intern_model_label(label: &str) -> &'static str {
    let mut pool = pool().lock().expect("model-label intern pool poisoned");
    if let Some(&existing) = pool.get(label) {
        return existing;
    }
    let leaked: &'static str = Box::leak(label.to_owned().into_boxed_str());
    pool.insert(leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_label_interns_to_same_pointer() {
        let a = intern_model_label("fastembed");
        let b = intern_model_label(&String::from("fastembed"));
        // Same content → identical interned pointer (set hit, no second leak).
        assert!(std::ptr::eq(a, b));
        assert_eq!(a, "fastembed");
    }

    #[test]
    fn test_distinct_labels_distinct_pointers() {
        let a = intern_model_label("model-a");
        let b = intern_model_label("model-b");
        assert!(!std::ptr::eq(a, b));
        assert_eq!(a, "model-a");
        assert_eq!(b, "model-b");
    }
}
