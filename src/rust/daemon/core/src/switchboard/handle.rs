//! The emitter handle — resolved once, stored per-instance.
//!
//! A `MetricHandle` is `{ id, model }`: `id` is the routing key, `model` is the
//! stable label baked in at registration (arch §1.6, §5c). Emitters obtain one
//! from `MetricsSwitchboard::handle(id, model)` at construction and store it as
//! an instance field; it is never created on the hot path.

use super::MetricId;

/// Resolved once at emitter init via `handle()`. `Copy` and minimal — two fields,
/// no heap — so passing it to every `emit*` call costs nothing.
#[derive(Debug, Clone, Copy)]
pub struct MetricHandle {
    pub(crate) id: MetricId,
    pub(crate) model: &'static str,
}

impl MetricHandle {
    /// The routing key for this handle.
    #[inline]
    pub fn id(&self) -> MetricId {
        self.id
    }

    /// The stable model label carried into each emitted sample.
    #[inline]
    pub fn model(&self) -> &'static str {
        self.model
    }
}
