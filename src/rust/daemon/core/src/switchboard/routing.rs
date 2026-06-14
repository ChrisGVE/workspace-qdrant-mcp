//! Routing table — id-keyed control dispatch, built once at init.
//!
//! One `RoutingEntry` per `MetricId`, indexed by discriminant (`id as usize`).
//! Each entry holds an optional control fn-pointer. Telemetry is the automatic
//! default for every id and needs no entry — the hot path always attempts the
//! buffer push when `telemetry_on` is set. The table is frozen as a
//! `Box<[RoutingEntry]>` after `seal()`: read-only, no locks, no `dyn` (arch
//! §2, §6c). A bare `fn` pointer is a single indirect call — no vtable — which
//! is why it satisfies the hot-path budget (arch §9).

use super::{ControlFanout, MetricId, MetricSample, METRIC_COUNT};

/// Control-dispatch function pointer: stores the sample's value into the fanout.
/// One indirect call, no vtable (arch §9 "why fn-pointer ≠ dynamic dispatch").
pub type ControlFn = fn(&ControlFanout, &MetricSample);

/// One routing-table slot. Telemetry needs no entry (automatic default); only
/// the optional control fn is held here.
#[derive(Clone, Copy)]
pub struct RoutingEntry {
    pub control_fn: Option<ControlFn>,
}

impl Default for RoutingEntry {
    fn default() -> Self {
        Self { control_fn: None }
    }
}

/// Builder for the routing table. `wire_control` mutates entries before `build`
/// freezes the table — there is no after-build mutation path.
pub struct RoutingTableBuilder {
    entries: Vec<RoutingEntry>,
}

impl RoutingTableBuilder {
    pub fn new() -> Self {
        Self {
            entries: vec![RoutingEntry::default(); METRIC_COUNT],
        }
    }

    /// Attach a control fn to a metric id. Overwrites any prior fn for that id.
    pub fn wire_control(&mut self, id: MetricId, f: ControlFn) {
        self.entries[id as usize].control_fn = Some(f);
    }

    /// Freeze the table into a read-only boxed slice indexed by discriminant.
    pub fn build(self) -> Box<[RoutingEntry]> {
        self.entries.into_boxed_slice()
    }
}

impl Default for RoutingTableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noop(_f: &ControlFanout, _s: &MetricSample) {}

    #[test]
    fn test_wire_sets_entry_by_discriminant() {
        let mut b = RoutingTableBuilder::new();
        b.wire_control(MetricId::EmbedderLatency, noop);
        let table = b.build();
        assert_eq!(table.len(), METRIC_COUNT);
        assert!(table[MetricId::EmbedderLatency as usize]
            .control_fn
            .is_some());
        assert!(table[MetricId::QueueItemMs as usize].control_fn.is_none());
    }
}
