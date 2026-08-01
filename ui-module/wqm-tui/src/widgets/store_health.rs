//! Store-health rows — VISUAL-LANGUAGE.md §4 and §7.
//!
//! Backends are swappable behind seams, so the UI labels by ROLE and shows the concrete
//! backend as a dim binding: `vector ● qdrant`. Only the glyph carries colour; the label
//! is muted and the binding is fainter still, so a healthy screen stays quiet and a
//! degraded one stands out without the panel shouting.
//!
//! # The component set is DATA, not a UI constant (`CR-036`(a))
//!
//! An earlier shape of this widget spelled its four components as `&'static str`, which
//! meant only a component known at compile time could be drawn — a UI change per component
//! the daemon grows. `CR-035` §9 carries the *whole* report on one channel precisely so the
//! set can grow without a wire change per component, and v0.1 already modelled it as
//! `components: Vec<ComponentHealth>` with a free-form `component_name`. Fixed slots defeat
//! that.
//!
//! So this widget renders whatever list arrives. [`StoreRow::unreadable`] is the graceful
//! path for a component whose reported state this build cannot map: it is shown *degraded*
//! rather than dropped or greened, the same property [`wqm_client::DaemonReport`]'s
//! `Unreachable` variant establishes for reachability.
//!
//! # These names are NOT yet N8-owned, and nothing downstream will catch that
//!
//! Unlike [`crate::panes::collections`], which draws every name from
//! `wqm_common::names`, the role and binding strings in the sample lists below are spelled
//! locally. N8's Phase-0 vocabulary covers collections, environment keys, and the access
//! sets — it does not yet spell the store roles (`vector`, `graph`, `relational`) or their
//! backend bindings (`qdrant`, `ladybug`, `sqlite`, `memexd`).
//!
//! The earlier claim here — that these literals *"become a hard CI failure the moment these
//! widgets migrate"* — is **refuted** (`CR-038`). `src/rust/ci/guard_name_registry.py`
//! matches a guarded literal inside Rust string quotes; a role name N8 never registered can
//! never match one, so no build failure is waiting. The divergence would arrive silently at
//! migration instead, which makes routing it more urgent, not less.
//!
//! Whether these roles become published components at all is a scope question and Chris's
//! (`CR-036`(b)). Until he answers it, this widget stays **unbound and marked unbound**:
//! contract-binding it to the invented four is what would make the fiction load-bearing.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens::{self, Health};

/// One component of the status report: the role the UI shows, the backend bound to it if
/// the report names one, and its state.
///
/// `role` is an owned `String` because it arrives from the wire, not from this crate — see
/// the module docs. `binding` is optional for the same reason: a component the report names
/// without a backend still has to render, and inventing a binding for it would be a lie in
/// the faint column.
pub struct StoreRow {
    pub role: String,
    pub binding: Option<String>,
    pub health: Health,
}

impl StoreRow {
    /// A component with a known state and a named backend behind it.
    pub fn bound(role: impl Into<String>, binding: impl Into<String>, health: Health) -> Self {
        Self {
            role: role.into(),
            binding: Some(binding.into()),
            health,
        }
    }

    /// A component with a known state and no backend to name.
    pub fn unbound(role: impl Into<String>, health: Health) -> Self {
        Self {
            role: role.into(),
            binding: None,
            health,
        }
    }

    /// A component whose reported state this build cannot read.
    ///
    /// Degraded, never healthy and never dropped: the daemon named a component and made a
    /// claim about it, and both halves of that are information. This is the same judgement
    /// `daemon_status::health_of` makes for `DaemonState::Unrecognized` — green would be a
    /// claim this build has no basis for.
    pub fn unreadable(role: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            binding: None,
            health: Health::Degraded,
        }
    }
}

/// The Service zone's component list.
pub struct StoreHealth {
    rows: Vec<StoreRow>,
}

impl StoreHealth {
    pub fn new(rows: Vec<StoreRow>) -> Self {
        Self { rows }
    }

    /// The federation as §7 describes it, all nominal.
    ///
    /// An **illustrative list, not the shape** — these four are what VISUAL-LANGUAGE §7
    /// draws, not a set this widget is limited to. `StatusResponse` cannot express them
    /// today (`CR-036`), so nothing here is contract-bound.
    pub fn nominal() -> Self {
        Self::new(vec![
            StoreRow::bound("daemon", "memexd", Health::Healthy),
            StoreRow::bound("vector", "qdrant", Health::Healthy),
            StoreRow::bound("graph", "ladybug", Health::Healthy),
            StoreRow::bound("relational", "sqlite", Health::Healthy),
        ])
    }
}

impl Widget for StoreHealth {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Role labels are padded to a common width so the glyph column aligns; a ragged
        // glyph column is what makes a status panel hard to scan. Measured in display
        // columns rather than bytes, because a component name off the wire may be neither
        // ASCII nor single-width.
        let label_width = self
            .rows
            .iter()
            .map(|r| Span::raw(r.role.as_str()).width())
            .max()
            .unwrap_or(0);

        let lines: Vec<Line> = self
            .rows
            .iter()
            .map(|row| {
                let pad = label_width.saturating_sub(Span::raw(row.role.as_str()).width());
                let mut spans = vec![
                    Span::styled(
                        format!("{}{}  ", row.role, " ".repeat(pad)),
                        tokens::muted_style(),
                    ),
                    Span::styled(row.health.glyph(), Style::default().fg(row.health.color())),
                ];
                // An absent binding leaves the faint column empty rather than filling it
                // with a placeholder: the glyph already carries the state, and a component
                // with nothing behind it should look like one.
                if let Some(binding) = &row.binding {
                    spans.push(Span::styled(format!(" {binding}"), tokens::faint_style()));
                }
                Line::from(spans)
            })
            .collect();

        Paragraph::new(lines).render(area, buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "role",
            ty: "String",
            description:
                "Component name as the report spells it; owned, because it arrives from the wire",
        },
        PropInfo {
            name: "binding",
            ty: "Option<String>",
            description:
                "Backend bound to the role, rendered faint; absent leaves the column empty",
        },
        PropInfo {
            name: "health",
            ty: "Health",
            description: "Healthy / Degraded / Offline — drives glyph shape AND hue",
        },
    ];

    macro_rules! variant {
        ($ty:ident, $name:literal, $desc:literal, $build:expr) => {
            struct $ty;
            impl Ingredient for $ty {
                fn group(&self) -> &str {
                    "Store Health"
                }
                fn name(&self) -> &str {
                    $name
                }
                fn source(&self) -> &str {
                    "wqm_tui::widgets::store_health"
                }
                fn description(&self) -> &str {
                    $desc
                }
                fn props(&self) -> &[PropInfo] {
                    PROPS
                }
                fn render(&self, area: Rect, buf: &mut Buffer) {
                    let w: StoreHealth = $build;
                    w.render(area, buf);
                }
            }
        };
    }

    variant!(
        Nominal,
        "All Healthy",
        "The quiet default: every store nominal, nothing competing for the eye",
        StoreHealth::nominal()
    );

    variant!(
        Degraded,
        "Graph Degraded",
        "One store past its freshness SLA — the yellow triangle should be the only thing that pulls",
        StoreHealth::new(vec![
            StoreRow::bound("daemon", "memexd", Health::Healthy),
            StoreRow::bound("vector", "qdrant", Health::Healthy),
            StoreRow::bound("graph", "ladybug", Health::Degraded),
            StoreRow::bound("relational", "sqlite", Health::Healthy),
        ])
    );

    variant!(
        VectorOffline,
        "Vector Offline",
        "Qdrant unreachable at QDRANT_URL — tests whether offline out-shouts degraded",
        StoreHealth::new(vec![
            StoreRow::bound("daemon", "memexd", Health::Healthy),
            StoreRow::bound("vector", "qdrant", Health::Offline),
            StoreRow::bound("graph", "ladybug", Health::Degraded),
            StoreRow::bound("relational", "sqlite", Health::Healthy),
        ])
    );

    variant!(
        ArrivingSet,
        "Arriving Set",
        "CR-036(a): six components this build never heard of, one unreadable — the list is data, and the layout must hold",
        StoreHealth::new(vec![
            StoreRow::bound("daemon", "memexd", Health::Healthy),
            StoreRow::bound("vector", "qdrant", Health::Healthy),
            // Named by the report with no backend behind it: the faint column stays empty
            // rather than being filled with an invented binding.
            StoreRow::unbound("queue_processor", Health::Degraded),
            StoreRow::unbound("embedding_provider", Health::Healthy),
            // A component whose state this build cannot map. Degraded and present, never
            // green and never dropped.
            StoreRow::unreadable("watch_supervisor"),
            // A long name proves the glyph column is computed, not assumed.
            StoreRow::bound("language_registry", "bundled", Health::Healthy),
        ])
    );

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Nominal),
            Box::new(Degraded),
            Box::new(VectorOffline),
            Box::new(ArrivingSet),
        ]
    }
}
