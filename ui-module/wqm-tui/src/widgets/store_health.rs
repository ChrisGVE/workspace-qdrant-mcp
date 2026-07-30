//! Store-health rows — VISUAL-LANGUAGE.md §4 and §7.
//!
//! Backends are swappable behind seams, so the UI labels by ROLE and shows the concrete
//! backend as a dim binding: `vector ● qdrant`. Only the glyph carries colour; the label
//! is muted and the binding is fainter still, so a healthy screen stays quiet and a
//! degraded one stands out without the panel shouting.
//!
//! # These names are NOT yet N8-owned
//!
//! Unlike [`crate::widgets::collections`], which draws every name from
//! `wqm_common::names`, the role and binding strings below are spelled locally. N8's
//! Phase-0 vocabulary covers collections, environment keys, and the access sets — it does
//! not yet spell the store roles (`vector`, `graph`, `relational`) or their backend
//! bindings (`qdrant`, `ladybug`, `sqlite`, `memexd`). Adding them is an N8 slice with a
//! work order behind it, not a change this storyboard may make on its own, so they stay
//! here and stay flagged. The CI guard (`src/rust/ci/guard_name_registry.py`) scans only
//! `src/rust/`, so nothing fails today — but these literals become a hard CI failure the
//! moment these widgets migrate into the workspace.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens::{self, Health};

/// One store: the role the UI commits to, the backend currently bound to it, its state.
pub struct StoreRow {
    pub role: &'static str,
    pub binding: &'static str,
    pub health: Health,
}

/// The Service zone's store list.
pub struct StoreHealth {
    rows: Vec<StoreRow>,
}

impl StoreHealth {
    pub fn new(rows: Vec<StoreRow>) -> Self {
        Self { rows }
    }

    /// The federation as §7 describes it, all nominal.
    pub fn nominal() -> Self {
        Self::new(vec![
            StoreRow {
                role: "daemon",
                binding: "memexd",
                health: Health::Healthy,
            },
            StoreRow {
                role: "vector",
                binding: "qdrant",
                health: Health::Healthy,
            },
            StoreRow {
                role: "graph",
                binding: "ladybug",
                health: Health::Healthy,
            },
            StoreRow {
                role: "relational",
                binding: "sqlite",
                health: Health::Healthy,
            },
        ])
    }
}

impl Widget for StoreHealth {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Role labels are padded to a common width so the glyph column aligns; a ragged
        // glyph column is what makes a status panel hard to scan.
        let label_width = self.rows.iter().map(|r| r.role.len()).max().unwrap_or(0);

        let lines: Vec<Line> = self
            .rows
            .iter()
            .map(|row| {
                Line::from(vec![
                    Span::styled(
                        format!("{:<label_width$}  ", row.role),
                        tokens::muted_style(),
                    ),
                    Span::styled(row.health.glyph(), Style::default().fg(row.health.color())),
                    Span::styled(format!(" {}", row.binding), tokens::faint_style()),
                ])
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
            ty: "&str",
            description: "The role the UI commits to; never the backend name",
        },
        PropInfo {
            name: "binding",
            ty: "&str",
            description: "Concrete backend currently bound to the role, rendered faint",
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
            StoreRow { role: "daemon", binding: "memexd", health: Health::Healthy },
            StoreRow { role: "vector", binding: "qdrant", health: Health::Healthy },
            StoreRow { role: "graph", binding: "ladybug", health: Health::Degraded },
            StoreRow { role: "relational", binding: "sqlite", health: Health::Healthy },
        ])
    );

    variant!(
        VectorOffline,
        "Vector Offline",
        "Qdrant unreachable at QDRANT_URL — tests whether offline out-shouts degraded",
        StoreHealth::new(vec![
            StoreRow {
                role: "daemon",
                binding: "memexd",
                health: Health::Healthy
            },
            StoreRow {
                role: "vector",
                binding: "qdrant",
                health: Health::Offline
            },
            StoreRow {
                role: "graph",
                binding: "ladybug",
                health: Health::Degraded
            },
            StoreRow {
                role: "relational",
                binding: "sqlite",
                health: Health::Healthy
            },
        ])
    );

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Nominal),
            Box::new(Degraded),
            Box::new(VectorOffline),
        ]
    }
}
