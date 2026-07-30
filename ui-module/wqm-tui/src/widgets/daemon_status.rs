//! The daemon-status zone — VISUAL-LANGUAGE.md §4 and §7, typed on N49.
//!
//! This widget renders [`DaemonReport`] straight from `wqm-client`. That matters more
//! than it looks: N49's load-bearing decision is that **unreachability is data, not an
//! error**, so "no daemon" arrives as a value with a reason attached rather than as a
//! failed call. UX-F021 asks the launch screen to show an unreachable state with alarm
//! chrome rather than an empty workspace — and because the report is a two-variant enum,
//! a frame that forgot the unreachable case would not compile.
//!
//! The health mapping is the one judgement this widget makes, and it is deliberate:
//! `Unrecognized` maps to *degraded*, not healthy. The daemon answered but made a claim
//! this build cannot read, and reporting that as green would be the lie N49's own comment
//! warns against.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};
use wqm_client::{DaemonReport, DaemonState, DaemonStatus, IndexState, UnreachableReason};

use crate::tokens::{self, Health};

/// The N49 report, reduced to the three states §4 has glyphs for.
fn health_of(report: &DaemonReport) -> Health {
    match report {
        DaemonReport::Reachable(status) => match status.state {
            DaemonState::Ok => Health::Healthy,
            DaemonState::Unrecognized(_) => Health::Degraded,
        },
        DaemonReport::Unreachable { .. } => Health::Offline,
    }
}

/// The short label beside the glyph. Kept terse because §4 wants the glyph to carry the
/// state and the label to stay muted.
fn state_label(report: &DaemonReport) -> String {
    match report {
        DaemonReport::Reachable(status) => match status.state {
            DaemonState::Ok => "ok".to_string(),
            // Showing the raw discriminant is the honest move: it is the only thing this
            // build actually knows about the state.
            DaemonState::Unrecognized(code) => format!("unrecognized ({code})"),
        },
        DaemonReport::Unreachable { reason, .. } => match reason {
            // The wire spelling comes from N49 so the screen, the agent and telemetry all
            // say the same word (MCP-SURFACE §4.5 rule 1).
            UnreachableReason::DaemonUnreachable => UnreachableReason::DaemonUnreachable
                .as_str()
                .replace('_', " "),
        },
    }
}

pub struct DaemonPanel {
    report: DaemonReport,
}

impl DaemonPanel {
    pub fn new(report: DaemonReport) -> Self {
        Self { report }
    }

    /// A daemon serving normally, with a caught-up index.
    pub fn nominal() -> Self {
        Self::new(DaemonReport::Reachable(DaemonStatus {
            state: DaemonState::Ok,
            detail: String::new(),
            since_unix_seconds: Some(1_753_000_000),
            version: "0.2.0".to_string(),
            index: Some(IndexState {
                files_tracked: 12_840,
                queue_pending: 0,
                complete: true,
                lag_seconds: 0,
            }),
        }))
    }

    fn index_line(index: Option<&IndexState>) -> Line<'static> {
        match index {
            // `None` is a different claim from a zero-filled block, and N49 says the
            // difference is the point — so the screen must not render it as "0 files".
            None => Line::from(vec![
                Span::styled("index       ", tokens::muted_style()),
                Span::styled("not built in this daemon", tokens::faint_style()),
            ]),
            Some(index) => {
                let lag = if index.complete {
                    Span::styled("caught up", tokens::faint_style())
                } else {
                    // The one datum that must be seen when the index is behind (goal 7).
                    Span::styled(
                        format!("behind by {}s", index.lag_seconds),
                        Style::default()
                            .fg(Health::Degraded.color())
                            .add_modifier(ratatui::style::Modifier::BOLD),
                    )
                };

                Line::from(vec![
                    Span::styled("index       ", tokens::muted_style()),
                    Span::styled(
                        format!("{} files", index.files_tracked),
                        tokens::normal_style(),
                    ),
                    Span::styled(" · ", tokens::faint_style()),
                    Span::styled(
                        format!("{} queued", index.queue_pending),
                        tokens::faint_style(),
                    ),
                    Span::styled(" · ", tokens::faint_style()),
                    lag,
                ])
            }
        }
    }
}

impl Widget for DaemonPanel {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let health = health_of(&self.report);

        let mut lines = vec![Line::from(vec![
            Span::styled("daemon      ", tokens::muted_style()),
            Span::styled(health.glyph(), Style::default().fg(health.color())),
            Span::styled(
                format!(" {}", state_label(&self.report)),
                tokens::normal_style(),
            ),
        ])];

        match &self.report {
            DaemonReport::Reachable(status) => {
                lines.push(Line::from(vec![
                    Span::styled("version     ", tokens::muted_style()),
                    Span::styled(status.version.clone(), tokens::faint_style()),
                ]));
                if !status.detail.is_empty() {
                    lines.push(Line::from(vec![
                        Span::styled("detail      ", tokens::muted_style()),
                        Span::styled(status.detail.clone(), tokens::normal_style()),
                    ]));
                }
                lines.push(Self::index_line(status.index.as_ref()));
            }
            DaemonReport::Unreachable { address, .. } => {
                // Where we looked is what makes the answer actionable — N49 carries the
                // address precisely so the screen can say it.
                lines.push(Line::from(vec![
                    Span::styled("address     ", tokens::muted_style()),
                    Span::styled(address.clone(), tokens::faint_style()),
                ]));
                lines.push(Line::from(vec![Span::styled(
                    "no daemon is answering — start it with `wqm service start`",
                    tokens::normal_style(),
                )]));
            }
        }

        Paragraph::new(lines).render(area, buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};
    use wqm_proto::Address;

    const PROPS: &[PropInfo] = &[PropInfo {
        name: "report",
        ty: "DaemonReport",
        description: "N49's two-variant answer; Unreachable carries reason + address",
    }];

    macro_rules! variant {
        ($ty:ident, $name:literal, $desc:literal, $build:expr) => {
            struct $ty;
            impl Ingredient for $ty {
                fn group(&self) -> &str {
                    "Daemon Status"
                }
                fn name(&self) -> &str {
                    $name
                }
                fn source(&self) -> &str {
                    "wqm_tui::widgets::daemon_status"
                }
                fn description(&self) -> &str {
                    $desc
                }
                fn props(&self) -> &[PropInfo] {
                    PROPS
                }
                fn render(&self, area: Rect, buf: &mut Buffer) {
                    let w: DaemonPanel = $build;
                    w.render(area, buf);
                }
            }
        };
    }

    variant!(
        Nominal,
        "Serving",
        "Daemon ok, index caught up — the quiet case the other variants are judged against",
        DaemonPanel::nominal()
    );

    variant!(
        Unreachable,
        "Unreachable",
        "UX-F021: no daemon at launch must read as an alarm state, never as an empty workspace",
        DaemonPanel::new(DaemonReport::Unreachable {
            reason: UnreachableReason::DaemonUnreachable,
            address: Address::Uds("/Users/chris/.wqm/memexd.sock".into()).to_string(),
        })
    );

    variant!(
        IndexLagging,
        "Index Behind",
        "Goal 7: the lag figure is the one datum pushed forward, while the daemon stays green",
        DaemonPanel::new(DaemonReport::Reachable(DaemonStatus {
            state: DaemonState::Ok,
            detail: String::new(),
            since_unix_seconds: Some(1_753_000_000),
            version: "0.2.0".to_string(),
            index: Some(IndexState {
                files_tracked: 12_840,
                queue_pending: 417,
                complete: false,
                lag_seconds: 96,
            }),
        }))
    );

    variant!(
        UnknownState,
        "Unrecognized State",
        "A future-dated daemon made a claim this build cannot read — degraded, never green",
        DaemonPanel::new(DaemonReport::Reachable(DaemonStatus {
            state: DaemonState::Unrecognized(7),
            detail: "reported by a newer build".to_string(),
            since_unix_seconds: None,
            version: "0.3.0-dev".to_string(),
            index: None,
        }))
    );

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Nominal),
            Box::new(Unreachable),
            Box::new(IndexLagging),
            Box::new(UnknownState),
        ]
    }
}
