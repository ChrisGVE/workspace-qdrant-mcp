//! The Service hub — STORYBOARD §4.1's screen, and the first full composition.
//!
//! *"Service = the hub: top band = detailed service status + telemetry + embedder status +
//! graph/index freshness; large lower band toggles Config ↔ Logs."* That is one screen made
//! of elements every one of which was already built and judged alone: the tab row, the store
//! federation, the daemon panel, the config table, and — over the top of them — the modal,
//! the toast deck and the condition wash.
//!
//! # What composing them is for
//!
//! Three things only a whole screen can settle:
//!
//! 1. **Flatness.** r02's opening complaint is that a screen *"read flat"*. §2's answer is a
//!    posture — *"most of the screen is muted or faint"* — which is a claim about proportions
//!    and therefore unjudgeable one widget at a time.
//! 2. **Whether the reserved treatments collide.** The selector cyan appears twice on this
//!    screen (the Service tab, the Config pane), the data cursor once, health hues in three
//!    places. Isolated frames cannot show them competing.
//! 3. **Whether the grid holds across zones.** The store roles, the config keys and the tab
//!    row are drawn by three widgets that have never seen each other's margins.
//!
//! # Nothing on this screen is stated twice
//!
//! The rollup dot, the screen-wide condition and the Service tab's alarm colour are all
//! *derived* from the same [`SystemHealth`] the store rows and the daemon panel are built
//! from. A frame with a green rollup over a degraded store, or a red wash under a healthy
//! daemon panel, is not a value [`ServiceView`] can take — which is the same discipline
//! `config_table::Entry::is_changed` and `health::SystemHealth::condition` already apply,
//! carried up to the screen.
//!
//! # The daemon is named once
//!
//! §7 lists the daemon among the federation's axes *and* makes it the liveness master with a
//! panel of its own. Rendering it in both places puts the same glyph on the screen twice and
//! invites them to disagree. So [`ServiceView`] takes the *backing stores* and the daemon
//! report separately: the panel says how the daemon is, the store list says how everything it
//! is responsible for is.
//!
//! # The Logs pane is not built
//!
//! §4.1's lower band toggles Config ↔ Logs. The Config pane exists; nothing in this crate
//! renders logs, so the selector always shows Config selected. A `Logs` variant would be a
//! frame of an empty band — a depiction of a screen that does not exist yet, which is the one
//! thing a storyboard must not produce.

use std::time::Instant;

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    widgets::Widget,
};
use wqm_client::DaemonReport;

use crate::health::{Component, Rollup, SystemHealth};
use crate::tokens::{self, Condition, Health};
use crate::panes::{ConfigPane, StatusBand};
use crate::widgets::chrome::{inset, Attention, Freshness, Rule, StatusLine, TitleBar};
use crate::widgets::{
    config_table::ConfigTable,
    modal::Modal,
    store_health::StoreRow,
    surface::{ConditionBand, Surface},
    tab_bar::{Tab, TabBar},
    toast::ToastDeck,
};

/// Which zone of the screen is live. The lower band is `1`, so a screen laid out with more
/// zones later keeps these two.
const ZONE_STATUS: usize = 0;
const ZONE_CONFIG: usize = 1;

/// The Service tab's number in §4.1's row, and therefore its index.
const SERVICE_TAB: usize = 9;

/// The Service hub, composed.
pub struct ServiceView<'a> {
    /// The federation's *backing* stores — not the daemon; see the module docs.
    stores: Vec<StoreRow>,
    daemon: DaemonReport,
    config: ConfigTable,
    freshness: Freshness,
    attention: Attention,
    modal: Option<Modal>,
    /// The deck and the moment it is being rendered at. A widget never reads the clock, and
    /// a screen is no exception — the frame is a still, so the time in it is an input.
    toasts: Option<(&'a ToastDeck, Instant)>,
}

impl<'a> ServiceView<'a> {
    pub fn new(
        stores: Vec<StoreRow>,
        daemon: DaemonReport,
        config: ConfigTable,
        freshness: Freshness,
    ) -> Self {
        Self {
            stores,
            daemon,
            config,
            freshness,
            attention: Attention::None,
            modal: None,
            toasts: None,
        }
    }

    pub fn attention(mut self, attention: Attention) -> Self {
        self.attention = attention;
        self
    }

    pub fn modal(mut self, modal: Modal) -> Self {
        self.modal = Some(modal);
        self
    }

    pub fn toasts(mut self, deck: &'a ToastDeck, now: Instant) -> Self {
        self.toasts = Some((deck, now));
        self
    }

    /// The one health value everything on this screen is answered from.
    fn system(&self) -> SystemHealth {
        SystemHealth::from_report(
            &self.daemon,
            self.stores
                .iter()
                .map(|row| Component::new(row.role.clone(), row.health))
                .collect(),
        )
    }

    /// §7's single rollup dot, derived from the rows the screen is already showing.
    pub fn rollup(&self) -> Rollup {
        self.system().rollup()
    }

    /// The screen-wide condition — the wash and the band, or neither.
    pub fn condition(&self) -> Condition {
        self.system().condition()
    }

    /// §4.1's tab row with §4's must-see rule applied: the tab that owns a store in trouble
    /// is recoloured, and it is recoloured **from the same rollup the status line shows**, so
    /// a red tab over a green dot is not constructible.
    fn tabs(&self) -> TabBar {
        let rollup = self.rollup();
        let mut tabs: Vec<Tab> = TabBar::storyboard_tabs();
        if rollup.health != Health::Healthy
            && let Some(tab) = tabs.get_mut(SERVICE_TAB)
        {
            tab.alarm = Some(rollup.health);
        }
        TabBar::new(tabs, SERVICE_TAB)
    }

    /// The keys the foot of the screen offers, which depend on whether an edit is open.
    fn hints(&self) -> Vec<(&'static str, &'static str)> {
        if self.config.edit_mode().is_some() {
            vec![("↵", "validate"), ("Esc", "abandon")]
        } else {
            vec![
                ("j/k", "move"),
                ("↵", "edit"),
                ("Tab", "pane"),
                ("?", "help"),
                ("q", "quit"),
            ]
        }
    }
}

/// Every row of the screen, in order, so the layout is read once rather than counted twice.
///
/// **Two of these are panes, not rows.** `band` and `config` are whole zones handed to
/// [`StatusBand`] and [`ConfigPane`], which then lay out their own insides. That is §16's split:
/// the view decides where a zone goes and which one is live; the zone decides everything
/// within itself. The view no longer knows that the band is a heading over four rows, or that
/// the config zone opens with a selector and a gap — and asking [`StatusBand::ROWS`] for the
/// first of those is what keeps a change to the band from being a change in two files.
struct Rows {
    tabs: Rect,
    top_rule: Rect,
    title: Rect,
    band: Rect,
    seam: Rect,
    config: Rect,
    bottom_rule: Rect,
    status_line: Rect,
}

fn rows(area: Rect) -> Rows {
    let [tabs, top_rule, title, _, band, _, seam, _, config, bottom_rule, status_line] =
        Layout::vertical([
            Constraint::Length(1), // tab row
            Constraint::Length(1), // the frame rule under it
            Constraint::Length(1), // title, with the freshness right-aligned
            Constraint::Length(1), // negative space — §6 divides with space, not boxes
            Constraint::Length(StatusBand::ROWS),
            Constraint::Length(1),
            Constraint::Length(1), // the internal seam between the two bands
            Constraint::Length(1),
            Constraint::Min(0), // the config zone takes what is left
            Constraint::Length(1),
            Constraint::Length(1), // merged status + help
        ])
        .areas(area);

    Rows {
        tabs,
        top_rule,
        title,
        band,
        seam,
        config,
        bottom_rule,
        status_line,
    }
}

impl Widget for ServiceView<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let condition = self.condition();
        let rollup = self.rollup();
        let tabs = self.tabs();
        let hints = self.hints();
        let mode = self.config.edit_mode();

        // The wash goes down first, under everything (§6.18). It paints nothing at all when
        // the condition is nominal.
        Surface::with_condition(condition).render(area, buf);

        // The band the condition owns is carved off the bottom before anything is laid out,
        // so no zone is ever drawn into it and then overwritten.
        let reserved = Surface::reserved_rows(condition);
        let body = Rect {
            height: area.height.saturating_sub(reserved),
            ..area
        };
        if body.height < 8 {
            return;
        }
        let r = rows(body);

        // Everything from here to the `drop` below is the PAGE, and while a modal owns the
        // input every colour on it goes muted — VL §6, the whole screen and not the three
        // elements someone remembered ([`crate::tokens::modal`]).
        //
        // Dropped *before* the modal deliberately: the modal is the thing being answered, so
        // it keeps its own rungs, and the toast and the condition band behind it are the
        // must-see channel §6 says a modal cannot suspend. All three paint over the stack, and
        // all three are drawn after the scope has closed.
        let page = self.modal.is_some().then(tokens::ModalScope::enter);

        tabs.render(inset(r.tabs), buf);
        Rule::frame().render(r.top_rule, buf);
        TitleBar::new("Service")
            .freshness(self.freshness)
            .render(inset(r.title), buf);

        // Two zones, each handed its area and told which one the screen says is live. What a
        // band or a selector looks like from there is the pane's business (§16).
        StatusBand::new(self.stores, self.daemon, ZONE_STATUS, self.attention).render(r.band, buf);

        Rule::internal().render(r.seam, buf);

        ConfigPane::new(self.config, ZONE_CONFIG, self.attention).render(r.config, buf);

        Rule::frame().render(r.bottom_rule, buf);
        let mut status = StatusLine::new(rollup).mode(mode);
        for (key, label) in hints {
            status = status.hint(key, label);
        }
        status.render(inset(r.status_line), buf);
        drop(page);

        // Above the screen, in §6's order: the modal takes the stack, the toast sits outside
        // it and is painted over whatever is there.
        if let Some(modal) = self.modal {
            modal.render(body, buf);
        }
        if let Some((deck, now)) = self.toasts {
            // The SCREEN's corner, over the chrome (Chris, 20260731) — the toast keeps the
            // same arrival point whatever furniture a screen carries, and its own margins put
            // it flush with the content's.
            crate::widgets::toast::ToastStack::new(deck, now).render(body, buf);
        }

        // Last of all, over nothing (§6.18) — the structural half of the condition.
        if reserved > 0 {
            ConditionBand::with_condition(condition).render(
                Rect {
                    y: area.y + area.height - reserved,
                    height: reserved,
                    ..area
                },
                buf,
            );
        }
    }
}

#[cfg(any(test, feature = "tui-pantry"))]
pub(crate) mod frames {
    use super::*;
    use crate::widgets::config_table::{Edit, Entry, Focus, Row, UNSET};
    use std::time::Duration;
    use wqm_client::{DaemonState, DaemonStatus, IndexState, UnreachableReason};
    use wqm_proto::Address;

    /// How old a reading may get before §4 calls it stale. **Not a decision** — §7 leaves the
    /// freshness SLA open (OQ-6); this is the number the frames are drawn against so the two
    /// treatments can be compared, and it is stated here rather than buried in each one.
    pub const FRAME_SLA: Duration = Duration::from_secs(60);

    pub fn freshness() -> Freshness {
        Freshness::new(Duration::from_secs(4), FRAME_SLA)
    }

    /// The three backing stores. The daemon is deliberately not among them — see the module
    /// docs — and the list is illustrative rather than contract-bound (`CR-036`).
    pub fn stores(vector: Health) -> Vec<StoreRow> {
        vec![
            StoreRow::bound("vector", "qdrant", vector),
            StoreRow::bound("graph", "ladybug", Health::Healthy),
            StoreRow::bound("relational", "sqlite", Health::Healthy),
        ]
    }

    /// `CR-036`(a)'s arriving set: six components, one of them unreadable, none of which this
    /// build was compiled knowing about.
    ///
    /// It is the *screen's* arriving set, so — like [`stores`] and for the same reason (§6.29,
    /// the daemon is named once) — **the daemon is not in it**. `store_health`'s own `Arriving
    /// Set` frame keeps its daemon row, because that frame is the widget alone rather than a
    /// screen's backing federation. The two lists look alike and differ on purpose.
    ///
    /// The Dashboard's Storage cell needs it because its overflow row means nothing against a
    /// federation that fits.
    pub fn arriving_stores() -> Vec<StoreRow> {
        vec![
            StoreRow::bound("vector", "qdrant", Health::Healthy),
            StoreRow::bound("graph", "ladybug", Health::Healthy),
            StoreRow::bound("relational", "sqlite", Health::Healthy),
            // Named by the report with no backend behind it: the faint column stays empty
            // rather than being filled with an invented binding.
            StoreRow::unbound("queue_processor", Health::Degraded),
            StoreRow::unbound("embedding_provider", Health::Healthy),
            // A long name proves the glyph column is computed, not assumed.
            StoreRow::bound("language_registry", "bundled", Health::Healthy),
        ]
    }

    pub fn serving() -> DaemonReport {
        DaemonReport::Reachable(DaemonStatus {
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
        })
    }

    pub fn unreachable() -> DaemonReport {
        DaemonReport::Unreachable {
            reason: UnreachableReason::DaemonUnreachable,
            address: Address::Uds("/Users/chris/.wqm/memexd.sock".into()).to_string(),
        }
    }

    /// The reviewed frame's own keys, so the config zone of this screen and the `Config
    /// Table` widget frames are the same table.
    pub fn config_rows() -> Vec<Row> {
        vec![
            Row::Group("qdrant".into()),
            Row::Entry(Entry::new(
                "URL",
                "http://localhost:6333",
                "http://localhost:6333",
            )),
            Row::Entry(Entry::new("API key", UNSET, UNSET)),
            Row::Group("watcher".into()),
            Row::Entry(Entry::new("Debounce [ms]", "2000", "1500")),
            Row::Entry(Entry::new("Recursive", "on", "on")),
            Row::Group("embedding".into()),
            Row::Entry(Entry::new("model", "all-MiniLM-L6-v2", "all-MiniLM-L6-v2")),
            Row::Entry(Entry::new("batch size", "32", "32")),
        ]
    }

    /// The debounce key — the one the reviewed frame edits — as an entry index.
    pub const DEBOUNCE: usize = 2;

    /// The hub with the vector store in a stated state — the screen a Dashboard cell drills
    /// into, parameterised so a cell can be checked against *the same* screen rather than
    /// against a second construction of it.
    pub fn hub(vector: Health) -> ServiceView<'static> {
        ServiceView::new(
            stores(vector),
            serving(),
            ConfigTable::new(config_rows()),
            freshness(),
        )
    }

    /// The quiet screen: everything nominal, no zone focused, nothing floating above it.
    pub fn base() -> ServiceView<'static> {
        hub(Health::Healthy)
    }

    /// An edit-in-place open in the lower band, which is therefore the live zone.
    pub fn editing() -> ServiceView<'static> {
        ServiceView::new(
            stores(Health::Healthy),
            serving(),
            ConfigTable::new(config_rows()).focus(Focus::Editing(DEBOUNCE, Edit::insert("2000"))),
            freshness(),
        )
        .attention(Attention::Zone(ZONE_CONFIG))
    }

    /// A Tier-1 confirm over the screen (§4.4), with the cursor left where it was.
    pub fn confirming() -> ServiceView<'static> {
        ServiceView::new(
            stores(Health::Healthy),
            serving(),
            ConfigTable::new(config_rows()).focus(Focus::Cursor(DEBOUNCE)),
            freshness(),
        )
        .attention(Attention::Zone(ZONE_CONFIG))
        .modal(confirm_modal())
    }

    /// The confirm box [`confirming`] carries, on its own.
    ///
    /// Named rather than inlined because a guard has to know **what the modal covers** —
    /// [`Modal::rect`] answers that — and the page-is-colourless sweep is exactly the caller
    /// that comment on `rect` anticipated. Two constructions of one box would be two boxes
    /// whose rectangles agree until someone edits one of them.
    pub fn confirm_modal() -> Modal {
        Modal::new(
            "Discard changes?",
            "watcher.debounce_ms has been edited and not saved.",
        )
        .action("y", "discard")
        .action("n", "keep editing")
    }

    /// One store degraded: the rollup, the tab colour and the toast all follow from it.
    ///
    /// **The toast is produced by `health::transitions`, not written here.** The first cut
    /// carried a hand-written sentence — *"vector store degraded — qdrant slow past its
    /// SLA"* — which is both redundant (Chris, 20260731) and a message the system cannot
    /// actually emit: the producer says `role verb`, so the real string is *"vector
    /// degraded"*. A frame whose text was invented is a frame of a screen that does not
    /// exist, which is the same rule the Logs pane is missing for.
    pub fn degraded_deck() -> (ToastDeck, Instant) {
        let now = Instant::now();
        let before = SystemHealth::new(
            Health::Healthy,
            stores(Health::Healthy)
                .iter()
                .map(|row| Component::new(row.role.clone(), row.health))
                .collect(),
        );
        let after = SystemHealth::new(
            Health::Healthy,
            stores(Health::Degraded)
                .iter()
                .map(|row| Component::new(row.role.clone(), row.health))
                .collect(),
        );

        let mut deck = ToastDeck::new();
        for toast in crate::health::transitions(&before, &after) {
            deck.push(toast, now);
        }
        assert!(!deck.is_empty(), "the frame's own transition raised nothing");
        (deck, now)
    }

    pub fn degraded(deck: &ToastDeck, now: Instant) -> ServiceView<'_> {
        ServiceView::new(
            stores(Health::Degraded),
            serving(),
            ConfigTable::new(config_rows()),
            freshness(),
        )
        .toasts(deck, now)
    }

    /// The liveness master is not answering: the wash, the band, and readings nobody should
    /// trust. The freshness is past its SLA because that is what an unreachable daemon means.
    pub fn unreachable_view() -> ServiceView<'static> {
        ServiceView::new(
            stores(Health::Healthy),
            unreachable(),
            ConfigTable::new(config_rows()),
            Freshness::new(Duration::from_secs(184), FRAME_SLA),
        )
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "stores",
            ty: "Vec<StoreRow>",
            description: "The backing stores. The daemon is NOT among them — it has the panel",
        },
        PropInfo {
            name: "daemon",
            ty: "DaemonReport",
            description: "N49's answer. Drives the panel, the rollup, the tab and the wash",
        },
        PropInfo {
            name: "config",
            ty: "ConfigTable",
            description: "The lower band. Its focus decides the status line's mode indicator",
        },
        PropInfo {
            name: "attention",
            ty: "Attention",
            description: "Which zone is live — screen-level, so two cannot be",
        },
    ];

    struct Variant(&'static str, &'static str, fn(&mut Buffer, Rect));

    impl Ingredient for Variant {
        fn tab(&self) -> &str {
            "Views"
        }
        fn group(&self) -> &str {
            "Service"
        }
        fn name(&self) -> &str {
            self.0
        }
        fn source(&self) -> &str {
            "wqm_tui::views::service"
        }
        fn description(&self) -> &str {
            self.1
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            (self.2)(buf, area);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Variant(
                "Default",
                "The quiet screen — this is the frame r02's flatness complaint is judged against",
                |buf, area| frames::base().render(area, buf),
            )),
            Box::new(Variant(
                "Editing",
                "An edit open in the lower band: the third highlight role, and the zone accent beside the pane selector",
                |buf, area| frames::editing().render(area, buf),
            )),
            Box::new(Variant(
                "Confirm",
                "A Tier-1 confirm centred over the screen — the layer stack against a live layer 0",
                |buf, area| frames::confirming().render(area, buf),
            )),
            Box::new(Variant(
                "Degraded + toast",
                "One store degraded: rollup, tab recolour and toast all derived from one health value",
                |buf, area| {
                    let (deck, now) = frames::degraded_deck();
                    frames::degraded(&deck, now).render(area, buf);
                },
            )),
            Box::new(Variant(
                "Daemon unreachable",
                "The condition: wash under everything, band under that, and no toast pretending it is news",
                |buf, area| frames::unreachable_view().render(area, buf),
            )),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::Encoding;
    use crate::terminal::{Endpoints, Rgb};
    use crate::tokens::{self, Palette};
    use crate::widgets::surface::UNREACHABLE_BAND;
    use ratatui::style::Color;

    struct Restore(Palette, Encoding, Endpoints);

    impl Restore {
        fn dark_truecolor() -> Self {
            let restore = Restore(Palette::current(), Encoding::current(), tokens::endpoints());
            Palette::set(Palette::Derived);
            Encoding::set(Encoding::TrueColor);
            tokens::set_endpoints(Endpoints {
                background: Rgb::new(0x1e, 0x1e, 0x2e),
                foreground: Rgb::new(0xcd, 0xd6, 0xf4),
            });
            restore
        }
    }

    impl Drop for Restore {
        fn drop(&mut self) {
            Palette::set(self.0);
            Encoding::set(self.1);
            tokens::set_endpoints(self.2);
        }
    }

    /// The storyboard's own geometry (§6): a half-screen terminal.
    const AREA: Rect = Rect {
        x: 0,
        y: 0,
        width: 125,
        height: 34,
    };

    fn render(view: ServiceView<'_>) -> Buffer {
        let mut buf = Buffer::empty(AREA);
        view.render(AREA, &mut buf);
        buf
    }

    fn row(buf: &Buffer, y: u16) -> String {
        (0..AREA.width)
            .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
            .collect()
    }

    fn rows_of(buf: &Buffer) -> Vec<String> {
        (0..AREA.height).map(|y| row(buf, y)).collect()
    }

    fn find(buf: &Buffer, needle: &str) -> Option<(u16, u16)> {
        rows_of(buf).iter().enumerate().find_map(|(y, line)| {
            line.find(needle)
                .map(|x| (line[..x].chars().count() as u16, y as u16))
        })
    }

    #[test]
    fn a_bundled_screen_paints_every_cell_it_owns() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();
        let previous_theme = tokens::theme();
        tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());
        Palette::set(Palette::Bundled);

        // §15's full paint, kept as a guard rather than as a remembered measurement. The
        // probe that established it painted layer 0 with the theme's SECOND anchor so that an
        // unpainted cell would show as a hole to the eye; the eye is not here every time this
        // is built, and the claim it settled — the paint reaches every cell — is the one that
        // has to keep being true once the colour is the theme's actual background.
        //
        // "Every cell has A background", not "every cell has THIS background": the cursor
        // tint, the edit fill and a modal's layers are all deliberate departures from the
        // base. What must not exist is a cell left at the terminal's own default, because
        // that is the one colour the theme does not own.
        let buf = render(frames::base());
        let unpainted: Vec<(u16, u16)> = (0..AREA.height)
            .flat_map(|y| (0..AREA.width).map(move |x| (x, y)))
            .filter(|(x, y)| {
                buf.cell((*x, *y))
                    .expect("cell in area")
                    .style()
                    .bg
                    .is_none_or(|bg| bg == Color::Reset)
            })
            .collect();
        assert!(
            unpainted.is_empty(),
            "{} of {} cells kept the terminal's background, first at {:?}",
            unpainted.len(),
            AREA.width as usize * AREA.height as usize,
            unpainted.first()
        );

        if let Some(theme) = previous_theme {
            tokens::set_theme(theme);
        }
    }

    #[test]
    fn the_zones_share_one_left_margin_across_three_widgets() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // Three widgets that have never seen each other's arithmetic: the tab bar is inset by
        // the screen, the store list is inset by the screen, and the config table reaches the
        // same column through its OWN margin. If any of the three moves, the grid §3 relies
        // on to say "this cell, not that one" stops being a grid.
        let buf = render(frames::base());
        let (tab, _) = find(&buf, "1 Dashboard").expect("the tab row");
        let (store, _) = find(&buf, "vector").expect("the store list");
        let (key, _) = find(&buf, "KEY").expect("the config header");
        assert_eq!(
            (tab, store, key),
            {
                let m = crate::widgets::chrome::MARGIN;
                (m, m, m)
            },
            "tabs, stores and the config table start in one column"
        );
    }

    #[test]
    fn the_rollup_the_tab_colour_and_the_wash_all_answer_to_one_health() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // Healthy: one dot, no alarm hue on the tab, no wash.
        let healthy = frames::base();
        assert_eq!(healthy.rollup().health, Health::Healthy);
        assert_eq!(healthy.condition(), Condition::Nominal);

        // Degrade exactly one store and nothing else. Every one of the three moves, because
        // there is only one place for them to read.
        let (deck, now) = frames::degraded_deck();
        let degraded = frames::degraded(&deck, now);
        let rollup = degraded.rollup();
        assert_eq!(rollup.health, Health::Degraded);
        assert_eq!(rollup.label, "1 degraded", "§7's `▲ 1 degraded`");
        assert_eq!(
            degraded.condition(),
            Condition::Nominal,
            "a store is not the master"
        );

        let buf = render(degraded);
        let (x, y) = find(&buf, "Service").expect("the Service tab");
        assert_eq!(
            buf.cell((x, y)).expect("cell in area").style().bg,
            Some(Health::Degraded.color()),
            "§4's must-see rule: the owning tab keeps its alarm colour UNDER inversion"
        );
        assert!(
            rows_of(&buf)
                .last()
                .expect("a bottom row")
                .contains("1 degraded"),
            "and the same count is what the status line says"
        );
    }

    #[test]
    fn an_unreachable_daemon_washes_the_screen_and_rolls_up_as_itself() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let view = frames::unreachable_view();
        assert_eq!(view.condition(), Condition::DaemonUnreachable);
        // §6.19: the master silences its subordinates. Three healthy stores are on this
        // screen and the rollup counts none of them, because their readings are unknown.
        assert_eq!(view.rollup().label, "daemon unreachable");

        let buf = render(view);
        let bottom = AREA.height - 1;
        assert!(
            row(&buf, bottom).contains(UNREACHABLE_BAND),
            "the structural half of the condition owns the bottom row"
        );

        // The wash is under EVERY zone, not one of them — including the row the config table
        // draws into, which is the zone that would otherwise look live.
        let wash = tokens::wash(Condition::DaemonUnreachable).expect("truecolor washes");
        let (_, key_y) = find(&buf, "KEY").expect("the config header");
        for y in [1, key_y, bottom - 1] {
            assert_eq!(
                buf.cell((AREA.width - 1, y))
                    .expect("cell in area")
                    .style()
                    .bg,
                Some(wash),
                "row {y} is not washed"
            );
        }
    }

    #[test]
    fn the_condition_band_survives_the_zones_that_are_drawn_over_it() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // The regression this pins is a real one: drawing the band before the zones put the
        // store rows on top of the words. The band is only proof of ordering if a zone would
        // otherwise have reached that row, so the screen is rendered at its full height and
        // the row below the band is checked to be occupied.
        let buf = render(frames::unreachable_view());
        let bottom = AREA.height - 1;
        assert!(row(&buf, bottom)
            .trim_start()
            .starts_with(Health::Offline.glyph()));
        assert!(
            row(&buf, bottom - 1).contains("healthy")
                || row(&buf, bottom - 1).contains("unreachable"),
            "the status line is directly above the band, so the band displaced nothing"
        );
    }

    #[test]
    fn an_open_edit_reaches_the_status_line_without_being_told_twice() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // The indicator is not a field on the screen: it is read off the table that owns the
        // edit, so a frame with a caret and no indicator — or the reverse — is unbuildable.
        let quiet = render(frames::base());
        assert!(!rows_of(&quiet).concat().contains("-- INSERT --"));

        let buf = render(frames::editing());
        let bottom = AREA.height - 1;
        assert!(
            row(&buf, bottom).trim_start().starts_with("-- INSERT --"),
            "{:?}",
            row(&buf, bottom)
        );
        assert!(
            rows_of(&buf).concat().contains("Esc abandon"),
            "and the hints follow the mode, not the default row"
        );
    }

    mod under_modal;

    #[test]
    fn the_modal_covers_the_zones_and_the_toast_covers_the_corner() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // §6 tells a modal and a toast apart by POSITION. On a full screen that is testable
        // in a way an isolated preview cell cannot be: the modal must land over the zones,
        // and the toast must land where nothing else on this screen draws.
        let confirm = render(frames::confirming());
        let (_, modal_y) = find(&confirm, "Discard changes?").expect("the modal");
        assert!(
            (AREA.height / 3..2 * AREA.height / 3).contains(&modal_y),
            "a modal is centred over the screen, not parked in a corner: row {modal_y}"
        );

        let (deck, now) = frames::degraded_deck();
        let toasted = render(frames::degraded(&deck, now));
        let (toast_x, toast_y) = find(&toasted, "vector degraded").expect("the toast");
        assert!(
            toast_x > AREA.width / 2 && toast_y > AREA.height / 2,
            "a toast is welded to the lower-right corner: ({toast_x}, {toast_y})"
        );

        // And it is over the config table rather than beside it — the toast sits outside the
        // layer stack and is painted last.
        let quiet = render(frames::base());
        assert_ne!(
            quiet
                .cell((toast_x, toast_y))
                .expect("cell in area")
                .symbol(),
            toasted
                .cell((toast_x, toast_y))
                .expect("cell in area")
                .symbol(),
            "the toast has to be covering something for `painted last` to mean anything"
        );
    }

    #[test]
    fn a_floating_box_occludes_the_screen_it_floats_over() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // A background fill is not occlusion: ratatui restyles the cells a block covers and
        // leaves their symbols standing. On an empty preview buffer that is indistinguishable
        // from an opaque box, which is why both the modal and the toast shipped with the
        // screen's text able to run straight through them.
        let base = render(frames::base());
        let confirm = render(frames::confirming());

        // The title rides the top border, so the box's left edge is two columns before it,
        // and the blank row between the body and the actions is two rows down.
        let (title_x, title_y) = find(&confirm, "Discard changes?").expect("the modal");
        let left = title_x - 2;
        let gap = title_y + 2;
        let span = |buf: &Buffer, y: u16| -> String {
            (left + 2..left + 50)
                .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
                .collect()
        };

        // The relation has to OCCUR before it can be asserted: if nothing was behind the
        // modal's blank row, an opaque box and a transparent one look identical there.
        assert!(
            !span(&base, gap).trim().is_empty(),
            "this row must carry screen text for occlusion to mean anything: {:?}",
            span(&base, gap)
        );
        assert!(
            span(&confirm, gap).trim().is_empty(),
            "the modal's blank row is blank: {:?}",
            span(&confirm, gap)
        );
    }

    #[test]
    fn most_of_the_quiet_screen_is_below_the_normal_rung() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // §2's default posture — "most of the screen is muted or faint … strong is the few
        // things that matter" — stated as a measurement. This is the flatness complaint made
        // checkable: a screen where everything shouts has no accent, one where nothing does
        // has no anchor.
        //
        // # The rules are excluded, and the first version of this test was wrong for
        // including them
        //
        // Three full-width rules are 375 cells on a 125×34 screen — more than the text. They
        // are structure carrying their own greys (§2), not content, so counting them measured
        // the frame's *geometry* and reported a posture the screen does not have. Measured
        // over text the numbers are 100 faint / 170 muted / 141 normal / 4 strong.
        let buf = render(frames::base());
        let mut recessive = 0usize;
        let mut strong = 0usize;
        let mut ink = 0usize;
        let recessive_fg = [tokens::muted(), tokens::faint()];
        for y in 0..AREA.height {
            for x in 0..AREA.width {
                let cell = buf.cell((x, y)).expect("cell in area");
                if cell.symbol() == " " || cell.symbol() == "─" {
                    continue;
                }
                ink += 1;
                if cell.style().fg.is_some_and(|c| recessive_fg.contains(&c)) {
                    recessive += 1;
                }
                if cell.style().fg == Some(tokens::strong()) {
                    strong += 1;
                }
            }
        }
        assert!(
            ink > 200,
            "the frame has to be a screen, not a stub: {ink} text cells"
        );
        assert!(
            recessive * 2 > ink,
            "most of the screen recedes: {recessive} of {ink} text cells"
        );
        assert!(
            strong * 20 < ink,
            "strong is the few things that matter, not a rung the screen leans on: \
             {strong} of {ink}"
        );
    }

    #[test]
    fn cyan_is_the_selector_and_appears_nowhere_else() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // The reserved-colour promise is a claim about a WHOLE screen; two widgets each
        // using cyan correctly is exactly how it gets broken. Here the tab row and the pane
        // selector both use it, and nothing else may.
        let buf = render(frames::base());
        let selector = tokens::selector();
        let mut blocks: Vec<(u16, u16)> = Vec::new();
        for y in 0..AREA.height {
            for x in 0..AREA.width {
                let cell = buf.cell((x, y)).expect("cell in area");
                if cell.style().bg == Some(selector) || cell.style().fg == Some(selector) {
                    blocks.push((x, y));
                }
            }
        }
        let rows_used: Vec<u16> = {
            let mut ys: Vec<u16> = blocks.iter().map(|(_, y)| *y).collect();
            ys.dedup();
            ys.sort_unstable();
            ys.dedup();
            ys
        };
        let (_, tab_row) = find(&buf, "1 Dashboard").expect("the tab row");
        let (_, pane_row) = find(&buf, "Config").expect("the pane selector");
        assert_eq!(
            rows_used,
            vec![tab_row, pane_row],
            "cyan appears on the tab row and the pane selector, and nowhere else"
        );
    }

    #[test]
    fn a_screen_too_short_to_hold_its_zones_draws_nothing_rather_than_a_ruin() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let mut buf = Buffer::empty(Rect {
            x: 0,
            y: 0,
            width: 125,
            height: 6,
        });
        frames::base().render(buf.area, &mut buf);
        let painted = (0..6)
            .flat_map(|y| (0..125u16).map(move |x| (x, y)))
            .filter(|(x, y)| {
                let cell = buf.cell((*x, *y)).expect("cell in area");
                cell.symbol() != " " || cell.style().bg != Some(Color::Reset)
            })
            .count();
        assert_eq!(
            painted, 0,
            "half a screen is a rendering artefact, not a screen"
        );
    }
}
