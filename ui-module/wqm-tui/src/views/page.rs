//! The main screen's frame — **PageTop**, the view's region, and **PageFoot**. The page-side
//! twin of [`crate::widgets::modal_frame`].
//!
//! Chris, 2026-09-13 20:30: *"each such window is made from three components: a modal
//! container, a decoration (the top 4-lines and the bottom 3 lines), and a view. These are
//! composable … every table or tabular view is reusable, as well as decoration"*. The main
//! screen is that same composition **with no container**: it has no border and no footprint
//! because it *is* the terminal, and what is left is a decoration above, a decoration below,
//! and a view between them.
//!
//! Chris, 2026-09-14 19:05, on the shape those two decorations take here: *"transform the main
//! window into a selector on top, a frame with the service status (it should be optional, since
//! we won't need it in the Service tab) and its hairline, and the bottom with the hairline and
//! the minimalist help line"*.
//!
//! ```text
//!   WQM TUI   1 Dashboard  2 Queue  …  10 Service     ← PageTop row 0, the selector
//! ───────────────────────────────────────────────────  ← row 1, the frame rule
//!   ● Service status  v0.2.0        updated 4s ago     ← the OPTIONAL status frame …
//!   ● daemon      ● vector db   ● graph db  …
//!   ● queue    11'236 pending   4 in progress  …
//! ───────────────────────────────────────────────────  ← … and its hairline
//!   … the VIEW's region, and nothing in here is the
//!     page's business                                  ← `draw` hands this back
//! ───────────────────────────────────────────────────  ← PageFoot, the closing hairline
//!                            ? Help   q Quit           ← and the minimalist help line
//! ```
//!
//! # Why one object rather than four assemblies
//!
//! Chris, 20260906: the first rows of every tab are the same rows. That claim only stays true
//! while there is one thing drawing them, and before this module there were four: the Shell,
//! the Dashboard and the Queue each called a shared top and then built their own foot, and the
//! Service hub laid every row of the screen out with a [`Layout`](ratatui::layout::Layout) of
//! its own — including a tab row that had quietly lost the product title the other three
//! carry. A copy is how *the same on every tab* becomes *the same on the tabs someone
//! remembered to update*, and the Service tab is the proof: nobody noticed, because nothing
//! could. There was no place the claim was written down as code.
//!
//! # `draw` returns the region rather than taking the view
//!
//! [`Page::draw`] paints the chrome and hands back the [`Rect`] between the two decorations.
//! The view then owns everything inside it, which is §16's split: the chrome decides where the
//! chrome goes, the view decides everything else. A callback would have inverted that and made
//! the frame the thing that lays out a screen.
//!
//! **The foot is painted by `draw`, before the view runs.** Both foot rows are fixed — they sit
//! at the bottom of the area whatever the view does — so painting them early costs nothing and
//! buys the one property worth having: a screen cannot forget them. The region the view gets
//! back already excludes them, so nothing is drawn twice.
//!
//! # Not the modal's [`Decoration`](crate::widgets::modal_frame::Decoration), and deliberately
//!
//! Same ROLE, different content. A window's decoration carries a breadcrumb, a title, an
//! optional search row and two help rows; a page's carries a tab selector, a service-status
//! frame and one help row. Forcing one struct to serve both would mean a struct that is mostly
//! `None` from either side. What they share is the shape — decoration, view, decoration — and
//! that shape is legible by reading the two modules beside each other, which is what it is for.

use ratatui::{buffer::Buffer, layout::Rect, widgets::Widget};

use crate::panes::status_block::{self, StatusBlock};
use crate::tokens::Condition;
use crate::widgets::chrome::{inset, AppBar, Rule, StatusLine};
use crate::widgets::config_table::EditMode;
use crate::widgets::surface::Surface;
use crate::widgets::tab_bar::Tab;

/// The two rows every screen carries above whatever comes next: the selector and the frame
/// rule under it. Constant because the Service tab has these two and nothing else.
pub const APP_BAR_ROW: u16 = 0;
pub const TOP_RULE_ROW: u16 = 1;
pub const CONSTANT_ROWS: u16 = 2;

/// Rows the foot costs a page: the hairline that closes the content, and the help line itself.
///
/// A view's own minimum-height arithmetic adds this rather than adding 1, so the day the foot
/// grows a third row there is one number to change instead of one per screen.
pub const FOOT_ROWS: u16 = 2;

/// One row of an area, by index.
pub fn row(area: Rect, n: u16) -> Rect {
    Rect {
        y: area.y + n,
        height: 1,
        ..area
    }
}

/// The top decoration: the selector, the frame rule, and the service-status frame a tab carries
/// — or does not.
pub struct PageTop {
    /// A stated tab row, for the one screen that recolours a tab it does not own. `None` is
    /// the storyboard's ten.
    tabs: Option<Vec<Tab>>,
    active: usize,
    /// Absent on the Service tab, which carries [`crate::panes::status_band`] instead — the
    /// band says the same thing in more detail, and two claims about one system on one screen
    /// is how the two come to disagree.
    status: Option<StatusBlock>,
    content_floor: u16,
}

impl PageTop {
    fn new(active: usize) -> Self {
        Self {
            tabs: None,
            active,
            status: None,
            content_floor: status_block::MIN_CONTENT_ROWS,
        }
    }

    /// Rows this top takes out of `area` — the two constant ones, plus whatever the status
    /// frame claims at that size.
    ///
    /// Takes the whole page rather than a row count because the frame's height is not a
    /// property of the frame: it collapses when the screen is short, and what it collapses
    /// against is the view's own floor. The mirror of
    /// [`Decoration::top_rows`](crate::widgets::modal_frame::Decoration::top_rows), which needs
    /// no argument only because a window's fifth row is declared rather than measured.
    pub fn rows(&self, area: Rect) -> u16 {
        match &self.status {
            Some(_) => CONSTANT_ROWS + StatusBlock::rows_for(Self::body(area), self.content_floor),
            None => CONSTANT_ROWS,
        }
    }

    /// What the status frame and the view share: everything below the frame rule.
    fn body(area: Rect) -> Rect {
        Rect {
            y: area.y + CONSTANT_ROWS,
            height: area.height.saturating_sub(CONSTANT_ROWS),
            ..area
        }
    }

    /// Paint, and hand back what is left below.
    fn draw(self, area: Rect, buf: &mut Buffer) -> Rect {
        match self.tabs {
            Some(tabs) => AppBar::with_tabs(tabs, self.active),
            None => AppBar::new(self.active),
        }
        .render(inset(row(area, APP_BAR_ROW)), buf);
        Rule::frame().render(row(area, TOP_RULE_ROW), buf);

        let body = Self::body(area);
        let taken = match self.status {
            Some(block) => {
                let rows = StatusBlock::rows_for(body, self.content_floor);
                block.render(
                    Rect {
                        height: rows,
                        ..body
                    },
                    buf,
                );
                rows
            }
            None => 0,
        };

        Rect {
            y: body.y + taken,
            height: body.height - taken,
            ..body
        }
    }
}

/// The bottom decoration: the hairline that closes the content, and the minimalist help line.
///
/// Chris, 2026-09-07: *"we should have a line just above the bottom line, valid for all views
/// as well."* Every screen ends the same way, so the rule is not something a view remembers to
/// draw — it is drawn by the frame that tells the view how much room it has. It runs edge to
/// edge, [`crate::widgets::chrome::MARGIN`] included, exactly as the frame rule above does: it
/// divides the screen rather than the content inside it. Only the Dashboard's row rules break
/// over its column gap, and they break because the GRID has two columns — this one has nothing
/// to break over.
///
/// A fifth screen asking for a foot gets both rows whether or not anybody thought about it,
/// which is the half that used to be missing: the rule was drawn by the call that carved the
/// room, so a screen that skipped the call had no foot at all. That is exactly what had
/// happened to the bare page.
///
/// The line itself carries no hue at all — [`StatusLine`] holds that, along with the rule that
/// `? Help · q Quit` survive a foot too narrow for the rest.
#[derive(Default)]
pub struct PageFoot {
    hints: Vec<(String, String)>,
    mode: Option<EditMode>,
}

impl PageFoot {
    /// Rows the foot always costs — the twin of
    /// [`BOTTOM_ROWS`](crate::widgets::modal_frame::BOTTOM_ROWS), which is 3 because a window
    /// affords two help rows and a page affords one.
    pub const fn rows(&self) -> u16 {
        FOOT_ROWS
    }

    /// Carve the foot off the bottom of `body`, draw both of its rows, and hand back the
    /// region above them.
    ///
    /// Returns the whole of `body` and draws nothing when `body` cannot hold both rows, so a
    /// caller that forgets to check still draws no rule rather than drawing one over its own
    /// last line.
    fn draw(self, body: Rect, buf: &mut Buffer) -> Rect {
        if body.height < FOOT_ROWS {
            return body;
        }
        Rule::internal().render(row(body, body.height - FOOT_ROWS), buf);

        let mut line = StatusLine::new().mode(self.mode);
        for (key, label) in self.hints {
            line = line.hint(key, label);
        }
        line.render(inset(row(body, body.height - 1)), buf);

        Rect {
            height: body.height - FOOT_ROWS,
            ..body
        }
    }
}

/// The main screen's frame: ground, top decoration, view region, bottom decoration.
pub struct Page {
    top: PageTop,
    foot: PageFoot,
    condition: Condition,
}

impl Page {
    /// A tab with no status frame. [`Page::status`] adds one.
    pub fn new(active: usize) -> Self {
        Self {
            top: PageTop::new(active),
            foot: PageFoot::default(),
            condition: Condition::Nominal,
        }
    }

    /// A stated tab row — how the one screen that recolours a tab says so.
    pub fn tabs(mut self, tabs: Vec<Tab>) -> Self {
        self.top.tabs = Some(tabs);
        self
    }

    pub fn status(mut self, block: StatusBlock) -> Self {
        self.top.status = Some(block);
        self
    }

    /// Rows the VIEW's region must keep before the status frame gives way. The **view's**
    /// number: a tab holding a six-cell grid can afford less top furniture than one holding a
    /// one-line summary.
    pub fn content_floor(mut self, rows: u16) -> Self {
        self.top.content_floor = rows;
        self
    }

    /// Offer a key on the help line, in the order the view names them.
    pub fn hint(mut self, key: impl Into<String>, label: impl Into<String>) -> Self {
        self.foot.hints.push((key.into(), label.into()));
        self
    }

    /// The vim mode an edit-in-place is in, if one is open. Taken from the table that owns the
    /// edit rather than restated, so the caret and the indicator cannot disagree.
    pub fn mode(mut self, mode: Option<EditMode>) -> Self {
        self.foot.mode = mode;
        self
    }

    /// The screen-wide condition the ground is painted under. Nominal paints the theme's own
    /// background and nothing else.
    pub fn condition(mut self, condition: Condition) -> Self {
        self.condition = condition;
        self
    }

    /// Rows the chrome costs before a view has anywhere to draw: both decorations.
    pub const CHROME_ROWS: u16 = CONSTANT_ROWS + FOOT_ROWS;

    /// Paint the page's ground and both decorations, and hand back the view's region.
    ///
    /// Returns an empty [`Rect`] — and paints nothing at all — when the area cannot hold the
    /// chrome and one row of view, so a caller that forgets to check draws nothing rather than
    /// drawing into negative space. Half a screen is a rendering artefact, not a screen.
    pub fn draw(self, area: Rect, buf: &mut Buffer) -> Rect {
        if area.height < Self::CHROME_ROWS + 1 {
            return Rect { height: 0, ..area };
        }
        // §15: the theme owns the background, and every full screen paints it first. Stated
        // rather than read from the process, because a frame is a still.
        Surface::with_condition(self.condition).render(area, buf);

        let below = self.top.draw(area, buf);
        self.foot.draw(below, buf)
    }
}

/// The `Page` group — the frame, browsable, and the one frame that puts it beside the
/// window's.
///
/// **This is the old `Shell` group, renamed and extended.** Its variants were always frames of
/// the page's chrome over an empty body; they were filed under the view that happened to draw
/// them, and now they are filed under the thing they are of. Two groups saying the same thing
/// is how a reader ends up comparing a frame with its own copy.
///
/// The frames themselves come from [`crate::views::shell`], which is `Page` with a placeholder
/// where a view goes — the cheapest body there is, and therefore the one that shows the frame
/// rather than competing with it.
#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use crate::views::shell::{frames, ShellView};
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "PageTop",
            ty: "views::page::PageTop",
            description: "The selector, the frame rule, and the OPTIONAL service-status frame",
        },
        PropInfo {
            name: "region",
            ty: "Rect",
            description: "What `draw` returns — the view's, and nothing the page knows about",
        },
        PropInfo {
            name: "PageFoot",
            ty: "views::page::PageFoot",
            description: "The closing hairline and the minimalist help line. No hue, ever",
        },
        PropInfo {
            name: "status",
            ty: "Option<StatusBlock>",
            description: "Absent on Service, which carries the status band instead",
        },
        PropInfo {
            name: "content_floor",
            ty: "u16",
            description: "Rows the region keeps before the status frame collapses — the VIEW's",
        },
    ];

    /// What a variant puts in the preview cell.
    ///
    /// Two shapes rather than one, because the composed frame is not a `ShellView`: it is a
    /// whole screen with a window over it, and flattening the two into one function pointer
    /// would mean every bare frame carrying a closure it does not need.
    enum Draw {
        /// The frame over a placeholder body.
        Bare(fn() -> ShellView),
        /// A whole composition drawn into the cell — used by `Page + window`.
        Composed(fn(Rect, &mut Buffer)),
    }

    /// A frame, and the size it is drawn at. `None` fills the preview cell, which is how the
    /// storyboard's 125 × 34 is judged against the terminal actually running the pantry.
    struct Variant(&'static str, &'static str, Draw, Option<(u16, u16)>);

    impl Ingredient for Variant {
        fn tab(&self) -> &str {
            "Views"
        }
        fn group(&self) -> &str {
            "Page"
        }
        fn name(&self) -> &str {
            self.0
        }
        fn source(&self) -> &str {
            "wqm_tui::views::page"
        }
        fn description(&self) -> &str {
            self.1
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let (width, height) = self.3.unwrap_or((area.width, area.height));
            let at = Rect {
                width: width.min(area.width),
                height: height.min(area.height),
                ..area
            };
            match self.2 {
                Draw::Bare(frame) => frame().render(at, buf),
                Draw::Composed(draw) => draw(at, buf),
            }
        }
    }

    /// The Queue page with a framework window over it — the composition ruling's own claim,
    /// rendered.
    ///
    /// Chris, 2026-09-13 20:30: *"these are composable … every table or tabular view is
    /// reusable, as well as decoration"*. This frame is where that stops being a sentence. The
    /// page under the window is a real [`crate::views::queue::Queue`] drawn by `Page`; the
    /// window over it is [`crate::widgets::modal_frame::Container`] and its own `Decoration`;
    /// and the table INSIDE the window is the Queue's own
    /// [`ListPane`](crate::panes::list::ListPane), pre-filtered. One view kind, two frames,
    /// and the only difference is what is wrapped around it.
    ///
    /// Drawn under round 1's tint proposal, like every frame in the `Modal Framework` group,
    /// so the two groups show the same window rather than two tunings of it.
    fn page_and_window(area: Rect, buf: &mut Buffer) {
        use crate::views::modal_framework::frames as mf;
        mf::with_tint(crate::tokens::ModalTint::Accent, mf::PROPOSED_WASH, || {
            mf::table_frame(area, buf, false);
        });
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Variant(
                "Dashboard, healthy",
                "The frame at rest — six rows that say the system is fine, then a region, then two rows that say what the keys do",
                Draw::Bare(frames::dashboard),
                None,
            )),
            Box::new(Variant(
                "Queue, degraded, backlog",
                "Tab 2 selected, one part degraded, work piling up: does the top pull harder than the empty region?",
                Draw::Bare(frames::queue_degraded),
                None,
            )),
            Box::new(Variant(
                "Service tab (no status frame)",
                "Tab 10, and the one tab with no status frame — its own band says the same thing better, so the region starts under the frame rule",
                Draw::Bare(frames::service_tab),
                None,
            )),
            Box::new(Variant(
                "Under modal",
                "The page beneath a modal: no accent, no inverse block, nothing live, foot included. The modal itself is not drawn",
                Draw::Bare(|| frames::dashboard().under_modal(true)),
                None,
            )),
            Box::new(Variant(
                "Small 80x20",
                "Eighty by twenty: the tab row runs off the right, the status frame keeps only its roll-up, and the foot keeps the two hints that always survive",
                Draw::Bare(frames::dashboard),
                Some((80, 20)),
            )),
            Box::new(Variant(
                "Wide 200x40",
                "Two hundred by forty: the tab row has all the room it wants, and the region below it deliberately does not take all of its own",
                Draw::Bare(frames::queue_degraded),
                Some((200, 40)),
            )),
            Box::new(Variant(
                "Page + window",
                "The ruling, rendered: a Page beneath, a Container and Decoration above, and the Queue's own table inside the window. Same view kind, two frames",
                Draw::Composed(page_and_window),
                None,
            )),
        ]
    }
}
