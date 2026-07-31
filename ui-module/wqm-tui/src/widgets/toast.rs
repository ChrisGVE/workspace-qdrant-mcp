//! Toasts — a small ephemeral rectangle in the lower-right corner (Chris, 20260731).
//!
//! # What a toast is, and why it may be a box
//!
//! VISUAL-LANGUAGE.md §6 says boxes appear ONLY for modals, and every toast implementation
//! measured in `PASS2-EVAL.md` §F draws a floating bordered box — which is why *"is a toast
//! a modal?"* was an open question rather than an implementation detail. Chris answered it
//! by specifying the surface: **a small rectangle in the lower-right corner, ephemeral but
//! on screen long enough to read, with an audible and configurable sound.**
//!
//! So a box now means one of two things, and the two are told apart by position and
//! lifetime rather than by their border:
//!
//! | | modal | toast |
//! |---|---|---|
//! | position | centred | anchored to the lower-right corner |
//! | focus | takes it; input is blocked behind it | never takes it; input carries on |
//! | lifetime | dismissed by the user | expires on its own |
//! | layer | the §6 stack (layer 1, then layer 2) | outside the stack — always painted last |
//!
//! # Two things toast, and nothing else (Chris, 20260731)
//!
//! *"I would not have too many of those. What matters are: errors and recovery — i.e. the
//! system is degraded, or off, and the system returns to green. Providing too many alerts
//! will reduce their impact, make them annoying, and the important ones will be missed."*
//!
//! So a toast reports a **settled change of system state**, never a fact about a request:
//! falling out of green, and returning to it. That is why [`Toast::transition`] is the only
//! constructor and why it takes *both* ends of the change — a state that did not change
//! yields [`None`], so a repeated report cannot become an alert. Volume discipline is a type
//! here rather than a habit.
//!
//! What deliberately does **not** toast, and where it goes instead: an N12 notice
//! ([`crate::widgets::envelope::Notices`], the status zone) and a `ToolError` from an action
//! the user just took ([`crate::widgets::envelope::ErrorPanel`], where it stays readable
//! instead of expiring). Neither is a change of system state, and both would spend the
//! corner's attention on things the user is already looking at.
//!
//! # The widget never reads the clock
//!
//! Every entry point that depends on time takes `now` as a parameter. This is the one rule
//! three unrelated crate families arrived at independently (`PASS2-EVAL.md` §F, §ST), and it
//! is what makes a frame reproducible: `cargo pantry dump` and the PNG capture render a
//! *stated* moment in a toast's life rather than whatever the wall clock happens to say.
//!
//! # This deck collapses repetition, NOT flapping
//!
//! [`ToastDeck::push`] merges a toast into the live entry above it when both the state
//! arrived at and the message match — so eight reports of the *same* transition are one
//! toast with a `×8`. It does **not** coalesce `degraded → healthy → degraded → healthy`,
//! which is four real transitions and therefore yields
//! four toasts. That is deliberate and it is where `HEALTH-MONITORING.md` property 3 puts the
//! fix: a settled-change detector belongs upstream, daemon-side, under `CR-035`. `hjkl-holler`
//! was measured making exactly this distinction look like debouncing when it is not
//! (`PASS2-EVAL.md` §F), so the limit is pinned by a test rather than left to a comment.
//!
//! # The sound is emitted by the host, never by this crate
//!
//! A widget crate does no I/O, and the sound is configurable — which per Chris means the
//! field is owned by the **N7 config nexus** and read from the config SSOT, not invented
//! here. [`ToastDeck::push`] therefore *reports* a [`SoundEvent`] and the host decides what
//! to play. The knobs are requested from the corpus session in `TO-CORPUS.md` (`UIQ-006`);
//! until they land, [`DWELL_FLOOR`], [`DWELL_PER_CHAR`] and [`DWELL_CEILING`] are this
//! module's provisional defaults and are named so the request and the code use one
//! vocabulary.
//!
//! One half of that request came back the same day as `CR-052`, and it constrains what this
//! module may later do with the answer: **there is exactly one `Config` type across all four
//! surfaces (N7, anchor `A-config`), and a per-surface config struct must not exist.** So a
//! `tui.*` toast grouping is a *section* of the shared config, never a struct owned here —
//! which is also why this module holds bare constants rather than a `ToastConfig`. The field
//! set itself stays deferred: `CR-052` found that it was routed to a phase with no engagement
//! to hold it, and this request is what exposed that.

use std::time::{Duration, Instant};

use crate::tokens::{self, Health};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Block, Padding, Paragraph, Widget},
};

/// Which sound the host plays. **Two**, because there are two things worth interrupting
/// someone for: something broke, and it is fixed.
///
/// A larger set was drafted (info / warn / error / health) and Chris cut it: more alert
/// classes dilute all of them. If the config ever wants finer grain, `Alarm` can be split by
/// the [`Toast::to`] state without this enum growing a class that has no event behind it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SoundEvent {
    /// The system left green — degraded, or off.
    Alarm,
    /// The system came back to green. The all-clear.
    Recovery,
}

/// The shortest a toast stays up, however short its message.
///
/// Provisional — an N7 knob is requested (`UIQ-006`). Chris's constraint is *"ephemeral but
/// long enough to read"*, and a floor is the half of that a formula cannot supply: a
/// four-character message still has to be noticed before it is read.
pub const DWELL_FLOOR: Duration = Duration::from_millis(2500);

/// Reading time granted per character. 60 ms/char ≈ 200 words per minute at five characters
/// and a space per word — the rate typography handbooks use for continuous prose, which is
/// conservative here because a toast is a fragment rather than a paragraph.
pub const DWELL_PER_CHAR: Duration = Duration::from_millis(60);

/// The longest a toast stays up, however long its message. Past this the message is too long
/// to be a toast, and the notice belongs in the status zone where it can be read at leisure.
pub const DWELL_CEILING: Duration = Duration::from_millis(8000);

/// The tail of a toast's life during which it renders faint — the visual half of
/// "ephemeral". Not a knob: it is a rendering property of the surface, not a preference.
pub const FADE_TAIL: Duration = Duration::from_millis(600);

/// A settled change of system state, and what to say about it.
///
/// The fields are private and [`Toast::transition`] is the only constructor, which is what
/// makes the admission rule structural: nothing in this crate can raise an alert for a state
/// that did not change.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Toast {
    /// The state arrived at. [`Health::Healthy`] here *is* the recovery case — there is no
    /// second field to disagree with it.
    to: Health,
    message: String,
}

impl Toast {
    /// A toast for a settled transition from `from` to `to`, or [`None`] when the state did
    /// not change.
    ///
    /// Taking both ends rather than only the new state is the whole discipline: a status
    /// report that repeats itself is not an event, and an alert raised for one is the noise
    /// Chris cut this surface down to avoid. The caller cannot get an alert by asking twice.
    ///
    /// It does **not** debounce flapping — that detector is daemon-side (`CR-035`); see the
    /// module docs.
    pub fn transition(from: Health, to: Health, message: impl Into<String>) -> Option<Self> {
        (from != to).then(|| Self {
            to,
            message: message.into(),
        })
    }

    /// The state this toast announces. `Healthy` means recovery.
    pub const fn to(&self) -> Health {
        self.to
    }

    /// Whether this is the all-clear rather than an alarm.
    pub fn is_recovery(&self) -> bool {
        self.to == Health::Healthy
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    /// How long this toast stays up: reading time for its own length, held between the floor
    /// and the ceiling.
    pub fn dwell(&self) -> Duration {
        (DWELL_PER_CHAR * self.message.chars().count() as u32).clamp(DWELL_FLOOR, DWELL_CEILING)
    }

    /// The sound class the host should play when this toast is raised.
    pub fn sound(&self) -> SoundEvent {
        if self.is_recovery() {
            SoundEvent::Recovery
        } else {
            SoundEvent::Alarm
        }
    }

    /// §4's glyph vocabulary, reused rather than extended — the same choice `envelope.rs`
    /// made for severity. Shape carries the state where the encoding refuses colour, which
    /// matters more here than anywhere: an alarm and an all-clear must not read alike on a
    /// terminal that emits no hue.
    fn marks(&self) -> (&'static str, Style) {
        (self.to.glyph(), Style::default().fg(self.to.color()))
    }
}

/// Where a toast is in its life. The caller's `now` decides, never the clock.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Live,
    /// The last [`FADE_TAIL`] of the dwell: still readable, visibly going.
    Fading,
    Expired,
}

/// A toast that has been raised — the toast itself plus when, and how many identical repeats
/// have folded into it.
#[derive(Clone, Debug)]
pub struct Live {
    pub toast: Toast,
    pub raised: Instant,
    /// 1 for a toast raised once. Repeats extend nothing: the dwell still runs from
    /// [`Live::raised`], which is refreshed when a repeat merges.
    pub repeats: u32,
}

impl Live {
    pub fn phase(&self, now: Instant) -> Phase {
        let age = now.saturating_duration_since(self.raised);
        let dwell = self.toast.dwell();
        if age >= dwell {
            Phase::Expired
        } else if age + FADE_TAIL >= dwell {
            Phase::Fading
        } else {
            Phase::Live
        }
    }
}

/// The live toasts, newest last.
///
/// A deck rather than a single slot because a settled-change stream can produce two
/// transitions close together, and dropping the first would lose information the user was
/// told they would get.
#[derive(Clone, Debug)]
pub struct ToastDeck {
    live: Vec<Live>,
    cap: usize,
}

impl Default for ToastDeck {
    fn default() -> Self {
        Self::new()
    }
}

impl ToastDeck {
    /// How many toasts may be on screen at once. Three is the corner's budget: a fourth
    /// rectangle stacked upward starts to read as a panel rather than as a passing notice.
    pub const DEFAULT_CAP: usize = 3;

    pub fn new() -> Self {
        Self::with_cap(Self::DEFAULT_CAP)
    }

    pub fn with_cap(cap: usize) -> Self {
        Self {
            live: Vec::new(),
            cap: cap.max(1),
        }
    }

    /// Raise a toast, returning the sound the host should play — or [`None`] when this was a
    /// repeat of the toast already on top, which merges instead of stacking and stays silent.
    ///
    /// Silent on merge is the point: the sound announces *news*, and a repeat is not news.
    /// Note what this does not do — see the module docs: it collapses repetition, never
    /// flapping.
    pub fn push(&mut self, toast: Toast, now: Instant) -> Option<SoundEvent> {
        self.expire(now);

        if let Some(last) = self.live.last_mut()
            && last.toast == toast
        {
            last.repeats += 1;
            // The repeat restarts the dwell: the user's evidence that it is still happening
            // is that the toast is still there.
            last.raised = now;
            return None;
        }

        let sound = toast.sound();
        self.live.push(Live {
            toast,
            raised: now,
            repeats: 1,
        });
        // The oldest goes when the corner is full — a new toast is the one the user has not
        // seen yet.
        while self.live.len() > self.cap {
            self.live.remove(0);
        }
        Some(sound)
    }

    /// Drop everything past its dwell. Idempotent, and the only mutation `now` causes.
    pub fn expire(&mut self, now: Instant) {
        self.live.retain(|live| live.phase(now) != Phase::Expired);
    }

    /// The toasts that would be drawn at `now`, oldest first. Non-mutating, so a render pass
    /// never has to own the deck.
    pub fn active(&self, now: Instant) -> Vec<&Live> {
        self.live
            .iter()
            .filter(|live| live.phase(now) != Phase::Expired)
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }
}

/// The lower-right corner stack, as a widget.
///
/// Renders into whatever area it is given — the full screen in the TUI, a preview cell in the
/// pantry — and anchors itself to that area's lower-right corner.
pub struct ToastStack<'a> {
    deck: &'a ToastDeck,
    now: Instant,
}

/// Cells kept clear between the stack and the two screen edges it sits against, so the
/// rectangle reads as floating above the screen rather than welded to its corner.
const MARGIN: u16 = 1;

/// The widest a toast's text may be before it wraps. Wide enough for a sentence, narrow
/// enough that the stack never becomes the screen's main content.
const MAX_TEXT_WIDTH: u16 = 40;

/// The most lines of text one toast shows. Past this the message is truncated with an
/// ellipsis: a toast that scrolls is a panel.
const MAX_TEXT_LINES: usize = 3;

impl<'a> ToastStack<'a> {
    pub fn new(deck: &'a ToastDeck, now: Instant) -> Self {
        Self { deck, now }
    }
}

impl Widget for ToastStack<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Border on both sides plus one cell of padding inside each border, and the glyph
        // and its space in front of the first line.
        const CHROME: u16 = 4;
        const GLYPH: u16 = 2;

        if area.width <= MARGIN + CHROME + GLYPH || area.height <= MARGIN {
            return;
        }

        let text_width = MAX_TEXT_WIDTH.min(area.width - MARGIN - CHROME - GLYPH);
        // Newest first: the stack grows upward from the corner, so the toast that just
        // arrived is always in the same place.
        let mut bottom = area.bottom().saturating_sub(MARGIN);

        for live in self.deck.active(self.now).into_iter().rev() {
            let (glyph, glyph_style) = live.toast.marks();
            let fading = live.phase(self.now) == Phase::Fading;
            let body_style = if fading {
                tokens::faint_style()
            } else {
                tokens::normal_style()
            };

            let mut text = live.toast.message().to_string();
            if live.repeats > 1 {
                text.push_str(&format!(" ×{}", live.repeats));
            }
            let wrapped = wrap(&text, text_width as usize);

            let height = wrapped.len() as u16 + 2;
            let width = wrapped
                .iter()
                .map(|line| line.chars().count() as u16)
                .max()
                .unwrap_or(0)
                + CHROME
                + GLYPH;

            // A toast that does not fit above the one below it is not drawn at all: half a
            // rectangle in the corner is a rendering artefact, not a notice.
            if bottom < area.top() + height || width + MARGIN > area.width {
                break;
            }

            let rect = Rect {
                x: area.right() - MARGIN - width,
                y: bottom - height,
                width,
                height,
            };

            let lines: Vec<Line> = wrapped
                .iter()
                .enumerate()
                .map(|(i, line)| {
                    let lead = if i == 0 {
                        Span::styled(format!("{glyph} "), glyph_style)
                    } else {
                        // Continuation lines hang under the text, not under the glyph.
                        Span::raw("  ")
                    };
                    Line::from(vec![lead, Span::styled(line.clone(), body_style)])
                })
                .collect();

            // The fill is the §6 layer-1 background: a toast floats above the screen even
            // though it takes no part in the modal stack.
            let block = Block::bordered()
                .padding(Padding::horizontal(1))
                .border_style(if fading {
                    tokens::faint_style()
                } else {
                    tokens::muted_style()
                })
                .style(Style::default().bg(tokens::layer1_bg()));
            Paragraph::new(lines).block(block).render(rect, buf);

            bottom = rect.y;
            // One clear row between stacked toasts, so two rectangles never read as one.
            if bottom <= area.top() {
                break;
            }
            bottom -= 1;
        }
    }
}

/// Wrap on whitespace to `width`, truncating past [`MAX_TEXT_LINES`].
///
/// Counts characters rather than display columns, which is right for the ASCII messages the
/// contracts produce and wrong for CJK; a message that needs the difference belongs in the
/// status zone, which has the room to be careful.
fn wrap(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return Vec::new();
    }

    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        let addition = if current.is_empty() {
            word.chars().count()
        } else {
            word.chars().count() + 1
        };
        if !current.is_empty() && current.chars().count() + addition > width {
            lines.push(std::mem::take(&mut current));
        }
        if word.chars().count() > width {
            // A single word longer than the box: hard-split it rather than overflow.
            for ch in word.chars() {
                if current.chars().count() == width {
                    lines.push(std::mem::take(&mut current));
                }
                current.push(ch);
            }
        } else {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }

    if lines.len() > MAX_TEXT_LINES {
        lines.truncate(MAX_TEXT_LINES);
        if let Some(last) = lines.last_mut() {
            while last.chars().count() >= width && last.chars().count() > 1 {
                last.pop();
            }
            last.push('…');
        }
    }
    lines
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "deck",
            ty: "&ToastDeck",
            description: "The live toasts, newest last; the stack grows upward from the corner",
        },
        PropInfo {
            name: "now",
            ty: "Instant",
            description: "The moment being rendered. The widget never reads the clock itself",
        },
    ];

    /// A deck whose entries were raised at stated offsets before `now`, so a frame can show a
    /// toast mid-life without waiting for one.
    fn deck_at(entries: &[(Toast, Duration, u32)]) -> (ToastDeck, Instant) {
        let now = Instant::now();
        let mut deck = ToastDeck::with_cap(entries.len().max(1));
        for (toast, age, repeats) in entries {
            deck.live.push(Live {
                toast: toast.clone(),
                raised: now - *age,
                repeats: *repeats,
            });
        }
        (deck, now)
    }

    macro_rules! variant {
        ($ty:ident, $name:literal, $desc:literal, $build:expr) => {
            struct $ty;
            impl Ingredient for $ty {
                fn group(&self) -> &str {
                    "Toast"
                }
                fn name(&self) -> &str {
                    $name
                }
                fn source(&self) -> &str {
                    "wqm_tui::widgets::toast"
                }
                fn description(&self) -> &str {
                    $desc
                }
                fn props(&self) -> &[PropInfo] {
                    PROPS
                }
                fn render(&self, area: Rect, buf: &mut Buffer) {
                    let (deck, now): (ToastDeck, Instant) = $build;
                    ToastStack::new(&deck, now).render(area, buf);
                }
            }
        };
    }

    /// The transition every frame below is built from, unwrapped: a preview whose toast is
    /// `None` would be a preview of a bug.
    fn toast(from: Health, to: Health, message: &str) -> Toast {
        Toast::transition(from, to, message).expect("a preview transition must change state")
    }

    variant!(
        Degraded,
        "Alarm — degraded",
        "Green to degraded: the common alarm, and the one that must not read like the all-clear",
        deck_at(&[(
            toast(
                Health::Healthy,
                Health::Degraded,
                "vector store degraded — qdrant slow past its SLA",
            ),
            Duration::ZERO,
            1,
        )])
    );

    variant!(
        Offline,
        "Alarm — off",
        "The loud one: a store went away. Same rectangle, §4's offline glyph and hue",
        deck_at(&[(
            toast(
                Health::Degraded,
                Health::Offline,
                "vector store offline — qdrant unreachable",
            ),
            Duration::ZERO,
            1,
        )])
    );

    variant!(
        Recovery,
        "Recovery",
        "Back to green — the other half of what Chris kept. Does the all-clear read as relief, not alarm?",
        deck_at(&[(
            toast(Health::Offline, Health::Healthy, "vector store recovered"),
            Duration::ZERO,
            1,
        )])
    );

    variant!(
        Stack,
        "Stack",
        "Alarm then all-clear, the corner's budget. Newest is lowest; the arrival point never moves",
        deck_at(&[
            (
                toast(Health::Healthy, Health::Offline, "vector store offline"),
                Duration::from_millis(900),
                1,
            ),
            (
                toast(Health::Offline, Health::Degraded, "vector store degraded"),
                Duration::from_millis(400),
                1,
            ),
            (
                toast(Health::Degraded, Health::Healthy, "vector store recovered"),
                Duration::ZERO,
                1,
            ),
        ])
    );

    variant!(
        Repeated,
        "Repeated",
        "Eight identical reports, one rectangle: repetition collapses to a count — flapping does NOT (CR-035)",
        deck_at(&[(
            toast(Health::Healthy, Health::Degraded, "vector store degraded"),
            Duration::ZERO,
            8,
        )])
    );

    variant!(
        Fading,
        "Fading",
        "The last 600 ms of the dwell: still readable, visibly going. The ephemeral half, held still",
        deck_at(&[(
            toast(
                Health::Healthy,
                Health::Degraded,
                "the index is 96s behind its sources",
            ),
            toast(
                Health::Healthy,
                Health::Degraded,
                "the index is 96s behind its sources",
            )
            .dwell()
                - Duration::from_millis(300),
            1,
        )])
    );

    variant!(
        LongMessage,
        "Long Message",
        "Wraps at 40 columns and truncates at three lines — a toast that scrolls is a panel",
        deck_at(&[(
            toast(
                Health::Healthy,
                Health::Offline,
                "the index is 96 seconds behind its sources and the queue has not drained \
                 since the daemon last restarted on this host",
            ),
            Duration::ZERO,
            1,
        )])
    );

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Degraded),
            Box::new(Offline),
            Box::new(Recovery),
            Box::new(Stack),
            Box::new(Repeated),
            Box::new(Fading),
            Box::new(LongMessage),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::Encoding;
    use crate::tokens::Palette;

    fn render(deck: &ToastDeck, now: Instant, width: u16, height: u16) -> Buffer {
        let area = Rect::new(0, 0, width, height);
        let mut buf = Buffer::empty(area);
        ToastStack::new(deck, now).render(area, &mut buf);
        buf
    }

    fn symbol(buf: &Buffer, x: u16, y: u16) -> String {
        buf.cell((x, y)).expect("cell in area").symbol().to_string()
    }

    /// Every non-blank cell in the buffer, as (x, y).
    fn painted(buf: &Buffer) -> Vec<(u16, u16)> {
        let area = *buf.area();
        (area.top()..area.bottom())
            .flat_map(|y| (area.left()..area.right()).map(move |x| (x, y)))
            .filter(|(x, y)| symbol(buf, *x, *y) != " ")
            .collect()
    }

    /// The only constructor, unwrapped — for the tests that are about something other than
    /// the admission rule.
    fn toast(from: Health, to: Health, message: &str) -> Toast {
        Toast::transition(from, to, message).expect("a test transition must change state")
    }

    /// A plain alarm, for tests that only need *some* toast.
    fn alarm(message: &str) -> Toast {
        toast(Health::Healthy, Health::Degraded, message)
    }

    #[test]
    fn only_a_change_of_state_can_raise_a_toast() {
        // The volume rule, as a type. A status report that repeats itself is not an event,
        // so no caller can turn one into an alert by asking twice.
        for state in [Health::Healthy, Health::Degraded, Health::Offline] {
            assert!(
                Toast::transition(state, state, "nothing happened").is_none(),
                "{state:?} to itself is not a transition"
            );
        }

        // Both directions across the boundary are events, and the recovery is not an alarm.
        let alarm = toast(Health::Healthy, Health::Offline, "store offline");
        let recovery = toast(Health::Offline, Health::Healthy, "store recovered");
        assert!(!alarm.is_recovery());
        assert!(recovery.is_recovery());
        assert_eq!(alarm.sound(), SoundEvent::Alarm);
        assert_eq!(recovery.sound(), SoundEvent::Recovery);

        // A move between two unhealthy states is still news, and still an alarm.
        let partial = toast(Health::Offline, Health::Degraded, "store degraded");
        assert!(!partial.is_recovery());
        assert_eq!(partial.sound(), SoundEvent::Alarm);
    }

    #[test]
    fn the_stack_is_anchored_to_the_lower_right_corner() {
        let now = Instant::now();
        let mut deck = ToastDeck::new();
        deck.push(alarm("vector store degraded"), now);

        let buf = render(&deck, now, 60, 20);
        let cells = painted(&buf);
        assert!(!cells.is_empty(), "nothing was drawn");

        let right = cells.iter().map(|(x, _)| *x).max().unwrap();
        let bottom = cells.iter().map(|(_, y)| *y).max().unwrap();
        let left = cells.iter().map(|(x, _)| *x).min().unwrap();

        // MARGIN clear cells against both edges, and nothing on the left half of a 60-column
        // screen: this is a corner surface, not a footer.
        assert_eq!(right, 60 - 1 - MARGIN, "not flush against the right margin");
        assert_eq!(
            bottom,
            20 - 1 - MARGIN,
            "not flush against the bottom margin"
        );
        assert!(
            left > 30,
            "the toast spilled into the left half of the screen"
        );
    }

    #[test]
    fn the_newest_toast_is_the_one_nearest_the_corner() {
        let now = Instant::now();
        let mut deck = ToastDeck::new();
        deck.push(alarm("older"), now);
        deck.push(alarm("newer"), now);

        let buf = render(&deck, now, 60, 20);
        let rows: Vec<String> = (0..20)
            .map(|y| (0..60).map(|x| symbol(&buf, x, y)).collect::<String>())
            .collect();

        let older = rows
            .iter()
            .position(|r| r.contains("older"))
            .expect("older");
        let newer = rows
            .iter()
            .position(|r| r.contains("newer"))
            .expect("newer");
        assert!(
            newer > older,
            "the newest toast must sit lowest — the arrival point has to stay fixed"
        );
    }

    #[test]
    fn a_widget_never_reads_the_clock() {
        let raised = Instant::now();
        let mut deck = ToastDeck::new();
        deck.push(alarm("a message worth eight seconds"), raised);
        let dwell = alarm("a message worth eight seconds").dwell();

        // Real time passing changes nothing; only the `now` the caller states does.
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(deck.active(raised).len(), 1);
        assert_eq!(
            deck.active(raised + dwell - Duration::from_millis(1)).len(),
            1
        );
        assert_eq!(
            deck.active(raised + dwell).len(),
            0,
            "the toast outlived its own dwell"
        );

        // And two renders of the same stated moment are byte-identical, which is what makes
        // a storyboard frame reproducible.
        let a = render(&deck, raised, 60, 20);
        std::thread::sleep(Duration::from_millis(20));
        let b = render(&deck, raised, 60, 20);
        assert_eq!(a, b);
    }

    #[test]
    fn the_last_tail_of_the_dwell_fades() {
        let raised = Instant::now();
        let going = alarm("going");
        let dwell = going.dwell();
        let mut deck = ToastDeck::new();
        deck.push(going, raised);

        let live = deck.active(raised)[0];
        assert_eq!(live.phase(raised), Phase::Live);
        assert_eq!(live.phase(raised + dwell - FADE_TAIL), Phase::Fading);
        assert_eq!(live.phase(raised + dwell), Phase::Expired);
    }

    #[test]
    fn repetition_collapses_and_stays_silent_but_flapping_does_not() {
        let now = Instant::now();
        let mut deck = ToastDeck::new();

        // Eight reports of the same transition — one toast, one sound, a count of 8.
        for i in 0..8 {
            let sound = deck.push(alarm("store degraded"), now);
            if i == 0 {
                assert_eq!(sound, Some(SoundEvent::Alarm), "the first raise is news");
            } else {
                assert_eq!(sound, None, "a repeat is not news and must not sound");
            }
        }
        let active = deck.active(now);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].repeats, 8);

        // Flapping is NOT coalesced, and that is where CR-035 puts the fix: a settled-change
        // detector belongs upstream, daemon-side. Pinned so nobody reads the merge above as
        // debouncing — and it matters more now that the surface is only alarms and all-clears,
        // because every one of these four carries a sound.
        let mut flapping = ToastDeck::with_cap(16);
        for _ in 0..2 {
            flapping.push(
                toast(Health::Healthy, Health::Degraded, "store degraded"),
                now,
            );
            flapping.push(
                toast(Health::Degraded, Health::Healthy, "store recovered"),
                now,
            );
        }
        assert_eq!(
            flapping.active(now).len(),
            4,
            "this deck must not appear to debounce — it collapses repeats only"
        );
    }

    #[test]
    fn the_cap_drops_the_oldest() {
        let now = Instant::now();
        let mut deck = ToastDeck::with_cap(2);
        deck.push(alarm("first"), now);
        deck.push(alarm("second"), now);
        deck.push(alarm("third"), now);

        let messages: Vec<&str> = deck.active(now).iter().map(|l| l.toast.message()).collect();
        assert_eq!(messages, vec!["second", "third"]);
    }

    #[test]
    fn dwell_is_reading_time_held_between_a_floor_and_a_ceiling() {
        assert_eq!(alarm("ok").dwell(), DWELL_FLOOR);
        assert_eq!(alarm(&"x".repeat(400)).dwell(), DWELL_CEILING);

        let middling = alarm(&"x".repeat(100)).dwell();
        assert!(middling > DWELL_FLOOR && middling < DWELL_CEILING);
        assert!(
            alarm(&"x".repeat(100)).dwell() > alarm(&"x".repeat(60)).dwell(),
            "a longer message must be given longer to read"
        );
    }

    #[test]
    fn an_alarm_and_an_all_clear_differ_where_the_encoding_refuses_colour() {
        let _serial = crate::global_state_lock();
        let previous = (Palette::current(), Encoding::current());
        Palette::set(Palette::Derived);
        Encoding::set(Encoding::NoColor);

        let now = Instant::now();
        let mut down = ToastDeck::new();
        down.push(toast(Health::Healthy, Health::Offline, "store"), now);
        let mut up = ToastDeck::new();
        up.push(toast(Health::Offline, Health::Healthy, "store"), now);

        let down_cells = render(&down, now, 40, 10);
        let up_cells = render(&up, now, 40, 10);
        // Same message, opposite meaning: with hue gone the §4 glyph is the entire signal,
        // and mistaking an all-clear for an alarm is the worst failure this surface has.
        assert_ne!(
            down_cells, up_cells,
            "alarm and recovery rendered identically without colour"
        );
        // The rectangle itself must still be there — a border is structure, not colour.
        assert!(
            painted(&down_cells).len() > 10,
            "the box vanished without colour"
        );

        Palette::set(previous.0);
        Encoding::set(previous.1);
    }

    #[test]
    fn a_toast_that_does_not_fit_is_dropped_rather_than_clipped() {
        let now = Instant::now();
        let mut deck = ToastDeck::new();
        deck.push(alarm("one"), now);
        deck.push(alarm("two"), now);
        deck.push(alarm("three"), now);

        // Room for one rectangle and its margin, no more.
        let buf = render(&deck, now, 30, 4);
        let cells = painted(&buf);
        assert!(!cells.is_empty(), "the newest toast should still be drawn");
        for (_, y) in &cells {
            assert!(*y < 4, "drew outside the area");
        }
        let rows: Vec<String> = (0..4)
            .map(|y| (0..30).map(|x| symbol(&buf, x, y)).collect::<String>())
            .collect();
        assert!(
            rows.iter().any(|r| r.contains("three")),
            "the newest is the one kept"
        );
        assert!(
            !rows.iter().any(|r| r.contains("one")),
            "an older toast was clipped in"
        );
    }

    #[test]
    fn a_long_message_wraps_and_then_truncates() {
        let long = "the index is 96 seconds behind its sources and the queue has not drained since the last restart of the daemon on this host";
        let lines = wrap(long, 40);
        assert_eq!(lines.len(), MAX_TEXT_LINES);
        for line in &lines {
            assert!(line.chars().count() <= 40, "{line:?} overflowed the box");
        }
        assert!(
            lines.last().unwrap().ends_with('…'),
            "truncation must say that it truncated"
        );
    }

    #[test]
    fn a_word_longer_than_the_box_is_split_rather_than_overflowed() {
        let lines = wrap("/very/long/path/with/no/spaces/at/all/in/it/anywhere", 12);
        for line in &lines {
            assert!(line.chars().count() <= 12, "{line:?} overflowed the box");
        }
    }
}
