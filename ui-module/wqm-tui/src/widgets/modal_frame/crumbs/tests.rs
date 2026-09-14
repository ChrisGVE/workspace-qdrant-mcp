//! The trail, checked against item (a) sentence by sentence.

use super::*;
use crate::encoding::Encoding;
use crate::tokens::{contrast, ModalTint, Palette};

struct Restore(Palette, Encoding, ModalTint, f32);

impl Restore {
    fn mocha() -> Self {
        let restore = Restore(
            Palette::current(),
            Encoding::current(),
            ModalTint::current(),
            tokens::tint_strength(),
        );
        Palette::set(Palette::Bundled);
        Encoding::set(Encoding::TrueColor);
        tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());
        ModalTint::set(ModalTint::Accent);
        tokens::set_tint_strength(0.40);
        restore
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        Palette::set(self.0);
        Encoding::set(self.1);
        ModalTint::set(self.2);
        tokens::set_tint_strength(self.3);
    }
}

fn trail() -> Vec<String> {
    ["Queue", "open-books", "reading_guide.py"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Every background the trail uses is distinguishable from the one beside it, or the segments
/// are one bar with text in it.
///
/// Three surfaces, in the order item (a) names them: the window's fill, the current crumb
/// *"mid-way"*, and the ancestors at *"full saturation"*. Each must clear a just-noticeable
/// difference from its neighbour, and they must be ORDERED — midway between is a claim about
/// position, not merely about being different.
#[test]
fn the_three_backgrounds_are_ordered_and_separable() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();

    for name in ratatui_themes::ThemeName::all() {
        tokens::set_theme(name.palette());
        let window = tokens::modal_fill(tokens::layer1_bg());
        let (mid, full) = (current_bg(), ancestor_bg());

        let to_mid = tokens::delta_e(window, mid);
        let mid_to_full = tokens::delta_e(mid, full);
        let to_full = tokens::delta_e(window, full);

        assert!(
            to_mid > contrast::JND,
            "{}: the current crumb is ΔE {to_mid:.1} off the window — one surface, not two",
            name.display_name()
        );
        assert!(
            mid_to_full > contrast::JND,
            "{}: the ancestors are ΔE {mid_to_full:.1} off the current crumb",
            name.display_name()
        );
        // *"mid-way"* is asserted as an ORDER and not as a position on a straight line, and the
        // difference is the legibility correction: `legible_ground` may move either ground's
        // lightness to keep its own text readable, which takes the midpoint off the chord
        // between the other two. Measured on One Dark Pro the correction leaves mid→full at
        // 34.2 against a window→full of 29.3 — plainly off-axis, and equally plainly still the
        // nearer of the two to the window, which is the half of "mid-way" a reader sees.
        assert!(
            to_mid < to_full,
            "{}: the current crumb must sit NEARER the window than the ancestors do — \
             window→mid {to_mid:.1}, window→full {to_full:.1}",
            name.display_name()
        );
    }
}

/// **The objection, measured.** Crumb text clears the body floor on every bundled theme — which
/// literal white does not.
///
/// This is the test that makes the substitution in [`super`]'s module docs a finding rather than
/// an opinion: it fails on the instruction as written and passes on the role it was read as.
#[test]
fn crumb_text_is_legible_on_every_theme_and_literal_white_is_not() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();

    let mut white_failures = Vec::new();
    for name in ratatui_themes::ThemeName::all() {
        tokens::set_theme(name.palette());
        for bg in [ancestor_bg(), current_bg()] {
            let chosen = contrast::contrast_ratio(contrast::text_on(bg), bg);
            assert!(
                chosen >= contrast::BODY_FLOOR,
                "{}: crumb text is {chosen:.1}:1 on its own background",
                name.display_name()
            );
        }
        let white = contrast::contrast_ratio(tokens::neutral_at(100), ancestor_bg());
        if white < contrast::BODY_FLOOR {
            white_failures.push((name.display_name(), white));
        }
    }
    assert!(
        !white_failures.is_empty(),
        "if literal white now passes everywhere, the substitution has stopped being needed and \
         this whole treatment should go back to Chris's wording"
    );
    println!("white on the full-saturation accent fails on: {white_failures:?}");
}

/// The shape of the trail, span by span: ancestors share one run, the current crumb has its
/// own, and both transitions are there.
///
/// Asserted on the SPANS rather than on a rendered buffer, because the claim is about which
/// colour each glyph carries and a buffer comparison would pass on a trail whose separators were
/// the right colour in the wrong places.
#[test]
fn the_trail_is_one_ancestor_run_a_transition_and_the_current_crumb() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let line = line(&trail(), CrumbStyle::Powerline);
    let (ancestor, live, window) = (
        ancestor_bg(),
        current_bg(),
        tokens::modal_fill(tokens::layer1_bg()),
    );

    let bg_of = |span: &Span<'static>| span.style.bg.expect("every crumb span states a bg");
    let spans = &line.spans;

    // Queue | sep | open-books | transition | reading_guide.py | transition
    assert_eq!(spans.len(), 6, "{spans:?}");
    assert_eq!(bg_of(&spans[0]), ancestor);
    assert_eq!(
        bg_of(&spans[1]),
        ancestor,
        "the inner separator stays inside the run"
    );
    assert_eq!(bg_of(&spans[2]), ancestor);
    assert_eq!(
        (spans[3].style.fg, bg_of(&spans[3])),
        (Some(ancestor), live),
        "the transition is the outgoing colour drawn on the incoming one"
    );
    assert_eq!(bg_of(&spans[4]), live);
    assert!(
        spans[4].style.add_modifier.contains(Modifier::BOLD),
        "the current crumb is bold"
    );
    assert_eq!(
        (spans[5].style.fg, bg_of(&spans[5])),
        (Some(live), window),
        "the current run finishes with a chevron onto the window"
    );
}

/// A trail of one is all current: no ancestor run, and the transition comes off the window.
///
/// The depth-1 frame. Without this the first crumb of a fresh window would be drawn as an
/// ancestor with nothing after it, which is a trail pointing at nowhere.
#[test]
fn a_trail_of_one_is_all_current() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let line = line(&["Configuration".to_string()], CrumbStyle::Powerline);
    let window = tokens::modal_fill(tokens::layer1_bg());

    assert_eq!(line.spans.len(), 3, "{:?}", line.spans);
    assert_eq!(line.spans[0].style.fg, Some(window));
    assert_eq!(line.spans[1].style.bg, Some(current_bg()));
    assert!(line.spans[1].content.contains("Configuration"));
}

/// An empty trail draws nothing rather than a bare pair of chevrons pointing at no name.
#[test]
fn an_empty_trail_draws_nothing() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    assert!(line(&[], CrumbStyle::Powerline).spans.is_empty());
    assert!(line(&[], CrumbStyle::Plain).spans.is_empty());
}

/// Round 1's trail is untouched, so the A/B pair really is a pair.
#[test]
fn the_plain_trail_is_still_round_ones() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let line = line(&trail(), CrumbStyle::Plain);
    let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    assert_eq!(text, "Queue › open-books › reading_guide.py");
    assert!(
        line.spans.iter().all(|s| s.style.bg.is_none()),
        "the plain trail spends no background at all"
    );
}
