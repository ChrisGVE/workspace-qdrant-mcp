//! What the categorical tier is pinned to.
//!
//! The orders asserted here were computed **outside this crate** — an independent CIE76
//! implementation over Catppuccin's published hex values, run once — and then written down.
//! That matters: a test that obtained its expectation by calling the function it is testing
//! asserts only that the function is deterministic, which was never in doubt.

use super::*;
use ratatui_themes::ThemeName;

/// The order the rule produces on Mocha, computed independently and pinned.
///
/// `blue` used to lead this list. It is gone since 20260906, when `accent` joined the reserved
/// set: the jump digits carry `accent` on **every** screen, so a data hue equal to it collides
/// everywhere rather than on one view, and that beats the earlier "affordance, not data"
/// argument. `lavender` went with it — see [`MOCHA_LAVENDER_TO_BLUE`].
const MOCHA_ORDER: [&str; 8] = [
    "pink",
    "mauve",
    "peach",
    "rosewater",
    "sapphire",
    "sky",
    "maroon",
    "flamingo",
];

/// Every flavour's count, computed the same way. They differ because the exclusion is a
/// measurement against that flavour's own reserved roles, not a fixed list of names.
///
/// Reserving `accent` cost every flavour its `blue`. It cost **Mocha alone** a second hue,
/// which is why Mocha is 8 rather than 9 — a prediction of 9 was made before the arithmetic
/// was run, and the arithmetic disagreed.
const COUNTS: [(&str, usize); 4] = [
    ("mocha", 8),
    ("macchiato", 8),
    ("frappe", 8),
    ("latte", 7),
];

/// Why Mocha lost two hues where the others lost one.
///
/// Mocha's `lavender` sits ΔE 11.0 from its `blue`, just under the floor of 12.0, so reserving
/// `accent` swept it up as well. The other three clear it — Macchiato by 0.6, Frappé by 1.6,
/// Latte by 9.3 — which is why only Mocha pays twice. Pinned because it is the whole
/// explanation of a count that otherwise looks like an off-by-one.
const MOCHA_LAVENDER_TO_BLUE: f32 = 11.0;

use super::mirrored_palette as flavour_palette;

/// The mirrored mapping is checked against GROUND TRUTH, not against itself.
///
/// `ratatui-themes` bundles Mocha and Latte, so for those two the mapping this crate
/// reproduces can be compared with the real thing. If it matches on both, the Frappé and
/// Macchiato frames — which have no real thing to compare against — are trustworthy by
/// construction rather than by assertion. If that crate ever retunes its Catppuccin rows, this
/// fails here rather than silently making two preview frames a fiction.
///
/// # `muted` is excluded, and that is an upstream discrepancy rather than a slack test
///
/// This guard found it on its first run. `ratatui-themes` sets Latte's `muted` to
/// `#8C8FA1` under a comment reading *"Overlay0"* — but Catppuccin's Latte `overlay0` is
/// `#9CA0B0`; `#8C8FA1` is `overlay1`. Mocha's row uses `overlay0` correctly, so the two
/// bundled flavours draw that field from different source colours. Nothing here depends on
/// `muted` — it is neither the detection key, nor a reserved role, nor a fallback candidate —
/// so the divergence is recorded and stepped around rather than papered over by relaxing the
/// comparison for every field.
#[test]
fn the_mirrored_mapping_reproduces_the_bundled_flavours_exactly() {
    for (identifier, name) in [
        ("mocha", ThemeName::CatppuccinMocha),
        ("latte", ThemeName::CatppuccinLatte),
    ] {
        let flavour = catppuccin::PALETTE
            .all_flavors()
            .into_iter()
            .find(|f| f.identifier() == identifier)
            .expect("a named flavour");
        let mine = mirrored_palette(flavour);
        let theirs = name.palette();
        // Every field this module READS. `bg` is the detection key and the four roles are the
        // exclusion set; `accent`/`secondary` are the fallback's two candidates, which a
        // Catppuccin flavour never reaches but which must still mirror correctly.
        for (field, a, b) in [
            ("bg", mine.bg, theirs.bg),
            ("success", mine.success, theirs.success),
            ("warning", mine.warning, theirs.warning),
            ("error", mine.error, theirs.error),
            ("info", mine.info, theirs.info),
            ("accent", mine.accent, theirs.accent),
            ("secondary", mine.secondary, theirs.secondary),
            ("fg", mine.fg, theirs.fg),
            ("selection", mine.selection, theirs.selection),
        ] {
            assert_eq!(a, b, "{identifier}.{field} does not match what ratatui-themes bundles");
        }
    }
}

/// The tie-break, exercised directly — because Catppuccin's own data contains no tie.
///
/// Reversing the walk over the real flavours changes nothing, so the "ties keep the earlier
/// colour" clause is unreachable from the public API and would rot unnoticed. A synthetic pool
/// of two identical hues makes the tie exact, and the only thing that can decide it is the
/// order the pool is walked in.
#[test]
fn an_exact_tie_keeps_the_colour_that_comes_first() {
    let theme = ThemeName::CatppuccinMocha.palette();
    let reserved = reserved_of(&theme);
    // Mocha's `pink` — a hue that SURVIVES the exclusion, so the pool reaches the ordering
    // walk at all. The first version used its `blue`, which became a reserved role on
    // 20260906 and was filtered out before the tie could happen.
    let same = Color::Rgb(0xf5, 0xc2, 0xe7);

    let order: Vec<&str> = farthest_first(vec![("first", same), ("second", same)], &reserved)
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    assert_eq!(
        order,
        ["first", "second"],
        "a tie must resolve to the earlier entry, or the whole result is unreproducible"
    );
}

#[test]
fn each_catppuccin_flavour_is_recognised_by_its_own_background() {
    for flavour in catppuccin::PALETTE.all_flavors() {
        let found = flavour_of(&flavour_palette(flavour))
            .unwrap_or_else(|| panic!("{} was not recognised", flavour.identifier()));
        assert_eq!(found.identifier(), flavour.identifier());
    }
}

/// Detection is by DATA, not by name — and the negative half is what makes that claim mean
/// something. Three themes that are emphatically not Catppuccin must come back as none.
#[test]
fn a_theme_that_is_not_a_catppuccin_flavour_is_recognised_as_none() {
    for name in [ThemeName::Everforest, ThemeName::Dracula, ThemeName::Nord] {
        assert!(
            flavour_of(&name.palette()).is_none(),
            "{} was mistaken for a Catppuccin flavour",
            name.display_name()
        );
    }
}

/// Every entry, on every flavour, clears the reserved floor against all five reserved roles.
/// This is the whole promise of the tier: a categorical hue never reads as a state, and never
/// reads as the key you press.
#[test]
fn no_entry_on_any_flavour_comes_within_the_reserved_floor_of_a_role() {
    for flavour in catppuccin::PALETTE.all_flavors() {
        let theme = flavour_palette(flavour);
        let reserved = reserved_of(&theme);
        for (name, colour) in Categorical::for_theme(&theme).iter() {
            let nearest = min_delta(colour, &reserved);
            assert!(
                nearest >= RESERVED_FLOOR,
                "{}/{name} is ΔE {nearest:.1} from a reserved role, under the floor of {RESERVED_FLOOR}",
                flavour.identifier()
            );
        }
    }
}

#[test]
fn the_order_on_mocha_is_the_one_the_rule_produces() {
    let theme = ThemeName::CatppuccinMocha.palette();
    let got: Vec<&str> = Categorical::for_theme(&theme)
        .iter()
        .map(|(name, _)| name)
        .collect();
    assert_eq!(got, MOCHA_ORDER, "the farthest-first order moved");
    assert_eq!(
        got.first().copied(),
        Some(MOCHA_ORDER[0]),
        "entry 0 is the survivor farthest from every reserved role"
    );
}

#[test]
fn each_flavour_yields_the_count_the_exclusion_leaves() {
    for (identifier, want) in COUNTS {
        let flavour = catppuccin::PALETTE
            .all_flavors()
            .into_iter()
            .find(|f| f.identifier() == identifier)
            .expect("a named flavour");
        assert_eq!(
            Categorical::for_theme(&flavour_palette(flavour)).len(),
            want,
            "{identifier} no longer leaves {want} survivors"
        );
    }
}

/// Farthest-first is not merely "sorted somehow": each pick must be at least as far from
/// everything already chosen as the pick after it. Asserted on the produced order, so a
/// greedy step that regressed would be caught even if the set were right.
#[test]
fn the_order_never_improves_as_it_goes() {
    let theme = ThemeName::CatppuccinMocha.palette();
    let reserved = reserved_of(&theme);
    let entries: Vec<(&str, Color)> = Categorical::for_theme(&theme).iter().collect();

    let mut previous = f32::MAX;
    for (i, (name, colour)) in entries.iter().enumerate() {
        let mut nearest = min_delta(*colour, &reserved);
        for (_, earlier) in &entries[..i] {
            nearest = nearest.min(crate::tokens::delta_e(*colour, *earlier));
        }
        assert!(
            nearest <= previous + f32::EPSILON,
            "{name} at {i} is further from the chosen set than the entry before it"
        );
        previous = nearest;
    }
}

/// A theme with no flavour still answers, honestly and very small.
///
/// One candidate now, not two: `accent` became a reserved role on 20260906, and a reserved
/// role cannot be its own alternative — it sits ΔE 0 from itself and fails the floor by
/// construction. So the fallback is `secondary` alone, and Everforest yields exactly one hue.
#[test]
fn a_non_catppuccin_theme_falls_back_to_secondary_alone() {
    let categorical = Categorical::for_theme(&ThemeName::Everforest.palette());
    assert_eq!(
        categorical.len(),
        1,
        "the fallback is `secondary` and nothing else"
    );
    assert_eq!(categorical.name(0), Some("secondary"));
    assert!(
        categorical.iter().all(|(name, _)| name != "accent"),
        "`accent` is reserved and can never be offered as a data hue"
    );
}

/// The reservation that produced the counts above, asserted directly rather than only through
/// them: no entry, on any flavour, may be the theme's own `accent`.
#[test]
fn no_entry_is_the_hue_the_jump_digits_carry() {
    for flavour in catppuccin::PALETTE.all_flavors() {
        let theme = flavour_palette(flavour);
        for (name, colour) in Categorical::for_theme(&theme).iter() {
            let distance = crate::tokens::delta_e(colour, theme.accent);
            assert!(
                distance >= RESERVED_FLOOR,
                "{}/{name} is ΔE {distance:.1} from `accent`, which is on every screen",
                flavour.identifier()
            );
        }
    }
}

/// The measurement that explains Mocha's count, checked rather than asserted in prose.
#[test]
fn mocha_loses_lavender_because_it_sits_inside_the_floor_of_blue() {
    let mocha = ThemeName::CatppuccinMocha.palette();
    let lavender: Color = catppuccin::PALETTE.mocha.colors.lavender.into();
    let distance = crate::tokens::delta_e(lavender, mocha.accent);
    assert!(
        (distance - MOCHA_LAVENDER_TO_BLUE).abs() < 0.1,
        "the distance that explains the count moved: {distance:.1}"
    );
    assert!(
        distance < RESERVED_FLOOR,
        "lavender must be inside the floor, or Mocha would have nine"
    );
}

/// `color(i)` cycles, so a consumer with more categories than hues never panics and never
/// reaches for a colour that is not in the set.
#[test]
fn indexing_past_the_end_wraps_rather_than_failing() {
    let theme = ThemeName::CatppuccinMocha.palette();
    let categorical = Categorical::for_theme(&theme);
    let n = categorical.len();
    assert_eq!(categorical.color(0), categorical.color(n));
    assert_eq!(categorical.color(1), categorical.color(n + 1));
    assert_eq!(categorical.name(0), categorical.name(n));
}

/// Where the stream refuses colour there is no categorical tier at all — `len() == 0`, so a
/// consumer reading it is told to fall back to structure rather than handed hues that will all
/// render as the same `Color::Reset`.
#[test]
fn an_encoding_that_refuses_colour_has_no_categorical_tier() {
    let _serial = crate::global_state_lock();
    let previous = (Palette::current(), Encoding::current(), crate::tokens::theme());
    Palette::set(Palette::Bundled);
    crate::tokens::set_theme(ThemeName::CatppuccinMocha.palette());

    Encoding::set(Encoding::TrueColor);
    assert!(
        !Categorical::current().is_empty(),
        "the tier must exist to be lost"
    );

    Encoding::set(Encoding::NoColor);
    assert_eq!(
        Categorical::current().len(),
        0,
        "no colour means no categorical distinctions to make"
    );

    Palette::set(previous.0);
    Encoding::set(previous.1);
    if let Some(theme) = previous.2 {
        crate::tokens::set_theme(theme);
    }
}
