//! Bundled-theme data, for judging a theme source rather than arguing about one.
//!
//! # Why the data is here and not behind a crate
//!
//! Base16 is a **format**, not a library. Its schemes live in
//! [`tinted-theming/schemes`](https://github.com/tinted-theming/schemes) as YAML under the
//! **MIT licence** (verified from the repository's own `LICENSE`, not from a metadata
//! column — `crate-screen.csv`'s licence field has twice misled). Two consequences:
//!
//! - `chromata` (1104 themes) is **GPL-3.0-only** and generates itself from exactly this
//!   data. We may read it as a design reference and may **not** ship it, or copy its output.
//! - `ratatui-base16` exposes the slots we want but pins **ratatui ^0.29**, whose `Color` is
//!   a different type from our 0.30. It is out as a dependency for that reason alone.
//!
//! So the shipping route is to generate our own table from the MIT YAML, and the four
//! schemes below are that route demonstrated at n=4 rather than described.
//!
//! # What a slot's name is, and is not
//!
//! Base16 names slots **positionally** — `base00`–`base0F` — and assigns hues by
//! *convention*. The spec's own words: *"These are just guidelines and will most often
//! provide best results when they are followed"*, and `ratatui-base16`'s docs go further:
//! *"scheme designers should pick whichever colours they desire, e.g. base0B (green by
//! default) could be replaced with red."*
//!
//! So **every theme has a slot that plays red's role; no theme guarantees that slot is red.**
//! [`Slot::scheme_name`] carries the scheme author's own name for the slot where the YAML
//! records one — Catppuccin annotates all sixteen, Solarized annotates none.

use ratatui::style::Color;

/// One palette slot: what base16 calls it, what the spec says it is for, what colour it is,
/// and what the scheme's own author called it.
pub struct Slot {
    /// The positional base16 name — `base00` … `base0F`. Universal across every scheme.
    pub slot: &'static str,
    /// The spec's role for this slot. Universal; the hue in parentheses is a guideline.
    pub spec_role: &'static str,
    pub colour: Color,
    /// The scheme author's own name for this slot, where the YAML records one. `None` is
    /// common and is not a defect — Solarized names nothing.
    pub scheme_name: Option<&'static str>,
}

/// Whether a scheme's ramp runs dark-to-light or light-to-dark. Declared in the YAML AND
/// derivable from the ramp itself, which is why [`Scheme::declared_variant_matches_ramp`]
/// can check one against the other.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Variant {
    Dark,
    Light,
}

pub struct Scheme {
    pub name: &'static str,
    pub author: &'static str,
    pub variant: Variant,
    pub slots: [Slot; 16],
}

/// The spec's role for each slot, in order. Quoted from
/// `tinted-theming/home/styling.md`; the hue is the guideline half.
const SPEC_ROLES: [&str; 16] = [
    "Default Background",
    "Lighter Background (status bars)",
    "Selection Background",
    "Comments, Invisibles, Line Highlighting",
    "Dark Foreground (status bars)",
    "Default Foreground, Caret, Delimiters",
    "Light Foreground",
    "The Lightest Foreground",
    "(Red) Variables, Diff Deleted",
    "(Orange) Integers, Constants",
    "(Yellow) Classes, Search Background",
    "(Green) Strings, Diff Inserted",
    "(Cyan) Support, Escapes, Quotes",
    "(Blue) Functions, Methods, Headings",
    "(Purple) Keywords, Diff Changed",
    "(Brown) Deprecated",
];

const SLOT_NAMES: [&str; 16] = [
    "base00", "base01", "base02", "base03", "base04", "base05", "base06", "base07", "base08",
    "base09", "base0A", "base0B", "base0C", "base0D", "base0E", "base0F",
];

const fn rgb(hex: u32) -> Color {
    Color::Rgb(
        ((hex >> 16) & 0xff) as u8,
        ((hex >> 8) & 0xff) as u8,
        (hex & 0xff) as u8,
    )
}

/// Build a scheme from its sixteen values and the author's own names for them.
const fn scheme(
    name: &'static str,
    author: &'static str,
    variant: Variant,
    values: [u32; 16],
    names: [Option<&'static str>; 16],
) -> Scheme {
    // Written out rather than looped: `const fn` cannot build an array in a loop on our
    // MSRV, and the alternative — building it at runtime — would put theme data behind a
    // `OnceLock` for no gain.
    macro_rules! slot {
        ($i:expr) => {
            Slot {
                slot: SLOT_NAMES[$i],
                spec_role: SPEC_ROLES[$i],
                colour: rgb(values[$i]),
                scheme_name: names[$i],
            }
        };
    }
    Scheme {
        name,
        author,
        variant,
        slots: [
            slot!(0),
            slot!(1),
            slot!(2),
            slot!(3),
            slot!(4),
            slot!(5),
            slot!(6),
            slot!(7),
            slot!(8),
            slot!(9),
            slot!(10),
            slot!(11),
            slot!(12),
            slot!(13),
            slot!(14),
            slot!(15),
        ],
    }
}

/// `base16/catppuccin-mocha.yaml`. The one scheme here that names all sixteen slots, which
/// is why it is the clearest demonstration that the two namings are independent.
pub fn catppuccin_mocha() -> Scheme {
    scheme(
        "Catppuccin Mocha",
        "https://github.com/catppuccin/catppuccin",
        Variant::Dark,
        [
            0x1e1e2e, 0x181825, 0x313244, 0x45475a, 0x585b70, 0xcdd6f4, 0xf5e0dc, 0xb4befe,
            0xf38ba8, 0xfab387, 0xf9e2af, 0xa6e3a1, 0x94e2d5, 0x89b4fa, 0xcba6f7, 0xf2cdcd,
        ],
        [
            Some("base"),
            Some("mantle"),
            Some("surface0"),
            Some("surface1"),
            Some("surface2"),
            Some("text"),
            Some("rosewater"),
            Some("lavender"),
            Some("red"),
            Some("peach"),
            Some("yellow"),
            Some("green"),
            Some("teal"),
            Some("blue"),
            Some("mauve"),
            Some("flamingo"),
        ],
    )
}

/// `base16/catppuccin-latte.yaml` — Mocha's light counterpart, and the pair that shows why
/// `base01` is not our layer 1.
///
/// **Catppuccin has two background ramps and base16 has room for one.** Its `base`/`mantle`/
/// `crust` run *away* from the base surface, and its `surface0`/`surface1`/`surface2` run
/// *toward* the reader. base16 maps `base01` to `mantle` — a deeper background — so the slot
/// the spec calls "Lighter Background" holds Catppuccin's darker one, and Catppuccin's raised
/// surface lands on `base02`. `crust` does not survive the mapping at all.
///
/// Chris, 20260731, on why the deeper ramp exists: *"crust is darker than mantle … meant to
/// prevent excessive contrast that is tiring for the eyes."* Measured against `text`, the
/// effect inverts with polarity even though the ordering does not — on Mocha deepening moves
/// AWAY from the text (182.9 → 196.6 luma distance), on Latte it moves TOWARD it
/// (159.8 → 142.6). Same ramp, opposite job: separation on a dark theme, contrast relief on
/// a light one.
pub fn catppuccin_latte() -> Scheme {
    scheme(
        "Catppuccin Latte",
        "https://github.com/catppuccin/catppuccin",
        Variant::Light,
        [
            0xeff1f5, 0xe6e9ef, 0xccd0da, 0xbcc0cc, 0xacb0be, 0x4c4f69, 0xdc8a78, 0x7287fd,
            0xd20f39, 0xfe640b, 0xdf8e1d, 0x40a02b, 0x179299, 0x1e66f5, 0x8839ef, 0xdd7878,
        ],
        [
            Some("base"),
            Some("mantle"),
            Some("surface0"),
            Some("surface1"),
            Some("surface2"),
            Some("text"),
            Some("rosewater"),
            Some("lavender"),
            Some("red"),
            Some("peach"),
            Some("yellow"),
            Some("green"),
            Some("teal"),
            Some("blue"),
            Some("mauve"),
            Some("flamingo"),
        ],
    )
}

/// `base16/dracula.yaml`. Kept because it is the counter-example: its slot COMMENTS say
/// Blue and Purple where the values are violet and pink, and `base06` equals `base05`
/// (the YAML carries a `TODO review` beside it). Both matter — see `theme_sheet`.
pub fn dracula() -> Scheme {
    scheme(
        "Dracula",
        "clach04 (https://github.com/clach04)",
        Variant::Dark,
        [
            0x282a36, 0x21222c, 0x44475A, 0x6272a4, 0x9ea8c7, 0xf8f8f2, 0xf8f8f2, 0xffffff,
            0xff5555, 0xFFB86C, 0xf1fa8c, 0x50fa7b, 0x8be9fd, 0xbd93f9, 0xff79c6, 0x993333,
        ],
        [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("Red"),
            Some("Orange"),
            Some("Yellow"),
            Some("Green"),
            Some("Aqua / Cyan"),
            Some("Blue"),
            Some("Purple / Magenta"),
            Some("Dark Red or Brown"),
        ],
    )
}

/// `base16/gruvbox-dark-medium.yaml`. A third naming style again — its ramp is annotated
/// `----` to `++++`, i.e. relative position and nothing else.
pub fn gruvbox_dark_medium() -> Scheme {
    scheme(
        "Gruvbox dark, medium",
        "Dawid Kurek, morhetz",
        Variant::Dark,
        [
            0x282828, 0x3c3836, 0x504945, 0x665c54, 0xbdae93, 0xd5c4a1, 0xebdbb2, 0xfbf1c7,
            0xfb4934, 0xfe8019, 0xfabd2f, 0xb8bb26, 0x8ec07c, 0x83a598, 0xd3869b, 0xd65d0e,
        ],
        [
            Some("----"),
            Some("---"),
            Some("--"),
            Some("-"),
            Some("+"),
            Some("++"),
            Some("+++"),
            Some("++++"),
            Some("red"),
            Some("orange"),
            Some("yellow"),
            Some("green"),
            Some("aqua/cyan"),
            Some("blue"),
            Some("purple"),
            Some("brown"),
        ],
    )
}

/// `base16/solarized-light.yaml`. The light-polarity check, and the scheme that names
/// nothing — so the `scheme_name` column is empty for all sixteen. That is normal.
pub fn solarized_light() -> Scheme {
    scheme(
        "Solarized Light",
        "Ethan Schoonover (modified by aramisgithub)",
        Variant::Light,
        [
            0xfdf6e3, 0xeee8d5, 0x93a1a1, 0x839496, 0x657b83, 0x586e75, 0x073642, 0x002b36,
            0xdc322f, 0xcb4b16, 0xb58900, 0x859900, 0x2aa198, 0x268bd2, 0x6c71c4, 0xd33682,
        ],
        [None; 16],
    )
}

pub fn all() -> Vec<Scheme> {
    vec![
        catppuccin_mocha(),
        catppuccin_latte(),
        dracula(),
        gruvbox_dark_medium(),
        solarized_light(),
    ]
}

/// Relative luminance, for the two checks below. Rec. 601 rather than WCAG's linearised
/// form: this is used to compare rungs of one ramp against each other, never to make a
/// contrast-ratio claim — that would need the WCAG formula and nothing here performs one.
pub fn luma(colour: Color) -> f32 {
    match colour {
        Color::Rgb(r, g, b) => 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32,
        _ => f32::NAN,
    }
}

impl Scheme {
    /// Whether the declared `variant` agrees with the ramp's own direction.
    ///
    /// The spec ties the two together — *"to create a dark theme, colours base00 to base07
    /// should span from dark to light; for a light theme, from light to dark"* — so the
    /// declaration is checkable rather than merely trusted. This is the same move the wash
    /// made in §9: a polarity read off the artifact, not asserted about it.
    pub fn declared_variant_matches_ramp(&self) -> bool {
        let rising = luma(self.slots[5].colour) > luma(self.slots[0].colour);
        match self.variant {
            Variant::Dark => rising,
            Variant::Light => !rising,
        }
    }

    /// How far a colour sits from `base00`, in the direction this scheme's ramp runs.
    ///
    /// Distance rather than "lighter", for the reason §9's wash established: on a light
    /// theme "lighter" inverts, while "further from the base" is the same claim on both
    /// polarities. Every check below is phrased in it.
    fn depth(&self, colour: Color) -> f32 {
        (luma(colour) - luma(self.slots[0].colour)).abs()
    }

    /// How many of `base06`/`base07` sit beyond `base05` — the headroom that would let
    /// `tokens::strong()` stop extrapolating (defect §8.5).
    ///
    /// **Measured, not assumed, and it is never 2 across the board.** Dracula's `base06`
    /// equals its `base05` (the YAML carries a `TODO review` beside it); Catppuccin's
    /// `base07` is *lavender*, an accent, which lands BELOW its `text`. So a bundled theme
    /// usually offers one rung of headroom and sometimes two — `strong()` keeps its
    /// extrapolating fallback for the schemes that offer none.
    pub fn headroom_rungs(&self) -> usize {
        let fg = self.depth(self.slots[5].colour);
        [6, 7]
            .into_iter()
            .filter(|i| self.depth(self.slots[*i].colour) > fg)
            .count()
    }

    /// Whether `base01` — which the spec calls *"Lighter Background (Used for status
    /// bars)"* — actually stands off the base at all.
    ///
    /// **It does not, in half of a four-scheme sample.** Catppuccin's `base01` is `mantle`
    /// and Dracula's is named "Darker Background" outright; both sit *under* `base00`. §6
    /// requires layer 1 to read as floating ABOVE layer 0, so mapping layer 1 to `base01`
    /// inverts the depth model on exactly those themes. `base02` (Selection Background)
    /// stands off the base in all four, which is why the layer fill should be **chosen by
    /// measured depth rather than by slot number**.
    pub fn lighter_background_is_actually_lighter(&self) -> bool {
        self.depth(self.slots[1].colour) > 0.0
            && luma(self.slots[1].colour) != luma(self.slots[0].colour)
            && (luma(self.slots[1].colour) > luma(self.slots[0].colour))
                == (self.variant == Variant::Dark)
    }

    /// The slot this scheme should supply layer 1 from: whichever of `base01`/`base02`
    /// stands furthest off the base. Never `base01` by assumption — see above.
    pub fn layer1_slot(&self) -> &Slot {
        if self.depth(self.slots[2].colour) > self.depth(self.slots[1].colour) {
            &self.slots[2]
        } else {
            &self.slots[1]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scheme_declares_a_variant_its_own_ramp_agrees_with() {
        // The declaration is data from a third party, so it is checked rather than trusted.
        for scheme in all() {
            assert!(
                scheme.declared_variant_matches_ramp(),
                "{} declares {:?} but its ramp runs the other way",
                scheme.name,
                scheme.variant
            );
        }
    }

    #[test]
    fn headroom_above_the_foreground_exists_but_never_reliably() {
        // §8.5 is unfixable under Theme/Indexed: nothing brighter than slot 15 exists to
        // reach for. base16 LOOKS like the fix — base06/base07 are described as beyond
        // base05 — and the measurement says "usually one rung, sometimes two, never
        // guaranteed". So strong() keeps its extrapolating fallback.
        assert_eq!(gruvbox_dark_medium().headroom_rungs(), 2);
        assert_eq!(solarized_light().headroom_rungs(), 2);
        // rosewater is above `text`, lavender is below it.
        assert_eq!(catppuccin_mocha().headroom_rungs(), 1);
        // base06 == base05 exactly; only base07 (#ffffff) clears it.
        assert_eq!(dracula().headroom_rungs(), 1);
    }

    /// Chris, 20260731: the deeper ramp is deliberate — *"meant to prevent excessive
    /// contrast that is tiring for the eyes."* The ordering does not invert between
    /// flavours; the JOB does, and only a measurement against `text` shows it.
    #[test]
    fn catppuccins_deeper_ramp_adds_contrast_on_mocha_and_relieves_it_on_latte() {
        let gap = |s: &Scheme, i: usize| (luma(s.slots[i].colour) - luma(s.slots[5].colour)).abs();

        // Mocha: mantle sits further from `text` than `base` does — separation.
        let mocha = catppuccin_mocha();
        assert!(gap(&mocha, 1) > gap(&mocha, 0));

        // Latte: mantle sits CLOSER to `text` than `base` does — contrast relief.
        let latte = catppuccin_latte();
        assert!(gap(&latte, 1) < gap(&latte, 0));
    }

    #[test]
    fn the_slot_the_spec_calls_lighter_background_is_often_darker() {
        // The finding that decides how a theme feeds our layer model. Mapping layer 1 to
        // base01 would put the modal fill UNDER layer 0 on these two, which is precisely
        // the inversion §6 forbids.
        assert!(!catppuccin_mocha().lighter_background_is_actually_lighter());
        assert!(!dracula().lighter_background_is_actually_lighter());

        // …and the asymmetry worth keeping: Latte PASSES with the very same `mantle`.
        // Catppuccin's deeper ramp always runs base -> mantle -> crust, while base16's
        // ramp reverses with polarity — so on a light theme the two happen to agree and on
        // a dark theme they oppose. One scheme, one ramp, and the slot semantics fit it
        // exactly half the time. That is the argument for choosing by measured depth: a
        // rule that reads the artifact is right on both polarities without knowing this.
        assert!(catppuccin_latte().lighter_background_is_actually_lighter());
        assert!(gruvbox_dark_medium().lighter_background_is_actually_lighter());
        assert!(solarized_light().lighter_background_is_actually_lighter());
    }

    #[test]
    fn the_layer_one_fill_is_chosen_by_depth_and_never_by_slot_number() {
        // Whichever slot actually stands off the base wins, so the depth model survives a
        // scheme that orders its own ramp differently.
        assert_eq!(catppuccin_mocha().layer1_slot().slot, "base02");
        assert_eq!(catppuccin_latte().layer1_slot().slot, "base02");
        assert_eq!(dracula().layer1_slot().slot, "base02");
        for scheme in all() {
            let layer1 = scheme.depth(scheme.layer1_slot().colour);
            assert!(layer1 > 0.0, "{} has no slot standing off its base", scheme.name);
        }
    }

    #[test]
    fn a_slots_own_name_is_optional_but_its_positional_name_never_is() {
        // The answer to "are the colours always represented the same way across themes":
        // positionally yes, by the author's own vocabulary no.
        assert_eq!(catppuccin_mocha().slots[8].scheme_name, Some("red"));
        assert_eq!(solarized_light().slots[8].scheme_name, None);
        for scheme in all() {
            for (i, slot) in scheme.slots.iter().enumerate() {
                assert_eq!(slot.slot, SLOT_NAMES[i]);
            }
        }
    }
}
