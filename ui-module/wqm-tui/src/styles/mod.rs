//! The Styles tier — the vocabulary the design speaks, rendered live.
//!
//! §16 gives the surface four tiers, and three of them already had a home: [`crate::widgets`]
//! for the atoms and the chrome, `widgets` again for the zones, [`crate::views`] for the
//! screens. The fourth — Styles — had none, because until now its entries were **not code**:
//! `pantry.toml` declares `[colors.*]` and `[typography]`, and `tui-pantry` turns those tables
//! into swatch and text-sample ingredients on its own.
//!
//! # Why anything here is code at all
//!
//! Static TOML cannot call [`crate::tokens`], so every value in it is a **transcription**, and
//! a transcription is a claim that can drift from what the renderer paints. It drifts silently:
//! the swatch is still a colour, the row is still a row, and nothing fails. Two of Chris's
//! three requests of 20260801 could not be answered in TOML at all —
//!
//! - the theme's own ten fields are not fixed values, they are whichever theme is in force
//!   ([`palette`]);
//! - a typography entry carries `color` and `description` and **nothing else**, so the tab
//!   could not show weight ([`typography`]).
//!
//! — so they are rendered from `tokens` here instead. That removes the drift as a side effect
//! rather than as a goal: these two entries cannot disagree with a screen, because they ask the
//! same functions the screen asks.
//!
//! The `[colors.*]` groups stay in `pantry.toml`. They are the **written** vocabulary — the
//! names the design uses and the reason each exists — and a written vocabulary is worth having
//! even where it is pinned to one theme. What is measured lives here; what is authored lives
//! there, and the two are labelled so a reader knows which is which.
//!
//! # The instruments are here too, under a section
//!
//! [`palette_reference`] and [`theme_sheet`] declare `tab() = "Styles"` with
//! `section() = "Instruments"` — §16's own Styles row is *the vocabulary, plus by section the
//! instruments*, and this directory is that row. They are how *we* judge the language rather
//! than part of it, and the section is what keeps the two readings apart; [`palette`] and
//! [`typography`] carry no section, which is what puts them beside the TOML groups.

pub mod palette;
pub mod palette_reference;
pub mod theme_sheet;
pub mod typography;
