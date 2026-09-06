//! wqm TUI storyboard widgets (P01-GT001).
//!
//! Every widget here renders against VISUAL-LANGUAGE.md r02. A widget that invents an
//! ad-hoc highlight is a defect — reach for [`tokens`] instead of a colour literal.
//!
//! The crate is a design instrument: it exists so storyboard frames are produced by the
//! real ratatui renderer rather than approximated, and so each frame can be browsed and
//! compared in `cargo pantry`.

#[cfg(feature = "png-capture")]
pub mod capture;
pub mod categorical;
pub mod encoding;
pub mod health;
pub mod names;
pub mod panes;
pub mod styles;
pub mod terminal;
pub mod tokens;
pub mod views;
pub mod widgets;

/// Every module's declared tier, checked against §16 and against the directory it lives in.
///
/// `Ingredient::tab()` defaults to `"Widgets"`, so a module that belongs anywhere else is one
/// forgotten method away from silently landing in the wrong tab and *looking like* an atom.
/// There is no error to see and no frame to judge — the entry is simply filed under the wrong
/// idea of what the thing is.
///
/// Since the directories were split by tier the two say the same thing twice, which is the
/// point: [`panes`] and [`styles`] are readable at a glance, and this is what stops the method
/// drifting away from the folder. One list of exceptions at the crate root rather than one per
/// directory, because three lists are three chances to forget one.
#[cfg(all(test, feature = "tui-pantry"))]
mod tier {
    use tui_pantry::Ingredient;

    fn assert_every(
        ingredients: Vec<Box<dyn Ingredient>>,
        tab: &str,
        section: Option<&str>,
        why: &str,
    ) {
        let want = (tab.to_string(), section.map(str::to_string));
        let got: Vec<(String, Option<String>)> = ingredients
            .iter()
            .map(|i| (i.tab().to_string(), i.section().map(str::to_string)))
            .collect();
        assert!(!got.is_empty(), "a module with no ingredients is unbrowsable");
        assert!(got.iter().all(|t| *t == want), "{why}: {got:?} != {want:?}");
    }

    #[test]
    fn every_module_declares_the_tier_its_directory_says_it_is() {
        // Panes — a zone of a screen. The collections list is one because it fills a tab's
        // body; the other two are the Service hub's own zones.
        for module in [
            crate::panes::collections::ingredient::ingredients(),
            crate::panes::config::ingredient::ingredients(),
            crate::panes::status_band::ingredient::ingredients(),
            crate::panes::status_block::ingredient::ingredients(),
            crate::panes::storage::ingredient::ingredients(),
        ] {
            assert_every(module, "Panes", None, "a zone of a screen is a pane");
        }

        // Styles, unsectioned — the vocabulary itself. `pantry.toml` no longer contributes any
        // colour groups (Chris, 20260801), so these two ARE the Styles tab's product half.
        for module in [
            crate::styles::categorical::ingredient::ingredients(),
            crate::styles::palette::ingredient::ingredients(),
            crate::styles::typography::ingredient::ingredients(),
        ] {
            assert_every(
                module,
                "Styles",
                None,
                "the vocabulary is Styles and is not an instrument",
            );
        }

        // Styles, sectioned — how *we* judge the language rather than part of it. The section
        // is the whole difference between the two Styles groups above and below.
        for module in [
            crate::styles::palette_reference::ingredient::ingredients(),
            crate::styles::theme_sheet::ingredient::ingredients(),
        ] {
            assert_every(
                module,
                "Styles",
                Some("Instruments"),
                "an instrument is sectioned off the product tabs",
            );
        }

        for module in [
            crate::views::service::ingredient::ingredients(),
            crate::views::shell::ingredient::ingredients(),
        ] {
            assert_every(module, "Views", None, "a full screen is a view");
        }
    }
}

/// Serialises every test that touches process-global rendering state.
///
/// The active palette ([`tokens::Palette`]) and the active encoding
/// ([`encoding::Encoding`]) are process-global by design — a widget deep in a render tree
/// must be able to ask what it is rendering into without being handed a context. The cost is
/// that a test which sets one of them, renders, and then asserts is racing every other test
/// that does the same, and cargo runs them in parallel by default.
///
/// One lock for the whole crate rather than one per module: two independent locks around the
/// same global is how a lock-ordering deadlock is built.
#[cfg(test)]
pub(crate) fn global_state_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A panicking test poisons the lock but leaves no invariant broken — every holder either
    // restores what it found or is followed by one that sets the value it needs. Recovering
    // keeps one failure from cascading into every later test as a poison error.
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
