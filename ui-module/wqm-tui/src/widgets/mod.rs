//! Storyboard widgets. Each module owns one element of the visual language and, behind
//! the `tui-pantry` feature, the preview variants that let it be judged in isolation.
//!
//! Where a widget depicts something the system already has a contract for, it is typed on
//! that contract rather than on a local mock: [`collections`] on N8's name registry,
//! [`daemon_status`] on N49's client seam, [`envelope`] on N12's response envelope. A
//! frame that cannot be built from the real type is a frame depicting a state the system
//! cannot produce.
//!
//! [`chrome`] is the exception to "a widget is a zone's content": §16's composition model
//! (Chris, 20260731) counts the screen furniture — title bar, status line, rules, selectors,
//! headings — as widgets too, and it lives there as one family.

pub mod chrome;
pub mod collections;
pub mod config_table;
pub mod daemon_status;
pub mod envelope;
pub mod modal;
pub mod palette_reference;
pub mod store_health;
pub mod surface;
pub mod tab_bar;
pub mod theme_sheet;
pub mod toast;

/// The tier every module claims, checked against §16 — crate-wide, not only this directory.
///
/// It lives here because this is where the departures started; [`crate::styles`] and
/// [`crate::views`] are checked from the same test on purpose. One list of exceptions is
/// auditable, three lists in three modules are three chances to forget one.
///
/// `Ingredient::tab()` defaults to `"Widgets"`, so a module that belongs anywhere else is one
/// forgotten method away from silently landing in the wrong tab and *looking like* a widget.
/// There is no error to see and no frame to judge — the entry is simply filed under the wrong
/// idea of what it is. The default tier needs no test; every departure from it does.
#[cfg(all(test, feature = "tui-pantry"))]
mod tier {
    use tui_pantry::Ingredient;

    fn tiers(ingredients: Vec<Box<dyn Ingredient>>) -> Vec<(String, Option<String>)> {
        ingredients
            .iter()
            .map(|i| (i.tab().to_string(), i.section().map(str::to_string)))
            .collect()
    }

    fn assert_every(
        ingredients: Vec<Box<dyn Ingredient>>,
        tab: &str,
        section: Option<&str>,
        why: &str,
    ) {
        let want = (tab.to_string(), section.map(str::to_string));
        let got = tiers(ingredients);
        assert!(!got.is_empty(), "a module with no ingredients is unbrowsable");
        assert!(got.iter().all(|t| *t == want), "{why}: {got:?} != {want:?}");
    }

    #[test]
    fn a_zone_is_a_pane_and_an_instrument_is_not_product() {
        assert_every(
            super::collections::ingredient::ingredients(),
            "Panes",
            None,
            "§16 lists the collections list among the PANES — a list that fills a tab's body is a zone",
        );

        // §16: the instruments are how *we* judge the language, not part of the surface, and
        // the Styles tab is where the vocabulary already lives.
        for module in [
            super::palette_reference::ingredient::ingredients(),
            super::theme_sheet::ingredient::ingredients(),
        ] {
            assert_every(
                module,
                "Styles",
                Some("Instruments"),
                "an instrument is sectioned off the product tabs",
            );
        }

        // §17.3: the live vocabulary is Styles with NO section. The section is what separates
        // how we judge the language from the language itself, so an unsectioned Styles entry
        // sits beside the `[colors.*]` groups the stylesheet contributes — which is where a
        // reader looking up "what does the design call this" already is.
        assert_every(
            crate::styles::palette::ingredient::ingredients(),
            "Styles",
            None,
            "the vocabulary is Styles and is not an instrument",
        );

        assert_every(
            crate::views::service::ingredient::ingredients(),
            "Views",
            None,
            "a full screen is a view",
        );
    }
}
