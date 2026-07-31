//! The canonical-collection list — the Library tab's spine.
//!
//! Every name rendered here comes from N8 (`wqm_common::names`), never from a literal in
//! this crate. ADR-001 closes the set at four, and N8 owns how each is spelled, so the
//! storyboard cannot drift from the vocabulary the daemon and MCP surface actually key
//! on. `images` is drawn from the same registry as a reserved, not-yet-active name.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};
use wqm_common::names::{Collection, RESERVED_IMAGES_COLLECTION};

use crate::tokens;

/// How each collection is partitioned — the one piece of prose the UI adds on top of the
/// name. Kept beside the variant it describes so a new collection cannot be added without
/// answering the question.
const fn partition_hint(collection: Collection) -> &'static str {
    match collection {
        Collection::Projects => "by tenant_id",
        Collection::Libraries => "by library_name",
        Collection::Rules => "global",
        Collection::Scratchpad => "global",
    }
}

pub struct Collections {
    /// Index of the data cursor. §3: the cursor sits on row 1 of a list by default and is
    /// a subtle fill, never the selector's inverse block.
    cursor: Option<usize>,
    /// Whether the reserved `images` name is shown as an inactive future entry.
    show_reserved: bool,
}

impl Collections {
    pub fn new(cursor: Option<usize>) -> Self {
        Self {
            cursor,
            show_reserved: false,
        }
    }

    pub fn with_reserved(mut self) -> Self {
        self.show_reserved = true;
        self
    }
}

impl Widget for Collections {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let name_width = Collection::ALL
            .iter()
            .map(|c| c.name().len())
            .max()
            .unwrap_or(0)
            .max(RESERVED_IMAGES_COLLECTION.len());

        let mut lines: Vec<Line> = Collection::ALL
            .iter()
            .enumerate()
            .map(|(i, collection)| {
                let on_cursor = self.cursor == Some(i);

                // The marker column is always present so names stay aligned whether or
                // not the row carries the cursor.
                let mark = if on_cursor { "▸ " } else { "  " };

                let mut line = Line::from(vec![
                    Span::styled(
                        mark,
                        ratatui::style::Style::default().fg(tokens::cursor_mark()),
                    ),
                    Span::styled(
                        format!("{:<name_width$}  ", collection.name()),
                        if on_cursor {
                            tokens::normal_style()
                        } else {
                            tokens::muted_style()
                        },
                    ),
                    Span::styled(partition_hint(*collection), tokens::faint_style()),
                ]);

                if on_cursor {
                    line = line.style(ratatui::style::Style::default().bg(tokens::cursor_bg()));
                }
                line
            })
            .collect();

        if self.show_reserved {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("{:<name_width$}  ", RESERVED_IMAGES_COLLECTION),
                    tokens::faint_style(),
                ),
                Span::styled("reserved — not yet active", tokens::faint_style()),
            ]));
        }

        Paragraph::new(lines).render(area, buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "cursor",
            ty: "Option<usize>",
            description: "Data-cursor row; a subtle fill plus ▸, never the selector block",
        },
        PropInfo {
            name: "show_reserved",
            ty: "bool",
            description: "Append N8's reserved `images` name as an inactive entry",
        },
    ];

    macro_rules! variant {
        ($ty:ident, $name:literal, $desc:literal, $build:expr) => {
            struct $ty;
            impl Ingredient for $ty {
                fn tab(&self) -> &str {
                    "Panes"
                }
                fn group(&self) -> &str {
                    "Collections"
                }
                fn name(&self) -> &str {
                    $name
                }
                fn source(&self) -> &str {
                    "wqm_tui::widgets::collections"
                }
                fn description(&self) -> &str {
                    $desc
                }
                fn props(&self) -> &[PropInfo] {
                    PROPS
                }
                fn render(&self, area: Rect, buf: &mut Buffer) {
                    let w: Collections = $build;
                    w.render(area, buf);
                }
            }
        };
    }

    variant!(
        Default_,
        "Default",
        "The four canonical collections, names straight from N8, cursor on row 1 per §3",
        Collections::new(Some(0))
    );

    variant!(
        NoCursor,
        "No Cursor",
        "The unfocused zone: no row carries the cursor, so nothing competes for the eye",
        Collections::new(None)
    );

    variant!(
        WithReserved,
        "With Reserved",
        "`images` shown as claimed-but-inactive — does faint read as 'not yet' or as 'broken'?",
        Collections::new(Some(1)).with_reserved()
    );

    /// The palette trade-off as a pair of frames rather than an argument. Both render the
    /// identical widget; only the neutral source differs, so any difference you see IS the
    /// cost of sourcing greys from the theme's four slots.
    struct PaletteVariant(tokens::Palette, &'static str, &'static str);

    impl Ingredient for PaletteVariant {
        fn tab(&self) -> &str {
            "Panes"
        }
        fn group(&self) -> &str {
            "Collections"
        }
        fn name(&self) -> &str {
            self.1
        }
        fn source(&self) -> &str {
            "wqm_tui::tokens::Palette"
        }
        fn description(&self) -> &str {
            self.2
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let previous = tokens::Palette::current();
            tokens::Palette::set(self.0);
            Collections::new(Some(0)).with_reserved().render(area, buf);
            tokens::Palette::set(previous);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Default_),
            Box::new(NoCursor),
            Box::new(WithReserved),
            Box::new(PaletteVariant(
                tokens::Palette::Theme,
                "Palette: Theme",
                "Neutrals from the 16 theme slots only — four available greys for eleven specified rungs",
            )),
            Box::new(PaletteVariant(
                tokens::Palette::Indexed,
                "Palette: Indexed",
                "Neutrals from the xterm 232–255 ramp — every rung survives, but greys stop following the theme",
            )),
            Box::new(PaletteVariant(
                tokens::Palette::Derived,
                "Palette: Derived",
                "Neutrals interpolated between the terminal's own background and foreground — keeps the rungs AND the theme's hue, at the cost of emitting RGB",
            )),
        ]
    }
}
