//! Custom tabled width peakers for smart column layout.
//!
//! Provides two `Peaker` implementations used with `Width::wrap` and
//! `Width::increase` to distribute terminal width intelligently across
//! categorical and content columns.

use std::cell::RefCell;

use tabled::settings::peaker::Peaker;

// ─── Column layout hints ─────────────────────────────────────────────────

thread_local! {
    /// Content column indices for custom peakers.
    pub(super) static CONTENT_COLUMNS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    /// Content-aware minimum widths per column (computed from data).
    pub(super) static COLUMN_MIN_WIDTHS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Peaker for `Width::wrap`: shrinks categorical columns first.
///
/// When the table exceeds terminal width, this shrinks the widest
/// non-content (categorical) column first. Only falls back to shrinking
/// content columns when all categorical columns are at minimum width.
#[derive(Debug, Default, Clone)]
pub(super) struct ShrinkCategoricalFirst;

impl Peaker for ShrinkCategoricalFirst {
    fn create() -> Self {
        Self
    }

    fn peak(&mut self, _min_widths: &[usize], widths: &[usize]) -> Option<usize> {
        CONTENT_COLUMNS.with(|cc| {
            COLUMN_MIN_WIDTHS.with(|cmw| {
                let content_cols = cc.borrow();
                let col_mins = cmw.borrow();

                // Phase 1: shrink widest categorical column above its content minimum
                let cat = (0..widths.len())
                    .filter(|i| !content_cols.contains(i))
                    .filter(|&i| widths[i] > col_mins.get(i).copied().unwrap_or(0))
                    .max_by_key(|&i| widths[i]);
                if cat.is_some() {
                    return cat;
                }

                // Phase 2: shrink widest content column above its minimum
                let content = (0..widths.len())
                    .filter(|i| content_cols.contains(i))
                    .filter(|&i| widths[i] > col_mins.get(i).copied().unwrap_or(0))
                    .max_by_key(|&i| widths[i]);
                if content.is_some() {
                    return content;
                }

                // Phase 3: last resort — shrink any column above 1 char
                (0..widths.len())
                    .filter(|&i| widths[i] > 1)
                    .max_by_key(|&i| widths[i])
            })
        })
    }
}

/// Peaker for `Width::increase`: expands only content columns.
///
/// When the table is narrower than the terminal, this distributes extra
/// space exclusively to content columns, picking the narrowest one each
/// time for even distribution. Falls back to `PriorityMax` if no content
/// columns are configured.
#[derive(Debug, Default, Clone)]
pub(super) struct ExpandContentOnly;

impl Peaker for ExpandContentOnly {
    fn create() -> Self {
        Self
    }

    fn peak(&mut self, _min_widths: &[usize], widths: &[usize]) -> Option<usize> {
        CONTENT_COLUMNS.with(|cc| {
            let content_cols = cc.borrow();

            if content_cols.is_empty() {
                // No hints — fall back to widest column
                return (0..widths.len()).max_by_key(|&i| widths[i]);
            }

            // Expand the narrowest content column (for even distribution)
            content_cols
                .iter()
                .filter(|&&i| i < widths.len())
                .min_by_key(|&&i| widths[i])
                .copied()
        })
    }
}
