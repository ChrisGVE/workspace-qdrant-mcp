//! Columnar (key-value) display template.
//!
//! Renders key-value pairs with bold keys, aligned values, optional gutter
//! indicators, nested dictionaries, and section separators.

use colored::Colorize;
use unicode_width::UnicodeWidthStr;

use super::canvas::{print_sized_dim_separator, print_sized_separator, title_case};
use super::gutter::Gutter;

/// A single entry in a columnar display.
pub enum ColumnarEntry {
    /// A key-value pair with optional gutter indicator.
    KeyValue {
        key: String,
        value: String,
        gutter: Gutter,
    },
    /// A section separator with optional header.
    Section { header: Option<String> },
    /// A nested group of key-value pairs (indented one level deeper).
    Nested {
        key: String,
        entries: Vec<ColumnarEntry>,
    },
    /// A raw pre-formatted line (for lists of items, etc.).
    Raw { text: String, gutter: Gutter },
}

/// Builder for columnar displays.
pub struct ColumnarBuilder {
    entries: Vec<ColumnarEntry>,
    indent: usize,
}

impl Default for ColumnarBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ColumnarBuilder {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            indent: 4,
        }
    }

    /// Set the indentation width for nested entries.
    pub fn indent(mut self, spaces: usize) -> Self {
        self.indent = spaces;
        self
    }

    /// Add a key-value pair.
    pub fn kv(mut self, key: &str, value: impl std::fmt::Display) -> Self {
        self.entries.push(ColumnarEntry::KeyValue {
            key: key.to_string(),
            value: value.to_string(),
            gutter: Gutter::None,
        });
        self
    }

    /// Add a key-value pair with a gutter indicator.
    pub fn kv_gutter(mut self, key: &str, value: impl std::fmt::Display, gutter: Gutter) -> Self {
        self.entries.push(ColumnarEntry::KeyValue {
            key: key.to_string(),
            value: value.to_string(),
            gutter,
        });
        self
    }

    /// Add a section separator with optional header.
    pub fn section(mut self, header: Option<&str>) -> Self {
        self.entries.push(ColumnarEntry::Section {
            header: header.map(|s| s.to_string()),
        });
        self
    }

    /// Add a nested group.
    pub fn nested(mut self, key: &str, builder: ColumnarBuilder) -> Self {
        self.entries.push(ColumnarEntry::Nested {
            key: key.to_string(),
            entries: builder.entries,
        });
        self
    }

    /// Add a raw pre-formatted line with optional gutter.
    pub fn raw(mut self, text: &str, gutter: Gutter) -> Self {
        self.entries.push(ColumnarEntry::Raw {
            text: text.to_string(),
            gutter,
        });
        self
    }

    /// Add a group of key-value pairs with right-aligned values.
    ///
    /// Used for decompositions (e.g., queue breakdown) where numbers
    /// should align vertically. Values are pre-padded so the widest
    /// value determines alignment for the group.
    pub fn aligned_group(mut self, entries: Vec<(&str, String, Gutter)>) -> Self {
        let max_width = entries
            .iter()
            .map(|(_, v, _)| v.chars().count())
            .max()
            .unwrap_or(0);
        for (key, value, gutter) in entries {
            let val_width = value.chars().count();
            let padding = max_width.saturating_sub(val_width);
            let padded = format!("{}{}", " ".repeat(padding), value);
            self.entries.push(ColumnarEntry::KeyValue {
                key: key.to_string(),
                value: padded,
                gutter,
            });
        }
        self
    }

    /// Render the columnar display to stdout.
    pub fn render(self) {
        let max_key_width = self.compute_max_key_width(&self.entries, 0);
        let content_width = self.compute_content_width(&self.entries, 0, max_key_width);

        // Opening line
        print_sized_separator(content_width);

        self.render_entries(&self.entries, 0, max_key_width, content_width);

        // Closing line
        print_sized_separator(content_width);
    }

    /// Compute the maximum key width across all entries at a given depth.
    ///
    /// Tracks section context: entries after a `Section` are at depth+1.
    fn compute_max_key_width(&self, entries: &[ColumnarEntry], depth: usize) -> usize {
        let mut effective_depth = depth;
        let mut max_w = 0usize;

        for entry in entries {
            match entry {
                ColumnarEntry::Section { .. } => {
                    effective_depth = depth + 1;
                }
                ColumnarEntry::KeyValue { key, .. } => {
                    let base_indent = Gutter::WIDTH + effective_depth * self.indent;
                    let w = base_indent + UnicodeWidthStr::width(key.as_str()) + 1; // +1 for colon
                    max_w = max_w.max(w);
                }
                ColumnarEntry::Nested { key, entries } => {
                    let base_indent = Gutter::WIDTH + effective_depth * self.indent;
                    let w = base_indent + UnicodeWidthStr::width(key.as_str()) + 1;
                    max_w = max_w.max(w);
                    let nested_w = self.compute_max_key_width(entries, effective_depth + 1);
                    max_w = max_w.max(nested_w);
                }
                _ => {}
            }
        }
        max_w
    }

    /// Compute the total content width (for separator lines).
    ///
    /// Tracks section context: entries after a `Section` are at depth+1.
    fn compute_content_width(
        &self,
        entries: &[ColumnarEntry],
        depth: usize,
        max_key_width: usize,
    ) -> usize {
        let mut effective_depth = depth;
        let mut max_w = max_key_width;

        for entry in entries {
            match entry {
                ColumnarEntry::Section { .. } => {
                    effective_depth = depth + 1;
                }
                ColumnarEntry::KeyValue { value, .. } => {
                    let w = max_key_width + 1 + UnicodeWidthStr::width(value.as_str()); // +1 for space after colon
                    max_w = max_w.max(w);
                }
                ColumnarEntry::Nested { entries, .. } => {
                    let nested_w =
                        self.compute_content_width(entries, effective_depth + 1, max_key_width);
                    max_w = max_w.max(nested_w);
                }
                ColumnarEntry::Raw { text, .. } => {
                    let w = Gutter::WIDTH
                        + effective_depth * self.indent
                        + UnicodeWidthStr::width(text.as_str());
                    max_w = max_w.max(w);
                }
            }
        }
        max_w
    }

    fn render_entries(
        &self,
        entries: &[ColumnarEntry],
        base_depth: usize,
        max_key_width: usize,
        content_width: usize,
    ) {
        let mut effective_depth = base_depth;

        for entry in entries {
            match entry {
                ColumnarEntry::KeyValue { key, value, gutter } => {
                    let base_indent = effective_depth * self.indent;
                    let gutter_str = gutter.colored();
                    let indent_str = " ".repeat(base_indent);
                    let titled_key = title_case(key);
                    let key_with_colon = format!("{titled_key}:");
                    let key_display_width = Gutter::WIDTH
                        + base_indent
                        + UnicodeWidthStr::width(key_with_colon.as_str());
                    let padding = if max_key_width > key_display_width {
                        max_key_width - key_display_width
                    } else {
                        0
                    };
                    println!(
                        "{gutter_str} {indent_str}{}{} {value}",
                        key_with_colon.bold(),
                        " ".repeat(padding),
                    );
                }
                ColumnarEntry::Section { header } => {
                    print_sized_dim_separator(content_width);
                    if let Some(h) = header {
                        let indent_str = " ".repeat(Gutter::WIDTH);
                        println!("{indent_str}{}", title_case(h).bold());
                    }
                    effective_depth = base_depth + 1;
                }
                ColumnarEntry::Nested { key, entries } => {
                    if key.is_empty() {
                        // Anonymous nested group — skip key line, just indent children
                        self.render_entries(
                            entries,
                            effective_depth + 1,
                            max_key_width,
                            content_width,
                        );
                    } else {
                        let base_indent = effective_depth * self.indent;
                        let gutter_str = Gutter::None.colored();
                        let indent_str = " ".repeat(base_indent);
                        let titled_key = title_case(key);
                        println!("{gutter_str} {indent_str}{}:", titled_key.bold());
                        self.render_entries(
                            entries,
                            effective_depth + 1,
                            max_key_width,
                            content_width,
                        );
                    }
                }
                ColumnarEntry::Raw { text, gutter } => {
                    let base_indent = effective_depth * self.indent;
                    let gutter_str = gutter.colored();
                    let indent_str = " ".repeat(base_indent);
                    println!("{gutter_str} {indent_str}{text}");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_renders_without_panic() {
        ColumnarBuilder::new()
            .kv("Project Name", "workspace-qdrant-mcp")
            .kv("Project Id", "4ed81466dec7")
            .kv("Status", "Active")
            .section(Some("Database Status"))
            .kv_gutter("Files in Sync", "42", Gutter::Sync)
            .kv_gutter("Files to Add", "3", Gutter::Add)
            .render();
    }

    #[test]
    fn nested_builder_renders_without_panic() {
        let inner = ColumnarBuilder::new()
            .kv("Branch", "main")
            .kv("Commit", "abc1234");

        ColumnarBuilder::new()
            .kv("Project Name", "test")
            .nested("Git Info", inner)
            .render();
    }

    #[test]
    fn empty_builder_renders_without_panic() {
        ColumnarBuilder::new().render();
    }

    #[test]
    fn raw_entries_render() {
        ColumnarBuilder::new()
            .raw("src/main.rs", Gutter::Add)
            .raw("src/lib.rs", Gutter::Update)
            .raw("src/old.rs", Gutter::Remove)
            .render();
    }
}
