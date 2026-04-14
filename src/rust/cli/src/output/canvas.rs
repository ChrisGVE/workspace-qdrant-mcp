//! Output canvas: the general wrapper for all command output.
//!
//! Every command outputs a title, content module, and optional footnotes.
//! This module provides the `Canvas` builder for composing these elements
//! consistently.

use colored::Colorize;

use super::terminal_width;

/// Capitalize the first letter of each word in a string.
///
/// Treats both whitespace and underscores as word boundaries so that
/// `"grpc_server"` becomes `"Grpc Server"` rather than `"Grpc_server"`.
pub fn title_case(s: &str) -> String {
    s.split_whitespace()
        .map(|word| {
            word.split('_')
                .map(|part| {
                    let mut chars = part.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(first) => {
                            let upper: String = first.to_uppercase().collect();
                            format!("{upper}{}", chars.as_str())
                        }
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Print a command title (bold, title-cased, no leading blank line).
pub fn print_title(title: &str) {
    println!("{}", title_case(title).bold());
}

/// Print a horizontal separator line spanning the full terminal width.
pub fn print_separator() {
    let width = terminal_width();
    println!("{}", "─".repeat(width));
}

/// Print a dimmer separator (for section breaks within a display).
pub fn print_dim_separator() {
    let width = terminal_width();
    println!("{}", "─".repeat(width).dimmed());
}

/// Print a double separator (for closing metrics sections).
pub fn print_double_separator() {
    let width = terminal_width();
    println!("{}", "═".repeat(width));
}

/// Print a separator from column 1 to the given width (for columnar displays).
pub fn print_sized_separator(width: usize) {
    println!("{}", "─".repeat(width));
}

/// Print a dim separator of a given width.
pub fn print_sized_dim_separator(width: usize) {
    println!("{}", "─".repeat(width).dimmed());
}

/// Print a blank line (used between title and content).
pub fn print_blank() {
    println!();
}

/// Print a footnote line (dimmed, indented).
pub fn print_footnote(text: &str) {
    println!("  {}", text.dimmed());
}

/// Canvas builder for composing command output.
///
/// Usage:
/// ```ignore
/// Canvas::new("Project List")
///     .blank()
///     .body(|| { /* render table/columnar content */ })
///     .footnote("* Status may be stale if daemon is not running")
///     .render();
/// ```
pub struct Canvas {
    title: String,
    sections: Vec<CanvasSection>,
}

enum CanvasSection {
    Blank,
    Separator,
    DimSeparator,
    DoubleSeparator,
    Footnote(String),
    Custom(Box<dyn FnOnce()>),
}

impl Canvas {
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            sections: Vec::new(),
        }
    }

    /// Add a blank line.
    pub fn blank(mut self) -> Self {
        self.sections.push(CanvasSection::Blank);
        self
    }

    /// Add a full-width separator line.
    pub fn separator(mut self) -> Self {
        self.sections.push(CanvasSection::Separator);
        self
    }

    /// Add a dimmed separator line.
    pub fn dim_separator(mut self) -> Self {
        self.sections.push(CanvasSection::DimSeparator);
        self
    }

    /// Add a double separator line.
    pub fn double_separator(mut self) -> Self {
        self.sections.push(CanvasSection::DoubleSeparator);
        self
    }

    /// Add a footnote.
    pub fn footnote(mut self, text: &str) -> Self {
        self.sections
            .push(CanvasSection::Footnote(text.to_string()));
        self
    }

    /// Add a custom render function (table, columnar, etc.).
    pub fn body(mut self, f: impl FnOnce() + 'static) -> Self {
        self.sections.push(CanvasSection::Custom(Box::new(f)));
        self
    }

    /// Render all sections to stdout.
    pub fn render(self) {
        print_title(&self.title);
        for section in self.sections {
            match section {
                CanvasSection::Blank => print_blank(),
                CanvasSection::Separator => print_separator(),
                CanvasSection::DimSeparator => print_dim_separator(),
                CanvasSection::DoubleSeparator => print_double_separator(),
                CanvasSection::Footnote(text) => print_footnote(&text),
                CanvasSection::Custom(f) => f(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_case_basic() {
        assert_eq!(title_case("project list"), "Project List");
        assert_eq!(title_case("HELLO WORLD"), "HELLO WORLD");
        assert_eq!(title_case("a"), "A");
        assert_eq!(title_case(""), "");
    }

    #[test]
    fn title_case_handles_underscores() {
        assert_eq!(title_case("grpc_server"), "Grpc Server");
        assert_eq!(title_case("tenant_id column"), "Tenant Id Column");
        assert_eq!(title_case("a_b_c"), "A B C");
    }

    #[test]
    fn title_case_multiple_spaces() {
        assert_eq!(title_case("hello   world"), "Hello World");
    }

    #[test]
    fn canvas_renders_without_panic() {
        // Just verify it doesn't panic — output goes to stdout
        Canvas::new("Test Title")
            .blank()
            .body(|| {
                println!("test content");
            })
            .separator()
            .footnote("test footnote")
            .render();
    }
}
