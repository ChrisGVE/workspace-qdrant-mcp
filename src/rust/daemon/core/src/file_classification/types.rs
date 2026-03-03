//! `FileType` enum and its trait implementations.

/// File type classification result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileType {
    /// Source code files
    Code,
    /// Plain text and lightweight markup (.txt, .md, .rst, .org, .adoc, .tex)
    Text,
    /// Binary/rich document formats (.pdf, .docx, .epub, .odt, .rtf)
    Docs,
    /// Web content files (.html, .css, .xml)
    Web,
    /// Presentation formats (.ppt, .pptx, .key, .odp)
    Slides,
    /// Configuration files (.yaml, .json, .toml, .ini)
    Config,
    /// Data files (.csv, .parquet, .xlsx, .ipynb)
    Data,
    /// Build artifacts and build system files
    Build,
    /// Unclassified files
    Other,
}

impl FileType {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            FileType::Code => "code",
            FileType::Text => "text",
            FileType::Docs => "docs",
            FileType::Web => "web",
            FileType::Slides => "slides",
            FileType::Config => "config",
            FileType::Data => "data",
            FileType::Build => "build",
            FileType::Other => "other",
        }
    }

    /// Parse from string representation
    pub(super) fn from_str(s: &str) -> Option<Self> {
        match s {
            "code" => Some(FileType::Code),
            "text" => Some(FileType::Text),
            "docs" => Some(FileType::Docs),
            "web" => Some(FileType::Web),
            "slides" => Some(FileType::Slides),
            "config" => Some(FileType::Config),
            "data" => Some(FileType::Data),
            "build" => Some(FileType::Build),
            "other" => Some(FileType::Other),
            _ => None,
        }
    }
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
