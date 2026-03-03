/// Core types for image extraction: `ImageFormat` and `EmbeddedImage`.

/// Detected image format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Tiff,
    Bmp,
    Gif,
    Unknown,
}

impl ImageFormat {
    /// Detect format from magic bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        if data.len() < 4 {
            return Self::Unknown;
        }
        match &data[..4] {
            [0xFF, 0xD8, 0xFF, ..] => Self::Jpeg,
            [0x89, 0x50, 0x4E, 0x47] => Self::Png,
            [0x49, 0x49, 0x2A, 0x00] | [0x4D, 0x4D, 0x00, 0x2A] => Self::Tiff,
            [0x42, 0x4D, ..] => Self::Bmp,
            [0x47, 0x49, 0x46, 0x38] => Self::Gif,
            _ => Self::Unknown,
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Png => "png",
            Self::Tiff => "tiff",
            Self::Bmp => "bmp",
            Self::Gif => "gif",
            Self::Unknown => "bin",
        }
    }
}

/// An image extracted from a document.
#[derive(Debug, Clone)]
pub struct EmbeddedImage {
    /// Raw image bytes.
    pub bytes: Vec<u8>,
    /// Detected image format.
    pub format: ImageFormat,
    /// Image width (if known from metadata).
    pub width: Option<u32>,
    /// Image height (if known from metadata).
    pub height: Option<u32>,
    /// Page number (1-based, for page-based formats like PDF).
    pub page_number: Option<u32>,
    /// Position index within the page/section (0-based).
    pub position_in_page: usize,
}

/// Minimum image dimension to extract (skip tiny icons/decorations).
pub const MIN_IMAGE_DIMENSION: u32 = 100;
