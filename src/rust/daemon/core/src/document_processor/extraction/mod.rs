//! Document content extraction for various file formats.

pub mod docx;
pub mod epub;
pub mod iwork;
pub mod jupyter;
pub mod opendocument;
pub mod pdf;
pub mod pptx;
pub mod rtf;
pub mod spreadsheet;
pub mod text;
pub mod xml_utils;

pub use self::docx::{count_docx_images, extract_docx, extract_text_from_docx_xml};
pub use self::epub::{count_epub_images, extract_epub};
pub use self::iwork::extract_iwork;
pub use self::jupyter::extract_jupyter;
pub use self::opendocument::extract_opendocument;
pub use self::pdf::{count_pdf_images, extract_pdf};
pub use self::pptx::extract_pptx;
pub use self::rtf::extract_rtf;
pub use self::spreadsheet::{extract_csv, extract_spreadsheet};
pub use self::text::{extract_code, extract_text_with_encoding};
pub use self::xml_utils::{clean_extracted_text, extract_text_from_xml_tags};
