//! File type classification for metadata-based routing
//!
//! This module provides file type classification to replace collection-based routing
//! with metadata-based differentiation. All extension mappings are derived from the
//! unified YAML reference in `wqm_common::classification`.
//!
//! File Types:
//! - code: Source code files (.py, .rs, .js, etc.)
//! - text: Plain text and lightweight markup (.txt, .md, .rst, .org, .adoc, .tex)
//! - docs: Binary/rich document formats (.pdf, .docx, .odt, .epub, .rtf)
//! - web: Web content files (.html, .css, .scss, .xml)
//! - slides: Presentation formats (.ppt, .pptx, .key, .odp)
//! - config: Configuration files (.yaml, .json, .toml, .ini)
//! - data: Data files (.csv, .parquet, .xlsx, .ipynb)
//! - build: Build artifacts (.whl, .tar.gz, .zip, .jar)
//! - other: Unclassified files
//!
//! Test detection is separate: `is_test_file()` returns a bool independent of file_type.
//! A file can be both `code` and a test file.

mod classify;
mod test_detection;
mod types;

pub use classify::{classify_file_type, get_extension_for_storage};
pub use test_detection::{is_test_directory, is_test_file};
pub use types::FileType;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    use classify::get_extension;

    #[test]
    fn test_code_files() {
        assert_eq!(classify_file_type(Path::new("main.py")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("lib.rs")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("app.js")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("component.tsx")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("handler.go")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("script.ps1")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("module.d")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("app.vue")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("page.svelte")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("layout.astro")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("schema.proto")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("main.zig")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("app.dart")), FileType::Code);
    }

    #[test]
    fn test_text_files() {
        assert_eq!(classify_file_type(Path::new("README.md")), FileType::Text);
        assert_eq!(classify_file_type(Path::new("guide.rst")), FileType::Text);
        assert_eq!(classify_file_type(Path::new("notes.txt")), FileType::Text);
        assert_eq!(classify_file_type(Path::new("doc.adoc")), FileType::Text);
        assert_eq!(classify_file_type(Path::new("notes.org")), FileType::Text);
        assert_eq!(classify_file_type(Path::new("paper.tex")), FileType::Text);
    }

    #[test]
    fn test_docs_files() {
        assert_eq!(classify_file_type(Path::new("manual.pdf")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("book.epub")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("report.docx")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("legacy.doc")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("document.odt")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("formatted.rtf")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("notes.pages")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("book.mobi")), FileType::Docs);
    }

    #[test]
    fn test_web_files() {
        assert_eq!(classify_file_type(Path::new("index.html")), FileType::Web);
        assert_eq!(classify_file_type(Path::new("page.htm")), FileType::Web);
        assert_eq!(classify_file_type(Path::new("doc.xhtml")), FileType::Web);
        assert_eq!(classify_file_type(Path::new("styles.css")), FileType::Web);
        assert_eq!(classify_file_type(Path::new("styles.scss")), FileType::Web);
        assert_eq!(classify_file_type(Path::new("styles.less")), FileType::Web);
        assert_eq!(classify_file_type(Path::new("data.xml")), FileType::Web);
    }

    #[test]
    fn test_slides_files() {
        assert_eq!(classify_file_type(Path::new("deck.pptx")), FileType::Slides);
        assert_eq!(classify_file_type(Path::new("legacy.ppt")), FileType::Slides);
        assert_eq!(classify_file_type(Path::new("presentation.key")), FileType::Slides);
        assert_eq!(classify_file_type(Path::new("slides.odp")), FileType::Slides);
    }

    #[test]
    fn test_config_files() {
        assert_eq!(classify_file_type(Path::new("config.yaml")), FileType::Config);
        assert_eq!(classify_file_type(Path::new("settings.toml")), FileType::Config);
        assert_eq!(classify_file_type(Path::new(".env")), FileType::Config);
        assert_eq!(classify_file_type(Path::new("app.ini")), FileType::Config);
    }

    #[test]
    fn test_json_xml_context_aware() {
        // JSON in config location → config
        let config_json = PathBuf::from("/project/config/app.json");
        assert_eq!(classify_file_type(&config_json), FileType::Config);

        // JSON in data location → data
        let data_json = PathBuf::from("/project/data/records.json");
        assert_eq!(classify_file_type(&data_json), FileType::Data);

        // XML → always web (moved from config-dependent)
        let xml = PathBuf::from("/project/exports/data.xml");
        assert_eq!(classify_file_type(&xml), FileType::Web);
    }

    #[test]
    fn test_data_files() {
        assert_eq!(classify_file_type(Path::new("data.csv")), FileType::Data);
        assert_eq!(classify_file_type(Path::new("export.parquet")), FileType::Data);
        assert_eq!(classify_file_type(Path::new("db.sqlite")), FileType::Data);
        assert_eq!(classify_file_type(Path::new("array.npy")), FileType::Data);
    }

    #[test]
    fn test_build_files() {
        assert_eq!(classify_file_type(Path::new("package.whl")), FileType::Build);
        assert_eq!(classify_file_type(Path::new("app.zip")), FileType::Build);
        assert_eq!(classify_file_type(Path::new("lib.so")), FileType::Build);
        assert_eq!(classify_file_type(Path::new("archive.tar.gz")), FileType::Build);
        assert_eq!(classify_file_type(Path::new("backup.tar.bz2")), FileType::Build);
    }

    #[test]
    fn test_other_files() {
        assert_eq!(classify_file_type(Path::new("unknown.xyz")), FileType::Other);
        assert_eq!(classify_file_type(Path::new("data")), FileType::Other);
    }

    // --- Test file detection (separate from file_type) ---

    #[test]
    fn test_is_test_file_by_filename() {
        assert!(is_test_file(Path::new("test_auth.py")));
        assert!(is_test_file(Path::new("main_test.go")));
        assert!(is_test_file(Path::new("app.test.js")));
        assert!(is_test_file(Path::new("component.spec.ts")));
        assert!(is_test_file(Path::new("conftest.py")));
        assert!(is_test_file(Path::new("test_utils.rs")));
    }

    #[test]
    fn test_is_test_file_non_code_excluded() {
        // Non-code files should NOT be classified as test even with test patterns
        assert!(!is_test_file(Path::new("test_data.txt")));
        assert!(!is_test_file(Path::new("test_fixture.json")));
        assert!(!is_test_file(Path::new("test_input.md")));
        assert!(!is_test_file(Path::new("test_config.yaml")));
    }

    #[test]
    fn test_is_test_file_by_directory() {
        assert!(is_test_file(Path::new("/project/tests/helper.py")));
        assert!(is_test_file(Path::new("/project/__tests__/utils.js")));
        assert!(is_test_file(Path::new("/project/spec/models.rb")));

        // Non-code files in test dirs are NOT tests
        assert!(!is_test_file(Path::new("/project/tests/fixture.txt")));
        assert!(!is_test_file(Path::new("/project/__tests__/mock_data.json")));
    }

    #[test]
    fn test_is_test_file_regular_code_not_test() {
        assert!(!is_test_file(Path::new("main.py")));
        assert!(!is_test_file(Path::new("utils.rs")));
        assert!(!is_test_file(Path::new("index.js")));
    }

    #[test]
    fn test_test_files_are_still_code() {
        // A test file should be classified as "code", not "test"
        assert_eq!(classify_file_type(Path::new("test_main.py")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("app.test.js")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("component.spec.ts")), FileType::Code);

        // But they should be detected by is_test_file
        assert!(is_test_file(Path::new("test_main.py")));
        assert!(is_test_file(Path::new("app.test.js")));
        assert!(is_test_file(Path::new("component.spec.ts")));
    }

    // --- Extension extraction ---

    #[test]
    fn test_get_extension_simple() {
        assert_eq!(get_extension(Path::new("main.py")), ".py");
        assert_eq!(get_extension(Path::new("lib.rs")), ".rs");
        assert_eq!(get_extension(Path::new("FILE.HTML")), ".html");
    }

    #[test]
    fn test_get_extension_compound() {
        assert_eq!(get_extension(Path::new("types.d.ts")), ".d.ts");
        assert_eq!(get_extension(Path::new("global.d.mts")), ".d.mts");
        assert_eq!(get_extension(Path::new("index.d.cts")), ".d.cts");
    }

    #[test]
    fn test_get_extension_for_storage() {
        assert_eq!(
            get_extension_for_storage(Path::new("main.py")),
            Some("py".to_string())
        );
        assert_eq!(
            get_extension_for_storage(Path::new("types.d.ts")),
            Some("d.ts".to_string())
        );
        assert_eq!(get_extension_for_storage(Path::new("noext")), None);
    }

    // --- Other ---

    #[test]
    fn test_test_directory_detection() {
        assert!(is_test_directory(Path::new("/project/tests")));
        assert!(is_test_directory(Path::new("/project/test")));
        assert!(is_test_directory(Path::new("/project/__tests__")));
        assert!(is_test_directory(Path::new("/project/spec")));
        assert!(is_test_directory(Path::new("/project/e2e")));
        assert!(is_test_directory(Path::new("/project/integration")));

        assert!(!is_test_directory(Path::new("/project/src")));
        assert!(!is_test_directory(Path::new("/project/lib")));
    }

    #[test]
    fn test_file_type_as_str() {
        assert_eq!(FileType::Code.as_str(), "code");
        assert_eq!(FileType::Text.as_str(), "text");
        assert_eq!(FileType::Docs.as_str(), "docs");
        assert_eq!(FileType::Web.as_str(), "web");
        assert_eq!(FileType::Slides.as_str(), "slides");
        assert_eq!(FileType::Config.as_str(), "config");
        assert_eq!(FileType::Data.as_str(), "data");
        assert_eq!(FileType::Build.as_str(), "build");
        assert_eq!(FileType::Other.as_str(), "other");
    }

    #[test]
    fn test_xml_is_web_not_config() {
        // XML is now always classified as web, not config
        assert_eq!(classify_file_type(Path::new("data.xml")), FileType::Web);
        // Even in config location, XML → web (it's a markup language)
        assert_eq!(
            classify_file_type(&PathBuf::from("/project/.github/workflow.xml")),
            FileType::Web
        );
    }

    #[test]
    fn test_compound_extension_classification() {
        // .d.ts files should be classified as code (TypeScript declarations)
        assert_eq!(classify_file_type(Path::new("types.d.ts")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("global.d.mts")), FileType::Code);
    }
}
