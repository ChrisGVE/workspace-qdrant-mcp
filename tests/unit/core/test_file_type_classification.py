"""
Comprehensive unit tests for file type classification (Task 374.5).

Tests the file type classification system that enables metadata-based routing
instead of collection suffix-based routing. All files go to single _{project_id}
collection with file_type in metadata.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "python"))

from pathlib import Path

import pytest

from common.utils.file_type_classifier import (
    determine_file_type,
    is_test_directory,
    CODE_EXTENSIONS,
    DOCS_EXTENSIONS,
    CONFIG_EXTENSIONS,
    DATA_EXTENSIONS,
    BUILD_EXTENSIONS,
)


class TestCodeFileDetection:
    """Test code file classification."""

    def test_python_files(self):
        """Test Python file detection."""
        assert determine_file_type(Path("main.py")) == "code"
        assert determine_file_type(Path("lib.pyx")) == "code"
        assert determine_file_type(Path("types.pyi")) == "code"

    def test_rust_files(self):
        """Test Rust file detection."""
        assert determine_file_type(Path("main.rs")) == "code"
        assert determine_file_type(Path("lib.rs")) == "code"

    def test_javascript_files(self):
        """Test JavaScript file detection."""
        assert determine_file_type(Path("app.js")) == "code"
        assert determine_file_type(Path("component.jsx")) == "code"
        assert determine_file_type(Path("module.mjs")) == "code"
        assert determine_file_type(Path("module.cjs")) == "code"

    def test_typescript_files(self):
        """Test TypeScript file detection."""
        assert determine_file_type(Path("app.ts")) == "code"
        assert determine_file_type(Path("component.tsx")) == "code"
        assert determine_file_type(Path("types.d.ts")) == "code"

    def test_go_files(self):
        """Test Go file detection."""
        assert determine_file_type(Path("main.go")) == "code"
        assert determine_file_type(Path("server.go")) == "code"

    def test_java_kotlin_scala(self):
        """Test JVM language file detection."""
        assert determine_file_type(Path("Main.java")) == "code"
        assert determine_file_type(Path("App.kt")) == "code"
        assert determine_file_type(Path("Service.scala")) == "code"

    def test_cpp_files(self):
        """Test C/C++ file detection."""
        assert determine_file_type(Path("main.c")) == "code"
        assert determine_file_type(Path("lib.cpp")) == "code"
        assert determine_file_type(Path("lib.cxx")) == "code"
        assert determine_file_type(Path("lib.cc")) == "code"
        assert determine_file_type(Path("header.h")) == "code"
        assert determine_file_type(Path("header.hpp")) == "code"

    def test_dotnet_files(self):
        """Test .NET language file detection."""
        assert determine_file_type(Path("Program.cs")) == "code"
        assert determine_file_type(Path("Module.fs")) == "code"
        assert determine_file_type(Path("Form.vb")) == "code"

    def test_ruby_files(self):
        """Test Ruby file detection."""
        assert determine_file_type(Path("app.rb")) == "code"
        assert determine_file_type(Path("template.erb")) == "code"

    def test_php_files(self):
        """Test PHP file detection."""
        assert determine_file_type(Path("index.php")) == "code"
        assert determine_file_type(Path("template.phtml")) == "code"

    def test_shell_scripts(self):
        """Test shell script detection."""
        assert determine_file_type(Path("install.sh")) == "code"
        assert determine_file_type(Path("setup.bash")) == "code"
        assert determine_file_type(Path("configure.zsh")) == "code"
        assert determine_file_type(Path("config.fish")) == "code"

    def test_sql_files(self):
        """Test SQL file detection."""
        assert determine_file_type(Path("schema.sql")) == "code"
        assert determine_file_type(Path("migrations.ddl")) == "code"
        assert determine_file_type(Path("queries.dml")) == "code"


class TestTestFileDetection:
    """Test test file classification."""

    def test_prefix_test_files(self):
        """Test files with test_ prefix."""
        assert determine_file_type(Path("test_auth.py")) == "test"
        assert determine_file_type(Path("test_api.rs")) == "test"
        assert determine_file_type(Path("test_utils.js")) == "test"

    def test_suffix_test_files(self):
        """Test files with _test suffix."""
        assert determine_file_type(Path("auth_test.py")) == "test"
        assert determine_file_type(Path("api_test.rs")) == "test"
        assert determine_file_type(Path("utils_test.go")) == "test"

    def test_spec_files(self):
        """Test spec files (common in JS/TS)."""
        assert determine_file_type(Path("login.spec.ts")) == "test"
        assert determine_file_type(Path("component.spec.js")) == "test"

    def test_dot_test_files(self):
        """Test files with .test. pattern."""
        assert determine_file_type(Path("api.test.js")) == "test"
        assert determine_file_type(Path("utils.test.ts")) == "test"

    def test_conftest_files(self):
        """Test conftest files."""
        assert determine_file_type(Path("conftest.py")) == "test"

    def test_test_priority_over_code(self):
        """Test that test classification has priority over code."""
        # These have code extensions but test names
        assert determine_file_type(Path("test_main.py")) == "test"
        assert determine_file_type(Path("server_test.rs")) == "test"

    def test_test_in_tests_directory(self):
        """Test files in test directories."""
        # The function doesn't check parent directory, but documents it
        # This test documents expected behavior if we add directory checking
        assert determine_file_type(Path("tests/test_auth.py")) == "test"
        assert determine_file_type(Path("spec/login.spec.ts")) == "test"


class TestDocsFileDetection:
    """Test documentation file classification."""

    def test_markdown_files(self):
        """Test Markdown file detection."""
        assert determine_file_type(Path("README.md")) == "docs"
        assert determine_file_type(Path("guide.markdown")) == "docs"

    def test_rst_files(self):
        """Test reStructuredText file detection."""
        assert determine_file_type(Path("index.rst")) == "docs"
        assert determine_file_type(Path("guide.rest")) == "docs"

    def test_txt_files(self):
        """Test plain text file detection."""
        assert determine_file_type(Path("notes.txt")) == "docs"
        assert determine_file_type(Path("README.text")) == "docs"

    def test_pdf_files(self):
        """Test PDF file detection."""
        assert determine_file_type(Path("manual.pdf")) == "docs"

    def test_epub_files(self):
        """Test EPUB file detection."""
        assert determine_file_type(Path("book.epub")) == "docs"

    def test_word_files(self):
        """Test Word document detection."""
        assert determine_file_type(Path("report.docx")) == "docs"
        assert determine_file_type(Path("old_report.doc")) == "docs"

    def test_odt_files(self):
        """Test OpenDocument Text file detection."""
        assert determine_file_type(Path("document.odt")) == "docs"

    def test_rtf_files(self):
        """Test Rich Text Format file detection."""
        assert determine_file_type(Path("formatted.rtf")) == "docs"

    def test_asciidoc_files(self):
        """Test AsciiDoc file detection."""
        assert determine_file_type(Path("guide.adoc")) == "docs"
        assert determine_file_type(Path("manual.asciidoc")) == "docs"

    def test_org_files(self):
        """Test Org mode file detection."""
        assert determine_file_type(Path("notes.org")) == "docs"

    def test_latex_files(self):
        """Test LaTeX file detection."""
        assert determine_file_type(Path("paper.tex")) == "docs"


class TestConfigFileDetection:
    """Test configuration file classification."""

    def test_yaml_files(self):
        """Test YAML file detection."""
        assert determine_file_type(Path("config.yaml")) == "config"
        assert determine_file_type(Path("settings.yml")) == "config"

    def test_json_config_files(self):
        """Test JSON configuration file detection."""
        # JSON in config directory should be config
        assert determine_file_type(Path("config/settings.json")) == "config"
        assert determine_file_type(Path(".vscode/settings.json")) == "config"

    def test_toml_files(self):
        """Test TOML file detection."""
        assert determine_file_type(Path("pyproject.toml")) == "config"
        assert determine_file_type(Path("Cargo.toml")) == "config"

    def test_ini_files(self):
        """Test INI file detection."""
        assert determine_file_type(Path("setup.ini")) == "config"
        assert determine_file_type(Path("config.cfg")) == "config"
        assert determine_file_type(Path("app.conf")) == "config"

    def test_env_files(self):
        """Test environment file detection."""
        assert determine_file_type(Path(".env")) == "config"
        assert determine_file_type(Path(".env.local")) == "config"

    def test_properties_files(self):
        """Test Java properties file detection."""
        assert determine_file_type(Path("application.properties")) == "config"

    def test_editorconfig_files(self):
        """Test EditorConfig file detection."""
        assert determine_file_type(Path(".editorconfig")) == "config"

    def test_gitconfig_files(self):
        """Test Git config file detection."""
        assert determine_file_type(Path(".gitconfig")) == "config"
        assert determine_file_type(Path(".gitignore")) == "config"
        assert determine_file_type(Path(".gitattributes")) == "config"


class TestDataFileDetection:
    """Test data file classification."""

    def test_csv_files(self):
        """Test CSV file detection."""
        assert determine_file_type(Path("data.csv")) == "data"
        assert determine_file_type(Path("records.tsv")) == "data"

    def test_parquet_files(self):
        """Test Parquet file detection."""
        assert determine_file_type(Path("dataset.parquet")) == "data"

    def test_json_data_files(self):
        """Test JSON data file detection."""
        # JSON not in config directory should be data
        assert determine_file_type(Path("data/records.json")) == "data"
        assert determine_file_type(Path("output.json")) == "data"
        assert determine_file_type(Path("data.jsonl")) == "data"
        assert determine_file_type(Path("stream.ndjson")) == "data"

    def test_xml_data_files(self):
        """Test XML data file detection."""
        # XML not in config directory should be data
        assert determine_file_type(Path("data/records.xml")) == "data"

    def test_arrow_files(self):
        """Test Apache Arrow file detection."""
        assert determine_file_type(Path("dataset.arrow")) == "data"
        assert determine_file_type(Path("table.feather")) == "data"

    def test_hdf5_files(self):
        """Test HDF5 file detection."""
        assert determine_file_type(Path("data.hdf5")) == "data"
        assert determine_file_type(Path("dataset.h5")) == "data"

    def test_sqlite_files(self):
        """Test SQLite file detection."""
        assert determine_file_type(Path("app.db")) == "data"
        assert determine_file_type(Path("cache.sqlite")) == "data"
        assert determine_file_type(Path("data.sqlite3")) == "data"

    def test_pickle_files(self):
        """Test Python pickle file detection."""
        assert determine_file_type(Path("model.pkl")) == "data"
        assert determine_file_type(Path("cache.pickle")) == "data"

    def test_numpy_files(self):
        """Test NumPy file detection."""
        assert determine_file_type(Path("array.npy")) == "data"
        assert determine_file_type(Path("arrays.npz")) == "data"

    def test_matlab_files(self):
        """Test MATLAB data file detection."""
        assert determine_file_type(Path("data.mat")) == "data"

    def test_r_data_files(self):
        """Test R data file detection."""
        assert determine_file_type(Path("dataset.rds")) == "data"
        assert determine_file_type(Path("workspace.rdata")) == "data"


class TestBuildArtifactDetection:
    """Test build artifact classification."""

    def test_wheel_files(self):
        """Test Python wheel file detection."""
        assert determine_file_type(Path("package-1.0.0-py3-none-any.whl")) == "build"

    def test_tarball_files(self):
        """Test tarball file detection."""
        assert determine_file_type(Path("package.tar.gz")) == "build"
        assert determine_file_type(Path("archive.tgz")) == "build"
        assert determine_file_type(Path("bundle.tar.bz2")) == "build"
        assert determine_file_type(Path("compressed.tbz2")) == "build"

    def test_zip_files(self):
        """Test ZIP file detection."""
        assert determine_file_type(Path("archive.zip")) == "build"

    def test_jar_files(self):
        """Test Java archive file detection."""
        assert determine_file_type(Path("app.jar")) == "build"
        assert determine_file_type(Path("webapp.war")) == "build"
        assert determine_file_type(Path("enterprise.ear")) == "build"

    def test_shared_libraries(self):
        """Test shared library file detection."""
        assert determine_file_type(Path("libmylib.so")) == "build"
        assert determine_file_type(Path("library.dylib")) == "build"
        assert determine_file_type(Path("module.dll")) == "build"

    def test_static_libraries(self):
        """Test static library file detection."""
        assert determine_file_type(Path("libstatic.a")) == "build"
        assert determine_file_type(Path("library.lib")) == "build"

    def test_object_files(self):
        """Test object file detection."""
        assert determine_file_type(Path("module.o")) == "build"
        assert determine_file_type(Path("object.obj")) == "build"

    def test_executables(self):
        """Test executable file detection."""
        assert determine_file_type(Path("app.exe")) == "build"
        assert determine_file_type(Path("MyApp.app")) == "build"

    def test_package_formats(self):
        """Test package format file detection."""
        assert determine_file_type(Path("package.deb")) == "build"
        assert determine_file_type(Path("package.rpm")) == "build"

    def test_disk_images(self):
        """Test disk image file detection."""
        assert determine_file_type(Path("installer.dmg")) == "build"
        assert determine_file_type(Path("system.iso")) == "build"


class TestEdgeCases:
    """Test classification edge cases."""

    def test_json_in_config_directory(self):
        """Test JSON in config directory is classified as config."""
        assert determine_file_type(Path("config/app.json")) == "config"
        assert determine_file_type(Path("settings/database.json")) == "config"
        assert determine_file_type(Path(".vscode/launch.json")) == "config"

    def test_json_in_data_directory(self):
        """Test JSON in data directory is classified as data."""
        assert determine_file_type(Path("data/records.json")) == "data"
        assert determine_file_type(Path("output/results.json")) == "data"

    def test_xml_in_config_directory(self):
        """Test XML in config directory is classified as config."""
        assert determine_file_type(Path("config/pom.xml")) == "config"
        assert determine_file_type(Path(".github/workflows/ci.xml")) == "config"

    def test_xml_in_data_directory(self):
        """Test XML in data directory is classified as data."""
        assert determine_file_type(Path("data/records.xml")) == "data"

    def test_unknown_extension_returns_other(self):
        """Test unknown extensions return 'other'."""
        assert determine_file_type(Path("file.unknown")) == "other"
        assert determine_file_type(Path("data.xyz")) == "other"
        assert determine_file_type(Path("noextension")) == "other"

    def test_multiple_dots_in_filename(self):
        """Test files with multiple dots in name."""
        assert determine_file_type(Path("test.component.spec.ts")) == "test"
        assert determine_file_type(Path("api.v2.test.js")) == "test"
        assert determine_file_type(Path("package-1.0.0.tar.gz")) == "build"

    def test_uppercase_extensions(self):
        """Test that uppercase extensions are normalized."""
        # The function converts to lowercase internally
        assert determine_file_type(Path("Main.PY")) == "code"
        assert determine_file_type(Path("README.MD")) == "docs"
        assert determine_file_type(Path("Config.YAML")) == "config"

    def test_no_extension(self):
        """Test files with no extension."""
        # Files without extension return 'other'
        assert determine_file_type(Path("Makefile")) == "other"
        assert determine_file_type(Path("Dockerfile")) == "other"
        assert determine_file_type(Path("README")) == "other"


class TestIsTestDirectory:
    """Test directory classification."""

    def test_common_test_directories(self):
        """Test common test directory names."""
        assert is_test_directory(Path("tests"))
        assert is_test_directory(Path("test"))
        assert is_test_directory(Path("__tests__"))
        assert is_test_directory(Path("spec"))
        assert is_test_directory(Path("specs"))

    def test_test_type_directories(self):
        """Test specific test type directories."""
        assert is_test_directory(Path("integration"))
        assert is_test_directory(Path("e2e"))
        assert is_test_directory(Path("unit"))
        assert is_test_directory(Path("functional"))
        assert is_test_directory(Path("acceptance"))

    def test_non_test_directories(self):
        """Test non-test directory names."""
        assert not is_test_directory(Path("src"))
        assert not is_test_directory(Path("lib"))
        assert not is_test_directory(Path("docs"))
        assert not is_test_directory(Path("config"))

    def test_case_insensitive(self):
        """Test directory name is case-insensitive."""
        assert is_test_directory(Path("Tests"))
        assert is_test_directory(Path("TEST"))
        assert is_test_directory(Path("Spec"))


class TestFileTypeExtensionSets:
    """Test that extension sets are properly defined."""

    def test_code_extensions_not_empty(self):
        """Test CODE_EXTENSIONS set is not empty."""
        assert len(CODE_EXTENSIONS) > 0
        assert '.py' in CODE_EXTENSIONS
        assert '.rs' in CODE_EXTENSIONS

    def test_docs_extensions_not_empty(self):
        """Test DOCS_EXTENSIONS set is not empty."""
        assert len(DOCS_EXTENSIONS) > 0
        assert '.md' in DOCS_EXTENSIONS
        assert '.rst' in DOCS_EXTENSIONS

    def test_config_extensions_not_empty(self):
        """Test CONFIG_EXTENSIONS set is not empty."""
        assert len(CONFIG_EXTENSIONS) > 0
        assert '.yaml' in CONFIG_EXTENSIONS
        assert '.json' in CONFIG_EXTENSIONS

    def test_data_extensions_not_empty(self):
        """Test DATA_EXTENSIONS set is not empty."""
        assert len(DATA_EXTENSIONS) > 0
        assert '.csv' in DATA_EXTENSIONS
        assert '.parquet' in DATA_EXTENSIONS

    def test_build_extensions_not_empty(self):
        """Test BUILD_EXTENSIONS set is not empty."""
        assert len(BUILD_EXTENSIONS) > 0
        assert '.whl' in BUILD_EXTENSIONS
        assert '.jar' in BUILD_EXTENSIONS

    def test_no_overlapping_extensions(self):
        """Test that extension sets don't have significant overlaps."""
        # JSON and XML can appear in both config and data (context-dependent)
        # But code, docs, and build should be distinct
        code_docs_overlap = CODE_EXTENSIONS & DOCS_EXTENSIONS
        code_build_overlap = CODE_EXTENSIONS & BUILD_EXTENSIONS
        docs_build_overlap = DOCS_EXTENSIONS & BUILD_EXTENSIONS

        assert len(code_docs_overlap) == 0
        assert len(code_build_overlap) == 0
        assert len(docs_build_overlap) == 0


class TestPriorityOrdering:
    """Test that file type classification priority is correct."""

    def test_test_has_priority_over_code(self):
        """Test files are classified as 'test' even if they have code extensions."""
        # test_file.py has .py extension (code) but test_ prefix (test)
        # Test classification should win
        assert determine_file_type(Path("test_main.py")) == "test"
        assert determine_file_type(Path("utils_test.rs")) == "test"

    def test_docs_has_priority_over_config(self):
        """Docs files are clearly identified even with similar extensions."""
        # .txt could be docs or other, but it's in DOCS_EXTENSIONS
        assert determine_file_type(Path("README.txt")) == "docs"
        assert determine_file_type(Path("notes.txt")) == "docs"

    def test_config_before_data_for_ambiguous_extensions(self):
        """Config classification takes priority for context-aware extensions."""
        # .json in config directory → config
        assert determine_file_type(Path("config/app.json")) == "config"
        # .json in data directory → data
        assert determine_file_type(Path("data/records.json")) == "data"

    def test_fallback_to_other(self):
        """Unknown files fall back to 'other'."""
        assert determine_file_type(Path("unknown.xyz")) == "other"
        assert determine_file_type(Path("binary.bin")) == "other"
