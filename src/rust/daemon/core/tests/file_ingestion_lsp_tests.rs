//! LSP analysis integration tests using Pipeline API (Task 315.4)
//!
//! These tests validate code file processing through the pipeline and prepare for
//! future LSP integration. The placeholder implementation doesn't perform actual
//! LSP analysis, but these tests ensure the pipeline correctly handles code files.
//!
//! Future enhancement: When LSP integration is implemented, these tests should
//! validate symbol extraction, language-specific parsing, dependency analysis,
//! and code structure metadata.

use shared_test_utils::TestResult;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;
use workspace_qdrant_core::{
    classify_file_type, FileType, Pipeline, TaskPayload, TaskPriority, TaskResult, TaskSource,
};

const TEST_COLLECTION: &str = "test_collection";
const TASK_TIMEOUT: Duration = Duration::from_secs(5);

/// Test helper to create a document file with a non-test name
/// This is needed because files starting with "test_" are classified as test files
async fn create_document_file(content: &str, extension: &str) -> TestResult<(TempDir, PathBuf)> {
    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join(format!("document.{}", extension));
    fs::write(&file_path, content).await?;
    Ok((temp_dir, file_path))
}

#[tokio::test]
async fn test_python_code_with_lsp_metadata_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, x: int, y: int) -> int:
        result = calculate_sum(x, y)
        self.history.append(result)
        return result
"#;
    let (_temp_dir, file_path) = create_document_file(content, "py").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(
        file_type,
        FileType::Code,
        "Python files should be classified as code"
    );

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "lsp_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate LSP extracted symbols (calculate_sum, Calculator, add, __init__)
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "Python code file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_javascript_code_with_lsp_metadata_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
function calculateSum(a, b) {
    return a + b;
}

class Calculator {
    constructor() {
        this.history = [];
    }

    add(x, y) {
        const result = calculateSum(x, y);
        this.history.push(result);
        return result;
    }
}

export { Calculator, calculateSum };
"#;
    let (_temp_dir, file_path) = create_document_file(content, "js").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(
        file_type,
        FileType::Code,
        "JavaScript files should be classified as code"
    );

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "lsp_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate LSP extracted symbols (calculateSum, Calculator, add, constructor)
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "JavaScript code file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_typescript_code_with_lsp_metadata_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
interface CalculatorInterface {
    add(x: number, y: number): number;
    history: number[];
}

function calculateSum(a: number, b: number): number {
    return a + b;
}

class Calculator implements CalculatorInterface {
    history: number[] = [];

    add(x: number, y: number): number {
        const result = calculateSum(x, y);
        this.history.push(result);
        return result;
    }
}

export { Calculator, calculateSum, CalculatorInterface };
"#;
    let (_temp_dir, file_path) = create_document_file(content, "ts").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(
        file_type,
        FileType::Code,
        "TypeScript files should be classified as code"
    );

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "lsp_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate LSP extracted symbols (CalculatorInterface, calculateSum, Calculator, add)
    // Future: Validate type information extraction
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "TypeScript code file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_rust_code_with_lsp_metadata_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
pub fn calculate_sum(a: i32, b: i32) -> i32 {
    a + b
}

pub struct Calculator {
    pub history: Vec<i32>,
}

impl Calculator {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }

    pub fn add(&mut self, x: i32, y: i32) -> i32 {
        let result = calculate_sum(x, y);
        self.history.push(result);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let mut calc = Calculator::new();
        assert_eq!(calc.add(2, 3), 5);
    }
}
"#;
    let (_temp_dir, file_path) = create_document_file(content, "rs").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(
        file_type,
        FileType::Code,
        "Rust files should be classified as code"
    );

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "lsp_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate LSP extracted symbols (calculate_sum, Calculator, new, add)
    // Future: Validate module structure (tests module)
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "Rust code file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_go_code_with_lsp_metadata_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
package calculator

func CalculateSum(a int, b int) int {
    return a + b
}

type Calculator struct {
    History []int
}

func NewCalculator() *Calculator {
    return &Calculator{
        History: make([]int, 0),
    }
}

func (c *Calculator) Add(x int, y int) int {
    result := CalculateSum(x, y)
    c.History = append(c.History, result)
    return result
}
"#;
    let (_temp_dir, file_path) = create_document_file(content, "go").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(
        file_type,
        FileType::Code,
        "Go files should be classified as code"
    );

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "lsp_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate LSP extracted symbols (CalculateSum, Calculator, NewCalculator, Add)
    // Future: Validate package structure
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "Go code file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_multiple_code_languages_concurrent() -> TestResult {
    let mut pipeline = Pipeline::new(5);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;

    // Create code files in different languages
    let code_files = vec![
        (
            "script.py",
            r#"def hello(): print("Hello from Python")"#,
            "Python",
        ),
        (
            "script.js",
            r#"function hello() { console.log("Hello from JavaScript"); }"#,
            "JavaScript",
        ),
        (
            "main.rs",
            r#"fn main() { println!("Hello from Rust"); }"#,
            "Rust",
        ),
        (
            "main.go",
            r#"package main; func main() { println("Hello from Go") }"#,
            "Go",
        ),
        (
            "App.tsx",
            r#"export const App = () => <div>Hello from TypeScript</div>;"#,
            "TypeScript",
        ),
    ];

    let mut handles = Vec::new();

    for (filename, content, language) in code_files.iter() {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).await?;

        // Verify classification
        let file_type = classify_file_type(&file_path);
        assert_eq!(
            file_type,
            FileType::Code,
            "{} files should be classified as code",
            language
        );

        // Submit for processing
        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "multi_language_lsp_test".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                    branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        handles.push((language, handle));
    }

    // All code files should process successfully
    let mut success_count = 0;
    for (language, handle) in handles {
        let result = handle.wait().await?;
        if matches!(result, TaskResult::Success { .. }) {
            success_count += 1;
        } else {
            panic!(
                "{} code file should process successfully, got: {:?}",
                language, result
            );
        }
    }

    assert_eq!(
        success_count, 5,
        "All 5 code files in different languages should process successfully"
    );

    // Future: Validate each language's LSP analysis produced appropriate symbols
    // Future: Validate language-specific metadata (imports, exports, etc.)

    Ok(())
}
