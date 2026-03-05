//! Jupyter notebook (.ipynb) text extraction.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use super::xml_utils::clean_extracted_text;
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// Extract code and markdown from Jupyter notebook (.ipynb) files
pub fn extract_jupyter(
    file_path: &Path,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "jupyter".to_string());

    let notebook = load_notebook_json(file_path)?;

    let language = detect_notebook_language(&notebook);
    metadata.insert("language".to_string(), language.clone());

    let cells = notebook
        .get("cells")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            DocumentProcessorError::JupyterExtraction(
                "No cells array found in notebook".to_string(),
            )
        })?;

    metadata.insert("cell_count".to_string(), cells.len().to_string());

    let (all_text, code_cells, markdown_cells) = render_cells(cells, &language);

    metadata.insert("code_cells".to_string(), code_cells.to_string());
    metadata.insert("markdown_cells".to_string(), markdown_cells.to_string());

    if all_text.trim().is_empty() {
        return Err(DocumentProcessorError::JupyterExtraction(
            "No content found in notebook".to_string(),
        ));
    }

    Ok((clean_extracted_text(&all_text), metadata))
}

fn load_notebook_json(file_path: &Path) -> DocumentProcessorResult<serde_json::Value> {
    let mut file = std::fs::File::open(file_path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    serde_json::from_str(&content).map_err(|e| {
        DocumentProcessorError::JupyterExtraction(format!("Invalid notebook JSON: {}", e))
    })
}

fn detect_notebook_language(notebook: &serde_json::Value) -> String {
    notebook
        .pointer("/metadata/kernelspec/language")
        .or_else(|| notebook.pointer("/metadata/language_info/name"))
        .and_then(|v| v.as_str())
        .unwrap_or("python")
        .to_string()
}

fn render_cells(cells: &[serde_json::Value], language: &str) -> (String, usize, usize) {
    let mut all_text = String::new();
    let mut code_cells = 0usize;
    let mut markdown_cells = 0usize;

    for cell in cells {
        let cell_type = cell
            .get("cell_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let source = read_cell_source(cell);
        if source.trim().is_empty() {
            continue;
        }
        if !all_text.is_empty() {
            all_text.push('\n');
        }
        match cell_type {
            "code" => {
                code_cells += 1;
                all_text.push_str(&format!("```{}\n{}\n```", language, source.trim()));
            }
            "markdown" => {
                markdown_cells += 1;
                all_text.push_str(source.trim());
            }
            _ => {
                all_text.push_str(source.trim());
            }
        }
    }

    (all_text, code_cells, markdown_cells)
}

fn read_cell_source(cell: &serde_json::Value) -> String {
    cell.get("source")
        .and_then(|v| v.as_array())
        .map(|lines| {
            lines
                .iter()
                .filter_map(|l| l.as_str())
                .collect::<Vec<&str>>()
                .join("")
        })
        .or_else(|| {
            cell.get("source")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .unwrap_or_default()
}
