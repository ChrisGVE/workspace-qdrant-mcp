//! Spreadsheet (XLSX, XLS) and CSV/TSV text extraction.

use std::collections::HashMap;
use std::path::Path;

use super::xml_utils::clean_extracted_text;
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// Extract text from Excel spreadsheet files (XLSX and XLS) using calamine
pub fn extract_spreadsheet(
    file_path: &Path,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    use calamine::{open_workbook_auto, Data, Reader};

    let mut metadata = HashMap::new();
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("xlsx")
        .to_lowercase();
    metadata.insert("source_format".to_string(), ext);

    let mut workbook = open_workbook_auto(file_path)
        .map_err(|e| DocumentProcessorError::SpreadsheetExtraction(e.to_string()))?;

    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();
    metadata.insert("sheet_count".to_string(), sheet_names.len().to_string());

    let mut all_text = String::new();
    let mut total_rows = 0usize;

    for sheet_name in &sheet_names {
        if let Ok(range) = workbook.worksheet_range(sheet_name) {
            if !all_text.is_empty() {
                all_text.push('\n');
            }
            all_text.push_str(&format!("## {}\n", sheet_name));

            for row in range.rows() {
                total_rows += 1;
                let cells: Vec<String> = row
                    .iter()
                    .map(|cell| match cell {
                        Data::Empty => String::new(),
                        Data::String(s) => s.clone(),
                        Data::Int(i) => i.to_string(),
                        Data::Float(f) => f.to_string(),
                        Data::Bool(b) => b.to_string(),
                        Data::DateTime(dt) => dt.to_string(),
                        Data::Error(e) => format!("#ERR:{:?}", e),
                        Data::DateTimeIso(s) => s.clone(),
                        Data::DurationIso(s) => s.clone(),
                    })
                    .collect();

                // Skip entirely empty rows
                if cells.iter().all(|c| c.is_empty()) {
                    continue;
                }

                all_text.push_str(&cells.join("\t"));
                all_text.push('\n');
            }
        }
    }

    metadata.insert("row_count".to_string(), total_rows.to_string());

    if all_text.trim().is_empty() {
        return Err(DocumentProcessorError::SpreadsheetExtraction(
            "No data found in spreadsheet".to_string(),
        ));
    }

    Ok((clean_extracted_text(&all_text), metadata))
}

/// Extract text from CSV/TSV files
pub fn extract_csv(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("csv")
        .to_lowercase();
    metadata.insert("source_format".to_string(), ext.clone());

    let delimiter = if ext == "tsv" { b'\t' } else { b',' };

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(true) // tolerate rows with varying column counts
        .has_headers(true)
        .from_path(file_path)
        .map_err(|e| DocumentProcessorError::CsvExtraction(e.to_string()))?;

    let mut all_text = String::new();
    let mut row_count = 0usize;

    // Include headers
    let headers = reader
        .headers()
        .map_err(|e| DocumentProcessorError::CsvExtraction(e.to_string()))?
        .clone();
    let col_count = headers.len();
    metadata.insert("column_count".to_string(), col_count.to_string());

    if col_count > 0 {
        let header_line: Vec<&str> = headers.iter().collect();
        all_text.push_str(&header_line.join("\t"));
        all_text.push('\n');
    }

    for result in reader.records() {
        let record = result.map_err(|e| DocumentProcessorError::CsvExtraction(e.to_string()))?;
        row_count += 1;
        let fields: Vec<&str> = record.iter().collect();
        all_text.push_str(&fields.join("\t"));
        all_text.push('\n');
    }

    metadata.insert("row_count".to_string(), row_count.to_string());

    if all_text.trim().is_empty() {
        return Err(DocumentProcessorError::CsvExtraction(
            "No data found in CSV/TSV file".to_string(),
        ));
    }

    Ok((clean_extracted_text(&all_text), metadata))
}
