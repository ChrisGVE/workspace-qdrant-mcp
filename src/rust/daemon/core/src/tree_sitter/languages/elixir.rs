//! Elixir language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::node_text;
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Elixir source code.
pub struct ElixirExtractor {
    language: Option<Language>,
}

impl ElixirExtractor {
    pub fn new() -> Self {
        Self { language: None }
    }

    pub fn with_language(language: Language) -> Self {
        Self {
            language: Some(language),
        }
    }

    fn create_parser(&self) -> Result<TreeSitterParser, DaemonError> {
        match &self.language {
            Some(lang) => TreeSitterParser::with_language("elixir", lang.clone()),
            None => TreeSitterParser::new("elixir"),
        }
    }

    fn extract_preamble(
        &self,
        root: &Node,
        source: &str,
        file_path: &str,
    ) -> Option<SemanticChunk> {
        let mut preamble_items = Vec::new();
        let mut last_preamble_line = 0;
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            match child.kind() {
                "comment" => {
                    if preamble_items.is_empty()
                        || child.start_position().row <= last_preamble_line + 1
                    {
                        preamble_items.push(node_text(&child, source).to_string());
                        last_preamble_line = child.end_position().row + 1;
                    }
                }
                "call" => {
                    let text = node_text(&child, source);
                    // Elixir uses `use`, `import`, `alias`, `require` as calls
                    let first_word = text.split_whitespace().next().unwrap_or("");
                    if matches!(first_word, "use" | "import" | "alias" | "require") {
                        preamble_items.push(text.to_string());
                        last_preamble_line = child.end_position().row + 1;
                    } else if !preamble_items.is_empty() {
                        break;
                    }
                }
                _ => {
                    if !preamble_items.is_empty() {
                        break;
                    }
                }
            }
        }

        if preamble_items.is_empty() {
            return None;
        }

        Some(SemanticChunk::preamble(
            preamble_items.join("\n"),
            last_preamble_line,
            "elixir",
            file_path,
        ))
    }

    fn is_def_call(text: &str) -> Option<(&str, ChunkType)> {
        let first_word = text.split_whitespace().next()?;
        match first_word {
            "def" => Some(("def", ChunkType::Function)),
            "defp" => Some(("defp", ChunkType::Function)),
            "defmacro" => Some(("defmacro", ChunkType::Macro)),
            "defmacrop" => Some(("defmacrop", ChunkType::Macro)),
            _ => None,
        }
    }

    fn is_module_call(text: &str) -> bool {
        text.split_whitespace()
            .next()
            .map(|w| w == "defmodule")
            .unwrap_or(false)
    }

    fn extract_function_from_call(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        parent: Option<&str>,
        chunk_type: ChunkType,
    ) -> SemanticChunk {
        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        // Extract function name from `def name(args)` pattern
        let name = content
            .split_whitespace()
            .nth(1)
            .map(|s| s.split('(').next().unwrap_or(s))
            .unwrap_or("anonymous");

        let signature = content
            .lines()
            .next()
            .map(|l| l.trim_end_matches(" do").trim().to_string());

        let docstring = self.extract_doc_comment(node, source);

        let symbol_kind = if content.starts_with("defp") {
            "private_function"
        } else if chunk_type == ChunkType::Macro {
            "macro"
        } else {
            "function"
        };

        let mut chunk = SemanticChunk::new(
            chunk_type, name, content, start_line, end_line, "elixir", file_path,
        )
        .with_symbol_kind(symbol_kind);

        if let Some(sig) = signature {
            chunk = chunk.with_signature(sig);
        }
        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }
        if let Some(p) = parent {
            chunk = chunk.with_parent(p);
        }

        chunk
    }

    fn extract_module(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
    ) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        // Extract module name from `defmodule Name do`
        let name = content
            .split_whitespace()
            .nth(1)
            .map(|s| s.trim_end_matches(" ").to_string())
            .unwrap_or_else(|| "anonymous".to_string());

        let docstring = self.extract_moduledoc(node, source);

        let mut module_chunk = SemanticChunk::new(
            ChunkType::Module,
            &name,
            content,
            start_line,
            end_line,
            "elixir",
            file_path,
        );
        if let Some(doc) = docstring {
            module_chunk = module_chunk.with_docstring(doc);
        }
        chunks.push(module_chunk);

        // Walk children for def/defp/defmacro/defmodule
        self.walk_children_for_defs(node, source, file_path, &name, &mut chunks);

        chunks
    }

    fn walk_children_for_defs(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        parent_name: &str,
        chunks: &mut Vec<SemanticChunk>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "call" {
                let text = node_text(&child, source);
                if let Some((_, chunk_type)) = Self::is_def_call(&text) {
                    chunks.push(self.extract_function_from_call(
                        &child,
                        source,
                        file_path,
                        Some(parent_name),
                        chunk_type,
                    ));
                } else if Self::is_module_call(&text) {
                    chunks.extend(self.extract_module(&child, source, file_path));
                }
            } else if child.kind() == "do_block" || child.kind() == "stab_clause" {
                self.walk_children_for_defs(&child, source, file_path, parent_name, chunks);
            } else if child.child_count() > 0 {
                self.walk_children_for_defs(&child, source, file_path, parent_name, chunks);
            }
        }
    }

    fn extract_doc_comment(&self, node: &Node, source: &str) -> Option<String> {
        // Look for @doc preceding the function
        let mut prev = node.prev_sibling();
        while let Some(sibling) = prev {
            let text = node_text(&sibling, source);
            if text.starts_with("@doc") {
                let doc = text
                    .trim_start_matches("@doc")
                    .trim()
                    .trim_start_matches("\"\"\"")
                    .trim_end_matches("\"\"\"")
                    .trim()
                    .to_string();
                return Some(doc);
            }
            if sibling.kind() == "comment" {
                prev = sibling.prev_sibling();
                continue;
            }
            break;
        }
        None
    }

    fn extract_moduledoc(&self, node: &Node, source: &str) -> Option<String> {
        // Look for @moduledoc inside the module body
        let content = node_text(node, source);
        if let Some(idx) = content.find("@moduledoc") {
            let rest = &content[idx + "@moduledoc".len()..];
            let doc = rest
                .trim()
                .trim_start_matches("\"\"\"")
                .lines()
                .take_while(|l| !l.trim_start().starts_with("\"\"\""))
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string();
            if !doc.is_empty() {
                return Some(doc);
            }
        }
        None
    }
}

impl ChunkExtractor for ElixirExtractor {
    fn extract_chunks(
        &self,
        source: &str,
        file_path: &Path,
    ) -> Result<Vec<SemanticChunk>, DaemonError> {
        let mut parser = self.create_parser()?;
        let tree = parser.parse(source)?;
        let root = tree.root_node();
        let file_path_str = file_path.to_string_lossy().to_string();

        let mut chunks = Vec::new();

        if let Some(preamble) = self.extract_preamble(&root, source, &file_path_str) {
            chunks.push(preamble);
        }

        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if child.kind() == "call" {
                let text = node_text(&child, source);
                if Self::is_module_call(&text) {
                    chunks.extend(self.extract_module(&child, source, &file_path_str));
                } else if let Some((_, chunk_type)) = Self::is_def_call(&text) {
                    chunks.push(self.extract_function_from_call(
                        &child,
                        source,
                        &file_path_str,
                        None,
                        chunk_type,
                    ));
                }
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "elixir"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    #[test]
    fn test_extract_module_and_functions() {
        let Some(lang) = get_language("elixir") else {
            return;
        };
        let source = r#"
defmodule Greeter do
  def hello(name) do
    "Hello, #{name}!"
  end

  defp internal_helper do
    :ok
  end
end
"#;
        let path = PathBuf::from("test.ex");
        let extractor = ElixirExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let module = chunks.iter().find(|c| c.chunk_type == ChunkType::Module);
        assert!(module.is_some());

        let fns: Vec<_> = chunks
            .iter()
            .filter(|c| c.chunk_type == ChunkType::Function)
            .collect();
        assert!(fns.len() >= 1);
    }
}
