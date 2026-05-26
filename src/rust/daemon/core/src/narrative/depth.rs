/// Depth level estimation for narrative nodes.
///
/// Classifies how deeply a document section covers a topic:
/// qualitative, introductory, intermediate, rigorous, reference.
use crate::graph::DepthLevel;

/// Estimate depth level from content characteristics.
pub fn estimate_depth(content: &str, _language: Option<&str>) -> DepthLevel {
    let word_count = content.split_whitespace().count();
    let has_code_blocks = content.contains("```");
    let has_equations = content.contains('$') || content.contains("\\(");
    let has_references = content.contains("[^") || content.contains("\\cite");

    if has_references && has_equations {
        return DepthLevel::Rigorous;
    }
    if has_code_blocks && word_count > 500 {
        return DepthLevel::Intermediate;
    }
    if word_count > 200 {
        return DepthLevel::Introductory;
    }
    if word_count < 50 {
        return DepthLevel::Reference;
    }
    DepthLevel::Qualitative
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_content_is_reference() {
        assert_eq!(
            estimate_depth("API: foo(x) -> y", None),
            DepthLevel::Reference
        );
    }

    #[test]
    fn test_medium_qualitative() {
        let content = "a ".repeat(100);
        assert_eq!(estimate_depth(&content, None), DepthLevel::Qualitative);
    }

    #[test]
    fn test_long_introductory() {
        let content = "word ".repeat(250);
        assert_eq!(estimate_depth(&content, None), DepthLevel::Introductory);
    }

    #[test]
    fn test_code_blocks_intermediate() {
        let mut content = "word ".repeat(600);
        content.push_str("\n```rust\nfn main() {}\n```\n");
        assert_eq!(estimate_depth(&content, None), DepthLevel::Intermediate);
    }

    #[test]
    fn test_equations_and_refs_rigorous() {
        let content = "The proof follows from $\\alpha$ and [^1] citation.";
        assert_eq!(estimate_depth(content, None), DepthLevel::Rigorous);
    }
}
