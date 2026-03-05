//! Line-level diff computation using imara-diff.
//!
//! Given old content (from `indexed_content`) and new content (from disk),
//! computes a minimal edit script of line-level changes. The output is a
//! sequence of `DiffOp` values that can be mapped to SQL operations on the
//! `code_lines` table:
//!
//! - `Unchanged { old_index, new_index }` → no-op
//! - `Changed { old_index, new_index, new_content }` → UPDATE code_lines SET content
//! - `Inserted { new_index, new_content }` → INSERT code_lines with midpoint seq
//! - `Deleted { old_index }` → DELETE FROM code_lines WHERE file_id AND seq
//!
//! Uses `imara_diff` with the Histogram algorithm for best performance
//! on typical source code files.

use imara_diff::{Algorithm, Diff, InternedInput};

/// A single diff operation between old and new file content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffOp {
    /// Line is unchanged (exists at old_index in old, new_index in new).
    Unchanged { old_index: usize, new_index: usize },
    /// Line content changed (old_index in old replaced by new content at new_index).
    Changed {
        old_index: usize,
        new_index: usize,
        new_content: String,
    },
    /// Line was inserted (new line at new_index, not present in old).
    Inserted {
        new_index: usize,
        new_content: String,
    },
    /// Line was deleted (old_index in old, not present in new).
    Deleted { old_index: usize },
}

/// Result of computing a diff between two file contents.
#[derive(Debug)]
pub struct DiffResult {
    /// The ordered list of diff operations.
    pub ops: Vec<DiffOp>,
    /// Number of lines in the old content.
    pub old_line_count: usize,
    /// Number of lines in the new content.
    pub new_line_count: usize,
}

impl DiffResult {
    /// Returns true if there are no changes.
    pub fn is_empty(&self) -> bool {
        self.ops
            .iter()
            .all(|op| matches!(op, DiffOp::Unchanged { .. }))
    }

    /// Count the number of changed lines (inserts + deletes + changes).
    pub fn change_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| !matches!(op, DiffOp::Unchanged { .. }))
            .count()
    }

    /// Count inserted lines.
    pub fn insert_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, DiffOp::Inserted { .. }))
            .count()
    }

    /// Count deleted lines.
    pub fn delete_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, DiffOp::Deleted { .. }))
            .count()
    }

    /// Count changed (modified) lines.
    pub fn modified_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, DiffOp::Changed { .. }))
            .count()
    }
}

/// Build per-line removed/added flags from diff hunks.
fn build_change_flags<'a>(
    old_lines: &[&'a str],
    new_lines: &[&'a str],
    old_content: &'a str,
    new_content: &'a str,
) -> (Vec<bool>, Vec<bool>) {
    let input = InternedInput::new(old_content, new_content);
    let diff = Diff::compute(Algorithm::Histogram, &input);

    let mut old_removed = vec![false; old_lines.len()];
    let mut new_added = vec![false; new_lines.len()];

    for hunk in diff.hunks() {
        for i in hunk.before.start..hunk.before.end {
            if (i as usize) < old_lines.len() {
                old_removed[i as usize] = true;
            }
        }
        for i in hunk.after.start..hunk.after.end {
            if (i as usize) < new_lines.len() {
                new_added[i as usize] = true;
            }
        }
    }

    (old_removed, new_added)
}

/// Compute a line-level diff between old and new file content.
///
/// Uses the Histogram algorithm (best general-purpose performance for code).
/// Lines are split on `\n`. Both inputs should be UTF-8 strings.
pub fn compute_line_diff(old_content: &str, new_content: &str) -> DiffResult {
    let old_lines: Vec<&str> = old_content.split('\n').collect();
    let new_lines: Vec<&str> = new_content.split('\n').collect();

    let (old_removed, new_added) =
        build_change_flags(&old_lines, &new_lines, old_content, new_content);

    // Walk both sequences to produce DiffOps.
    // Unchanged lines advance both cursors. Removed/added lines are paired
    // as Changed when they appear in the same hunk, or emitted as pure
    // Delete/Insert otherwise.
    let mut ops = Vec::new();
    let mut old_idx = 0usize;
    let mut new_idx = 0usize;

    while old_idx < old_lines.len() || new_idx < new_lines.len() {
        let old_is_removed = old_idx < old_lines.len() && old_removed[old_idx];
        let new_is_added = new_idx < new_lines.len() && new_added[new_idx];

        match (old_is_removed, new_is_added) {
            (false, false) => {
                // Both lines are unchanged — advance both
                if old_idx < old_lines.len() && new_idx < new_lines.len() {
                    ops.push(DiffOp::Unchanged {
                        old_index: old_idx,
                        new_index: new_idx,
                    });
                    old_idx += 1;
                    new_idx += 1;
                } else if old_idx < old_lines.len() {
                    // Shouldn't happen in well-formed diff, but handle gracefully
                    ops.push(DiffOp::Deleted { old_index: old_idx });
                    old_idx += 1;
                } else {
                    ops.push(DiffOp::Inserted {
                        new_index: new_idx,
                        new_content: new_lines[new_idx].to_string(),
                    });
                    new_idx += 1;
                }
            }
            (true, true) => {
                // Both removed and added — pair as Changed
                ops.push(DiffOp::Changed {
                    old_index: old_idx,
                    new_index: new_idx,
                    new_content: new_lines[new_idx].to_string(),
                });
                old_idx += 1;
                new_idx += 1;
            }
            (true, false) => {
                // Old line removed, no corresponding new line
                ops.push(DiffOp::Deleted { old_index: old_idx });
                old_idx += 1;
            }
            (false, true) => {
                // New line added, no corresponding old line
                ops.push(DiffOp::Inserted {
                    new_index: new_idx,
                    new_content: new_lines[new_idx].to_string(),
                });
                new_idx += 1;
            }
        }
    }

    DiffResult {
        ops,
        old_line_count: old_lines.len(),
        new_line_count: new_lines.len(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_identical_content() {
        let content = "line1\nline2\nline3";
        let result = compute_line_diff(content, content);
        assert!(result.is_empty());
        assert_eq!(result.change_count(), 0);
        assert_eq!(result.old_line_count, 3);
        assert_eq!(result.new_line_count, 3);
        assert_eq!(result.ops.len(), 3);
        for op in &result.ops {
            assert!(matches!(op, DiffOp::Unchanged { .. }));
        }
    }

    #[test]
    fn test_single_line_change() {
        let old = "line1\nline2\nline3";
        let new = "line1\nmodified\nline3";
        let result = compute_line_diff(old, new);
        assert!(!result.is_empty());
        assert_eq!(result.modified_count(), 1);
        assert_eq!(result.insert_count(), 0);
        assert_eq!(result.delete_count(), 0);

        // First and last lines unchanged
        assert!(matches!(
            result.ops[0],
            DiffOp::Unchanged {
                old_index: 0,
                new_index: 0
            }
        ));
        assert!(matches!(
            result.ops[2],
            DiffOp::Unchanged {
                old_index: 2,
                new_index: 2
            }
        ));

        // Middle line changed
        match &result.ops[1] {
            DiffOp::Changed {
                old_index,
                new_index,
                new_content,
            } => {
                assert_eq!(*old_index, 1);
                assert_eq!(*new_index, 1);
                assert_eq!(new_content, "modified");
            }
            other => panic!("Expected Changed, got {:?}", other),
        }
    }

    #[test]
    fn test_insert_at_end() {
        let old = "line1\nline2";
        let new = "line1\nline2\nline3";
        let result = compute_line_diff(old, new);
        assert_eq!(result.insert_count(), 1);
        assert_eq!(result.delete_count(), 0);

        let last = result.ops.last().unwrap();
        match last {
            DiffOp::Inserted {
                new_index,
                new_content,
            } => {
                assert_eq!(*new_index, 2);
                assert_eq!(new_content, "line3");
            }
            other => panic!("Expected Inserted, got {:?}", other),
        }
    }

    #[test]
    fn test_insert_at_beginning() {
        let old = "line2\nline3";
        let new = "line1\nline2\nline3";
        let result = compute_line_diff(old, new);
        assert_eq!(result.insert_count(), 1);
        assert_eq!(result.delete_count(), 0);

        let first = &result.ops[0];
        match first {
            DiffOp::Inserted {
                new_index,
                new_content,
            } => {
                assert_eq!(*new_index, 0);
                assert_eq!(new_content, "line1");
            }
            other => panic!("Expected Inserted, got {:?}", other),
        }
    }

    #[test]
    fn test_delete_first_line() {
        let old = "line1\nline2\nline3";
        let new = "line2\nline3";
        let result = compute_line_diff(old, new);
        assert_eq!(result.delete_count(), 1);
        assert_eq!(result.insert_count(), 0);

        let first = &result.ops[0];
        assert!(matches!(first, DiffOp::Deleted { old_index: 0 }));
    }

    #[test]
    fn test_delete_last_line() {
        let old = "line1\nline2\nline3";
        let new = "line1\nline2";
        let result = compute_line_diff(old, new);
        assert_eq!(result.delete_count(), 1);
        assert_eq!(result.insert_count(), 0);

        let last = result.ops.last().unwrap();
        assert!(matches!(last, DiffOp::Deleted { old_index: 2 }));
    }

    #[test]
    fn test_multi_line_change() {
        let old = "a\nb\nc\nd\ne";
        let new = "a\nB\nC\nd\ne";
        let result = compute_line_diff(old, new);
        assert_eq!(result.modified_count(), 2);
        assert_eq!(result.insert_count(), 0);
        assert_eq!(result.delete_count(), 0);
    }

    #[test]
    fn test_empty_to_content() {
        let old = "";
        let new = "line1\nline2";
        let result = compute_line_diff(old, new);
        // old has 1 line (""), new has 2 lines
        assert!(result.insert_count() + result.modified_count() > 0);
    }

    #[test]
    fn test_content_to_empty() {
        let old = "line1\nline2";
        let new = "";
        let result = compute_line_diff(old, new);
        // old has 2 lines, new has 1 line ("")
        assert!(result.delete_count() + result.modified_count() > 0);
    }

    #[test]
    fn test_completely_different() {
        let old = "aaa\nbbb\nccc";
        let new = "xxx\nyyy\nzzz";
        let result = compute_line_diff(old, new);
        assert_eq!(result.change_count(), 3);
        // All lines should be Changed (3 removed, 3 added → 3 Changed pairs)
        assert_eq!(result.modified_count(), 3);
    }

    #[test]
    fn test_insert_in_middle() {
        let old = "line1\nline3";
        let new = "line1\nline2\nline3";
        let result = compute_line_diff(old, new);
        assert_eq!(result.insert_count(), 1);
        assert_eq!(result.delete_count(), 0);
        assert_eq!(result.modified_count(), 0);
    }

    #[test]
    fn test_delete_in_middle() {
        let old = "line1\nline2\nline3";
        let new = "line1\nline3";
        let result = compute_line_diff(old, new);
        assert_eq!(result.delete_count(), 1);
        assert_eq!(result.insert_count(), 0);
        assert_eq!(result.modified_count(), 0);
    }

    #[test]
    fn test_realistic_rust_code() {
        let old = r#"fn main() {
    println!("hello");
    let x = 42;
    process(x);
}

fn process(val: i32) {
    println!("{}", val);
}"#;

        let new = r#"fn main() {
    println!("hello, world!");
    let x = 42;
    let y = 10;
    process(x, y);
}

fn process(val: i32, extra: i32) {
    println!("{} {}", val, extra);
}"#;

        let result = compute_line_diff(old, new);
        assert!(!result.is_empty());
        // Lines 1 (println changed), 4 (new let y), 5 (process call), 8 (fn sig), 9 (println) changed
        assert!(result.change_count() > 0);
    }

    #[test]
    fn test_diff_performance_300_lines_1_change() {
        // Build a 300-line file, change 1 line
        let lines: Vec<String> = (0..300)
            .map(|i| format!("line {} content here", i))
            .collect();
        let old = lines.join("\n");

        let mut new_lines = lines.clone();
        new_lines[150] = "CHANGED LINE 150".to_string();
        let new = new_lines.join("\n");

        let start = Instant::now();
        let result = compute_line_diff(&old, &new);
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_millis() < 10,
            "Diff of 300 lines with 1 change took {}ms (target: <10ms)",
            elapsed.as_millis()
        );
        assert_eq!(result.modified_count(), 1);
        assert_eq!(result.insert_count(), 0);
        assert_eq!(result.delete_count(), 0);
    }

    #[test]
    fn test_diff_performance_1000_lines() {
        let lines: Vec<String> = (0..1000)
            .map(|i| format!("line {} with some content", i))
            .collect();
        let old = lines.join("\n");

        // Change 10 scattered lines
        let mut new_lines = lines.clone();
        for i in (0..1000).step_by(100) {
            new_lines[i] = format!("CHANGED {}", i);
        }
        let new = new_lines.join("\n");

        let start = Instant::now();
        let result = compute_line_diff(&old, &new);
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_millis() < 50,
            "Diff of 1000 lines with 10 changes took {}ms (target: <50ms)",
            elapsed.as_millis()
        );
        assert_eq!(result.modified_count(), 10);
    }

    #[test]
    fn test_diff_op_counts_consistency() {
        let old = "a\nb\nc\nd\ne";
        let new = "a\nB\ninserted\nc\ne";
        let result = compute_line_diff(old, new);

        // Verify: unchanged + changed + deleted ops consume all old lines
        let old_consumed: usize = result
            .ops
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    DiffOp::Unchanged { .. } | DiffOp::Changed { .. } | DiffOp::Deleted { .. }
                )
            })
            .count();

        // Verify: unchanged + changed + inserted ops consume all new lines
        let new_consumed: usize = result
            .ops
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    DiffOp::Unchanged { .. } | DiffOp::Changed { .. } | DiffOp::Inserted { .. }
                )
            })
            .count();

        assert_eq!(
            old_consumed, result.old_line_count,
            "All old lines should be accounted for"
        );
        assert_eq!(
            new_consumed, result.new_line_count,
            "All new lines should be accounted for"
        );
    }

    #[test]
    fn test_trailing_newline_handling() {
        // Files with trailing newline
        let old = "line1\nline2\n";
        let new = "line1\nline2\n";
        let result = compute_line_diff(old, new);
        assert!(result.is_empty());

        // File gains trailing newline — split('\n') adds an extra empty element
        // "line1\nline2" → ["line1", "line2"] (2 lines)
        // "line1\nline2\n" → ["line1", "line2", ""] (3 lines)
        let old = "line1\nline2";
        let new = "line1\nline2\n";
        let result = compute_line_diff(old, new);
        assert_eq!(result.old_line_count, 2);
        assert_eq!(result.new_line_count, 3);
        // The diff should report some change (the new empty line)
        assert!(result.change_count() > 0);
    }
}
