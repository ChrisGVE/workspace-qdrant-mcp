//! Disambiguation path computation for multi-clone repositories

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Computes the disambiguation path for a new project
///
/// Given a new project path and existing projects with the same remote,
/// finds the first differing directory component from the common ancestor.
pub struct DisambiguationPathComputer;

impl DisambiguationPathComputer {
    /// Compute disambiguation path for a new project
    pub fn compute(new_path: &Path, existing_paths: &[PathBuf]) -> String {
        if existing_paths.is_empty() {
            return String::new();
        }

        let new_components: Vec<_> = new_path.components().collect();
        let mut min_common_idx = new_components.len();

        for existing_path in existing_paths {
            let existing_components: Vec<_> = existing_path.components().collect();

            let mut common_idx = 0;
            for (i, (a, b)) in new_components.iter().zip(&existing_components).enumerate() {
                if a != b {
                    common_idx = i;
                    break;
                }
                common_idx = i + 1;
            }

            min_common_idx = min_common_idx.min(common_idx);
        }

        if min_common_idx < new_components.len() {
            new_components[min_common_idx..]
                .iter()
                .map(|c| c.as_os_str().to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join("/")
        } else {
            new_path.to_string_lossy().to_string()
        }
    }

    /// Recompute disambiguation paths for all clones of a repository
    pub fn recompute_all(paths: &[PathBuf]) -> HashMap<PathBuf, String> {
        let mut result = HashMap::new();

        if paths.len() <= 1 {
            for path in paths {
                result.insert(path.clone(), String::new());
            }
            return result;
        }

        for path in paths {
            let others: Vec<_> = paths
                .iter()
                .filter(|p| *p != path)
                .cloned()
                .collect();

            let disambig = Self::compute(path, &others);
            result.insert(path.clone(), disambig);
        }

        result
    }
}
