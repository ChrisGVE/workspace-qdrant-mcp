/// Custom grep-searcher `Sink` that collects matches with before/after context lines.
///
/// grep-searcher interleaves context and match callbacks:
///   context(Before) → matched → context(After) → context_break → ...
///
/// We buffer context-before lines and attach context-after to the previous
/// match as they arrive.
use grep_searcher::{Searcher, Sink, SinkContextKind};

use crate::text_search::SearchMatch;

use super::types::FileInfo;

pub(super) struct ContextSink {
    file_path: String,
    tenant_id: String,
    branch: Option<String>,
    context_lines: usize,
    pub(super) matches: Vec<SearchMatch>,
    /// Pending context-before lines for the next match.
    pending_before: Vec<String>,
    /// How many context-after lines we've added to the last match.
    after_count: usize,
}

impl ContextSink {
    pub(super) fn new(file_info: &FileInfo, context_lines: usize) -> Self {
        Self {
            file_path: file_info.file_path.clone(),
            tenant_id: file_info.tenant_id.clone(),
            branch: file_info.branch.clone(),
            context_lines,
            matches: Vec::new(),
            pending_before: Vec::new(),
            after_count: 0,
        }
    }

    /// Consume and return all collected matches.
    pub(super) fn finish_collecting(self) -> Vec<SearchMatch> {
        self.matches
    }
}

impl Sink for ContextSink {
    type Error = std::io::Error;

    fn matched(
        &mut self,
        _searcher: &Searcher,
        mat: &grep_searcher::SinkMatch<'_>,
    ) -> Result<bool, Self::Error> {
        let content = String::from_utf8_lossy(mat.bytes())
            .trim_end_matches('\n')
            .to_string();
        let line_number = mat.line_number().unwrap_or(0) as i64;

        let context_before = std::mem::take(&mut self.pending_before);
        self.after_count = 0;

        self.matches.push(SearchMatch {
            line_id: 0,
            file_id: 0,
            line_number,
            content,
            file_path: self.file_path.clone(),
            tenant_id: self.tenant_id.clone(),
            branch: self.branch.clone(),
            context_before,
            context_after: vec![],
        });

        Ok(true)
    }

    fn context(
        &mut self,
        _searcher: &Searcher,
        context: &grep_searcher::SinkContext<'_>,
    ) -> Result<bool, Self::Error> {
        let line = String::from_utf8_lossy(context.bytes())
            .trim_end_matches('\n')
            .to_string();

        match context.kind() {
            &SinkContextKind::Before => {
                self.pending_before.push(line);
                if self.pending_before.len() > self.context_lines {
                    self.pending_before.remove(0);
                }
            }
            &SinkContextKind::After => {
                if let Some(last) = self.matches.last_mut() {
                    if self.after_count < self.context_lines {
                        last.context_after.push(line);
                        self.after_count += 1;
                    }
                }
            }
            &SinkContextKind::Other => {
                self.pending_before.push(line);
                if self.pending_before.len() > self.context_lines {
                    self.pending_before.remove(0);
                }
            }
        }

        Ok(true)
    }
}
