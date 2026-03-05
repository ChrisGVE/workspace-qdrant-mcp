//! Core recursive-descent parsing logic for regex literal extraction.

use super::super::types::RegexLiterals;

/// Core extraction logic, separated for recursion on group contents.
pub(super) fn extract_literals_recursive(pattern: &str, result: &mut RegexLiterals) {
    let mut current = String::new();
    let mut chars = pattern.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '\\' => {
                // Escape sequence
                if let Some(&next) = chars.peek() {
                    match next {
                        // Metacharacter classes — end the current literal run
                        'd' | 'D' | 'w' | 'W' | 's' | 'S' | 'b' | 'B' | 'A' | 'z' | 'Z' | 'G' => {
                            flush_to_mandatory(&mut current, &mut result.mandatory);
                            chars.next();
                        }
                        // Escaped literals — add the literal character
                        _ => {
                            chars.next();
                            current.push(next);
                        }
                    }
                }
            }
            // Character class — skip everything until closing `]`
            '[' => {
                flush_to_mandatory(&mut current, &mut result.mandatory);
                while let Some(inner) = chars.next() {
                    if inner == '\\' {
                        chars.next();
                    } else if inner == ']' {
                        break;
                    }
                }
            }
            // Group start — extract group content, check for alternation
            '(' => {
                let prefix = std::mem::take(&mut current);
                if prefix.len() >= 3 {
                    result.mandatory.push(prefix.clone());
                }
                let group_content = extract_group_content(&mut chars);
                let suffix = collect_literal_suffix(&mut chars);
                process_group_with_affixes(&prefix, &suffix, &group_content, result);
                if suffix.len() >= 3 {
                    result.mandatory.push(suffix);
                }
            }
            // Alternation at top level — treat remaining pattern as alternate branch.
            '|' => {
                flush_to_mandatory(&mut current, &mut result.mandatory);
                let rest: String = chars.collect();
                let mut left_lits = std::mem::take(&mut result.mandatory);
                let mut right_result = RegexLiterals {
                    mandatory: Vec::new(),
                    alternations: Vec::new(),
                };
                extract_literals_recursive(&rest, &mut right_result);
                let mut right_lits = right_result.mandatory;
                result.alternations.extend(right_result.alternations);
                if !left_lits.is_empty() || !right_lits.is_empty() {
                    let mut group = Vec::new();
                    if !left_lits.is_empty() {
                        group.append(&mut left_lits);
                    }
                    if !right_lits.is_empty() {
                        // For top-level alternation, each side becomes a branch
                        // We need to restructure: put all left as one alt, all right as another
                    }
                    group.append(&mut right_lits);
                    if !group.is_empty() {
                        result.alternations.push(group);
                    }
                }
                return; // rest already consumed
            }
            // Other metacharacters that end a literal run
            '.' | '*' | '+' | '?' | ']' | ')' | '{' | '}' | '^' | '$' => {
                flush_to_mandatory(&mut current, &mut result.mandatory);
            }
            // Literal character
            _ => {
                current.push(ch);
            }
        }
    }

    flush_to_mandatory(&mut current, &mut result.mandatory);
}

/// Extract the content of a parenthesized group, handling nested parens.
pub(super) fn extract_group_content(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> String {
    let mut content = String::new();
    let mut depth = 1;
    while let Some(ch) = chars.next() {
        match ch {
            '(' => {
                depth += 1;
                content.push(ch);
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
                content.push(ch);
            }
            '\\' => {
                content.push(ch);
                if let Some(next) = chars.next() {
                    content.push(next);
                }
            }
            _ => content.push(ch),
        }
    }
    content
}

/// Collect literal characters immediately following a group close `)`.
pub(super) fn collect_literal_suffix(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> String {
    let mut suffix = String::new();
    while let Some(&ch) = chars.peek() {
        match ch {
            '\\' => {
                let mut lookahead = chars.clone();
                lookahead.next();
                if let Some(&next) = lookahead.peek() {
                    match next {
                        'd' | 'D' | 'w' | 'W' | 's' | 'S' | 'b' | 'B' | 'A' | 'z' | 'Z' | 'G' => {
                            break
                        }
                        _ => {
                            chars.next();
                            chars.next();
                            suffix.push(next);
                        }
                    }
                } else {
                    break;
                }
            }
            '.' | '*' | '+' | '?' | '[' | ']' | '(' | ')' | '{' | '}' | '|' | '^' | '$' => {
                break;
            }
            _ => {
                suffix.push(ch);
                chars.next();
            }
        }
    }
    suffix
}

/// Process a group's content with optional prefix/suffix affixes.
pub(super) fn process_group_with_affixes(
    prefix: &str,
    suffix: &str,
    content: &str,
    result: &mut RegexLiterals,
) {
    let branches = split_alternation(content);
    if branches.len() <= 1 {
        extract_literals_recursive(content, result);
    } else {
        let mut alt_group: Vec<String> = Vec::new();
        for branch in &branches {
            let mut branch_result = RegexLiterals {
                mandatory: Vec::new(),
                alternations: Vec::new(),
            };
            extract_literals_recursive(branch, &mut branch_result);
            if branch_result.mandatory.is_empty() {
                let combined = format!("{}{}{}", prefix, branch, suffix);
                if combined.len() >= 3 && is_all_literal(branch) {
                    alt_group.push(combined);
                }
            } else {
                for lit in &branch_result.mandatory {
                    let combined = format!("{}{}{}", prefix, lit, suffix);
                    if combined.len() >= 3 {
                        alt_group.push(combined);
                    } else if lit.len() >= 3 {
                        alt_group.push(lit.clone());
                    }
                }
            }
            result.alternations.extend(branch_result.alternations);
        }
        if !alt_group.is_empty() {
            result.alternations.push(alt_group);
        }
    }
}

/// Split a group's content by top-level `|` (respecting nested parens).
fn split_alternation(content: &str) -> Vec<String> {
    let mut branches = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let mut chars = content.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '(' => {
                depth += 1;
                current.push(ch);
            }
            ')' => {
                depth -= 1;
                current.push(ch);
            }
            '\\' => {
                current.push(ch);
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            }
            '|' if depth == 0 => {
                branches.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }
    branches.push(current);
    branches
}

/// Check if a string contains only literal characters (no regex metacharacters).
fn is_all_literal(s: &str) -> bool {
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => {
                if let Some(next) = chars.next() {
                    match next {
                        'd' | 'D' | 'w' | 'W' | 's' | 'S' | 'b' | 'B' | 'A' | 'z' | 'Z' | 'G' => {
                            return false
                        }
                        _ => {}
                    }
                }
            }
            '.' | '*' | '+' | '?' | '[' | ']' | '(' | ')' | '{' | '}' | '|' | '^' | '$' => {
                return false;
            }
            _ => {}
        }
    }
    true
}

/// Flush the current literal buffer into the mandatory list if >= 3 chars.
pub(super) fn flush_to_mandatory(current: &mut String, mandatory: &mut Vec<String>) {
    if current.len() >= 3 {
        mandatory.push(current.clone());
    }
    current.clear();
}
