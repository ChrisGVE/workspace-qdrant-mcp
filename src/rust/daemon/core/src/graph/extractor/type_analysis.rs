//! Type reference extraction and keyword/primitive detection for signatures.
//!
//! Parses function signatures to identify type references (used for USES_TYPE edges)
//! and filters out language keywords and primitive/builtin types.

/// Parse a qualified name like `self.method`, `module::function`, or `pkg.Func`.
///
/// Returns (optional_qualifier, base_name). The base_name is always the
/// last component which is the actual callable symbol.
pub fn parse_qualified_name(call: &str) -> (Option<String>, String) {
    let call = call.trim();
    if call.is_empty() {
        return (None, String::new());
    }

    // Rust/C++ qualified names: `foo::bar::baz`
    if let Some(pos) = call.rfind("::") {
        let qualifier = &call[..pos];
        let name = &call[pos + 2..];
        if !name.is_empty() {
            return (Some(qualifier.to_string()), name.to_string());
        }
    }

    // Method calls: `self.method`, `obj.method`, `pkg.Func`
    if let Some(pos) = call.rfind('.') {
        let qualifier = &call[..pos];
        let name = &call[pos + 1..];
        if !name.is_empty() {
            return (Some(qualifier.to_string()), name.to_string());
        }
    }

    // Simple unqualified name
    (None, call.to_string())
}

/// Extract type references from a function/method signature.
///
/// Uses simple regex-free parsing to find capitalized type names and
/// well-known generic wrappers. Not a full parser -- focuses on common
/// patterns that cover ~80% of real-world code.
pub fn extract_type_references(signature: &str, language: &str) -> Vec<String> {
    let mut types = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Tokenize the signature into potential type identifiers
    let tokens = tokenize_signature(signature);

    for token in &tokens {
        if is_type_name(token, language) && seen.insert(token.clone()) {
            types.push(token.clone());
        }
    }

    types
}

/// Tokenize a signature string into identifier-like tokens.
///
/// Splits on non-identifier characters, keeping only tokens that look
/// like identifiers (start with letter/underscore, contain alphanum/_).
fn tokenize_signature(sig: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in sig.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
        } else {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// Check if a token looks like a type name rather than a keyword or variable.
fn is_type_name(token: &str, language: &str) -> bool {
    if token.len() < 2 {
        return false;
    }

    // Skip language keywords
    if is_keyword(token, language) {
        return false;
    }

    // Skip common parameter names and primitives
    if is_primitive_or_builtin(token, language) {
        return false;
    }

    // In most languages, types start with uppercase
    let first = token.chars().next().unwrap();
    if first.is_uppercase() {
        return true;
    }

    // Rust generic types can be lowercase: `vec`, `option`, `result` -- but
    // canonically they're PascalCase. We only accept PascalCase for now.
    false
}

/// Check if a token is a language keyword (not a type).
fn is_keyword(token: &str, language: &str) -> bool {
    match language {
        "rust" => matches!(
            token,
            "fn" | "pub"
                | "self"
                | "Self"
                | "mut"
                | "let"
                | "const"
                | "static"
                | "async"
                | "await"
                | "impl"
                | "trait"
                | "struct"
                | "enum"
                | "type"
                | "where"
                | "for"
                | "in"
                | "if"
                | "else"
                | "match"
                | "return"
                | "mod"
                | "use"
                | "crate"
                | "super"
                | "dyn"
                | "ref"
                | "unsafe"
                | "extern"
        ),
        "python" => matches!(
            token,
            "def" | "self"
                | "cls"
                | "class"
                | "return"
                | "import"
                | "from"
                | "as"
                | "if"
                | "else"
                | "elif"
                | "for"
                | "in"
                | "while"
                | "with"
                | "try"
                | "except"
                | "raise"
                | "pass"
                | "lambda"
                | "yield"
                | "async"
                | "await"
                | "None"
                | "True"
                | "False"
        ),
        "javascript" | "typescript" | "tsx" | "jsx" => matches!(
            token,
            "function"
                | "const"
                | "let"
                | "var"
                | "return"
                | "if"
                | "else"
                | "for"
                | "while"
                | "class"
                | "extends"
                | "implements"
                | "import"
                | "export"
                | "default"
                | "new"
                | "this"
                | "super"
                | "async"
                | "await"
                | "yield"
                | "typeof"
                | "instanceof"
                | "void"
                | "null"
                | "undefined"
                | "true"
                | "false"
        ),
        "go" => matches!(
            token,
            "func" | "return"
                | "if"
                | "else"
                | "for"
                | "range"
                | "switch"
                | "case"
                | "type"
                | "struct"
                | "interface"
                | "package"
                | "import"
                | "var"
                | "const"
                | "defer"
                | "go"
                | "chan"
                | "select"
                | "nil"
                | "true"
                | "false"
                | "map"
        ),
        _ => false,
    }
}

/// Check if a token is a primitive type or common builtin.
fn is_primitive_or_builtin(token: &str, language: &str) -> bool {
    match language {
        "rust" => matches!(
            token,
            "i8" | "i16"
                | "i32"
                | "i64"
                | "i128"
                | "isize"
                | "u8"
                | "u16"
                | "u32"
                | "u64"
                | "u128"
                | "usize"
                | "f32"
                | "f64"
                | "bool"
                | "char"
                | "str"
        ),
        "python" => matches!(
            token,
            "int" | "float" | "str" | "bool" | "bytes" | "list" | "dict" | "set" | "tuple"
        ),
        "javascript" | "typescript" | "tsx" | "jsx" => matches!(
            token,
            "string" | "number" | "boolean" | "any" | "never" | "unknown" | "void" | "object"
        ),
        "go" => matches!(
            token,
            "int" | "int8"
                | "int16"
                | "int32"
                | "int64"
                | "uint"
                | "uint8"
                | "uint16"
                | "uint32"
                | "uint64"
                | "float32"
                | "float64"
                | "bool"
                | "string"
                | "byte"
                | "rune"
                | "error"
        ),
        _ => false,
    }
}
