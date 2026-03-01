# Language Support Test Specification

## Purpose

This test suite validates the workspace-qdrant-mcp language support pipeline across 25 programming languages. It identifies flaws, measures quality, and surfaces optimization opportunities in:

1. **Tree-sitter grammar management** — automatic download, caching, parsing
2. **Semantic chunking quality** — chunk sizes, boundary alignment, coverage
3. **LSP detection** — automatic server discovery and startup
4. **LSP enrichment** — symbol resolution, type info, reference extraction
5. **LSP fallback** — graceful degradation when LSP is unavailable

## Current System Capabilities

As of v0.1.0-beta1, the system has:

| Capability | Languages |
|---|---|
| Tree-sitter grammar (compiled-in) | C, C++, Go, Java, JavaScript, Python, Rust, TypeScript |
| LSP detection + enrichment | C, C++, Go, Python, Rust, TypeScript |
| Dynamic grammar download | Infrastructure exists, not yet populated |
| Text fallback chunking | All other extensions in the allowlist |

This test suite exercises all three tiers: full support, partial support, and fallback-only.

## The Bookshelf Project

### Concept

A small library management system with 4 source files per language. The project exercises data structures, business logic, utilities, and cross-file imports — the building blocks our chunking and analysis pipeline must handle.

### Canonical File Structure

Each language implementation has 4 logical units:

#### 1. Models (data structures)

```
struct Book {
    title: String
    author: String
    year: Integer
    isbn: String
    available: Boolean
}

struct Shelf {
    name: String
    capacity: Integer
    books: List<Book>
}

enum Genre {
    Fiction
    NonFiction
    Science
    History
    Biography
    Technology
    Philosophy
}
```

Requirements:
- Book must have all 5 fields with the types above
- Shelf must contain a books collection, name, and capacity
- Genre must have exactly 7 variants
- Include constructors or factory functions where idiomatic
- File should be 40-80 lines (language-dependent)

#### 2. Storage (business logic)

Functions:
- `add_book(shelf, book) -> Result/Error` — adds book to shelf, error if at capacity
- `remove_book(shelf, isbn) -> Result/Error` — removes by ISBN, error if not found
- `find_by_author(shelf, author) -> List<Book>` — case-insensitive substring match
- `find_by_year_range(shelf, start_year, end_year) -> List<Book>` — inclusive range
- `sort_by_title(shelf) -> List<Book>` — returns sorted copy, does not mutate
- `is_full(shelf) -> Boolean` — capacity check
- `generate_report(shelf) -> String` — multi-line formatted report (see below)

The `generate_report` function must be 30-50 lines and produce output like:
```
=== Library Report: [shelf name] ===
Total books: N
Available: N / N (XX%)
Capacity: N / N (XX% full)

Authors (N unique):
  - Author Name (N books)
  - ...

Year range: YYYY - YYYY

Books by availability:
  [+] Title by Author (YYYY) - ISBN
  [-] Title by Author (YYYY) - ISBN
```

Requirements:
- All functions take shelf as first parameter (or method on shelf)
- Error handling must be explicit (Result types, exceptions, error codes)
- `find_by_author` must be case-insensitive
- `sort_by_title` must not mutate the original shelf
- File should be 100-180 lines (language-dependent)

#### 3. Utils (helper functions)

Functions:
- `validate_isbn(isbn) -> Boolean` — ISBN-13 check digit validation using the standard algorithm:
  1. ISBN must be exactly 13 digits
  2. Sum: alternate multiply by 1 and 3, sum all products
  3. Valid if sum mod 10 == 0
- `format_book(book) -> String` — single-line format: `"Title" by Author (YYYY) [ISBN: XXXXXXXXXXXXX]`
- `parse_csv_line(line) -> Result<Book, Error>` — parse `title,author,year,isbn,available` where available is `true`/`false`

Requirements:
- ISBN validation must implement the real ISBN-13 algorithm
- `parse_csv_line` must handle basic error cases (wrong field count, invalid year, invalid boolean)
- File should be 40-80 lines

#### 4. Main (entry point and demonstration)

Must:
1. Create a shelf named "Computer Science" with capacity 10
2. Add these books:
   - "The Art of Computer Programming" by Donald Knuth, 1968, ISBN 9780201896831, available=true
   - "Structure and Interpretation of Computer Programs" by Harold Abelson, 1996, ISBN 9780262510875, available=true
   - "Introduction to Algorithms" by Thomas Cormen, 2009, ISBN 9780262033848, available=false
   - "Design Patterns" by Erich Gamma, 1994, ISBN 9780201633610, available=true
   - "The Pragmatic Programmer" by David Thomas, 2019, ISBN 9780135957059, available=true
3. Print the report
4. Search by author "knuth" (case-insensitive test)
5. Search by year range 1990-2010
6. Parse a CSV line: `"Clean Code,Robert Martin,2008,9780132350884,true"`
7. Validate ISBN of each book
8. Print results of each operation

Requirements:
- Must import from all other modules (cross-file dependency)
- Output must be deterministic (same every run)
- File should be 40-80 lines

### Language-Specific Adaptations

Some languages require structural adjustments:

| Language | Adaptation |
|---|---|
| Shell (Bash) | Use associative arrays or positional fields instead of structs. Enum as constants. Capacity as global. |
| Perl | Use hash references for structs. OO via bless or Moo/Moose if available. |
| Lua | Use tables for structs. Modules via return table pattern. |
| Fortran | Use derived types for structs. Modules with use statements. |
| Lisp | Use defstruct or CLOS classes. Packages for modules. |
| Clojure | Use maps/records for data. Namespaces for modules. |
| Erlang | Use records or maps. Modules with -export. |
| Pascal | Use record types and units. |

For all languages: maintain functional equivalence even if structural representation differs.

### Sample Book Data

All implementations must use these exact books (for deterministic output comparison):

| Title | Author | Year | ISBN | Available |
|---|---|---|---|---|
| The Art of Computer Programming | Donald Knuth | 1968 | 9780201896831 | true |
| Structure and Interpretation of Computer Programs | Harold Abelson | 1996 | 9780262510875 | true |
| Introduction to Algorithms | Thomas Cormen | 2009 | 9780262033848 | false |
| Design Patterns | Erich Gamma | 1994 | 9780201633610 | true |
| The Pragmatic Programmer | David Thomas | 2019 | 9780135957059 | true |

Note: These ISBNs are the real ISBN-13 values for these books and pass check digit validation.

## Repository Layout

```
tests/language-support/
├── SPEC.md                    # This file
├── FINDINGS.md                # Results analysis (generated after validation)
├── validate.sh                # Orchestration script
├── helpers/
│   ├── qdrant_query.sh        # Qdrant REST API helpers
│   ├── report_generator.sh    # YAML report generation
│   └── expected_output.txt    # Reference output for comparison
├── results/                   # Per-language YAML reports (gitignored)
│   └── .gitkeep
├── ada/
│   ├── src/
│   │   ├── models.ads / models.adb
│   │   ├── storage.ads / storage.adb
│   │   ├── utils.ads / utils.adb
│   │   └── main.adb
│   ├── build.sh
│   └── run.sh
├── c/
│   ├── src/
│   │   ├── models.h / models.c
│   │   ├── storage.h / storage.c
│   │   ├── utils.h / utils.c
│   │   └── main.c
│   ├── Makefile
│   └── run.sh
├── clojure/
│   ├── src/bookshelf/
│   │   ├── models.clj
│   │   ├── storage.clj
│   │   ├── utils.clj
│   │   └── core.clj
│   ├── project.clj
│   └── run.sh
├── cpp/
│   ├── src/
│   │   ├── models.hpp / models.cpp
│   │   ├── storage.hpp / storage.cpp
│   │   ├── utils.hpp / utils.cpp
│   │   └── main.cpp
│   ├── Makefile
│   └── run.sh
├── elixir/
│   ├── lib/
│   │   ├── models.ex
│   │   ├── storage.ex
│   │   ├── utils.ex
│   │   └── bookshelf.ex
│   ├── mix.exs
│   └── run.sh
├── erlang/
│   ├── src/
│   │   ├── models.erl
│   │   ├── storage.erl
│   │   ├── utils.erl
│   │   └── main.erl
│   ├── rebar.config
│   └── run.sh
├── fortran/
│   ├── src/
│   │   ├── models.f90
│   │   ├── storage.f90
│   │   ├── utils.f90
│   │   └── main.f90
│   ├── build.sh
│   └── run.sh
├── go/
│   ├── models.go
│   ├── storage.go
│   ├── utils.go
│   ├── main.go
│   ├── go.mod
│   └── run.sh
├── haskell/
│   ├── src/
│   │   ├── Models.hs
│   │   ├── Storage.hs
│   │   ├── Utils.hs
│   │   └── Main.hs
│   ├── bookshelf.cabal
│   └── run.sh
├── java/
│   ├── src/bookshelf/
│   │   ├── Models.java
│   │   ├── Storage.java
│   │   ├── Utils.java
│   │   └── Main.java
│   ├── build.sh
│   └── run.sh
├── javascript/
│   ├── src/
│   │   ├── models.js
│   │   ├── storage.js
│   │   ├── utils.js
│   │   └── main.js
│   ├── package.json
│   └── run.sh
├── lisp/
│   ├── src/
│   │   ├── models.lisp
│   │   ├── storage.lisp
│   │   ├── utils.lisp
│   │   └── main.lisp
│   ├── bookshelf.asd
│   └── run.sh
├── lua/
│   ├── models.lua
│   ├── storage.lua
│   ├── utils.lua
│   ├── main.lua
│   └── run.sh
├── ocaml/
│   ├── lib/
│   │   ├── models.ml
│   │   ├── storage.ml
│   │   └── utils.ml
│   ├── bin/
│   │   └── main.ml
│   ├── dune-project
│   └── run.sh
├── odin/
│   ├── models.odin
│   ├── storage.odin
│   ├── utils.odin
│   ├── main.odin
│   └── run.sh
├── pascal/
│   ├── src/
│   │   ├── models.pas
│   │   ├── storage.pas
│   │   ├── utils.pas
│   │   └── main.pas
│   ├── build.sh
│   └── run.sh
├── perl/
│   ├── lib/
│   │   ├── Models.pm
│   │   ├── Storage.pm
│   │   └── Utils.pm
│   ├── main.pl
│   └── run.sh
├── python/
│   ├── bookshelf/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── storage.py
│   │   └── utils.py
│   ├── main.py
│   └── run.sh
├── ruby/
│   ├── lib/
│   │   ├── models.rb
│   │   ├── storage.rb
│   │   └── utils.rb
│   ├── main.rb
│   └── run.sh
├── rust/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs
│   │   ├── models.rs
│   │   ├── storage.rs
│   │   └── utils.rs
│   └── run.sh
├── scala/
│   ├── src/main/scala/bookshelf/
│   │   ├── Models.scala
│   │   ├── Storage.scala
│   │   ├── Utils.scala
│   │   └── Main.scala
│   ├── build.sbt
│   └── run.sh
├── shell/
│   ├── models.sh
│   ├── storage.sh
│   ├── utils.sh
│   ├── main.sh
│   └── run.sh
├── swift/
│   ├── Sources/Bookshelf/
│   │   ├── Models.swift
│   │   ├── Storage.swift
│   │   ├── Utils.swift
│   │   └── main.swift
│   ├── Package.swift
│   └── run.sh
├── typescript/
│   ├── src/
│   │   ├── models.ts
│   │   ├── storage.ts
│   │   ├── utils.ts
│   │   └── main.ts
│   ├── tsconfig.json
│   ├── package.json
│   └── run.sh
└── zig/
    ├── src/
    │   ├── models.zig
    │   ├── storage.zig
    │   ├── utils.zig
    │   └── main.zig
    ├── build.zig
    └── run.sh
```

## Validation Methodology

### Phase 1: Compilation and Execution

For each language:
1. Run `build.sh` (if present) — record exit code, stderr
2. Run `run.sh` — capture stdout, record exit code
3. Compare stdout against `helpers/expected_output.txt` (fuzzy match allowing formatting differences)

**Metrics:**
- `compilation_pass`: boolean
- `execution_pass`: boolean
- `output_match`: boolean (fuzzy comparison)
- `errors`: list of error messages

### Phase 2: Ingestion

Register the language subfolder as a workspace-qdrant project:
1. Use `wqm project register --path <language_dir>` (or equivalent)
2. Wait for queue processing to complete: poll `wqm queue stats` until pending=0
3. Query tracked files: `wqm project files --tenant <tenant_id>`

**Metrics:**
- `files_detected`: integer (expected: 4 source files)
- `files_processed`: integer
- `files_failed`: integer
- `processing_errors`: list of error messages
- `ingestion_time_seconds`: float

### Phase 3: Tree-sitter Analysis

Query Qdrant for all points from this project's tenant:
1. Use Qdrant REST API: `POST /collections/projects/points/scroll` with tenant filter
2. Analyze chunks:
   - Count chunks per file
   - Extract token counts from metadata
   - Check chunk type labels (function, struct, class, enum, preamble, text)
   - Verify chunk boundaries align with source code structure

**Metrics:**
- `grammar_available`: boolean (check if tree-sitter or text fallback was used)
- `grammar_auto_downloaded`: boolean
- `chunks_total`: integer
- `chunks_per_file`: map of filename to count
- `chunks_min_tokens`: integer
- `chunks_max_tokens`: integer
- `chunks_mean_tokens`: float
- `chunk_types`: map of type to count (function, struct, enum, text, etc.)
- `boundary_alignment`: good | fair | poor
  - good: all chunks align with function/struct/class boundaries
  - fair: most chunks align, some mid-function splits
  - poor: arbitrary text-based splitting
- `coverage`: percentage of source functions appearing in chunks

### Phase 4: LSP Analysis

Check LSP detection and enrichment for this language:
1. Query tracked_files for LSP metadata columns
2. Query Qdrant points for `lsp_enrichment_status` field
3. Check daemon logs for LSP server startup attempts

**Metrics:**
- `lsp_detected`: boolean
- `lsp_server_name`: string or null
- `enrichment_attempted`: boolean
- `enrichment_rate`: percentage (points with successful LSP enrichment / total points)
- `symbols_resolved`: integer
- `references_found`: integer
- `type_info_found`: integer
- `fallback_to_treesitter`: boolean (enrichment skipped, tree-sitter chunks used)

### Phase 5: Search Quality

Run 4 standardized search queries:
1. **Semantic: function search** — "find books by a specific author" → expect `find_by_author` in top 3
2. **Semantic: utility search** — "validate an ISBN number" → expect `validate_isbn` in top 3
3. **Exact: function name** — grep for `generate_report` → expect storage file match
4. **Semantic: data structure** — "book data structure with title and author" → expect models file in top 3

**Metrics:**
- `search_semantic_function`: hit | miss (was target in top 3?)
- `search_semantic_utility`: hit | miss
- `search_exact_function`: hit | miss
- `search_semantic_structure`: hit | miss
- `search_precision`: N/4 (count of hits)

## Report Format

### Per-Language Report (`results/<language>.yaml`)

```yaml
language: <language_name>
timestamp: <ISO 8601>
phase1_compilation:
  compilation_pass: true
  execution_pass: true
  output_match: true
  errors: []
phase2_ingestion:
  files_detected: 4
  files_processed: 4
  files_failed: 0
  processing_errors: []
  ingestion_time_seconds: 2.3
phase3_treesitter:
  grammar_available: true
  grammar_auto_downloaded: false
  chunks_total: 14
  chunks_per_file:
    models.rs: 4
    storage.rs: 6
    utils.rs: 3
    main.rs: 1
  chunks_min_tokens: 28
  chunks_max_tokens: 185
  chunks_mean_tokens: 87.5
  chunk_types:
    function: 8
    struct: 2
    enum: 1
    preamble: 2
    impl: 1
  boundary_alignment: good
  coverage: 100
phase4_lsp:
  lsp_detected: true
  lsp_server_name: rust-analyzer
  enrichment_attempted: true
  enrichment_rate: 85
  symbols_resolved: 12
  references_found: 24
  type_info_found: 10
  fallback_to_treesitter: false
phase5_search:
  search_semantic_function: hit
  search_semantic_utility: hit
  search_exact_function: hit
  search_semantic_structure: hit
  search_precision: 4
verdict: PASS
issues: []
improvements: []
```

### Verdict Criteria

| Verdict | Criteria |
|---|---|
| **PASS** | Compilation OK, tree-sitter parses correctly, chunks are logical, LSP works (where expected), search returns relevant results |
| **PARTIAL** | Compilation OK, but: chunks too small/large, boundary misalignment, low LSP enrichment, or search misses |
| **FAIL** | Compilation fails, tree-sitter grammar unavailable when it should be, no chunking produced, or search returns irrelevant results |

For languages without tree-sitter support:
- `PASS` if text fallback produces reasonable chunks and search works
- The absence of tree-sitter is recorded as an `improvement` opportunity, not a `failure`

### Summary Report (`results/summary.yaml`)

```yaml
timestamp: <ISO 8601>
total_languages: 25
verdicts:
  pass: N
  partial: N
  fail: N
treesitter_support:
  compiled_in: [list of languages]
  auto_downloaded: [list of languages]
  text_fallback: [list of languages]
  failed: [list of languages]
lsp_support:
  detected: [list of languages]
  enriched: [list of languages]
  skipped: [list of languages]
search_quality:
  mean_precision: N.N
  perfect_scores: [list of languages with 4/4]
  zero_scores: [list of languages with 0/4]
issues_by_severity:
  critical: N
  high: N
  medium: N
  low: N
top_improvements: [prioritized list of improvement recommendations]
```

## Expected Outcomes by Language Tier

### Tier A: Full Support (tree-sitter + LSP)
**Languages**: C, C++, Go, Python, Rust, TypeScript

Expected: PASS verdict with semantic chunking and LSP enrichment. These are the gold standard.

### Tier B: Tree-sitter Only
**Languages**: Java, JavaScript

Expected: PASS with semantic chunking but no/limited LSP enrichment. JavaScript shares TypeScript LSP infrastructure but may behave differently.

### Tier C: Text Fallback Only
**Languages**: Ada, Clojure, Elixir, Erlang, Fortran, Haskell, Lisp, Lua, OCaml, Odin, Pascal, Perl, Ruby, Scala, Shell, Swift, Zig

Expected: PARTIAL verdict — text-based chunking will work but chunk boundaries will not align with function/struct boundaries. Search quality may be lower. Each finding becomes an improvement opportunity.

## Running the Tests

```bash
# Full validation (all languages)
./tests/language-support/validate.sh

# Single language
./tests/language-support/validate.sh --language rust

# Specific phase only
./tests/language-support/validate.sh --language rust --phase 3

# Verbose output
./tests/language-support/validate.sh --verbose

# Generate summary only (from existing results)
./tests/language-support/validate.sh --summary-only
```

## Notes

- Results in `results/` are gitignored (environment-specific). Only the summary FINDINGS.md is committed.
- The expected_output.txt provides a reference for output comparison but allows formatting differences.
- ISBNs in the sample data are real and pass validation.
- Languages may require toolchain installation on the test machine (compilers, interpreters, build tools). The validate.sh script should detect and report missing toolchains rather than failing silently.
