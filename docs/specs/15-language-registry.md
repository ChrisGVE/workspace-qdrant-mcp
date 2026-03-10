## Language Registry

The Language Registry is the central data-driven system for language support. All language knowledge — file extensions, grammar sources, semantic chunking patterns, LSP server metadata — is defined in YAML rather than Rust code. Adding support for a new language requires only a YAML definition.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  LanguageRegistry                     │
│  (merges all providers, serves LanguageDefinitions)  │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │   Bundled    │  │  Linguist    │  │  nvim-ts   │  │
│  │  (pri 255)   │  │  (pri 10)    │  │  (pri 20)  │  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  ts-grammars │  │    mason     │  │ User YAML  │  │
│  │  org (pri 15)│  │  (pri 30)    │  │ (highest)  │  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              GenericExtractor                         │
│  (reads SemanticPatterns, walks AST via patterns)    │
└─────────────────────────────────────────────────────┘
```

### Provider Abstraction

All providers implement the `LanguageSourceProvider` trait:

```rust
#[async_trait]
pub trait LanguageSourceProvider: Send + Sync {
    fn name(&self) -> &str;
    fn priority(&self) -> u8;               // Higher = more authoritative
    fn is_enabled(&self) -> bool;
    fn last_updated(&self) -> Option<DateTime<Utc>>;
    async fn fetch_languages(&self) -> Result<Vec<LanguageEntry>>;
    async fn fetch_grammars(&self) -> Result<Vec<GrammarEntry>>;
    async fn fetch_lsp_servers(&self) -> Result<Vec<LspEntry>>;
    async fn refresh(&self) -> Result<ProviderData>;
    fn full_definitions(&self) -> Option<Vec<LanguageDefinition>>;
}
```

### Providers

| Provider | Priority | Source | Data |
| --- | --- | --- | --- |
| **RegistryProvider** | 255 | Embedded YAML (compile-time) | Full definitions: identity, grammar, patterns, LSP |
| **MasonProvider** | 30 | mason-registry/registry.json | LSP server metadata with install methods |
| **NvimTreesitterProvider** | 20 | nvim-treesitter/lockfile.json | Grammar-to-repo mappings (200+ languages) |
| **TreeSitterGrammarsOrgProvider** | 15 | GitHub tree-sitter-grammars org | Curated grammar repos |
| **LinguistProvider** | 10 | GitHub Linguist languages.yml | Language identity (name, extensions, aliases, type) |

**Merge strategy:** Providers are loaded in reverse priority order (lowest first). Higher-priority providers overwrite identity fields. Grammar sources and LSP servers are deduplicated and accumulated. Semantic patterns come only from providers with `full_definitions()` (currently: RegistryProvider).

### YAML Schema

Each bundled language is defined in `assets/languages/*.yaml`:

```yaml
language: Python
aliases: [py, python3]
extensions: [".py", ".pyi", ".pyw"]
type: programming

grammar:
  sources:
    - repo: tree-sitter/tree-sitter-python
      quality: curated
  has_cpp_scanner: false

semantic_patterns:
  docstring_style: FirstStringInBody
  name_node: identifier
  body_node: block

  preamble:
    node_types: [import_statement, import_from_statement, future_import_statement]
  function:
    node_types: [function_definition]
    async_node_types: [function_definition]  # async keyword in children
  class:
    node_types: [class_definition]
  method:
    node_types: [function_definition]  # detected by parent context
  struct_def:
    node_types: []
  enum_def:
    node_types: []
  trait_def:
    node_types: []
  interface:
    node_types: []
  module:
    node_types: [module]
  constant:
    node_types: []
  macro_def:
    node_types: []
  type_alias:
    node_types: []
  impl_block:
    node_types: []

lsp_servers:
  - name: ruff
    binary: ruff
    args: [server]
    priority: 1
    install_methods:
      - manager: pip
        command: pip install ruff
      - manager: brew
        command: brew install ruff
  - name: pylsp
    binary: pylsp
    args: []
    priority: 2
    install_methods:
      - manager: pip
        command: pip install python-lsp-server
```

### GenericExtractor

The `GenericExtractor` replaces all per-language extractors (25 files, 8,300+ lines removed). It reads `SemanticPatterns` from the registry and:

1. **Classifies AST nodes** by matching `node.kind()` against pattern lists
2. **Extracts names** via configurable `name_node` (default: `"identifier"`)
3. **Extracts docstrings** using the language's `DocstringStyle` variant
4. **Extracts methods** from class/struct/impl body nodes
5. **Extracts preamble** from top-level import/use/include nodes

Supported `DocstringStyle` variants:
- `PrecedingComments` — comment nodes before the definition (Rust, C, Go, etc.)
- `FirstStringInBody` — first string expression in function body (Python)
- `Javadoc` — `/** ... */` preceding the definition (Java, JavaScript)
- `Haddock` — `-- |` or `{- | -}` comments (Haskell)
- `ElixirAttr` — `@doc` / `@moduledoc` attributes (Elixir)
- `OcamlDoc` — `(** ... *)` comments (OCaml)
- `Pod` — POD blocks (`=head`, `=cut`) (Perl)
- `None` — no docstring extraction

### CLI Commands

```bash
wqm language list [--installed] [--category <cat>] [--verbose]
wqm language info <lang>
wqm language status [<lang>] [--verbose]
wqm language refresh
wqm language ts-install <lang> [--force]
wqm language ts-remove <lang|all>
wqm language ts-search <lang>
wqm language lsp-install <lang>
wqm language lsp-remove <lang>
wqm language lsp-search <lang>
```

### Adding a New Language

To add support for a new language (e.g., COBOL):

1. Create `assets/languages/cobol.yaml` with the schema above
2. Define file extensions, grammar repo, and semantic patterns
3. Rebuild — the `RegistryProvider` embeds the YAML at compile time
4. No Rust code changes required

For user-local additions without rebuilding, place YAML files in `~/.workspace-qdrant/languages/`. These take highest precedence and override bundled definitions.

---
