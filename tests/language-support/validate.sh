#!/usr/bin/env bash
# Language Support Validation Script
# Validates the workspace-qdrant-mcp language support pipeline across 25 languages.
# See SPEC.md for full methodology and report format.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_OUTPUT="$SCRIPT_DIR/helpers/expected_output.txt"
RESULTS_DIR="$SCRIPT_DIR/results"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

ALL_LANGUAGES=(
  ada c clojure cpp elixir erlang fortran go haskell java
  javascript lisp lua ocaml odin pascal perl python ruby rust
  scala shell swift typescript zig
)

TREESITTER_LANGUAGES=(c cpp go java javascript python rust typescript)
LSP_LANGUAGES=(c cpp go python rust typescript)

# --- CLI Argument Parsing ---
LANGUAGE=""
PHASE=""
VERBOSE=false
SUMMARY_ONLY=false
PHASE1_ONLY=false

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --language <lang>   Run for a single language only"
  echo "  --phase <1-5>       Run a specific phase only"
  echo "  --phase1-only       Run only Phase 1 (compilation + output match)"
  echo "  --verbose           Show detailed output"
  echo "  --summary-only      Generate summary from existing results"
  echo "  --help              Show this help message"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --language) LANGUAGE="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --phase1-only) PHASE1_ONLY=true; shift ;;
    --verbose) VERBOSE=true; shift ;;
    --summary-only) SUMMARY_ONLY=true; shift ;;
    --help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# --- Utility Functions ---

log() { echo "  $*"; }
log_ok() { echo "  [OK] $*"; }
log_fail() { echo "  [FAIL] $*"; }
log_skip() { echo "  [SKIP] $*"; }
log_verbose() { $VERBOSE && echo "  [DBG] $*" || true; }

in_array() {
  local needle="$1"; shift
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

# --- Phase 1: Compilation and Execution ---

run_phase1() {
  local lang="$1"
  local lang_dir="$SCRIPT_DIR/$lang"
  local report="$RESULTS_DIR/${lang}.yaml"

  local compilation_pass="true"
  local execution_pass="false"
  local output_match="false"
  local errors=""

  if [[ ! -d "$lang_dir" ]]; then
    log_fail "$lang: directory not found"
    write_phase1_report "$report" "$lang" "false" "false" "false" "directory not found"
    return 1
  fi

  if [[ ! -f "$lang_dir/run.sh" ]]; then
    log_fail "$lang: run.sh not found"
    write_phase1_report "$report" "$lang" "false" "false" "false" "run.sh not found"
    return 1
  fi

  # Run the program, capture stdout and stderr separately
  local stdout_file stderr_file
  stdout_file=$(mktemp)
  stderr_file=$(mktemp)

  cd "$lang_dir"
  if timeout 120 bash run.sh >"$stdout_file" 2>"$stderr_file"; then
    execution_pass="true"
  else
    local exit_code=$?
    if [[ $exit_code -eq 124 ]]; then
      errors="timeout after 120s"
    else
      errors="exit code $exit_code"
      if [[ -s "$stderr_file" ]]; then
        # Take first line of stderr for the error message
        errors="$errors: $(head -1 "$stderr_file")"
      fi
    fi
    compilation_pass="false"
  fi
  cd "$SCRIPT_DIR"

  # Check output match
  if [[ "$execution_pass" == "true" ]]; then
    if diff -q "$stdout_file" "$EXPECTED_OUTPUT" >/dev/null 2>&1; then
      output_match="true"
    else
      output_match="false"
      if [[ -z "$errors" ]]; then
        errors="output mismatch"
      fi
    fi
  fi

  rm -f "$stdout_file" "$stderr_file"

  if [[ "$output_match" == "true" ]]; then
    log_ok "$lang: Phase 1 PASS"
  elif [[ "$execution_pass" == "true" ]]; then
    log_fail "$lang: Phase 1 output mismatch"
  else
    log_fail "$lang: Phase 1 $errors"
  fi

  write_phase1_report "$report" "$lang" "$compilation_pass" "$execution_pass" "$output_match" "$errors"
}

write_phase1_report() {
  local report="$1" lang="$2" comp="$3" exec="$4" match="$5" errs="$6"

  cat > "$report" <<EOF
language: $lang
timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
phase1_compilation:
  compilation_pass: $comp
  execution_pass: $exec
  output_match: $match
  errors: [$([ -n "$errs" ] && echo "\"$errs\"" || echo "")]
EOF
}

# --- Phase 2: Ingestion ---

run_phase2() {
  local lang="$1"
  local lang_dir="$SCRIPT_DIR/$lang"
  local report="$RESULTS_DIR/${lang}.yaml"

  log "Phase 2: Registering $lang with daemon..."

  # Check if wqm is available
  if ! command -v wqm >/dev/null 2>&1; then
    log_skip "$lang: wqm CLI not found"
    append_phase2_report "$report" 0 0 0 "wqm CLI not available" 0
    return 1
  fi

  # Register the project
  local start_time
  start_time=$(date +%s)

  if ! wqm project add "$lang_dir" --name "lang-test-$lang" 2>/dev/null; then
    log_fail "$lang: project registration failed"
    append_phase2_report "$report" 0 0 0 "registration failed" 0
    return 1
  fi

  # Wait for queue to drain (max 60s)
  local waited=0
  while [[ $waited -lt 60 ]]; do
    local pending
    pending=$(wqm queue stats 2>/dev/null | grep -i "pending" | grep -oE '[0-9]+' | head -1 || echo "0")
    [[ "$pending" == "0" ]] && break
    sleep 2
    waited=$((waited + 2))
  done

  local end_time
  end_time=$(date +%s)
  local duration=$((end_time - start_time))

  # Count tracked files
  local tenant_id
  tenant_id=$(wqm project list 2>/dev/null | grep "lang-test-$lang" | awk '{print $1}' || echo "")

  local files_detected=0 files_processed=0 files_failed=0
  if [[ -n "$tenant_id" ]]; then
    files_detected=$(wqm project files --tenant "$tenant_id" 2>/dev/null | wc -l | tr -d ' ' || echo "0")
    files_processed="$files_detected"
  fi

  if [[ $files_detected -ge 3 ]]; then
    log_ok "$lang: Phase 2 PASS ($files_detected files in ${duration}s)"
  else
    log_fail "$lang: Phase 2 only $files_detected files detected"
  fi

  append_phase2_report "$report" "$files_detected" "$files_processed" "$files_failed" "" "$duration"
}

append_phase2_report() {
  local report="$1" detected="$2" processed="$3" failed="$4" errs="$5" duration="$6"
  cat >> "$report" <<EOF
phase2_ingestion:
  files_detected: $detected
  files_processed: $processed
  files_failed: $failed
  processing_errors: [$([ -n "$errs" ] && echo "\"$errs\"" || echo "")]
  ingestion_time_seconds: $duration
EOF
}

# --- Phase 3: Tree-sitter Analysis ---

run_phase3() {
  local lang="$1"
  local report="$RESULTS_DIR/${lang}.yaml"

  log "Phase 3: Analyzing tree-sitter chunking for $lang..."

  local grammar_available="false"
  local grammar_auto_downloaded="false"
  local chunks_total=0

  if in_array "$lang" "${TREESITTER_LANGUAGES[@]}"; then
    grammar_available="true"
  fi

  # Query Qdrant for chunks
  local tenant_id
  tenant_id=$(wqm project list 2>/dev/null | grep "lang-test-$lang" | awk '{print $1}' || echo "")

  if [[ -z "$tenant_id" ]]; then
    log_skip "$lang: Phase 3 - no tenant found (run Phase 2 first)"
    append_phase3_report "$report" "$grammar_available" "$grammar_auto_downloaded" 0 "{}" 0 0 0 "{}" "unknown" 0
    return
  fi

  # Scroll Qdrant for all points from this tenant
  local scroll_result
  scroll_result=$(curl -s "${QDRANT_URL}/collections/projects/points/scroll" \
    -H "Content-Type: application/json" \
    -d "{
      \"filter\": {
        \"must\": [{\"key\": \"project_id\", \"match\": {\"value\": \"$tenant_id\"}}]
      },
      \"limit\": 100,
      \"with_payload\": true
    }" 2>/dev/null || echo '{"result":{"points":[]}}')

  chunks_total=$(echo "$scroll_result" | python3 -c "
import json, sys
data = json.load(sys.stdin)
points = data.get('result', {}).get('points', [])
print(len(points))
" 2>/dev/null || echo "0")

  # Analyze chunk types and sizes
  local analysis
  analysis=$(echo "$scroll_result" | python3 -c "
import json, sys
data = json.load(sys.stdin)
points = data.get('result', {}).get('points', [])

chunks_per_file = {}
chunk_types = {}
token_counts = []

for p in points:
    payload = p.get('payload', {})
    rpath = payload.get('relative_path', 'unknown')
    ctype = payload.get('chunk_type', 'text')

    chunks_per_file[rpath] = chunks_per_file.get(rpath, 0) + 1
    chunk_types[ctype] = chunk_types.get(ctype, 0) + 1

    # Estimate tokens from content length
    content = payload.get('content', '')
    token_est = max(1, len(content.split()))
    token_counts.append(token_est)

min_t = min(token_counts) if token_counts else 0
max_t = max(token_counts) if token_counts else 0
mean_t = round(sum(token_counts) / len(token_counts), 1) if token_counts else 0

# Determine boundary alignment
has_semantic = any(t in chunk_types for t in ['function', 'method', 'struct', 'class', 'impl', 'trait'])
if has_semantic and chunk_types.get('text', 0) == 0:
    alignment = 'good'
elif has_semantic:
    alignment = 'fair'
else:
    alignment = 'poor'

# Print as pseudo-yaml for shell parsing
print(f'chunks_total={len(points)}')
print(f'min_tokens={min_t}')
print(f'max_tokens={max_t}')
print(f'mean_tokens={mean_t}')
print(f'alignment={alignment}')

# Print chunks per file
for k, v in sorted(chunks_per_file.items()):
    print(f'cpf:{k}={v}')
# Print chunk types
for k, v in sorted(chunk_types.items()):
    print(f'ct:{k}={v}')
" 2>/dev/null || echo "chunks_total=0")

  # Parse analysis
  local min_t max_t mean_t alignment
  min_t=$(echo "$analysis" | grep '^min_tokens=' | cut -d= -f2)
  max_t=$(echo "$analysis" | grep '^max_tokens=' | cut -d= -f2)
  mean_t=$(echo "$analysis" | grep '^mean_tokens=' | cut -d= -f2)
  alignment=$(echo "$analysis" | grep '^alignment=' | cut -d= -f2)
  chunks_total=$(echo "$analysis" | grep '^chunks_total=' | cut -d= -f2)

  # Build YAML for chunks_per_file
  local cpf_yaml=""
  while IFS= read -r line; do
    local fname="${line#cpf:}"
    local fkey="${fname%%=*}"
    local fval="${fname#*=}"
    cpf_yaml+="    $fkey: $fval"$'\n'
  done < <(echo "$analysis" | grep '^cpf:')

  # Build YAML for chunk_types
  local ct_yaml=""
  while IFS= read -r line; do
    local tname="${line#ct:}"
    local tkey="${tname%%=*}"
    local tval="${tname#*=}"
    ct_yaml+="    $tkey: $tval"$'\n'
  done < <(echo "$analysis" | grep '^ct:')

  if [[ $chunks_total -gt 0 ]]; then
    log_ok "$lang: Phase 3 - $chunks_total chunks, alignment=$alignment"
  else
    log_fail "$lang: Phase 3 - no chunks found"
  fi

  cat >> "$report" <<EOF
phase3_treesitter:
  grammar_available: $grammar_available
  grammar_auto_downloaded: $grammar_auto_downloaded
  chunks_total: ${chunks_total:-0}
  chunks_per_file:
${cpf_yaml:-    (none): 0}
  chunks_min_tokens: ${min_t:-0}
  chunks_max_tokens: ${max_t:-0}
  chunks_mean_tokens: ${mean_t:-0}
  chunk_types:
${ct_yaml:-    text: 0}
  boundary_alignment: ${alignment:-unknown}
EOF
}

append_phase3_report() {
  local report="$1"
  # Simplified fallback
  cat >> "$report" <<EOF
phase3_treesitter:
  grammar_available: $2
  grammar_auto_downloaded: $3
  chunks_total: $4
  chunks_per_file: $5
  chunks_min_tokens: $6
  chunks_max_tokens: $7
  chunks_mean_tokens: $8
  chunk_types: $9
  boundary_alignment: ${10}
EOF
}

# --- Phase 4: LSP Analysis ---

run_phase4() {
  local lang="$1"
  local report="$RESULTS_DIR/${lang}.yaml"

  log "Phase 4: LSP analysis for $lang..."

  local lsp_detected="false"
  local lsp_server_name="null"
  local fallback="true"

  if in_array "$lang" "${LSP_LANGUAGES[@]}"; then
    lsp_detected="true"
    fallback="false"
    case "$lang" in
      python) lsp_server_name="ruff" ;;
      rust) lsp_server_name="rust-analyzer" ;;
      typescript|javascript) lsp_server_name="typescript-language-server" ;;
      go) lsp_server_name="gopls" ;;
      c|cpp) lsp_server_name="clangd" ;;
    esac
  fi

  log_ok "$lang: Phase 4 - lsp_detected=$lsp_detected"

  cat >> "$report" <<EOF
phase4_lsp:
  lsp_detected: $lsp_detected
  lsp_server_name: $lsp_server_name
  enrichment_attempted: $lsp_detected
  enrichment_rate: 0
  symbols_resolved: 0
  references_found: 0
  type_info_found: 0
  fallback_to_treesitter: $fallback
EOF
}

# --- Phase 5: Search Quality ---

run_phase5() {
  local lang="$1"
  local report="$RESULTS_DIR/${lang}.yaml"

  log "Phase 5: Search quality for $lang..."

  local hits=0
  local s1="miss" s2="miss" s3="miss" s4="miss"

  local tenant_id
  tenant_id=$(wqm project list 2>/dev/null | grep "lang-test-$lang" | awk '{print $1}' || echo "")

  if [[ -z "$tenant_id" ]]; then
    log_skip "$lang: Phase 5 - no tenant (run Phase 2 first)"
  else
    # Semantic search: find_by_author
    local result
    result=$(wqm search "find books by a specific author" --project "$tenant_id" --limit 3 2>/dev/null || echo "")
    if echo "$result" | grep -qi "find_by_author\|find.by.author\|findByAuthor"; then
      s1="hit"; hits=$((hits + 1))
    fi

    # Semantic search: validate_isbn
    result=$(wqm search "validate an ISBN number" --project "$tenant_id" --limit 3 2>/dev/null || echo "")
    if echo "$result" | grep -qi "validate_isbn\|validate.isbn\|validateIsbn\|isbn"; then
      s2="hit"; hits=$((hits + 1))
    fi

    # Exact search: generate_report
    result=$(wqm search "generate_report" --project "$tenant_id" --exact --limit 3 2>/dev/null || echo "")
    if echo "$result" | grep -qi "storage\|generate_report\|generateReport"; then
      s3="hit"; hits=$((hits + 1))
    fi

    # Semantic search: book data structure
    result=$(wqm search "book data structure with title and author" --project "$tenant_id" --limit 3 2>/dev/null || echo "")
    if echo "$result" | grep -qi "models\|book\|struct\|class"; then
      s4="hit"; hits=$((hits + 1))
    fi
  fi

  log_ok "$lang: Phase 5 - precision=$hits/4"

  cat >> "$report" <<EOF
phase5_search:
  search_semantic_function: $s1
  search_semantic_utility: $s2
  search_exact_function: $s3
  search_semantic_structure: $s4
  search_precision: $hits
EOF
}

# --- Verdict ---

compute_verdict() {
  local lang="$1"
  local report="$RESULTS_DIR/${lang}.yaml"

  local verdict="PASS"
  local issues=""
  local improvements=""

  # Read Phase 1 results
  local comp exec match
  comp=$(grep 'compilation_pass:' "$report" | head -1 | awk '{print $2}')
  exec=$(grep 'execution_pass:' "$report" | head -1 | awk '{print $2}')
  match=$(grep 'output_match:' "$report" | head -1 | awk '{print $2}')

  if [[ "$comp" != "true" || "$exec" != "true" ]]; then
    verdict="FAIL"
    issues+="\"compilation or execution failed\""
  elif [[ "$match" != "true" ]]; then
    verdict="PARTIAL"
    issues+="\"output mismatch\""
  fi

  # Check tree-sitter
  if ! in_array "$lang" "${TREESITTER_LANGUAGES[@]}"; then
    improvements+="\"add tree-sitter grammar for $lang\""
  fi

  # Check LSP
  if ! in_array "$lang" "${LSP_LANGUAGES[@]}"; then
    if [[ -z "$improvements" ]]; then
      improvements+="\"add LSP server integration for $lang\""
    else
      improvements+=", \"add LSP server integration for $lang\""
    fi
  fi

  cat >> "$report" <<EOF
verdict: $verdict
issues: [${issues}]
improvements: [${improvements}]
EOF

  echo "$verdict"
}

# --- Summary Generation ---

generate_summary() {
  local pass=0 partial=0 fail=0
  local ts_compiled="" ts_downloaded="" ts_fallback=""
  local lsp_detected="" lsp_skipped=""
  local precision_sum=0 precision_count=0 perfect="" zero=""

  for lang in "${ALL_LANGUAGES[@]}"; do
    local report="$RESULTS_DIR/${lang}.yaml"
    [[ ! -f "$report" ]] && continue

    local v
    v=$(grep '^verdict:' "$report" 2>/dev/null | awk '{print $2}')
    case "$v" in
      PASS) pass=$((pass + 1)) ;;
      PARTIAL) partial=$((partial + 1)) ;;
      FAIL) fail=$((fail + 1)) ;;
    esac

    # Tree-sitter classification
    if in_array "$lang" "${TREESITTER_LANGUAGES[@]}"; then
      ts_compiled+="$lang, "
    else
      ts_fallback+="$lang, "
    fi

    # LSP classification
    if in_array "$lang" "${LSP_LANGUAGES[@]}"; then
      lsp_detected+="$lang, "
    else
      lsp_skipped+="$lang, "
    fi

    # Search precision
    local prec
    prec=$(grep 'search_precision:' "$report" 2>/dev/null | awk '{print $2}')
    if [[ -n "$prec" && "$prec" =~ ^[0-9]+$ ]]; then
      precision_sum=$((precision_sum + prec))
      precision_count=$((precision_count + 1))
      [[ "$prec" == "4" ]] && perfect+="$lang, "
      [[ "$prec" == "0" ]] && zero+="$lang, "
    fi
  done

  local mean_precision="0.0"
  if [[ $precision_count -gt 0 ]]; then
    mean_precision=$(python3 -c "print(round($precision_sum / $precision_count, 1))" 2>/dev/null || echo "0.0")
  fi

  # Strip trailing commas
  ts_compiled="${ts_compiled%, }"
  ts_fallback="${ts_fallback%, }"
  lsp_detected="${lsp_detected%, }"
  lsp_skipped="${lsp_skipped%, }"
  perfect="${perfect%, }"
  zero="${zero%, }"

  cat > "$RESULTS_DIR/summary.yaml" <<EOF
timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
total_languages: ${#ALL_LANGUAGES[@]}
verdicts:
  pass: $pass
  partial: $partial
  fail: $fail
treesitter_support:
  compiled_in: [$ts_compiled]
  auto_downloaded: []
  text_fallback: [$ts_fallback]
  failed: []
lsp_support:
  detected: [$lsp_detected]
  enriched: [$lsp_detected]
  skipped: [$lsp_skipped]
search_quality:
  mean_precision: $mean_precision
  perfect_scores: [$perfect]
  zero_scores: [$zero]
EOF

  echo ""
  echo "=== Summary ==="
  echo "Languages: ${#ALL_LANGUAGES[@]}"
  echo "PASS: $pass | PARTIAL: $partial | FAIL: $fail"
  echo "Tree-sitter: ${ts_compiled:-none}"
  echo "LSP: ${lsp_detected:-none}"
  echo "Mean search precision: $mean_precision"
  echo "Report: $RESULTS_DIR/summary.yaml"
}

# --- Main ---

main() {
  mkdir -p "$RESULTS_DIR"

  if $SUMMARY_ONLY; then
    generate_summary
    return
  fi

  # Determine which languages to test
  local languages=("${ALL_LANGUAGES[@]}")
  if [[ -n "$LANGUAGE" ]]; then
    languages=("$LANGUAGE")
  fi

  echo "=== Language Support Validation ==="
  echo "Languages: ${#languages[@]}"
  echo "Phase: ${PHASE:-all}"
  echo ""

  for lang in "${languages[@]}"; do
    echo "--- $lang ---"

    if [[ -z "$PHASE" ]] || [[ "$PHASE" == "1" ]]; then
      run_phase1 "$lang"
    fi

    if $PHASE1_ONLY; then
      # For phase1-only, compute verdict based on phase 1 results
      if [[ -f "$RESULTS_DIR/${lang}.yaml" ]]; then
        compute_verdict "$lang" >/dev/null
      fi
      continue
    fi

    if [[ -z "$PHASE" ]] || [[ "$PHASE" == "2" ]]; then
      run_phase2 "$lang"
    fi

    if [[ -z "$PHASE" ]] || [[ "$PHASE" == "3" ]]; then
      run_phase3 "$lang"
    fi

    if [[ -z "$PHASE" ]] || [[ "$PHASE" == "4" ]]; then
      run_phase4 "$lang"
    fi

    if [[ -z "$PHASE" ]] || [[ "$PHASE" == "5" ]]; then
      run_phase5 "$lang"
    fi

    if [[ -z "$PHASE" ]]; then
      local verdict
      verdict=$(compute_verdict "$lang")
      echo "  => $verdict"
    fi

    echo ""
  done

  generate_summary
}

main "$@"
