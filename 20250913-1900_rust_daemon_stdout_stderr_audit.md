# Rust Daemon Direct Output Audit Report

**Date:** 2025-09-13 19:00
**Scope:** src/rust/daemon/ (all Rust source files, excluding build artifacts)
**Objective:** Identify all direct stdout/stderr output bypassing the logging system

## Executive Summary

**CRITICAL FINDINGS:**
- **67 direct println!/eprintln! calls** found across source files
- **29 panic! macros** that output before program abort
- **242 unwrap() calls** that may panic with output
- **0 dbg! macros** (good - no debug output)

**SEVERITY BREAKDOWN:**
- **Critical:** 4 instances (always outputs in production daemon)
- **Conditional:** 63 instances (outputs in specific conditions)
- **Debug-only:** 0 instances (outputs only in debug builds)

---

## CRITICAL ISSUES - IMMEDIATE ACTION REQUIRED

These instances ALWAYS output to stdout/stderr in production daemon mode:

### FILE: core/src/embedding.rs:283
**PATTERN:** `println!`
**SEVERITY:** Critical
**CONTENT:** `"Downloading model: {} from {}", model_name, model_info.url`
**RECOMMENDATION:** Replace with `info!("Downloading model: {} from {}", model_name, model_info.url);`

### FILE: core/src/embedding.rs:291
**PATTERN:** `println!`
**SEVERITY:** Critical
**CONTENT:** `"Downloading tokenizer for: {}", model_name`
**RECOMMENDATION:** Replace with `info!("Downloading tokenizer for: {}", model_name);`

### FILE: core/src/bin/memexd.rs:108
**PATTERN:** `.unwrap()`
**SEVERITY:** Critical
**CONTENT:** Command-line argument parsing (will panic with message on invalid args)
**RECOMMENDATION:** Replace with proper error handling using `?` operator and return Result

### FILE: core/src/bin/memexd.rs:109
**PATTERN:** `.unwrap()`
**SEVERITY:** Critical
**CONTENT:** Command-line argument parsing (will panic with message on invalid args)
**RECOMMENDATION:** Replace with proper error handling using `?` operator and return Result

---

## CONDITIONAL OUTPUT ISSUES

### Service Demo Binary (memexd_service_demo.rs)
**PATTERN:** Multiple `println!` and `eprintln!` calls
**SEVERITY:** Conditional (demo/test binary)
**COUNT:** 31 instances
**CONTENT:** Status messages, startup info, shutdown messages
**RECOMMENDATION:**
- If this is a user-facing demo: Keep output but add `#[cfg(not(feature = "daemon-mode"))]` guards
- If used in production: Replace all with appropriate log levels (info!/warn!/error!)

### Test Files
**PATTERN:** `println!` calls
**SEVERITY:** Conditional (test-only)
**COUNT:** 28 instances across test files
**CONTENT:** Test output, performance metrics, debug information
**RECOMMENDATION:**
- Keep for test files but add `#[cfg(test)]` guards to ensure they only compile in test builds
- Consider using `eprintln!` for test output to avoid mixing with captured stdout

### Build Scripts
**PATTERN:** `println!` calls
**SEVERITY:** Conditional (build-time only)
**COUNT:** 3 instances in build.rs files
**CONTENT:** Cargo directives (`cargo:rerun-if-changed`, `cargo:rustc-env`)
**RECOMMENDATION:** Keep as-is (these are cargo build directives, not runtime output)

---

## PANIC/UNWRAP ANALYSIS

### High-Risk Panic Calls
**COUNT:** 29 panic! macros found
**LOCATIONS:** Primarily in test code and IPC response handling
**SEVERITY:** Conditional (most are in test code)
**RECOMMENDATION:**
- Test panics: Keep but ensure they're under `#[cfg(test)]`
- IPC panics: Replace with proper error handling and logging

### Unwrap Usage
**COUNT:** 242 unwrap() calls
**SEVERITY:** Conditional (depends on context)
**RECOMMENDATION:** Audit each unwrap() call:
- Convert to proper error handling where possible
- Use `.expect("detailed error message")` for better panic messages
- Add logging before potential panics

---

## BUILD SCRIPT OUTPUT (SAFE)

### FILE: build.rs:15
**PATTERN:** `println!`
**SEVERITY:** Debug-only (build-time)
**CONTENT:** `"cargo:rerun-if-changed=../proto/ingestion.proto"`
**RECOMMENDATION:** Keep (cargo directive)

### FILE: build.rs:18-19
**PATTERN:** `println!`
**SEVERITY:** Debug-only (build-time)
**CONTENT:** Build timestamp and git hash env vars
**RECOMMENDATION:** Keep (cargo directives)

### FILE: grpc/build.rs:11
**PATTERN:** `println!`
**SEVERITY:** Debug-only (build-time)
**CONTENT:** `"cargo:rerun-if-changed=../proto/ingestion.proto"`
**RECOMMENDATION:** Keep (cargo directive)

---

## DETAILED RECOMMENDATIONS

### 1. IMMEDIATE FIXES (Critical Priority)

```rust
// BEFORE: core/src/embedding.rs:283
println!("Downloading model: {} from {}", model_name, model_info.url);

// AFTER:
info!("Downloading model: {} from {}", model_name, model_info.url);
```

```rust
// BEFORE: core/src/embedding.rs:291
println!("Downloading tokenizer for: {}", model_name);

// AFTER:
info!("Downloading tokenizer for: {}", model_name);
```

```rust
// BEFORE: core/src/bin/memexd.rs:108-109
log_level: matches.get_one::<String>("log-level").unwrap().clone(),
pid_file: matches.get_one::<PathBuf>("pid-file").unwrap().clone(),

// AFTER:
log_level: matches.get_one::<String>("log-level")
    .ok_or("Missing log-level argument")?
    .clone(),
pid_file: matches.get_one::<PathBuf>("pid-file")
    .ok_or("Missing pid-file argument")?
    .clone(),
```

### 2. SERVICE DEMO CLEANUP

Add daemon mode detection:
```rust
// Add this helper function
fn should_suppress_console_output() -> bool {
    std::env::var("WQM_SERVICE_MODE").is_ok() ||
    std::env::var("WQM_CLI_MODE").is_err()
}

// Replace println! calls:
if !should_suppress_console_output() {
    println!("Status message");
} else {
    info!("Status message");
}
```

### 3. TEST OUTPUT GUARDS

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn performance_test() {
        // Test output is OK here
        println!("Performance results: ...");
    }
}

// Or for integration tests:
#[cfg(test)]
println!("Integration test output");
```

### 4. ERROR HANDLING IMPROVEMENTS

```rust
// Replace panic! in production code:
// BEFORE:
other => panic!("Expected HealthCheckOk, got: {:?}", other),

// AFTER:
other => {
    error!("IPC protocol error: expected HealthCheckOk, got: {:?}", other);
    return Err(IpcError::ProtocolError(format!("Unexpected response: {:?}", other)));
}
```

---

## COMPLIANCE VERIFICATION

To verify daemon silence compliance:

1. **Run daemon in service mode:**
   ```bash
   RUST_LOG=error WQM_SERVICE_MODE=true cargo run --bin memexd
   ```

2. **Monitor output:**
   ```bash
   # Should produce NO stdout/stderr output after startup
   timeout 10s cargo run --bin memexd 2>&1 | wc -l  # Should be 0
   ```

3. **Test model download scenario:**
   ```bash
   # This currently breaks silence - needs immediate fix
   rm -rf ~/.cache/workspace-qdrant-models/
   WQM_SERVICE_MODE=true cargo run --bin memexd  # Will output download messages
   ```

---

## IMPLEMENTATION PRIORITY

1. **Phase 1 (Critical):** Fix embedding.rs println! calls (breaks daemon silence)
2. **Phase 2 (High):** Fix memexd.rs unwrap() calls (causes panics with output)
3. **Phase 3 (Medium):** Clean up service demo or add guards
4. **Phase 4 (Low):** Test output improvements and panic handling

**SUCCESS CRITERIA:**
- Zero stdout/stderr output in daemon service mode during normal operation
- Model download progress logged via tracing instead of println!
- No panic output during normal error conditions
- Test output properly isolated with guards

**COMPLIANCE TEST:**
```bash
# Should produce ZERO console output
WQM_SERVICE_MODE=true RUST_LOG=off timeout 30s ./memexd --foreground >/dev/null 2>&1
echo $?  # Should be 0 (clean exit) or 124 (timeout - daemon running silently)
```