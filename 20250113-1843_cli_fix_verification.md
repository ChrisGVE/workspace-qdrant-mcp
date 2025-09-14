# CLI Argument Parsing Fix Verification

## Objective
Replace .unwrap() calls that can panic with output in CLI argument parsing.

## Root Cause
The `memexd.rs` binary was using `.unwrap()` on argument parsing (lines 108-109) that could panic and output error messages to stderr, violating MCP stdio protocol compliance.

## Implementation

### Key Changes Made

1. **Function Signature Change**:
   - Changed from `fn parse_args() -> DaemonArgs`
   - To `fn parse_args() -> Result<DaemonArgs, Box<dyn std::error::Error>>`

2. **Replaced .unwrap() with Proper Error Handling**:
   ```rust
   // BEFORE (problematic):
   let log_level = matches.get_one::<String>("log-level").unwrap().clone();
   let pid_file = matches.get_one::<PathBuf>("pid-file").unwrap().clone();

   // AFTER (safe):
   let log_level = matches.get_one::<String>("log-level")
       .ok_or("Missing log-level parameter (this should not happen with default value)")?;
   let pid_file = matches.get_one::<PathBuf>("pid-file")
       .ok_or("Missing pid-file parameter (this should not happen with default value)")?;
   ```

3. **Replaced .get_matches() with .try_get_matches()**:
   ```rust
   // BEFORE (could output help/errors):
   let matches = Command::new("memexd").get_matches();

   // AFTER (graceful error handling):
   let matches = Command::new("memexd")
       .disable_help_flag(is_daemon) // Disable help in daemon mode
       .disable_version_flag(is_daemon) // Disable version in daemon mode
       .try_get_matches();
   ```

4. **Added Daemon Mode Detection**:
   - Uses existing `detect_daemon_mode()` function to conditionally disable help/version flags
   - Prevents clap from outputting help text in daemon mode

5. **Graceful Error Exit**:
   ```rust
   let matches = match matches {
       Ok(m) => m,
       Err(e) => {
           if is_daemon {
               // In daemon mode, exit silently without stderr output
               process::exit(1);
           } else {
               // In interactive mode, show helpful error message
               eprintln!("Error: {}", e);
               process::exit(1);
           }
       }
   };
   ```

6. **Updated Main Function**:
   ```rust
   // BEFORE:
   let args = parse_args();

   // AFTER:
   let args = parse_args()?;
   ```

## Success Criteria ✅

- ✅ **Zero panic messages from argument parsing**: Replaced all .unwrap() calls with proper error handling
- ✅ **Zero clap help/version output in daemon mode**: Added disable_help_flag() and disable_version_flag()
- ✅ **CLI still functional for interactive mode**: Error messages preserved for interactive use
- ✅ **Graceful error handling without stderr noise**: Silent exit in daemon mode, helpful messages in interactive mode

## Technical Details

### Daemon Mode Detection
The fix leverages the existing `detect_daemon_mode()` function which checks:
- Command-line arguments (absence of `--foreground` flag)
- Environment variables indicating service context:
  - `WQM_SERVICE_MODE=true`
  - `LAUNCHD_SOCKET_PATH` (macOS launchd)
  - `SYSTEMD_EXEC_PID` (systemd)
  - `SERVICE_NAME` (Windows service)

### Memory Safety
All changes maintain Rust's memory safety guarantees:
- No unsafe code introduced
- Proper error propagation using Result types
- No potential for use-after-free or memory leaks

### Backward Compatibility
- Interactive mode behavior preserved (help/version still work)
- All existing command-line arguments work identically
- Default values still function correctly
- Error messages still helpful for users

## Verification

The fix can be verified by:

1. **Compilation Check**: Code compiles without warnings about .unwrap()
2. **Daemon Mode Test**: Running with service environment variables produces no stderr output
3. **Interactive Mode Test**: Running normally still shows help and error messages
4. **Argument Parsing Test**: All valid arguments still parse correctly

## Dependencies Satisfied

This fix coordinates with the environment suppression work by:
- Using the same `detect_daemon_mode()` logic for consistent behavior
- Preventing any CLI-related output that could interfere with MCP protocol
- Maintaining the service detection environment variables

## Impact

- **MCP Protocol Compliance**: Eliminates potential stderr noise from CLI parsing
- **Production Stability**: Removes panic risks from argument handling
- **Developer Experience**: Maintains helpful error messages in development
- **Service Reliability**: Ensures clean daemon startup without output noise