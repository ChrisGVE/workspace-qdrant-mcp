# Handover — 2026-03-12

## Current State

**v0.0.1 released.** All 6 platform builds green. Release published at:
https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/tag/v0.0.1

Branch protection on `main` is active (requires CI + Security Audit status checks).

## Completed This Session

### v0.0.1 Release — Cross-platform build fixes

All CI checks were already green from the previous session. This session fixed the
Release Build workflow which cross-compiles for 6 targets.

**Linux fixes:**
- Added `liblzma.so`, `libz.so`, `libssl.so`, `libcrypto.so` to Linux verify allowlist
  (from libgit2 for compression/HTTPS; standard on all modern Linux)
- Changed `reqwest = "0.11"` to use `rustls-tls` (removes OpenSSL dynamic dep)
- Changed `git2 = "0.20"` to `default-features = false` (disables HTTPS/SSH transport,
  only local git ops needed; removes OpenSSL dynamic dep from macOS and Linux)

**macOS fixes:**
- Disabling git2 HTTPS feature removed the Homebrew OpenSSL dylib dependency
  (binaries are now self-contained — no Homebrew needed)

**Windows fixes:**
- `watching/platform/mod.rs`: Replaced `windows::Win32::Storage::FileSystem::*` constants
  with `winapi::um::winnt` equivalents (removed `windows = 0.52` dep, which had feature
  resolution issues on Windows CI)
- `storage/client.rs` + `memexd/src/main.rs`: Fixed `SetStdHandle` cast from
  `*mut std::ffi::c_void` to `*mut winapi::ctypes::c_void` (type mismatch on MSVC)
- Added missing Windows DLLs to verify allowlist: `bcryptprimitives.dll`, `pdh.dll`,
  `powrprof.dll`, `d3d12.dll`, `directml.dll`, `dxgi.dll`, `setupapi.dll`, `MSVCP140_1.dll`
- Skip smoke test for `aarch64-pc-windows-msvc` (ARM64 binary can't run on x64 runner)

**Release publishing:**
- Retag: removed `-alpha` suffix per user request → released as `v0.0.1`
- `softprops/action-gh-release@v2` had a transient "Not Found" error on one asset update,
  leaving release in Draft state → manually published via `gh release edit --draft=false`

### Branch protection on `main`
- Required status checks: CI (ubuntu-latest), CI (macos-latest), TypeScript MCP server,
  Security Audit
- Force pushes disabled, deletions disabled
- Admins not enforced (can push directly if needed)

## Branch Status

- `main`: all CI green, v0.0.1 released and published

## No Further Work Required

All tasks complete. The project is at v0.0.1 with:
- 39 release assets (binaries, archives, checksums, napi addons)
- 6 platforms: linux-x64, linux-arm64, darwin-arm64, darwin-x64, windows-x64, windows-arm64
- Branch protection on main
