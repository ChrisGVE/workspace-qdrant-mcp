#!/usr/bin/env python3
"""
Rust binary installer for workspace-qdrant-mcp.

This module compiles and installs the wqm CLI and memexd daemon binaries.
It handles version checking to avoid unnecessary rebuilds.

Usage:
    wqm-install          # Install/update both binaries
    wqm-install --check  # Check if binaries need updating
    wqm-install --force  # Force rebuild even if up-to-date
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


# Package version - should match pyproject.toml
PACKAGE_VERSION = "0.3.0"

# Binary configurations
BINARIES = {
    "wqm": {
        "source_dir": "src/rust/cli",
        "cargo_target": None,  # Uses default target dir
        "features": ["phase3"],  # Enable all phases
        "description": "CLI for workspace-qdrant-mcp",
    },
    "memexd": {
        "source_dir": "src/rust/daemon",
        "cargo_target": "core",  # Workspace member that builds memexd
        "features": [],
        "description": "Daemon for file watching and processing",
    },
}


class BinaryStatus(NamedTuple):
    """Status of a binary installation."""
    name: str
    installed: bool
    installed_version: str | None
    needs_update: bool
    install_path: Path | None


def get_install_dir() -> Path:
    """Get the binary installation directory."""
    if platform.system() == "Windows":
        # Windows: use %LOCALAPPDATA%\Programs\wqm
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "Programs" / "wqm"
        return Path.home() / "AppData" / "Local" / "Programs" / "wqm"
    else:
        # Unix-like: use ~/.local/bin
        return Path.home() / ".local" / "bin"


def get_package_root() -> Path | None:
    """Find the package root directory containing Rust source."""
    # First, try to find it relative to this file (development mode)
    this_file = Path(__file__).resolve()

    # Walk up looking for src/rust directory
    for parent in [this_file.parent] + list(this_file.parents):
        rust_dir = parent / "src" / "rust"
        if rust_dir.exists():
            return parent

    # Try finding via package resources
    try:
        import importlib.resources as pkg_resources
        # In installed mode, we need the source to be included
        # This typically won't work for source builds, but try anyway
        pass
    except Exception:
        pass

    return None


def check_cargo() -> bool:
    """Check if Cargo (Rust toolchain) is available."""
    try:
        result = subprocess.run(
            ["cargo", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def is_python_shim(binary_path: Path) -> bool:
    """Check if a file is a Python script shim (not a real binary)."""
    if not binary_path.exists():
        return False

    try:
        with open(binary_path, "rb") as f:
            header = f.read(100)
            # Python shims start with #! and contain "python"
            if header.startswith(b"#!") and b"python" in header.lower():
                return True
    except Exception:
        pass

    return False


def get_binary_version(binary_path: Path) -> str | None:
    """Get the version of an installed binary."""
    if not binary_path.exists():
        return None

    # Skip Python shims - they can't provide version
    if is_python_shim(binary_path):
        return "python-shim"

    try:
        result = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Parse version from output like "wqm 0.3.0" or "memexd 0.3.0"
            output = result.stdout.strip()
            parts = output.split()
            if len(parts) >= 2:
                return parts[1]
            return output
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        pass

    return None


def check_binary_status(name: str, install_dir: Path) -> BinaryStatus:
    """Check the installation status of a binary."""
    binary_name = f"{name}.exe" if platform.system() == "Windows" else name
    install_path = install_dir / binary_name

    installed = install_path.exists()
    installed_version = get_binary_version(install_path) if installed else None

    # Need update if not installed or version mismatch
    needs_update = not installed or installed_version != PACKAGE_VERSION

    return BinaryStatus(
        name=name,
        installed=installed,
        installed_version=installed_version,
        needs_update=needs_update,
        install_path=install_path if installed else None
    )


def build_binary(name: str, config: dict, package_root: Path, verbose: bool = False) -> Path | None:
    """Build a Rust binary and return the path to the built executable."""
    source_dir = package_root / config["source_dir"]

    if not source_dir.exists():
        print(f"  ✗ Source directory not found: {source_dir}", file=sys.stderr)
        return None

    # Build command
    cmd = ["cargo", "build", "--release"]

    # Add features if specified
    if config["features"]:
        cmd.extend(["--features", ",".join(config["features"])])

    # For workspace builds, specify the package
    if config["cargo_target"]:
        cmd.extend(["-p", f"workspace-qdrant-{config['cargo_target']}"])

    print(f"  Building {name}...")
    if verbose:
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Directory: {source_dir}")

    try:
        result = subprocess.run(
            cmd,
            cwd=source_dir,
            capture_output=not verbose,
            text=True,
            timeout=600  # 10 minute timeout for compilation
        )

        if result.returncode != 0:
            print(f"  ✗ Build failed for {name}", file=sys.stderr)
            if not verbose and result.stderr:
                print(result.stderr, file=sys.stderr)
            return None

    except subprocess.TimeoutExpired:
        print(f"  ✗ Build timed out for {name}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ✗ Build error for {name}: {e}", file=sys.stderr)
        return None

    # Find the built binary
    binary_name = f"{name}.exe" if platform.system() == "Windows" else name

    # Determine target directory
    if config["cargo_target"]:
        # Workspace: target is at workspace root
        target_dir = source_dir / "target" / "release"
    else:
        target_dir = source_dir / "target" / "release"

    built_binary = target_dir / binary_name

    if not built_binary.exists():
        print(f"  ✗ Built binary not found at: {built_binary}", file=sys.stderr)
        return None

    print(f"  ✓ Built {name}")
    return built_binary


def install_binary(built_path: Path, install_dir: Path, name: str) -> bool:
    """Install a built binary to the installation directory."""
    binary_name = f"{name}.exe" if platform.system() == "Windows" else name
    install_path = install_dir / binary_name

    # Ensure install directory exists
    install_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy the binary
        shutil.copy2(built_path, install_path)

        # Make executable on Unix
        if platform.system() != "Windows":
            install_path.chmod(0o755)

        print(f"  ✓ Installed {name} to {install_path}")
        return True

    except Exception as e:
        print(f"  ✗ Failed to install {name}: {e}", file=sys.stderr)
        return False


def check_path_configured(install_dir: Path) -> bool:
    """Check if the install directory is in PATH."""
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    return str(install_dir) in path_dirs


def print_path_instructions(install_dir: Path):
    """Print instructions for adding install dir to PATH."""
    shell = os.environ.get("SHELL", "")

    print(f"\n⚠️  {install_dir} is not in your PATH")
    print("\nAdd it to your shell configuration:")

    if "zsh" in shell:
        print(f'  echo \'export PATH="{install_dir}:$PATH"\' >> ~/.zshrc')
        print("  source ~/.zshrc")
    elif "bash" in shell:
        print(f'  echo \'export PATH="{install_dir}:$PATH"\' >> ~/.bashrc')
        print("  source ~/.bashrc")
    elif "fish" in shell:
        print(f'  fish_add_path {install_dir}')
    elif platform.system() == "Windows":
        print(f"  Add {install_dir} to your PATH environment variable")
        print("  Settings → System → About → Advanced system settings → Environment Variables")
    else:
        print(f'  export PATH="{install_dir}:$PATH"')


def main():
    """Main entry point for the installer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Install workspace-qdrant-mcp Rust binaries (wqm CLI and memexd daemon)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if binaries need updating without installing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if binaries are up-to-date"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed build output"
    )
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=None,
        help=f"Installation directory (default: {get_install_dir()})"
    )

    args = parser.parse_args()

    install_dir = args.install_dir or get_install_dir()

    print(f"workspace-qdrant-mcp binary installer v{PACKAGE_VERSION}")
    print(f"Install directory: {install_dir}")
    print()

    # Check for Cargo
    if not check_cargo():
        print("✗ Rust toolchain not found", file=sys.stderr)
        print("\nInstall Rust from: https://rustup.rs/", file=sys.stderr)
        sys.exit(1)

    # Find package root
    package_root = get_package_root()
    if not package_root:
        print("✗ Could not find package source directory", file=sys.stderr)
        print("\nThis installer requires the source code to be available.", file=sys.stderr)
        print("If installed via pip, ensure you used editable mode: pip install -e .", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Package root: {package_root}")

    # Check status of all binaries
    print("Checking binary status...")
    statuses = {}
    for name in BINARIES:
        status = check_binary_status(name, install_dir)
        statuses[name] = status

        if status.installed:
            version_str = status.installed_version or "unknown"
            if version_str == "python-shim":
                print(f"  {name}: Python shim found (will replace with Rust binary)")
            elif status.needs_update:
                print(f"  {name}: v{version_str} → v{PACKAGE_VERSION} (needs update)")
            else:
                print(f"  {name}: v{version_str} ✓")
        else:
            print(f"  {name}: not installed")

    # If just checking, exit here
    if args.check:
        needs_update = any(s.needs_update for s in statuses.values())
        sys.exit(1 if needs_update else 0)

    # Determine what needs to be built
    to_build = []
    for name, status in statuses.items():
        if args.force or status.needs_update:
            to_build.append(name)

    if not to_build:
        print("\n✓ All binaries are up-to-date")

        if not check_path_configured(install_dir):
            print_path_instructions(install_dir)

        sys.exit(0)

    # Build and install
    print(f"\nBuilding {len(to_build)} binary(ies)...")

    success_count = 0
    for name in to_build:
        config = BINARIES[name]
        print(f"\n[{name}] {config['description']}")

        # Build
        built_path = build_binary(name, config, package_root, args.verbose)
        if not built_path:
            continue

        # Install
        if install_binary(built_path, install_dir, name):
            success_count += 1

    print()

    if success_count == len(to_build):
        print(f"✓ Successfully installed {success_count} binary(ies)")
    elif success_count > 0:
        print(f"⚠ Installed {success_count}/{len(to_build)} binaries")
    else:
        print("✗ No binaries were installed", file=sys.stderr)
        sys.exit(1)

    # Check PATH
    if not check_path_configured(install_dir):
        print_path_instructions(install_dir)
    else:
        print(f"\nBinaries are ready to use:")
        print(f"  wqm --help     # CLI commands")
        print(f"  memexd --help  # Daemon options")


if __name__ == "__main__":
    main()
