"""Grammar management CLI commands for tree-sitter.

This module provides commands for installing, compiling, and managing
tree-sitter grammars for enhanced code parsing and analysis.
"""

from pathlib import Path

import typer
from common.core.grammar_compiler import CompilerDetector, GrammarCompiler
from common.core.grammar_config import ConfigManager
from common.core.grammar_discovery import GrammarDiscovery
from common.core.grammar_installer import GrammarInstaller
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..utils import (
    confirm,
    create_command_app,
    error_message,
    force_option,
    json_output_option,
    success_message,
    verbose_option,
    warning_message,
)

console = Console()

# Create the grammar app using shared utilities
grammar_app = create_command_app(
    name="grammar",
    help_text="""Tree-sitter grammar management.

Manage tree-sitter grammars for enhanced code parsing and analysis.

Examples:
    wqm grammar list                                    # Show installed grammars
    wqm grammar install https://github.com/...          # Install from URL
    wqm grammar install tree-sitter-python              # Install by name
    wqm grammar compile python                          # Compile a grammar
    wqm grammar remove python                           # Remove a grammar
    wqm grammar info python                             # Show grammar details
    wqm grammar config                                  # Show configuration
    wqm grammar config --set-c-compiler=clang           # Set compiler""",
    no_args_is_help=True,
)


@grammar_app.command("list")
def list_grammars(
    json_output: bool = json_output_option(),
    verbose: bool = verbose_option(),
):
    """List all installed tree-sitter grammars."""
    try:
        discovery = GrammarDiscovery()
        grammars = discovery.discover_grammars()

        if json_output:
            import json as json_lib
            output = {
                "grammars": {
                    name: info.to_dict()
                    for name, info in grammars.items()
                },
                "total": len(grammars)
            }
            typer.echo(json_lib.dumps(output, indent=2))
            return

        if not grammars:
            warning_message("No grammars installed")
            console.print("\n💡 Install grammars with: wqm grammar install <url>")
            return

        # Create rich table
        table = Table(title=f"Installed Tree-sitter Grammars ({len(grammars)})")
        table.add_column("Language", style="cyan", no_wrap=True)
        table.add_column("Path", style="blue")
        table.add_column("Parser", style="green")
        table.add_column("Scanner", style="yellow")

        for name, info in sorted(grammars.items()):
            parser_status = "✓" if info.parser_path else "✗"
            scanner_status = "✓" if info.has_external_scanner else "−"

            if verbose:
                path_str = str(info.path)
            else:
                path_str = str(info.path).replace(str(Path.home()), "~")

            table.add_row(
                name,
                path_str,
                parser_status,
                scanner_status
            )

        console.print(table)

        # Show summary
        compiled = sum(1 for g in grammars.values() if g.parser_path)
        console.print(f"\n📊 {compiled}/{len(grammars)} grammars compiled")

    except Exception as e:
        error_message(f"Failed to list grammars: {e}")
        raise typer.Exit(1)


@grammar_app.command("install")
def install_grammar(
    url_or_name: str = typer.Argument(..., help="Git URL or grammar name"),
    version: str | None = typer.Option(None, "--version", "-v", help="Version (tag/branch/commit)"),
    name: str | None = typer.Option(None, "--name", "-n", help="Custom grammar name"),
    compile: bool = typer.Option(True, "--compile/--no-compile", help="Compile after installation"),
    force: bool = force_option(),
    verbose: bool = verbose_option(),
):
    """Install a tree-sitter grammar from Git repository.

    Can install from full Git URL or short name (will try common repos).
    """
    try:
        config_manager = ConfigManager()
        config = config_manager.load()

        installer = GrammarInstaller(
            installation_dir=config_manager.get_installation_directory()
        )

        # Convert short names to URLs
        if not url_or_name.startswith(("http://", "https://", "git@")):
            # Assume it's a short name like "python" or "tree-sitter-python"
            grammar_name = url_or_name.replace("tree-sitter-", "")
            url = f"https://github.com/tree-sitter/tree-sitter-{grammar_name}"
            console.print(f"📦 Resolving short name to: {url}")
        else:
            url = url_or_name

        # Install with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Installing grammar...", total=None)

            result = installer.install(
                grammar_url=url,
                grammar_name=name,
                version=version,
                force=force,
                progress=progress,
                progress_task=task
            )

        if not result.success:
            error_message(f"Installation failed: {result.error}")
            raise typer.Exit(1)

        success_message(f"Installed '{result.grammar_name}' to {result.installation_path}")

        if result.version:
            console.print(f"📌 Version: {result.version}")

        # Compile if requested
        if compile and config.auto_compile:
            console.print("\n🔨 Compiling grammar...")
            compile_result = _compile_grammar(result.grammar_name, result.installation_path)

            if compile_result:
                success_message("Compiled successfully")
            else:
                warning_message("Compilation failed (grammar still installed)")

    except Exception as e:
        error_message(f"Failed to install grammar: {e}")
        if verbose:
            logger.exception("Installation error")
        raise typer.Exit(1)


@grammar_app.command("compile")
def compile_grammar(
    grammar_name: str = typer.Argument(..., help="Grammar name to compile"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = verbose_option(),
):
    """Compile a tree-sitter grammar to shared library."""
    try:
        discovery = GrammarDiscovery()
        grammar_info = discovery.get_grammar(grammar_name)

        if not grammar_info:
            error_message(f"Grammar '{grammar_name}' not found")
            console.print("\n💡 List available grammars with: wqm grammar list")
            raise typer.Exit(1)

        success = _compile_grammar(grammar_name, grammar_info.path, output_dir)

        if not success:
            raise typer.Exit(1)

    except Exception as e:
        error_message(f"Failed to compile grammar: {e}")
        if verbose:
            logger.exception("Compilation error")
        raise typer.Exit(1)


def _compile_grammar(grammar_name: str, grammar_path: Path, output_dir: Path | None = None) -> bool:
    """Helper function to compile a grammar."""
    config_manager = ConfigManager()
    config = config_manager.load()

    # Get compilers from config or auto-detect
    compiler_detector = CompilerDetector()
    c_compiler = None
    cpp_compiler = None

    if config.c_compiler:
        # User specified compiler
        import shutil
        c_path = shutil.which(config.c_compiler)
        if c_path:
            from common.core.grammar_compiler import CompilerInfo
            c_compiler = CompilerInfo(
                name=config.c_compiler,
                path=c_path,
                version=compiler_detector._get_compiler_version(config.c_compiler, c_path)
            )

    if config.cpp_compiler:
        import shutil
        cpp_path = shutil.which(config.cpp_compiler)
        if cpp_path:
            from common.core.grammar_compiler import CompilerInfo
            cpp_compiler = CompilerInfo(
                name=config.cpp_compiler,
                path=cpp_path,
                version=compiler_detector._get_compiler_version(config.cpp_compiler, cpp_path),
                is_cpp=True
            )

    compiler = GrammarCompiler(
        c_compiler=c_compiler,
        cpp_compiler=cpp_compiler
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Compiling grammar...", total=None)
        result = compiler.compile(
            grammar_path,
            output_dir=output_dir,
            progress=progress,
            progress_task=task
        )

    if not result.success:
        error_message(f"Compilation failed: {result.error}")
        return False

    success_message(f"Compiled '{grammar_name}' to {result.output_path}")

    if result.warnings:
        warning_message(f"{len(result.warnings)} warnings")
        for warning in result.warnings[:5]:  # Show first 5
            console.print(f"  ⚠️  {warning}")

    return True


@grammar_app.command("remove")
def remove_grammar(
    grammar_name: str = typer.Argument(..., help="Grammar name to remove"),
    force: bool = force_option(),
    verbose: bool = verbose_option(),
):
    """Remove an installed tree-sitter grammar."""
    try:
        config_manager = ConfigManager()
        installer = GrammarInstaller(
            installation_dir=config_manager.get_installation_directory()
        )

        if not installer.is_installed(grammar_name):
            error_message(f"Grammar '{grammar_name}' is not installed")
            raise typer.Exit(1)

        install_path = installer.get_installation_path(grammar_name)

        # Confirm removal
        if not force:
            if not confirm(
                f"Remove grammar '{grammar_name}' from {install_path}?",
                default=False
            ):
                console.print("❌ Cancelled")
                return

        success, message = installer.uninstall(grammar_name)

        if success:
            success_message(message)
        else:
            error_message(message)
            raise typer.Exit(1)

    except Exception as e:
        error_message(f"Failed to remove grammar: {e}")
        if verbose:
            logger.exception("Removal error")
        raise typer.Exit(1)


@grammar_app.command("info")
def grammar_info(
    grammar_name: str = typer.Argument(..., help="Grammar name"),
    json_output: bool = json_output_option(),
    verbose: bool = verbose_option(),
):
    """Show detailed information about a grammar."""
    try:
        discovery = GrammarDiscovery()
        grammar = discovery.get_grammar(grammar_name)

        if not grammar:
            error_message(f"Grammar '{grammar_name}' not found")
            raise typer.Exit(1)

        if json_output:
            import json as json_lib
            typer.echo(json_lib.dumps(grammar.to_dict(), indent=2))
            return

        # Rich display
        console.print(f"\n[bold cyan]Grammar Information: {grammar_name}[/bold cyan]\n")

        table = Table(show_header=False, box=None)
        table.add_column("Property", style="yellow")
        table.add_column("Value", style="white")

        table.add_row("Name", grammar.name)
        table.add_row("Path", str(grammar.path))

        if grammar.parser_path:
            table.add_row("Parser", f"✓ {grammar.parser_path}")
        else:
            table.add_row("Parser", "✗ Not compiled")

        table.add_row("External Scanner", "✓ Yes" if grammar.has_external_scanner else "− No")

        if grammar.version:
            table.add_row("Version", grammar.version)

        console.print(table)

        # Validation
        is_valid, msg = discovery.validate_grammar(grammar_name)
        console.print()
        if is_valid:
            console.print(f"[green]✓ {msg}[/green]")
        else:
            console.print(f"[red]✗ {msg}[/red]")

    except Exception as e:
        error_message(f"Failed to get grammar info: {e}")
        if verbose:
            logger.exception("Info error")
        raise typer.Exit(1)


@grammar_app.command("config")
def show_config(
    set_c_compiler: str | None = typer.Option(None, "--set-c-compiler", help="Set C compiler"),
    set_cpp_compiler: str | None = typer.Option(None, "--set-cpp-compiler", help="Set C++ compiler"),
    set_auto_compile: bool = typer.Option(False, "--auto-compile", help="Enable auto-compile"),
    set_no_auto_compile: bool = typer.Option(False, "--no-auto-compile", help="Disable auto-compile"),
    set_parallel_builds: int | None = typer.Option(None, "--parallel-builds", help="Parallel build jobs"),
    add_directory: str | None = typer.Option(None, "--add-directory", help="Add grammar search directory"),
    remove_directory: str | None = typer.Option(None, "--remove-directory", help="Remove search directory"),
    reset: bool = typer.Option(False, "--reset", help="Reset to defaults"),
    export: Path | None = typer.Option(None, "--export", help="Export config to file"),
    import_config: Path | None = typer.Option(None, "--import", help="Import config from file"),
    json_output: bool = json_output_option(),
    verbose: bool = verbose_option(),
):
    """Show or modify grammar configuration."""
    try:
        config_manager = ConfigManager()

        # Handle modifications
        if reset:
            if confirm("Reset configuration to defaults?", default=False):
                config_manager.reset_to_defaults()
                success_message("Configuration reset to defaults")
            return

        if import_config:
            config_manager.import_config(import_config)
            success_message(f"Imported configuration from {import_config}")
            return

        if export:
            config_manager.export_config(export)
            success_message(f"Exported configuration to {export}")
            return

        # Update fields
        updates = {}
        if set_c_compiler is not None:
            updates['c_compiler'] = set_c_compiler
        if set_cpp_compiler is not None:
            updates['cpp_compiler'] = set_cpp_compiler
        if set_auto_compile:
            updates['auto_compile'] = True
        if set_no_auto_compile:
            updates['auto_compile'] = False
        if set_parallel_builds is not None:
            updates['parallel_builds'] = set_parallel_builds

        if updates:
            config_manager.update(**updates)
            success_message("Configuration updated")

        if add_directory:
            config_manager.add_grammar_directory(add_directory)
            success_message(f"Added directory: {add_directory}")

        if remove_directory:
            config_manager.remove_grammar_directory(remove_directory)
            success_message(f"Removed directory: {remove_directory}")

        # Show configuration
        config = config_manager.load()

        if json_output:
            import json as json_lib
            typer.echo(json_lib.dumps(config.to_dict(), indent=2))
            return

        # Rich display
        console.print("\n[bold cyan]Grammar Configuration[/bold cyan]\n")

        table = Table(show_header=False, box=None)
        table.add_column("Setting", style="yellow")
        table.add_column("Value", style="white")

        table.add_row("C Compiler", config.c_compiler or "Auto-detect")
        table.add_row("C++ Compiler", config.cpp_compiler or "Auto-detect")
        table.add_row("Auto Compile", "✓" if config.auto_compile else "✗")
        table.add_row("Parallel Builds", str(config.parallel_builds))
        table.add_row("Optimization", config.optimization_level)
        table.add_row("Keep Artifacts", "✓" if config.keep_build_artifacts else "✗")

        if config.grammar_directories:
            dirs = "\n".join(config.grammar_directories)
            table.add_row("Grammar Directories", dirs)

        install_dir = config.installation_directory or str(config_manager.get_installation_directory())
        table.add_row("Installation Directory", install_dir)

        console.print(table)
        console.print(f"\n📁 Config file: {config_manager.config_file}")

    except Exception as e:
        error_message(f"Failed to manage configuration: {e}")
        if verbose:
            logger.exception("Config error")
        raise typer.Exit(1)


@grammar_app.command("update")
def update_grammar(
    grammar_name: str = typer.Argument(..., help="Grammar name to update"),
    version: str | None = typer.Option(None, "--version", "-v", help="Version to update to"),
    compile: bool = typer.Option(True, "--compile/--no-compile", help="Compile after update"),
    verbose: bool = verbose_option(),
):
    """Update a grammar to a new version."""
    try:
        # This is essentially a force reinstall
        config_manager = ConfigManager()
        installer = GrammarInstaller(
            installation_dir=config_manager.get_installation_directory()
        )

        if not installer.is_installed(grammar_name):
            error_message(f"Grammar '{grammar_name}' is not installed")
            console.print("\n💡 Install it with: wqm grammar install tree-sitter-{grammar_name}")
            raise typer.Exit(1)

        # Get current installation to derive URL
        installer.get_installation_path(grammar_name)

        console.print(f"🔄 Updating grammar '{grammar_name}'...")

        # Derive URL from common pattern
        url = f"https://github.com/tree-sitter/tree-sitter-{grammar_name}"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Updating grammar...", total=None)

            result = installer.install(
                grammar_url=url,
                grammar_name=grammar_name,
                version=version,
                force=True,  # Force reinstall
                progress=progress,
                progress_task=task
            )

        if not result.success:
            error_message(f"Update failed: {result.error}")
            raise typer.Exit(1)

        success_message(f"Updated '{grammar_name}'")

        if result.version:
            console.print(f"📌 New version: {result.version}")

        # Compile if requested
        if compile:
            console.print("\n🔨 Compiling grammar...")
            _compile_grammar(grammar_name, result.installation_path)

    except Exception as e:
        error_message(f"Failed to update grammar: {e}")
        if verbose:
            logger.exception("Update error")
        raise typer.Exit(1)


# Export the app
__all__ = ["grammar_app"]
