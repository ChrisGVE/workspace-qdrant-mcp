"""Web UI command for workspace-qdrant-mcp.

This module implements the `wqm web` command to serve the integrated
qdrant-web-ui with workspace-specific features and safety modifications.

Usage:
    wqm web start [--port=8080] [--host=localhost]    # Start web server
    wqm web build                                     # Build production assets
    wqm web dev                                       # Start development server
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ...observability import get_logger

console = Console()
logger = get_logger(__name__)

# Create the web app
web_app = typer.Typer(
    name="web",
    help="Web UI server for workspace-qdrant-mcp",
    add_completion=False,
    rich_markup_mode=None,
)


def get_web_ui_path() -> Path:
    """Get the path to the web UI directory."""
    # Get the project root (where this package is installed)
    current_dir = Path(__file__).parent.parent.parent.parent.parent
    web_ui_path = current_dir / "web-ui"

    if not web_ui_path.exists():
        console.print(f"[red]Error: Web UI not found at {web_ui_path}[/red]")
        console.print("Please ensure the web-ui submodule is properly initialized:")
        console.print("  git submodule update --init --recursive")
        raise typer.Exit(1)

    return web_ui_path


def ensure_dependencies(web_ui_path: Path) -> None:
    """Ensure Node.js dependencies are installed."""
    package_json = web_ui_path / "package.json"
    node_modules = web_ui_path / "node_modules"

    if not package_json.exists():
        console.print(f"[red]Error: package.json not found at {package_json}[/red]")
        raise typer.Exit(1)

    if not node_modules.exists():
        console.print("[yellow]Installing Node.js dependencies...[/yellow]")
        try:
            subprocess.run(
                ["npm", "install"], cwd=web_ui_path, check=True, capture_output=False
            )
            console.print("[green]Dependencies installed successfully![/green]")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to install dependencies: {e}[/red]")
            raise typer.Exit(1)
        except FileNotFoundError:
            console.print(
                "[red]Error: npm not found. Please install Node.js and npm.[/red]"
            )
            console.print("Visit: https://nodejs.org/")
            raise typer.Exit(1)


@web_app.command()
def start(
    port: int = typer.Option(8080, "--port", "-p", help="Port to serve on"),
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to"),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open browser automatically"
    ),
) -> None:
    """Start the workspace web UI server."""
    web_ui_path = get_web_ui_path()
    ensure_dependencies(web_ui_path)

    console.print(f"[green]Starting web UI server on {host}:{port}[/green]")
    console.print(f"Web UI path: {web_ui_path}")

    try:
        # Build the project first
        console.print("[yellow]Building web UI...[/yellow]")
        subprocess.run(
            ["npm", "run", "build"], cwd=web_ui_path, check=True, capture_output=False
        )

        # Start the preview server
        env = os.environ.copy()
        env["PORT"] = str(port)
        env["HOST"] = host

        process = subprocess.run(
            ["npm", "run", "serve", "--", "--port", str(port), "--host", host],
            cwd=web_ui_path,
            env=env,
            capture_output=False,
        )

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to start web server: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Web server stopped.[/yellow]")
        raise typer.Exit(0)


@web_app.command()
def dev(
    port: int = typer.Option(3000, "--port", "-p", help="Development server port"),
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to"),
) -> None:
    """Start the development server with hot reloading."""
    web_ui_path = get_web_ui_path()
    ensure_dependencies(web_ui_path)

    console.print(f"[green]Starting development server on {host}:{port}[/green]")
    console.print("[cyan]Hot reloading enabled - changes will auto-refresh[/cyan]")

    try:
        env = os.environ.copy()
        env["PORT"] = str(port)
        env["HOST"] = host

        # Start Vite dev server
        subprocess.run(["npm", "start"], cwd=web_ui_path, env=env, capture_output=False)

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to start development server: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Development server stopped.[/yellow]")
        raise typer.Exit(0)


@web_app.command()
def build(
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Custom output directory"
    ),
) -> None:
    """Build the web UI for production."""
    web_ui_path = get_web_ui_path()
    ensure_dependencies(web_ui_path)

    console.print("[yellow]Building web UI for production...[/yellow]")

    try:
        env = os.environ.copy()
        if output_dir:
            # Note: Vite uses outDir in vite.config.js, may need custom config for this
            env["BUILD_PATH"] = output_dir

        subprocess.run(
            ["npm", "run", "build"],
            cwd=web_ui_path,
            env=env,
            check=True,
            capture_output=False,
        )

        dist_path = web_ui_path / "dist"
        console.print(f"[green]Build completed successfully![/green]")
        console.print(f"Output directory: {output_dir or dist_path}")

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        raise typer.Exit(1)


@web_app.command()
def install():
    """Install or update Node.js dependencies."""
    web_ui_path = get_web_ui_path()

    console.print("[yellow]Installing Node.js dependencies...[/yellow]")
    try:
        subprocess.run(
            ["npm", "install"], cwd=web_ui_path, check=True, capture_output=False
        )
        console.print("[green]Dependencies installed successfully![/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to install dependencies: {e}[/red]")
        raise typer.Exit(1)
    except FileNotFoundError:
        console.print(
            "[red]Error: npm not found. Please install Node.js and npm.[/red]"
        )
        console.print("Visit: https://nodejs.org/")
        raise typer.Exit(1)


@web_app.command()
def status():
    """Show web UI status and information."""
    web_ui_path = get_web_ui_path()
    package_json = web_ui_path / "package.json"
    node_modules = web_ui_path / "node_modules"
    dist_path = web_ui_path / "dist"

    console.print("[bold]Web UI Status:[/bold]")
    console.print(f"Path: {web_ui_path}")
    console.print(f"Package.json: {'✓' if package_json.exists() else '✗'}")
    console.print(
        f"Dependencies: {'✓' if node_modules.exists() else '✗ (run: wqm web install)'}"
    )
    console.print(
        f"Built assets: {'✓' if dist_path.exists() else '✗ (run: wqm web build)'}"
    )

    if package_json.exists():
        try:
            import json

            with open(package_json) as f:
                pkg = json.load(f)
            console.print(f"Version: {pkg.get('version', 'unknown')}")
            console.print(f"License: {pkg.get('license', 'unknown')}")
        except Exception as e:
            console.print(f"Could not read package.json: {e}")


if __name__ == "__main__":
    web_app()
