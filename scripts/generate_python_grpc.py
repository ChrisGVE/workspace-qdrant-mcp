#!/usr/bin/env python3
"""
Generate Python gRPC stubs from Protocol Buffer definitions.

This script uses grpcio-tools to generate Python code from the workspace_daemon.proto file.
The generated files are placed in src/python/common/grpc/generated/ directory.

Usage:
    python scripts/generate_python_grpc.py

Requirements:
    - grpcio-tools (install via: pip install grpcio-tools)
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Generate Python gRPC stubs from proto files."""
    # Project root directory
    project_root = Path(__file__).parent.parent.absolute()

    # Proto file location
    proto_dir = project_root / "rust-engine-legacy" / "proto"
    proto_file = proto_dir / "workspace_daemon.proto"

    # Output directory for generated Python code
    output_dir = project_root / "src" / "python" / "common" / "grpc" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py if it doesn't exist
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Generated gRPC stubs for workspace-qdrant-mcp daemon."""\n')

    print(f"Project root: {project_root}")
    print(f"Proto file: {proto_file}")
    print(f"Output directory: {output_dir}")

    if not proto_file.exists():
        print(f"Error: Proto file not found at {proto_file}", file=sys.stderr)
        sys.exit(1)

    # Generate Python code using grpc_tools.protoc
    print("\nGenerating Python gRPC stubs...")

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        f"--pyi_out={output_dir}",  # Generate type stubs
        str(proto_file),
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        print("\n✅ Successfully generated Python gRPC stubs!")
        print(f"\nGenerated files in {output_dir}:")
        for file in sorted(output_dir.glob("*.py")):
            print(f"  - {file.name}")

        # Fix imports in generated files to use relative imports
        print("\nFixing imports in generated files...")
        grpc_file = output_dir / "workspace_daemon_pb2_grpc.py"
        if grpc_file.exists():
            content = grpc_file.read_text()
            # Replace absolute import with relative import
            content = content.replace(
                "import workspace_daemon_pb2 as workspace__daemon__pb2",
                "from . import workspace_daemon_pb2 as workspace__daemon__pb2"
            )
            grpc_file.write_text(content)
            print("  ✅ Fixed imports in workspace_daemon_pb2_grpc.py")

        print("\nYou can now import the generated stubs in Python:")
        print("  from src.python.common.grpc.generated import workspace_daemon_pb2")
        print("  from src.python.common.grpc.generated import workspace_daemon_pb2_grpc")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error generating Python gRPC stubs: {e}", file=sys.stderr)
        if e.stdout:
            print(f"stdout: {e.stdout}", file=sys.stderr)
        if e.stderr:
            print(f"stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(
            "\n❌ Error: grpcio-tools not found. Install it with:",
            file=sys.stderr
        )
        print("  pip install grpcio-tools", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
