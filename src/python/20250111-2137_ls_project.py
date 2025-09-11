#!/usr/bin/env python3
import os

def list_directory(path, max_depth=3, current_depth=0):
    """Recursively list directories up to max_depth"""
    if current_depth > max_depth:
        return
        
    try:
        items = sorted(os.listdir(path))
        
        for item in items:
            item_path = os.path.join(path, item)
            indent = "  " * current_depth
            
            if os.path.isdir(item_path):
                print(f"{indent}{item}/")
                if current_depth < max_depth:
                    list_directory(item_path, max_depth, current_depth + 1)
            else:
                # Highlight Python files and CLI-related files
                if item.endswith('.py'):
                    if 'cli' in item.lower() or 'service' in item.lower():
                        print(f"{indent}{item} ⭐")
                    else:
                        print(f"{indent}{item}")
                elif item in ['pyproject.toml', 'setup.py', 'uv.lock']:
                    print(f"{indent}{item} 📦")
                else:
                    print(f"{indent}{item}")
                    
    except PermissionError:
        print(f"{indent}[Permission Denied]")
    except Exception as e:
        print(f"{indent}[Error: {e}]")

if __name__ == "__main__":
    project_root = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"
    print(f"Project structure for: {project_root}")
    print("="*60)
    list_directory(project_root)