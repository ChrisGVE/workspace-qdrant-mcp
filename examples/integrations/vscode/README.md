# VS Code Integration

Comprehensive VS Code workspace integration for workspace-qdrant-mcp, including settings, extensions, and workflows.

## 🎯 Overview

This integration provides:

- **Workspace Configuration** - Pre-configured VS Code settings for optimal workflow
- **Extension Recommendations** - Essential extensions for enhanced productivity  
- **Code Snippets** - Custom snippets for common workspace-qdrant-mcp patterns
- **Task Integration** - VS Code tasks for workspace operations
- **Debug Configuration** - Debugging setups for development
- **Keybind Shortcuts** - Custom keyboard shortcuts for efficiency

## 🏗️ Integration Structure

```
integrations/vscode/
├── README.md                    # This file
├── .vscode/                     # VS Code workspace configuration
│   ├── settings.json           # Workspace settings
│   ├── extensions.json         # Recommended extensions
│   ├── tasks.json             # Custom tasks
│   ├── launch.json            # Debug configurations
│   └── keybindings.json       # Custom keybindings
├── snippets/                   # Code snippets
│   ├── python-workspace-qdrant.json  # Python snippets
│   ├── markdown-notes.json           # Markdown note snippets
│   └── claude-prompts.json          # Claude prompt snippets
├── templates/                  # Project templates
│   ├── workspace_project/      # Complete project template
│   ├── research_project/       # Research project template
│   └── business_project/       # Business project template
└── scripts/                   # Automation scripts
    ├── setup_workspace.py     # Workspace setup automation
    ├── sync_collections.py    # Collection synchronization
    └── backup_knowledge.py    # Knowledge backup utility
```

## 🚀 Quick Setup

### 1. Install VS Code Extension

```bash
# Install VS Code if not already installed
# Download from: https://code.visualstudio.com/

# Install recommended extensions via command line
code --install-extension ms-python.python
code --install-extension ms-python.pylint
code --install-extension ms-toolsai.jupyter
code --install-extension ms-vscode.vscode-json
code --install-extension yzhang.markdown-all-in-one
code --install-extension ms-vscode-remote.remote-containers
```

### 2. Copy Configuration Files

```bash
# Navigate to your project directory
cd your-project-directory

# Copy VS Code configuration
cp -r examples/integrations/vscode/.vscode .vscode/

# Copy snippets to user directory (macOS example)
cp examples/integrations/vscode/snippets/* ~/Library/Application\ Support/Code/User/snippets/
```

### 3. Workspace Configuration

The provided `.vscode/settings.json` includes optimized settings for workspace-qdrant-mcp development:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "88"],
  
  "files.associations": {
    "*.md": "markdown",
    "*.txt": "plaintext",
    "claude_desktop_config.json": "jsonc"
  },
  
  "markdown.preview.fontSize": 14,
  "markdown.preview.lineHeight": 1.6,
  
  "json.schemas": [
    {
      "fileMatch": ["**/claude_desktop_config.json"],
      "url": "./schemas/claude-desktop-config.schema.json"
    }
  ],
  
  "workspace.saveWorkspaceOnExit": "always",
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 1000,
  
  "terminal.integrated.defaultProfile.osx": "zsh",
  "terminal.integrated.env.osx": {
    "QDRANT_URL": "http://localhost:6333"
  },
  
  "search.exclude": {
    "**/.venv": true,
    "**/venv": true,
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/node_modules": true
  }
}
```

## 📝 Code Snippets

### Python Snippets for workspace-qdrant-mcp

**Client Connection Snippet** (`wq-client`):
```python
from workspace_qdrant_mcp.client import WorkspaceClient

def create_client():
    """Create workspace-qdrant-mcp client connection."""
    client = WorkspaceClient()
    return client

# Usage
client = create_client()
```

**Store Document Snippet** (`wq-store`):
```python
# Store document in collection
client.store(
    content="${1:document_content}",
    metadata={
        "type": "${2:document_type}",
        "author": "${3:author_name}",
        "created_date": datetime.now().isoformat()
    },
    collection="${4:collection_name}"
)
```

**Search Documents Snippet** (`wq-search`):
```python
# Search documents in collection
results = client.search(
    query="${1:search_query}",
    collection="${2:collection_name}",
    metadata_filter={"type": "${3:document_type}"},
    limit=${4:10}
)

for result in results:
    print(f"Title: {result.get('metadata', {}).get('title', 'Unknown')}")
    print(f"Content: {result.get('content', '')[:200]}...")
    print("---")
```

**Collection Management Snippet** (`wq-collection`):
```python
# Collection management operations
from workspace_qdrant_mcp.collection_manager import CollectionManager

cm = CollectionManager(client)

# List collections
collections = cm.list_collections()
print(f"Available collections: {collections}")

# Create collection
cm.create_collection(
    name="${1:collection_name}",
    description="${2:collection_description}"
)

# Collection info
info = cm.get_collection_info("${3:collection_name}")
print(f"Collection info: {info}")
```

### Markdown Note Snippets

**Daily Note Template** (`daily-note`):
```markdown
# Daily Note - ${CURRENT_YEAR}-${CURRENT_MONTH}-${CURRENT_DATE}

## Today's Focus
- [ ] ${1:Primary focus area}
- [ ] ${2:Secondary focus area}
- [ ] ${3:Important task}

## Ideas & Insights
${4:Capture thoughts, ideas, and insights}

## Learnings
${5:What did I learn today?}

## Tomorrow's Preparation
${6:What needs attention tomorrow?}

## Gratitude
${7:What am I grateful for today?}

---
**Tags:** daily-note, ${CURRENT_YEAR}-${CURRENT_MONTH}
**Created:** ${CURRENT_YEAR}-${CURRENT_MONTH}-${CURRENT_DATE}
```

**Meeting Notes Template** (`meeting-notes`):
```markdown
# Meeting: ${1:Meeting Title}

**Date:** ${CURRENT_YEAR}-${CURRENT_MONTH}-${CURRENT_DATE}
**Time:** ${2:Meeting Time}
**Attendees:** ${3:List attendees}
**Meeting Type:** ${4:standup|planning|review|client}

## Agenda
1. ${5:Agenda item 1}
2. ${6:Agenda item 2}
3. ${7:Agenda item 3}

## Discussion Notes
${8:Meeting discussion and notes}

## Decisions Made
- ${9:Decision 1}
- ${10:Decision 2}

## Action Items
- [ ] **${11:Assignee}**: ${12:Action item 1} (Due: ${13:Date})
- [ ] **${14:Assignee}**: ${15:Action item 2} (Due: ${16:Date})

## Next Steps
${17:Next steps and follow-up}

---
**Tags:** meeting, ${4}, ${CURRENT_YEAR}-${CURRENT_MONTH}
```

## ⚙️ Custom Tasks

VS Code tasks for workspace-qdrant-mcp operations (`.vscode/tasks.json`):

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Start Qdrant Server",
            "type": "shell",
            "command": "docker",
            "args": ["run", "-d", "--name", "qdrant", "-p", "6333:6333", "qdrant/qdrant"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "panel": "new"
            },
            "problemMatcher": []
        },
        {
            "label": "Stop Qdrant Server",
            "type": "shell",
            "command": "docker",
            "args": ["stop", "qdrant", "&&", "docker", "rm", "qdrant"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "panel": "new"
            },
            "problemMatcher": []
        },
        {
            "label": "Test Workspace Connection",
            "type": "shell",
            "command": "python",
            "args": ["-c", "from workspace_qdrant_mcp.client import WorkspaceClient; client = WorkspaceClient(); print('✅ Connection successful')"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "panel": "new"
            },
            "problemMatcher": []
        },
        {
            "label": "List Collections",
            "type": "shell",
            "command": "python",
            "args": ["scripts/list_collections.py"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "panel": "new"
            },
            "problemMatcher": []
        },
        {
            "label": "Backup Collections",
            "type": "shell",
            "command": "python",
            "args": ["scripts/backup_knowledge.py", "--output", "backup_${workspaceFolderBasename}_$(date +%Y%m%d).json"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "panel": "new"
            },
            "problemMatcher": []
        },
        {
            "label": "Import Sample Data",
            "type": "shell",
            "command": "python",
            "args": ["examples/sample_data_import.py", "--project-name", "${workspaceFolderBasename}"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "panel": "new"
            },
            "problemMatcher": []
        },
        {
            "label": "Generate Daily Note",
            "type": "shell",
            "command": "python",
            "args": ["scripts/create_daily_note.py"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "panel": "new"
            },
            "problemMatcher": []
        }
    ]
}
```

## 🔍 Debug Configuration

Debug configurations for workspace-qdrant-mcp development (`.vscode/launch.json`):

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug MCP Server",
            "type": "python",
            "request": "launch",
            "module": "workspace_qdrant_mcp.server",
            "args": ["--transport", "stdio", "--log-level", "DEBUG"],
            "console": "integratedTerminal",
            "env": {
                "QDRANT_URL": "http://localhost:6333",
                "LOG_LEVEL": "DEBUG"
            }
        },
        {
            "name": "Debug Client Script",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {
                "QDRANT_URL": "http://localhost:6333",
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Debug with Claude Desktop",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "."
                }
            ]
        },
        {
            "name": "Test Suite",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v", "--tb=short"],
            "console": "integratedTerminal",
            "env": {
                "QDRANT_URL": "http://localhost:6333"
            }
        }
    ]
}
```

## ⌨️ Custom Keybindings

Productivity keybindings (`.vscode/keybindings.json`):

```json
[
    {
        "key": "ctrl+shift+q",
        "command": "workbench.action.tasks.runTask",
        "args": "Test Workspace Connection"
    },
    {
        "key": "ctrl+shift+l",
        "command": "workbench.action.tasks.runTask",
        "args": "List Collections"
    },
    {
        "key": "ctrl+shift+d",
        "command": "workbench.action.tasks.runTask",
        "args": "Generate Daily Note"
    },
    {
        "key": "ctrl+shift+b",
        "command": "workbench.action.tasks.runTask",
        "args": "Backup Collections"
    },
    {
        "key": "ctrl+shift+s",
        "command": "workbench.action.tasks.runTask",
        "args": "Start Qdrant Server"
    },
    {
        "key": "ctrl+shift+x",
        "command": "workbench.action.tasks.runTask",
        "args": "Stop Qdrant Server"
    }
]
```

## 🔄 Automated Scripts

### Workspace Setup Script

```python
#!/usr/bin/env python3
"""
setup_workspace.py - Automated VS Code workspace setup for workspace-qdrant-mcp
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

def setup_vscode_workspace(project_path: str, project_type: str = "general"):
    """
    Set up VS Code workspace for workspace-qdrant-mcp project.
    
    Args:
        project_path: Path to the project directory
        project_type: Type of project (general, research, business, personal)
    """
    project_dir = Path(project_path)
    vscode_dir = project_dir / ".vscode"
    
    # Create .vscode directory
    vscode_dir.mkdir(exist_ok=True)
    
    # Copy configuration files
    config_source = Path(__file__).parent / ".vscode"
    
    for config_file in config_source.glob("*.json"):
        dest_file = vscode_dir / config_file.name
        
        # Customize settings based on project type
        if config_file.name == "settings.json":
            settings = customize_settings_for_project_type(config_file, project_type)
            with open(dest_file, 'w') as f:
                json.dump(settings, f, indent=2)
        else:
            shutil.copy2(config_file, dest_file)
    
    # Create workspace file
    workspace_config = create_workspace_config(project_dir.name, project_type)
    workspace_file = project_dir / f"{project_dir.name}.code-workspace"
    
    with open(workspace_file, 'w') as f:
        json.dump(workspace_config, f, indent=2)
    
    # Copy snippets to user directory
    setup_user_snippets()
    
    # Create initial project structure
    create_project_structure(project_dir, project_type)
    
    print(f"✅ VS Code workspace setup complete for {project_dir.name}")
    print(f"📁 Open workspace: code {workspace_file}")

def customize_settings_for_project_type(settings_file: Path, project_type: str) -> dict:
    """Customize VS Code settings based on project type."""
    with open(settings_file) as f:
        settings = json.load(f)
    
    # Project-specific customizations
    if project_type == "research":
        settings["files.associations"].update({
            "*.bib": "bibtex",
            "*.tex": "latex"
        })
        settings["python.analysis.extraPaths"] = ["./research_tools"]
        
    elif project_type == "business":
        settings["files.associations"].update({
            "*.ppt": "plaintext",
            "*.pptx": "plaintext"
        })
        settings["markdown.preview.theme"] = "business"
        
    elif project_type == "personal":
        settings["files.autoSave"] = "onFocusChange"
        settings["markdown.preview.fontSize"] = 16
    
    # Add project-specific environment variables
    env_vars = {
        "COLLECTIONS": get_collections_for_project_type(project_type),
        "GLOBAL_COLLECTIONS": get_global_collections_for_project_type(project_type)
    }
    
    settings.setdefault("terminal.integrated.env.osx", {}).update(env_vars)
    settings.setdefault("terminal.integrated.env.linux", {}).update(env_vars)
    settings.setdefault("terminal.integrated.env.windows", {}).update(env_vars)
    
    return settings

def get_collections_for_project_type(project_type: str) -> str:
    """Get collection configuration for project type."""
    collections_map = {
        "general": "project,docs,notes",
        "research": "papers,notes,reviews,experiments",
        "business": "meetings,documents,projects,processes",
        "personal": "notes,ideas,learning,journal"
    }
    return collections_map.get(project_type, "project,docs,notes")

def get_global_collections_for_project_type(project_type: str) -> str:
    """Get global collection configuration for project type."""
    global_collections_map = {
        "general": "references,templates,best-practices",
        "research": "citations,methodologies,datasets",
        "business": "knowledge-base,policies,templates",
        "personal": "references,templates,inspiration"
    }
    return global_collections_map.get(project_type, "references,templates")

def create_workspace_config(project_name: str, project_type: str) -> dict:
    """Create VS Code workspace configuration."""
    return {
        "folders": [
            {
                "name": project_name,
                "path": "."
            }
        ],
        "settings": {
            "workspace.name": f"{project_name} - {project_type.title()}",
            "workspace.description": f"workspace-qdrant-mcp {project_type} project"
        },
        "extensions": {
            "recommendations": [
                "ms-python.python",
                "ms-python.pylint",
                "ms-toolsai.jupyter",
                "yzhang.markdown-all-in-one",
                "ms-vscode.vscode-json",
                "streetsidesoftware.code-spell-checker"
            ]
        }
    }

def setup_user_snippets():
    """Copy snippets to VS Code user directory."""
    import platform
    
    # Determine user snippets directory
    system = platform.system()
    if system == "Darwin":  # macOS
        snippets_dir = Path.home() / "Library/Application Support/Code/User/snippets"
    elif system == "Linux":
        snippets_dir = Path.home() / ".config/Code/User/snippets"
    elif system == "Windows":
        snippets_dir = Path.home() / "AppData/Roaming/Code/User/snippets"
    else:
        print(f"⚠️  Unknown system {system}, skipping snippets setup")
        return
    
    snippets_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy snippet files
    source_snippets = Path(__file__).parent / "snippets"
    for snippet_file in source_snippets.glob("*.json"):
        dest_file = snippets_dir / snippet_file.name
        shutil.copy2(snippet_file, dest_file)
    
    print(f"📝 Snippets installed to {snippets_dir}")

def create_project_structure(project_dir: Path, project_type: str):
    """Create initial project structure based on type."""
    # Create common directories
    common_dirs = ["docs", "scripts", "templates"]
    
    # Project-specific directories
    type_specific_dirs = {
        "research": ["papers", "data", "analysis", "notes"],
        "business": ["meetings", "reports", "processes", "templates"],
        "personal": ["notes", "ideas", "learning", "projects"],
        "general": ["src", "tests", "examples"]
    }
    
    dirs_to_create = common_dirs + type_specific_dirs.get(project_type, [])
    
    for dir_name in dirs_to_create:
        dir_path = project_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        
        # Create .gitkeep files
        (dir_path / ".gitkeep").touch()
    
    # Create initial README
    create_initial_readme(project_dir, project_type)

def create_initial_readme(project_dir: Path, project_type: str):
    """Create initial README file for the project."""
    readme_content = f"""# {project_dir.name}

A {project_type} project using workspace-qdrant-mcp for knowledge management.

## Setup

1. Ensure Qdrant server is running:
   ```bash
   docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
   ```

2. Install workspace-qdrant-mcp:
   ```bash
   pip install workspace-qdrant-mcp
   ```

3. Configure environment variables:
   ```bash
   export QDRANT_URL="http://localhost:6333"
   export COLLECTIONS="{get_collections_for_project_type(project_type)}"
   export GLOBAL_COLLECTIONS="{get_global_collections_for_project_type(project_type)}"
   ```

## VS Code Integration

This project is configured with:
- Custom tasks for workspace-qdrant-mcp operations
- Code snippets for common patterns
- Debug configurations
- Recommended extensions

### Quick Commands

- `Ctrl+Shift+Q`: Test workspace connection
- `Ctrl+Shift+L`: List collections
- `Ctrl+Shift+D`: Generate daily note
- `Ctrl+Shift+B`: Backup collections

## Usage with Claude

Store documents:
- "Store this note in my {project_type} collection: [content]"

Search knowledge:
- "Find all notes about [topic] in my {project_type} project"

## Project Structure

```
{project_dir.name}/
├── docs/           # Documentation files
├── scripts/        # Utility scripts
├── templates/      # Document templates
└── .vscode/        # VS Code configuration
```

Created: {datetime.now().strftime('%Y-%m-%d')}
"""
    
    readme_file = project_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up VS Code workspace for workspace-qdrant-mcp")
    parser.add_argument("project_path", help="Path to project directory")
    parser.add_argument("--type", choices=["general", "research", "business", "personal"], 
                       default="general", help="Type of project")
    
    args = parser.parse_args()
    
    setup_vscode_workspace(args.project_path, args.type)
```

## 💡 Best Practices

### Workspace Organization

1. **Use project-specific workspaces** for different contexts
2. **Configure environment variables** per workspace
3. **Utilize custom tasks** for common operations
4. **Leverage code snippets** for consistent patterns
5. **Set up debug configurations** for development

### Productivity Tips

1. **Keyboard shortcuts** for frequent operations
2. **Auto-save configuration** to prevent data loss
3. **Extension recommendations** for team consistency
4. **Search exclusions** to improve performance
5. **Terminal integration** with environment variables

### Integration Workflows

1. **Daily note creation** with automated templates
2. **Meeting note capture** with structured templates
3. **Code documentation** with workspace-qdrant-mcp snippets
4. **Knowledge backup** with scheduled tasks
5. **Collection synchronization** across team members

## 🔧 Troubleshooting

### Common Issues

**Extension not loading:**
```bash
# Reload VS Code window
Ctrl+Shift+P → "Developer: Reload Window"

# Check extension installation
code --list-extensions | grep python
```

**Task not running:**
```bash
# Verify task configuration
cat .vscode/tasks.json

# Test task manually
python scripts/test_connection.py
```

**Snippets not appearing:**
```bash
# Check snippets installation
ls ~/Library/Application\ Support/Code/User/snippets/

# Reload snippets
Ctrl+Shift+P → "Preferences: Configure User Snippets"
```

---

**Next Steps:**
1. Set up your [first workspace](templates/workspace_project/)
2. Try [Cursor Integration](../cursor/README.md) for AI-powered development
3. Explore [Automation Scripts](../automation/README.md) for workflow automation