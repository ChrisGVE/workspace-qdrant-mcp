"""FastAPI web server for memory curation interface."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..core.client import create_qdrant_client
from ..core.collection_naming import create_naming_manager
from ..core.config import Config
from ..memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryManager,
    create_memory_manager,
)
from ..memory.types import MemoryRule


class MemoryWebServer:
    """Web server for memory rule management."""

    def __init__(self, config: Config, port: int = 8000, host: str = "127.0.0.1"):
        self.config = config
        self.port = port
        self.host = host
        self.app = None
        self.memory_manager = None

    async def initialize(self):
        """Initialize the memory manager and FastAPI app."""
        # Initialize Qdrant client and memory manager
        client = create_qdrant_client(self.config.qdrant_client_config)
        naming_manager = create_naming_manager(self.config.workspace.global_collections)
        self.memory_manager = create_memory_manager(client, naming_manager)

        # Ensure memory collection exists
        await self.memory_manager.initialize_memory_collection()

        # Create FastAPI app
        self.app = create_web_app(self.memory_manager)

    async def start(self):
        """Start the web server."""
        if not self.app:
            await self.initialize()

        # Use uvicorn to run the server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False,
        )
        server = uvicorn.Server(config)
        await server.serve()


def create_web_app(memory_manager: MemoryManager) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Memory Curation Interface",
        description="Web interface for managing memory rules",
        version="1.0.0",
    )

    # Add CORS middleware for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Get the web directory path
    web_dir = Path(__file__).parent
    static_dir = web_dir / "static"
    templates_dir = web_dir / "templates"

    # Mount static files
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Initialize templates
    templates = Jinja2Templates(directory=str(templates_dir))

    # Store memory manager in app state
    app.state.memory_manager = memory_manager

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "memory-curation-web"}

    # Main interface route
    @app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request):
        """Serve the main interface."""
        return templates.TemplateResponse("index.html", {"request": request})

    # API Routes

    @app.get("/api/rules")
    async def get_rules(
        category: str | None = None,
        authority: str | None = None,
        scope: str | None = None,
    ):
        """Get all memory rules with optional filtering."""
        try:
            # Convert string parameters to enums
            category_enum = MemoryCategory(category) if category else None
            authority_enum = AuthorityLevel(authority) if authority else None

            # Get rules from memory manager
            rules = await memory_manager.list_memory_rules(
                category=category_enum, authority=authority_enum, scope=scope
            )

            # Convert to JSON-serializable format
            rules_data = []
            for rule in rules:
                rule_dict = {
                    "id": rule.id,
                    "category": rule.category.value,
                    "name": rule.name,
                    "rule": rule.rule,
                    "authority": rule.authority.value,
                    "scope": rule.scope,
                    "source": rule.source,
                    "created_at": rule.created_at.isoformat()
                    if rule.created_at
                    else None,
                    "updated_at": rule.updated_at.isoformat()
                    if rule.updated_at
                    else None,
                }
                rules_data.append(rule_dict)

            return {"rules": rules_data}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/rules/{rule_id}")
    async def get_rule(rule_id: str):
        """Get a specific memory rule."""
        try:
            rule = await memory_manager.get_memory_rule(rule_id)
            if not rule:
                raise HTTPException(status_code=404, detail="Rule not found")

            return {
                "id": rule.id,
                "category": rule.category.value,
                "name": rule.name,
                "rule": rule.rule,
                "authority": rule.authority.value,
                "scope": rule.scope,
                "source": rule.source,
                "created_at": rule.created_at.isoformat() if rule.created_at else None,
                "updated_at": rule.updated_at.isoformat() if rule.updated_at else None,
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/rules")
    async def create_rule(
        category: str = Form(...),
        name: str = Form(...),
        rule: str = Form(...),
        authority: str = Form(...),
        scope: str = Form(""),
    ):
        """Create a new memory rule."""
        try:
            # Parse scope
            scope_list = []
            if scope:
                scope_list = [s.strip() for s in scope.split(",") if s.strip()]

            # Convert to enums
            category_enum = MemoryCategory(category)
            authority_enum = AuthorityLevel(authority)

            # Add the rule
            rule_id = await memory_manager.add_memory_rule(
                category=category_enum,
                name=name,
                rule=rule,
                authority=authority_enum,
                scope=scope_list,
                source="web_user",
            )

            return {"id": rule_id, "message": "Rule created successfully"}

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.put("/api/rules/{rule_id}")
    async def update_rule(
        rule_id: str,
        name: str = Form(None),
        rule: str = Form(None),
        authority: str = Form(None),
        scope: str = Form(None),
    ):
        """Update an existing memory rule."""
        try:
            updates = {}

            if name is not None:
                updates["name"] = name
            if rule is not None:
                updates["rule"] = rule
            if authority is not None:
                updates["authority"] = AuthorityLevel(authority)
            if scope is not None:
                scope_list = (
                    [s.strip() for s in scope.split(",") if s.strip()] if scope else []
                )
                updates["scope"] = scope_list

            if not updates:
                return {"message": "No changes provided"}

            success = await memory_manager.update_memory_rule(rule_id, updates)

            if success:
                return {"message": "Rule updated successfully"}
            else:
                raise HTTPException(status_code=404, detail="Rule not found")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/rules/{rule_id}")
    async def delete_rule(rule_id: str):
        """Delete a memory rule."""
        try:
            success = await memory_manager.delete_memory_rule(rule_id)

            if success:
                return {"message": "Rule deleted successfully"}
            else:
                raise HTTPException(status_code=404, detail="Rule not found")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/stats")
    async def get_memory_stats():
        """Get memory usage statistics."""
        try:
            stats = await memory_manager.get_memory_stats()

            return {
                "total_rules": stats.total_rules,
                "estimated_tokens": stats.estimated_tokens,
                "rules_by_category": {
                    cat.value: count for cat, count in stats.rules_by_category.items()
                },
                "rules_by_authority": {
                    auth.value: count
                    for auth, count in stats.rules_by_authority.items()
                },
                "last_optimization": stats.last_optimization.isoformat()
                if stats.last_optimization
                else None,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/conflicts")
    async def get_conflicts():
        """Get memory rule conflicts."""
        try:
            conflicts = await memory_manager.detect_conflicts()

            conflicts_data = []
            for conflict in conflicts:
                conflict_dict = {
                    "conflict_type": conflict.conflict_type,
                    "confidence": conflict.confidence,
                    "description": conflict.description,
                    "rule1": {
                        "id": conflict.rule1.id,
                        "name": conflict.rule1.name,
                        "rule": conflict.rule1.rule,
                        "authority": conflict.rule1.authority.value,
                    },
                    "rule2": {
                        "id": conflict.rule2.id,
                        "name": conflict.rule2.name,
                        "rule": conflict.rule2.rule,
                        "authority": conflict.rule2.authority.value,
                    },
                    "resolution_options": conflict.resolution_options,
                }
                conflicts_data.append(conflict_dict)

            return {"conflicts": conflicts_data}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/enums")
    async def get_enums():
        """Get available enum values for dropdowns."""
        return {
            "categories": [cat.value for cat in MemoryCategory],
            "authorities": [auth.value for auth in AuthorityLevel],
        }

    @app.get("/api/optimize/suggestions")
    async def get_optimization_suggestions(target_tokens: int = 2000):
        """Get memory optimization suggestions."""
        try:
            # Get all rules
            rules = await memory_manager.list_memory_rules()

            # Get optimization suggestions
            suggestions = memory_manager.token_counter.suggest_memory_optimizations(
                rules, target_tokens
            )

            return suggestions

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/optimize/preview")
    async def preview_optimization(
        target_tokens: int = Form(...), preserve_absolute: bool = Form(True)
    ):
        """Preview rule optimization without applying changes."""
        try:
            # Get all rules
            rules = await memory_manager.list_memory_rules()

            # Get optimized rules
            optimized_rules, token_usage = (
                memory_manager.token_counter.optimize_rules_for_context(
                    rules, target_tokens, preserve_absolute
                )
            )

            # Convert to JSON-serializable format
            optimized_data = []
            removed_data = []

            optimized_ids = {rule.id for rule in optimized_rules}

            for rule in rules:
                rule_dict = {
                    "id": rule.id,
                    "name": rule.name,
                    "rule": rule.rule,
                    "authority": rule.authority.value,
                    "category": rule.category.value,
                    "tokens": memory_manager.token_counter.count_rule_tokens(rule),
                }

                if rule.id in optimized_ids:
                    optimized_data.append(rule_dict)
                else:
                    removed_data.append(rule_dict)

            return {
                "current_tokens": memory_manager.token_counter.count_rules_tokens(
                    rules
                ).total_tokens,
                "target_tokens": target_tokens,
                "optimized_tokens": token_usage.total_tokens,
                "rules_kept": len(optimized_rules),
                "rules_removed": len(rules) - len(optimized_rules),
                "optimized_rules": optimized_data,
                "removed_rules": removed_data,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/rules/reorder")
    async def reorder_rules(rule_ids: list[str] = Form(...)):
        """Reorder rules by priority (drag and drop)."""
        try:
            # This would need to be implemented with rule priority updates
            # For now, return success
            return {"message": "Rule reordering not yet implemented"}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/import/preview")
    async def preview_import(file: bytes = Form(...), filename: str = Form(...)):
        """Preview imported rules without applying them."""
        try:
            # Parse the imported file
            if filename.endswith(".json"):
                import_data = json.loads(file.decode("utf-8"))
            else:
                raise HTTPException(
                    status_code=400, detail="Only JSON files are supported"
                )

            # Validate the data structure
            if not isinstance(import_data, list):
                raise HTTPException(
                    status_code=400, detail="Expected an array of rules"
                )

            # Get current rules for conflict detection
            current_rules = await memory_manager.list_memory_rules()
            current_rule_names = {rule.name.lower() for rule in current_rules}

            # Analyze imported rules
            valid_rules = []
            invalid_rules = []
            conflicts = []

            for i, rule_data in enumerate(import_data):
                try:
                    # Basic validation
                    required_fields = ["name", "rule", "category", "authority"]
                    if not all(field in rule_data for field in required_fields):
                        invalid_rules.append(
                            {
                                "index": i,
                                "rule": rule_data,
                                "error": f"Missing required fields. Expected: {required_fields}",
                            }
                        )
                        continue

                    # Validate enum values
                    if rule_data["category"] not in [
                        cat.value for cat in MemoryCategory
                    ]:
                        invalid_rules.append(
                            {
                                "index": i,
                                "rule": rule_data,
                                "error": f"Invalid category: {rule_data['category']}",
                            }
                        )
                        continue

                    if rule_data["authority"] not in [
                        auth.value for auth in AuthorityLevel
                    ]:
                        invalid_rules.append(
                            {
                                "index": i,
                                "rule": rule_data,
                                "error": f"Invalid authority: {rule_data['authority']}",
                            }
                        )
                        continue

                    # Check for name conflicts
                    if rule_data["name"].lower() in current_rule_names:
                        conflicts.append(
                            {
                                "index": i,
                                "rule": rule_data,
                                "conflict_type": "name_duplicate",
                            }
                        )

                    valid_rules.append({"index": i, "rule": rule_data})

                except Exception as e:
                    invalid_rules.append(
                        {"index": i, "rule": rule_data, "error": str(e)}
                    )

            return {
                "total_rules": len(import_data),
                "valid_rules": valid_rules,
                "invalid_rules": invalid_rules,
                "conflicts": conflicts,
                "can_import": len(valid_rules) > 0,
            }

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/import/apply")
    async def apply_import(
        rules_to_import: str = Form(...), conflict_resolution: str = Form("skip")
    ):
        """Apply the import of selected rules."""
        try:
            # Parse the rules to import
            rules_data = json.loads(rules_to_import)

            imported_count = 0
            errors = []

            for rule_data in rules_data:
                try:
                    # Parse scope
                    scope_list = []
                    if "scope" in rule_data and rule_data["scope"]:
                        if isinstance(rule_data["scope"], list):
                            scope_list = rule_data["scope"]
                        else:
                            scope_list = [
                                s.strip()
                                for s in str(rule_data["scope"]).split(",")
                                if s.strip()
                            ]

                    # Convert to enums
                    category_enum = MemoryCategory(rule_data["category"])
                    authority_enum = AuthorityLevel(rule_data["authority"])

                    # Check if rule already exists
                    existing_rules = await memory_manager.list_memory_rules()
                    existing_names = [rule.name.lower() for rule in existing_rules]

                    if rule_data["name"].lower() in existing_names:
                        if conflict_resolution == "skip":
                            continue
                        elif conflict_resolution == "overwrite":
                            # Find and delete existing rule
                            for existing_rule in existing_rules:
                                if (
                                    existing_rule.name.lower()
                                    == rule_data["name"].lower()
                                ):
                                    await memory_manager.delete_memory_rule(
                                        existing_rule.id
                                    )
                                    break

                    # Add the rule
                    await memory_manager.add_memory_rule(
                        category=category_enum,
                        name=rule_data["name"],
                        rule=rule_data["rule"],
                        authority=authority_enum,
                        scope=scope_list,
                        source="web_import",
                    )

                    imported_count += 1

                except Exception as e:
                    errors.append(
                        {"rule_name": rule_data.get("name", "Unknown"), "error": str(e)}
                    )

            return {
                "imported_count": imported_count,
                "errors": errors,
                "message": f"Successfully imported {imported_count} rules",
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


async def start_web_server(config: Config, port: int = 8000, host: str = "127.0.0.1"):
    """Start the memory curation web server."""
    server = MemoryWebServer(config, port, host)
    await server.start()
