"""FastAPI web server for memory curation interface."""

import json
import os
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..core.config import Config
from ..core.client import create_qdrant_client
from ..core.collection_naming import create_naming_manager
from ..memory import (
    MemoryManager,
    MemoryCategory,
    AuthorityLevel,
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
            access_log=False
        )
        server = uvicorn.Server(config)
        await server.serve()


def create_web_app(memory_manager: MemoryManager) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Memory Curation Interface",
        description="Web interface for managing memory rules",
        version="1.0.0"
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
        category: Optional[str] = None,
        authority: Optional[str] = None,
        scope: Optional[str] = None
    ):
        """Get all memory rules with optional filtering."""
        try:
            # Convert string parameters to enums
            category_enum = MemoryCategory(category) if category else None
            authority_enum = AuthorityLevel(authority) if authority else None
            
            # Get rules from memory manager
            rules = await memory_manager.list_memory_rules(
                category=category_enum,
                authority=authority_enum,
                scope=scope
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
                    "created_at": rule.created_at.isoformat() if rule.created_at else None,
                    "updated_at": rule.updated_at.isoformat() if rule.updated_at else None
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
                "updated_at": rule.updated_at.isoformat() if rule.updated_at else None
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
                source="web_user"
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
                scope_list = [s.strip() for s in scope.split(",") if s.strip()] if scope else []
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
                "rules_by_category": {cat.value: count for cat, count in stats.rules_by_category.items()},
                "rules_by_authority": {auth.value: count for auth, count in stats.rules_by_authority.items()},
                "last_optimization": stats.last_optimization.isoformat() if stats.last_optimization else None
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
                    "resolution_options": conflict.resolution_options
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
            "authorities": [auth.value for auth in AuthorityLevel]
        }
    
    return app


async def start_web_server(config: Config, port: int = 8000, host: str = "127.0.0.1"):
    """Start the memory curation web server."""
    server = MemoryWebServer(config, port, host)
    await server.start()