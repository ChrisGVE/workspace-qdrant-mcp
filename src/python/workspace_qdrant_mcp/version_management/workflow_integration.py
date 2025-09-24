"""
User Workflow Integration for Version Management System.

This module implements Task 262 workflow integration requirements:

4. Workflow Integration:
   - Automatic conflict detection during ingestion
   - User notification and approval workflows
   - Batch conflict resolution for large imports
   - Integration with document processing pipeline
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass

from loguru import logger

from python.common.core.client import QdrantWorkspaceClient
from .version_manager import VersionManager, VersionConflict, ResolutionStrategy
from .archive_manager import ArchiveManager


class WorkflowStatus(Enum):
    """Status of version management workflows."""
    PENDING = "pending"                 # Awaiting user action
    IN_PROGRESS = "in_progress"         # Being processed
    COMPLETED = "completed"             # Successfully completed
    FAILED = "failed"                   # Failed with errors
    CANCELLED = "cancelled"             # User cancelled
    PARTIALLY_COMPLETED = "partially_completed"  # Some items succeeded, some failed


class UserDecision(Enum):
    """User decisions for conflict resolution."""
    KEEP_NEW = "keep_new"              # Keep the new version
    KEEP_EXISTING = "keep_existing"    # Keep the existing version
    MERGE = "merge"                    # Attempt to merge versions
    ARCHIVE_BOTH = "archive_both"      # Archive both versions
    MANUAL_REVIEW = "manual_review"    # Requires manual review
    SKIP = "skip"                      # Skip this conflict for now


@dataclass
class WorkflowStep:
    """Represents a step in the version management workflow."""
    step_id: str
    step_type: str
    description: str
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    dependencies: List[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class UserPrompt:
    """Represents a user prompt for conflict resolution."""
    prompt_id: str
    conflict: VersionConflict
    message: str
    options: List[Dict[str, Any]]
    created_at: datetime
    expires_at: Optional[datetime] = None
    user_response: Optional[UserDecision] = None
    response_metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchOperation:
    """Represents a batch version management operation."""
    batch_id: str
    operation_type: str
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    conflicts: List[VersionConflict]
    pending_prompts: List[UserPrompt]
    audit_log: List[Dict[str, Any]]


class WorkflowIntegrator:
    """
    Integrates version management with user workflows and batch operations.

    Provides automatic conflict detection, user notification systems,
    batch processing capabilities, and audit logging for version management operations.
    """

    def __init__(
        self,
        client: QdrantWorkspaceClient,
        version_manager: VersionManager,
        archive_manager: ArchiveManager
    ):
        """Initialize workflow integrator with required managers."""
        self.client = client
        self.version_manager = version_manager
        self.archive_manager = archive_manager

        # Workflow state storage
        self.active_workflows = {}
        self.pending_prompts = {}
        self.batch_operations = {}

        # Configuration
        self.auto_resolve_threshold = 0.3  # Auto-resolve conflicts below this severity
        self.prompt_timeout_hours = 24      # Hours before prompts expire
        self.max_concurrent_workflows = 10  # Maximum concurrent workflows

        # User notification callbacks
        self.notification_callbacks = []

    def register_notification_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for user notifications."""
        self.notification_callbacks.append(callback)

    async def process_document_ingestion_with_workflow(
        self,
        content: str,
        collection: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        auto_resolve_conflicts: bool = True,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process document ingestion with automatic conflict detection and workflow integration.

        This is the main entry point for document ingestion with version management.
        """
        workflow_id = str(uuid.uuid4())
        logger.info("Starting document ingestion workflow %s", workflow_id)

        try:
            # Step 1: Detect document type and extract version information
            doc_type = self.version_manager.detect_document_type(metadata, content)
            version_string = self.version_manager.extract_version_from_content(content, doc_type)

            # Step 2: Create version info for the new document
            from .version_manager import VersionInfo, FileFormat
            new_version_info = VersionInfo(
                version_string=version_string or datetime.now(timezone.utc).isoformat(),
                version_type="extracted" if version_string else "timestamp",
                document_type=doc_type,
                authority_level=metadata.get("authority_level", 0.7),
                timestamp=datetime.now(timezone.utc),
                content_hash=self.version_manager.calculate_content_hash(content),
                format=self.version_manager.get_file_format(metadata),
                metadata=metadata,
                point_id=""  # Will be set after successful ingestion
            )

            # Step 3: Check for conflicts
            document_id = document_id or str(uuid.uuid4())
            conflicts = await self.version_manager.find_conflicting_versions(
                document_id, collection, new_version_info
            )

            # Step 4: Handle conflicts based on configuration
            if conflicts and auto_resolve_conflicts:
                resolution_result = await self._handle_conflicts_automatically(
                    conflicts, new_version_info, collection, workflow_id
                )

                if resolution_result["requires_user_input"]:
                    # Create workflow for user interaction
                    return await self._create_user_workflow(
                        workflow_id, conflicts, new_version_info, content,
                        collection, document_id, user_context
                    )
            elif conflicts:
                # Always require user input when auto-resolution is disabled
                return await self._create_user_workflow(
                    workflow_id, conflicts, new_version_info, content,
                    collection, document_id, user_context
                )

            # Step 5: No conflicts or conflicts resolved automatically
            from workspace_qdrant_mcp.tools.documents import ingest_new_version
            result = await ingest_new_version(
                client=self.client,
                content=content,
                collection=collection,
                metadata=metadata,
                document_id=document_id,
                version=version_string,
                document_type=doc_type.value
            )

            if "error" not in result:
                logger.info("Successfully ingested document without conflicts: %s", workflow_id)
                return {
                    "success": True,
                    "workflow_id": workflow_id,
                    "document_id": document_id,
                    "conflicts_resolved": len(conflicts),
                    "ingestion_result": result
                }
            else:
                logger.error("Failed to ingest document: %s", result["error"])
                return {
                    "success": False,
                    "workflow_id": workflow_id,
                    "error": result["error"]
                }

        except Exception as e:
            logger.error("Document ingestion workflow %s failed: %s", workflow_id, e)
            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": f"Workflow failed: {e}"
            }

    async def _handle_conflicts_automatically(
        self,
        conflicts: List[VersionConflict],
        new_version_info,
        collection: str,
        workflow_id: str
    ) -> Dict[str, Any]:
        """Automatically resolve conflicts that meet auto-resolution criteria."""
        auto_resolved = []
        requires_user_input = []

        for conflict in conflicts:
            if conflict.conflict_severity < self.auto_resolve_threshold:
                # Auto-resolve low-severity conflicts
                try:
                    resolution_result = await self._apply_resolution_strategy(
                        conflict, conflict.recommended_strategy, workflow_id, collection
                    )
                    if resolution_result["success"]:
                        auto_resolved.append(conflict)
                    else:
                        requires_user_input.append(conflict)
                except Exception as e:
                    logger.warning("Auto-resolution failed for conflict: %s", e)
                    requires_user_input.append(conflict)
            else:
                requires_user_input.append(conflict)

        return {
            "auto_resolved": auto_resolved,
            "requires_user_input": requires_user_input,
            "requires_user_input": len(requires_user_input) > 0
        }

    async def _create_user_workflow(
        self,
        workflow_id: str,
        conflicts: List[VersionConflict],
        new_version_info,
        content: str,
        collection: str,
        document_id: str,
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create a user workflow for conflict resolution."""

        # Create user prompts for each conflict
        prompts = []
        for i, conflict in enumerate(conflicts):
            prompt = UserPrompt(
                prompt_id=f"{workflow_id}_prompt_{i}",
                conflict=conflict,
                message=self._generate_user_message(conflict, new_version_info),
                options=self._generate_resolution_options(conflict),
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc).replace(
                    hour=datetime.now(timezone.utc).hour + self.prompt_timeout_hours
                ) if self.prompt_timeout_hours > 0 else None
            )
            prompts.append(prompt)
            self.pending_prompts[prompt.prompt_id] = prompt

        # Create workflow
        workflow = {
            "workflow_id": workflow_id,
            "status": WorkflowStatus.PENDING,
            "conflicts": conflicts,
            "new_version_info": new_version_info,
            "content": content,
            "collection": collection,
            "document_id": document_id,
            "prompts": prompts,
            "user_context": user_context or {},
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }

        self.active_workflows[workflow_id] = workflow

        # Send notifications
        await self._send_user_notifications(workflow)

        return {
            "success": True,
            "workflow_id": workflow_id,
            "status": "pending_user_input",
            "prompts": [
                {
                    "prompt_id": p.prompt_id,
                    "message": p.message,
                    "options": p.options
                } for p in prompts
            ]
        }

    def _generate_user_message(
        self,
        conflict: VersionConflict,
        new_version_info
    ) -> str:
        """Generate user-friendly conflict resolution message."""

        base_message = f"Version conflict detected: {conflict.user_message}"

        conflict_details = []

        # Add version comparison details
        if len(conflict.conflicting_versions) >= 2:
            existing = conflict.conflicting_versions[1]
            new = conflict.conflicting_versions[0]

            conflict_details.append(f"Existing version: {existing.version_string} ({existing.format.extension})")
            conflict_details.append(f"New version: {new.version_string} ({new.format.extension})")
            conflict_details.append(f"Conflict severity: {conflict.conflict_severity:.2f}")

        # Add recommended action
        if conflict.recommended_strategy:
            strategy_descriptions = {
                ResolutionStrategy.FORMAT_PRECEDENCE: "Recommended: Use format precedence (higher quality format wins)",
                ResolutionStrategy.VERSION_PRECEDENCE: "Recommended: Use version precedence (newer version wins)",
                ResolutionStrategy.TIMESTAMP_PRECEDENCE: "Recommended: Use timestamp precedence (more recent wins)",
                ResolutionStrategy.USER_DECISION: "Recommended: Manual review required",
                ResolutionStrategy.ARCHIVE_ALL: "Recommended: Archive existing and keep new"
            }

            recommended_desc = strategy_descriptions.get(
                conflict.recommended_strategy,
                f"Recommended: {conflict.recommended_strategy.value}"
            )
            conflict_details.append(recommended_desc)

        return base_message + "\n\nDetails:\n" + "\n".join(f"• {detail}" for detail in conflict_details)

    def _generate_resolution_options(self, conflict: VersionConflict) -> List[Dict[str, Any]]:
        """Generate resolution options for user selection."""
        options = []

        # Standard options based on conflict type and available strategies
        for resolution_option in conflict.resolution_options:
            option = {
                "value": resolution_option["action"],
                "label": resolution_option["description"],
                "strategy": resolution_option["strategy"].value,
                "recommended": resolution_option["strategy"] == conflict.recommended_strategy
            }
            options.append(option)

        # Always add manual review option
        options.append({
            "value": "manual_review",
            "label": "Require manual review and intervention",
            "strategy": "manual_override",
            "recommended": False
        })

        # Add skip option for batch operations
        options.append({
            "value": "skip",
            "label": "Skip this conflict and continue",
            "strategy": "defer",
            "recommended": False
        })

        return options

    async def _send_user_notifications(self, workflow: Dict[str, Any]):
        """Send notifications to registered callbacks."""
        notification = {
            "type": "version_conflict",
            "workflow_id": workflow["workflow_id"],
            "conflicts_count": len(workflow["conflicts"]),
            "document_id": workflow["document_id"],
            "collection": workflow["collection"],
            "prompts": [
                {
                    "prompt_id": p.prompt_id,
                    "message": p.message,
                    "severity": p.conflict.conflict_severity
                } for p in workflow["prompts"]
            ],
            "created_at": workflow["created_at"].isoformat()
        }

        for callback in self.notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                logger.warning("Notification callback failed: %s", e)

    async def respond_to_prompt(
        self,
        prompt_id: str,
        decision: UserDecision,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process user response to a conflict resolution prompt."""

        if prompt_id not in self.pending_prompts:
            return {"error": f"Prompt {prompt_id} not found or already resolved"}

        prompt = self.pending_prompts[prompt_id]

        # Check if prompt has expired
        if (prompt.expires_at and
            datetime.now(timezone.utc) > prompt.expires_at):
            return {"error": "Prompt has expired"}

        try:
            # Record user response
            prompt.user_response = decision
            prompt.response_metadata = metadata or {}

            # Find the associated workflow
            workflow = None
            workflow_id = prompt_id.split("_prompt_")[0]

            if workflow_id in self.active_workflows:
                workflow = self.active_workflows[workflow_id]

            if not workflow:
                return {"error": "Associated workflow not found"}

            # Apply the user's decision
            resolution_result = await self._apply_user_decision(
                prompt, decision, workflow, metadata
            )

            # Remove from pending prompts
            del self.pending_prompts[prompt_id]

            # Check if all prompts in workflow are resolved
            remaining_prompts = [
                p for p in workflow["prompts"]
                if p.prompt_id in self.pending_prompts
            ]

            if not remaining_prompts:
                # All prompts resolved, complete the workflow
                completion_result = await self._complete_workflow(workflow_id)
                resolution_result.update(completion_result)

            return resolution_result

        except Exception as e:
            logger.error("Failed to process user response for prompt %s: %s", prompt_id, e)
            return {"error": f"Failed to process response: {e}"}

    async def _apply_user_decision(
        self,
        prompt: UserPrompt,
        decision: UserDecision,
        workflow: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply user decision to resolve conflict."""

        conflict = prompt.conflict
        workflow_id = workflow["workflow_id"]

        # Create audit log entry
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "user_decision",
            "user_decision": decision.value,
            "prompt_id": prompt.prompt_id,
            "conflict_type": conflict.conflict_type.value,
            "conflict_severity": conflict.conflict_severity,
            "metadata": metadata or {}
        }

        try:
            if decision == UserDecision.KEEP_NEW:
                # Archive existing version and keep new
                for existing_version in conflict.conflicting_versions[1:]:
                    await self.archive_manager.archive_document_version(
                        existing_version.point_id,
                        workflow["collection"],
                        superseded_by_point_id=None  # Will be set after new version is ingested
                    )

            elif decision == UserDecision.KEEP_EXISTING:
                # Don't ingest new version, keep existing
                audit_entry["action_taken"] = "rejected_new_version"

            elif decision == UserDecision.ARCHIVE_BOTH:
                # Archive all conflicting versions
                for version in conflict.conflicting_versions:
                    if version.point_id:  # Existing versions
                        await self.archive_manager.archive_document_version(
                            version.point_id,
                            workflow["collection"]
                        )

            elif decision == UserDecision.MANUAL_REVIEW:
                # Mark for manual review
                audit_entry["action_taken"] = "marked_for_manual_review"
                workflow["status"] = WorkflowStatus.PENDING
                # Don't auto-complete this workflow

            elif decision == UserDecision.SKIP:
                # Skip this conflict
                audit_entry["action_taken"] = "skipped"

            # Add audit entry to workflow
            if "audit_log" not in workflow:
                workflow["audit_log"] = []
            workflow["audit_log"].append(audit_entry)

            return {
                "success": True,
                "decision_applied": decision.value,
                "audit_entry": audit_entry
            }

        except Exception as e:
            logger.error("Failed to apply user decision %s: %s", decision.value, e)
            audit_entry["error"] = str(e)
            workflow["audit_log"].append(audit_entry)
            return {"error": f"Failed to apply decision: {e}"}

    async def _complete_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Complete a workflow after all prompts are resolved."""

        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}

        workflow = self.active_workflows[workflow_id]

        try:
            # Check if we should proceed with ingestion
            should_ingest = True
            skip_reasons = []

            for prompt in workflow["prompts"]:
                if prompt.user_response == UserDecision.KEEP_EXISTING:
                    should_ingest = False
                    skip_reasons.append("User chose to keep existing version")
                elif prompt.user_response == UserDecision.MANUAL_REVIEW:
                    should_ingest = False
                    skip_reasons.append("Manual review required")
                elif prompt.user_response == UserDecision.SKIP:
                    skip_reasons.append("User skipped conflict")

            if should_ingest:
                # Proceed with document ingestion
                from workspace_qdrant_mcp.tools.documents import ingest_new_version

                result = await ingest_new_version(
                    client=self.client,
                    content=workflow["content"],
                    collection=workflow["collection"],
                    metadata=workflow["new_version_info"].metadata,
                    document_id=workflow["document_id"],
                    version=workflow["new_version_info"].version_string,
                    document_type=workflow["new_version_info"].document_type.value
                )

                if "error" not in result:
                    workflow["status"] = WorkflowStatus.COMPLETED
                    workflow["ingestion_result"] = result
                else:
                    workflow["status"] = WorkflowStatus.FAILED
                    workflow["error"] = result["error"]
            else:
                workflow["status"] = WorkflowStatus.CANCELLED
                workflow["skip_reasons"] = skip_reasons

            workflow["completed_at"] = datetime.now(timezone.utc)

            # Remove from active workflows
            del self.active_workflows[workflow_id]

            logger.info("Completed workflow %s with status %s", workflow_id, workflow["status"].value)

            return {
                "success": True,
                "workflow_completed": True,
                "status": workflow["status"].value,
                "ingestion_result": workflow.get("ingestion_result"),
                "skip_reasons": workflow.get("skip_reasons", [])
            }

        except Exception as e:
            logger.error("Failed to complete workflow %s: %s", workflow_id, e)
            workflow["status"] = WorkflowStatus.FAILED
            workflow["error"] = str(e)
            return {"error": f"Failed to complete workflow: {e}"}

    async def _apply_resolution_strategy(
        self,
        conflict: VersionConflict,
        strategy: ResolutionStrategy,
        workflow_id: str,
        collection: str
    ) -> Dict[str, Any]:
        """Apply a specific resolution strategy to a conflict."""

        try:
            if strategy == ResolutionStrategy.FORMAT_PRECEDENCE:
                # Use format with higher precedence
                versions = sorted(
                    conflict.conflicting_versions,
                    key=lambda v: v.format.precedence,
                    reverse=True
                )
                preferred_version = versions[0]

                # Archive other versions
                for version in versions[1:]:
                    if version.point_id:
                        await self.archive_manager.archive_document_version(
                            version.point_id,
                            collection,
                            superseded_by_point_id=preferred_version.point_id
                        )

            elif strategy == ResolutionStrategy.VERSION_PRECEDENCE:
                # Use version comparison to determine precedence
                versions = conflict.conflicting_versions
                if len(versions) >= 2:
                    new_version, existing_version = versions[0], versions[1]
                    version_cmp = self.version_manager.compare_versions(
                        new_version.version_string,
                        existing_version.version_string,
                        new_version.document_type
                    )

                    if version_cmp <= 0:  # Existing version is newer or equal
                        # Don't ingest new version
                        return {"success": True, "action": "rejected_new_version"}

            elif strategy == ResolutionStrategy.TIMESTAMP_PRECEDENCE:
                # Use most recent timestamp
                versions = sorted(
                    conflict.conflicting_versions,
                    key=lambda v: v.timestamp,
                    reverse=True
                )
                preferred_version = versions[0]

                # Archive older versions
                for version in versions[1:]:
                    if version.point_id:
                        await self.archive_manager.archive_document_version(
                            version.point_id,
                            collection,
                            superseded_by_point_id=preferred_version.point_id
                        )

            elif strategy == ResolutionStrategy.ARCHIVE_ALL:
                # Archive all existing versions
                for version in conflict.conflicting_versions:
                    if version.point_id:
                        await self.archive_manager.archive_document_version(
                            version.point_id,
                            collection
                        )

            return {"success": True, "strategy_applied": strategy.value}

        except Exception as e:
            logger.error("Failed to apply resolution strategy %s: %s", strategy.value, e)
            return {"success": False, "error": str(e)}

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a specific workflow."""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            return {
                "workflow_id": workflow_id,
                "status": workflow["status"].value,
                "conflicts_count": len(workflow["conflicts"]),
                "pending_prompts": len([p for p in workflow["prompts"] if p.prompt_id in self.pending_prompts]),
                "created_at": workflow["created_at"].isoformat(),
                "updated_at": workflow["updated_at"].isoformat()
            }
        elif workflow_id in self.batch_operations:
            batch = self.batch_operations[workflow_id]
            return {
                "batch_id": workflow_id,
                "status": batch.status.value,
                "total_items": batch.total_items,
                "processed_items": batch.processed_items,
                "successful_items": batch.successful_items,
                "failed_items": batch.failed_items,
                "created_at": batch.created_at.isoformat(),
                "updated_at": batch.updated_at.isoformat()
            }
        else:
            return {"error": "Workflow not found"}

    def list_pending_prompts(self) -> List[Dict[str, Any]]:
        """List all pending user prompts."""
        return [
            {
                "prompt_id": prompt.prompt_id,
                "message": prompt.message,
                "options": prompt.options,
                "created_at": prompt.created_at.isoformat(),
                "expires_at": prompt.expires_at.isoformat() if prompt.expires_at else None,
                "severity": prompt.conflict.conflict_severity
            }
            for prompt in self.pending_prompts.values()
        ]