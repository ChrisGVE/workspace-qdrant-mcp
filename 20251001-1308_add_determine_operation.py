#!/usr/bin/env python3
"""
Script to add _determine_operation_type() method to FileWatcher class.

This script surgically inserts the new method and updates related methods.
"""

from pathlib import Path

file_path = Path(__file__).parent / "src" / "python" / "common" / "core" / "file_watcher.py"

# Read the file
with open(file_path, 'r') as f:
    lines = f.readlines()

# Method to insert
new_method = '''    def _determine_operation_type(self, change_type: Change, file_path: Path) -> str:
        """
        Determine the operation type for a file change event.

        This method maps watchfiles Change types to queue operation types,
        handling race conditions where files may be deleted between detection
        and processing.

        Args:
            change_type: The watchfiles Change type (added, modified, deleted)
            file_path: Path to the file being changed

        Returns:
            Operation type string: 'ingest', 'update', or 'delete'

        Logic:
            - Change.added → 'ingest' (new file added to watch directory)
            - Change.modified + file exists → 'update' (existing file modified)
            - Change.modified + file missing → 'delete' (file deleted during processing)
            - Change.deleted → 'delete' (file explicitly deleted)

        Edge cases handled:
            - Race condition: File deleted between event and this check
            - Symlinks: Checked via exists() which follows symlinks by default
            - Broken symlinks: Treated as delete since target is unavailable
            - Permission errors: Default to 'update' to allow queue processor to handle
            - Special files: Filtered earlier in _handle_changes(), won't reach here

        Note:
            This method performs filesystem checks and should be called
            during the debounce period to handle transient file states.
            It's called AFTER filtering, so we know the file is relevant.
        """
        if change_type == Change.added:
            operation = 'ingest'
            logger.debug(f"Operation type 'ingest' determined for added file: {file_path}")
            return operation
        elif change_type == Change.deleted:
            operation = 'delete'
            logger.debug(f"Operation type 'delete' determined for deleted file: {file_path}")
            return operation
        elif change_type == Change.modified:
            # Check if file still exists to disambiguate modify vs delete
            # This handles race conditions where a file is deleted between
            # the event being generated and this method being called
            try:
                # Check for symlinks first
                if file_path.is_symlink():
                    # Symlink exists - check if target exists
                    if file_path.exists():
                        operation = 'update'
                        logger.debug(f"Operation type 'update' determined for modified symlink: {file_path}")
                    else:
                        # Broken symlink - treat as delete
                        operation = 'delete'
                        logger.debug(f"Operation type 'delete' determined for broken symlink: {file_path}")
                elif file_path.exists():
                    operation = 'update'
                    logger.debug(f"Operation type 'update' determined for modified file: {file_path}")
                else:
                    # File was deleted between event and check
                    operation = 'delete'
                    logger.debug(
                        f"Operation type 'delete' determined for modified file "
                        f"(race condition - file no longer exists): {file_path}"
                    )
                return operation
            except (OSError, PermissionError) as e:
                # If we can't check the file (permissions, disk error, etc.),
                # assume it's an update and let the queue processor handle errors
                logger.warning(
                    f"Could not check file existence for {file_path}: {e}. "
                    f"Defaulting to 'update' operation."
                )
                return 'update'
        else:
            # Unknown change type - default to ingest for safety
            logger.warning(f"Unknown change type {change_type} for {file_path}, defaulting to 'ingest'")
            return 'ingest'

'''

# Find insertion point (before _handle_changes at line 258)
insert_line = 257

# Insert the new method
lines.insert(insert_line, new_method)

# Now update _handle_changes to pass change_type
# Find the line that calls _debounce_ingestion (originally line 310, now shifted)
for i, line in enumerate(lines):
    if 'await self._debounce_ingestion(str(file_path))' in line:
        # Replace with version that passes change_type
        lines[i] = line.replace(
            'await self._debounce_ingestion(str(file_path))',
            'await self._debounce_ingestion(str(file_path), change_type)'
        )
        break

# Update _debounce_ingestion signature to accept change_type
for i, line in enumerate(lines):
    if 'async def _debounce_ingestion(self, file_path: str)' in line:
        lines[i] = '    async def _debounce_ingestion(self, file_path: str, change_type: Change) -> None:\n'
        # Update the call to _delayed_ingestion
        for j in range(i, min(i+10, len(lines))):
            if 'self._delayed_ingestion(file_path)' in lines[j]:
                lines[j] = lines[j].replace(
                    'self._delayed_ingestion(file_path)',
                    'self._delayed_ingestion(file_path, change_type)'
                )
                break
        break

# Update _delayed_ingestion to use _determine_operation_type
for i, line in enumerate(lines):
    if 'async def _delayed_ingestion(self, file_path: str)' in line:
        lines[i] = '    async def _delayed_ingestion(self, file_path: str, change_type: Change) -> None:\n'
        # Find and update the _trigger_operation call
        for j in range(i, min(i+15, len(lines))):
            if 'await self._trigger_operation(file_path, self.config.collection, "ingest")' in lines[j]:
                # Replace with version that determines operation type
                indent = ' ' * 12  # Preserve indentation
                lines[j] = f'''{indent}# Determine operation type after debounce period
{indent}# File state may have changed during the delay
{indent}path_obj = Path(file_path)
{indent}operation = self._determine_operation_type(change_type, path_obj)

{indent}# Trigger operation with determined type
{indent}await self._trigger_operation(file_path, self.config.collection, operation)
'''
                break
        break

# Write the modified file
with open(file_path, 'w') as f:
    f.writelines(lines)

print(f"Successfully updated {file_path}")
print("Changes made:")
print("1. Added _determine_operation_type() method")
print("2. Updated _debounce_ingestion() to accept change_type")
print("3. Updated _delayed_ingestion() to use _determine_operation_type()")
