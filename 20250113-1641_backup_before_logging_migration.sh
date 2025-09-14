#!/bin/bash
# Backup script for Task 215: Logging Migration
# Create backup branch before mass logging system changes

set -e

echo "Creating backup branch before logging migration..."

# Create backup branch
git checkout -b backup/logging-migration-$(date +%Y%m%d-%H%M)

# Return to main/current branch
git checkout -

echo "Backup branch created successfully"
echo "Current branch: $(git branch --show-current)"
echo "Ready to proceed with logging migration"