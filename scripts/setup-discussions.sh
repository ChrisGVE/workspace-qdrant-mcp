#!/bin/bash

# GitHub Discussions Setup Script
# This script helps maintainers set up GitHub Discussions for the Qdrant MCP repository

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_step() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Check if gh CLI is installed
check_gh_cli() {
    print_step "Checking GitHub CLI Installation"
    
    if ! command -v gh &> /dev/null; then
        print_error "GitHub CLI (gh) is not installed."
        print_info "Please install it from: https://cli.github.com/"
        exit 1
    fi
    
    print_success "GitHub CLI is installed"
    
    # Check if user is authenticated
    if ! gh auth status &> /dev/null; then
        print_error "You are not authenticated with GitHub CLI."
        print_info "Please run: gh auth login"
        exit 1
    fi
    
    print_success "GitHub CLI is authenticated"
}

# Check if we're in the right repository
check_repository() {
    print_step "Checking Repository Context"
    
    if [ ! -d ".git" ]; then
        print_error "This script must be run from the root of the repository."
        exit 1
    fi
    
    REPO_NAME=$(basename "$(git rev-parse --show-toplevel)")
    if [ "$REPO_NAME" != "workspace-qdrant-mcp" ]; then
        print_warning "Repository name '$REPO_NAME' doesn't match expected 'workspace-qdrant-mcp'"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    print_success "Repository context verified"
}

# Check if discussions are already enabled
check_discussions_status() {
    print_step "Checking Discussions Status"
    
    # Try to access discussions - this will fail if discussions are not enabled
    if gh api repos/:owner/:repo/discussions --paginate &> /dev/null; then
        print_warning "Discussions appear to already be enabled for this repository."
        read -p "Continue with setup anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Exiting. You can manually review your discussions setup."
            exit 0
        fi
    else
        print_info "Discussions are not yet enabled. This script will guide you through the setup."
    fi
}

# Enable discussions (this must be done manually via web interface)
enable_discussions() {
    print_step "Enabling GitHub Discussions"
    
    REPO_URL=$(gh repo view --json url -q .url)
    
    print_warning "Discussions must be enabled manually through the GitHub web interface."
    print_info "Please follow these steps:"
    print_info "1. Go to: ${REPO_URL}/settings"
    print_info "2. Scroll down to the 'Features' section"
    print_info "3. Check the box next to 'Discussions'"
    print_info "4. Click 'Set up discussions'"
    
    read -p "Press Enter after you've enabled discussions..." -r
    
    # Verify discussions are now enabled
    if gh api repos/:owner/:repo/discussions --paginate &> /dev/null; then
        print_success "Discussions are now enabled!"
    else
        print_error "Discussions don't appear to be enabled yet. Please try again."
        exit 1
    fi
}

# Create discussion categories
create_categories() {
    print_step "Creating Discussion Categories"
    
    print_info "GitHub doesn't provide API access to create categories programmatically."
    print_info "Please create the following categories manually:"
    
    echo ""
    echo "📂 Categories to create:"
    echo "===================="
    
    categories=(
        "💬|General|open-ended|Community introductions, casual conversation, general questions about vector databases and MCP, and community announcements."
        "❓|Q&A|question-answer|Technical questions, troubleshooting help, and \"how do I...?\" discussions. Mark helpful answers to build our searchable knowledge base."
        "💡|Ideas|open-ended|Feature requests, enhancement suggestions, and community-driven development proposals. Help shape the future of Qdrant MCP."
        "🎉|Show and tell|open-ended|Share your projects, integrations, performance results, and success stories. Inspire the community with your creativity!"
        "🆘|Help|question-answer|Specific troubleshooting requests, installation issues, configuration problems, and urgent support needs."
        "🛠️|Development|open-ended|Technical architecture discussions, code reviews, performance optimization, and contributor coordination."
    )
    
    for category in "${categories[@]}"; do
        IFS='|' read -r emoji name format description <<< "$category"
        echo ""
        echo "Name: $name"
        echo "Emoji: $emoji"
        echo "Format: $format"
        echo "Description: $description"
        echo "---"
    done
    
    echo ""
    REPO_URL=$(gh repo view --json url -q .url)
    print_info "Create these at: ${REPO_URL}/discussions/categories"
    
    read -p "Press Enter after you've created the categories..." -r
}

# Create initial discussions
create_initial_discussions() {
    print_step "Creating Initial Discussions"
    
    print_info "Creating welcome discussion in General category..."
    
    # Check if welcome discussion already exists
    if gh api repos/:owner/:repo/discussions --paginate | jq -r '.[].title' | grep -q "Welcome to the Qdrant MCP Community"; then
        print_warning "Welcome discussion already exists. Skipping creation."
    else
        # We'll create a simplified version since we can't easily upload the full template
        WELCOME_BODY="# Welcome to the Qdrant MCP Community! 🎉

Welcome to GitHub Discussions for Qdrant MCP! 

For the full welcome post content, see: docs/discussion-templates/welcome-post.md

## Quick Links
- [Community Guidelines](docs/COMMUNITY_GUIDELINES.md)
- [Discussion Guide](docs/DISCUSSIONS_GUIDE.md)
- [Installation Guide](INSTALLATION.md)
- [Examples](examples/)

We're excited to have you here! Feel free to introduce yourself and let us know what brings you to Qdrant MCP."

        if gh api repos/:owner/:repo/discussions \
            --method POST \
            --field title="Welcome to the Qdrant MCP Community! 🎉" \
            --field body="$WELCOME_BODY" \
            --field category_id="$(get_category_id "General")" &> /dev/null; then
            print_success "Welcome discussion created!"
        else
            print_warning "Could not create welcome discussion automatically. Please create it manually."
        fi
    fi
    
    print_info "For additional initial discussions, see templates in docs/discussion-templates/"
}

# Helper function to get category ID (simplified - may need manual adjustment)
get_category_id() {
    local category_name="$1"
    # This is a placeholder - GitHub's API for category management is limited
    echo "general_category_id"
}

# Verify discussion templates
verify_templates() {
    print_step "Verifying Discussion Templates"
    
    templates=(
        ".github/DISCUSSION_TEMPLATE/question.yml"
        ".github/DISCUSSION_TEMPLATE/idea.yml"
        ".github/DISCUSSION_TEMPLATE/show-and-tell.yml"
        ".github/DISCUSSION_TEMPLATE/help.yml"
        ".github/DISCUSSION_TEMPLATE/development.yml"
    )
    
    all_exist=true
    for template in "${templates[@]}"; do
        if [ -f "$template" ]; then
            print_success "✓ $template"
        else
            print_error "✗ $template (missing)"
            all_exist=false
        fi
    done
    
    if [ "$all_exist" = true ]; then
        print_success "All discussion templates are in place!"
    else
        print_error "Some discussion templates are missing. Please ensure all templates are committed to the repository."
        exit 1
    fi
}

# Verify documentation
verify_documentation() {
    print_step "Verifying Community Documentation"
    
    docs=(
        "docs/COMMUNITY_GUIDELINES.md"
        "docs/DISCUSSIONS_GUIDE.md"
        "docs/DISCUSSIONS_SETUP.md"
        "docs/discussion-templates/welcome-post.md"
        "docs/discussion-templates/feature-roadmap.md"
        "docs/discussion-templates/use-case-sharing.md"
    )
    
    all_exist=true
    for doc in "${docs[@]}"; do
        if [ -f "$doc" ]; then
            print_success "✓ $doc"
        else
            print_error "✗ $doc (missing)"
            all_exist=false
        fi
    done
    
    if [ "$all_exist" = true ]; then
        print_success "All community documentation is in place!"
    else
        print_error "Some documentation files are missing. Please ensure all docs are committed to the repository."
        exit 1
    fi
}

# Main setup flow
main() {
    echo -e "${BLUE}"
    echo "🚀 GitHub Discussions Setup for Qdrant MCP"
    echo "=========================================="
    echo -e "${NC}"
    
    check_gh_cli
    check_repository
    verify_templates
    verify_documentation
    check_discussions_status
    
    print_info "This script will guide you through setting up GitHub Discussions."
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Setup cancelled."
        exit 0
    fi
    
    enable_discussions
    create_categories
    create_initial_discussions
    
    print_step "Setup Complete!"
    
    REPO_URL=$(gh repo view --json url -q .url)
    
    print_success "GitHub Discussions setup is complete!"
    print_info ""
    print_info "Next steps:"
    print_info "1. Visit your discussions: ${REPO_URL}/discussions"
    print_info "2. Pin important discussions (welcome, roadmap)"
    print_info "3. Set up notifications for your team"
    print_info "4. Review and customize categories as needed"
    print_info "5. Create additional seed discussions from templates"
    print_info ""
    print_info "For detailed setup instructions, see: docs/DISCUSSIONS_SETUP.md"
    print_info ""
    print_success "Happy community building! 🎉"
}

# Run the main function
main "$@"