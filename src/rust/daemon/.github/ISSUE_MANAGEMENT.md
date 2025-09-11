# Issue Management Guide

This document describes how we manage issues in the Workspace Qdrant MCP project.

## Issue Templates

We provide structured templates for different types of issues:

### 🐛 Bug Reports
Use the **Bug Report** template for:
- Application crashes or errors
- Unexpected behavior
- Performance issues
- Installation problems

### ✨ Feature Requests
Use the **Feature Request** template for:
- New functionality suggestions
- API improvements
- Workflow enhancements
- Integration requests

### 💬 General Feedback
Use the **General Feedback** template for:
- Questions about usage
- Documentation improvements
- General suggestions
- User experience feedback

## Labels

We use a comprehensive labeling system to organize issues:

### Component Labels
- `daemon` - Issues related to the Rust memexd daemon
- `mcp-server` - Issues with the Python MCP server
- `cli` - Command-line interface (wqm) issues
- `web-ui` - Web interface problems
- `auto-ingestion` - File monitoring and ingestion
- `service-management` - System service installation/management
- `configuration` - Configuration management
- `documentation` - Documentation improvements

### Priority Labels
- `critical` - Blocking issues, system down
- `high-priority` - Important issues affecting core functionality
- `medium-priority` - Standard issues
- `low-priority` - Minor improvements, nice-to-haves

### Type Labels
- `bug` - Something isn't working correctly
- `enhancement` - New feature or improvement
- `question` - General questions
- `documentation` - Documentation related
- `good-first-issue` - Good for newcomers

### Platform Labels
- `macos` - macOS-specific issues
- `linux` - Linux-specific issues  
- `windows` - Windows-specific issues

### Status Labels
- `triage` - Needs initial review
- `in-progress` - Currently being worked on
- `blocked` - Waiting on external dependencies
- `duplicate` - Duplicate of another issue
- `wontfix` - Will not be fixed

## Automated Issue Management

Our GitHub Actions workflow automatically:

### Auto-Labeling
When issues are opened, they're automatically labeled based on:
- Keywords in title and description
- Component mentions (daemon, CLI, etc.)
- Platform information
- Priority indicators

### Auto-Closing
Issues are automatically closed when commits reference them with:
- `fixes #123`
- `closes #123`
- `resolves #123`

The system adds a comment with the commit details before closing.

### PR Integration
When pull requests are merged:
- Referenced issues get notified with PR links
- Related issues are updated with progress

## Issue Lifecycle

1. **Opened** - New issue created, gets `triage` label
2. **Triaged** - Team reviews, adds appropriate labels and milestone
3. **In Progress** - Work begins, issue marked `in-progress`
4. **Testing** - Fix implemented, testing in progress
5. **Resolved** - Issue closed via commit or manual closure

## Creating Quality Issues

### For Bug Reports
- Provide clear reproduction steps
- Include version information
- Add relevant logs and error messages
- Specify your operating system
- Describe expected vs actual behavior

### For Feature Requests
- Explain the problem you're trying to solve
- Describe your proposed solution
- Consider alternative approaches
- Provide use cases and examples
- Indicate priority and impact

### For Questions
- Check existing documentation first
- Search existing issues to avoid duplicates
- Provide context about your setup
- Be specific about what you're trying to achieve

## Milestones

We organize work using milestones:

- **v0.3.0** - Next minor release
- **v1.0.0** - Stable release
- **Future** - Ideas for later consideration
- **Documentation** - Documentation improvements

## Issue Assignment

Issues are assigned based on:
- Component expertise (daemon → Rust developers, MCP → Python developers)
- Availability and workload
- Contributor interest and skills

## Contributing

Want to help with issues?

1. Look for `good-first-issue` labels
2. Comment on issues you're interested in
3. Fork the repo and create a PR
4. Reference the issue number in your PR

## Communication

- Use issue comments for technical discussion
- @ mention relevant team members
- Keep discussions focused and constructive
- Use reactions (👍/👎) for simple agreement/disagreement

## Closing Issues

Issues should be closed when:
- The problem is fixed and verified
- The feature is implemented and tested
- The issue is a duplicate
- The issue is out of scope
- The reporter confirms resolution

Always add a comment explaining why an issue is being closed.

## Getting Help

If you need help with:
- **Using the software**: Create a question issue or check discussions
- **Contributing**: Look at good-first-issue labels or join discussions
- **Reporting bugs**: Use the bug report template
- **Requesting features**: Use the feature request template

For urgent issues affecting the core functionality, use the `critical` priority label.