# Test Result Notification System

Comprehensive notification system for CI/CD test results with multi-channel alerting, failure categorization, and escalation policies.

## Features

- **Multi-Channel Notifications**: Slack, email, GitHub issues
- **Intelligent Categorization**: Automatic failure classification
- **Escalation Policies**: Critical failures trigger enhanced alerting
- **Workflow Integration**: Monitors all test workflows automatically
- **Deduplication**: Avoids duplicate notifications for same issues
- **Rich Context**: Includes test counts, logs, and actionable links

## Notification Channels

### 1. Slack Notifications

**Triggered for**: All test failures and warnings

**Configuration**:
```yaml
# Required secrets in GitHub repository settings
SLACK_WEBHOOK_URL: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
SLACK_TEST_CHANNEL: #test-notifications  # Optional, defaults to #test-notifications
```

**Setup Slack Webhook**:
1. Go to Slack App settings: https://api.slack.com/apps
2. Create new app or select existing
3. Enable "Incoming Webhooks"
4. Add new webhook for desired channel
5. Copy webhook URL to GitHub secrets as `SLACK_WEBHOOK_URL`

**Message Format**:
- **Color-coded**: Red (critical), yellow (warning), gray (info)
- **Severity level**: CRITICAL, WARNING, INFO
- **Test metrics**: Failed/total tests
- **Context**: Branch, commit, author
- **Quick links**: Direct links to workflow run

### 2. Email Notifications

**Triggered for**: Critical failures only (10+ failed tests or regression detected)

**Configuration**:
```yaml
# Required secrets
EMAIL_USERNAME: github-actions@example.com
EMAIL_PASSWORD: app-specific-password

# Optional environment variable
EMAIL_RECIPIENTS: dev-team@example.com,ops@example.com
```

**Setup Email**:
1. **Gmail**: Use App-Specific Password (2FA required)
   - Go to: https://myaccount.google.com/apppasswords
   - Generate password for "GitHub Actions"
   - Add to `EMAIL_PASSWORD` secret

2. **Other SMTP**: Update server settings in workflow
   ```yaml
   server_address: smtp.yourserver.com
   server_port: 465  # or 587 for TLS
   ```

**Email Content**:
- HTML formatted with critical failure styling
- Test metrics and failure details
- Commit information
- Direct link to workflow run
- Action required section

### 3. GitHub Issues

**Triggered for**: Critical failures on main branch

**Configuration**: No secrets required (uses `GITHUB_TOKEN`)

**Behavior**:
- **New issue**: Created for first critical failure of workflow type
- **Update existing**: Adds comment for repeated failures
- **Labels**: `critical-test-failure`, `automated`, `bug`, `high-priority`
- **Auto-escalation**: Repeated failures get `high-priority` label

**Issue Template**:
```markdown
## 🚨 Critical Test Failure Alert

**Workflow:** [Workflow Name]
**Category:** [unit_test_failure|integration_failure|stability_issue|performance_regression]
**Failed Tests:** X / Y

### Details
- Branch, commit, author
- Run URL
- Timestamp

### Action Required
Investigation steps and resolution checklist
```

## Failure Categorization

Automatic classification based on workflow type and failure count:

| Category | Criteria | Notification Level | Escalation |
|----------|----------|-------------------|------------|
| `unit_test_failure` | Unit test workflow failed | Warning | Critical if ≥10 failures |
| `integration_failure` | Integration test failed | Error | Critical if ≥5 failures |
| `stability_issue` | Stress/E2E test failed | Error | Always critical |
| `performance_regression` | Performance threshold exceeded | Error | Always critical |
| `test_failure` | Other test failures | Warning | Critical if ≥10 failures |

## Escalation Thresholds

Configurable thresholds (set in workflow env):

```yaml
env:
  CRITICAL_FAILURE_THRESHOLD: 10        # Failed tests for critical escalation
  REGRESSION_THRESHOLD: 3               # Performance regressions for critical
  CONSECUTIVE_FAILURE_THRESHOLD: 3      # Consecutive failures for escalation
```

## Monitored Workflows

The notification system automatically monitors:

1. **Unit Tests (Fast Feedback)** - `unit-tests.yml`
2. **Integration Tests (PR Validation)** - `integration-tests.yml`
3. **Nightly Stress Tests (Stability)** - `nightly-stress-tests.yml`
4. **Performance Regression Detection** - `performance-regression.yml`
5. **E2E System Tests** - (when created)

Add new workflows to monitoring:
```yaml
on:
  workflow_run:
    workflows:
      - "Your New Test Workflow"
    types:
      - completed
```

## Testing Notifications

### Manual Trigger

Test notification system without running full test suite:

```bash
# Via GitHub CLI
gh workflow run test-notifications.yml \
  -f test_workflow="Unit Tests (Fast Feedback)" \
  -f test_status="failure" \
  -f failure_count="15"

# Via GitHub UI
# Actions → Test Result Notifications → Run workflow
# Select workflow, status, and failure count
```

### Test Scenarios

1. **Warning notification**:
   ```bash
   gh workflow run test-notifications.yml \
     -f test_workflow="Unit Tests (Fast Feedback)" \
     -f test_status="failure" \
     -f failure_count="5"
   ```
   Result: Slack notification only (non-critical)

2. **Critical notification**:
   ```bash
   gh workflow run test-notifications.yml \
     -f test_workflow="Unit Tests (Fast Feedback)" \
     -f test_status="failure" \
     -f failure_count="15"
   ```
   Result: Slack + email + GitHub issue (critical)

3. **Regression notification**:
   ```bash
   gh workflow run test-notifications.yml \
     -f test_workflow="Performance Regression Detection" \
     -f test_status="failure" \
     -f failure_count="3"
   ```
   Result: All channels (always critical)

## Notification Flow

```
Test Workflow Completes
         ↓
  workflow_run trigger
         ↓
  Analyze Results Job
    - Download artifacts
    - Parse JUnit XML
    - Categorize failure
    - Determine severity
         ↓
    ┌────┴────┐
    ↓         ↓         ↓
  Slack   Email    GitHub Issue
  (all)  (critical)  (critical + main)
    ↓         ↓         ↓
  Notification Summary
```

## Customization

### Adjust Notification Levels

Edit thresholds in `test-notifications.yml`:

```yaml
env:
  CRITICAL_FAILURE_THRESHOLD: 15  # Increase for less sensitive alerts
  REGRESSION_THRESHOLD: 5         # More regressions before critical
```

### Customize Slack Message

Edit `slack-message.json` generation in workflow:

```json
{
  "attachments": [{
    "color": "$COLOR",
    "fields": [
      // Add custom fields here
      {
        "title": "Custom Field",
        "value": "Custom Value",
        "short": true
      }
    ]
  }]
}
```

### Add Custom Notification Channel

Example: Discord webhook

```yaml
- name: Send to Discord
  if: needs.analyze-results.outputs.is_critical == 'true'
  run: |
    curl -X POST "${{ secrets.DISCORD_WEBHOOK_URL }}" \
      -H "Content-Type: application/json" \
      -d '{
        "content": "🚨 Critical test failure detected",
        "embeds": [{
          "title": "${{ needs.analyze-results.outputs.workflow_name }}",
          "description": "Failed tests: ${{ needs.analyze-results.outputs.failed_tests }}",
          "color": 15158332
        }]
      }'
```

## Troubleshooting

### Slack notifications not sending

1. **Check webhook URL**: Verify `SLACK_WEBHOOK_URL` secret is set correctly
2. **Test webhook**:
   ```bash
   curl -X POST "$SLACK_WEBHOOK_URL" \
     -H 'Content-Type: application/json' \
     -d '{"text": "Test message"}'
   ```
3. **Check Slack app permissions**: Ensure app has `incoming-webhook` scope
4. **Review workflow logs**: Check for curl errors in `slack-notification` job

### Email notifications not sending

1. **Gmail App Password**: Ensure using App-Specific Password, not account password
2. **2FA Required**: Gmail requires 2FA for App Passwords
3. **SMTP settings**: Verify server address and port
4. **Firewall**: Ensure GitHub Actions can reach SMTP server (port 465/587)
5. **Test manually**:
   ```bash
   echo "Test" | mail -s "Test" your-email@example.com
   ```

### GitHub issues not creating

1. **Check permissions**: Workflow needs `issues: write` permission (already set)
2. **Branch restriction**: Issues only created on `main` branch by design
3. **Existing issues**: System updates existing issue rather than creating duplicate
4. **Review logs**: Check `create-issue` job for API errors

### Test results not parsing

1. **Artifact format**: Ensure test workflows upload JUnit XML artifacts
2. **Artifact naming**: Must include "test-results" in artifact name
3. **XML schema**: Verify JUnit XML schema compatibility
4. **Manual verification**:
   ```bash
   # Download artifact manually and parse
   unzip test-results.zip
   python3 -c "import xml.etree.ElementTree as ET; tree = ET.parse('results.xml'); print(tree.getroot().attrib)"
   ```

## Security Considerations

1. **Secrets Management**:
   - Use GitHub repository secrets for sensitive data
   - Rotate webhook URLs and passwords regularly
   - Limit secret access to necessary environments

2. **Information Disclosure**:
   - Notifications may expose commit messages and branch names
   - Consider private Slack channels for sensitive projects
   - Review email recipients list regularly

3. **Webhook Security**:
   - Slack: Webhook URLs should be kept secret
   - Email: Use App-Specific passwords, not account passwords
   - Consider IP whitelisting if supported

4. **Rate Limiting**:
   - Slack: 1 message per second limit
   - Email: Varies by provider
   - GitHub Issues: Handled by GitHub API rate limiting

## Maintenance

### Regular Tasks

- [ ] Review notification recipients quarterly
- [ ] Rotate webhook URLs annually
- [ ] Update email passwords on 2FA changes
- [ ] Close resolved critical failure issues
- [ ] Archive old test result artifacts (automatic after 90 days)

### Monitoring

Track notification system health:

```bash
# Check recent notification workflows
gh run list --workflow=test-notifications.yml --limit=10

# Review failed notification runs
gh run list --workflow=test-notifications.yml --status=failure

# Check for orphaned critical issues
gh issue list --label=critical-test-failure --state=open
```

## Examples

### Complete Setup Checklist

```bash
# 1. Configure Slack
# - Create webhook in Slack App settings
# - Add to GitHub secrets

gh secret set SLACK_WEBHOOK_URL --body "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
gh secret set SLACK_TEST_CHANNEL --body "#test-notifications"

# 2. Configure Email
# - Generate App-Specific Password in Gmail
# - Add to GitHub secrets

gh secret set EMAIL_USERNAME --body "github-actions@example.com"
gh secret set EMAIL_PASSWORD --body "your-app-password"
gh secret set TEST_NOTIFICATION_EMAILS --body "team@example.com"

# 3. Test the system
gh workflow run test-notifications.yml \
  -f test_workflow="Unit Tests (Fast Feedback)" \
  -f test_status="failure" \
  -f failure_count="15"

# 4. Verify in Slack, email, and GitHub issues
gh issue list --label=critical-test-failure

# 5. Clean up test notifications
gh issue close <issue-number> --comment "Test notification - closing"
```

## Integration with Existing Workflows

No changes needed to existing test workflows - notification system automatically monitors via `workflow_run` triggers.

**Optional enhancement**: Add explicit artifact uploads for better result parsing:

```yaml
# In your test workflow
- name: Upload test results
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: test-results-${{ github.run_number }}
    path: |
      **/test-results.xml
      **/pytest-results.xml
      **/.pytest_cache/
    retention-days: 7
```

## Support

For issues or questions:
- GitHub Issues: Create issue with `notification-system` label
- Documentation: Review workflow comments in `test-notifications.yml`
- Testing: Use manual trigger to debug notification behavior
