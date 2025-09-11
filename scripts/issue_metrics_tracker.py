#!/usr/bin/env python3
"""
GitHub Issue Metrics Tracker for workspace-qdrant-mcp

This script provides basic issue metrics tracking and reporting functionality
for the GitHub issue management system.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional

class IssueMetricsTracker:
    """Track and analyze GitHub issue metrics."""
    
    def __init__(self, repo: str = "ChrisGVE/workspace-qdrant-mcp"):
        self.repo = repo
        self.issues_data = []
        
    def fetch_issues(self, state: str = "all", limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch issues from GitHub API via gh CLI."""
        try:
            cmd = [
                "gh", "issue", "list",
                "--repo", self.repo,
                "--state", state,
                "--limit", str(limit),
                "--json", "number,title,labels,state,createdAt,closedAt,updatedAt,milestone,assignees,author"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.issues_data = json.loads(result.stdout)
            return self.issues_data
            
        except subprocess.CalledProcessError as e:
            print(f"Error fetching issues: {e}")
            print(f"stderr: {e.stderr}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return []
    
    def analyze_issue_distribution(self) -> Dict[str, Dict[str, int]]:
        """Analyze issue distribution by various categories."""
        analysis = {
            "by_state": Counter(),
            "by_priority": Counter(),
            "by_component": Counter(),
            "by_milestone": Counter(),
            "by_type": Counter()
        }
        
        for issue in self.issues_data:
            # State analysis
            analysis["by_state"][issue["state"]] += 1
            
            # Label-based analysis
            labels = [label["name"] for label in issue["labels"]]
            
            # Priority analysis
            priority_labels = [l for l in labels if any(p in l for p in ["critical", "high-priority", "medium-priority", "low-priority"])]
            if priority_labels:
                analysis["by_priority"][priority_labels[0]] += 1
            else:
                analysis["by_priority"]["no-priority"] += 1
            
            # Component analysis
            component_labels = [l for l in labels if l in [
                "daemon", "mcp-server", "cli", "web-ui", "auto-ingestion", 
                "service-management", "configuration", "documentation", "tests"
            ]]
            if component_labels:
                for component in component_labels:
                    analysis["by_component"][component] += 1
            else:
                analysis["by_component"]["no-component"] += 1
            
            # Milestone analysis
            milestone = issue.get("milestone")
            if milestone:
                analysis["by_milestone"][milestone["title"]] += 1
            else:
                analysis["by_milestone"]["no-milestone"] += 1
            
            # Type analysis
            type_labels = [l for l in labels if l in ["bug", "enhancement", "documentation"]]
            if type_labels:
                for type_label in type_labels:
                    analysis["by_type"][type_label] += 1
            else:
                analysis["by_type"]["no-type"] += 1
        
        return analysis
    
    def calculate_resolution_times(self) -> Dict[str, float]:
        """Calculate average resolution times for closed issues."""
        resolution_times = []
        
        for issue in self.issues_data:
            if issue["state"] == "CLOSED" and issue["closedAt"]:
                created = datetime.fromisoformat(issue["createdAt"].replace("Z", "+00:00"))
                closed = datetime.fromisoformat(issue["closedAt"].replace("Z", "+00:00"))
                resolution_time = (closed - created).total_seconds() / 3600  # hours
                resolution_times.append(resolution_time)
        
        if not resolution_times:
            return {"average": 0, "count": 0}
        
        return {
            "average": sum(resolution_times) / len(resolution_times),
            "count": len(resolution_times),
            "min": min(resolution_times),
            "max": max(resolution_times)
        }
    
    def identify_stale_issues(self, days_threshold: int = 30) -> List[Dict[str, Any]]:
        """Identify issues that haven't been updated recently."""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        stale_issues = []
        
        for issue in self.issues_data:
            if issue["state"] == "OPEN":
                updated = datetime.fromisoformat(issue["updatedAt"].replace("Z", "+00:00"))
                if updated.replace(tzinfo=None) < cutoff_date:
                    stale_issues.append({
                        "number": issue["number"],
                        "title": issue["title"],
                        "updated": issue["updatedAt"],
                        "days_stale": (datetime.now() - updated.replace(tzinfo=None)).days
                    })
        
        return sorted(stale_issues, key=lambda x: x["days_stale"], reverse=True)
    
    def generate_priority_report(self) -> Dict[str, Any]:
        """Generate priority-focused metrics report."""
        priority_metrics = defaultdict(lambda: {"open": 0, "closed": 0, "total": 0})
        
        for issue in self.issues_data:
            labels = [label["name"] for label in issue["labels"]]
            priority = "no-priority"
            
            for label in labels:
                if any(p in label for p in ["critical", "high-priority", "medium-priority", "low-priority"]):
                    priority = label
                    break
            
            priority_metrics[priority]["total"] += 1
            if issue["state"] == "OPEN":
                priority_metrics[priority]["open"] += 1
            else:
                priority_metrics[priority]["closed"] += 1
        
        return dict(priority_metrics)
    
    def generate_milestone_progress(self) -> Dict[str, Any]:
        """Generate milestone progress report."""
        milestone_progress = defaultdict(lambda: {"open": 0, "closed": 0, "total": 0})
        
        for issue in self.issues_data:
            milestone_title = "no-milestone"
            if issue.get("milestone"):
                milestone_title = issue["milestone"]["title"]
            
            milestone_progress[milestone_title]["total"] += 1
            if issue["state"] == "OPEN":
                milestone_progress[milestone_title]["open"] += 1
            else:
                milestone_progress[milestone_title]["closed"] += 1
        
        # Calculate completion percentages
        for milestone in milestone_progress:
            total = milestone_progress[milestone]["total"]
            closed = milestone_progress[milestone]["closed"]
            milestone_progress[milestone]["completion_percentage"] = (closed / total * 100) if total > 0 else 0
        
        return dict(milestone_progress)
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive metrics report."""
        if not self.issues_data:
            self.fetch_issues()
        
        distribution = self.analyze_issue_distribution()
        resolution_times = self.calculate_resolution_times()
        stale_issues = self.identify_stale_issues()
        priority_report = self.generate_priority_report()
        milestone_progress = self.generate_milestone_progress()
        
        report = f"""
# GitHub Issues Metrics Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Repository: {self.repo}
Total Issues Analyzed: {len(self.issues_data)}

## Issue Distribution

### By State
"""
        for state, count in distribution["by_state"].items():
            report += f"- {state}: {count}\n"
        
        report += "\n### By Priority\n"
        for priority, count in sorted(distribution["by_priority"].items()):
            report += f"- {priority}: {count}\n"
        
        report += "\n### By Component\n"
        for component, count in sorted(distribution["by_component"].items()):
            report += f"- {component}: {count}\n"
        
        report += "\n### By Milestone\n"
        for milestone, count in sorted(distribution["by_milestone"].items()):
            report += f"- {milestone}: {count}\n"
        
        report += f"""
## Resolution Metrics
- Average Resolution Time: {resolution_times['average']:.1f} hours
- Total Resolved Issues: {resolution_times['count']}
"""
        if resolution_times['count'] > 0:
            report += f"- Fastest Resolution: {resolution_times['min']:.1f} hours\n"
            report += f"- Slowest Resolution: {resolution_times['max']:.1f} hours\n"
        
        report += f"""
## Milestone Progress
"""
        for milestone, progress in milestone_progress.items():
            completion = progress["completion_percentage"]
            report += f"- **{milestone}**: {completion:.1f}% complete ({progress['closed']}/{progress['total']})\n"
        
        report += f"""
## Priority Issues Status
"""
        for priority, metrics in priority_report.items():
            total = metrics["total"]
            open_count = metrics["open"]
            report += f"- **{priority}**: {open_count}/{total} open\n"
        
        if stale_issues:
            report += f"""
## Stale Issues (>30 days since update)
"""
            for issue in stale_issues[:5]:  # Show top 5 stale issues
                report += f"- #{issue['number']}: {issue['title']} ({issue['days_stale']} days)\n"
        
        report += f"""
## Recommendations

### Critical Actions Needed
"""
        critical_open = priority_report.get("critical", {}).get("open", 0)
        high_open = priority_report.get("high-priority", {}).get("open", 0)
        
        if critical_open > 0:
            report += f"- ⚠️  {critical_open} critical issues require immediate attention\n"
        if high_open > 0:
            report += f"- 🔴 {high_open} high-priority issues should be addressed soon\n"
        if len(stale_issues) > 10:
            report += f"- 📅 {len(stale_issues)} stale issues need review and updates\n"
        
        # Milestone progress alerts
        for milestone, progress in milestone_progress.items():
            if milestone != "no-milestone" and progress["completion_percentage"] < 20 and progress["total"] > 3:
                report += f"- 🎯 {milestone} milestone progress is low ({progress['completion_percentage']:.1f}%)\n"
        
        return report


def main():
    """Main function to run the metrics tracker."""
    if len(sys.argv) > 1:
        repo = sys.argv[1]
    else:
        repo = "ChrisGVE/workspace-qdrant-mcp"
    
    tracker = IssueMetricsTracker(repo)
    
    print("Fetching issues...")
    tracker.fetch_issues()
    
    print("Generating comprehensive report...")
    report = tracker.generate_comprehensive_report()
    
    # Save report to file
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    filename = f"20250911-1635_issue_metrics_report_{timestamp}.md"
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {filename}")
    print("\nReport Summary:")
    print("=" * 50)
    print(report)


if __name__ == "__main__":
    main()