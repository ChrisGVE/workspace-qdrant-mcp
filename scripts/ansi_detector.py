#!/usr/bin/env python3
"""
ANSI Escape Sequence Detector for Git History

Scans git commit messages and identifies commits containing ANSI escape sequences.
Outputs list of affected commit hashes with examples of detected sequences.

ANSI patterns detected:
- \\[38;2;R;G;Bm - 24-bit true color codes
- \\[0m - Reset codes
- \\033[ - Escape sequences
- \\x1b[ - Hex escape sequences
- \\[<digits>m - General ANSI codes
"""

import re
import subprocess
import sys
from collections import defaultdict
from typing import List, Tuple, Dict

# ANSI escape sequence patterns
ANSI_PATTERNS = [
    (r'\[38;2;\d+;\d+;\d+m', '24-bit true color'),
    (r'\[0m', 'Reset code'),
    (r'\033\[', 'Octal escape sequence'),
    (r'\\x1b\[', 'Hex escape sequence'),
    (r'\[[\d;]+m', 'General ANSI code'),
]


def get_all_commits() -> List[str]:
    """
    Get list of all commit hashes in the repository.

    Returns:
        List of commit SHA hashes
    """
    try:
        result = subprocess.run(
            ['git', 'log', '--format=%H', '--color=never'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Error getting git commits: {e}", file=sys.stderr)
        sys.exit(1)


def get_commit_message(commit_hash: str) -> str:
    """
    Get the full commit message for a specific commit.

    Args:
        commit_hash: Git commit SHA hash

    Returns:
        Full commit message (subject + body)
    """
    try:
        result = subprocess.run(
            ['git', 'log', '--format=%s%n%b', '-n', '1', commit_hash, '--color=never'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error getting commit message for {commit_hash}: {e}", file=sys.stderr)
        return ""


def detect_ansi_in_text(text: str) -> List[Tuple[str, str, str]]:
    """
    Detect ANSI escape sequences in text.

    Args:
        text: Text to scan for ANSI codes

    Returns:
        List of tuples: (pattern, description, matched_text)
    """
    detections = []
    for pattern, description in ANSI_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            detections.append((pattern, description, match.group(0)))
    return detections


def scan_commits() -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Scan all commits for ANSI escape sequences.

    Returns:
        Dictionary mapping commit hash to list of detected ANSI codes
    """
    print("Scanning git commit history for ANSI escape sequences...")
    print()

    commits = get_all_commits()
    total_commits = len(commits)
    affected_commits = {}

    for i, commit_hash in enumerate(commits, 1):
        if i % 100 == 0:
            print(f"Progress: {i}/{total_commits} commits scanned...", file=sys.stderr)

        message = get_commit_message(commit_hash)
        detections = detect_ansi_in_text(message)

        if detections:
            affected_commits[commit_hash] = detections

    print(f"\nCompleted scanning {total_commits} commits.", file=sys.stderr)
    print()

    return affected_commits


def print_results(affected_commits: Dict[str, List[Tuple[str, str, str]]]):
    """
    Print scan results in a readable format.

    Args:
        affected_commits: Dictionary of commits with ANSI codes
    """
    if not affected_commits:
        print("✓ No ANSI escape sequences found in commit history!")
        return

    print(f"Found {len(affected_commits)} commits with ANSI escape sequences:\n")
    print("=" * 80)

    # Group by pattern type for summary
    pattern_counts = defaultdict(int)

    for commit_hash, detections in affected_commits.items():
        # Get commit subject for context
        subject = subprocess.run(
            ['git', 'log', '--format=%s', '-n', '1', commit_hash, '--color=never'],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

        print(f"\nCommit: {commit_hash[:12]}")
        print(f"Subject: {subject[:70]}...")
        print("Detected ANSI codes:")

        # Track unique patterns in this commit
        seen_in_commit = set()

        for pattern, description, matched_text in detections:
            if (description, matched_text) not in seen_in_commit:
                seen_in_commit.add((description, matched_text))
                pattern_counts[description] += 1
                print(f"  - {description}: {matched_text}")

    print("\n" + "=" * 80)
    print("\nSummary by pattern type:")
    for pattern_type, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {pattern_type}: {count} occurrences")

    print(f"\nTotal affected commits: {len(affected_commits)}")


def save_commit_list(affected_commits: Dict[str, List[Tuple[str, str, str]]], output_file: str = "affected_commits.txt"):
    """
    Save list of affected commit hashes to a file.

    Args:
        affected_commits: Dictionary of commits with ANSI codes
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        f.write("# Commits containing ANSI escape sequences\n")
        f.write(f"# Total: {len(affected_commits)}\n")
        f.write("# Format: commit_hash (short) - subject\n\n")

        for commit_hash in affected_commits.keys():
            subject = subprocess.run(
                ['git', 'log', '--format=%s', '-n', '1', commit_hash, '--color=never'],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()

            f.write(f"{commit_hash[:12]} - {subject}\n")

    print(f"\nAffected commit list saved to: {output_file}")


def main():
    """Main entry point."""
    print("ANSI Escape Sequence Detector for Git History")
    print("=" * 80)
    print()

    # Scan commits
    affected_commits = scan_commits()

    # Print results
    print_results(affected_commits)

    # Save commit list
    if affected_commits:
        save_commit_list(affected_commits)

    # Exit with appropriate code
    sys.exit(0 if not affected_commits else 1)


if __name__ == '__main__':
    main()
