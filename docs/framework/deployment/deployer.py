"""Documentation deployer for GitHub Pages and other hosting platforms."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


class DocumentationDeployer:
    """Deployer for documentation to various hosting platforms."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the deployer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.deployment_config = config.get('deployment', {})

    def deploy_to_github_pages(self, docs_dir: Path,
                              branch: str = 'gh-pages') -> Dict[str, Any]:
        """Deploy documentation to GitHub Pages.

        Args:
            docs_dir: Directory containing built documentation
            branch: Git branch to deploy to

        Returns:
            Deployment result dictionary
        """
        result = {
            'success': False,
            'message': '',
            'url': None
        }

        try:
            # Check if in a git repository
            if not self._is_git_repo():
                result['message'] = "Not in a git repository"
                return result

            # Create or switch to deployment branch
            self._setup_deployment_branch(branch)

            # Copy documentation files
            self._copy_docs_to_branch(docs_dir)

            # Commit and push
            self._commit_and_push(branch)

            result['success'] = True
            result['message'] = f"Successfully deployed to {branch} branch"
            result['url'] = self._get_github_pages_url()

        except Exception as e:
            result['message'] = f"Deployment failed: {e}"

        return result

    def _is_git_repo(self) -> bool:
        """Check if current directory is a git repository."""
        try:
            subprocess.run(['git', 'status'], capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _setup_deployment_branch(self, branch: str):
        """Set up the deployment branch."""
        # Check if branch exists
        try:
            subprocess.run(['git', 'checkout', branch],
                          capture_output=True, check=True)
        except subprocess.CalledProcessError:
            # Create new orphan branch
            subprocess.run(['git', 'checkout', '--orphan', branch],
                          capture_output=True, check=True)
            subprocess.run(['git', 'rm', '-rf', '.'],
                          capture_output=True, check=True)

    def _copy_docs_to_branch(self, docs_dir: Path):
        """Copy documentation files to the deployment branch."""
        # Clear existing files (except .git)
        for item in Path('.').iterdir():
            if item.name != '.git':
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        # Copy documentation files
        for item in docs_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, item.name)
            else:
                shutil.copy2(item, item.name)

    def _commit_and_push(self, branch: str):
        """Commit and push changes."""
        subprocess.run(['git', 'add', '.'], check=True)

        try:
            subprocess.run(['git', 'commit', '-m', 'Update documentation'],
                          capture_output=True, check=True)
        except subprocess.CalledProcessError:
            # No changes to commit
            pass

        subprocess.run(['git', 'push', 'origin', branch], check=True)

    def _get_github_pages_url(self) -> Optional[str]:
        """Get the GitHub Pages URL for the repository."""
        try:
            # Get remote origin URL
            result = subprocess.run(['git', 'remote', 'get-url', 'origin'],
                                  capture_output=True, text=True, check=True)
            remote_url = result.stdout.strip()

            # Parse GitHub URL
            if 'github.com' in remote_url:
                if remote_url.startswith('https://'):
                    # https://github.com/user/repo.git -> user/repo
                    parts = remote_url.replace('https://github.com/', '').replace('.git', '').split('/')
                elif remote_url.startswith('git@'):
                    # git@github.com:user/repo.git -> user/repo
                    parts = remote_url.split(':')[1].replace('.git', '').split('/')
                else:
                    return None

                if len(parts) >= 2:
                    user, repo = parts[0], parts[1]
                    return f"https://{user}.github.io/{repo}/"

        except Exception:
            pass

        return None