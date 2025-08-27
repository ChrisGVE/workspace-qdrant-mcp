"""
Project detection logic with Git and GitHub integration.

Detects project names and subprojects based on Git repositories and GitHub user ownership.
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import git
from git.exc import GitError, InvalidGitRepositoryError

logger = logging.getLogger(__name__)


class ProjectDetector:
    """
    Detects project information from Git repositories.
    
    Handles project name resolution, submodule detection, and GitHub user filtering.
    """
    
    def __init__(self, github_user: Optional[str] = None):
        self.github_user = github_user
    
    def get_project_name(self, path: str = ".") -> str:
        """
        Get project name following the PRD algorithm.
        
        Args:
            path: Path to analyze (defaults to current directory)
            
        Returns:
            Project name string
        """
        try:
            git_root = self._find_git_root(path)
            if not git_root:
                return os.path.basename(os.path.abspath(path))
            
            remote_url = self._get_git_remote_url(git_root)
            if self.github_user and remote_url and self._belongs_to_user(remote_url):
                return self._extract_repo_name_from_remote(remote_url)
            else:
                return os.path.basename(git_root)
                
        except Exception as e:
            logger.warning("Failed to detect project name from %s: %s", path, e)
            return os.path.basename(os.path.abspath(path))
    
    def get_project_and_subprojects(self, path: str = ".") -> Tuple[str, List[str]]:
        """
        Get main project name and filtered subprojects.
        
        Args:
            path: Path to analyze
            
        Returns:
            Tuple of (main_project_name, list_of_subproject_names)
        """
        main_project = self.get_project_name(path)
        subprojects = self.get_subprojects(path)
        
        return main_project, subprojects
    
    def get_subprojects(self, path: str = ".") -> List[str]:
        """
        Get list of subprojects (Git submodules filtered by GitHub user).
        
        Args:
            path: Path to analyze
            
        Returns:
            List of subproject names
        """
        try:
            git_root = self._find_git_root(path)
            if not git_root:
                return []
            
            repo = git.Repo(git_root)
            submodules = []
            
            # Get all submodules
            for submodule in repo.submodules:
                try:
                    submodule_url = submodule.url
                    
                    # Filter by GitHub user if specified
                    if self.github_user:
                        if not self._belongs_to_user(submodule_url):
                            continue
                    
                    # Extract project name from submodule
                    project_name = self._extract_repo_name_from_remote(submodule_url)
                    if project_name:
                        submodules.append(project_name)
                        
                except Exception as e:
                    logger.warning("Failed to process submodule %s: %s", submodule.name, e)
                    continue
            
            return sorted(list(set(submodules)))  # Remove duplicates and sort
            
        except Exception as e:
            logger.warning("Failed to get subprojects from %s: %s", path, e)
            return []
    
    def _find_git_root(self, path: str) -> Optional[str]:
        """
        Find the root directory of a Git repository.
        
        Args:
            path: Starting path
            
        Returns:
            Git root directory path or None
        """
        try:
            repo = git.Repo(path, search_parent_directories=True)
            return repo.working_dir
        except (InvalidGitRepositoryError, GitError):
            return None
    
    def _get_git_remote_url(self, git_root: str) -> Optional[str]:
        """
        Get the remote URL for the Git repository.
        
        Args:
            git_root: Git repository root directory
            
        Returns:
            Remote URL string or None
        """
        try:
            repo = git.Repo(git_root)
            
            # Try origin first, then any remote
            for remote_name in ["origin", "upstream"]:
                if hasattr(repo.remotes, remote_name):
                    remote = getattr(repo.remotes, remote_name)
                    return remote.url
            
            # Fall back to first available remote
            if repo.remotes:
                return repo.remotes[0].url
                
            return None
            
        except Exception as e:
            logger.warning("Failed to get remote URL from %s: %s", git_root, e)
            return None
    
    def _belongs_to_user(self, remote_url: str) -> bool:
        """
        Check if a remote URL belongs to the configured GitHub user.
        
        Args:
            remote_url: Git remote URL
            
        Returns:
            True if URL belongs to the user
        """
        if not self.github_user or not remote_url:
            return False
            
        try:
            # Handle different URL formats
            if remote_url.startswith("git@github.com:"):
                # SSH format: git@github.com:user/repo.git
                match = re.match(r"git@github\.com:([^/]+)/", remote_url)
                if match:
                    return match.group(1) == self.github_user
                    
            elif "github.com" in remote_url:
                # HTTPS format: https://github.com/user/repo.git
                parsed = urlparse(remote_url)
                if parsed.hostname == "github.com" and parsed.path:
                    path_parts = parsed.path.strip("/").split("/")
                    if len(path_parts) >= 1:
                        return path_parts[0] == self.github_user
            
            return False
            
        except Exception as e:
            logger.warning("Failed to parse remote URL %s: %s", remote_url, e)
            return False
    
    def _extract_repo_name_from_remote(self, remote_url: str) -> Optional[str]:
        """
        Extract repository name from remote URL.
        
        Args:
            remote_url: Git remote URL
            
        Returns:
            Repository name or None
        """
        if not remote_url:
            return None
            
        try:
            # Handle SSH format: git@github.com:user/repo.git
            if remote_url.startswith("git@github.com:"):
                match = re.match(r"git@github\.com:[^/]+/([^/]+)(?:\.git)?$", remote_url)
                if match:
                    return match.group(1)
            
            # Handle HTTPS format: https://github.com/user/repo.git
            elif "github.com" in remote_url:
                parsed = urlparse(remote_url)
                if parsed.path:
                    path_parts = parsed.path.strip("/").split("/")
                    if len(path_parts) >= 2:
                        repo_name = path_parts[1]
                        # Remove .git suffix if present
                        if repo_name.endswith(".git"):
                            repo_name = repo_name[:-4]
                        return repo_name
            
            # Fallback: try to extract from any URL
            match = re.search(r"/([^/]+?)(?:\.git)?/?$", remote_url)
            if match:
                return match.group(1)
                
            return None
            
        except Exception as e:
            logger.warning("Failed to extract repo name from %s: %s", remote_url, e)
            return None
    
    def get_project_info(self, path: str = ".") -> dict:
        """
        Get comprehensive project information.
        
        Args:
            path: Path to analyze
            
        Returns:
            Dictionary with project information
        """
        try:
            main_project, subprojects = self.get_project_and_subprojects(path)
            git_root = self._find_git_root(path)
            remote_url = self._get_git_remote_url(git_root) if git_root else None
            
            return {
                "main_project": main_project,
                "subprojects": subprojects,
                "git_root": git_root,
                "remote_url": remote_url,
                "github_user": self.github_user,
                "path": os.path.abspath(path),
                "is_git_repo": git_root is not None,
                "belongs_to_user": self._belongs_to_user(remote_url) if remote_url else False,
            }
            
        except Exception as e:
            logger.error("Failed to get project info from %s: %s", path, e)
            return {
                "main_project": os.path.basename(os.path.abspath(path)),
                "subprojects": [],
                "git_root": None,
                "remote_url": None,
                "github_user": self.github_user,
                "path": os.path.abspath(path),
                "is_git_repo": False,
                "belongs_to_user": False,
                "error": str(e)
            }