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
        submodules = self.get_detailed_submodules(path)
        return [sm["project_name"] for sm in submodules if sm["project_name"]]
    
    def get_detailed_submodules(self, path: str = ".") -> List[dict]:
        """
        Get detailed information about submodules.
        
        Args:
            path: Path to analyze
            
        Returns:
            List of submodule information dictionaries
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
                    submodule_info = self._analyze_submodule(submodule, git_root)
                    if submodule_info:
                        submodules.append(submodule_info)
                        
                except Exception as e:
                    logger.warning("Failed to process submodule %s: %s", submodule.name, e)
                    continue
            
            # Sort by project name
            submodules.sort(key=lambda x: x.get("project_name", ""))
            
            return submodules
            
        except Exception as e:
            logger.warning("Failed to get submodules from %s: %s", path, e)
            return []
    
    def _analyze_submodule(self, submodule, git_root: str) -> Optional[dict]:
        """Analyze a single submodule and extract information."""
        try:
            submodule_url = submodule.url
            submodule_path = os.path.join(git_root, submodule.path)
            
            # Parse URL information
            url_info = self._parse_git_url(submodule_url)
            
            # Check if user filtering is required
            user_owned = False
            if self.github_user:
                user_owned = self._belongs_to_user(submodule_url)
                # Skip if user filtering is enabled but this doesn't belong to user
                if not user_owned:
                    return None
            
            # Extract project name
            project_name = self._extract_repo_name_from_remote(submodule_url)
            
            # Check if submodule is initialized
            is_initialized = os.path.exists(submodule_path) and bool(os.listdir(submodule_path))
            
            # Try to get commit info
            commit_sha = None
            try:
                commit_sha = submodule.hexsha
            except Exception:
                pass
            
            return {
                "name": submodule.name,
                "path": submodule.path,
                "url": submodule_url,
                "project_name": project_name,
                "is_initialized": is_initialized,
                "user_owned": user_owned,
                "commit_sha": commit_sha,
                "url_info": url_info,
                "local_path": submodule_path
            }
            
        except Exception as e:
            logger.error("Failed to analyze submodule %s: %s", submodule.name, e)
            return None
    
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
    
    def _parse_git_url(self, remote_url: str) -> dict:
        """
        Parse a Git remote URL and extract components.
        
        Args:
            remote_url: Git remote URL
            
        Returns:
            Dictionary with URL components
        """
        url_info = {
            "original": remote_url,
            "hostname": None,
            "username": None,
            "repository": None,
            "protocol": None,
            "is_github": False,
            "is_ssh": False
        }
        
        if not remote_url:
            return url_info
            
        try:
            # SSH format: git@github.com:user/repo.git
            if remote_url.startswith("git@"):
                url_info["is_ssh"] = True
                url_info["protocol"] = "ssh"
                
                # Parse SSH format
                ssh_match = re.match(r"git@([^:]+):([^/]+)/(.+?)(?:\.git)?$", remote_url)
                if ssh_match:
                    url_info["hostname"] = ssh_match.group(1)
                    url_info["username"] = ssh_match.group(2)
                    url_info["repository"] = ssh_match.group(3)
                    
            # HTTPS/HTTP format: https://github.com/user/repo.git
            elif remote_url.startswith(("http://", "https://")):
                parsed = urlparse(remote_url)
                url_info["protocol"] = parsed.scheme
                url_info["hostname"] = parsed.hostname
                
                if parsed.path:
                    path_parts = parsed.path.strip("/").split("/")
                    if len(path_parts) >= 2:
                        url_info["username"] = path_parts[0]
                        repo_name = path_parts[1]
                        if repo_name.endswith(".git"):
                            repo_name = repo_name[:-4]
                        url_info["repository"] = repo_name
            
            # Check if it's GitHub
            if url_info["hostname"] == "github.com":
                url_info["is_github"] = True
                
        except Exception as e:
            logger.warning("Failed to parse Git URL %s: %s", remote_url, e)
            
        return url_info
    
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
            url_info = self._parse_git_url(remote_url)
            return (url_info["is_github"] and 
                   url_info["username"] == self.github_user)
            
        except Exception as e:
            logger.warning("Failed to check user ownership for URL %s: %s", remote_url, e)
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
            url_info = self._parse_git_url(remote_url)
            return url_info.get("repository")
            
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
            detailed_submodules = self.get_detailed_submodules(path)
            
            # Parse main project URL info
            main_url_info = self._parse_git_url(remote_url) if remote_url else {}
            
            return {
                "main_project": main_project,
                "subprojects": subprojects,
                "git_root": git_root,
                "remote_url": remote_url,
                "main_url_info": main_url_info,
                "github_user": self.github_user,
                "path": os.path.abspath(path),
                "is_git_repo": git_root is not None,
                "belongs_to_user": self._belongs_to_user(remote_url) if remote_url else False,
                "detailed_submodules": detailed_submodules,
                "submodule_count": len(detailed_submodules),
                "user_owned_submodules": [sm for sm in detailed_submodules if sm.get("user_owned", False)],
            }
            
        except Exception as e:
            logger.error("Failed to get project info from %s: %s", path, e)
            return {
                "main_project": os.path.basename(os.path.abspath(path)),
                "subprojects": [],
                "git_root": None,
                "remote_url": None,
                "main_url_info": {},
                "github_user": self.github_user,
                "path": os.path.abspath(path),
                "is_git_repo": False,
                "belongs_to_user": False,
                "detailed_submodules": [],
                "submodule_count": 0,
                "user_owned_submodules": [],
                "error": str(e)
            }