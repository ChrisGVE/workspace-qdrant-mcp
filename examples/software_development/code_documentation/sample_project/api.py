"""
Sample API module demonstrating various code documentation patterns.

This module provides RESTful API endpoints for user management and authentication,
showcasing different documentation styles and patterns for workspace-qdrant-mcp integration.
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import jwt
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr


logger = logging.getLogger(__name__)
security = HTTPBearer()


@dataclass
class UserProfile:
    """
    User profile data structure.
    
    Represents a user's profile information including authentication
    and authorization details.
    
    Attributes:
        id (int): Unique user identifier
        email (str): User's email address
        username (str): Unique username
        full_name (str): User's full name
        is_active (bool): Whether the user account is active
        roles (List[str]): List of user roles for authorization
        created_at (datetime): Account creation timestamp
        last_login (Optional[datetime]): Last successful login timestamp
    """
    id: int
    email: str
    username: str
    full_name: str
    is_active: bool = True
    roles: List[str] = None
    created_at: datetime = None
    last_login: Optional[datetime] = None


class UserCreate(BaseModel):
    """
    User creation request model.
    
    Validates and structures data for creating new user accounts.
    """
    email: EmailStr
    username: str
    full_name: str
    password: str
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "username": "johndoe",
                "full_name": "John Doe",
                "password": "securepassword123"
            }
        }


class UserResponse(BaseModel):
    """User response model for API responses (excludes sensitive data)."""
    id: int
    email: str
    username: str
    full_name: str
    is_active: bool
    roles: List[str]
    created_at: datetime
    last_login: Optional[datetime]


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class AuthService:
    """
    Authentication and authorization service.
    
    Provides secure user authentication, JWT token management, and role-based
    access control for the application.
    
    Features:
    - Password hashing with salt
    - JWT token generation and validation
    - Role-based access control
    - Session management
    - Password reset functionality
    
    Security considerations:
    - Passwords are hashed using SHA-256 with salt (in production, use bcrypt)
    - JWT tokens expire after 24 hours
    - Failed login attempts are logged for security monitoring
    """
    
    def __init__(self, secret_key: str, token_expiry_hours: int = 24):
        """
        Initialize authentication service.
        
        Args:
            secret_key (str): Secret key for JWT token signing
            token_expiry_hours (int): JWT token expiration time in hours
        """
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        self.users_db: Dict[int, UserProfile] = {}  # In production, use real database
        self.username_index: Dict[str, int] = {}
        
        # Add sample users for demonstration
        self._initialize_sample_users()
    
    def _initialize_sample_users(self):
        """Initialize sample users for demonstration purposes."""
        sample_users = [
            UserProfile(
                id=1,
                email="admin@example.com",
                username="admin",
                full_name="System Administrator",
                roles=["admin", "user"],
                created_at=datetime.now() - timedelta(days=30)
            ),
            UserProfile(
                id=2,
                email="john@example.com",
                username="johndoe",
                full_name="John Doe",
                roles=["user"],
                created_at=datetime.now() - timedelta(days=15)
            )
        ]
        
        for user in sample_users:
            self.users_db[user.id] = user
            self.username_index[user.username] = user.id
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """
        Hash password with salt for secure storage.
        
        Args:
            password (str): Plain text password to hash
            salt (str, optional): Salt for hashing. Generated if not provided.
        
        Returns:
            tuple: (hashed_password, salt) for storage
            
        Note:
            In production, use bcrypt or similar library instead of SHA-256.
        """
        if not salt:
            salt = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16]
        
        hashed = hashlib.sha256((password + salt).encode()).hexdigest()
        return hashed, salt
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """
        Verify password against stored hash.
        
        Args:
            password (str): Plain text password to verify
            hashed_password (str): Stored hashed password
            salt (str): Salt used for hashing
            
        Returns:
            bool: True if password is correct, False otherwise
        """
        test_hash, _ = self.hash_password(password, salt)
        return test_hash == hashed_password
    
    def generate_token(self, user: UserProfile) -> str:
        """
        Generate JWT token for authenticated user.
        
        Args:
            user (UserProfile): User profile for token generation
            
        Returns:
            str: JWT token string
            
        Raises:
            HTTPException: If token generation fails
        """
        try:
            payload = {
                'user_id': user.id,
                'username': user.username,
                'email': user.email,
                'roles': user.roles,
                'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
                'iat': datetime.utcnow()
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm='HS256')
            logger.info(f"Generated token for user {user.username}")
            return token
            
        except Exception as e:
            logger.error(f"Token generation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token generation failed"
            )
    
    def verify_token(self, token: str) -> UserProfile:
        """
        Verify JWT token and return user profile.
        
        Args:
            token (str): JWT token to verify
            
        Returns:
            UserProfile: User profile if token is valid
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            user_id = payload.get('user_id')
            
            if user_id not in self.users_db:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            user = self.users_db[user_id]
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User account is deactivated"
                )
            
            return user
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserProfile]:
        """
        Authenticate user credentials.
        
        Args:
            username (str): Username for authentication
            password (str): Password for authentication
            
        Returns:
            Optional[UserProfile]: User profile if authentication succeeds, None otherwise
        """
        # In production, this would check against database with hashed passwords
        user_id = self.username_index.get(username)
        if not user_id:
            logger.warning(f"Authentication failed: username {username} not found")
            return None
        
        user = self.users_db[user_id]
        if not user.is_active:
            logger.warning(f"Authentication failed: user {username} is deactivated")
            return None
        
        # In production, verify hashed password
        # For demo, we'll accept any password for existing users
        user.last_login = datetime.now()
        logger.info(f"User {username} authenticated successfully")
        return user
    
    def has_role(self, user: UserProfile, required_role: str) -> bool:
        """
        Check if user has required role.
        
        Args:
            user (UserProfile): User profile to check
            required_role (str): Required role for access
            
        Returns:
            bool: True if user has required role, False otherwise
        """
        return required_role in (user.roles or [])


class UserAPI:
    """
    User management API endpoints.
    
    Provides RESTful endpoints for user management operations including
    user creation, authentication, profile management, and role-based access control.
    """
    
    def __init__(self, auth_service: AuthService):
        """Initialize user API with authentication service."""
        self.auth = auth_service
        self.app = FastAPI(
            title="User Management API",
            description="RESTful API for user management and authentication",
            version="1.0.0"
        )
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up API routes and middleware."""
        # Authentication endpoints
        self.app.post("/auth/login")(self.login)
        self.app.post("/auth/logout")(self.logout)
        
        # User management endpoints
        self.app.post("/users/", response_model=UserResponse)(self.create_user)
        self.app.get("/users/me", response_model=UserResponse)(self.get_current_user)
        self.app.get("/users/{user_id}", response_model=UserResponse)(self.get_user)
        self.app.put("/users/{user_id}", response_model=UserResponse)(self.update_user)
        self.app.delete("/users/{user_id}")(self.delete_user)
        
        # Admin endpoints
        self.app.get("/admin/users", response_model=List[UserResponse])(self.list_all_users)
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserProfile:
        """
        Get current authenticated user from JWT token.
        
        This dependency function extracts and validates the JWT token from the
        Authorization header and returns the authenticated user profile.
        
        Args:
            credentials: HTTP Bearer token credentials
            
        Returns:
            UserProfile: Currently authenticated user
            
        Raises:
            HTTPException: If token is invalid or user not found
        """
        return self.auth.verify_token(credentials.credentials)
    
    async def login(self, login_data: LoginRequest) -> Dict[str, Any]:
        """
        Authenticate user and return JWT token.
        
        Args:
            login_data: Login credentials (username and password)
            
        Returns:
            Dict containing access token and user information
            
        Raises:
            HTTPException: If authentication fails
        """
        user = self.auth.authenticate_user(login_data.username, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        token = self.auth.generate_token(user)
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": self.auth.token_expiry_hours * 3600,  # seconds
            "user": UserResponse(**user.__dict__)
        }
    
    async def logout(self, current_user: UserProfile = Depends(get_current_user)) -> Dict[str, str]:
        """
        Logout current user.
        
        Note: With JWT tokens, logout is primarily client-side.
        In production, you might maintain a token blacklist.
        
        Args:
            current_user: Currently authenticated user
            
        Returns:
            Dict with logout confirmation message
        """
        logger.info(f"User {current_user.username} logged out")
        return {"message": "Successfully logged out"}
    
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """
        Create new user account.
        
        Args:
            user_data: User creation data
            
        Returns:
            UserResponse: Created user profile (without password)
            
        Raises:
            HTTPException: If user creation fails or username/email exists
        """
        # Check for existing username
        if user_data.username in self.auth.username_index:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Check for existing email
        existing_emails = {user.email for user in self.auth.users_db.values()}
        if user_data.email in existing_emails:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )
        
        # Create new user
        new_id = max(self.auth.users_db.keys(), default=0) + 1
        new_user = UserProfile(
            id=new_id,
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            roles=["user"],  # Default role
            created_at=datetime.now()
        )
        
        # Store user (in production, hash and store password too)
        self.auth.users_db[new_id] = new_user
        self.auth.username_index[user_data.username] = new_id
        
        logger.info(f"Created user: {user_data.username}")
        return UserResponse(**new_user.__dict__)
    
    async def get_user(self, user_id: int, current_user: UserProfile = Depends(get_current_user)) -> UserResponse:
        """
        Get user profile by ID.
        
        Users can access their own profile, admins can access any profile.
        
        Args:
            user_id: ID of user to retrieve
            current_user: Currently authenticated user
            
        Returns:
            UserResponse: User profile data
            
        Raises:
            HTTPException: If user not found or access denied
        """
        if user_id not in self.auth.users_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Check authorization: users can see their own profile, admins can see any
        if current_user.id != user_id and not self.auth.has_role(current_user, "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        user = self.auth.users_db[user_id]
        return UserResponse(**user.__dict__)
    
    async def list_all_users(self, current_user: UserProfile = Depends(get_current_user)) -> List[UserResponse]:
        """
        List all users (admin only).
        
        Args:
            current_user: Currently authenticated user
            
        Returns:
            List[UserResponse]: List of all user profiles
            
        Raises:
            HTTPException: If user doesn't have admin role
        """
        if not self.auth.has_role(current_user, "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin role required"
            )
        
        return [UserResponse(**user.__dict__) for user in self.auth.users_db.values()]


# Example of complex business logic with comprehensive documentation
async def process_user_batch_operation(
    user_ids: List[int],
    operation: str,
    auth_service: AuthService,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Process batch operations on multiple users asynchronously.
    
    This function demonstrates complex business logic that might be found in
    real-world applications. It processes operations on multiple users in
    batches to optimize performance and prevent resource exhaustion.
    
    Supported operations:
    - 'activate': Activate user accounts
    - 'deactivate': Deactivate user accounts  
    - 'reset_password': Send password reset emails
    - 'update_roles': Update user roles (requires additional parameters)
    
    Args:
        user_ids (List[int]): List of user IDs to process
        operation (str): Operation to perform on users
        auth_service (AuthService): Authentication service instance
        batch_size (int): Number of users to process per batch (default: 100)
        
    Returns:
        Dict[str, Any]: Operation results including:
            - total_processed: Number of users processed
            - successful: Number of successful operations
            - failed: Number of failed operations
            - errors: List of error messages for failed operations
            - duration: Processing time in seconds
            
    Raises:
        ValueError: If operation is not supported
        HTTPException: If batch operation fails completely
        
    Example:
        >>> results = await process_user_batch_operation(
        ...     user_ids=[1, 2, 3, 4, 5],
        ...     operation='activate',
        ...     auth_service=auth_service,
        ...     batch_size=2
        ... )
        >>> print(f"Processed {results['total_processed']} users")
        
    Performance Considerations:
        - Operations are batched to prevent memory issues with large datasets
        - Each batch is processed asynchronously for better performance
        - Failed operations don't stop processing of remaining batches
        - Detailed error reporting for troubleshooting
        
    Security Considerations:
        - Validates user existence before operations
        - Logs all batch operations for audit trail
        - Rate limiting should be implemented at the API level
    """
    start_time = datetime.now()
    
    # Validate operation
    supported_operations = ['activate', 'deactivate', 'reset_password', 'update_roles']
    if operation not in supported_operations:
        raise ValueError(f"Unsupported operation: {operation}. Supported: {supported_operations}")
    
    results = {
        'total_processed': 0,
        'successful': 0,
        'failed': 0,
        'errors': [],
        'duration': 0,
        'operation': operation,
        'batch_size': batch_size
    }
    
    # Process users in batches
    for i in range(0, len(user_ids), batch_size):
        batch = user_ids[i:i + batch_size]
        batch_results = await _process_user_batch(batch, operation, auth_service)
        
        # Aggregate results
        results['total_processed'] += len(batch)
        results['successful'] += batch_results['successful']
        results['failed'] += batch_results['failed']
        results['errors'].extend(batch_results['errors'])
        
        # Log batch completion
        logger.info(f"Completed batch {i//batch_size + 1}: {len(batch)} users processed")
    
    # Calculate duration
    end_time = datetime.now()
    results['duration'] = (end_time - start_time).total_seconds()
    
    # Log final results
    logger.info(f"Batch operation '{operation}' completed: "
               f"{results['successful']} successful, {results['failed']} failed")
    
    return results


async def _process_user_batch(user_ids: List[int], operation: str, auth_service: AuthService) -> Dict[str, Any]:
    """
    Process a single batch of users.
    
    Internal helper function for batch processing. Handles the actual
    operation execution for a subset of users.
    
    Args:
        user_ids: List of user IDs in this batch
        operation: Operation to perform
        auth_service: Authentication service instance
        
    Returns:
        Dict with batch processing results
    """
    batch_results = {'successful': 0, 'failed': 0, 'errors': []}
    
    for user_id in user_ids:
        try:
            if user_id not in auth_service.users_db:
                batch_results['errors'].append(f"User {user_id} not found")
                batch_results['failed'] += 1
                continue
            
            user = auth_service.users_db[user_id]
            
            # Execute operation
            if operation == 'activate':
                user.is_active = True
            elif operation == 'deactivate':
                user.is_active = False
            elif operation == 'reset_password':
                # In real implementation, send password reset email
                logger.info(f"Password reset email sent to {user.email}")
            elif operation == 'update_roles':
                # This would require additional parameters in real implementation
                pass
            
            batch_results['successful'] += 1
            
        except Exception as e:
            batch_results['errors'].append(f"Error processing user {user_id}: {str(e)}")
            batch_results['failed'] += 1
            logger.error(f"Batch operation failed for user {user_id}: {str(e)}")
    
    return batch_results


# Initialize the application
def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    auth_service = AuthService(secret_key="your-secret-key-here")
    user_api = UserAPI(auth_service)
    
    return user_api.app


# Global application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)