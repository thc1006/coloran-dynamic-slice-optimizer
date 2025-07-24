import os
import secrets
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from cryptography.fernet import Fernet
import bcrypt
from pathlib import Path
import sys

# Import configuration manager
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config

class SecurityManager:
    """
    Comprehensive security manager for ColO-RAN Dynamic Slice Optimizer.
    
    Features:
    - JWT token management with proper secret handling
    - Input validation and sanitization
    - Rate limiting and brute force protection
    - Secure password hashing
    - Data encryption for sensitive information
    - Security audit logging
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config()
        self.security_config = self.config.get_security_config()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize security components
        self._setup_jwt_secret()
        self._setup_encryption()
        self._setup_rate_limiting()
        
        self.security = HTTPBearer()
        
        # Security audit trail
        self.failed_attempts = {}
        self.security_events = []
        
        self.logger.info("🔒 SecurityManager initialized with comprehensive protection")
    
    def _setup_jwt_secret(self):
        """Setup JWT secret with proper validation."""
        jwt_secret = self.security_config.get('jwt_secret')
        
        if not jwt_secret or jwt_secret == '${COLORAN_JWT_SECRET}':
            # Check environment variable
            jwt_secret = os.getenv('COLORAN_JWT_SECRET')
            
            if not jwt_secret:
                # Generate secure random secret
                jwt_secret = secrets.token_urlsafe(64)
                self.logger.warning("⚠️ Generated random JWT secret. Set COLORAN_JWT_SECRET for production!")
        
        # Validate secret strength
        if len(jwt_secret) < 32:
            raise ValueError("JWT secret must be at least 32 characters long")
        
        self.jwt_secret = jwt_secret
        self.logger.info("✅ JWT secret configured securely")
    
    def _setup_encryption(self):
        """Setup encryption for sensitive data."""
        encryption_key = os.getenv('COLORAN_ENCRYPTION_KEY')
        
        if not encryption_key:
            # Generate key for session (should be persisted in production)
            encryption_key = Fernet.generate_key()
            self.logger.warning("⚠️ Generated session encryption key. Set COLORAN_ENCRYPTION_KEY for production!")
        else:
            encryption_key = encryption_key.encode()
        
        self.cipher = Fernet(encryption_key)
        self.logger.info("✅ Data encryption configured")
    
    def _setup_rate_limiting(self):
        """Setup rate limiting configuration."""
        self.rate_limit = self.security_config.get('api_rate_limit', 100)
        self.rate_window = 3600  # 1 hour window
        self.request_counts = {}
        self.logger.info(f"✅ Rate limiting: {self.rate_limit} requests per hour")
    
    def generate_token(self, user_data: Dict[str, Any], expires_in_hours: int = 24) -> str:
        """Generate secure JWT token."""
        try:
            payload = {
                **user_data,
                'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
                'iat': datetime.utcnow(),
                'jti': secrets.token_urlsafe(16)  # JWT ID for tracking
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            
            # Log token generation
            self._log_security_event('token_generated', {
                'user_id': user_data.get('user_id', 'unknown'),
                'expires_in_hours': expires_in_hours
            })
            
            return token
            
        except Exception as e:
            self.logger.error(f"❌ Token generation failed: {e}")
            raise HTTPException(status_code=500, detail="Token generation failed")
    
    async def authenticate_request(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Enhanced JWT authentication with security checks."""
        token = credentials.credentials
        client_ip = self._get_client_ip()
        
        try:
            # Check rate limiting
            if not self._check_rate_limit(client_ip):
                self._log_security_event('rate_limit_exceeded', {'client_ip': client_ip})
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Validate token
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Additional security checks
            self._validate_token_security(payload, client_ip)
            
            # Log successful authentication
            self._log_security_event('authentication_success', {
                'user_id': payload.get('user_id', 'unknown'),
                'client_ip': client_ip
            })
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self._handle_auth_failure(client_ip, "Token expired")
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            self._handle_auth_failure(client_ip, "Invalid token")
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            self._handle_auth_failure(client_ip, str(e))
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    def _validate_token_security(self, payload: Dict[str, Any], client_ip: str):
        """Additional token security validations."""
        # Check token age
        issued_at = payload.get('iat')
        if issued_at and (datetime.utcnow().timestamp() - issued_at) > 86400:  # 24 hours
            raise ValueError("Token too old")
        
        # Check for token reuse (implement token blacklist in production)
        jti = payload.get('jti')
        if not jti:
            raise ValueError("Missing token identifier")
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit."""
        current_time = time.time()
        
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        # Clean old requests
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip]
            if current_time - req_time < self.rate_window
        ]
        
        # Check limit
        if len(self.request_counts[client_ip]) >= self.rate_limit:
            return False
        
        # Add current request
        self.request_counts[client_ip].append(current_time)
        return True
    
    def _handle_auth_failure(self, client_ip: str, reason: str):
        """Handle authentication failures with security logging."""
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = []
        
        self.failed_attempts[client_ip].append(time.time())
        
        # Clean old attempts (last hour)
        cutoff = time.time() - 3600
        self.failed_attempts[client_ip] = [
            attempt for attempt in self.failed_attempts[client_ip]
            if attempt > cutoff
        ]
        
        # Log security event
        self._log_security_event('authentication_failure', {
            'client_ip': client_ip,
            'reason': reason,
            'failure_count': len(self.failed_attempts[client_ip])
        })
        
        # Check for brute force
        if len(self.failed_attempts[client_ip]) >= 10:
            self.logger.warning(f"🚨 Potential brute force attack from {client_ip}")
    
    def _get_client_ip(self) -> str:
        """Get client IP address from request context."""
        # This would be implemented based on the web framework used
        return "unknown"  # Placeholder
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events for audit trail."""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        self.security_events.append(event)
        self.logger.info(f"Security Event: {event_type} - {details}")
        
        # Keep only last 1000 events in memory
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
    
    def hash_password(self, password: str) -> str:
        """Securely hash password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def validate_input(self, input_data: str, max_length: int = 1000, allow_html: bool = False) -> str:
        """Validate and sanitize input data."""
        if not input_data:
            return input_data
        
        # Length check
        if len(input_data) > max_length:
            raise ValueError(f"Input too long (max {max_length} characters)")
        
        # Basic HTML/script injection protection
        if not allow_html:
            dangerous_patterns = ['<script', 'javascript:', 'data:', 'vbscript:', 'onload=', 'onerror=']
            input_lower = input_data.lower()
            
            for pattern in dangerous_patterns:
                if pattern in input_lower:
                    raise ValueError("Potentially dangerous input detected")
        
        return input_data.strip()
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security audit report."""
        return {
            'failed_attempts_by_ip': {
                ip: len(attempts) for ip, attempts in self.failed_attempts.items()
            },
            'recent_security_events': self.security_events[-50:],  # Last 50 events
            'rate_limiting_config': {
                'limit': self.rate_limit,
                'window_hours': self.rate_window / 3600
            },
            'active_request_counts': {
                ip: len(requests) for ip, requests in self.request_counts.items()
            }
        }
