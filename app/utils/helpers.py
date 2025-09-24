"""
Utility functions and helpers
"""
import hashlib
import secrets
import string
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 (in production, use bcrypt)"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(password) == hashed_password


def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA-256 hash of file content"""
    return hashlib.sha256(content).hexdigest()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def extract_file_extension(filename: str) -> str:
    """Extract file extension from filename"""
    return Path(filename).suffix.lower()


def is_valid_file_type(filename: str, allowed_extensions: List[str]) -> bool:
    """Check if file type is allowed"""
    extension = extract_file_extension(filename)
    return extension in [ext.lower() for ext in allowed_extensions]


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace problematic characters
    safe_chars = set(string.ascii_letters + string.digits + ".-_")
    sanitized = ''.join(c if c in safe_chars else '_' for c in filename)
    
    # Ensure it doesn't start with a dot or dash
    if sanitized.startswith(('.', '-')):
        sanitized = 'file_' + sanitized
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        max_name_length = 255 - len(ext)
        sanitized = name[:max_name_length] + ext
    
    return sanitized or "unnamed_file"


def parse_sources_from_response(rag_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse and format sources from RAG response"""
    sources = rag_response.get("sources", [])
    if not sources:
        return []
    
    formatted_sources = []
    for source in sources:
        if isinstance(source, dict):
            formatted_sources.append({
                "content": source.get("page_content", ""),
                "metadata": source.get("metadata", {}),
                "score": source.get("score", 0.0)
            })
        elif hasattr(source, 'page_content'):
            # LangChain document object
            formatted_sources.append({
                "content": source.page_content,
                "metadata": source.metadata,
                "score": getattr(source, 'score', 0.0)
            })
    
    return formatted_sources


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (4 chars ≈ 1 token)"""
    return len(text) // 4


def format_response_time(seconds: float) -> str:
    """Format response time in human readable format"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


async def async_retry(
    func,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Async retry decorator with exponential backoff"""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except exceptions as e:
            last_exception = e
            if attempt == max_retries:
                break
            
            wait_time = delay * (backoff_factor ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
    
    raise last_exception


def validate_email(email: str) -> bool:
    """Basic email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength"""
    issues = []
    
    if len(password) < 8:
        issues.append("Password must be at least 8 characters long")
    
    if not any(c.isupper() for c in password):
        issues.append("Password must contain at least one uppercase letter")
    
    if not any(c.islower() for c in password):
        issues.append("Password must contain at least one lowercase letter")
    
    if not any(c.isdigit() for c in password):
        issues.append("Password must contain at least one digit")
    
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        issues.append("Password must contain at least one special character")
    
    return {
        "is_strong": len(issues) == 0,
        "issues": issues,
        "score": max(0, 5 - len(issues))  # Score out of 5
    }


def create_pagination_info(total: int, skip: int, limit: int) -> Dict[str, Any]:
    """Create pagination information"""
    total_pages = (total + limit - 1) // limit  # Ceiling division
    current_page = (skip // limit) + 1
    
    return {
        "total": total,
        "page": current_page,
        "per_page": limit,
        "total_pages": total_pages,
        "has_next": current_page < total_pages,
        "has_previous": current_page > 1
    }


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.utcnow()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"{self.description} took {format_response_time(duration)}")
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


def safe_json_loads(json_str: str, default=None):
    """Safely load JSON string with fallback"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj, default=None) -> str:
    """Safely dump object to JSON string"""
    try:
        return json.dumps(obj, default=str)  # Use str for datetime serialization
    except (TypeError, ValueError):
        return json.dumps(default) if default is not None else "{}"


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Merge two dictionaries, with dict2 values taking precedence"""
    merged = dict1.copy()
    merged.update(dict2)
    return merged


def get_client_ip(request) -> str:
    """Extract client IP from request headers"""
    # Check for common proxy headers
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to client host
    return request.client.host if request.client else "unknown"
