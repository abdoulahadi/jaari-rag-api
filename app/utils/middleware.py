"""
Middleware components for FastAPI application
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from typing import Callable
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging middleware for request/response tracking"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} - "
            f"Time: {process_time:.3f}s - "
            f"Path: {request.url.path}"
        )
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Content Security Policy: allow self, CDN for Swagger UI assets, and inline scripts/styles
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
            "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
            "img-src 'self' data: https://fastapi.tiangolo.com"
        )
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        current_time = time.time()
        
        if client_ip not in self.clients:
            self.clients[client_ip] = {"calls": 0, "reset_time": current_time + self.period}
        
        client_data = self.clients[client_ip]
        
        # Reset if period expired
        if current_time > client_data["reset_time"]:
            client_data["calls"] = 0
            client_data["reset_time"] = current_time + self.period
        
        # Check if limit exceeded
        if client_data["calls"] >= self.calls:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Increment call count
        client_data["calls"] += 1
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(self.calls - client_data["calls"])
        response.headers["X-RateLimit-Reset"] = str(int(client_data["reset_time"]))
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Metrics collection middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = {}
        self.response_times = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Track request
        path = request.url.path
        method = request.method
        
        key = f"{method}:{path}"
        self.request_count[key] = self.request_count.get(key, 0) + 1
        
        # Process request
        response = await call_next(request)
        
        # Track response time
        duration = time.time() - start_time
        if key not in self.response_times:
            self.response_times[key] = []
        self.response_times[key].append(duration)
        
        # Keep only last 100 response times per endpoint
        if len(self.response_times[key]) > 100:
            self.response_times[key] = self.response_times[key][-100:]
        
        return response
    
    def get_metrics(self):
        """Get collected metrics"""
        metrics = {
            "request_counts": self.request_count.copy(),
            "avg_response_times": {}
        }
        
        for key, times in self.response_times.items():
            if times:
                metrics["avg_response_times"][key] = sum(times) / len(times)
        
        return metrics


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Unhandled error in {request.url.path}: {str(e)}", exc_info=True)
            
            from fastapi import HTTPException
            from fastapi.responses import JSONResponse
            
            # Don't handle HTTPException - let FastAPI handle it
            if isinstance(e, HTTPException):
                raise
            
            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )


class CacheMiddleware(BaseHTTPMiddleware):
    """Simple response caching middleware"""
    
    def __init__(self, app, cache_time: int = 300):
        super().__init__(app)
        self.cache_time = cache_time
        self.cache = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Generate cache key
        cache_key = f"{request.url.path}?{request.url.query}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cached_response, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_time:
                logger.debug(f"Cache hit for {cache_key}")
                # Return cached response (simplified)
                return cached_response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            self.cache[cache_key] = (response, current_time)
            
            # Clean old cache entries
            if len(self.cache) > 1000:  # Simple cleanup
                oldest_keys = sorted(
                    self.cache.keys(),
                    key=lambda k: self.cache[k][1]
                )[:100]
                for key in oldest_keys:
                    del self.cache[key]
        
        return response
