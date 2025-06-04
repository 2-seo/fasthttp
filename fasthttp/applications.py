"""
FastHTTP - A fast and elegant HTTP client library with decorator-based request handling.

This module provides the main FastHTTP client class built on top of aiohttp,
offering decorator-based request handling for clean and intuitive API design.
"""

import aiohttp
import logging
from typing import Any, Callable, Dict, Optional, Union, TypeVar, Awaitable
from functools import wraps

from fasthttp.lifecycle import register_instance
from fasthttp.serializers import to_dict

# Type variables for better type hints
# AsyncCallable represents async function types with preserved signatures
AsyncCallable = TypeVar('AsyncCallable', bound=Callable[..., Awaitable[Any]])


class FastHTTP:
    """
    Async HTTP client with decorator-based request handling.
    
    Built on aiohttp, providing clean decorator syntax for HTTP requests
    with automatic session management and object serialization.
    
    Args:
        base_url: Base URL for all requests
        timeout: Request timeout (seconds or ClientTimeout)  
        headers: Default headers
        connector: Custom aiohttp connector
        auth: Basic authentication
        cookies: Default cookies
        debug: Enable debug logging
        auto_cleanup: Auto cleanup resources on exit
        
    Example:
        >>> http = FastHTTP(base_url="https://api.example.com")
        >>> 
        >>> @http.get("/users/{user_id}")
        >>> async def get_user(response, user_id: int):
        >>>     return await response.json()
        >>> 
        >>> user = await get_user(user_id=123)
    """
    
    def __init__(
        self, 
        base_url: Optional[str] = None,
        timeout: Optional[Union[int, float, aiohttp.ClientTimeout]] = None,
        headers: Optional[Dict[str, str]] = None,
        connector: Optional[aiohttp.BaseConnector] = None,
        auth: Optional[aiohttp.BasicAuth] = None,
        cookies: Optional[Dict[str, str]] = None,
        debug: bool = False,
        auto_cleanup: bool = True
    ):
        """
        Initialize FastHTTP client.
        
        Args:
            base_url: Base URL for all requests
            timeout: Request timeout (seconds or ClientTimeout object)
            headers: Default headers for all requests
            connector: Custom aiohttp connector for advanced configuration
            auth: Basic authentication credentials
            cookies: Default cookies for all requests
            debug: Enable debug logging
            auto_cleanup: Automatically cleanup resources on exit
        """
        self.base_url = base_url.rstrip('/') if base_url else None
        self.default_headers = headers or {}
        self.default_cookies = cookies or {}
        self.auth = auth
        self.debug = debug
        self._closed = False
        
        # Handle timeout parameter - convert int/float to ClientTimeout
        if isinstance(timeout, (int, float)):
            self.timeout = aiohttp.ClientTimeout(total=timeout)
        else:
            self.timeout = timeout or aiohttp.ClientTimeout(total=30)
        
        # Store connector configuration but don't create it yet
        # This avoids "no running event loop" error when creating FastHTTP at module level
        self._custom_connector = connector
        self._connector: Optional[aiohttp.BaseConnector] = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        
        # Session will be created lazily
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Register for automatic cleanup if enabled
        if auto_cleanup:
            register_instance(self)
    
    @property
    def connector(self) -> aiohttp.BaseConnector:
        """Lazy-create connector when first accessed."""
        if self._connector is None:
            if self._custom_connector is not None:
                self._connector = self._custom_connector
            else:
                # Create default connector only when needed (inside event loop)
                self._connector = aiohttp.TCPConnector(
                    limit=100,  # Total connection pool size
                    limit_per_host=30,  # Per-host connection limit
                    ttl_dns_cache=300,  # DNS cache TTL in seconds
                    use_dns_cache=True,
                )
        return self._connector
    
    async def __aenter__(self) -> 'FastHTTP':
        """Async context manager entry."""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create session with lazy initialization."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=self.connector,  # This will create connector if needed
                timeout=self.timeout,
                headers=self.default_headers,
                cookies=self.default_cookies,
                auth=self.auth
            )
        return self._session
    
    async def _cleanup_sync(self) -> None:
        """Internal cleanup method (prevents duplicate calls)."""
        if self._closed:
            return
            
        self._closed = True
        
        if self._session and not self._session.closed:
            await self._session.close()
        if self._connector and not self._connector.closed:
            await self._connector.close()
    
    async def close(self) -> None:
        """Close the session and cleanup resources."""
        await self._cleanup_sync()
        
        # Disable finalizer if set (already cleaned up manually)
        if hasattr(self, '_finalizer'):
            self._finalizer.detach()
    
    def _build_url(self, url: str, **kwargs) -> str:
        """Build complete URL from base_url and format with kwargs."""
        if self.base_url and not url.startswith(('http://', 'https://')):
            url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
        return url.format(**kwargs)
    
    def _make_request(self, method: str, **decorator_kwargs) -> Callable[[AsyncCallable], AsyncCallable]:
        """
        Common request logic for all HTTP methods.
        
        Returns a decorator that wraps async functions to handle HTTP requests.
        The decorator preserves the original function's type signature.
        """
        def decorator(func: AsyncCallable) -> AsyncCallable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                session = await self._get_session()
                
                try:
                    # Build URL
                    url = self._build_url(decorator_kwargs['url'], **kwargs)
                    
                    if self.debug:
                        self.logger.debug(f"Making {method} request to: {url}")
                        self.logger.debug(f"Request kwargs: {kwargs}")
                    
                    # Prepare request parameters
                    request_kwargs = {
                        'params': kwargs.get('params', {}),
                        'data': to_dict(kwargs.get('data')),
                        'json': to_dict(kwargs.get('json')),
                        'headers': {**self.default_headers, **kwargs.get('headers', {})},
                        'cookies': kwargs.get('cookies'),
                        'auth': kwargs.get('auth', self.auth),
                        'timeout': kwargs.get('timeout', self.timeout),
                        'ssl': kwargs.get('ssl'),
                        'proxy': kwargs.get('proxy'),
                        'allow_redirects': kwargs.get('allow_redirects', True),
                    }
                    
                    # Remove None values
                    request_kwargs = {k: v for k, v in request_kwargs.items() if v is not None}
                    
                    async with session.request(method, url, **request_kwargs) as response:
                        if self.debug:
                            self.logger.debug(f"Response status: {response.status}")
                            self.logger.debug(f"Response headers: {dict(response.headers)}")
                        
                        # Auto-raise for HTTP errors if requested
                        if kwargs.get('raise_for_status', False):
                            response.raise_for_status()
                        
                        return await func(response, **kwargs)
                        
                except aiohttp.ClientError as e:
                    if self.debug:
                        self.logger.error(f"Request failed: {e}")
                    raise
                    
            return wrapper
        return decorator

    def get(self, url: str, **kwargs) -> Callable[[AsyncCallable], AsyncCallable]:
        """GET request decorator"""
        return self._make_request('GET', url=url, **kwargs)
    
    def post(self, url: str, **kwargs) -> Callable[[AsyncCallable], AsyncCallable]:
        """POST request decorator"""
        return self._make_request('POST', url=url, **kwargs)
    
    def put(self, url: str, **kwargs) -> Callable[[AsyncCallable], AsyncCallable]:
        """PUT request decorator"""
        return self._make_request('PUT', url=url, **kwargs)
    
    def patch(self, url: str, **kwargs) -> Callable[[AsyncCallable], AsyncCallable]:
        """PATCH request decorator"""
        return self._make_request('PATCH', url=url, **kwargs)
    
    def delete(self, url: str, **kwargs) -> Callable[[AsyncCallable], AsyncCallable]:
        """DELETE request decorator"""
        return self._make_request('DELETE', url=url, **kwargs)
    
    def head(self, url: str, **kwargs) -> Callable[[AsyncCallable], AsyncCallable]:
        """HEAD request decorator"""
        return self._make_request('HEAD', url=url, **kwargs)
    
    def options(self, url: str, **kwargs) -> Callable[[AsyncCallable], AsyncCallable]:
        """OPTIONS request decorator"""
        return self._make_request('OPTIONS', url=url, **kwargs)
