import aiohttp
import asyncio
import logging
import atexit
import weakref
from typing import Any, Callable, Dict, Optional, Union, TypeVar, Awaitable, Set
from functools import wraps

# Type variables for better type hints
# AsyncCallable은 비동기 함수 타입을 나타내는 제네릭 타입 변수입니다.
# bound=Callable[..., Awaitable[Any]]는 AsyncCallable이 async 함수 타입으로만 제한됨을 의미합니다.
# 이를 통해 데코레이터 적용 후에도 원본 함수의 타입 정보가 보존됩니다.
AsyncCallable = TypeVar('AsyncCallable', bound=Callable[..., Awaitable[Any]])

# 전역 FastHTTP 인스턴스들을 추적하여 프로세스 종료 시 정리
_active_instances: Set[weakref.ReferenceType] = set()
_cleanup_registered = False

def _cleanup_all_instances():
    """프로세스 종료 시 모든 활성 FastHTTP 인스턴스를 정리"""
    for instance_ref in list(_active_instances):
        instance = instance_ref()
        if instance is not None:
            try:
                # 이벤트 루프가 실행 중인지 확인
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 루프가 실행 중이면 태스크로 추가
                    loop.create_task(instance._cleanup_sync())
                else:
                    # 루프가 없거나 중지되었으면 새 루프에서 실행
                    asyncio.run(instance._cleanup_sync())
            except:
                # 에러가 발생해도 다른 인스턴스들은 계속 정리
                pass

def _register_global_cleanup():
    """전역 cleanup 핸들러를 한 번만 등록"""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(_cleanup_all_instances)
        _cleanup_registered = True

class FastHTTP:
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
        Initialize FastHTTP client with optional configuration.
        
        Args:
            base_url: Base URL for all requests
            timeout: Default timeout for requests (seconds or ClientTimeout object)
            headers: Default headers for all requests
            connector: Custom connector for connection pooling
            auth: Basic authentication
            cookies: Default cookies
            debug: Enable debug logging
            auto_cleanup: Enable automatic resource cleanup on process exit (default: True)
        """
        self.base_url = base_url
        self.default_headers = headers or {}
        self.default_cookies = cookies or {}
        self.auth = auth
        self.debug = debug
        self._closed = False
        
        # Setup timeout
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
        
        # 자동 cleanup 설정
        if auto_cleanup:
            # 전역 cleanup 핸들러 등록
            _register_global_cleanup()
            
            # 이 인스턴스를 전역 추적 목록에 추가
            _active_instances.add(weakref.ref(self, self._remove_from_tracking))
            
            # weakref.finalize를 사용하여 객체가 GC될 때도 정리
            self._finalizer = weakref.finalize(
                self, 
                self._cleanup_finalizer, 
                weakref.ref(self._get_session), 
                weakref.ref(lambda: self._connector)
            )
    
    @staticmethod
    def _remove_from_tracking(instance_ref):
        """인스턴스가 GC될 때 추적 목록에서 제거"""
        _active_instances.discard(instance_ref)
    
    @staticmethod
    def _cleanup_finalizer(session_ref, connector_ref):
        """finalizer에서 사용할 cleanup 함수 (동기적)"""
        try:
            session_func = session_ref()
            if session_func:
                session = session_func()
                if session and not session.closed:
                    try:
                        # 동기적으로 세션 종료 시도
                        session._connector.close()
                    except:
                        pass
                        
            connector_func = connector_ref()
            if connector_func:
                connector = connector_func()
                if connector and not connector.closed:
                    try:
                        connector.close()
                    except:
                        pass
        except:
            # finalizer에서는 예외를 무시
            pass
    
    @property
    def connector(self) -> aiohttp.BaseConnector:
        """Lazy-create connector when first accessed"""
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
        """Async context manager entry"""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create session with lazy initialization"""
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
        """내부 cleanup 메서드 (중복 호출 방지)"""
        if self._closed:
            return
            
        self._closed = True
        
        if self._session and not self._session.closed:
            await self._session.close()
        if self._connector and not self._connector.closed:
            await self._connector.close()
    
    async def close(self) -> None:
        """Close the session and cleanup resources"""
        await self._cleanup_sync()
        
        # finalizer가 설정되어 있으면 비활성화 (이미 수동으로 정리했으므로)
        if hasattr(self, '_finalizer'):
            self._finalizer.detach()
    
    def _build_url(self, url: str, **kwargs) -> str:
        """Build complete URL from base_url and format with kwargs"""
        if self.base_url and not url.startswith(('http://', 'https://')):
            url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
        return url.format(**kwargs)
    
    def _make_request(self, method: str, **decorator_kwargs) -> Callable[[AsyncCallable], AsyncCallable]:
        """
        Common request logic for all HTTP methods.
        
        AsyncCallable 타입 사용 설명:
        - AsyncCallable은 데코레이터가 적용되는 비동기 함수의 타입을 나타냅니다
        - Callable[[AsyncCallable], AsyncCallable]는 "AsyncCallable 타입의 함수를 받아서 AsyncCallable 타입의 함수를 반환"한다는 의미입니다
        - 이를 통해 데코레이터 적용 후에도 원본 함수의 시그니처가 IDE에서 그대로 보입니다
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
                        'data': kwargs.get('data'),
                        'json': kwargs.get('json'),
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
