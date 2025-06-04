# FastHTTP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fast and elegant HTTP client library with decorator-based request handling, built on top of aiohttp.

FastHTTP provides a clean, intuitive interface for making HTTP requests in Python, featuring automatic resource management, connection pooling, and a unique decorator-based API that makes your code more readable and maintainable.

## ✨ Features

- **Decorator-based API**: Clean and intuitive request handling
- **Async/Await Support**: Built on aiohttp for high performance
- **Automatic Resource Management**: No more connection leaks
- **Connection Pooling**: Efficient connection reuse
- **URL Template Support**: Dynamic URL building with parameters
- **Request/Response Middleware**: Easy customization
- **Type Hints**: Full typing support for better IDE experience
- **Debug Mode**: Comprehensive logging for development

## 🚀 Quick Start

### Installation

```bash
# pip
pip install fastaiohttp

# uv
uv add fastaiohttp
```

### Basic Usage

```python
import asyncio
from fasthttp import FastHTTP

# Create a client instance
http = FastHTTP(base_url="https://jsonplaceholder.typicode.com")

@http.get("/posts/{post_id}")
async def get_post(response, post_id):
    """Fetch a single post by ID."""
    return await response.json()

@http.post("/posts")
async def create_post(response, json=None):
    """Create a new post."""
    return await response.json()

async def main():
    # Fetch a post
    post = await get_post(post_id=1)
    print(f"Post title: {post['title']}")
    
    # Create a new post
    new_post = await create_post(
        json={
            "title": "My New Post",
            "body": "This is the content of my post",
            "userId": 1
        }
    )
    print(f"Created post with ID: {new_post['id']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 📖 Documentation

### Client Configuration

```python
from fasthttp import FastHTTP
import aiohttp

http = FastHTTP(
    base_url="https://api.example.com",
    timeout=30,  # Request timeout in seconds
    headers={"User-Agent": "MyApp/1.0"},
    auth=aiohttp.BasicAuth("username", "password"),
    cookies={"session": "abc123"},
    debug=True,  # Enable debug logging
    auto_cleanup=True  # Automatic resource cleanup
)
```

### HTTP Methods

FastHTTP supports all standard HTTP methods:

```python
@http.get("/users")
async def list_users(response):
    return await response.json()

@http.post("/users")
async def create_user(response, json=None):
    return await response.json()

@http.put("/users/{user_id}")
async def update_user(response, user_id, json=None):
    return await response.json()

@http.patch("/users/{user_id}")
async def partial_update_user(response, user_id, json=None):
    return await response.json()

@http.delete("/users/{user_id}")
async def delete_user(response, user_id):
    return response.status == 204

@http.head("/users/{user_id}")
async def user_exists(response, user_id):
    return response.status == 200

@http.options("/users")
async def get_allowed_methods(response):
    return response.headers.get("Allow", "").split(", ")
```

### URL Templates

Use URL templates with dynamic parameters:

```python
@http.get("/users/{user_id}/posts/{post_id}")
async def get_user_post(response, user_id, post_id):
    return await response.json()

# Usage
post = await get_user_post(user_id=123, post_id=456)
```

### Request Parameters

Pass various request parameters:

```python
# Query parameters example
@http.get("/search")
async def search(response, params=None):
    """Search with query parameters"""
    return await response.json()

# Usage with query parameters
results = await search(
    params={"query": "python", "page": 1, "limit": 10}
)

# JSON body example
@http.post("/users")
async def create_user(response, json=None):
    """Create user with JSON data"""
    return await response.json()

# Usage with JSON body
user = await create_user(
    json={"name": "John Doe", "email": "john@example.com"}
)

# Form data example
@http.post("/upload")
async def upload_file(response, data=None):
    """Upload file with form data"""
    return await response.json()

# Usage with form data
result = await upload_file(
    data={"file": open("document.pdf", "rb")}
)

# Multiple parameters with URL templating
@http.get("/users/{user_id}/posts")
async def get_user_posts(response, user_id, params=None, headers=None):
    """Get user posts with query parameters and custom headers"""
    return await response.json()

# Usage with URL parameter, query parameters, and headers
posts = await get_user_posts(
    user_id=123,
    params={"page": 1, "limit": 5},
    headers={"Authorization": "Bearer token123"}
)

# Advanced example with multiple HTTP options
@http.post("/api/data")
async def send_data(response, json=None, params=None, headers=None, timeout=None):
    """Send data with multiple request options"""
    return await response.json()

# Usage with multiple options
result = await send_data(
    json={"data": "example"},
    params={"version": "v1"},
    headers={"Content-Type": "application/json"},
    timeout=60
)
```

### Error Handling

```python
import aiohttp

@http.get("/users/{user_id}")
async def get_user(response, user_id):
    if response.status == 404:
        return None
    response.raise_for_status()  # Raise exception for HTTP errors
    return await response.json()

# Or use raise_for_status parameter
@http.get("/users/{user_id}")
async def get_user_safe(response, user_id, raise_for_status=None):
    return await response.json()

try:
    user = await get_user_safe(user_id=999, raise_for_status=True)
except aiohttp.ClientResponseError as e:
    print(f"HTTP error: {e.status}")
```

### Context Manager

FastHTTP automatically cleans up resources when your program exits (enabled by default), but you can also use it as an async context manager for explicit control:

```python
# Option 1: Automatic cleanup (default behavior)
http = FastHTTP(base_url="https://api.example.com")

@http.get("/data")
async def get_data(response):
    return await response.json()

async def main():
    data = await get_data()
    print(data)
    # Resources are automatically cleaned up when program exits

# Option 2: Explicit cleanup with context manager
async def main():
    async with FastHTTP(base_url="https://api.example.com") as http:
        @http.get("/data")
        async def get_data(response):
            return await response.json()
        
        data = await get_data()
        print(data)
    # Resources are explicitly cleaned up here

# Option 3: Manual cleanup (if auto_cleanup=False)
async def main():
    http = FastHTTP(base_url="https://api.example.com", auto_cleanup=False)
    
    @http.get("/data")
    async def get_data(response):
        return await response.json()
    
    try:
        data = await get_data()
        print(data)
    finally:
        await http.close()  # Manual cleanup required
```

### Custom Connectors

For advanced use cases, you can provide custom connectors:

```python
import aiohttp

# Custom connector with specific settings
connector = aiohttp.TCPConnector(
    limit=100,
    limit_per_host=10,
    ttl_dns_cache=300,
    use_dns_cache=True,
    keepalive_timeout=30,
    enable_cleanup_closed=True
)

http = FastHTTP(
    base_url="https://api.example.com",
    connector=connector
)
```

## 🔧 Advanced Usage

### Debugging

Enable debug mode for detailed logging:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create client with debug enabled
http = FastHTTP(
    base_url="https://api.example.com",
    debug=True
)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass
6. Submit a pull request

## 📋 Requirements

- Python 3.8+
- aiohttp 3.8+

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of the excellent [aiohttp](https://github.com/aio-libs/aiohttp) library
- Inspired by modern API design patterns
- Thanks to all contributors and users

## 📞 Support

- 📫 Issues: [GitHub Issues](https://github.com/2-seo/fasthttp/issues)
- ✉️ Email: seohyun.develop@gmail.com


---

Made with ❤️ by the FastHTTP team

