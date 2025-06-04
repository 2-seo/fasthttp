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
async def get_post(response, post_id: int):
    """Fetch a single post by ID."""
    return await response.json()

@http.post("/posts")
async def create_post(response, title: str, body: str, userId: int):
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
async def create_user(response, **user_data):
    return await response.json()

@http.put("/users/{user_id}")
async def update_user(response, user_id: int, **user_data):
    return await response.json()

@http.patch("/users/{user_id}")
async def partial_update_user(response, user_id: int, **changes):
    return await response.json()

@http.delete("/users/{user_id}")
async def delete_user(response, user_id: int):
    return response.status == 204

@http.head("/users/{user_id}")
async def user_exists(response, user_id: int):
    return response.status == 200

@http.options("/users")
async def get_allowed_methods(response):
    return response.headers.get("Allow", "").split(", ")
```

### URL Templates

Use URL templates with dynamic parameters:

```python
@http.get("/users/{user_id}/posts/{post_id}")
async def get_user_post(response, user_id: int, post_id: int):
    return await response.json()

# Usage
post = await get_user_post(user_id=123, post_id=456)
```

### Request Parameters

Pass various request parameters:

```python
@http.get("/search")
async def search(response, query: str, page: int = 1):
    return await response.json()

# Usage with query parameters
results = await search(
    query="python",
    params={"page": 1, "limit": 10}
)

# Usage with JSON body
@http.post("/users")
async def create_user(response):
    return await response.json()

user = await create_user(
    json={"name": "John Doe", "email": "john@example.com"}
)

# Usage with form data
@http.post("/upload")
async def upload_file(response):
    return await response.json()

result = await upload_file(
    data={"file": open("document.pdf", "rb")}
)
```

### Error Handling

```python
import aiohttp

@http.get("/users/{user_id}")
async def get_user(response, user_id: int):
    if response.status == 404:
        return None
    response.raise_for_status()  # Raise exception for HTTP errors
    return await response.json()

# Or use raise_for_status parameter
@http.get("/users/{user_id}")
async def get_user_safe(response, user_id: int):
    return await response.json()

try:
    user = await get_user_safe(user_id=999, raise_for_status=True)
except aiohttp.ClientResponseError as e:
    print(f"HTTP error: {e.status}")
```

### Context Manager

Use FastHTTP as an async context manager for automatic resource cleanup:

```python
async def main():
    async with FastHTTP(base_url="https://api.example.com") as http:
        @http.get("/data")
        async def get_data(response):
            return await response.json()
        
        data = await get_data()
        print(data)
    # Resources are automatically cleaned up here
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
- 💬 Discussions: [GitHub Discussions](https://github.com/2-seo/fasthttp/discussions)


---

Made with ❤️ by the FastHTTP team

