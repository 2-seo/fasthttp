"""
FastHTTP - A fast and elegant HTTP client library with decorator-based request handling.

Built on top of aiohttp, FastHTTP provides a clean, decorator-based interface
for making HTTP requests with automatic resource management.
"""

from .applications import FastHTTP

__version__ = "0.1.0"
__all__ = ["FastHTTP"]
