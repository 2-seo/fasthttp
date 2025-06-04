"""
FastHTTP - A fast and elegant HTTP client library with decorator-based request handling.

Built on top of aiohttp, FastHTTP provides a clean, decorator-based interface
for making HTTP requests with automatic resource management.
"""

from fasthttp.applications import FastHTTP
from fasthttp.serializers import to_dict

__version__ = "0.1.1"
__all__ = ["FastHTTP", "to_dict"]
