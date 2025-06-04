"""
Object serialization utilities for FastHTTP.

This module provides smart serialization for various Python objects,
converting them to JSON-serializable dictionaries for HTTP requests.
"""

import io
from typing import Any, Dict, List, Union, TypeAlias

# Type aliases for better readability and type safety
JsonSerializable: TypeAlias = Union[
    None, bool, int, float, str, 
    Dict[str, 'JsonSerializable'], 
    List['JsonSerializable']
]

SerializableInput: TypeAlias = Union[
    # Primitives
    None, bool, int, float, str,
    # Collections
    Dict[str, Any], List[Any], tuple,
    # Objects that can be serialized
    Any  # For objects with __dict__, dataclasses, Pydantic models, etc.
]


def to_dict(data: SerializableInput) -> JsonSerializable:
    """
    Convert Python objects to JSON-serializable dictionaries recursively.
    
    Automatically converts objects to JSON-serializable dictionaries:
    - Pydantic BaseModel -> dict (with nested models)
    - Dataclasses -> dict (with nested dataclasses) 
    - Regular classes with __dict__ -> dict (with nested objects)
    - Other objects with .dict() method -> dict
    - Lists/tuples -> list (with nested objects serialized)
    - Dictionaries -> dict (with nested objects serialized)
    - File objects (IOBase) -> unchanged (for file uploads)
    - Regular primitives -> unchanged
    
    This function is used for both 'json' and 'data' parameters to ensure
    consistent object handling across different request types.
    
    Args:
        data: The object to convert to a dictionary
        
    Returns:
        JSON-serializable representation of the object
        
    Examples:
        >>> from dataclasses import dataclass
        >>> from pydantic import BaseModel
        
        >>> @dataclass
        ... class User:
        ...     name: str
        ...     age: int
        
        >>> user = User("John", 30)
        >>> to_dict(user)
        {'name': 'John', 'age': 30}
        
        >>> class PostRequest(BaseModel):
        ...     title: str
        ...     content: str
        
        >>> post = PostRequest(title="Hello", content="World")
        >>> to_dict(post)
        {'title': 'Hello', 'content': 'World'}
    """
    if data is None:
        return None
    
    # Check for file objects first (IOBase subclasses) - don't serialize these
    if isinstance(data, io.IOBase):
        return data
        
    # Check for Pydantic BaseModel
    if hasattr(data, 'model_dump'):
        # Pydantic v2 - handles nested models automatically
        return data.model_dump()
    elif hasattr(data, 'dict'):
        # Pydantic v1 or other objects with .dict() method
        return data.dict()
    
    # Check for dataclass
    elif hasattr(data, '__dataclass_fields__'):
        import dataclasses
        # dataclasses.asdict() handles nested dataclasses automatically
        return dataclasses.asdict(data)
    
    # Check for regular class with __dict__ attribute
    elif hasattr(data, '__dict__'):
        # Recursively serialize nested objects
        result = {}
        for key, value in vars(data).items():
            result[key] = to_dict(value)  # Recursive call
        return result
    
    # Handle lists/tuples
    elif isinstance(data, (list, tuple)):
        return [to_dict(item) for item in data]
    
    # Handle dictionaries
    elif isinstance(data, dict):
        return {key: to_dict(value) for key, value in data.items()}
    
    # Return as-is for regular primitives (str, int, float, bool, None)
    return data 