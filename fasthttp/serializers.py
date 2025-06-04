"""
Object serialization utilities for FastHTTP.

This module provides smart serialization for various Python objects,
converting them to JSON-serializable dictionaries for HTTP requests.
"""

import io
from datetime import date, datetime, time
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
    - Date/DateTime/Time objects -> ISO format strings
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
        >>> from datetime import date, datetime
        
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
        ...     created_at: datetime
        
        >>> post = PostRequest(title="Hello", content="World", created_at=datetime.now())
        >>> result = to_dict(post)
        >>> # result['created_at'] will be ISO format string
    """
    if data is None:
        return None
    
    # Check for file objects first (IOBase subclasses) - don't serialize these
    if isinstance(data, io.IOBase):
        return data
    
    # Handle date/time objects - convert to ISO format strings
    if isinstance(data, datetime):
        return data.isoformat()
    elif isinstance(data, date):
        return data.isoformat()
    elif isinstance(data, time):
        return data.isoformat()
        
    # Check for Pydantic BaseModel
    if hasattr(data, 'model_dump'):
        # Pydantic v2 - handles nested models automatically
        result = data.model_dump()
        # Recursively process the result to handle date/time objects
        return to_dict(result)
    elif hasattr(data, 'dict'):
        # Pydantic v1 or other objects with .dict() method
        result = data.dict()
        # Recursively process the result to handle date/time objects
        return to_dict(result)
    
    # Check for dataclass
    elif hasattr(data, '__dataclass_fields__'):
        import dataclasses
        # dataclasses.asdict() handles nested dataclasses automatically
        result = dataclasses.asdict(data)
        # Recursively process the result to handle date/time objects
        return to_dict(result)
    
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