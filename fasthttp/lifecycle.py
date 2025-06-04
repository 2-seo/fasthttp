"""
FastHTTP instance lifecycle management module.

This module handles automatic cleanup of FastHTTP instances when the process exits,
preventing resource leaks and connection warnings.
"""

import asyncio
import atexit
import weakref
from typing import Set, TYPE_CHECKING

if TYPE_CHECKING:
    from fasthttp.applications import FastHTTP

# Global tracking of active FastHTTP instances
_active_instances: Set[weakref.ReferenceType] = set()
_cleanup_registered = False


def _cleanup_all_instances():
    """Clean up all active FastHTTP instances when the process exits."""
    for instance_ref in list(_active_instances):
        instance = instance_ref()
        if instance is not None:
            try:
                # Check if event loop is running
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Add as task if loop is running
                    loop.create_task(instance._cleanup_sync())
                else:
                    # Run in new loop if no loop or stopped
                    asyncio.run(instance._cleanup_sync())
            except Exception:
                # Continue cleaning other instances even if one fails
                pass


def _register_global_cleanup():
    """Register global cleanup handler only once."""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(_cleanup_all_instances)
        _cleanup_registered = True


def register_instance(instance: 'FastHTTP') -> None:
    """Register a FastHTTP instance for automatic cleanup."""
    # Register global cleanup handler
    _register_global_cleanup()
    
    # Add this instance to global tracking list
    _active_instances.add(weakref.ref(instance, _remove_from_tracking))
    
    # Use weakref.finalize for cleanup when object is GC'd
    instance._finalizer = weakref.finalize(
        instance,
        _cleanup_finalizer,
        weakref.ref(instance._get_session),
        weakref.ref(lambda: instance._connector)
    )


def _remove_from_tracking(instance_ref: weakref.ReferenceType) -> None:
    """Remove instance from tracking list when it's garbage collected."""
    _active_instances.discard(instance_ref)


def _cleanup_finalizer(session_ref: weakref.ReferenceType, connector_ref: weakref.ReferenceType) -> None:
    """Cleanup function for finalizer (synchronous)."""
    try:
        session_func = session_ref()
        if session_func:
            session = session_func()
            if session and not session.closed:
                try:
                    # Attempt to close session synchronously
                    session._connector.close()
                except Exception:
                    pass
                    
        connector_func = connector_ref()
        if connector_func:
            connector = connector_func()
            if connector and not connector.closed:
                try:
                    connector.close()
                except Exception:
                    pass
    except Exception:
        # Ignore exceptions in finalizer
        pass 