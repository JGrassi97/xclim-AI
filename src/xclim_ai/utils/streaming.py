# xclim_ai.utils.streaming
# =================================================

# Centralized event streaming system for real-time monitoring of agent activities.
# Provides a thread-safe way to emit and consume events across the entire package.

import threading
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class EventType(Enum):
    """Types of events that can be streamed."""
    INFO = "info"
    DEBUG = "debug" 
    WARNING = "warning"
    ERROR = "error"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_THINKING = "agent_thinking"
    RAG_QUERY_REFINED = "rag_query_refined"
    RAG_RETRIEVAL = "rag_retrieval"
    RAG_EVALUATION = "rag_evaluation"
    RAG_AGGREGATION = "rag_aggregation"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_OBSERVATION = "tool_observation"


@dataclass
class StreamEvent:
    """A single event in the stream."""
    type: EventType
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"


class EventStreamer:
    """Thread-safe event streamer for real-time monitoring."""
    
    def __init__(self):
        self._events: List[StreamEvent] = []
        self._lock = threading.Lock()
        self._listeners: List[Callable[[StreamEvent], None]] = []
        
    def emit(self, event_type: EventType, message: str, source: str = "unknown", **metadata):
        """Emit a new event to all listeners."""
        event = StreamEvent(
            type=event_type,
            message=message,
            source=source,
            metadata=metadata
        )
        
        with self._lock:
            self._events.append(event)
            
        # Notify all listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass  # Don't let listener errors break the stream
                
    def add_listener(self, listener: Callable[[StreamEvent], None]):
        """Add a listener function to receive events."""
        self._listeners.append(listener)
        
    def remove_listener(self, listener: Callable[[StreamEvent], None]):
        """Remove a listener function."""
        if listener in self._listeners:
            self._listeners.remove(listener)
            
    def get_events(self, since_timestamp: Optional[float] = None) -> List[StreamEvent]:
        """Get all events, optionally filtered by timestamp."""
        with self._lock:
            if since_timestamp is None:
                return self._events.copy()
            return [e for e in self._events if e.timestamp >= since_timestamp]
            
    def clear(self):
        """Clear all stored events."""
        with self._lock:
            self._events.clear()


# Global instance for the entire package
_global_streamer = EventStreamer()


def get_streamer() -> EventStreamer:
    """Get the global event streamer instance."""
    return _global_streamer


def stream_info(message: str, source: str = "unknown", **metadata):
    """Convenience function to emit an info event."""
    _global_streamer.emit(EventType.INFO, message, source, **metadata)


def stream_debug(message: str, source: str = "unknown", **metadata):
    """Convenience function to emit a debug event."""
    _global_streamer.emit(EventType.DEBUG, message, source, **metadata)


def stream_warning(message: str, source: str = "unknown", **metadata):
    """Convenience function to emit a warning event.""" 
    _global_streamer.emit(EventType.WARNING, message, source, **metadata)


def stream_error(message: str, source: str = "unknown", **metadata):
    """Convenience function to emit an error event."""
    _global_streamer.emit(EventType.ERROR, message, source, **metadata)
