"""
WebSocket handler for real-time log streaming
"""

import asyncio
import logging
from datetime import datetime
from collections import deque
from typing import Dict, Set, Optional
from fastapi import WebSocket


class LogStreamManager:
    """Manages WebSocket connections for log streaming"""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize log stream manager
        
        Args:
            max_history: Maximum number of log entries to keep in history
        """
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._log_history: deque = deque(maxlen=max_history)
        self._max_history = max_history
    
    async def subscribe(self, websocket: WebSocket, channel: str = "logs") -> None:
        """
        Subscribe a WebSocket connection to a channel
        
        Args:
            websocket: WebSocket connection
            channel: Channel name (default: "logs")
        """
        await websocket.accept()
        
        if channel not in self._connections:
            self._connections[channel] = set()
        
        self._connections[channel].add(websocket)
        
        # Send last 50 log entries to new subscriber
        recent_logs = list(self._log_history)[-50:]
        for log_entry in recent_logs:
            try:
                await websocket.send_json(log_entry)
            except:
                # Connection might be closed
                self._connections[channel].discard(websocket)
                break
    
    async def unsubscribe(self, websocket: WebSocket, channel: str = "logs") -> None:
        """
        Unsubscribe a WebSocket connection from a channel
        
        Args:
            websocket: WebSocket connection
            channel: Channel name
        """
        if channel in self._connections:
            self._connections[channel].discard(websocket)
    
    async def publish(self, level: str, message: str, source: str = "server") -> None:
        """
        Publish a log entry to all subscribers
        
        Args:
            level: Log level (INFO, WARNING, ERROR)
            message: Log message
            source: Log source
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "source": source
        }
        
        # Add to history
        self._log_history.append(entry)
        
        # Broadcast to all connections
        if "logs" in self._connections:
            disconnected = set()
            for ws in self._connections["logs"]:
                try:
                    await ws.send_json(entry)
                except:
                    disconnected.add(ws)
            
            # Remove disconnected clients
            for ws in disconnected:
                self._connections["logs"].discard(ws)
    
    async def broadcast(self, channel: str, message: dict) -> None:
        """
        Broadcast a message to all connections on a channel
        
        Args:
            channel: Channel name
            message: Message dict to send
        """
        if channel in self._connections:
            disconnected = set()
            for ws in self._connections[channel]:
                try:
                    await ws.send_json(message)
                except:
                    disconnected.add(ws)
            
            for ws in disconnected:
                self._connections[channel].discard(ws)
    
    def get_history(self, limit: Optional[int] = None) -> list:
        """
        Get log history
        
        Args:
            limit: Maximum number of entries to return
        
        Returns:
            List of log entries
        """
        if limit:
            return list(self._log_history)[-limit:]
        return list(self._log_history)
    
    def get_connection_count(self, channel: str = "logs") -> int:
        """Get number of active connections for a channel"""
        return len(self._connections.get(channel, set()))
    
    def clear_history(self) -> None:
        """Clear log history"""
        self._log_history.clear()


class WebSocketLogHandler(logging.Handler):
    """
    Custom logging handler that streams logs to WebSocket connections
    
    This integrates with Python's logging system and broadcasts
    log messages to all connected WebSocket clients.
    """
    
    def __init__(self, manager: LogStreamManager):
        """
        Initialize WebSocket log handler
        
        Args:
            manager: LogStreamManager instance
        """
        super().__init__()
        self.manager = manager
    
    def emit(self, record):
        """Emit a log record to WebSocket connections"""
        try:
            # Format the log message
            log_entry = self.format(record)
            
            # Create async task to publish
            # Note: This is called from logging thread, so we need to
            # schedule the coroutine in the event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(
                    self.manager.publish(
                        level=record.levelname,
                        message=log_entry,
                        source="server"
                    )
                )
        except Exception:
            # Silently fail if we can't send logs
            pass


# Global log stream manager instance
_log_stream_manager: Optional[LogStreamManager] = None


def get_log_stream_manager() -> LogStreamManager:
    """
    Get global log stream manager instance
    
    Returns:
        LogStreamManager instance
    """
    global _log_stream_manager
    if _log_stream_manager is None:
        _log_stream_manager = LogStreamManager()
    return _log_stream_manager


def setup_log_streaming(logger: logging.Logger) -> None:
    """
    Setup log streaming for a logger
    
    Args:
        logger: Python logger instance to add WebSocket handler to
    """
    manager = get_log_stream_manager()
    handler = WebSocketLogHandler(manager)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)