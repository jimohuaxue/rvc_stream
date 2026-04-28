"""
Admin panel module initialization
"""

from .models import (
    StreamingConfig,
    ServerConfig,
    ModelConfig,
    ConfigUpdateRequest,
    ConfigResponse,
    ConnectionInfo,
    ServerStats,
    LogEntry,
    LoadModelRequest,
    LoadModelResponse,
)
from .config_manager import ConfigManager, get_config_manager
from .websocket import LogStreamManager, get_log_stream_manager, setup_log_streaming
from .utils import (
    SystemMonitor,
    LogFilter,
    ParamsValidator,
    format_bytes,
    format_duration,
)
from .routes import router as admin_router, init_admin_panel

__all__ = [
    # Models
    "StreamingConfig",
    "ServerConfig",
    "ModelConfig",
    "ConfigUpdateRequest",
    "ConfigResponse",
    "ConnectionInfo",
    "ServerStats",
    "LogEntry",
    "LoadModelRequest",
    "LoadModelResponse",
    # Managers
    "ConfigManager",
    "get_config_manager",
    "LogStreamManager",
    "get_log_stream_manager",
    "setup_log_streaming",
    # Utils
    "SystemMonitor",
    "LogFilter",
    "ParamsValidator",
    "format_bytes",
    "format_duration",
    # Routes
    "admin_router",
    "init_admin_panel",
]
