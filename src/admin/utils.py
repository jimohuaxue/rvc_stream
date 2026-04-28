"""
Utility functions for admin panel
"""

import os
import psutil
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class SystemMonitor:
    """System resource monitoring utilities"""

    _start_time: float = time.time()

    @classmethod
    def get_cpu_usage(cls) -> float:
        """Get CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0

    @classmethod
    def get_memory_usage(cls) -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            mem = psutil.virtual_memory()
            return {
                "total": mem.total,
                "available": mem.available,
                "used": mem.used,
                "percent": mem.percent,
            }
        except:
            return {"total": 0, "available": 0, "used": 0, "percent": 0.0}

    @classmethod
    def get_disk_usage(cls, path: str = "/") -> Dict[str, Any]:
        """Get disk usage information"""
        try:
            disk = psutil.disk_usage(path)
            return {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            }
        except:
            return {"total": 0, "used": 0, "free": 0, "percent": 0.0}

    @classmethod
    def get_uptime_seconds(cls) -> float:
        """Get server uptime in seconds"""
        return time.time() - cls._start_time

    @classmethod
    def get_uptime_formatted(cls) -> str:
        """Get server uptime formatted string"""
        uptime = cls.get_uptime_seconds()
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        return f"{hours}h {minutes}m {seconds}s"


class LogFilter:
    """Log filtering utilities"""

    @staticmethod
    def filter_by_level(logs: list, level: str) -> list:
        """Filter logs by level"""
        if level == "ALL":
            return logs
        return [log for log in logs if log.get("level") == level]

    @staticmethod
    def filter_by_keyword(logs: list, keyword: str) -> list:
        """Filter logs by keyword in message"""
        if not keyword:
            return logs
        keyword_lower = keyword.lower()
        return [log for log in logs if keyword_lower in log.get("message", "").lower()]

    @staticmethod
    def filter_by_time_range(
        logs: list, start_time: Optional[datetime], end_time: Optional[datetime]
    ) -> list:
        """Filter logs by time range"""
        if not start_time and not end_time:
            return logs

        filtered = []
        for log in logs:
            try:
                log_time = datetime.fromisoformat(log.get("timestamp", ""))
                if start_time and log_time < start_time:
                    continue
                if end_time and log_time > end_time:
                    continue
                filtered.append(log)
            except:
                # If timestamp parsing fails, include the log
                filtered.append(log)

        return filtered


class ParamsValidator:
    """Parameter validation utilities"""

    @staticmethod
    def validate_sample_rate(value: int) -> bool:
        """Validate sample rate"""
        return 8000 <= value <= 48000

    @staticmethod
    def validate_chunk_size(value: int) -> bool:
        """Validate chunk size (must be power of 2)"""
        return 128 <= value <= 4096 and (value & (value - 1)) == 0

    @staticmethod
    def validate_latency_target(value: int) -> bool:
        """Validate latency target"""
        return 10 <= value <= 500

    @staticmethod
    def validate_index_rate(value: float) -> bool:
        """Validate index rate"""
        return 0.0 <= value <= 1.0


def format_bytes(size: int) -> str:
    """Format bytes to human readable string"""
    display_size = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if display_size < 1024.0:
            return f"{display_size:.2f} {unit}"
        display_size /= 1024.0
    return f"{display_size:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format duration to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
