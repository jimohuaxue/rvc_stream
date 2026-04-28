"""
Pydantic models for admin panel API request/response validation
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class StreamingConfig(BaseModel):
    """Audio streaming configuration"""

    sample_rate: int = Field(
        default=16000, ge=8000, le=48000, description="Audio sample rate (Hz)"
    )
    chunk_size: int = Field(
        default=512, ge=128, le=4096, description="Samples per chunk"
    )
    channels: int = Field(default=1, ge=1, le=2, description="Number of audio channels")
    dtype: str = Field(default="float32", description="Audio data type")
    latency_target_ms: int = Field(
        default=100, ge=10, le=500, description="Target latency in milliseconds"
    )

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        valid_dtypes = ["float32", "float64", "int16", "int32"]
        if v not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}")
        return v


class ServerConfig(BaseModel):
    """Server configuration"""

    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8080, ge=1, le=65535, description="Server port")
    max_queue_size: int = Field(
        default=100, ge=1, le=1000, description="Max queue size"
    )
    model_cache_size: int = Field(
        default=3, ge=1, le=10, description="Max cached models"
    )


class ModelConfig(BaseModel):
    """RVC model configuration"""

    name: str = Field(..., description="Model name")
    pth_path: str = Field(..., description="Path to .pth model file")
    index_path: Optional[str] = Field(default="", description="Path to .index file")
    index_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Index search rate"
    )
    tgt_sr: int = Field(default=40000, description="Target sample rate")
    n_cpu: int = Field(default=4, ge=1, le=16, description="CPU threads for processing")
    f0method: str = Field(default="fcpe", description="F0 extraction method")

    @field_validator("f0method")
    @classmethod
    def validate_f0method(cls, v: str) -> str:
        valid_methods = ["fcpe", "pm", "harvest", "crepe", "rmvpe"]
        if v not in valid_methods:
            raise ValueError(f"f0method must be one of {valid_methods}")
        return v


class RvcConfig(BaseModel):
    """RVC 实时推理参数（对应 gui_v1.py 的全部推理设置）"""

    pitch: int = Field(default=0, ge=-24, le=24, description="音调偏移（半音）")
    formant: float = Field(default=0.0, ge=-1.0, le=1.0, description="音色/声线粗细")
    index_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="索引使用率")
    rms_mix_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="响度混合率")
    f0method: str = Field(default="rmvpe", description="基频提取算法")
    block_time: float = Field(default=0.25, ge=0.05, le=3.0, description="处理块时长（秒）")
    crossfade_length: float = Field(default=0.05, ge=0.01, le=0.5, description="交叉淡化时长（秒）")
    extra_time: float = Field(default=2.5, ge=0.5, le=10.0, description="额外上下文时长（秒）")
    n_cpu: int = Field(default=4, ge=1, le=16, description="Harvest CPU 线程数")
    I_noise_reduce: bool = Field(default=False, description="输入降噪")
    O_noise_reduce: bool = Field(default=False, description="输出降噪")
    threshold: float = Field(default=-60.0, ge=-100.0, le=0.0, description="响应阈值（dBFS）")
    sr_type: str = Field(default="sr_model", description="采样率来源：sr_model / sr_device")

    @field_validator("f0method")
    @classmethod
    def validate_f0method(cls, v: str) -> str:
        valid = ["pm", "harvest", "crepe", "rmvpe", "fcpe"]
        if v not in valid:
            raise ValueError(f"f0method must be one of {valid}")
        return v

    @field_validator("sr_type")
    @classmethod
    def validate_sr_type(cls, v: str) -> str:
        if v not in ("sr_model", "sr_device"):
            raise ValueError("sr_type must be 'sr_model' or 'sr_device'")
        return v


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration"""

    streaming: Optional[StreamingConfig] = None
    server: Optional[ServerConfig] = None
    rvc: Optional[RvcConfig] = None


class ConfigResponse(BaseModel):
    """Response with current configuration"""

    streaming: StreamingConfig
    server: ServerConfig
    models: List[ModelConfig]


class ConnectionInfo(BaseModel):
    """Active connection information"""

    id: str = Field(..., description="Connection ID")
    model: Optional[str] = Field(None, description="Current model name")
    sample_rate: int = Field(default=16000)
    chunk_size: int = Field(default=512)
    connected_at: datetime = Field(default_factory=datetime.now)
    bytes_sent: int = Field(default=0)
    bytes_received: int = Field(default=0)


class ServerStats(BaseModel):
    """Server statistics"""

    total_connections: int = Field(default=0, description="Total connections count")
    active_connections: int = Field(default=0, description="Active connections count")
    models_loaded: int = Field(default=0, description="Number of loaded models")
    uptime_seconds: float = Field(default=0.0, description="Server uptime in seconds")
    cpu_percent: float = Field(default=0.0, description="CPU usage percentage")
    memory_percent: float = Field(default=0.0, description="Memory usage percentage")
    avg_latency_ms: float = Field(default=0.0, description="Average processing latency")


class LogEntry(BaseModel):
    """Log entry"""

    timestamp: str = Field(..., description="ISO format timestamp")
    level: str = Field(..., description="Log level (INFO, WARNING, ERROR)")
    message: str = Field(..., description="Log message")
    source: str = Field(default="server", description="Log source")


class LoadModelRequest(BaseModel):
    """Request to load a model"""

    pth_path: str
    index_path: str = ""
    index_rate: float = 0.0


class LoadModelResponse(BaseModel):
    """Response from loading a model"""

    status: str
    model_name: Optional[str] = None
    message: Optional[str] = None
