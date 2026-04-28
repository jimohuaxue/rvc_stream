"""
Admin panel API routes
"""

import aiofiles
from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
from typing import List, Optional

from .models import (
    StreamingConfig,
    ServerConfig,
    ModelConfig,
    RvcConfig,
    ConfigUpdateRequest,
    ConfigResponse,
    ServerStats,
    LogEntry,
    LoadModelRequest,
    LoadModelResponse,
)
from .config_manager import get_config_manager
from .websocket import get_log_stream_manager
from .utils import SystemMonitor

# Create router
router = APIRouter(prefix="/admin", tags=["admin"])

# Global server reference (will be set during initialization)
_server_instance = None


def init_admin_panel(server_instance):
    """Initialize admin panel with server instance"""
    global _server_instance
    _server_instance = server_instance


# ============== HTML Pages ==============

@router.get("/", response_class=HTMLResponse)
async def admin_dashboard():
    """Admin dashboard main page"""
    template_path = Path(__file__).parent.parent / "templates" / "admin" / "dashboard.html"
    if template_path.exists():
        return HTMLResponse(content=template_path.read_text(encoding='utf-8'))
    else:
        return HTMLResponse(content="<h1>Admin Dashboard</h1><p>Template not found</p>")


# ============== Configuration API ==============

@router.get("/api/config")
async def get_config():
    """Get current server configuration"""
    try:
        manager = get_config_manager()
        config = manager.get()
        return {
            "status": "success",
            "config": config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")


@router.post("/api/config")
async def update_config(request: ConfigUpdateRequest):
    """Update server configuration"""
    try:
        manager = get_config_manager()
        
        # Update streaming config
        if request.streaming:
            manager.update_streaming(request.streaming)
        
        # Update server config
        if request.server:
            manager.update_server(request.server)
        
        # Broadcast update to all WebSocket clients
        ws_manager = get_log_stream_manager()
        await ws_manager.broadcast("config", {
            "type": "config_updated",
            "config": manager.get()
        })
        
        return {
            "status": "success",
            "message": "Configuration updated",
            "config": manager.get()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@router.post("/api/config/streaming")
async def update_streaming_config(config: StreamingConfig):
    """Update streaming configuration only"""
    try:
        manager = get_config_manager()
        manager.update_streaming(config)
        
        # Broadcast update
        ws_manager = get_log_stream_manager()
        await ws_manager.broadcast("config", {
            "type": "streaming_config_updated",
            "config": config.model_dump()
        })
        
        return {
            "status": "success",
            "message": "Streaming configuration updated",
            "streaming": config.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update streaming config: {str(e)}")


@router.get("/api/config/rvc")
async def get_rvc_config():
    """Get RVC inference configuration"""
    try:
        manager = get_config_manager()
        return {"status": "success", "rvc": manager.get_rvc_config().model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/config/rvc")
async def update_rvc_config(config: RvcConfig):
    """Update RVC inference configuration and apply to live processor immediately."""
    try:
        manager = get_config_manager()
        manager.update_rvc(config)

        # Push to live processor so changes take effect without restart
        if _server_instance and hasattr(_server_instance, "processor"):
            _server_instance.processor.update_rvc_params(config.model_dump())

        ws_manager = get_log_stream_manager()
        await ws_manager.publish(
            "INFO",
            f"RVC params updated: pitch={config.pitch}, f0method={config.f0method}, "
            f"index_rate={config.index_rate}, threshold={config.threshold}",
        )

        return {"status": "success", "rvc": config.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/config/reset")
async def reset_config():
    """Reset configuration to defaults"""
    try:
        manager = get_config_manager()
        manager.reset_to_defaults()
        
        return {
            "status": "success",
            "message": "Configuration reset to defaults",
            "config": manager.get()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset config: {str(e)}")


# ============== Models API ==============

@router.get("/api/models")
async def list_models():
    """List loaded models and available models"""
    try:
        manager = get_config_manager()
        
        # Get loaded models from server instance
        loaded_models = []
        if _server_instance and hasattr(_server_instance, 'model_manager'):
            loaded_models = list(_server_instance.model_manager.models.keys())
        
        # Get available models from config
        available_models = manager.list_available_models()
        
        return {
            "status": "success",
            "loaded": loaded_models,
            "available": available_models,
            "configured": manager.get_models_config()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.post("/api/models/load")
async def load_model(request: LoadModelRequest):
    """Load a model into memory"""
    try:
        if not _server_instance:
            raise HTTPException(status_code=503, detail="Server not initialized")
        
        # Load model through server instance
        if hasattr(_server_instance, 'processor'):
            await _server_instance.processor.set_model(
                request.pth_path,
                request.index_path,
                request.index_rate
            )
            
            model_name = Path(request.pth_path).stem
            
            # Broadcast model loaded event
            ws_manager = get_log_stream_manager()
            await ws_manager.publish("INFO", f"Model loaded: {model_name}")
            
            return {
                "status": "success",
                "model_name": model_name,
                "message": f"Model {model_name} loaded successfully"
            }
        else:
            raise HTTPException(status_code=503, detail="Model processor not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a .pth model file using streaming (chunked) I/O."""
    if not file.filename or not file.filename.endswith(".pth"):
        raise HTTPException(status_code=400, detail="只接受 .pth 文件")

    upload_dir = Path("models")
    upload_dir.mkdir(exist_ok=True)

    # Prevent path traversal
    safe_name = Path(file.filename).name
    dest = upload_dir / safe_name

    try:
        async with aiofiles.open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):   # 1 MB chunks
                await out.write(chunk)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"写入失败: {e}")

    size = dest.stat().st_size
    ws_manager = get_log_stream_manager()
    await ws_manager.publish("INFO", f"Model uploaded: {safe_name} ({size // 1024} KB)")

    return {"status": "success", "name": safe_name, "path": str(dest), "size": size}


@router.get("/api/models/uploaded")
async def list_uploaded_models():
    """List model files in the models/ upload directory."""
    manager = get_config_manager()
    return {"status": "success", "models": manager.list_uploaded_models()}


@router.delete("/api/models/uploaded/{model_name}")
async def delete_uploaded_model(model_name: str):
    """Delete an uploaded model file."""
    upload_dir = Path("models").resolve()
    target = (upload_dir / Path(model_name).name).resolve()

    # Path traversal guard
    if not str(target).startswith(str(upload_dir)):
        raise HTTPException(status_code=400, detail="非法文件名")
    if not target.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    target.unlink()
    ws_manager = get_log_stream_manager()
    await ws_manager.publish("INFO", f"Model deleted: {model_name}")
    return {"status": "success"}


@router.delete("/api/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a model from memory"""
    try:
        if not _server_instance:
            raise HTTPException(status_code=503, detail="Server not initialized")
        
        # Unload model through server instance
        if hasattr(_server_instance, 'model_manager'):
            if model_name in _server_instance.model_manager.models:
                del _server_instance.model_manager.models[model_name]
                
                # Broadcast model unloaded event
                ws_manager = get_log_stream_manager()
                await ws_manager.publish("INFO", f"Model unloaded: {model_name}")
                
                return {"status": "success", "message": f"Model {model_name} unloaded"}
            else:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        else:
            raise HTTPException(status_code=503, detail="Model manager not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


# ============== Statistics API ==============

@router.get("/api/stats")
async def get_stats():
    """Get server statistics"""
    try:
        # Get system stats
        cpu_percent = SystemMonitor.get_cpu_usage()
        memory_info = SystemMonitor.get_memory_usage()
        uptime = SystemMonitor.get_uptime_seconds()
        
        # Get connection count
        ws_manager = get_log_stream_manager()
        log_connections = ws_manager.get_connection_count("logs")
        
        # Get model count
        model_count = 0
        if _server_instance and hasattr(_server_instance, 'model_manager'):
            model_count = len(_server_instance.model_manager.models)
        
        # Get active connections
        active_connections = 0
        if _server_instance and hasattr(_server_instance, 'connections'):
            active_connections = len(_server_instance.connections)
        
        return {
            "status": "success",
            "stats": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_info['percent'],
                "memory_used": memory_info['used'],
                "memory_total": memory_info['total'],
                "uptime_seconds": uptime,
                "uptime_formatted": SystemMonitor.get_uptime_formatted(),
                "models_loaded": model_count,
                "active_connections": active_connections,
                "log_connections": log_connections
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/api/connections")
async def list_connections():
    """List active WebSocket connections"""
    try:
        connections = []
        
        if _server_instance and hasattr(_server_instance, 'connections'):
            for conn in _server_instance.connections:
                connection_info = {
                    "id": str(id(conn)),
                    "model": None,
                    "sample_rate": getattr(conn, 'config', {}).get('sample_rate', 16000) if hasattr(conn, 'config') else 16000,
                    "chunk_size": getattr(conn, 'config', {}).get('chunk_size', 512) if hasattr(conn, 'config') else 512,
                }
                
                # Get current model if available
                if hasattr(conn, 'processor') and hasattr(conn.processor, 'current_model'):
                    if conn.processor.current_model:
                        connection_info["model"] = conn.processor.current_model.name
                
                connections.append(connection_info)
        
        return {
            "status": "success",
            "count": len(connections),
            "connections": connections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list connections: {str(e)}")


# ============== Logs API ==============

@router.get("/api/logs")
async def get_logs(limit: int = 100, level: Optional[str] = None):
    """Get log history"""
    try:
        ws_manager = get_log_stream_manager()
        logs = ws_manager.get_history(limit)
        
        # Filter by level if specified
        if level and level != "ALL":
            logs = [log for log in logs if log.get('level') == level]
        
        return {
            "status": "success",
            "count": len(logs),
            "logs": logs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


@router.delete("/api/logs")
async def clear_logs():
    """Clear log history"""
    try:
        ws_manager = get_log_stream_manager()
        ws_manager.clear_history()
        
        return {
            "status": "success",
            "message": "Log history cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear logs: {str(e)}")


# ============== WebSocket Endpoints ==============

@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming"""
    ws_manager = get_log_stream_manager()
    
    try:
        await ws_manager.subscribe(websocket, "logs")
        
        # Keep connection alive
        while True:
            # Wait for any message from client (ping/pong)
            data = await websocket.receive_json()
            
            # Handle ping
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": data.get("timestamp")})
            
    except WebSocketDisconnect:
        await ws_manager.unsubscribe(websocket, "logs")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await ws_manager.unsubscribe(websocket, "logs")


@router.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    """WebSocket endpoint for real-time statistics"""
    ws_manager = get_log_stream_manager()
    
    try:
        await ws_manager.subscribe(websocket, "stats")
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "get_stats":
                # Send current stats
                stats = await get_stats()
                await websocket.send_json({
                    "type": "stats_update",
                    "stats": stats["stats"]
                })
                
    except WebSocketDisconnect:
        await ws_manager.unsubscribe(websocket, "stats")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await ws_manager.unsubscribe(websocket, "stats")