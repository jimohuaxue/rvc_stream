"""
RVC Streaming Server
WebSocket server for remote RVC inference

NOTE: This server can start without torch/fairseq, but requires them
for actual RVC inference. It's designed to run on a GPU machine with RVC models.

For a standalone client that connects to this server, see rvc_client.py
"""

import os
import sys
import json
import asyncio
import argparse
import logging
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Deque
from collections import deque
import time

import numpy as np

# Lazy imports for RVC/torch - loaded only when needed
_torch = None
_rvc_modules = {}


def _load_torch():
    """Lazy load torch module"""
    global _torch
    if _torch is None:
        try:
            import torch

            _torch = torch
            logger.info(f"PyTorch loaded: {torch.__version__}")
        except ImportError:
            logger.error("PyTorch not installed. Install with: pip install torch")
            raise
    return _torch


def _load_rvc_module(module_name: str):
    """Lazy load RVC modules"""
    if module_name not in _rvc_modules:
        try:
            if module_name == "config":
                from configs.config import Config

                _rvc_modules[module_name] = Config
            elif module_name == "rtrvc":
                from infer.lib import rtrvc

                _rvc_modules[module_name] = rtrvc
            elif module_name == "get_synthesizer":
                from infer.lib.jit.get_synthesizer import get_synthesizer

                _rvc_modules[module_name] = get_synthesizer
            elif module_name == "models":
                from infer.lib.infer_pack.models import (
                    SynthesizerTrnMs256NSFsid,
                    SynthesizerTrnMs256NSFsid_nono,
                    SynthesizerTrnMs768NSFsid,
                    SynthesizerTrnMs768NSFsid_nono,
                )

                _rvc_modules[module_name] = {
                    "SynthesizerTrnMs256NSFsid": SynthesizerTrnMs256NSFsid,
                    "SynthesizerTrnMs256NSFsid_nono": SynthesizerTrnMs256NSFsid_nono,
                    "SynthesizerTrnMs768NSFsid": SynthesizerTrnMs768NSFsid,
                    "SynthesizerTrnMs768NSFsid_nono": SynthesizerTrnMs768NSFsid_nono,
                }
        except ImportError as e:
            logger.error(f"Failed to load RVC module {module_name}: {e}")
            raise
    return _rvc_modules.get(module_name)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 512  # samples per chunk
DEFAULT_PORT = 8080
MAX_QUEUE_SIZE = 100
MODEL_CACHE_SIZE = 3


@dataclass
class StreamConfig:
    """Streaming configuration"""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    channels: int = 1
    dtype: str = "float32"
    latency_target_ms: int = 100


@dataclass
class RVCModel:
    """Cached RVC model instance (placeholder for type hint)"""

    name: str
    pth_path: str
    index_path: str = ""
    index_rate: float = 0.0
    tgt_sr: int = 40000
    last_used: float = field(default_factory=time.time)


class ModelManager:
    """Manages RVC model instances with caching"""

    def __init__(self, max_cache: int = MODEL_CACHE_SIZE):
        self.max_cache = max_cache
        self.models: Dict[str, RVCModel] = {}
        self._config = None
        self._rvc = None

    def _get_config(self):
        """Lazy load Config"""
        if self._config is None:
            Config = _load_rvc_module("config")
            self._config = Config()
            self._config.use_jit = False
        return self._config

    def _get_rvc(self):
        """Lazy load RVC module"""
        if self._rvc is None:
            self._rvc = _load_rvc_module("rtrvc")
        return self._rvc

    def _make_key(self, pth_path: str, index_path: str, index_rate: float) -> str:
        return f"{pth_path}:{index_path}:{index_rate}"

    async def load_model(
        self,
        pth_path: str,
        index_path: str = "",
        index_rate: float = 0.0,
        n_cpu: int = 4,
    ) -> RVCModel:
        """
        Load or retrieve cached model.

        NOTE: Actual model loading requires full RVC environment.
        This is a placeholder that creates a mock model for testing.
        """
        key = self._make_key(pth_path, index_path, index_rate)

        # Return cached model
        if key in self.models:
            self.models[key].last_used = time.time()
            return self.models[key]

        # Evict oldest if cache full
        if len(self.models) >= self.max_cache:
            oldest_key = min(self.models.keys(), key=lambda k: self.models[k].last_used)
            old_model = self.models.pop(oldest_key)
            logger.info(f"Evicted model: {old_model.name}")

        # NOTE: In production, load actual RVC model here
        # For now, create a placeholder
        logger.info(f"Loading model: {pth_path}")
        logger.warning("NOTE: Model loading requires full RVC environment")

        model = RVCModel(
            name=os.path.basename(pth_path),
            pth_path=pth_path,
            index_path=index_path,
            index_rate=index_rate,
            tgt_sr=40000,
        )

        self.models[key] = model
        logger.info(f"Model loaded: {model.name}, target SR: {model.tgt_sr}")

        return model


class CircularBuffer:
    """Thread-safe circular buffer for audio processing"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        self.lock = asyncio.Lock()

    async def write(self, data: np.ndarray) -> int:
        """Write data to buffer, return bytes written"""
        async with self.lock:
            n = min(len(data), self.capacity - self.size)
            end_pos = (self.write_pos + n) % self.capacity

            if end_pos > self.write_pos:
                self.buffer[self.write_pos : end_pos] = data[:n]
            else:
                first_part = min(n, self.capacity - self.write_pos)
                self.buffer[self.write_pos :] = data[:first_part]
                if n > first_part:
                    self.buffer[: n - first_part] = data[first_part:n]

            self.write_pos = end_pos
            self.size = min(self.size + n, self.capacity)
            return n

    async def read(self, n: int) -> np.ndarray:
        """Read n samples from buffer"""
        async with self.lock:
            n = min(n, self.size)
            if n == 0:
                return np.array([], dtype=np.float32)

            end_pos = (self.read_pos + n) % self.capacity

            if end_pos > self.read_pos:
                data = self.buffer[self.read_pos : end_pos].copy()
            else:
                first_part = self.buffer[self.read_pos :].copy()
                second_part = (
                    self.buffer[:end_pos].copy()
                    if end_pos > 0
                    else np.array([], dtype=np.float32)
                )
                data = np.concatenate([first_part, second_part])

            self.read_pos = end_pos
            self.size -= n
            return data

    async def available(self) -> int:
        async with self.lock:
            return self.size


class AudioProcessor:
    """Processes audio chunks through RVC"""

    # Default RVC inference parameters (mirrors RvcConfig in admin/models.py)
    _RVC_DEFAULTS = {
        "pitch": 0,
        "formant": 0.0,
        "index_rate": 0.0,
        "rms_mix_rate": 0.0,
        "f0method": "rmvpe",
        "block_time": 0.25,
        "crossfade_length": 0.05,
        "extra_time": 2.5,
        "n_cpu": 4,
        "I_noise_reduce": False,
        "O_noise_reduce": False,
        "threshold": -60.0,
        "sr_type": "sr_model",
    }

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.current_model: Optional[RVCModel] = None
        self.stream_config = StreamConfig()
        self.latency_history: Deque[int] = deque(maxlen=100)
        # Live RVC inference parameters — updated by admin panel in real time
        self.rvc_params: dict = dict(self._RVC_DEFAULTS)

    def update_rvc_params(self, params: dict) -> None:
        """Apply a partial or full RVC parameter dict. Unknown keys are ignored."""
        for k, v in params.items():
            if k in self._RVC_DEFAULTS:
                self.rvc_params[k] = v
        logger.info(
            f"RVC params updated: pitch={self.rvc_params['pitch']}, "
            f"f0method={self.rvc_params['f0method']}, "
            f"index_rate={self.rvc_params['index_rate']}"
        )

    async def set_model(
        self, pth_path: str, index_path: str = "", index_rate: float = 0.0
    ):
        """Set active RVC model"""
        self.current_model = await self.model_manager.load_model(
            pth_path, index_path, index_rate
        )
        # Keep index_rate in sync with rvc_params
        self.rvc_params["index_rate"] = index_rate
        logger.info(f"Active model: {self.current_model.name}")

    async def process_chunk(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio chunk through RVC using current rvc_params.

        NOTE: In production, this calls actual RVC inference with all params.
        Currently returns audio passthrough for testing.
        """
        t0 = time.perf_counter()

        if self.current_model is None:
            return audio_data

        # In production, call RVC inference here with self.rvc_params:
        #   output = rvc.infer(
        #       audio_data,
        #       pitch=self.rvc_params["pitch"],
        #       f0method=self.rvc_params["f0method"],
        #       index_rate=self.rvc_params["index_rate"],
        #       rms_mix_rate=self.rvc_params["rms_mix_rate"],
        #       ...
        #   )
        await asyncio.sleep(0.01)  # Simulate inference time

        latency_ms = int((time.perf_counter() - t0) * 1000)
        self.latency_history.append(latency_ms)

        return audio_data

    async def get_stats(self) -> Dict:
        """Get processing statistics"""
        if self.latency_history:
            avg_latency = sum(self.latency_history) / len(self.latency_history)
            max_latency = max(self.latency_history)
            min_latency = min(self.latency_history)
        else:
            avg_latency = max_latency = min_latency = 0

        return {
            "avg_latency_ms": round(avg_latency, 1),
            "max_latency_ms": max_latency,
            "min_latency_ms": min_latency,
            "model": self.current_model.name if self.current_model else None,
            "model_sr": self.current_model.tgt_sr if self.current_model else None,
        }


class StreamingConnection:
    """Manages a single WebSocket streaming connection"""

    def __init__(self, websocket, processor: AudioProcessor):
        self.websocket = websocket
        self.processor = processor
        self.config = StreamConfig()
        self.running = False
        self.sequence = 0

    async def send_config(self):
        """Send initial configuration to client"""
        await self.websocket.send_text(
            json.dumps(
                {
                    "type": "config",
                    "sample_rate": self.config.sample_rate,
                    "chunk_size": self.config.chunk_size,
                    "channels": self.config.channels,
                    "latency_target_ms": self.config.latency_target_ms,
                }
            )
        )

    async def handle_message(self, message: bytes):
        """Handle incoming binary audio message"""
        try:
            data = json.loads(message.decode("utf-8"))
            msg_type = data.get("type")

            if msg_type == "config":
                self.config.sample_rate = data.get("sample_rate", DEFAULT_SAMPLE_RATE)
                self.config.chunk_size = data.get("chunk_size", DEFAULT_CHUNK_SIZE)
                self.config.latency_target_ms = data.get("latency_target_ms", 100)
                logger.info(
                    f"Client config: SR={self.config.sample_rate}, chunk={self.config.chunk_size}"
                )

            elif msg_type == "model":
                pth_path = data.get("pth_path", "")
                index_path = data.get("index_path", "")
                index_rate = float(data.get("index_rate", 0.0))

                if pth_path:
                    await self.processor.set_model(pth_path, index_path, index_rate)

                    # Client may also send RVC params with the model message
                    rvc_overrides = {
                        k: data[k]
                        for k in AudioProcessor._RVC_DEFAULTS
                        if k in data
                    }
                    if rvc_overrides:
                        self.processor.update_rvc_params(rvc_overrides)

                    await self.websocket.send_text(
                        json.dumps(
                            {
                                "type": "model_loaded",
                                "model": os.path.basename(pth_path),
                                "rvc_params": self.processor.rvc_params,
                            }
                        )
                    )

            elif msg_type == "audio":
                audio_bytes = data.get("data")
                if audio_bytes:
                    import base64

                    audio_b64 = (
                        audio_bytes.encode("utf-8")
                        if isinstance(audio_bytes, str)
                        else audio_bytes
                    )
                    audio_data = np.frombuffer(
                        base64.b64decode(audio_b64), dtype=np.float32
                    )

                    output = await self.processor.process_chunk(audio_data)

                    self.sequence += 1
                    stats = await self.processor.get_stats()

                    response = {
                        "type": "audio",
                        "seq": self.sequence,
                        "latency_ms": stats.get("avg_latency_ms", 0),
                        "data": base64.b64encode(output.tobytes()).decode("utf-8"),
                    }
                    await self.websocket.send_text(json.dumps(response))

            elif msg_type == "stats":
                stats = await self.processor.get_stats()
                await self.websocket.send_text(
                    json.dumps({"type": "stats_response", **stats})
                )

            elif msg_type == "ping":
                await self.websocket.send_text(
                    json.dumps({"type": "pong", "timestamp": data.get("timestamp", 0)})
                )

        except json.JSONDecodeError:
            logger.error("Invalid JSON message")
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            traceback.print_exc()

    async def run(self):
        """Main connection loop"""
        self.running = True
        await self.send_config()

        try:
            while self.running:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.receive_text(), timeout=30.0
                    )
                    await self.handle_message(message.encode("utf-8"))

                except asyncio.TimeoutError:
                    try:
                        await self.websocket.send_text(json.dumps({"type": "keepalive"}))
                    except:
                        break

        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            self.running = False


class RVCStreamingServer:
    """RVC WebSocket Streaming Server"""

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.model_manager = ModelManager()
        self.processor = AudioProcessor(self.model_manager)
        self.connections = set()

        # Setup FastAPI (lazy import)
        self.app = None
        self._setup_routes()

        self.running = False

    def _setup_routes(self):
        """Setup FastAPI routes"""
        try:
            from fastapi import FastAPI, WebSocket
            from fastapi import HTTPException
        except ImportError:
            logger.error(
                "FastAPI not installed. Install with: pip install fastapi uvicorn"
            )
            return

        self.app = FastAPI(title="RVC Streaming Server")

        @self.app.get("/")
        async def root():
            return {"service": "RVC Streaming Server", "status": "running"}

        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}

        @self.app.websocket("/rvc")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            conn = StreamingConnection(websocket, self.processor)
            self.connections.add(conn)
            try:
                await conn.run()
            finally:
                self.connections.discard(conn)

    async def broadcast_stats(self):
        """Broadcast stats to all connections"""
        stats = await self.processor.get_stats()
        message = json.dumps({"type": "broadcast_stats", **stats})
        for conn in self.connections:
            try:
                await conn.websocket.send_text(message)
            except:
                pass

    def run(self, threaded: bool = False):
        """Run the server"""
        if self.app is None:
            logger.error("FastAPI not available. Install dependencies first.")
            return

        try:
            import uvicorn
        except ImportError:
            logger.error("uvicorn not installed. Install with: pip install uvicorn")
            return

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False,
            # wsproto backend skips Origin header checking, which allows
            # connections from Electron (file:// origin) and other non-browser clients.
            ws="wsproto",
        )
        server = uvicorn.Server(config)

        logger.info(f"Starting RVC Streaming Server on {self.host}:{self.port}")
        logger.info(f"WebSocket endpoint: ws://{self.host}:{self.port}/rvc")

        if threaded:
            import threading

            thread = threading.Thread(target=lambda: server.run(), daemon=True)
            thread.start()
            return thread
        else:
            asyncio.run(server.serve())


def main():
    parser = argparse.ArgumentParser(description="RVC WebSocket Streaming Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to bind to"
    )
    parser.add_argument(
        "--max-cache", type=int, default=MODEL_CACHE_SIZE, help="Max cached models"
    )
    args = parser.parse_args()

    server = RVCStreamingServer(host=args.host, port=args.port)
    server.model_manager.max_cache = args.max_cache
    server.run()


if __name__ == "__main__":
    main()
