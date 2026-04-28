#!/usr/bin/env python3
"""
RVC WebSocket Streaming Client
Real-time voice conversion client with virtual audio device output

Architecture:
    [Microphone] --> [sounddevice InputStream] --> [WebSocket Client] --> [Server]
    
    [Server] --> [WebSocket Client] --> [sounddevice OutputStream] --> [Virtual Mic]

Requirements:
    - sounddevice (for audio I/O)
    - websockets (for server communication)
    - numpy (for audio processing)

Virtual Audio Device Options:
    1. Linux: PulseAudio/JACK virtual source or ALSA loopback
    2. macOS: BlackHole virtual audio driver
    3. Windows: Virtual Audio Cable, VB-Audio

Usage:
    # Connect to server and use default devices
    python rvc_client.py --server ws://localhost:8080/rvc --model model.pth
    
    # With custom devices
    python rvc_client.py --server ws://localhost:8080/rvc \
        --input-device "MacBook Pro Microphone" \
        --output-device "BlackHole 2ch" \
        --model model.pth
"""

import os
import sys
import json
import asyncio
import argparse
import logging
import threading
import time
import traceback
import base64
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ClientState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass
class StreamConfig:
    """Audio streaming configuration"""

    sample_rate: int = 16000
    chunk_size: int = 512  # samples per chunk
    channels: int = 1
    latency_target_ms: int = 100


class AudioDevice:
    """Audio device information"""

    def __init__(
        self, index: int, name: str, hostapi: str, channels: int, sample_rate: float
    ):
        self.index = index
        self.name = name
        self.hostapi = hostapi
        self.channels = channels
        self.sample_rate = sample_rate

    def __str__(self):
        return f"[{self.index}] {self.name} ({self.hostapi})"


class AudioDeviceManager:
    """Manages audio devices using sounddevice"""

    @staticmethod
    def list_devices() -> Dict[str, list]:
        """List all available audio devices"""
        import sounddevice as sd

        devices = {"input": [], "output": []}
        info = sd.query_devices()

        hostapis = sd.query_hostapis()

        for i, dev in enumerate(info):
            hostapi_idx = dev.get("hostapi", 0)
            hostapi_name = (
                hostapis[hostapi_idx]["name"]
                if hostapi_idx < len(hostapis)
                else "Unknown"
            )

            device = AudioDevice(
                index=i,
                name=dev.get("name", f"Device {i}"),
                hostapi=hostapi_name,
                channels=dev.get(
                    "max_input_channels"
                    if dev.get("max_input_channels", 0) > 0
                    else "max_output_channels",
                    0,
                ),
                sample_rate=dev.get("default_samplerate", 44100),
            )

            if dev.get("max_input_channels", 0) > 0:
                devices["input"].append(device)
            if dev.get("max_output_channels", 0) > 0:
                devices["output"].append(device)

        return devices

    @staticmethod
    def print_devices():
        """Print all available audio devices"""
        devices = AudioDeviceManager.list_devices()

        print("\n=== Available Input Devices ===")
        for d in devices["input"]:
            print(f"  {d}")

        print("\n=== Available Output Devices ===")
        for d in devices["output"]:
            print(f"  {d}")
        print()


class RingBuffer:
    """Thread-safe ring buffer for audio data"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        """Write data to buffer, return samples written"""
        with self.lock:
            n = min(len(data), self.capacity)

            end_pos = (self.write_pos + n) % self.capacity

            if end_pos > self.write_pos:
                self.buffer[self.write_pos : end_pos] = data[:n]
            else:
                first_part = min(n, self.capacity - self.write_pos)
                self.buffer[self.write_pos :] = data[:first_part]
                if n > first_part:
                    self.buffer[: n - first_part] = data[first_part:n]

            self.write_pos = end_pos
            return n

    def read(self, n: int) -> np.ndarray:
        """Read n samples from buffer"""
        with self.lock:
            if self.write_pos >= self.read_pos:
                available = self.write_pos - self.read_pos
            else:
                available = self.capacity - self.read_pos + self.write_pos

            n = min(n, available)
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
            return data

    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.read_pos = 0
            self.write_pos = 0


class VirtualAudioOutput:
    """Handles virtual audio output for different platforms"""

    @staticmethod
    def get_platform_output_name(target_name: str = None) -> Optional[str]:
        """Get the appropriate virtual audio device name for the platform"""
        import sounddevice as sd

        devices = sd.query_devices()

        if sys.platform == "darwin":
            candidates = ["BlackHole", "Aggregate", "Loopback", "Virtual"]
        elif sys.platform == "linux":
            candidates = ["Virtual", "Loopback", "Monitor", "null"]
        elif sys.platform == "win32":
            candidates = ["Virtual", "CABLE", "VB-Audio", "VoiceMeeter"]
        else:
            return None

        for dev in devices:
            name = dev.get("name", "").lower()
            for candidate in candidates:
                if candidate.lower() in name:
                    return dev["name"]

        return None


class RVCStreamClient:
    """WebSocket client for RVC streaming"""

    def __init__(
        self,
        server_url: str,
        model_path: str = "",
        index_path: str = "",
        index_rate: float = 0.0,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_size: int = 512,
    ):
        self.server_url = server_url
        self.model_path = model_path
        self.index_path = index_path
        self.index_rate = index_rate

        self.config = StreamConfig(sample_rate=sample_rate, chunk_size=chunk_size)

        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate

        # Buffers
        self.input_buffer = RingBuffer(sample_rate * 3)  # 3 seconds
        self.output_buffer = RingBuffer(sample_rate * 3)

        # State
        self.state = ClientState.DISCONNECTED
        self.websocket = None
        self.input_stream = None
        self.output_stream = None

        # Statistics
        self.bytes_sent = 0
        self.bytes_received = 0
        self.latency_ms = 0
        self.frames_processed = 0
        self.start_time = 0

        # Threading
        self.running = False
        self.recv_thread: Optional[threading.Thread] = None

    async def connect(self) -> bool:
        """Connect to the RVC streaming server"""
        import websockets

        try:
            self.state = ClientState.CONNECTING
            logger.info(f"Connecting to {self.server_url}...")

            self.websocket = await websockets.connect(
                self.server_url, ping_interval=30, ping_timeout=10
            )

            config_msg = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
            config_data = json.loads(config_msg)

            if config_data.get("type") == "config":
                self.config.sample_rate = config_data.get("sample_rate", 16000)
                self.config.chunk_size = config_data.get("chunk_size", 512)
                self.config.latency_target_ms = config_data.get(
                    "latency_target_ms", 100
                )
                logger.info(
                    f"Server config: SR={self.config.sample_rate}, chunk={self.config.chunk_size}"
                )

            if self.model_path:
                await self.load_model(self.model_path, self.index_path, self.index_rate)

            self.state = ClientState.CONNECTED
            logger.info("Connected to server")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.state = ClientState.ERROR
            return False

    async def load_model(
        self, pth_path: str, index_path: str = "", index_rate: float = 0.0
    ):
        """Request server to load a model"""
        if not self.websocket:
            return

        await self.websocket.send(
            json.dumps(
                {
                    "type": "model",
                    "pth_path": pth_path,
                    "index_path": index_path,
                    "index_rate": index_rate,
                }
            )
        )

        try:
            response = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)
            data = json.loads(response)
            if data.get("type") == "model_loaded":
                logger.info(f"Model loaded: {data.get('model')}")
        except asyncio.TimeoutError:
            logger.warning("Model loading timeout")

    def _audio_input_callback(self, indata, frames, times, status):
        """Callback for audio input stream"""
        if status:
            if status.input_overflow:
                logger.warning("Input overflow")
            elif status.input_underflow:
                logger.warning("Input underflow")

        if indata.shape[1] > 1:
            audio = np.mean(indata, axis=1)
        else:
            audio = indata[:, 0]

        audio = audio.astype(np.float32)
        self.input_buffer.write(audio)

    def _audio_output_callback(self, outdata, frames, times, status):
        """Callback for audio output stream"""
        if status:
            if status.output_underflow:
                logger.warning("Output underflow")
            elif status.output_overflow:
                logger.warning("Output overflow")

        audio = self.output_buffer.read(frames)

        if len(audio) < frames:
            audio = np.pad(audio, (0, frames - len(audio)))

        if outdata.shape[1] > 1:
            outdata[:, 0] = audio
            outdata[:, 1] = audio
        else:
            outdata[:, 0] = audio

    def _start_audio_streams(self):
        """Start sounddevice input and output streams"""
        import sounddevice as sd

        try:
            self.input_stream = sd.InputStream(
                device=self.input_device,
                channels=1,
                samplerate=self.sample_rate,
                dtype="float32",
                blocksize=self.config.chunk_size,
                callback=self._audio_input_callback,
            )

            self.output_stream = sd.OutputStream(
                device=self.output_device,
                channels=1,
                samplerate=self.sample_rate,
                dtype="float32",
                blocksize=self.config.chunk_size,
                callback=self._audio_output_callback,
            )

            self.input_stream.start()
            self.output_stream.start()

            logger.info("Audio streams started")

        except Exception as e:
            logger.error(f"Failed to start audio streams: {e}")
            raise

    async def _send_loop(self):
        """Send audio chunks to server"""
        while self.running and self.websocket:
            try:
                audio = self.input_buffer.read(self.config.chunk_size)

                if len(audio) < self.config.chunk_size:
                    await asyncio.sleep(0.01)
                    continue

                audio_b64 = base64.b64encode(audio.tobytes()).decode("utf-8")

                await self.websocket.send(
                    json.dumps(
                        {"type": "audio", "data": audio_b64, "timestamp": time.time()}
                    )
                )

                self.bytes_sent += len(audio) * 4  # float32 = 4 bytes

            except Exception as e:
                logger.error(f"Send error: {e}")
                break

    async def _recv_loop(self):
        """Receive processed audio from server"""
        while self.running and self.websocket:
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)

                if isinstance(message, str):
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "audio":
                        audio_b64 = data.get("data", "")
                        audio_bytes = base64.b64decode(audio_b64)
                        audio = np.frombuffer(audio_bytes, dtype=np.float32)

                        self.output_buffer.write(audio)

                        self.bytes_received += len(audio) * 4
                        self.latency_ms = data.get("latency_ms", 0)
                        self.frames_processed += 1

                    elif msg_type == "keepalive":
                        pass

                    elif msg_type == "pong":
                        rtt = (time.time() - data.get("timestamp", 0)) * 1000
                        logger.debug(f"RTT: {rtt:.1f}ms")

            except asyncio.TimeoutError:
                try:
                    await self.websocket.send(
                        json.dumps({"type": "ping", "timestamp": time.time()})
                    )
                except:
                    break
            except Exception as e:
                logger.error(f"Recv error: {e}")
                break

    async def stream(self):
        """Start streaming audio"""
        if self.state != ClientState.CONNECTED:
            logger.error("Not connected")
            return False

        try:
            self.running = True
            self.state = ClientState.STREAMING
            self.start_time = time.time()

            audio_thread = threading.Thread(target=self._start_audio_streams)
            audio_thread.start()

            await asyncio.sleep(0.5)

            logger.info("Streaming started")

            await asyncio.gather(self._send_loop(), self._recv_loop())

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            self.state = ClientState.ERROR
            return False
        finally:
            self.running = False
            self._stop_audio_streams()

    def _stop_audio_streams(self):
        """Stop audio streams"""
        try:
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
                self.input_stream = None

            if self.output_stream:
                self.output_stream.stop()
                self.output_stream.close()
                self.output_stream = None

            logger.info("Audio streams stopped")
        except Exception as e:
            logger.error(f"Error stopping audio streams: {e}")

    async def disconnect(self):
        """Disconnect from server"""
        self.running = False

        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None

        self._stop_audio_streams()
        self.state = ClientState.DISCONNECTED

        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info(
            f"Session stats: {elapsed:.1f}s, {self.frames_processed} frames, "
            f"{self.bytes_sent / 1024 / 1024:.1f}MB sent, {self.bytes_received / 1024 / 1024:.1f}MB received"
        )

    def get_stats(self) -> Dict:
        """Get current statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 1
        return {
            "state": self.state.value,
            "latency_ms": self.latency_ms,
            "frames_processed": self.frames_processed,
            "fps": self.frames_processed / elapsed if elapsed > 0 else 0,
            "bytes_sent_mb": self.bytes_sent / 1024 / 1024,
            "bytes_received_mb": self.bytes_received / 1024 / 1024,
        }


async def run_client(client: RVCStreamClient):
    """Run the client"""
    try:
        if not await client.connect():
            return

        await client.stream()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Client error: {e}")
        traceback.print_exc()
    finally:
        await client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="RVC WebSocket Streaming Client")

    # Server configuration
    parser.add_argument(
        "--server", "-s", default="ws://localhost:8080/rvc", help="WebSocket server URL"
    )

    # Model configuration
    parser.add_argument(
        "--model", "-m", default="", help="Path to .pth model file (on server)"
    )
    parser.add_argument(
        "--index", "-i", default="", help="Path to .index file (on server)"
    )
    parser.add_argument(
        "--index-rate", type=float, default=0.0, help="Index search rate (0.0-1.0)"
    )

    # Audio configuration
    parser.add_argument(
        "--input-device",
        "-d",
        type=int,
        default=None,
        help="Input device index (microphone)",
    )
    parser.add_argument(
        "--output-device",
        "-o",
        type=int,
        default=None,
        help="Output device index (virtual mic)",
    )
    parser.add_argument(
        "--sample-rate", "-r", type=int, default=16000, help="Audio sample rate"
    )
    parser.add_argument(
        "--chunk-size", "-c", type=int, default=512, help="Audio chunk size in samples"
    )

    # List devices mode
    parser.add_argument(
        "--list-devices",
        "-l",
        action="store_true",
        help="List available audio devices and exit",
    )

    args = parser.parse_args()

    if args.list_devices:
        AudioDeviceManager.print_devices()
        return

    if not args.model:
        logger.warning("No model specified. Server will use default or loaded model.")

    client = RVCStreamClient(
        server_url=args.server,
        model_path=args.model,
        index_path=args.index,
        index_rate=args.index_rate,
        input_device=args.input_device,
        output_device=args.output_device,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
    )

    asyncio.run(run_client(client))


if __name__ == "__main__":
    main()
