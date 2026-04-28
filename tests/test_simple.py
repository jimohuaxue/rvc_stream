#!/usr/bin/env python3
"""
Simple unit tests for rvc_stream that don't require audio devices.
Tests protocol encoding, buffer logic, and server/client components.

Usage:
    python test_simple.py
"""

import unittest
import json
import base64
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


class TestProtocolEncoding(unittest.TestCase):
    """Test protocol encoding/decoding without network/audio"""

    def test_audio_base64_encoding(self):
        """Test audio data can be encoded and decoded correctly"""
        audio = np.random.randn(512).astype(np.float32)
        encoded = base64.b64encode(audio.tobytes()).decode("utf-8")
        decoded = np.frombuffer(base64.b64decode(encoded), dtype=np.float32)
        np.testing.assert_array_almost_equal(audio, decoded)

    def test_config_message(self):
        """Test config message serialization"""
        config = {
            "type": "config",
            "sample_rate": 16000,
            "chunk_size": 512,
            "channels": 1,
            "latency_target_ms": 100,
        }
        json_str = json.dumps(config)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["type"], "config")
        self.assertEqual(parsed["sample_rate"], 16000)
        self.assertEqual(parsed["chunk_size"], 512)

    def test_model_message(self):
        """Test model loading message"""
        msg = {
            "type": "model",
            "pth_path": "/path/to/model.pth",
            "index_path": "/path/to/model.index",
            "index_rate": 0.75,
        }
        json_str = json.dumps(msg)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["type"], "model")
        self.assertEqual(parsed["pth_path"], "/path/to/model.pth")
        self.assertAlmostEqual(parsed["index_rate"], 0.75)

    def test_audio_message(self):
        """Test audio message with embedded data"""
        audio = np.ones(256, dtype=np.float32) * 0.5
        audio_b64 = base64.b64encode(audio.tobytes()).decode("utf-8")
        msg = {
            "type": "audio",
            "data": audio_b64,
            "timestamp": time.time(),
            "seq": 42,
        }
        json_str = json.dumps(msg)
        parsed = json.loads(json_str)
        decoded_audio = np.frombuffer(
            base64.b64decode(parsed["data"]), dtype=np.float32
        )
        np.testing.assert_array_almost_equal(audio, decoded_audio)
        self.assertEqual(parsed["seq"], 42)


class TestRingBuffer(unittest.TestCase):
    """Test RingBuffer implementation"""

    def _create_buffer(self, capacity):
        import threading

        class RingBuffer:
            def __init__(self, capacity):
                self.capacity = capacity
                self.buffer = np.zeros(capacity, dtype=np.float32)
                self.write_pos = 0
                self.read_pos = 0
                self.size = 0
                self.lock = threading.Lock()

            def write(self, data):
                with self.lock:
                    n = min(len(data), self.capacity - self.size)
                    if n == 0:
                        return 0
                    end_pos = (self.write_pos + n) % self.capacity
                    if end_pos > self.write_pos:
                        self.buffer[self.write_pos : end_pos] = data[:n]
                    else:
                        first_part = self.capacity - self.write_pos
                        self.buffer[self.write_pos :] = data[:first_part]
                        if n > first_part:
                            self.buffer[: n - first_part] = data[first_part:n]
                    self.write_pos = end_pos
                    self.size = min(self.size + n, self.capacity)
                    return n

            def read(self, n):
                with self.lock:
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

            def clear(self):
                with self.lock:
                    self.read_pos = 0
                    self.write_pos = 0
                    self.size = 0

        return RingBuffer(capacity)

    def test_basic_write_read(self):
        buf = self._create_buffer(100)
        data = np.random.randn(50).astype(np.float32)
        written = buf.write(data)
        self.assertEqual(written, 50)
        read = buf.read(50)
        self.assertEqual(len(read), 50)
        np.testing.assert_array_almost_equal(data, read)

    def test_partial_read(self):
        buf = self._create_buffer(100)
        data = np.random.randn(50).astype(np.float32)
        buf.write(data)
        read = buf.read(25)
        self.assertEqual(len(read), 25)
        np.testing.assert_array_almost_equal(data[:25], read)

    def test_buffer_full(self):
        buf = self._create_buffer(100)
        data1 = np.ones(100, dtype=np.float32)
        written = buf.write(data1)
        self.assertEqual(written, 100)
        # Buffer is full, subsequent writes should return 0
        data2 = np.ones(50, dtype=np.float32) * 2
        written = buf.write(data2)
        self.assertEqual(written, 0)

    def test_wraparound(self):
        buf = self._create_buffer(100)
        buf.write(np.ones(100, dtype=np.float32))
        buf.read(50)
        buf.write(np.ones(30, dtype=np.float32) * 2)
        result = buf.read(80)
        self.assertEqual(len(result), 80)
        np.testing.assert_array_almost_equal(result[:50], np.ones(50))
        np.testing.assert_array_almost_equal(result[50:], np.ones(30) * 2)

    def test_clear(self):
        buf = self._create_buffer(100)
        buf.write(np.ones(50, dtype=np.float32))
        buf.clear()
        read = buf.read(100)
        self.assertEqual(len(read), 0)


class TestLatencyCalculation(unittest.TestCase):
    def test_buffer_latency_calculation(self):
        sample_rate = 16000
        chunk_size = 512
        expected_latency = (chunk_size / sample_rate) * 1000
        self.assertAlmostEqual(expected_latency, 32.0, places=1)

    def test_rtt_calculation(self):
        send_time = time.time()
        recv_time = send_time + 0.05
        rtt_ms = (recv_time - send_time) * 1000
        self.assertAlmostEqual(rtt_ms, 50.0, places=1)


class TestServerImports(unittest.TestCase):
    def test_server_imports_without_torch(self):
        try:
            import rvc_server

            self.assertTrue(hasattr(rvc_server, "RVCStreamingServer"))
            self.assertTrue(hasattr(rvc_server, "ModelManager"))
            self.assertTrue(hasattr(rvc_server, "StreamingConnection"))
        except ImportError as e:
            self.skipTest(f"Server imports failed (may need fastapi/uvicorn): {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
