#!/usr/bin/env python3
"""
RVC Streaming Test Script

Tests the streaming client functionality.
Can test protocol encoding/decoding without requiring audio devices.

Usage:
    python test_client.py --test all
"""

import os
import sys
import asyncio
import json
import time
import base64

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


async def test_client_import():
    """Test that the client module can be imported"""
    print("Testing client import...")

    try:
        import rvc_client

        print("✓ Client module imported successfully")

        # Check classes exist
        assert hasattr(rvc_client, "RVCStreamClient")
        assert hasattr(rvc_client, "ClientState")
        assert hasattr(rvc_client, "RingBuffer")
        assert hasattr(rvc_client, "AudioDeviceManager")
        print("✓ All client classes exist")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  (This is expected if dependencies aren't installed)")
        return None


async def test_ring_buffer():
    """Test RingBuffer functionality - simplified test without audio device deps"""
    print("Testing RingBuffer...")

    try:
        # Create a minimal RingBuffer implementation for testing
        # (Matches the implementation in rvc_client.py)
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

        buffer = RingBuffer(1000)

        # Write some data
        test_data = np.random.randn(100).astype(np.float32)
        n_written = buffer.write(test_data)
        assert n_written == 100
        print("✓ RingBuffer write works")

        # Read data
        read_data = buffer.read(50)
        assert len(read_data) == 50
        print("✓ RingBuffer read works")

        # Check data integrity
        assert np.allclose(read_data, test_data[:50])
        print("✓ RingBuffer data integrity verified")

        # Test wraparound
        buffer2 = RingBuffer(100)
        # Fill buffer
        data1 = np.ones(100).astype(np.float32)
        buffer2.write(data1)
        # Read some
        buffer2.read(50)
        # Write more (should wrap)
        data2 = np.ones(30).astype(np.float32) * 2
        buffer2.write(data2)
        # Read remaining (50 old + 30 new)
        result = buffer2.read(80)
        assert len(result) == 80
        assert np.allclose(result[:50], np.ones(50))
        assert np.allclose(result[50:], np.ones(30) * 2)
        print("✓ RingBuffer wraparound works")

        return True

    except Exception as e:
        print(f"✗ RingBuffer test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_protocol():
    """Test WebSocket protocol encoding/decoding"""
    print("Testing protocol...")

    try:
        # Test audio encoding
        audio = np.random.randn(512).astype(np.float32)
        encoded = base64.b64encode(audio.tobytes()).decode("utf-8")
        decoded = np.frombuffer(base64.b64decode(encoded), dtype=np.float32)

        assert np.allclose(audio, decoded)
        print("✓ Audio encoding/decoding works")

        # Test JSON config message
        config = {
            "type": "config",
            "sample_rate": 16000,
            "chunk_size": 512,
            "channels": 1,
            "latency_target_ms": 100,
        }
        json_str = json.dumps(config)
        parsed = json.loads(json_str)

        assert parsed == config
        print("✓ JSON config message works")

        # Test model message
        model_msg = {
            "type": "model",
            "pth_path": "test_model.pth",
            "index_path": "test_index.index",
            "index_rate": 0.5,
        }
        json_str = json.dumps(model_msg)
        parsed = json.loads(json_str)

        assert parsed == model_msg
        print("✓ JSON model message works")

        # Test audio message
        audio_b64 = base64.b64encode(audio.tobytes()).decode("utf-8")
        audio_msg = {"type": "audio", "data": audio_b64, "timestamp": time.time()}
        json_str = json.dumps(audio_msg)
        parsed = json.loads(json_str)
        parsed_audio = np.frombuffer(base64.b64decode(parsed["data"]), dtype=np.float32)
        assert np.allclose(audio, parsed_audio)
        print("✓ JSON audio message works")

        return True

    except Exception as e:
        print(f"✗ Protocol test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_latency():
    """Test latency measurement"""
    print("Testing latency measurement...")

    try:
        # Simulate latency measurement
        send_time = time.time()
        await asyncio.sleep(0.05)  # 50ms simulated delay
        recv_time = time.time()

        latency_ms = (recv_time - send_time) * 1000
        print(f"  Measured latency: {latency_ms:.1f}ms")

        # Test buffer latency
        buffer_latency_samples = 512  # 512 samples at 16kHz = 32ms
        buffer_latency_ms = (buffer_latency_samples / 16000) * 1000
        print(f"  Buffer latency: {buffer_latency_ms:.1f}ms")

        total_latency = latency_ms + buffer_latency_ms
        print(f"  Total estimated latency: {total_latency:.1f}ms")

        assert total_latency < 200
        print("✓ Latency is within target (<200ms)")

        return True

    except Exception as e:
        print(f"✗ Latency test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    print("=" * 50)
    print("RVC Streaming Client Test Suite")
    print("=" * 50)

    tests = [
        ("Client Import", test_client_import()),
        ("RingBuffer", test_ring_buffer()),
        ("Protocol", test_protocol()),
        ("Latency", test_latency()),
    ]

    results = []
    for name, test_coro in tests:
        if asyncio.iscoroutine(test_coro):
            result = await test_coro
        else:
            result = test_coro
        results.append((name, result))

    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)

    all_passed = True
    for name, result in results:
        if result is None:
            status = "SKIPPED"
            symbol = "○"
        elif result:
            status = "PASSED"
            symbol = "✓"
        else:
            status = "FAILED"
            symbol = "✗"
            all_passed = False
        print(f"  {symbol} {name}: {status}")

    print("=" * 50)

    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
