#!/usr/bin/env python3
"""Setup script for rvc-stream package"""

from setuptools import setup, find_packages

setup(
    name="rvc-stream",
    version="0.1.0",
    description="WebSocket-based real-time voice conversion streaming",
    author="RVC Project",
    python_requires=">=3.10",
    packages=find_packages(where=".", include=["src*"]),
    install_requires=[
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "websockets>=11.0",
    ],
    extras_require={
        "client": [
            "sounddevice>=0.4.5",
        ],
        "server": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "soundfile>=0.12.1",
            "librosa>=0.10.0",
        ],
        "admin": [
            "psutil>=5.9.0",
            "pydantic>=2.9.0",
        ],
        "rvc": [
            "fairseq>=0.12.2",
            "faiss-cpu>=1.7.3",
        ],
        "full": [
            "sounddevice>=0.4.5",
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "soundfile>=0.12.1",
            "librosa>=0.10.0",
            "psutil>=5.9.0",
            "pydantic>=2.9.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rvc-server=src.rvc_server:main",
            "rvc-client=src.rvc_client:main",
        ],
    },
)
