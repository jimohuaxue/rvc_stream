#!/bin/bash
# RVC Stream - Run Server Container
# Starts the RVC server with GPU support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "RVC Stream Server Startup"
echo "========================================="

cd "$PROJECT_ROOT"

RVC_RUNTIME_DIR="${RVC_RUNTIME_DIR:-$PROJECT_ROOT/runtime}"
MODELS_DIR="${RVC_MODELS_DIR:-$RVC_RUNTIME_DIR/models}"
LOGS_DIR="${RVC_LOGS_DIR:-$RVC_RUNTIME_DIR/logs}"
PORT="${RVC_SERVER_PORT:-8080}"

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "Warning: Models directory not found at $MODELS_DIR"
    echo "Creating directory..."
    mkdir -pv "$MODELS_DIR"
fi

# Check if logs directory exists
if [ ! -d "$LOGS_DIR" ]; then
    echo "Creating logs directory..."
    mkdir -pv "$LOGS_DIR"
fi

echo ""
echo "Starting RVC Server..."
echo "  - Port: $PORT"
echo "  - Models: $MODELS_DIR"
echo "  - Logs: $LOGS_DIR"
echo ""

docker run -d \
    --name rvc-server \
    --gpus all \
    -p "$PORT:8080" \
    -v "$MODELS_DIR:/app/assets/weights:ro" \
    -v "$LOGS_DIR:/app/logs" \
    -e PYTHONUNBUFFERED=1 \
    -e CUDA_VISIBLE_DEVICES=0 \
    rvc-stream-server:latest

echo ""
echo "Server started!"
echo ""
echo "Check logs: docker logs -f rvc-server"
echo "Test health: curl http://localhost:$PORT/health"
echo "WebSocket endpoint: ws://localhost:$PORT/rvc"
