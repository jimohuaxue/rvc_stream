#!/bin/bash
# RVC Stream - Run Client Container
# Starts the RVC client (requires running server)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "RVC Stream Client Startup"
echo "========================================="

cd "$PROJECT_ROOT"

RVC_RUNTIME_DIR="${RVC_RUNTIME_DIR:-$PROJECT_ROOT/runtime}"
SHARED_DIR="${RVC_SHARED_DIR:-$RVC_RUNTIME_DIR/shared}"

# Check if server is running
if ! docker ps | grep -q rvc-server; then
    echo "Warning: Server container not running"
    echo "Start it first with: ./scripts/docker-run-server.sh"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for audio devices (Linux)
if [ -d "/dev/snd" ]; then
    AUDIO_DEVICE="--device /dev/snd:/dev/snd"
    echo "Audio devices found at /dev/snd"
else
    AUDIO_DEVICE=""
    echo "No audio devices found, running in test mode only"
fi

# Server URL (default to localhost)
SERVER_URL="${RVC_SERVER_URL:-ws://localhost:8080/rvc}"

echo ""
echo "Starting RVC Client..."
echo "  - Server: $SERVER_URL"
echo "  - Shared dir: $SHARED_DIR"
echo "  - Audio: $([ -n "$AUDIO_DEVICE" ] && echo "Enabled" || echo "Disabled")"
echo ""

mkdir -pv "$SHARED_DIR"

# Interactive mode - show help first
docker run --rm \
    --name rvc-client \
    $AUDIO_DEVICE \
    -v "$SHARED_DIR:/app/shared:ro" \
    -e RVC_SERVER_URL="$SERVER_URL" \
    -e PYTHONUNBUFFERED=1 \
    rvc-stream-client:latest \
    python3 -m src.rvc_client --help

echo ""
echo "To run the client with specific options:"
echo "  docker run --rm rvc-stream-client:latest \\"
echo "    python3 -m src.rvc_client --server ws://YOUR_SERVER:8080/rvc \\"
echo "    --list-devices"
