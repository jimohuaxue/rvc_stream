#!/bin/bash
# RVC Stream - Docker Build Script
# Builds both server and client images

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "RVC Stream Docker Build"
echo "========================================="

cd "$PROJECT_ROOT"

# Build server image
echo ""
echo "[1/2] Building RVC Server image (GPU-enabled)..."
docker build \
    -f Dockerfile.server \
    -t rvc-stream-server:latest \
    -t rvc-stream-server:v0.1.0 \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    .

if [ $? -eq 0 ]; then
    echo "✓ Server image built successfully"
    docker images | grep rvc-stream-server
else
    echo "✗ Server image build failed"
    exit 1
fi

# Build client image
echo ""
echo "[2/2] Building RVC Client image (minimal)..."
docker build \
    -f Dockerfile.client \
    -t rvc-stream-client:latest \
    -t rvc-stream-client:v0.1.0 \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    .

if [ $? -eq 0 ]; then
    echo "✓ Client image built successfully"
    docker images | grep rvc-stream-client
else
    echo "✗ Client image build failed"
    exit 1
fi

echo ""
echo "========================================="
echo "Build Complete!"
echo "========================================="
echo ""
echo "Images created:"
docker images | grep rvc-stream
echo ""
echo "Next steps:"
echo "  - Run server: docker-compose up rvc-server"
echo "  - Run client: docker-compose up rvc-client"
echo "  - Or use scripts: ./scripts/docker-run-server.sh"