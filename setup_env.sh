#!/bin/bash
# Environment setup script for rvc_stream

set -e

echo "=== RVC Stream Environment Setup ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CUDA_VARIANT="${CUDA_VARIANT:-cpu}"
INSTALL_RVC_EXTRAS="${INSTALL_RVC_EXTRAS:-0}"

# Try conda first
setup_conda() {
    if ! command -v conda &> /dev/null; then
        return 1
    fi
    
    echo "Creating conda environment 'rvcstream'..."
    if conda env create -f environment.yml 2>&1; then
        conda run -n rvcstream python scripts/install_torch.py --variant "$CUDA_VARIANT" || true
        echo ""
        echo "=== Conda Setup Complete ==="
        echo "To activate the environment, run:"
        echo "  conda activate rvcstream"
        return 0
    fi
    return 1
}

# Fallback to venv
setup_venv() {
    echo "Conda setup failed or not available. Trying Python venv..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    elif command -v python &> /dev/null; then
        PYTHON_CMD=python
    else
        echo "Error: Python not found"
        return 1
    fi
    
    echo "Creating Python venv at rvcstream_venv..."
    $PYTHON_CMD -m venv --system-site-packages rvcstream_venv
    
    echo "Installing packages..."
    source rvcstream_venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements-client.txt
    pip install -r requirements-server.txt
    pip install -r requirements-admin.txt
    pip install -r requirements-dev.txt
    if [ "$INSTALL_RVC_EXTRAS" = "1" ]; then
        pip install -r requirements-rvc.txt
    fi
    python scripts/install_torch.py --variant "$CUDA_VARIANT" || true
    
    echo ""
    echo "=== Venv Setup Complete ==="
    echo "To activate the environment, run:"
    echo "  source rvcstream_venv/bin/activate"
    return 0
}

# Try conda, fallback to venv
if ! setup_conda; then
    setup_venv
fi

echo ""
echo "=== Verification ==="
echo "Running tests..."
python -m pytest tests/ -v --tb=short || true

echo ""
echo "To run the client:"
echo "  python -m src.rvc_client --help"
echo ""
echo "To run the server (requires RVC dependencies):"
echo "  python -m src.rvc_server --help"
echo ""
echo "To install a CUDA-specific PyTorch build later:"
echo "  CUDA_VARIANT=cu124 ./setup_env.sh"
echo "  CUDA_VARIANT=cu126 ./setup_env.sh"
echo ""
echo "To opt into legacy RVC inference extras:"
echo "  INSTALL_RVC_EXTRAS=1 ./setup_env.sh"
