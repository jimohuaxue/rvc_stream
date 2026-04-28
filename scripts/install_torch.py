#!/usr/bin/env python3
"""Install a PyTorch build matching the local CUDA runtime.

Usage:
    python scripts/install_torch.py           # auto-detect CUDA
    python scripts/install_torch.py --auto    # same, explicit
    python scripts/install_torch.py --variant cu128
    python scripts/install_torch.py --variant cpu
    python scripts/install_torch.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Variant catalogue
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "cpu": {
        "index_url": "https://download.pytorch.org/whl/cpu",
        "note": "CPU-only — CI / machines without NVIDIA GPU.",
    },
    "cu121": {
        "index_url": "https://download.pytorch.org/whl/cu121",
        "note": "CUDA 12.1 — GTX/RTX 10/20/30 series.",
    },
    "cu124": {
        "index_url": "https://download.pytorch.org/whl/cu124",
        "note": "CUDA 12.4 — RTX 40 series, safe default for CUDA 12.x.",
    },
    "cu126": {
        "index_url": "https://download.pytorch.org/whl/cu126",
        "note": "CUDA 12.6 — RTX 40/50 series.",
    },
    "cu128": {
        "index_url": "https://download.pytorch.org/whl/cu128",
        "note": "CUDA 12.8 — RTX 50 series (Blackwell) and latest drivers.",
    },
}

PACKAGES = ["torch", "torchvision", "torchaudio"]


# ---------------------------------------------------------------------------
# CUDA detection
# ---------------------------------------------------------------------------

def _nvidia_smi_cuda() -> Optional[str]:
    """Read CUDA version from nvidia-smi output."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=8
        ).stdout
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    except Exception:
        pass
    return None


def _nvcc_cuda() -> Optional[str]:
    """Read CUDA version from nvcc."""
    if not shutil.which("nvcc"):
        return None
    try:
        out = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=8
        ).stdout
        m = re.search(r"release\s+(\d+)\.(\d+)", out)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    except Exception:
        pass
    return None


def detect_cuda() -> Optional[str]:
    return _nvidia_smi_cuda() or _nvcc_cuda()


def cuda_to_variant(cuda: str) -> str:
    """Map a 'major.minor' CUDA string to the best available variant."""
    try:
        major, minor = (int(x) for x in cuda.split(".")[:2])
        v = major * 100 + minor
    except ValueError:
        return "cpu"

    if v >= 1208:
        return "cu128"
    if v >= 1206:
        return "cu126"
    if v >= 1200:
        return "cu124"
    if v >= 1106:
        return "cu121"
    # CUDA < 11.6 is too old for current torch; fall back to CPU wheels
    return "cpu"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install torch/torchvision/torchaudio for the local CUDA runtime."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        help="Force a specific wheel variant.",
    )
    group.add_argument(
        "--auto",
        action="store_true",
        default=True,
        help="Auto-detect CUDA version (default).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pip command without executing it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.variant:
        variant = args.variant
        print(f"[install_torch] variant forced: {variant}")
    else:
        cuda = detect_cuda()
        if cuda:
            variant = cuda_to_variant(cuda)
            print(f"[install_torch] detected CUDA {cuda} → variant '{variant}'")
        else:
            variant = "cpu"
            print("[install_torch] no CUDA detected → falling back to CPU wheels")

    info = VARIANTS[variant]
    print(f"[install_torch] {info['note']}")

    cmd = [
        sys.executable, "-m", "pip", "install", "--upgrade",
        *PACKAGES,
        "--index-url", info["index_url"],
    ]
    print("[install_torch]", " ".join(cmd))

    if args.dry_run:
        return 0

    rc = subprocess.run(cmd, check=False).returncode
    if rc == 0:
        # Show what was actually installed
        subprocess.run(
            [sys.executable, "-c",
             "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"],
            check=False,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
