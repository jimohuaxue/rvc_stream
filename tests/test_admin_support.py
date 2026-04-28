from __future__ import annotations

from pathlib import Path

from src.admin.config_manager import ConfigManager
from src.admin.models import StreamingConfig
from src.admin.utils import format_bytes


def test_config_manager_accepts_model_objects(tmp_path: Path) -> None:
    config_path = tmp_path / "server.yaml"
    manager = ConfigManager(str(config_path))

    manager.update(streaming=StreamingConfig(sample_rate=22050, chunk_size=512))

    assert manager.get()["streaming"]["sample_rate"] == 22050
    assert config_path.exists()


def test_format_bytes_scales_units() -> None:
    assert format_bytes(512) == "512.00 B"
    assert format_bytes(2048) == "2.00 KB"
