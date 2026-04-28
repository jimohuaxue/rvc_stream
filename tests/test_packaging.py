from __future__ import annotations

from importlib import util


def test_src_package_is_importable() -> None:
    assert util.find_spec("src") is not None


def test_client_module_is_importable() -> None:
    assert util.find_spec("src.rvc_client") is not None


def test_server_module_is_importable() -> None:
    assert util.find_spec("src.rvc_server") is not None
