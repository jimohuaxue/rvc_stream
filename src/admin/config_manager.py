"""
Configuration manager with YAML persistence for admin panel
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from .models import StreamingConfig, ServerConfig, ModelConfig, RvcConfig


class ConfigManager:
    """Manages server configuration with YAML persistence"""

    def __init__(self, config_file: str = "configs/server.yaml"):
        """
        Initialize config manager

        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = Path(config_file)
        self.config_dir = self.config_file.parent

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Default configuration
        self._defaults = {
            "streaming": StreamingConfig().model_dump(),
            "server": ServerConfig().model_dump(),
            "rvc": RvcConfig().model_dump(),
            "models": [],
        }

        # Load or create configuration
        self._config = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    if config:
                        # Merge with defaults
                        merged = self._defaults.copy()
                        merged.update(config)
                        return merged
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_file}: {e}")

        # Create default config file
        self._save(self._defaults)
        return self._defaults.copy()

    def _save(self, config: Dict[str, Any]) -> None:
        """Save configuration to YAML file"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    config, f, default_flow_style=False, allow_unicode=True, indent=2
                )
        except Exception as e:
            print(f"Error: Failed to save config to {self.config_file}: {e}")
            raise

    def get(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self._config.copy()

    def get_streaming_config(self) -> StreamingConfig:
        """Get streaming configuration"""
        return StreamingConfig(**self._config.get("streaming", {}))

    def get_server_config(self) -> ServerConfig:
        """Get server configuration"""
        return ServerConfig(**self._config.get("server", {}))

    def get_models_config(self) -> list[ModelConfig]:
        """Get models configuration"""
        models_data = self._config.get("models", [])
        return [ModelConfig(**model) for model in models_data]

    def update(self, **kwargs) -> Dict[str, Any]:
        """
        Update configuration

        Args:
            **kwargs: Configuration key-value pairs to update
                      Examples:
                      - update(streaming=StreamingConfig(...))
                      - update(streaming__sample_rate=48000)

        Returns:
            Updated configuration dict
        """
        # Handle nested updates (e.g., streaming__sample_rate)
        for key, value in kwargs.items():
            if "__" in key:
                # Nested key update
                parts = key.split("__")
                if len(parts) == 2:
                    section, param = parts
                    if section in self._config:
                        self._config[section][param] = value
            else:
                # Direct key update
                if key in self._config:
                    if hasattr(value, "model_dump"):
                        self._config[key] = value.model_dump()
                    else:
                        self._config[key] = value

        # Save to disk
        self._save(self._config)
        return self._config.copy()

    def update_streaming(self, config: StreamingConfig) -> None:
        """Update streaming configuration"""
        self._config["streaming"] = config.model_dump()
        self._save(self._config)

    def update_server(self, config: ServerConfig) -> None:
        """Update server configuration"""
        self._config["server"] = config.model_dump()
        self._save(self._config)

    def get_rvc_config(self) -> RvcConfig:
        """Get RVC inference configuration"""
        return RvcConfig(**self._config.get("rvc", {}))

    def update_rvc(self, config: RvcConfig) -> None:
        """Update RVC inference configuration"""
        self._config["rvc"] = config.model_dump()
        self._save(self._config)

    def add_model(self, model_config: ModelConfig) -> None:
        """Add model configuration"""
        if "models" not in self._config:
            self._config["models"] = []

        # Check if model already exists
        existing_names = [m.get("name") for m in self._config["models"]]
        if model_config.name in existing_names:
            # Update existing model
            for i, m in enumerate(self._config["models"]):
                if m.get("name") == model_config.name:
                    self._config["models"][i] = model_config.model_dump()
                    break
        else:
            # Add new model
            self._config["models"].append(model_config.model_dump())

        self._save(self._config)

    def remove_model(self, model_name: str) -> bool:
        """Remove model configuration"""
        if "models" not in self._config:
            return False

        initial_count = len(self._config["models"])
        self._config["models"] = [
            m for m in self._config["models"] if m.get("name") != model_name
        ]

        if len(self._config["models"]) < initial_count:
            self._save(self._config)
            return True
        return False

    def list_available_models(self) -> list[Dict[str, str]]:
        """List available models from assets/weights and models/ directories."""
        models = []
        for search_dir in [Path("assets/weights"), Path("models")]:
            if search_dir.exists():
                for pth_file in sorted(search_dir.glob("*.pth")):
                    models.append({"name": pth_file.stem, "path": str(pth_file)})
        return models

    def list_uploaded_models(self) -> list[Dict[str, Any]]:
        """List model files in the models/ upload directory."""
        upload_dir = Path("models")
        upload_dir.mkdir(exist_ok=True)
        result = []
        for p in sorted(upload_dir.glob("*.pth")):
            stat = p.stat()
            result.append({
                "name": p.name,
                "stem": p.stem,
                "path": str(p),
                "size": stat.st_size,
            })
        return result

    def reload(self) -> None:
        """Reload configuration from file"""
        self._config = self._load()

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""
        self._config = self._defaults.copy()
        self._save(self._config)


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: str = "configs/server.yaml") -> ConfigManager:
    """
    Get global config manager instance

    Args:
        config_file: Path to configuration file

    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager
