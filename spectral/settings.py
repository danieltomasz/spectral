import os
from pathlib import Path
from typing import Any, Dict, Union, Optional


class Settings:
    """Class to store and access settings loaded from a file."""

    def __init__(self, settings_dict: Dict[str, Any]):
        self._settings = settings_dict

    def __getattr__(self, name: str) -> Any:
        try:
            return self._settings[name]
        except KeyError:
            raise AttributeError(f"Setting '{name}' not found.")

    def get(self, key: str, default: Any = None) -> Any:
        """Safely get a value, returning a default if not found."""
        keys = key.split(".")
        value = self._settings
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def get_path(self, key: str, **kwargs) -> Path:
        """Get a path, replacing placeholders with provided values or other settings."""
        value = self.get(key)
        if not isinstance(value, str):
            raise ValueError(f"The key '{key}' does not correspond to a path string.")

        for k, v in kwargs.items():
            value = value.replace(f"{{{k}}}", str(v))

        while "{" in value and "}" in value:
            start, end = value.find("{"), value.find("}")
            if start == -1 or end == -1:
                break
            placeholder = value[start + 1 : end]
            replacement = str(self.get(placeholder, ""))
            value = value[:start] + replacement + value[end + 1 :]

        return Path(value)


def find_project_root(start_path: Union[str, Path] = None) -> Path:
    """Find the project root by looking for a settings file or common project markers."""
    if start_path is None:
        start_path = Path.cwd()
    start_path = Path(start_path).resolve()

    markers = [
        "settings.toml",
        "settings.yaml",
        "settings.yml",
        "settings.json",
        ".git",
        "pyproject.toml",
    ]

    for path in [start_path] + list(start_path.parents):
        if any((path / marker).exists() for marker in markers):
            return path

    raise FileNotFoundError("Could not find project root.")


def load_settings(
    filename: str = "settings.toml", root_dir: Union[str, Path] = None
) -> Settings:
    """Load settings from a file and return a Settings object."""
    if root_dir is None:
        root_dir = find_project_root()
    settings_path = Path(root_dir) / filename

    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    file_format = settings_path.suffix.lower().lstrip(".")

    if file_format in ("toml", "tml"):
        import tomllib

        with open(settings_path, "rb") as file:
            settings_dict = tomllib.load(file)
    elif file_format in ("yaml", "yml"):
        import yaml

        with open(settings_path, "r") as file:
            settings_dict = yaml.safe_load(file)
    elif file_format == "json":
        import json

        with open(settings_path, "r") as file:
            settings_dict = json.load(file)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    return Settings(settings_dict)


# Usage example
if __name__ == "__main__":
    try:
        settings = load_settings()
        print("Settings loaded successfully:")

        # Accessing values (assuming a structure similar to the TOML example)
        print(f"Project name: {settings.project.name}")
        print(f"Analysis method: {settings.get('analysis.method')}")
        print(f"Data path: {settings.get_path('paths.data')}")

        # Using default values
        print(f"Unknown setting: {settings.get('unknown.setting', 'default_value')}")

    except Exception as e:
        print(f"Error: {e}")
