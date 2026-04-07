"""
Configuration helper utilities for YAML-driven experiments.
"""

import os
from typing import Any, Dict, Optional

import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, 'r') as handle:
        config = yaml.safe_load(handle) or {}
    return config


def merge_config(base: Dict[str, Any], override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge two configuration dictionaries recursively."""
    if override is None:
        return base
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


class Config:
    """Simple YAML-backed configuration container."""

    def __init__(self, path: str):
        self._data = load_yaml_config(path)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)
