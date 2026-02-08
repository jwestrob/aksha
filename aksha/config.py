"""Configuration management for Aksha.

Path resolution precedence:
1. Environment variables (AKSHA_DATA_DIR, AKSHA_CONFIG_DIR)
2. User config file (~/.config/aksha/config.toml)
3. XDG defaults (~/.local/share/aksha, ~/.config/aksha)

Example config.toml:
    data_dir = "/scratch/shared/hmm_databases"
    
    [defaults]
    threads = 8
    cut_ga = true
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    # Python 3.10 compatibility
    import tomli as tomllib

from aksha.types import DatabaseInfo, MoleculeType


@dataclass
class AkshaConfig:
    """Global configuration for Aksha."""

    data_dir: Path
    config_dir: Path
    defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "AkshaConfig":
        """Load configuration from files and environment."""
        config_dir = cls._resolve_config_dir()
        config_file = config_dir / "config.toml"

        user_config: dict[str, Any] = {}
        if config_file.exists():
            with open(config_file, "rb") as f:
                user_config = tomllib.load(f)

        data_dir = cls._resolve_data_dir(user_config)

        return cls(
            data_dir=data_dir,
            config_dir=config_dir,
            defaults=user_config.get("defaults", {}),
        )

    @staticmethod
    def _resolve_config_dir() -> Path:
        """Resolve config directory path."""
        if env := os.environ.get("AKSHA_CONFIG_DIR"):
            return Path(env).expanduser()
        if xdg := os.environ.get("XDG_CONFIG_HOME"):
            return Path(xdg) / "aksha"
        return Path.home() / ".config" / "aksha"

    @staticmethod
    def _resolve_data_dir(user_config: dict[str, Any]) -> Path:
        """Resolve data directory path."""
        if env := os.environ.get("AKSHA_DATA_DIR"):
            return Path(env).expanduser()
        if config_path := user_config.get("data_dir"):
            return Path(config_path).expanduser()
        if xdg := os.environ.get("XDG_DATA_HOME"):
            return Path(xdg) / "aksha"
        return Path.home() / ".local" / "share" / "aksha"

    def ensure_dirs(self) -> None:
        """Create config and data directories if needed."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def resolve_database_path(self, db_path: str) -> Path:
        """Resolve database path (absolute or relative to data_dir)."""
        p = Path(db_path)
        if p.is_absolute():
            return p
        return self.data_dir / p


class DatabaseRegistry:
    """Manages HMM database metadata and installation state.
    
    Separates bundled database definitions (immutable) from local state
    (installed status, paths, versions).
    """

    def __init__(self, config: AkshaConfig):
        self.config = config
        self._state_file = config.data_dir / "databases.json"
        self._entries: dict[str, DatabaseInfo] = {}
        self._load()

    def _load(self) -> None:
        """Load database definitions and local state."""
        # Load bundled definitions
        bundled_path = Path(__file__).parent / "data" / "databases.json"
        with open(bundled_path) as f:
            bundled = json.load(f)

        # Load local state
        local_state: dict[str, dict] = {}
        if self._state_file.exists():
            with open(self._state_file) as f:
                local_state = json.load(f)

        # Merge bundled + local
        for db in bundled["databases"]:
            name = db["name"]
            state = local_state.get(name, {})

            self._entries[name] = DatabaseInfo(
                name=name,
                url=db["url"],
                molecule_type=MoleculeType[db["molecule_type"].upper()],
                citation=db.get("citation", ""),
                notes=db.get("notes"),
                domain=db.get("domain", False),
                has_thresholds=db.get("has_thresholds", False),
                installed=state.get("installed", False),
                path=state.get("path"),
                version=state.get("version"),
            )

        # Load custom databases
        for name, state in local_state.items():
            if name not in self._entries and state.get("custom"):
                self._entries[name] = DatabaseInfo(
                    name=name,
                    url=state.get("url", ""),
                    molecule_type=MoleculeType[state["molecule_type"].upper()],
                    citation=state.get("citation", ""),
                    installed=state.get("installed", False),
                    path=state.get("path"),
                )

    def _save_state(self) -> None:
        """Persist installation state to disk."""
        state = {}
        for name, entry in self._entries.items():
            if entry.installed or entry.path:
                state[name] = {
                    "installed": entry.installed,
                    "path": entry.path,
                    "version": entry.version,
                }

        self.config.ensure_dirs()
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)

    def get(self, name: str) -> Optional[DatabaseInfo]:
        """Get database by name."""
        return self._entries.get(name)

    def list_available(
        self,
        molecule_type: Optional[MoleculeType] = None,
        installed_only: bool = False,
    ) -> list[DatabaseInfo]:
        """List databases, optionally filtered."""
        entries = list(self._entries.values())

        if molecule_type:
            entries = [e for e in entries if e.molecule_type == molecule_type]
        if installed_only:
            entries = [e for e in entries if e.installed]

        return sorted(entries, key=lambda e: e.name)

    def mark_installed(
        self,
        name: str,
        path: str,
        version: Optional[str] = None,
    ) -> None:
        """Mark database as installed."""
        if name not in self._entries:
            raise ValueError(f"Unknown database: {name}")

        self._entries[name].installed = True
        self._entries[name].path = path
        self._entries[name].version = version
        self._save_state()

    def mark_uninstalled(self, name: str) -> None:
        """Mark database as uninstalled."""
        if name in self._entries:
            self._entries[name].installed = False
            self._entries[name].path = None
            self._save_state()

    def get_path(self, name: str) -> Optional[Path]:
        """Get resolved filesystem path for installed database."""
        entry = self.get(name)
        if entry and entry.installed and entry.path:
            return entry.resolve_path(self.config.data_dir)
        return None

    def register_custom(
        self,
        name: str,
        path: str,
        molecule_type: MoleculeType,
        citation: str = "",
    ) -> None:
        """Register a custom database not in bundled list."""
        if name in self._entries:
            raise ValueError(f"Database already exists: {name}")

        self._entries[name] = DatabaseInfo(
            name=name,
            url="",
            molecule_type=molecule_type,
            citation=citation,
            installed=True,
            path=path,
        )
        self._save_state()


# Module-level convenience functions
_config: Optional[AkshaConfig] = None
_registry: Optional[DatabaseRegistry] = None


def get_config() -> AkshaConfig:
    """Get global config instance (lazy loaded)."""
    global _config
    if _config is None:
        _config = AkshaConfig.load()
    return _config


def get_registry() -> DatabaseRegistry:
    """Get global database registry (lazy loaded)."""
    global _registry
    if _registry is None:
        _registry = DatabaseRegistry(get_config())
    return _registry


def reset_globals() -> None:
    """Reset global instances (useful for testing)."""
    global _config, _registry
    _config = None
    _registry = None
