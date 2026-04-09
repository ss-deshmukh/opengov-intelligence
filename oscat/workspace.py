"""Workspace initialization and management for OSCAT.

Handles:
- oscat.md creation and reading (project context file)
- .env secret vault (store secrets without passing through LLM)
- Non-empty folder detection and user confirmation
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

OSCAT_MD_TEMPLATE = """\
# OSCAT Workspace

Created: {date}

<!-- Add project context, conventions, and notes below.
     OSCAT reads this file at the start of every conversation. -->
"""


class Workspace:
    """Manages the .oscat/ workspace directory and its files."""

    def __init__(self, base: Path) -> None:
        self._base = base
        self._oscat_dir = base / ".oscat"
        self._oscat_md = self._oscat_dir / "oscat.md"
        self._env_file = self._oscat_dir / ".env"
        self._oscat_md_last_read: datetime | None = None

    @property
    def base(self) -> Path:
        return self._base

    @property
    def oscat_md_path(self) -> Path:
        return self._oscat_md

    @property
    def env_path(self) -> Path:
        return self._env_file

    # ── Folder state checks ──────────────────────────────────────

    def is_initialized(self) -> bool:
        """Check if this workspace has been initialized (oscat.md exists)."""
        return self._oscat_md.is_file()

    def has_non_oscat_files(self) -> bool:
        """Check if the folder contains files that aren't part of OSCAT."""
        if not self._base.exists():
            return False
        for item in self._base.iterdir():
            name = item.name
            # Skip OSCAT's own files/dirs
            if name in (".oscat", ".env"):
                continue
            # Skip common hidden files
            if name.startswith("."):
                continue
            return True
        return False

    def needs_confirmation(self) -> bool:
        """Check if the user should confirm before initializing.

        Returns True if the folder is non-empty and doesn't have oscat.md.
        """
        return not self.is_initialized() and self.has_non_oscat_files()

    # ── Initialization ───────────────────────────────────────────

    def initialize(self) -> list[str]:
        """Create the workspace structure. Returns list of actions taken."""
        actions: list[str] = []

        # Create .oscat/ directory and memory subdirectory
        self._oscat_dir.mkdir(parents=True, exist_ok=True)
        (self._oscat_dir / "memory").mkdir(exist_ok=True)
        actions.append(f"Created {self._oscat_dir}")

        # Create oscat.md if it doesn't exist
        if not self._oscat_md.is_file():
            self._oscat_md.write_text(
                OSCAT_MD_TEMPLATE.format(date=datetime.now().strftime("%Y-%m-%d"))
            )
            actions.append(f"Created {self._oscat_md}")

        # Create .env if it doesn't exist
        if not self._env_file.is_file():
            self._env_file.write_text("# OSCAT environment variables\n")
            actions.append(f"Created {self._env_file}")

        return actions

    # ── oscat.md reading ─────────────────────────────────────────

    def read_oscat_md(self) -> str | None:
        """Read oscat.md content. Returns None if it doesn't exist."""
        if not self._oscat_md.is_file():
            return None
        return self._oscat_md.read_text()

    def oscat_md_modified_since_last_read(self) -> bool:
        """Check if oscat.md has been modified since last read_oscat_md_tracked()."""
        if not self._oscat_md.is_file():
            return False
        mtime = datetime.fromtimestamp(self._oscat_md.stat().st_mtime)
        if self._oscat_md_last_read is None:
            return True
        return mtime > self._oscat_md_last_read

    def read_oscat_md_tracked(self) -> str | None:
        """Read oscat.md and track the read timestamp."""
        content = self.read_oscat_md()
        if content is not None:
            self._oscat_md_last_read = datetime.now()
        return content

    def build_oscat_md_context(self) -> str:
        """Build a prompt section from oscat.md content, if any."""
        content = self.read_oscat_md_tracked()
        if not content or not content.strip():
            return ""

        return (
            "\n\n## Project Context (oscat.md)\n"
            "The following was written by the user in .oscat/oscat.md:\n\n"
            f"{content.strip()}\n"
        )

    # ── Secret vault (.env management) ───────────────────────────

    def load_env(self) -> dict[str, str]:
        """Load all variables from .oscat/.env. Returns key=value dict."""
        result: dict[str, str] = {}
        if not self._env_file.is_file():
            return result
        for line in self._env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
        return result

    def get_secret(self, key: str) -> str | None:
        """Get a specific secret from .oscat/.env."""
        env = self.load_env()
        return env.get(key)

    def has_secret(self, key: str) -> bool:
        """Check if a secret exists in .oscat/.env."""
        return self.get_secret(key) is not None

    def set_secret(self, key: str, value: str) -> None:
        """Store a secret in .oscat/.env without passing it through the LLM.

        The value is written directly to the .env file, and the
        environment variable is set in the current process.
        """
        self._oscat_dir.mkdir(parents=True, exist_ok=True)

        # Read existing lines
        lines: list[str] = []
        replaced = False
        if self._env_file.is_file():
            for line in self._env_file.read_text().splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    existing_key = stripped.partition("=")[0].strip()
                    if existing_key == key:
                        lines.append(f"{key}={value}")
                        replaced = True
                        continue
                lines.append(line)

        if not replaced:
            lines.append(f"{key}={value}")

        self._env_file.write_text("\n".join(lines) + "\n")

        # Also set in current process environment
        os.environ[key] = value

    def remove_secret(self, key: str) -> bool:
        """Remove a secret from .oscat/.env.

        Returns True if the key was found and removed, False otherwise.
        """
        if not self._env_file.is_file():
            return False

        lines: list[str] = []
        found = False
        for line in self._env_file.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                existing_key = stripped.partition("=")[0].strip()
                if existing_key == key:
                    found = True
                    continue
            lines.append(line)

        if found:
            self._env_file.write_text("\n".join(lines) + "\n")
            os.environ.pop(key, None)

        return found

    def apply_env_to_process(self) -> int:
        """Load .oscat/.env variables into os.environ. Returns count loaded."""
        env = self.load_env()
        count = 0
        for key, value in env.items():
            if key not in os.environ:
                os.environ[key] = value
                count += 1
        return count
