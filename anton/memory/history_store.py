"""Chat history persistence — save/load full conversation history for resume.

Stores conversation history as JSON files alongside episodic JSONL files
in the `.anton/episodes/` directory.  Fire-and-forget writes (never raises).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class HistoryStore:
    """Persist and retrieve full chat history for session resume."""

    def __init__(self, episodes_dir: Path) -> None:
        self._dir = episodes_dir

    def save(self, session_id: str, history: list[dict]) -> None:
        """Atomically write history to ``{session_id}_history.json``.

        Fire-and-forget: silently ignores errors to avoid disrupting chat.
        """
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            target = self._dir / f"{session_id}_history.json"
            fd, tmp = tempfile.mkstemp(
                dir=str(self._dir), suffix=".tmp", prefix=".hist_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False)
                os.replace(tmp, str(target))
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        except Exception:
            pass  # Fire-and-forget

    def load(self, session_id: str) -> list[dict] | None:
        """Load history for *session_id*.  Returns ``None`` on missing/corrupt."""
        path = self._dir / f"{session_id}_history.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return None
        except Exception:
            return None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """List recent sessions with history, newest-first.

        Returns a list of dicts with keys:
        ``session_id``, ``date``, ``turns``, ``preview``.
        """
        if not self._dir.is_dir():
            return []

        files = sorted(self._dir.glob("*_history.json"), reverse=True)
        results: list[dict] = []
        for path in files:
            if len(results) >= limit:
                break
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, list) or not data:
                continue

            session_id = path.stem.removesuffix("_history")

            # Count user turns
            turns = sum(1 for m in data if m.get("role") == "user")
            if turns == 0:
                continue

            # Extract date from session_id (format: YYYYMMDD_HHMMSS)
            try:
                dt = datetime.strptime(session_id, "%Y%m%d_%H%M%S").replace(
                    tzinfo=timezone.utc
                )
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                date_str = session_id

            # First user message as preview
            preview = ""
            for m in data:
                if m.get("role") == "user":
                    content = m.get("content", "")
                    if isinstance(content, str):
                        preview = content.strip()
                    elif isinstance(content, list):
                        # Multimodal content — find first text block
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                preview = block.get("text", "").strip()
                                break
                    break
            if len(preview) > 60:
                preview = preview[:57] + "..."

            results.append({
                "session_id": session_id,
                "date": date_str,
                "turns": turns,
                "preview": preview,
            })

        return results
