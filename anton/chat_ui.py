from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console


@dataclass
class _ToolActivity:
    tool_id: str
    name: str
    json_parts: list[str] = field(default_factory=list)
    description: str = ""
    current_progress: str = ""
    step_count: int = 0
    eta_str: str = ""


_TOOL_LABELS: dict[str, str] = {
    "scratchpad": "Scratchpad",
    "memorize": "Memory",
    "recall": "Recall",
}

_MAX_DESC = 60

_REFRESH_FPS = 6


def _tool_display_text(name: str, input_json: str) -> str:
    """Map tool name + raw JSON input to a human-readable description."""
    label = _TOOL_LABELS.get(name, name)
    try:
        data = json.loads(input_json)
    except (json.JSONDecodeError, TypeError):
        return label

    desc = ""
    if name == "scratchpad":
        desc = data.get("one_line_description") or data.get("action", "")
    elif name == "memorize":
        entries = data.get("entries", [])
        desc = f"{len(entries)} entry/entries"
    if desc:
        if len(desc) > _MAX_DESC:
            desc = desc[:_MAX_DESC - 1] + "\u2026"
        return f"{label}({desc})"
    return label

THINKING_MESSAGES = [
    "Consulting the sacred docs...",
    "Rebasing my neurons...",
    "Spinning up inference hamsters...",
    "Parsing the vibes...",
    "Asking the rubber duck...",
    "Aligning my attention heads...",
    "Searching the latent space...",
    "Unrolling the loops...",
    "Compiling thoughts...",
    "Warming up the transformer...",
    "Descending the gradient...",
    "Sampling from the posterior...",
    "Tokenizing reality...",
    "Running a forward pass...",
    "Traversing the context window...",
    "Optimizing the objective...",
    "Softmaxing the options...",
    "Backpropagating insights...",
    "Loading weights...",
    "Crunching embeddings...",
]

WORKING_FOOTER_MESSAGES = [
    "working through your request",
    "piecing together a solution",
    "reasoning through the problem",
    "exploring the best approach",
    "connecting the dots for you",
    "building your answer step by step",
    "untangling the problem for you",
    "chewing on this one carefully",
    "cooking up a solid answer",
    "wiring together a solution",
]

TOOL_MESSAGES = [
    "Rolling up sleeves...",
    "Firing up the agent...",
    "Handing off to the crew...",
    "Dispatching the task...",
    "Engaging autopilot...",
    "Letting the tools cook...",
]

ANALYZING_MESSAGES = [
    "Analyzing results...",
    "Reading the output...",
    "Digesting the results...",
    "Making sense of the output...",
    "Processing results...",
    "Reviewing the output...",
]

CANCEL_MESSAGES = [
    "Ok, dropping everything\u2026",
    "Alright, pulling the plug\u2026",
    "Stopping the presses\u2026",
    "Hitting the brakes\u2026",
    "Winding down\u2026",
    "Wrapping it up\u2026",
    "Ok, letting go of this one\u2026",
    "Understood, shutting it down\u2026",
    "Copy that, standing down\u2026",
    "Roger, aborting mission\u2026",
]

PHASE_LABELS = {
    "memory_recall": "Memory",
    "planning": "Planning",
    "executing": "Executing",
    "complete": "Complete",
    "failed": "Failed",
    "scratchpad": "Scratchpad",
}


class StreamDisplay:
    """Manages a Rich Live display for streaming LLM responses.

    Uses segmented Live sessions: Live is active only while content is
    actively streaming. Between phases (e.g. after a tool result, before
    analysis), content is printed permanently so the user can scroll.
    """

    def __init__(self, console: Console, toolbar: dict | None = None) -> None:
        self._console = console
        self._live: Live | None = None
        self._initial_text = ""
        self._buffer = ""
        self._started = False
        self._toolbar = toolbar
        self._activities: list[_ToolActivity] = []
        self._thinking_msg: str = ""
        self._in_tool_phase = False
        self._answer_started = False
        self._last_was_tool = False
        self._footer_msg: str = ""
        self._cancel_msg: str = ""
        self._active = False  # True between start() and finish()/abort()

    def _set_status(self, text: str) -> None:
        if self._toolbar is not None:
            self._toolbar["status"] = text

    # --- Live lifecycle helpers ---

    def _start_live(self, spinner_text: str | None = None) -> None:
        """Start a new Live segment (or no-op if already running)."""
        if self._live is not None:
            return
        text = spinner_text or self._thinking_msg
        spinner = Spinner("dots", text=Text(f" {text}", style="anton.muted"))
        self._live = Live(
            spinner,
            console=self._console,
            refresh_per_second=_REFRESH_FPS,
            transient=True,
        )
        self._live.start()

    def _stop_live(self) -> None:
        """Stop the current Live segment (content vanishes — caller prints permanently)."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _flush_text(self) -> None:
        """Stop Live and print accumulated *text* permanently (not the activity tree).

        The activity tree is printed once in finish(). Intermediate flushes
        only emit text so tool results become scrollable between phases.
        """
        self._stop_live()

        # Print initial text (muted "inner speech")
        if self._initial_text:
            if self._activities:
                self._console.print(Text(self._initial_text.rstrip(), style="anton.muted"))
            else:
                self._console.print(Text("anton> ", style="anton.cyan"), end="")
                self._console.print(Markdown(self._initial_text))
            self._initial_text = ""

        # Print buffered answer text
        if self._buffer:
            if not self._answer_header_printed:
                self._console.print(Text("anton> ", style="anton.cyan"), end="")
                self._answer_header_printed = True
            self._console.print(Markdown(self._buffer))
            self._buffer = ""

    # --- Public API ---

    def start(self) -> None:
        msg = random.choice(THINKING_MESSAGES)  # noqa: S311
        self._thinking_msg = msg
        self._set_status(msg)
        self._initial_text = ""
        self._buffer = ""
        self._started = False
        self._activities = []
        self._answer_header_printed = False
        self._in_tool_phase = False
        self._answer_started = False
        self._last_was_tool = False
        self._cancel_msg = ""
        self._footer_msg = random.choice(WORKING_FOOTER_MESSAGES)  # noqa: S311
        self._active = True
        self._start_live()

    def append_text(self, delta: str) -> None:
        if not self._active:
            return
        # Restart Live if it was stopped between phases
        if self._live is None:
            self._start_live()
        if self._in_tool_phase:
            if self._buffer and self._last_was_tool:
                self._buffer += "\n\n"
            self._buffer += delta
            self._answer_started = True
            self._last_was_tool = False
        else:
            self._initial_text += delta
        self._started = True
        self._refresh_live()

    def show_tool_result(self, content: str) -> None:
        """Flush current text permanently, then print tool result permanently."""
        if not self._active:
            return
        # Flush any accumulated text so far
        self._flush_text()
        # Print the tool result directly — scrollable immediately
        self._console.print(Markdown(content))
        self._last_was_tool = True
        self._started = True

    def show_tool_execution(self, task: str) -> None:
        """Backward-compatible wrapper — delegates to on_tool_use_start."""
        self.on_tool_use_start(f"_compat_{id(task)}", task)

    def on_tool_use_start(self, tool_id: str, name: str) -> None:
        """Track a new tool use and update the live display."""
        if not self._active:
            return
        # Restart Live if stopped between phases
        if self._live is None:
            self._start_live()
        self._in_tool_phase = True
        self._last_was_tool = True
        activity = _ToolActivity(tool_id=tool_id, name=name)
        self._activities.append(activity)
        self._refresh_live()

    def on_tool_use_delta(self, tool_id: str, json_delta: str) -> None:
        """Accumulate JSON input deltas for a tool use."""
        for act in self._activities:
            if act.tool_id == tool_id:
                act.json_parts.append(json_delta)
                return

    def on_tool_use_end(self, tool_id: str) -> None:
        """Finalize a tool use: parse accumulated JSON and set description."""
        for act in self._activities:
            if act.tool_id == tool_id:
                raw = "".join(act.json_parts)
                act.description = _tool_display_text(act.name, raw)
                self._refresh_live()
                return

    def update_progress(self, phase: str, message: str, eta: float | None = None) -> None:
        """Update the Live display with agent progress (phase + message + optional ETA)."""
        if not self._active:
            return

        # Tools finished, LLM is now analyzing — flush text, restart with analyzing spinner
        if phase == "analyzing":
            self._flush_text()
            self._thinking_msg = random.choice(ANALYZING_MESSAGES)  # noqa: S311
            self._start_live(self._thinking_msg)
            return

        # Scratchpad is about to start — set description + ETA immediately
        if phase == "scratchpad_start" and self._activities:
            for act in reversed(self._activities):
                if act.name == "scratchpad":
                    act.description = _tool_display_text(act.name, "".join(act.json_parts)) or f"Scratchpad({message})"
                    if eta:
                        act.eta_str = f"~{int(eta)}s"
                    break
            self._refresh_live()
            return

        # For scratchpad streaming, show progress on the activity line itself
        if phase == "scratchpad" and self._activities:
            for act in reversed(self._activities):
                if act.name == "scratchpad":
                    act.current_progress = message
                    break
            self._refresh_live()
            return

        label = PHASE_LABELS.get(phase, phase)
        eta_str = f"  ~{int(eta)}s" if eta else ""
        status = f"{label}  {message}{eta_str}"
        self._set_status(status)
        self._refresh_live()

    def finish(self) -> None:
        self._stop_live()

        # Finalize any activities that never got on_tool_use_end
        for act in self._activities:
            if not act.description and act.json_parts:
                raw = "".join(act.json_parts)
                act.description = _tool_display_text(act.name, raw)

        # Print initial text
        if self._initial_text:
            if self._activities:
                self._console.print(Text(self._initial_text.rstrip(), style="anton.muted"))
            else:
                self._console.print(Text("anton> ", style="anton.cyan"), end="")
                self._console.print(Markdown(self._initial_text))
            self._initial_text = ""

        # Print activity tree once
        if self._activities:
            self._console.print(self._build_activity_tree(final=True))

        # Print any remaining buffered answer text
        if self._buffer:
            if not self._answer_header_printed:
                self._console.print(Text("anton> ", style="anton.cyan"), end="")
            self._console.print(Markdown(self._buffer))
            self._buffer = ""

        # If no activities and no text was printed at all, nothing to show
        self._active = False
        self._console.print()

    def abort(self) -> None:
        self._stop_live()
        self._active = False

    def show_context_compacted(self, message: str) -> None:
        """Show a notification that context was compacted."""
        if not self._active:
            return
        if self._buffer:
            self._buffer += "\n\n"
        self._buffer += f"> *{message}*"
        self._started = True
        self._refresh_live()

    def show_cancelling(self) -> None:
        """Update the footer to acknowledge that cancellation is in progress."""
        self._cancel_msg = random.choice(CANCEL_MESSAGES)  # noqa: S311
        self._refresh_live()

    # --- Private helpers ---

    def _build_activity_tree(self, final: bool = False) -> Text:
        """Render the activity tree as styled Text."""
        lines = Text()
        for i, act in enumerate(self._activities):
            label = act.description or _TOOL_LABELS.get(act.name, act.name)
            if i == 0:
                lines.append("\u23bf ")
            else:
                lines.append("  ")
            lines.append(label, style="bold")
            if act.eta_str and not final:
                lines.append(f" {act.eta_str}", style="anton.muted")
            if act.current_progress:
                lines.append(f" \u2190 {act.current_progress}", style="anton.muted")
            lines.append("\n")
        return lines

    def _refresh_live(self) -> None:
        """Recompose the live display: spinner, tree, streaming text."""
        if self._live is None:
            return

        parts: list = []

        if self._activities:
            # Show initial text as muted "inner speech" at top
            if self._initial_text:
                parts.append(Text(self._initial_text.rstrip(), style="anton.muted"))

            # Spinner while tools are running (before answer streams)
            if not self._answer_started:
                spinner = Spinner("dots", text=Text(f" {self._thinking_msg}", style="anton.muted"))
                parts.append(spinner)

            # Activity tree
            parts.append(self._build_activity_tree())

            # Answer text below tree
            if self._buffer:
                parts.append(Text())
                parts.append(Markdown(self._buffer))
        elif self._initial_text:
            # Pure text streaming, no tools yet
            parts.append(Markdown(self._initial_text))
        else:
            # Nothing yet — just spinner
            spinner = Spinner("dots", text=Text(f" {self._thinking_msg}", style="anton.muted"))
            parts.append(spinner)

        # Working footer — visible while streaming
        if self._cancel_msg:
            footer = Text(f"\n\u23f5\u23f5 {self._cancel_msg}", style="#ff69b4")
        else:
            footer = Text(f"\n\u23f5\u23f5 Esc to stop \u2014 {self._footer_msg}", style="#ff69b4")
        parts.append(footer)

        self._live.update(Group(*parts) if len(parts) > 1 else parts[0])
