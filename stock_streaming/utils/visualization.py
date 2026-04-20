"""
visualization.py — ASCII/ANSI terminal visualisations for the streaming
classification system: live ticker bar, confidence meter, label timeline,
and window-size history sparkline.  No external libraries required.
"""

import os
import sys
import math
from collections import deque
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"
ANSI_RED    = "\033[91m"
ANSI_GREEN  = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN   = "\033[96m"
ANSI_WHITE  = "\033[97m"
ANSI_DIM    = "\033[2m"

_COLOUR_ENABLED = sys.stdout.isatty()


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{ANSI_RESET}" if _COLOUR_ENABLED else text


LABEL_COLOURS = {
    "STRONG_BUY":  ANSI_GREEN,
    "BUY":         "\033[32m",
    "HOLD":        ANSI_YELLOW,
    "SELL":        "\033[31m",
    "STRONG_SELL": ANSI_RED,
}

LABEL_ICONS = {
    "STRONG_BUY":  "▲▲",
    "BUY":         "▲ ",
    "HOLD":        "──",
    "SELL":        "▼ ",
    "STRONG_SELL": "▼▼",
}


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------

SPARK_CHARS = "▁▂▃▄▅▆▇█"

def sparkline(values: List[float], width: int = 40) -> str:
    """Render a list of floats as a compact unicode sparkline."""
    if not values:
        return " " * width
    lo, hi = min(values), max(values)
    span   = hi - lo + 1e-9
    n      = len(SPARK_CHARS) - 1
    chars  = [SPARK_CHARS[int((v - lo) / span * n)] for v in values[-width:]]
    return "".join(chars).ljust(width)


# ---------------------------------------------------------------------------
# Confidence bar
# ---------------------------------------------------------------------------

def confidence_bar(confidence: float, width: int = 20) -> str:
    """Render a confidence value as a filled progress bar."""
    filled = int(confidence * width)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = f"{confidence * 100:5.1f}%"
    return f"[{bar}] {pct}"


# ---------------------------------------------------------------------------
# Live ticker line
# ---------------------------------------------------------------------------

def format_tick_line(event) -> str:
    """
    Format one PredictionEvent into a compact, coloured terminal line.

    Example:
        [00142] AAPL  ▲  BUY       [████████░░░░░░░░░░░░] 42.1%  ws=60  1.23ms
    """
    label  = event.label
    colour = LABEL_COLOURS.get(label, ANSI_WHITE)
    icon   = LABEL_ICONS.get(label, "  ")
    lbl_str = _c(f"{icon} {label:<12}", colour)

    conf_bar = confidence_bar(event.confidence, width=18)
    latency  = f"{event.latency_ms:5.1f}ms"
    ws       = f"ws={event.window_size}"

    return (f"[{event.tick_index:05d}] {event.symbol:<6} "
            f"{lbl_str} {conf_bar}  {ws:<6} {latency}")


# ---------------------------------------------------------------------------
# Dashboard (printed to terminal on each update)
# ---------------------------------------------------------------------------

class StreamingDashboard:
    """
    Full-screen terminal dashboard refreshed on each prediction event.
    Uses ANSI escape codes to move the cursor — works on most Unix terminals
    and Windows 10+ with ANSI mode enabled.
    """

    HEIGHT       = 18  # Lines to keep reserved for the dashboard
    HISTORY_LEN  = 60  # Number of past predictions to display/spark

    def __init__(self):
        self._pred_history:   deque = deque(maxlen=self.HISTORY_LEN)
        self._ws_history:     deque = deque(maxlen=self.HISTORY_LEN)
        self._conf_history:   deque = deque(maxlen=self.HISTORY_LEN)
        self._latency_history: deque = deque(maxlen=self.HISTORY_LEN)
        self._label_counts:   Dict[str, int] = {}
        self._total           = 0
        self._first_render    = True

    def update(self, event) -> None:
        label = event.smoothed_label
        self._pred_history.append(event)
        self._ws_history.append(event.window_size)
        self._conf_history.append(event.smoothed_confidence)
        self._latency_history.append(event.latency_ms)
        self._label_counts[label] = self._label_counts.get(label, 0) + 1
        self._total += 1
        self._render()

    def _render(self) -> None:
        if not self._first_render:
            # Move cursor up to overwrite the previous dashboard
            sys.stdout.write(f"\033[{self.HEIGHT}A\033[J")
        self._first_render = False

        lines = self._build_lines()
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    def _build_lines(self) -> List[str]:
        w = min(72, os.get_terminal_size().columns - 2) if _COLOUR_ENABLED else 72
        sep = "─" * w

        # Latest event
        latest = self._pred_history[-1] if self._pred_history else None
        lbl    = latest.smoothed_label if latest else "HOLD"
        conf   = latest.smoothed_confidence if latest else 0.0
        ws     = latest.window_size if latest else 60
        colour = LABEL_COLOURS.get(lbl, ANSI_WHITE)
        icon   = LABEL_ICONS.get(lbl, "  ")

        lines = [
            _c("  ◆ Stock Streaming Classification — Live Dashboard", ANSI_BOLD + ANSI_CYAN),
            sep,
            f"  Signal  : {_c(icon + ' ' + lbl, colour)}",
            f"  Confidence : {confidence_bar(conf, 24)}",
            f"  Predictions: {self._total:<6}  Window size: {ws}",
            sep,
            "  Confidence history (last 60):",
            "  " + _c(sparkline(list(self._conf_history)), ANSI_CYAN),
            "  Window-size history:",
            "  " + _c(sparkline(list(self._ws_history)), ANSI_YELLOW),
            "  Latency (ms) history:",
            "  " + _c(sparkline(list(self._latency_history)), ANSI_DIM),
            sep,
            "  Label distribution:",
        ]

        total = max(1, self._total)
        for lbl_name in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]:
            count = self._label_counts.get(lbl_name, 0)
            bar   = "▪" * int(count / total * 30)
            pct   = f"{count / total * 100:4.1f}%"
            clr   = LABEL_COLOURS.get(lbl_name, ANSI_WHITE)
            lbl_padded = lbl_name.ljust(13)
            lines.append(f"    {_c(lbl_padded, clr)} {pct}  {_c(bar, clr)}")

        lines.append(sep)
        return lines


# ---------------------------------------------------------------------------
# Simple non-dashboard formatter (for logging / non-TTY contexts)
# ---------------------------------------------------------------------------

class SimplePrinter:
    """Falls back to simple one-line-per-event printing when no TTY."""

    def __init__(self, every_n: int = 1):
        self._every_n = every_n
        self._count   = 0

    def update(self, event) -> None:
        self._count += 1
        if self._count % self._every_n == 0:
            print(format_tick_line(event))


def make_display(force_simple: bool = False):
    """Factory: returns a StreamingDashboard or SimplePrinter."""
    if force_simple or not _COLOUR_ENABLED:
        return SimplePrinter(every_n=5)
    return StreamingDashboard()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time, random
    rng = random.Random(0)

    class FakeEvent:
        def __init__(self, i):
            labels = ["STRONG_BUY","BUY","HOLD","SELL","STRONG_SELL"]
            self.tick_index         = i
            self.symbol             = "AAPL"
            self.label              = rng.choice(labels)
            self.smoothed_label     = self.label
            self.confidence         = rng.uniform(0.4, 0.95)
            self.smoothed_confidence = self.confidence
            self.latency_ms         = rng.uniform(0.5, 8.0)
            self.window_size        = rng.randint(30, 90)

    display = make_display()
    for i in range(30):
        display.update(FakeEvent(i))
        time.sleep(0.12)
