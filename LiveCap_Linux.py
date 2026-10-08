#!/usr/bin/env python3
"""
livecaption - offline live captions for Linux.

Captures system audio (whatever you hear) or your microphone, transcribes it
locally with faster-whisper, and shows rolling captions in a draggable,
always-on-top overlay (or in the terminal).

Requirements
------------
  pip install --user --break-system-packages faster-whisper
  sudo apt install pulseaudio-utils python3-tk numpy   # parec/pactl + tkinter

Usage
-----
  python3 livecaption.py                  # captions for system audio
  python3 livecaption.py --source mic     # captions for your microphone
  python3 livecaption.py --no-gui         # terminal only
  python3 livecaption.py --list-sources   # show audio sources
  python3 livecaption.py --reset-config   # forget saved settings

Overlay controls
----------------
  Drag with left mouse button  - move
  Double-click / right-click   - settings and menu
Settings you "Save" are stored in ~/.config/livecaption/config.json.
Command-line flags override saved settings for that run.
"""

import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2  # s16le mono

HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching",
    "bye", "you",
}

DEFAULTS = dict(
    # audio
    source="system", threshold=0.01, silence_ms=700, max_utterance=12.0,
    # recognition
    model="base", language=None, translate=False, device="cpu",
    compute_type="auto", partial_interval=1.0,
    # display
    lines=2, font_family="DejaVu Sans", font_size=32, bold=True,
    fg="#ffffff", bg="#000000", opacity=0.85, idle_opacity=0.25,
    width_pct=80, max_chars=0, clear_after=6.0, pos=None,
    # output
    log=None,
)

CONFIG_PATH = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
    "livecaption", "config.json",
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if k in DEFAULTS}
    except (OSError, ValueError):
        return {}


def save_config(cfg) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({k: getattr(cfg, k) for k in DEFAULTS}, f, indent=2)


def build_config(argv=None):
    p = argparse.ArgumentParser(description="Offline live captions for Linux.")
    p.add_argument("--source", help="'system', 'mic', or a PulseAudio/PipeWire source name")
    p.add_argument("--list-sources", action="store_true", help="list audio sources and exit")
    p.add_argument("--model", help="tiny, base, small, medium, large-v3, distil-large-v3, *.en ...")
    p.add_argument("--language", help="e.g. en, es, de, or 'auto'")
    p.add_argument("--translate", action=argparse.BooleanOptionalAction,
                   help="translate speech to English")
    p.add_argument("--device", help="cpu or cuda (cuda needs CUDA 12 libraries)")
    p.add_argument("--compute-type", dest="compute_type", help="auto, int8, float16, ...")
    p.add_argument("--threshold", type=float, help="min RMS level counted as speech")
    p.add_argument("--silence-ms", dest="silence_ms", type=int, help="silence that ends a caption")
    p.add_argument("--max-utterance", dest="max_utterance", type=float,
                   help="max seconds per caption chunk")
    p.add_argument("--partial-interval", dest="partial_interval", type=float,
                   help="seconds between live updates")
    p.add_argument("--lines", type=int, help="overlay lines")
    p.add_argument("--font-size", dest="font_size", type=int, help="overlay font size in pixels")
    p.add_argument("--opacity", type=float, help="overlay opacity 0-1")
    p.add_argument("--clear-after", dest="clear_after", type=float,
                   help="seconds before captions clear")
    p.add_argument("--log", help="append finalized captions to this file")
    p.add_argument("--no-gui", action="store_true", help="print captions in the terminal")
    p.add_argument("--reset-config", action="store_true", help="delete saved settings and exit")
    args = vars(p.parse_args(argv))

    extra = SimpleNamespace(
        list_sources=args.pop("list_sources"),
        no_gui=args.pop("no_gui"),
        reset_config=args.pop("reset_config"),
    )

    merged = dict(DEFAULTS)
    merged.update(load_config())
    merged.update({k: v for k, v in args.items() if v is not None})
    if merged["language"] in ("auto", ""):
        merged["language"] = None
    cfg = SimpleNamespace(**merged)
    cfg.paused = False
    return cfg, extra


# --------------------------------------------------------------------------- #
# Audio capture
# --------------------------------------------------------------------------- #
def resolve_source(name: str) -> str:
    if name == "system":
        return "@DEFAULT_MONITOR@"
    if name == "mic":
        return "@DEFAULT_SOURCE@"
    return name


def get_sources() -> list:
    if not shutil.which("pactl"):
        return []
    try:
        out = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    names = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            names.append(parts[1])
    return names


def list_sources() -> None:
    if not shutil.which("pactl"):
        sys.exit("pactl not found (install pulseaudio-utils).")

    subprocess.run(["pactl", "list", "short", "sources"])
    print("\nUse the name in column 2 with --source, or use 'system' / 'mic'.")


class AudioReader(threading.Thread):
    def __init__(self, cfg, audio_q, ui_q, stop, fatal):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.audio_q = audio_q
        self.ui_q = ui_q
        self.stop = stop
        self.fatal = fatal
        self.proc = None
        self.restart = threading.Event()

    def restart_now(self):
        """Switch to cfg.source."""
        self.restart.set()
        p = self.proc
        if p is not None:
            try:
                p.terminate()
            except OSError:
                pass

    def run(self):
        while not self.stop.is_set():
            self.restart.clear()

            cmd = [
                "parec",
                f"--device={resolve_source(self.cfg.source)}",
                "--format=s16le",
                f"--rate={SAMPLE_RATE}",
                "--channels=1",
                "--latency-msec=50",
            ]

            try:
                self.proc = proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as e:
                self.ui_q.put(("error", f"Cannot start parec: {e}"))
                self.stop.set()
                return

            try:
                while not self.stop.is_set() and not self.restart.is_set():
                    data = proc.stdout.read(CHUNK_BYTES)

                    if len(data) < CHUNK_BYTES:
                        break

                    if self.cfg.paused:
                        continue

                    self.audio_q.put(
                        np.frombuffer(
                            data,
                            dtype=np.int16,
                        ).astype(np.float32) / 32768.0
                    )
            finally:
                proc.terminate()

                try:
                    proc.wait(timeout=1)
                except subprocess.SubprocessError:
                    proc.kill()

            if self.restart.is_set() or self.stop.is_set():
                continue

            self.ui_q.put(
                (
                    "error",
                    f"Audio source '{self.cfg.source}' stopped or doesn't exist.",
                )
            )

            if self.fatal:
                self.stop.set()
                return

            while not self.stop.is_set() and not self.restart.wait(0.5):
                pass


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
class Transcriber(threading.Thread):
    def __init__(self, cfg, audio_q, ui_q, stop):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.audio_q = audio_q
        self.ui_q = ui_q
        self.stop = stop
        self.model = None
        self.loaded_key = None
        self._WhisperModel = None

    def _key(self):
        return (
            self.cfg.model,
            self.cfg.device,
            self.cfg.compute_type,
        )

    def _load(self) -> bool:
        """Load or reload the model."""
        c = self.cfg

        self.ui_q.put(("status", f"Loading model '{c.model}'…"))

        try:
            model = self._WhisperModel(
                c.model,
                device=c.device,
                compute_type=c.compute_type,
            )
        except Exception as e:
            if self.model is None:
                self.ui_q.put(("error", f"Model load failed: {e}"))
                self.stop.set()
                return False

            c.model, c.device, c.compute_type = self.loaded_key
            self.ui_q.put(("error", f"Could not load that model ({e}). Reverted."))
            self.ui_q.put(("status", "Listening…"))
            return True

        self.model = model
        self.loaded_key = self._key()

        while True:
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

        self.ui_q.put(("status", "Listening…"))
        return True

    def _transcribe(self, audio, final):
        c = self.cfg

        try:
            segments, _ = self.model.transcribe(
                audio,
                language=c.language,
                task="translate" if c.translate else "transcribe",
                beam_size=3 if final else 1,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
                condition_on_previous_text=False,
                without_timestamps=True,
            )

            text = " ".join(
                s.text.strip()
                for s in segments
            ).strip()

        except Exception as e:
            if c.device != "cpu":
                c.device = "cpu"
                self.model = None
                self.loaded_key = None
                self.ui_q.put(
                    (
                        "error",
                        f"GPU transcription failed ({e}). Falling back to CPU.",
                    )
                )
            else:
                self.ui_q.put(("error", f"Transcription failed: {e}"))
                self.stop.set()

            return ""

        if text.lower().strip(" .!?,") in HALLUCINATIONS:
            return ""

        return text

    def _emit_final(self, buf, voiced_ms):
        if voiced_ms < 250 or not buf or self.model is None:
            return

        text = self._transcribe(
            np.concatenate(buf),
            final=True,
        )

        if not text:
            return

        self.ui_q.put(("final", text))

        if self.cfg.log:
            try:
                with open(self.cfg.log, "a", encoding="utf-8") as f:
                    f.write(
                        f"[{time.strftime('%H:%M:%S')}] {text}\n"
                    )
            except OSError as e:
                self.ui_q.put(("error", f"Cannot write log: {e}"))

    def run(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.ui_q.put(
                (
                    "error",
                    "Missing dependency: pip install --user --break-system-packages faster-whisper",
                )
            )
            self.stop.set()
            return

        self._WhisperModel = WhisperModel

        if not self._load():
            return

        a = self.cfg

        preroll = deque(maxlen=4)
        buf = []
        speaking = False
        silence_ms = 0
        voiced_ms = 0
        last_partial = 0.0
        noise = 0.003

        while not self.stop.is_set():

            if not speaking and (
                self.model is None
                or self._key() != self.loaded_key
            ):
                if not self._load():
                    return

                preroll.clear()

            try:
                chunk = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            voiced = rms > max(a.threshold, noise * 3)

            if not speaking:
                if voiced:
                    speaking = True
                    buf = list(preroll) + [chunk]
                    silence_ms = 0
                    voiced_ms = CHUNK_MS
                    last_partial = time.monotonic()
                else:
                    preroll.append(chunk)
                    noise = 0.95 * noise + 0.05 * rms

                continue

            buf.append(chunk)

            if voiced:
                silence_ms = 0
                voiced_ms += CHUNK_MS
            else:
                silence_ms += CHUNK_MS

            dur = len(buf) * CHUNK_MS / 1000

            if (
                silence_ms >= a.silence_ms
                or dur >= a.max_utterance
            ):
                self._emit_final(buf, voiced_ms)
                buf = []
                speaking = False
                preroll.clear()

            elif (
                time.monotonic() - last_partial >= a.partial_interval
                and dur >= 0.6
                and self.audio_q.qsize() < 5
                and self.model is not None
            ):
                text = self._transcribe(
                    np.concatenate(buf),
                    final=False,
                )

                last_partial = time.monotonic()

                if text:
                    self.ui_q.put(("partial", text))


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
class CaptionState:
    """Holds committed + in-progress text."""

    def __init__(self):
        self.committed = ""
        self.partial = ""
        self.status = ""
        self.last_update = 0.0

    def handle(self, kind, text):
        self.last_update = time.monotonic()

        if kind in ("final", "preview"):
            self.committed = (
                self.committed + " " + text
            ).strip()[-2000:]
            self.partial = ""

        elif kind == "partial":
            self.partial = text

        else:
            self.status = text

    def clear(self):
        self.committed = ""
        self.partial = ""

    def render(self, max_chars):
        """
        Return caption text.

        The old implementation relied only on max_chars, which meant
        Tkinter could wrap the resulting text onto a third visual line.

        This implementation deliberately inserts newlines based on the
        actual available pixel width and font metrics. The Overlay passes
        the configured line count to this method.
        """
        full = (
            self.committed + " " + self.partial
        ).strip()

        if not full:
            return self.status

        if max_chars and len(full) > max_chars:
            full = full[-max_chars:]
            i = full.find(" ")

            if i != -1:
                full = "…" + full[i:]
            else:
                full = "…" + full

        return full


def _contrast(hexcolor):
    try:
        r, g, b = (
            int(hexcolor[i:i + 2], 16)
            for i in (1, 3, 5)
        )

        return (
            "black"
            if (
                0.299 * r
                + 0.587 * g
                + 0.114 * b
            ) > 150
            else "white"
        )

    except (ValueError, IndexError):
        return "black"


class Overlay:
    def __init__(self, cfg, ui_q, stop, reader):
        import tkinter as tk
        import tkinter.font as tkfont
        import tkinter.ttk as ttk
        from tkinter import colorchooser, filedialog, messagebox

        self.tk = tk
        self.tkfont = tkfont
        self.ttk = ttk

        self.colorchooser = colorchooser
        self.filedialog = filedialog
        self.messagebox = messagebox

        self.cfg = cfg
        self.ui_q = ui_q
        self.stop = stop
        self.reader = reader

        self.state = CaptionState()

        self.transcript = []
        self.transcript_win = None
        self.transcript_text = None
        self.settings_win = None

        self.root = tk.Tk()
        self.root.title("livecaption")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        self.font = tkfont.Font(
            family=cfg.font_family,
            size=-int(cfg.font_size),
            weight="bold" if cfg.bold else "normal",
        )

        self.label = tk.Label(
            self.root,
            text="",
            font=self.font,
            justify="center",
            anchor="center",
        )

        self.label.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10,
        )

        self.sw = self.root.winfo_screenwidth()
        self.sh = self.root.winfo_screenheight()

        self.apply_display(initial=True)

        self._drag = (0, 0)

        for w in (self.root, self.label):
            w.bind("<Button-1>", self._start_drag)
            w.bind("<B1-Motion>", self._do_drag)
            w.bind("<ButtonRelease-1>", self._end_drag)
            w.bind(
                "<Double-Button-1>",
                lambda e: self.open_settings(),
            )
            w.bind("<Button-3>", self._menu)

        self.pause_var = tk.BooleanVar(value=False)

        pos = tk.Menu(self.root, tearoff=0)

        pos.add_command(
            label="Top of screen",
            command=lambda: self._place("top"),
        )

        pos.add_command(
            label="Middle of screen",
            command=lambda: self._place("center"),
        )

        pos.add_command(
            label="Bottom of screen",
            command=lambda: self._place("bottom"),
        )

        self.menu = tk.Menu(self.root, tearoff=0)

        self.menu.add_checkbutton(
            label="Pause captions",
            variable=self.pause_var,
            command=self.toggle_pause,
        )

        self.menu.add_command(
            label="Settings…",
            command=self.open_settings,
        )

        self.menu.add_command(
            label="Transcript…",
            command=self.open_transcript,
        )

        self.menu.add_cascade(
            label="Position",
            menu=pos,
        )

        self.menu.add_command(
            label="Clear captions",
            command=self.state.clear,
        )

        self.menu.add_separator()

        self.menu.add_command(
            label="Quit",
            command=self.stop.set,
        )

        self.root.after(50, self._poll)

    # -- layout ---------------------------------------------------------- #
    def apply_display(self, initial=False):
        c = self.cfg

        self.font.configure(
            family=c.font_family,
            size=-int(c.font_size),
            weight="bold" if c.bold else "normal",
        )

        self.label.configure(
            fg=c.fg,
            bg=c.bg,
        )

        self.root.configure(bg=c.bg)

        self.width = max(
            200,
            int(self.sw * c.width_pct / 100),
        )

        # Use actual font metrics instead of a character approximation.
        self._line_width = max(
            100,
            self.width - 40,
        )

        height = (
            self.font.metrics("linespace")
            * int(c.lines)
            + 30
        )

        self.label.configure(
            wraplength=self._line_width,
        )

        if initial:
            if c.pos:
                x, y = c.pos
            else:
                x = (self.sw - self.width) // 2
                y = (
                    self.sh
                    - height
                    - int(self.sh * 0.06)
                )
        else:
            x = self.root.winfo_x()
            y = self.root.winfo_y()

        x = max(
            0,
            min(int(x), self.sw - 100),
        )

        y = max(
            0,
            min(int(y), self.sh - 40),
        )

        self.root.geometry(
            f"{self.width}x{height}+{x}+{y}"
        )

    def _place(self, where):
        height = self.root.winfo_height()

        x = (self.sw - self.width) // 2

        y = {
            "top": int(self.sh * 0.05),
            "center": (self.sh - height) // 2,
            "bottom": (
                self.sh
                - height
                - int(self.sh * 0.06)
            ),
        }[where]

        self.cfg.pos = [x, y]

        self.root.geometry(
            f"{self.width}x{height}+{x}+{y}"
        )

    def _max_chars(self):
        """
        Estimate a character limit as a secondary safety measure.

        The actual line limit is enforced by _fit_caption_to_lines().
        """
        if self.cfg.max_chars:
            return int(self.cfg.max_chars)

        per_line = max(
            10,
            int(
                self._line_width
                / max(1, self.font.measure("n"))
            ),
        )

        return int(
            per_line
            * int(self.cfg.lines)
            * 0.95
        )

    def _fit_caption_to_lines(self, text):
        """
        Force the caption to fit within exactly cfg.lines visual lines.

        Tkinter's wrapping is pixel-based, so this uses the actual font
        measurement rather than assuming every character has the same width.
        """
        if not text:
            return text

        max_lines = max(
            1,
            int(self.cfg.lines),
        )

        width = max(
            50,
            self._line_width,
        )

        words = text.split()

        if not words:
            return ""

        lines = []
        current = ""

        for word in words:
            candidate = (
                word
                if not current
                else current + " " + word
            )

            if self.font.measure(candidate) <= width:
                current = candidate
                continue

            if current:
                lines.append(current)

            current = word

            if len(lines) >= max_lines:
                break

            # Handle a single word wider than the available width.
            if self.font.measure(current) > width:
                while (
                    len(current) > 1
                    and self.font.measure(current + "…") > width
                ):
                    current = current[:-1]

        if current and len(lines) < max_lines:
            lines.append(current)

        # If there was more text than we displayed, add an ellipsis.
        displayed_words = " ".join(lines)

        if len(displayed_words) < len(text):
            if lines:
                last = lines[-1]

                while (
                    last
                    and self.font.measure(last + "…") > width
                ):
                    last = last[:-1]

                lines[-1] = (
                    last.rstrip() + "…"
                    if last
                    else "…"
                )

        return "\n".join(lines[:max_lines])

    def _start_drag(self, e):
        self._drag = (
            e.x_root - self.root.winfo_x(),
            e.y_root - self.root.winfo_y(),
        )

    def _do_drag(self, e):
        self.root.geometry(
            f"+{e.x_root - self._drag[0]}"
            f"+{e.y_root - self._drag[1]}"
        )

    def _end_drag(self, _e):
        self.cfg.pos = [
            self.root.winfo_x(),
            self.root.winfo_y(),
        ]

    def _menu(self, e):
        try:
            self.menu.tk_popup(
                e.x_root,
                e.y_root,
            )
        finally:
            self.menu.grab_release()

    def toggle_pause(self):
        self.cfg.paused = self.pause_var.get()

        self.state.clear()

        self.state.status = (
            "⏸ Paused"
            if self.cfg.paused
            else "Listening…"
        )

    # -- transcript window ----------------------------------------------- #
    def open_transcript(self):
        tk = self.tk

        if (
            self.transcript_win is not None
            and self.transcript_win.winfo_exists()
        ):
            self.transcript_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Live Caption – Transcript")
        win.geometry("580x380")
        win.attributes("-topmost", True)

        bar = tk.Frame(win)
        bar.pack(
            side="bottom",
            fill="x",
            padx=6,
            pady=6,
        )

        text = tk.Text(
            win,
            wrap="word",
            state="disabled",
            padx=8,
            pady=6,
        )

        sb = tk.Scrollbar(
            win,
            command=text.yview,
        )

        text.configure(
            yscrollcommand=sb.set,
        )

        sb.pack(
            side="right",
            fill="y",
        )

        text.pack(
            side="left",
            fill="both",
            expand=True,
        )

        def save():
            path = self.filedialog.asksaveasfilename(
                parent=win,
                defaultextension=".txt",
                initialfile="captions.txt",
                filetypes=[
                    ("Text", "*.txt"),
                    ("All files", "*"),
                ],
            )

            if path:
                with open(
                    path,
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        self._transcript_text()
                    )

        def copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(
                self._transcript_text()
            )

        def clear():
            self.transcript.clear()

            text.configure(state="normal")
            text.delete("1.0", "end")
            text.configure(state="disabled")

        for label, cmd in (
            ("Save…", save),
            ("Copy all", copy),
            ("Clear", clear),
        ):
            tk.Button(
                bar,
                text=label,
                command=cmd,
            ).pack(
                side="left",
                padx=4,
            )

        self.transcript_win = win
        self.transcript_text = text

        for ts, t in self.transcript:
            self._append_transcript(ts, t)

    def _transcript_text(self):
        return "".join(
            f"[{ts}] {t}\n"
            for ts, t in self.transcript
        )

    def _append_transcript(self, ts, t):
        w = self.transcript_text

        if (
            self.transcript_win is None
            or not self.transcript_win.winfo_exists()
        ):
            return

        w.configure(state="normal")

        w.insert(
            "end",
            f"[{ts}] {t}\n",
        )

        w.see("end")
        w.configure(state="disabled")

    # -- settings window ------------------------------------------------- #
    def open_settings(self):
        tk, ttk, c = (
            self.tk,
            self.ttk,
            self.cfg,
        )

        if (
            self.settings_win is not None
            and self.settings_win.winfo_exists()
        ):
            self.settings_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Live Caption – Settings")
        win.attributes("-topmost", True)
        win.resizable(False, False)

        self.settings_win = win

        v = dict(
            source=tk.StringVar(value=c.source),
            threshold=tk.DoubleVar(value=c.threshold),
            silence_ms=tk.IntVar(value=c.silence_ms),
            max_utterance=tk.DoubleVar(value=c.max_utterance),
            model=tk.StringVar(value=c.model),
            language=tk.StringVar(
                value=c.language or "auto"
            ),
            translate=tk.BooleanVar(
                value=c.translate
            ),
            device=tk.StringVar(value=c.device),
            compute_type=tk.StringVar(
                value=c.compute_type
            ),
            partial_interval=tk.DoubleVar(
                value=c.partial_interval
            ),
            font_family=tk.StringVar(
                value=c.font_family
            ),
            font_size=tk.IntVar(
                value=c.font_size
            ),
            bold=tk.BooleanVar(value=c.bold),
            lines=tk.IntVar(value=c.lines),
            width_pct=tk.IntVar(value=c.width_pct),
            opacity=tk.DoubleVar(value=c.opacity),
            idle_opacity=tk.DoubleVar(
                value=c.idle_opacity
            ),
            fg=tk.StringVar(value=c.fg),
            bg=tk.StringVar(value=c.bg),
            clear_after=tk.DoubleVar(
                value=c.clear_after
            ),
            log_on=tk.BooleanVar(
                value=bool(c.log)
            ),
            log=tk.StringVar(
                value=c.log
                or os.path.expanduser(
                    "~/livecaption-log.txt"
                )
            ),
        )

        nb = ttk.Notebook(win)

        nb.pack(
            fill="both",
            expand=True,
            padx=8,
            pady=8,
        )

        tabs = {}

        for name in (
            "Audio",
            "Recognition",
            "Display",
            "Output",
        ):
            f = ttk.Frame(nb)

            nb.add(
                f,
                text=name,
            )

            f.columnconfigure(
                1,
                weight=1,
            )

            tabs[name] = f

        def add(tab, r, text, widget):
            ttk.Label(
                tabs[tab],
                text=text,
            ).grid(
                row=r,
                column=0,
                sticky="w",
                padx=8,
                pady=5,
            )

            widget.grid(
                row=r,
                column=1,
                sticky="we",
                padx=8,
                pady=5,
            )

        def scale(tab, var, lo, hi, step):
            return tk.Scale(
                tabs[tab],
                from_=lo,
                to=hi,
                resolution=step,
                orient="horizontal",
                variable=var,
                length=240,
            )

        def spin(tab, var, lo, hi, step):
            return ttk.Spinbox(
                tabs[tab],
                from_=lo,
                to=hi,
                increment=step,
                textvariable=var,
                width=8,
            )

        def color_btn(var):
            b = tk.Button(
                tabs["Display"],
                width=12,
                textvariable=var,
                relief="solid",
            )

            def refresh(*_):
                try:
                    b.configure(
                        bg=var.get(),
                        fg=_contrast(var.get()),
                        activebackground=var.get(),
                        activeforeground=_contrast(
                            var.get()
                        ),
                    )
                except tk.TclError:
                    pass

            def pick():
                _, hexv = self.colorchooser.askcolor(
                    color=var.get(),
                    parent=win,
                )

                if hexv:
                    var.set(hexv)

            var.trace_add(
                "write",
                refresh,
            )

            refresh()

            b.configure(command=pick)

            return b

        # Audio tab
        src_box = ttk.Combobox(
            tabs["Audio"],
            textvariable=v["source"],
            values=[
                "system",
                "mic",
            ] + get_sources(),
            width=44,
        )

        add(
            "Audio",
            0,
            "Audio source",
            src_box,
        )

        def refresh_sources():
            src_box.configure(
                values=[
                    "system",
                    "mic",
                ] + get_sources()
            )

        ttk.Button(
            tabs["Audio"],
            text="Refresh source list",
            command=refresh_sources,
        ).grid(
            row=1,
            column=1,
            sticky="w",
            padx=8,
        )

        ttk.Label(
            tabs["Audio"],
            text="'system' = what you hear, 'mic' = microphone",
            foreground="gray",
        ).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            padx=8,
        )

        add(
            "Audio",
            3,
            "Speech sensitivity\n(lower = more sensitive)",
            scale(
                "Audio",
                v["threshold"],
                0.002,
                0.1,
                0.002,
            ),
        )

        add(
            "Audio",
            4,
            "End caption after silence (ms)",
            spin(
                "Audio",
                v["silence_ms"],
                300,
                3000,
                100,
            ),
        )

        add(
            "Audio",
            5,
            "Max seconds per caption",
            spin(
                "Audio",
                v["max_utterance"],
                4,
                30,
                1,
            ),
        )

        # Recognition tab
        add(
            "Recognition",
            0,
            "Model",
            ttk.Combobox(
                tabs["Recognition"],
                textvariable=v["model"],
                width=22,
                values=[
                    "tiny",
                    "tiny.en",
                    "base",
                    "base.en",
                    "small",
                    "small.en",
                    "medium",
                    "medium.en",
                    "large-v3",
                    "distil-large-v3",
                ],
            ),
        )

        add(
            "Recognition",
            1,
            "Language",
            ttk.Combobox(
                tabs["Recognition"],
                textvariable=v["language"],
                width=22,
                values=[
                    "auto",
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "nl",
                    "ru",
                    "uk",
                    "pl",
                    "tr",
                    "ar",
                    "hi",
                    "zh",
                    "ja",
                    "ko",
                    "sv",
                ],
            ),
        )

        add(
            "Recognition",
            2,
            "",
            ttk.Checkbutton(
                tabs["Recognition"],
                text="Translate speech to English",
                variable=v["translate"],
            ),
        )

        add(
            "Recognition",
            3,
            "Device",
            ttk.Combobox(
                tabs["Recognition"],
                textvariable=v["device"],
                values=[
                    "cpu",
                    "cuda",
                ],
                width=22,
                state="readonly",
            ),
        )

        add(
            "Recognition",
            4,
            "Compute type",
            ttk.Combobox(
                tabs["Recognition"],
                textvariable=v["compute_type"],
                width=22,
                values=[
                    "auto",
                    "int8",
                    "int8_float16",
                    "float16",
                    "float32",
                ],
            ),
        )

        add(
            "Recognition",
            5,
            "Live update interval (s)",
            scale(
                "Recognition",
                v["partial_interval"],
                0.5,
                3.0,
                0.1,
            ),
        )

        ttk.Label(
            tabs["Recognition"],
            text="Changing model/device reloads the model (first use downloads it).",
            foreground="gray",
        ).grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="w",
            padx=8,
            pady=4,
        )

        # Display tab
        families = sorted(
            set(
                self.tkfont.families()
            )
        )

        add(
            "Display",
            0,
            "Font",
            ttk.Combobox(
                tabs["Display"],
                textvariable=v["font_family"],
                values=families,
                width=26,
            ),
        )

        add(
            "Display",
            1,
            "Font size (px)",
            spin(
                "Display",
                v["font_size"],
                14,
                120,
                2,
            ),
        )

        add(
            "Display",
            2,
            "",
            ttk.Checkbutton(
                tabs["Display"],
                text="Bold",
                variable=v["bold"],
            ),
        )

        add(
            "Display",
            3,
            "Lines of text",
            spin(
                "Display",
                v["lines"],
                1,
                6,
                1,
            ),
        )

        add(
            "Display",
            4,
            "Width (% of screen)",
            scale(
                "Display",
                v["width_pct"],
                20,
                100,
                5,
            ),
        )

        add(
            "Display",
            5,
            "Text color",
            color_btn(v["fg"]),
        )

        add(
            "Display",
            6,
            "Background color",
            color_btn(v["bg"]),
        )

        add(
            "Display",
            7,
            "Opacity (with captions)",
            scale(
                "Display",
                v["opacity"],
                0.2,
                1.0,
                0.05,
            ),
        )

        add(
            "Display",
            8,
            "Opacity (idle)",
            scale(
                "Display",
                v["idle_opacity"],
                0.0,
                1.0,
                0.05,
            ),
        )

        add(
            "Display",
            9,
            "Clear captions after (s)",
            spin(
                "Display",
                v["clear_after"],
                2,
                60,
                1,
            ),
        )

        # Output tab
        add(
            "Output",
            0,
            "",
            ttk.Checkbutton(
                tabs["Output"],
                text="Save captions to a log file",
                variable=v["log_on"],
            ),
        )

        row = ttk.Frame(tabs["Output"])

        ttk.Entry(
            row,
            textvariable=v["log"],
            width=34,
        ).pack(
            side="left",
            fill="x",
            expand=True,
        )

        def browse():
            p = self.filedialog.asksaveasfilename(
                parent=win,
                initialfile=os.path.basename(
                    v["log"].get()
                ),
            )

            if p:
                v["log"].set(p)

        ttk.Button(
            row,
            text="Browse…",
            command=browse,
        ).pack(
            side="left",
            padx=4,
        )

        add(
            "Output",
            1,
            "Log file",
            row,
        )

        ttk.Button(
            tabs["Output"],
            text="Open transcript window…",
            command=self.open_transcript,
        ).grid(
            row=2,
            column=1,
            sticky="w",
            padx=8,
            pady=8,
        )

        # Actions
        def apply(save=False):
            try:
                new = dict(
                    source=v["source"].get().strip() or "system",
                    threshold=float(
                        v["threshold"].get()
                    ),
                    silence_ms=int(
                        v["silence_ms"].get()
                    ),
                    max_utterance=float(
                        v["max_utterance"].get()
                    ),
                    model=v["model"].get().strip() or "base",
                    language=(
                        None
                        if v["language"].get().strip()
                        in ("", "auto")
                        else v["language"].get().strip()
                    ),
                    translate=bool(
                        v["translate"].get()
                    ),
                    device=v["device"].get(),
                    compute_type=(
                        v["compute_type"].get().strip()
                        or "auto"
                    ),
                    partial_interval=float(
                        v["partial_interval"].get()
                    ),
                    font_family=(
                        v["font_family"].get().strip()
                        or "DejaVu Sans"
                    ),
                    font_size=max(
                        10,
                        int(v["font_size"].get()),
                    ),
                    bold=bool(
                        v["bold"].get()
                    ),
                    lines=max(
                        1,
                        int(v["lines"].get()),
                    ),
                    width_pct=max(
                        20,
                        min(
                            100,
                            int(v["width_pct"].get()),
                        ),
                    ),
                    opacity=max(
                        0.1,
                        min(
                            1.0,
                            float(v["opacity"].get()),
                        ),
                    ),
                    idle_opacity=max(
                        0.0,
                        min(
                            1.0,
                            float(v["idle_opacity"].get()),
                        ),
                    ),
                    fg=v["fg"].get(),
                    bg=v["bg"].get(),
                    clear_after=float(
                        v["clear_after"].get()
                    ),
                    log=(
                        v["log"].get().strip()
                        if v["log_on"].get()
                        and v["log"].get().strip()
                        else None
                    ),
                )

            except (tk.TclError, ValueError) as e:
                self.messagebox.showerror(
                    "Invalid setting",
                    str(e),
                    parent=win,
                )
                return

            source_changed = (
                new["source"] != c.source
            )

            for k, val in new.items():
                setattr(c, k, val)

            self.apply_display()

            if source_changed:
                self.reader.restart_now()

            if save:
                try:
                    save_config(c)
                    self.ui_q.put(
                        ("status", "Settings saved")
                    )
                except OSError as e:
                    self.messagebox.showerror(
                        "Save failed",
                        str(e),
                        parent=win,
                    )

        def test():
            apply()

            self.ui_q.put(
                (
                    "preview",
                    "This is a sample caption to preview your settings.",
                )
            )

        btns = ttk.Frame(win)

        btns.pack(
            fill="x",
            padx=8,
            pady=(0, 8),
        )

        ttk.Button(
            btns,
            text="Test caption",
            command=test,
        ).pack(side="left")

        ttk.Button(
            btns,
            text="Close",
            command=win.destroy,
        ).pack(side="right")

        ttk.Button(
            btns,
            text="Save",
            command=lambda: apply(save=True),
        ).pack(
            side="right",
            padx=4,
        )

        ttk.Button(
            btns,
            text="Apply",
            command=apply,
        ).pack(side="right")

    # -- main loop ------------------------------------------------------- #
    def _poll(self):
        try:
            while True:
                kind, text = self.ui_q.get_nowait()

                self.state.handle(
                    kind,
                    text,
                )

                if kind == "final":
                    ts = time.strftime("%H:%M:%S")

                    self.transcript.append(
                        (ts, text)
                    )

                    self._append_transcript(
                        ts,
                        text,
                    )

                    print(
                        text,
                        flush=True,
                    )

                elif kind == "error":
                    print(
                        f"ERROR: {text}",
                        file=sys.stderr,
                        flush=True,
                    )

                elif kind == "status":
                    print(
                        text,
                        file=sys.stderr,
                        flush=True,
                    )

        except queue.Empty:
            pass

        if self.stop.is_set():
            self.root.destroy()
            return

        s = self.state
        c = self.cfg

        if (
            (s.committed or s.partial)
            and time.monotonic() - s.last_update
            > float(c.clear_after)
        ):
            s.clear()
            s.status = (
                "⏸ Paused"
                if c.paused
                else ""
            )

        # -----------------------------------------------------------------
        # IMPORTANT:
        #
        # Fit the caption to the configured number of lines before giving
        # it to Tkinter. This prevents Tkinter's automatic wrapping from
        # creating a third line.
        # -----------------------------------------------------------------
        rendered = s.render(
            self._max_chars()
        )

        if s.committed or s.partial:
            rendered = self._fit_caption_to_lines(
                rendered
            )

        self.label.configure(
            text=rendered
        )

        active = bool(
            s.committed or s.partial
        )

        try:
            self.root.attributes(
                "-alpha",
                float(
                    c.opacity
                    if active
                    else c.idle_opacity
                ),
            )
        except self.tk.TclError:
            pass

        self.root.after(
            50,
            self._poll,
        )

    def run(self):
        self.root.mainloop()


def run_terminal(ui_q, stop):
    cols = (
        shutil.get_terminal_size().columns - 1
    )

    partial_shown = False

    while not stop.is_set():
        try:
            kind, text = ui_q.get(
                timeout=0.2
            )
        except queue.Empty:
            continue

        if kind == "partial":
            print(
                "\r"
                + text[-cols:].ljust(cols),
                end="",
                flush=True,
            )

            partial_shown = True

        else:
            if partial_shown:
                print(
                    "\r"
                    + " " * cols
                    + "\r",
                    end="",
                )

                partial_shown = False

            print(
                text
                if kind == "final"
                else f"[{kind}] {text}",
                flush=True,
            )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    cfg, extra = build_config()

    if extra.list_sources:
        list_sources()
        return

    if extra.reset_config:
        try:
            os.remove(CONFIG_PATH)
            print(
                f"Removed {CONFIG_PATH}"
            )
        except OSError:
            print(
                "No saved settings to remove."
            )
        return

    if not shutil.which("parec"):
        sys.exit(
            "parec not found. Install it: "
            "sudo apt install pulseaudio-utils"
        )

    stop = threading.Event()

    signal.signal(
        signal.SIGINT,
        lambda *_: stop.set(),
    )

    signal.signal(
        signal.SIGTERM,
        lambda *_: stop.set(),
    )

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    ui_q: "queue.Queue[tuple]" = queue.Queue()

    reader = AudioReader(
        cfg,
        audio_q,
        ui_q,
        stop,
        fatal=extra.no_gui,
    )

    reader.start()

    Transcriber(
        cfg,
        audio_q,
        ui_q,
        stop,
    ).start()

    if extra.no_gui:
        run_terminal(
            ui_q,
            stop,
        )

    else:
        try:
            Overlay(
                cfg,
                ui_q,
                stop,
                reader,
            ).run()

        except Exception as e:
            print(
                f"GUI unavailable ({e}); falling back to terminal.",
                file=sys.stderr,
            )

            run_terminal(
                ui_q,
                stop,
            )

    stop.set()

    sys.stdout.flush()
    sys.stderr.flush()

    os._exit(0)


if __name__ == "__main__":
    main()
