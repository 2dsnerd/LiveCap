#!/usr/bin/env python3
"""
LiveCaption - offline live captions for Windows.

System audio:
    Uses PyAudioWPatch's WASAPI loopback support.

Microphone:
    Uses sounddevice.

Transcription:
    faster-whisper.

Display:
    Tkinter always-on-top caption overlay.

Requirements
------------
Recommended Python:
    Python 3.13 on Windows

Install:
    py -m pip install numpy faster-whisper sounddevice PyAudioWPatch

Usage:
    python livecaption_windows.py
    python livecaption_windows.py --source system
    python livecaption_windows.py --source mic
    python livecaption_windows.py --no-gui
    python livecaption_windows.py --list-sources
    python livecaption_windows.py --diagnose-audio
    python livecaption_windows.py --reset-config

PyInstaller:
    py -m PyInstaller --noconfirm --onedir --windowed --collect-all faster_whisper --collect-all ctranslate2 --collect-all onnxruntime --collect-all soundcard livecaption_windows.py
"""

import argparse
import json
import os
import queue
import shutil
import signal
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_MS / 1000)

HALLUCINATIONS = {
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "bye",
    "you",
    "subtitles by the amara.org community",
}

DEFAULTS = dict(
    source="system",
    threshold=0.01,
    silence_ms=700,
    max_utterance=12.0,

    model="base",
    language=None,
    translate=False,
    device="cpu",
    compute_type="auto",
    partial_interval=1.0,

    lines=2,
    font_family="Segoe UI",
    font_size=32,
    bold=True,
    fg="#ffffff",
    bg="#000000",
    opacity=0.85,
    idle_opacity=0.25,
    width_pct=80,
    max_chars=0,
    clear_after=6.0,
    pos=None,

    log=None,
)

CONFIG_PATH = os.path.join(
    os.environ.get("APPDATA", os.path.expanduser("~")),
    "livecaption",
    "config.json",
)


# --------------------------------------------------------------------------- #
# Safe console helpers
# --------------------------------------------------------------------------- #

def safe_print(*args, **kwargs):
    """
    Safe print for normal Python and PyInstaller --windowed.

    With --windowed, Windows can set sys.stdout/sys.stderr to None.
    """
    stream = kwargs.get("file")

    if stream is None:
        stream = sys.stdout

    if stream is None:
        return

    try:
        print(*args, **kwargs)
    except (AttributeError, OSError):
        pass


def safe_flush():
    """Flush stdout/stderr without crashing a --windowed executable."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            try:
                stream.flush()
            except (AttributeError, OSError):
                pass


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)

        return {
            k: v
            for k, v in data.items()
            if k in DEFAULTS
        }

    except (OSError, ValueError, TypeError):
        return {}


def save_config(cfg) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                k: getattr(cfg, k)
                for k in DEFAULTS
            },
            f,
            indent=2,
        )


# --------------------------------------------------------------------------- #
# Dependency imports
# --------------------------------------------------------------------------- #

def _import_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        raise RuntimeError(
            "sounddevice is not installed.\n\n"
            "Install it with:\n"
            "  py -3.13 -m pip install sounddevice"
        )


def _import_pyaudiowpatch():
    try:
        import pyaudiowpatch as pyaudio
        return pyaudio
    except ImportError:
        raise RuntimeError(
            "PyAudioWPatch is not installed.\n\n"
            "Install it with:\n"
            "  py -3.13 -m pip install PyAudioWPatch"
        )


# --------------------------------------------------------------------------- #
# Audio device helpers
# --------------------------------------------------------------------------- #

def get_sounddevice_devices():
    sd = _import_sounddevice()

    result = []

    for i, d in enumerate(sd.query_devices()):
        result.append(
            {
                "index": i,
                "name": d["name"],
                "inputs": int(d["max_input_channels"]),
                "outputs": int(d["max_output_channels"]),
                "hostapi": sd.query_hostapis(d["hostapi"])["name"],
                "hostapi_idx": int(d["hostapi"]),
                "default_samplerate": float(
                    d.get("default_samplerate", SAMPLE_RATE)
                ),
            }
        )

    return result


def get_pyaudio_loopbacks():
    """
    Return WASAPI loopback devices from PyAudioWPatch.

    PyAudioWPatch exposes the speaker loopback as an input device.
    """
    pyaudio = _import_pyaudiowpatch()

    devices = []

    with pyaudio.PyAudio() as p:
        try:
            for info in p.get_loopback_device_info_generator():
                devices.append(dict(info))
        except Exception:
            pass

    return devices


def get_default_loopback():
    """
    Find the default Windows playback device's WASAPI loopback device.
    """
    pyaudio = _import_pyaudiowpatch()

    with pyaudio.PyAudio() as p:
        try:
            return dict(p.get_default_wasapi_loopback())
        except LookupError:
            raise RuntimeError(
                "Windows WASAPI loopback device was not found."
            )
        except OSError as e:
            raise RuntimeError(
                f"WASAPI is not available: {e}"
            )


def find_loopback_by_name(name):
    """
    Find a loopback device by a case-insensitive name match.
    """
    wanted = name.lower().strip()

    devices = get_pyaudio_loopbacks()

    for d in devices:
        device_name = str(d.get("name", ""))

        if wanted in device_name.lower():
            return d

    return None


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def diagnose_audio():
    safe_print()
    safe_print("=" * 70)
    safe_print("LiveCaption audio diagnostic")
    safe_print("=" * 70)

    # PyAudioWPatch
    try:
        pyaudio = _import_pyaudiowpatch()

        safe_print()
        safe_print("WASAPI loopback devices:")

        with pyaudio.PyAudio() as p:
            try:
                default = p.get_default_wasapi_loopback()

                safe_print(
                    f"  DEFAULT: [{default['index']}] "
                    f"{default['name']}"
                )

                safe_print(
                    f"  rate={default['defaultSampleRate']} "
                    f"channels={default['maxInputChannels']}"
                )

            except Exception as e:
                safe_print(f"  Could not find default loopback: {e}")

            safe_print()

            try:
                for d in p.get_loopback_device_info_generator():
                    safe_print(
                        f"  [{d['index']}] {d['name']}"
                    )
                    safe_print(
                        f"      rate={d.get('defaultSampleRate')} "
                        f"channels={d.get('maxInputChannels')}"
                    )

            except Exception as e:
                safe_print(
                    f"  Failed enumerating loopback devices: {e}"
                )

    except Exception as e:
        safe_print()
        safe_print("PyAudioWPatch:")
        safe_print(f"  ERROR: {e}")

    # Microphone / sounddevice
    try:
        sd = _import_sounddevice()

        safe_print()
        safe_print("sounddevice input devices:")

        for d in get_sounddevice_devices():
            if d["inputs"] > 0:
                safe_print(
                    f"  [{d['index']}] {d['name']} "
                    f"(inputs={d['inputs']}, "
                    f"rate={d['default_samplerate']})"
                )

    except Exception as e:
        safe_print()
        safe_print(f"sounddevice diagnostic failed: {e}")

    safe_print()
    safe_print("=" * 70)
    safe_print()


def list_sources():
    safe_print()
    safe_print("Available audio sources:")
    safe_print()

    # System audio
    try:
        default = get_default_loopback()

        safe_print(
            f"system  -> [{default['index']}] "
            f"{default['name']}"
        )

    except Exception as e:
        safe_print(f"system  -> ERROR: {e}")

    # Microphone
    try:
        devices = get_sounddevice_devices()

        for d in devices:
            if d["inputs"] > 0:
                safe_print(
                    f"mic     -> [{d['index']}] "
                    f"{d['name']}"
                )

    except Exception as e:
        safe_print(f"mic     -> ERROR: {e}")

    # Loopbacks
    try:
        for d in get_pyaudio_loopbacks():
            safe_print(
                f"loopback -> [{d['index']}] "
                f"{d['name']}"
            )

    except Exception as e:
        safe_print(f"loopback -> ERROR: {e}")

    safe_print()


# --------------------------------------------------------------------------- #
# Command-line config
# --------------------------------------------------------------------------- #

def build_config(argv=None):
    p = argparse.ArgumentParser(
        description="Offline live captions for Windows."
    )

    p.add_argument(
        "--source",
        help="'system', 'mic', or an audio device name/index",
    )

    p.add_argument(
        "--list-sources",
        action="store_true",
        help="list audio devices and exit",
    )

    p.add_argument(
        "--model",
        help=(
            "tiny, base, small, medium, large-v3, "
            "distil-large-v3, *.en ..."
        ),
    )

    p.add_argument(
        "--language",
        help="e.g. en, es, de, or 'auto'",
    )

    p.add_argument(
        "--translate",
        action=argparse.BooleanOptionalAction,
        help="translate speech to English",
    )

    p.add_argument(
        "--device",
        help="cpu or cuda",
    )

    p.add_argument(
        "--compute-type",
        dest="compute_type",
        help="auto, int8, float16, ...",
    )

    p.add_argument(
        "--threshold",
        type=float,
        help="minimum RMS level counted as speech",
    )

    p.add_argument(
        "--silence-ms",
        dest="silence_ms",
        type=int,
        help="silence that ends a caption",
    )

    p.add_argument(
        "--max-utterance",
        dest="max_utterance",
        type=float,
        help="maximum seconds per caption chunk",
    )

    p.add_argument(
        "--partial-interval",
        dest="partial_interval",
        type=float,
        help="seconds between live updates",
    )

    p.add_argument(
        "--lines",
        type=int,
        help="overlay lines",
    )

    p.add_argument(
        "--font-size",
        dest="font_size",
        type=int,
        help="overlay font size",
    )

    p.add_argument(
        "--opacity",
        type=float,
        help="overlay opacity 0-1",
    )

    p.add_argument(
        "--clear-after",
        dest="clear_after",
        type=float,
        help="seconds before captions clear",
    )

    p.add_argument(
        "--log",
        help="append finalized captions to this file",
    )

    p.add_argument(
        "--no-gui",
        action="store_true",
        help="print captions in terminal",
    )

    p.add_argument(
        "--reset-config",
        action="store_true",
        help="delete saved settings and exit",
    )

    p.add_argument(
        "--diagnose-audio",
        action="store_true",
        help="test Windows audio devices and exit",
    )

    args = vars(p.parse_args(argv))

    extra = SimpleNamespace(
        list_sources=args.pop("list_sources"),
        no_gui=args.pop("no_gui"),
        reset_config=args.pop("reset_config"),
        diagnose_audio=args.pop("diagnose_audio"),
    )

    merged = dict(DEFAULTS)
    merged.update(load_config())

    merged.update(
        {
            k: v
            for k, v in args.items()
            if v is not None
        }
    )

    if merged["language"] in ("auto", ""):
        merged["language"] = None

    cfg = SimpleNamespace(**merged)
    cfg.paused = False

    return cfg, extra


# --------------------------------------------------------------------------- #
# Audio reader
# --------------------------------------------------------------------------- #

class AudioReader(threading.Thread):
    """
    Windows audio capture.

    SYSTEM:
        PyAudioWPatch WASAPI loopback.

    MIC:
        sounddevice input stream.

    Important:
        The system-audio path does NOT use sounddevice's WASAPI loopback.
        PyAudioWPatch exposes the loopback endpoint as an input device,
        avoiding the Invalid number of channels (-9998) problem encountered
        with the previous implementation.
    """

    def __init__(
        self,
        cfg,
        audio_q,
        ui_q,
        stop,
        fatal=False,
    ):
        super().__init__(daemon=True)

        self.cfg = cfg
        self.audio_q = audio_q
        self.ui_q = ui_q
        self.stop = stop
        self.fatal = fatal

        self._restart = threading.Event()

        self._stream = None
        self._pyaudio = None
        self._sd_stream = None

    def restart_now(self):
        self._restart.set()

        stream = self._stream

        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass

            try:
                stream.close()
            except Exception:
                pass

        sd_stream = self._sd_stream

        if sd_stream is not None:
            try:
                sd_stream.abort()
            except Exception:
                pass

    @staticmethod
    def _resample(
        data: np.ndarray,
        from_rate: int,
    ) -> np.ndarray:

        if from_rate == SAMPLE_RATE:
            return data.astype(np.float32, copy=False)

        if len(data) <= 1:
            return data.astype(np.float32)

        ratio = SAMPLE_RATE / from_rate

        n_out = max(
            1,
            int(round(len(data) * ratio)),
        )

        x_old = np.linspace(
            0.0,
            1.0,
            len(data),
        )

        x_new = np.linspace(
            0.0,
            1.0,
            n_out,
        )

        return np.interp(
            x_new,
            x_old,
            data,
        ).astype(np.float32)

    def _queue_pcm16(self, raw, channels, rate):
        """
        Convert PyAudio int16 bytes to mono float32 at 16 kHz.
        """
        if not raw:
            return

        samples = np.frombuffer(
            raw,
            dtype=np.int16,
        )

        if channels > 1:
            usable = (
                len(samples) // channels
            ) * channels

            if usable <= 0:
                return

            samples = samples[:usable].reshape(
                -1,
                channels,
            )

            mono = samples.astype(
                np.float32
            ).mean(axis=1)

        else:
            mono = samples.astype(
                np.float32
            )

        mono /= 32768.0

        result = self._resample(
            mono,
            int(rate),
        )

        if len(result):
            self.audio_q.put(result)

    def _run_system(self):
        pyaudio = _import_pyaudiowpatch()

        self.ui_q.put(
            (
                "status",
                "Finding Windows WASAPI loopback device…",
            )
        )

        p = pyaudio.PyAudio()
        self._pyaudio = p

        try:
            # PyAudioWPatch has a direct helper for this.
            try:
                device = dict(
                    p.get_default_wasapi_loopback()
                )

            except Exception:
                # Fallback: manually locate the loopback.
                wasapi_info = p.get_host_api_info_by_type(
                    pyaudio.paWASAPI
                )

                default_output = p.get_device_info_by_index(
                    wasapi_info["defaultOutputDevice"]
                )

                device = None

                for loopback in p.get_loopback_device_info_generator():
                    if (
                        default_output["name"]
                        in loopback["name"]
                    ):
                        device = dict(loopback)
                        break

                if device is None:
                    raise RuntimeError(
                        "Could not find the WASAPI loopback "
                        "device for the default speakers."
                    )

            device_index = int(device["index"])
            rate = int(
                float(device["defaultSampleRate"])
            )

            channels = int(
                device.get(
                    "maxInputChannels",
                    2,
                )
            )

            if channels < 1:
                channels = 2

            # Keep this within sane bounds.
            channels = min(channels, 8)

            frames = max(
                256,
                int(rate * CHUNK_MS / 1000),
            )

            self.ui_q.put(
                (
                    "status",
                    f"Opening WASAPI loopback: "
                    f"{device['name']} "
                    f"at {rate} Hz, "
                    f"ch={channels}…",
                )
            )

            self.ui_q.put(
                (
                    "status",
                    "Opening WASAPI loopback…",
                )
            )

            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                frames_per_buffer=frames,
                input=True,
                input_device_index=device_index,
            )

            self._stream = stream

            self.ui_q.put(
                (
                    "status",
                    f"Listening to system audio: "
                    f"{device['name']}",
                )
            )

            while (
                not self.stop.is_set()
                and not self._restart.is_set()
            ):
                try:
                    raw = stream.read(
                        frames,
                        exception_on_overflow=False,
                    )

                except Exception as e:
                    if (
                        self.stop.is_set()
                        or self._restart.is_set()
                    ):
                        break

                    raise RuntimeError(
                        f"WASAPI loopback read failed: {e}"
                    )

                self._queue_pcm16(
                    raw,
                    channels,
                    rate,
                )

        finally:
            stream = self._stream
            self._stream = None

            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:
                    pass

                try:
                    stream.close()
                except Exception:
                    pass

            self._pyaudio = None

            try:
                p.terminate()
            except Exception:
                pass

    def _run_mic(self):
        sd = _import_sounddevice()

        self.ui_q.put(
            (
                "status",
                "Opening microphone input…",
            )
        )

        def callback(
            indata,
            frames,
            time_info,
            status,
        ):
            if self.cfg.paused:
                return

            if self.stop.is_set():
                return

            if status:
                # Don't spam the UI with routine PortAudio status messages.
                pass

            if indata is None:
                return

            data = np.asarray(
                indata,
                dtype=np.float32,
            )

            if data.ndim == 1:
                mono = data.copy()

            else:
                mono = data.mean(
                    axis=1
                ).astype(np.float32)

            if len(mono):
                self.audio_q.put(
                    self._resample(
                        mono,
                        int(native_rate),
                    )
                )

        try:
            device = None

            source = str(
                self.cfg.source
            )

            # Numeric source.
            try:
                device = int(source)
            except ValueError:
                pass

            # Name source.
            if device is None and source not in (
                "mic",
                "system",
            ):
                wanted = source.lower()

                for i, d in enumerate(
                    sd.query_devices()
                ):
                    if (
                        wanted
                        in str(d["name"]).lower()
                        and d["max_input_channels"] > 0
                    ):
                        device = i
                        break

            # Default microphone.
            if device is None:
                default_input = sd.default.device[0]

                if (
                    default_input is not None
                    and int(default_input) >= 0
                ):
                    device = int(default_input)

            if device is not None:
                info = sd.query_devices(device)

                native_rate = int(
                    float(
                        info.get(
                            "default_samplerate",
                            SAMPLE_RATE,
                        )
                    )
                )

                channels = max(
                    1,
                    min(
                        2,
                        int(
                            info[
                                "max_input_channels"
                            ]
                        ),
                    ),
                )

            else:
                native_rate = SAMPLE_RATE
                channels = 1

            with sd.InputStream(
                samplerate=native_rate,
                channels=channels,
                dtype="float32",
                blocksize=max(
                    256,
                    int(
                        native_rate
                        * CHUNK_MS
                        / 1000
                    ),
                ),
                callback=callback,
                device=device,
            ) as stream:

                self._sd_stream = stream

                self.ui_q.put(
                    (
                        "status",
                        "Listening to microphone…",
                    )
                )

                while (
                    not self.stop.is_set()
                    and not self._restart.is_set()
                ):
                    time.sleep(0.2)

        finally:
            self._sd_stream = None

    def run(self):
        while not self.stop.is_set():
            self._restart.clear()

            try:
                if self.cfg.source == "system":
                    self._run_system()
                else:
                    self._run_mic()

            except Exception as e:
                if self.stop.is_set():
                    break

                self.ui_q.put(
                    (
                        "error",
                        f"Audio capture error: {e}",
                    )
                )

                if self.fatal:
                    self.stop.set()
                    return

                # Avoid an aggressive restart loop.
                self.ui_q.put(
                    (
                        "status",
                        "Retrying audio capture in 2 seconds…",
                    )
                )

                for _ in range(20):
                    if self.stop.is_set():
                        return

                    if self._restart.is_set():
                        break

                    time.sleep(0.1)


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #

class Transcriber(threading.Thread):
    def __init__(
        self,
        cfg,
        audio_q,
        ui_q,
        stop,
    ):
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

    def _load(self):
        c = self.cfg

        self.ui_q.put(
            (
                "status",
                f"Loading model '{c.model}'…",
            )
        )

        try:
            model = self._WhisperModel(
                c.model,
                device=c.device,
                compute_type=c.compute_type,
            )

        except Exception as e:
            if self.model is None:
                self.ui_q.put(
                    (
                        "error",
                        f"Model load failed: {e}",
                    )
                )

                self.stop.set()
                return False

            old_key = self.loaded_key

            if old_key is not None:
                (
                    c.model,
                    c.device,
                    c.compute_type,
                ) = old_key

            self.ui_q.put(
                (
                    "error",
                    f"Could not load that model ({e}). "
                    f"Reverted.",
                )
            )

            return True

        self.model = model
        self.loaded_key = self._key()

        # Clear old audio after model loading.
        while True:
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

        self.ui_q.put(
            (
                "status",
                "Listening…",
            )
        )

        return True

    def _transcribe(
        self,
        audio,
        final,
    ):
        c = self.cfg

        try:
            segments, _ = self.model.transcribe(
                audio,
                language=c.language,
                task=(
                    "translate"
                    if c.translate
                    else "transcribe"
                ),
                beam_size=3 if final else 1,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300
                ),
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
                        "GPU transcription failed "
                        f"({e}). Falling back to CPU.",
                    )
                )

            else:
                self.ui_q.put(
                    (
                        "error",
                        f"Transcription failed: {e}",
                    )
                )

                self.stop.set()

            return ""

        normalized = text.lower().strip(
            " .!?,"
        )

        if normalized in HALLUCINATIONS:
            return ""

        return text

    def _emit_final(
        self,
        buf,
        voiced_ms,
    ):
        if (
            voiced_ms < 250
            or not buf
            or self.model is None
        ):
            return

        audio = np.concatenate(buf)

        text = self._transcribe(
            audio,
            final=True,
        )

        if not text:
            return

        self.ui_q.put(
            (
                "final",
                text,
            )
        )

        if self.cfg.log:
            try:
                with open(
                    self.cfg.log,
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"{text}\n"
                    )

            except OSError as e:
                self.ui_q.put(
                    (
                        "error",
                        f"Cannot write log: {e}",
                    )
                )

    def run(self):
        try:
            from faster_whisper import (
                WhisperModel,
            )

        except ImportError:
            self.ui_q.put(
                (
                    "error",
                    "Missing dependency: "
                    "pip install faster-whisper",
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

            if (
                not speaking
                and (
                    self.model is None
                    or self._key()
                    != self.loaded_key
                )
            ):
                if not self._load():
                    return

                preroll.clear()

            try:
                chunk = self.audio_q.get(
                    timeout=0.2
                )

            except queue.Empty:
                continue

            if chunk is None:
                continue

            chunk = np.asarray(
                chunk,
                dtype=np.float32,
            )

            if len(chunk) == 0:
                continue

            rms = float(
                np.sqrt(
                    np.mean(
                        chunk ** 2
                    )
                )
            )

            voiced = (
                rms
                > max(
                    a.threshold,
                    noise * 3,
                )
            )

            if not speaking:

                if voiced:
                    speaking = True

                    buf = (
                        list(preroll)
                        + [chunk]
                    )

                    silence_ms = 0
                    voiced_ms = CHUNK_MS

                    last_partial = (
                        time.monotonic()
                    )

                else:
                    preroll.append(chunk)

                    noise = (
                        0.95 * noise
                        + 0.05 * rms
                    )

                continue

            buf.append(chunk)

            if voiced:
                silence_ms = 0
                voiced_ms += CHUNK_MS
            else:
                silence_ms += CHUNK_MS

            dur = (
                len(buf)
                * CHUNK_MS
                / 1000
            )

            if (
                silence_ms >= a.silence_ms
                or dur >= a.max_utterance
            ):
                self._emit_final(
                    buf,
                    voiced_ms,
                )

                buf = []
                speaking = False
                preroll.clear()

            elif (
                time.monotonic()
                - last_partial
                >= a.partial_interval
                and dur >= 0.6
                and self.audio_q.qsize() < 5
                and self.model is not None
            ):
                text = self._transcribe(
                    np.concatenate(buf),
                    final=False,
                )

                last_partial = (
                    time.monotonic()
                )

                if text:
                    self.ui_q.put(
                        (
                            "partial",
                            text,
                        )
                    )


# --------------------------------------------------------------------------- #
# Caption state
# --------------------------------------------------------------------------- #

class CaptionState:
    def __init__(self):
        self.committed = ""
        self.partial = ""
        self.status = ""
        self.last_update = 0.0

    def handle(
        self,
        kind,
        text,
    ):
        self.last_update = (
            time.monotonic()
        )

        if kind in (
            "final",
            "preview",
        ):
            self.committed = (
                self.committed
                + " "
                + text
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
        full = (
            self.committed
            + " "
            + self.partial
        ).strip()

        if not full:
            return self.status

        if (
            max_chars
            and len(full) > max_chars
        ):
            full = full[-max_chars:]

            i = full.find(" ")

            if i != -1:
                full = "…" + full[i:]
            else:
                full = "…" + full

        return full


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #

def _contrast(hexcolor):
    try:
        r, g, b = (
            int(
                hexcolor[i:i + 2],
                16,
            )
            for i in (
                1,
                3,
                5,
            )
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

    except (
        ValueError,
        IndexError,
    ):
        return "black"


# --------------------------------------------------------------------------- #
# Overlay
# --------------------------------------------------------------------------- #

class Overlay:
    def __init__(
        self,
        cfg,
        ui_q,
        stop,
        reader,
    ):
        import tkinter as tk
        import tkinter.font as tkfont
        import tkinter.ttk as ttk
        from tkinter import (
            colorchooser,
            filedialog,
            messagebox,
        )

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

        self.root.title(
            "livecaption"
        )

        self.root.overrideredirect(
            True
        )

        self.root.attributes(
            "-topmost",
            True,
        )

        self.font = tkfont.Font(
            family=cfg.font_family,
            size=-int(
                cfg.font_size
            ),
            weight=(
                "bold"
                if cfg.bold
                else "normal"
            ),
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

        self.sw = (
            self.root.winfo_screenwidth()
        )

        self.sh = (
            self.root.winfo_screenheight()
        )

        self.apply_display(
            initial=True
        )

        self._drag = (0, 0)

        for w in (
            self.root,
            self.label,
        ):
            w.bind(
                "<Button-1>",
                self._start_drag,
            )

            w.bind(
                "<B1-Motion>",
                self._do_drag,
            )

            w.bind(
                "<ButtonRelease-1>",
                self._end_drag,
            )

            w.bind(
                "<Double-Button-1>",
                lambda e: self.open_settings(),
            )

            w.bind(
                "<Button-3>",
                self._menu,
            )

        self.pause_var = (
            tk.BooleanVar(
                value=False
            )
        )

        pos = tk.Menu(
            self.root,
            tearoff=0,
        )

        pos.add_command(
            label="Top of screen",
            command=lambda: self._place(
                "top"
            ),
        )

        pos.add_command(
            label="Middle of screen",
            command=lambda: self._place(
                "center"
            ),
        )

        pos.add_command(
            label="Bottom of screen",
            command=lambda: self._place(
                "bottom"
            ),
        )

        self.menu = tk.Menu(
            self.root,
            tearoff=0,
        )

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

        self.root.after(
            50,
            self._poll,
        )

    # -- layout ---------------------------------------------------------- #

    def apply_display(
        self,
        initial=False,
    ):
        c = self.cfg

        self.font.configure(
            family=c.font_family,
            size=-int(
                c.font_size
            ),
            weight=(
                "bold"
                if c.bold
                else "normal"
            ),
        )

        self.label.configure(
            fg=c.fg,
            bg=c.bg,
        )

        self.root.configure(
            bg=c.bg
        )

        self.width = max(
            200,
            int(
                self.sw
                * c.width_pct
                / 100
            ),
        )

        self._line_width = max(
            100,
            self.width - 40,
        )

        height = (
            self.font.metrics(
                "linespace"
            )
            * int(c.lines)
            + 30
        )

        self.label.configure(
            wraplength=self._line_width
        )

        if initial:
            if c.pos:
                x, y = c.pos

            else:
                x = (
                    self.sw
                    - self.width
                ) // 2

                y = (
                    self.sh
                    - height
                    - int(
                        self.sh
                        * 0.06
                    )
                )

        else:
            x = self.root.winfo_x()
            y = self.root.winfo_y()

        x = max(
            0,
            min(
                int(x),
                self.sw - 100,
            ),
        )

        y = max(
            0,
            min(
                int(y),
                self.sh - 40,
            ),
        )

        self.root.geometry(
            f"{self.width}x{height}"
            f"+{x}+{y}"
        )

    def _place(self, where):
        height = (
            self.root.winfo_height()
        )

        x = (
            self.sw
            - self.width
        ) // 2

        y = {
            "top": int(
                self.sh * 0.05
            ),
            "center": (
                self.sh - height
            ) // 2,
            "bottom": (
                self.sh
                - height
                - int(
                    self.sh
                    * 0.06
                )
            ),
        }[where]

        self.cfg.pos = [x, y]

        self.root.geometry(
            f"{self.width}x{height}"
            f"+{x}+{y}"
        )

    def _max_chars(self):
        if self.cfg.max_chars:
            return int(
                self.cfg.max_chars
            )

        per_line = max(
            10,
            int(
                self._line_width
                / max(
                    1,
                    self.font.measure(
                        "n"
                    ),
                )
            ),
        )

        return int(
            per_line
            * int(self.cfg.lines)
            * 0.95
        )

    def _fit_caption_to_lines(
        self,
        text,
    ):
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
                else current
                + " "
                + word
            )

            if (
                self.font.measure(
                    candidate
                )
                <= width
            ):
                current = candidate
                continue

            if current:
                lines.append(
                    current
                )

            current = word

            if (
                len(lines)
                >= max_lines
            ):
                break

            if (
                self.font.measure(
                    current
                )
                > width
            ):
                while (
                    len(current) > 1
                    and self.font.measure(
                        current + "…"
                    ) > width
                ):
                    current = current[:-1]

        if (
            current
            and len(lines)
            < max_lines
        ):
            lines.append(
                current
            )

        displayed_words = (
            " ".join(lines)
        )

        if len(displayed_words) < len(text):
            if lines:
                last = lines[-1]

                while (
                    last
                    and self.font.measure(
                        last + "…"
                    ) > width
                ):
                    last = last[:-1]

                lines[-1] = (
                    last.rstrip()
                    + "…"
                    if last
                    else "…"
                )

        return "\n".join(
            lines[:max_lines]
        )

    # -- dragging -------------------------------------------------------- #

    def _start_drag(self, e):
        self._drag = (
            e.x_root
            - self.root.winfo_x(),
            e.y_root
            - self.root.winfo_y(),
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

        except Exception:
            pass

        finally:
            try:
                self.menu.grab_release()
            except Exception:
                pass

    def toggle_pause(self):
        self.cfg.paused = (
            self.pause_var.get()
        )

        self.state.clear()

        self.state.status = (
            "⏸ Paused"
            if self.cfg.paused
            else "Listening…"
        )

    # -- transcript ------------------------------------------------------ #

    def open_transcript(self):
        tk = self.tk

        if (
            self.transcript_win is not None
            and self.transcript_win.winfo_exists()
        ):
            self.transcript_win.lift()
            return

        win = tk.Toplevel(
            self.root
        )

        win.title(
            "Live Caption – Transcript"
        )

        win.geometry(
            "580x380"
        )

        win.attributes(
            "-topmost",
            True,
        )

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
            yscrollcommand=sb.set
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
            path = (
                self.filedialog
                .asksaveasfilename(
                    parent=win,
                    defaultextension=".txt",
                    initialfile="captions.txt",
                    filetypes=[
                        ("Text", "*.txt"),
                        ("All files", "*"),
                    ],
                )
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

            text.configure(
                state="normal"
            )

            text.delete(
                "1.0",
                "end",
            )

            text.configure(
                state="disabled"
            )

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
            self._append_transcript(
                ts,
                t,
            )

    def _transcript_text(self):
        return "".join(
            f"[{ts}] {t}\n"
            for ts, t in self.transcript
        )

    def _append_transcript(
        self,
        ts,
        t,
    ):
        w = self.transcript_text

        if (
            self.transcript_win is None
            or not self.transcript_win.winfo_exists()
        ):
            return

        w.configure(
            state="normal"
        )

        w.insert(
            "end",
            f"[{ts}] {t}\n",
        )

        w.see("end")

        w.configure(
            state="disabled"
        )

    # -- settings -------------------------------------------------------- #

    def open_settings(self):
        tk = self.tk
        ttk = self.ttk
        c = self.cfg

        if (
            self.settings_win is not None
            and self.settings_win.winfo_exists()
        ):
            self.settings_win.lift()
            return

        win = tk.Toplevel(
            self.root
        )

        win.title(
            "Live Caption – Settings"
        )

        win.attributes(
            "-topmost",
            True,
        )

        win.resizable(
            False,
            False,
        )

        self.settings_win = win

        v = dict(
            source=tk.StringVar(
                value=c.source
            ),
            threshold=tk.DoubleVar(
                value=c.threshold
            ),
            silence_ms=tk.IntVar(
                value=c.silence_ms
            ),
            max_utterance=tk.DoubleVar(
                value=c.max_utterance
            ),
            model=tk.StringVar(
                value=c.model
            ),
            language=tk.StringVar(
                value=c.language or "auto"
            ),
            translate=tk.BooleanVar(
                value=c.translate
            ),
            device=tk.StringVar(
                value=c.device
            ),
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
            bold=tk.BooleanVar(
                value=c.bold
            ),
            lines=tk.IntVar(
                value=c.lines
            ),
            width_pct=tk.IntVar(
                value=c.width_pct
            ),
            opacity=tk.DoubleVar(
                value=c.opacity
            ),
            idle_opacity=tk.DoubleVar(
                value=c.idle_opacity
            ),
            fg=tk.StringVar(
                value=c.fg
            ),
            bg=tk.StringVar(
                value=c.bg
            ),
            clear_after=tk.DoubleVar(
                value=c.clear_after
            ),
            log=tk.StringVar(
                value=c.log or ""
            ),
        )

        nb = ttk.Notebook(win)

        nb.pack(
            fill="both",
            expand=True,
            padx=8,
            pady=8,
        )

        # Audio tab
        af = ttk.Frame(nb)

        nb.add(
            af,
            text="Audio",
        )

        source_choices = [
            "system",
            "mic",
        ]

        try:
            for d in get_sounddevice_devices():
                if (
                    d["inputs"] > 0
                    or d["outputs"] > 0
                ):
                    source_choices.append(
                        d["name"]
                    )
        except Exception:
            pass

        for row, (
            lbl,
            widget_factory,
        ) in enumerate(
            [
                (
                    "Source",
                    lambda f: ttk.Combobox(
                        f,
                        textvariable=v["source"],
                        values=source_choices,
                        width=30,
                    ),
                ),
                (
                    "Threshold",
                    lambda f: ttk.Entry(
                        f,
                        textvariable=v["threshold"],
                        width=10,
                    ),
                ),
                (
                    "Silence (ms)",
                    lambda f: ttk.Entry(
                        f,
                        textvariable=v["silence_ms"],
                        width=10,
                    ),
                ),
                (
                    "Max utterance (s)",
                    lambda f: ttk.Entry(
                        f,
                        textvariable=v["max_utterance"],
                        width=10,
                    ),
                ),
            ]
        ):
            ttk.Label(
                af,
                text=lbl,
            ).grid(
                row=row,
                column=0,
                sticky="w",
                padx=8,
                pady=4,
            )

            widget_factory(
                af
            ).grid(
                row=row,
                column=1,
                sticky="w",
                padx=8,
                pady=4,
            )

        # Model tab
        mf = ttk.Frame(nb)

        nb.add(
            mf,
            text="Model",
        )

        model_choices = [
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large-v2",
            "large-v3",
            "distil-large-v3",
            "distil-medium.en",
            "distil-small.en",
        ]

        for row, (
            lbl,
            widget_factory,
        ) in enumerate(
            [
                (
                    "Model",
                    lambda f: ttk.Combobox(
                        f,
                        textvariable=v["model"],
                        values=model_choices,
                        width=20,
                    ),
                ),
                (
                    "Language",
                    lambda f: ttk.Entry(
                        f,
                        textvariable=v["language"],
                        width=10,
                    ),
                ),
                (
                    "Translate to EN",
                    lambda f: ttk.Checkbutton(
                        f,
                        variable=v["translate"],
                    ),
                ),
                (
                    "Device",
                    lambda f: ttk.Combobox(
                        f,
                        textvariable=v["device"],
                        values=[
                            "cpu",
                            "cuda",
                        ],
                        width=10,
                    ),
                ),
                (
                    "Compute type",
                    lambda f: ttk.Combobox(
                        f,
                        textvariable=v["compute_type"],
                        values=[
                            "auto",
                            "int8",
                            "float16",
                            "float32",
                            "int8_float16",
                        ],
                        width=14,
                    ),
                ),
                (
                    "Partial interval (s)",
                    lambda f: ttk.Entry(
                        f,
                        textvariable=v[
                            "partial_interval"
                        ],
                        width=10,
                    ),
                ),
            ]
        ):
            ttk.Label(
                mf,
                text=lbl,
            ).grid(
                row=row,
                column=0,
                sticky="w",
                padx=8,
                pady=4,
            )

            widget_factory(
                mf
            ).grid(
                row=row,
                column=1,
                sticky="w",
                padx=8,
                pady=4,
            )

        # Display tab
        df = ttk.Frame(nb)

        nb.add(
            df,
            text="Display",
        )

        for row, (
            lbl,
            widget_factory,
        ) in enumerate(
            [
                (
                    "Font family",
                    lambda f: ttk.Entry(
                        f,
                        textvariable=v["font_family"],
                        width=20,
                    ),
                ),
                (
                    "Font size (px)",
                    lambda f: ttk.Entry(
                        f,
                        textvariable=v["font_size"],
                        width=8,
                    ),
                ),
                (
                    "Bold",
                    lambda f: ttk.Checkbutton(
                        f,
                        variable=v["bold"],
                    ),
                ),
                (
                    "Lines",
                    lambda f: ttk.Spinbox(
                        f,
                        textvariable=v["lines"],
                        from_=1,
                        to=6,
                        width=5,
                    ),
                ),
                (
                    "Width %",
                    lambda f: ttk.Spinbox(
                        f,
                        textvariable=v["width_pct"],
                        from_=20,
                        to=100,
                        width=5,
                    ),
                ),
                (
                    "Opacity",
                    lambda f: ttk.Entry(
                        f,
                        textvariable=v["opacity"],
                        width=8,
                    ),
                ),
                (
                    "Idle opacity",
                    lambda f: ttk.Entry(
                        f,
                        textvariable=v["idle_opacity"],
                        width=8,
                    ),
                ),
                (
                    "Clear after (s)",
                    lambda f: ttk.Entry(
                        f,
                        textvariable=v["clear_after"],
                        width=8,
                    ),
                ),
            ]
        ):
            ttk.Label(
                df,
                text=lbl,
            ).grid(
                row=row,
                column=0,
                sticky="w",
                padx=8,
                pady=4,
            )

            widget_factory(
                df
            ).grid(
                row=row,
                column=1,
                sticky="w",
                padx=8,
                pady=4,
            )

        def pick_color(var, btn):
            color = (
                self.colorchooser
                .askcolor(
                    color=var.get(),
                    parent=win,
                )[1]
            )

            if color:
                var.set(color)

                btn.configure(
                    bg=color,
                    fg=_contrast(color),
                )

        for row_offset, (
            lbl,
            var_key,
        ) in enumerate(
            [
                ("Text colour", "fg"),
                ("Background", "bg"),
            ],
            start=8,
        ):
            btn = tk.Button(
                df,
                text="  ",
                bg=v[var_key].get(),
                fg=_contrast(
                    v[var_key].get()
                ),
            )

            ttk.Label(
                df,
                text=lbl,
            ).grid(
                row=row_offset,
                column=0,
                sticky="w",
                padx=8,
                pady=4,
            )

            btn.grid(
                row=row_offset,
                column=1,
                sticky="w",
                padx=8,
                pady=4,
            )

            btn.configure(
                command=lambda vv=v[var_key],
                b=btn: pick_color(
                    vv,
                    b,
                )
            )

        # Output tab
        of = ttk.Frame(nb)

        nb.add(
            of,
            text="Output",
        )

        ttk.Label(
            of,
            text="Log file",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=8,
            pady=4,
        )

        ttk.Entry(
            of,
            textvariable=v["log"],
            width=30,
        ).grid(
            row=0,
            column=1,
            padx=8,
            pady=4,
        )

        def browse_log():
            path = (
                self.filedialog
                .asksaveasfilename(
                    parent=win,
                    defaultextension=".txt",
                    filetypes=[
                        ("Text", "*.txt"),
                        ("All files", "*"),
                    ],
                )
            )

            if path:
                v["log"].set(path)

        ttk.Button(
            of,
            text="Browse…",
            command=browse_log,
        ).grid(
            row=0,
            column=2,
            padx=4,
        )

        # Apply
        def apply(save=False):
            c.source = (
                v["source"].get()
            )

            c.threshold = (
                v["threshold"].get()
            )

            c.silence_ms = (
                v["silence_ms"].get()
            )

            c.max_utterance = (
                v["max_utterance"].get()
            )

            c.model = (
                v["model"].get()
            )

            lang = (
                v["language"]
                .get()
                .strip()
            )

            c.language = (
                None
                if lang in (
                    "auto",
                    "",
                )
                else lang
            )

            c.translate = (
                v["translate"].get()
            )

            c.device = (
                v["device"].get()
            )

            c.compute_type = (
                v["compute_type"].get()
            )

            c.partial_interval = (
                v["partial_interval"].get()
            )

            c.font_family = (
                v["font_family"].get()
            )

            c.font_size = (
                v["font_size"].get()
            )

            c.bold = (
                v["bold"].get()
            )

            c.lines = (
                v["lines"].get()
            )

            c.width_pct = (
                v["width_pct"].get()
            )

            c.opacity = (
                v["opacity"].get()
            )

            c.idle_opacity = (
                v["idle_opacity"].get()
            )

            c.fg = (
                v["fg"].get()
            )

            c.bg = (
                v["bg"].get()
            )

            c.clear_after = (
                v["clear_after"].get()
            )

            c.log = (
                v["log"].get()
                or None
            )

            self.apply_display()

            self.reader.restart_now()

            if save:
                save_config(c)

                self.messagebox.showinfo(
                    "Saved",
                    f"Settings saved to:\n"
                    f"{CONFIG_PATH}",
                    parent=win,
                )

        def test():
            apply()

            self.ui_q.put(
                (
                    "preview",
                    "This is a sample caption "
                    "to preview your settings.",
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
        ).pack(
            side="left"
        )

        ttk.Button(
            btns,
            text="Close",
            command=win.destroy,
        ).pack(
            side="right"
        )

        ttk.Button(
            btns,
            text="Save",
            command=lambda: apply(
                save=True
            ),
        ).pack(
            side="right",
            padx=4,
        )

        ttk.Button(
            btns,
            text="Apply",
            command=apply,
        ).pack(
            side="right"
        )

    # -- main loop ------------------------------------------------------- #

    def _poll(self):
        try:
            while True:
                kind, text = (
                    self.ui_q.get_nowait()
                )

                self.state.handle(
                    kind,
                    text,
                )

                if kind == "final":
                    ts = time.strftime(
                        "%H:%M:%S"
                    )

                    self.transcript.append(
                        (
                            ts,
                            text,
                        )
                    )

                    self._append_transcript(
                        ts,
                        text,
                    )

                    safe_print(
                        text,
                        flush=True,
                    )

                elif kind == "error":
                    safe_print(
                        f"ERROR: {text}",
                        file=sys.stderr,
                        flush=True,
                    )

                elif kind == "status":
                    safe_print(
                        text,
                        file=sys.stderr,
                        flush=True,
                    )

        except queue.Empty:
            pass

        if self.stop.is_set():
            try:
                self.root.destroy()
            except Exception:
                pass

            return

        s = self.state
        c = self.cfg

        if (
            s.committed
            or s.partial
        ) and (
            time.monotonic()
            - s.last_update
            > float(c.clear_after)
        ):
            s.clear()

            s.status = (
                "⏸ Paused"
                if c.paused
                else ""
            )

        rendered = s.render(
            self._max_chars()
        )

        if (
            s.committed
            or s.partial
        ):
            rendered = (
                self._fit_caption_to_lines(
                    rendered
                )
            )

        self.label.configure(
            text=rendered
        )

        active = bool(
            s.committed
            or s.partial
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


# --------------------------------------------------------------------------- #
# Terminal mode
# --------------------------------------------------------------------------- #

def run_terminal(
    ui_q,
    stop,
):
    cols = max(
        20,
        shutil.get_terminal_size().columns - 1,
    )

    partial_shown = False

    while not stop.is_set():
        try:
            kind, text = (
                ui_q.get(
                    timeout=0.2
                )
            )

        except queue.Empty:
            continue

        if kind == "partial":
            safe_print(
                "\r"
                + text[-cols:].ljust(cols),
                end="",
                flush=True,
            )

            partial_shown = True

        else:
            if partial_shown:
                safe_print(
                    "\r"
                    + " " * cols
                    + "\r",
                    end="",
                    flush=True,
                )

                partial_shown = False

            safe_print(
                text
                if kind == "final"
                else f"[{kind}] {text}",
                flush=True,
            )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    cfg, extra = build_config()

    if extra.list_sources:
        list_sources()
        return

    if extra.diagnose_audio:
        diagnose_audio()
        return

    if extra.reset_config:
        try:
            os.remove(CONFIG_PATH)
            safe_print(
                f"Removed {CONFIG_PATH}"
            )

        except OSError:
            safe_print(
                "No saved settings to remove."
            )

        return

    # Make sure dependencies are present.
    try:
        _import_sounddevice()

        if cfg.source == "system":
            _import_pyaudiowpatch()

    except Exception as e:
        safe_print(
            f"ERROR: {e}",
            file=sys.stderr,
        )

        return

    stop = threading.Event()

    # Ctrl+C
    try:
        signal.signal(
            signal.SIGINT,
            lambda *_: stop.set(),
        )

    except Exception:
        pass

    if hasattr(
        signal,
        "SIGTERM",
    ):
        try:
            signal.signal(
                signal.SIGTERM,
                lambda *_: stop.set(),
            )
        except Exception:
            pass

    audio_q = queue.Queue()
    ui_q = queue.Queue()

    reader = AudioReader(
        cfg,
        audio_q,
        ui_q,
        stop,
        fatal=extra.no_gui,
    )

    reader.start()

    transcriber = Transcriber(
        cfg,
        audio_q,
        ui_q,
        stop,
    )

    transcriber.start()

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
            safe_print(
                f"GUI unavailable ({e}); "
                f"falling back to terminal.",
                file=sys.stderr,
            )

            run_terminal(
                ui_q,
                stop,
            )

    stop.set()

    try:
        reader.restart_now()
    except Exception:
        pass

    safe_flush()


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        pass

    except Exception as e:
        safe_print(
            f"Fatal error: {e}",
            file=sys.stderr,
        )

        safe_flush()
