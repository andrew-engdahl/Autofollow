"""Audio capture — owns a sounddevice input stream and exposes a recent-audio
ring buffer to consumers (VAD, speaker recognizer, audio classifier).

Captures at 16 kHz mono float32. 16 kHz is the native sample rate for both
SpeechBrain ECAPA-TDNN and YAMNet, so consumers can read directly without
resampling.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False


SAMPLE_RATE = 16000               # Hz — matches YAMNet + ECAPA
CHANNELS = 1
BUFFER_SECONDS = 3.0              # rolling window kept in memory
BLOCK_SIZE = 480                  # 30 ms at 16 kHz — webrtcvad-friendly


def list_input_devices() -> list[dict]:
    """Return [{'index': int, 'name': str, 'max_channels': int}] of input devices.

    Returns an empty list if sounddevice isn't available.
    """
    if not SOUNDDEVICE_AVAILABLE:
        return []
    devices = []
    try:
        for i, info in enumerate(sd.query_devices()):
            if info.get('max_input_channels', 0) > 0:
                devices.append({
                    'index': i,
                    'name': info.get('name', f'Device {i}'),
                    'max_channels': info.get('max_input_channels', 0),
                })
    except Exception as e:
        print(f"sounddevice query failed: {e}")
    return devices


def default_input_device_index() -> int | None:
    """Return the system default input device index, or None."""
    if not SOUNDDEVICE_AVAILABLE:
        return None
    try:
        default = sd.default.device
        if isinstance(default, (list, tuple)) and len(default) >= 1:
            idx = default[0]
            return int(idx) if idx is not None and idx >= 0 else None
        return int(default) if default is not None and default >= 0 else None
    except Exception:
        return None


class AudioCapture:
    """Background-thread audio capture with a fixed-size ring buffer.

    Usage::

        cap = AudioCapture()
        cap.start(device_index)
        waveform = cap.get_recent(seconds=1.0)   # latest 1.0s as float32 mono
        cap.stop()

    The ring buffer is always exactly BUFFER_SECONDS long. get_recent() returns
    a slice of it. If start() is never called or the stream is stopped, the
    buffer is empty and get_recent() returns a zero-filled array.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 buffer_seconds: float = BUFFER_SECONDS):
        self.sample_rate = sample_rate
        self._buffer_samples = int(sample_rate * buffer_seconds)
        self._lock = threading.Lock()
        # Pre-allocate the ring buffer to avoid per-callback allocations.
        self._ring = np.zeros(self._buffer_samples, dtype=np.float32)
        self._write_pos = 0
        self._filled = 0
        self._stream = None
        self._device_index: int | None = None
        self._on_error_cb = None
        self._last_rms = 0.0

    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def device_index(self) -> int | None:
        return self._device_index

    @property
    def last_rms(self) -> float:
        """Most recent block RMS — useful for an input-level UI indicator."""
        return self._last_rms

    # ------------------------------------------------------------------

    def start(self, device_index: int | None = None) -> bool:
        """Start capturing from the given device. Returns True on success.

        If a stream is already running, it is stopped first. Passing
        device_index=None uses the system default input.
        """
        if not SOUNDDEVICE_AVAILABLE:
            return False
        self.stop()
        try:
            self._stream = sd.InputStream(
                device=device_index,
                channels=CHANNELS,
                samplerate=self.sample_rate,
                blocksize=BLOCK_SIZE,
                dtype='float32',
                callback=self._callback,
            )
            self._stream.start()
            self._device_index = device_index
            return True
        except Exception as e:
            print(f"AudioCapture.start failed (device={device_index}): {e}")
            self._stream = None
            self._device_index = None
            if self._on_error_cb:
                self._on_error_cb(str(e))
            return False

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        with self._lock:
            self._ring.fill(0)
            self._write_pos = 0
            self._filled = 0
        self._last_rms = 0.0

    def set_error_callback(self, cb):
        self._on_error_cb = cb

    # ------------------------------------------------------------------

    def _callback(self, indata, frames, time_info, status):
        # indata: (frames, channels). We capture mono so collapse to 1-D.
        if status:
            # Buffer underruns / overflows happen on heavy CPU load — log once.
            # Don't spam: callback runs hundreds of times per second.
            pass
        if indata.ndim == 2:
            block = indata[:, 0]
        else:
            block = indata
        # Update last-RMS for UI indicator
        self._last_rms = float(np.sqrt(np.mean(block * block) + 1e-12))

        with self._lock:
            n = block.shape[0]
            buf = self._ring
            end = self._write_pos + n
            if end <= self._buffer_samples:
                buf[self._write_pos:end] = block
            else:
                first = self._buffer_samples - self._write_pos
                buf[self._write_pos:] = block[:first]
                buf[:n - first] = block[first:]
            self._write_pos = end % self._buffer_samples
            self._filled = min(self._buffer_samples, self._filled + n)

    # ------------------------------------------------------------------

    def get_recent(self, seconds: float) -> np.ndarray:
        """Return the most recent ``seconds`` of audio as a 1-D float32 array.

        Result is always exactly int(seconds * sample_rate) samples long;
        if the buffer hasn't filled yet, leading samples are zero.
        """
        n = int(seconds * self.sample_rate)
        n = min(n, self._buffer_samples)
        with self._lock:
            if self._filled == 0:
                return np.zeros(n, dtype=np.float32)
            # Reconstruct linear order: oldest to newest
            buf = self._ring
            start = (self._write_pos - n) % self._buffer_samples
            if start + n <= self._buffer_samples:
                out = buf[start:start + n].copy()
            else:
                first = self._buffer_samples - start
                out = np.concatenate([buf[start:], buf[:n - first]])
            return out

    def get_all(self) -> np.ndarray:
        """Return the full ring buffer (BUFFER_SECONDS) as float32 mono."""
        return self.get_recent(self._buffer_samples / self.sample_rate)
