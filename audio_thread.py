"""Audio analysis thread — owns audio capture, VAD, speaker recognition,
and YAMNet classification. Emits per-window state via Qt signals.

State machine for music mode (Phase 2 spec from the user):

    DEFAULT (speech mode) ─┬→ candidate music: music_dominant=True AND no recognized speaker
                           │  │
                           │  └─ 5 s sustained → enter MUSIC mode
                           │
    MUSIC mode ────────────┴→ candidate exit: speech_dominant OR recognized speaker
                              │
                              ├─ recognized speaker present → exit immediately
                              └─ 10 s of speech/silence    → exit
"""

from __future__ import annotations

import time
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal

from audio_capture import AudioCapture, SAMPLE_RATE
from profiles import ProfileStore


# Analysis cadence — 2 Hz strikes a balance between responsiveness and CPU.
_TICK_INTERVAL_MS = 500

# Audio window lengths fed to the two analyses.
_SPEAKER_WINDOW_S = 1.5     # ECAPA needs >= 0.5s; 1.5s gives a solid embedding.
_CLASSIFY_WINDOW_S = 1.5    # YAMNet emits ~16 frames at this length.

# Music-mode hysteresis (seconds of sustained candidate state to flip).
_MUSIC_ENTER_HOLD = 5.0
_MUSIC_EXIT_HOLD = 10.0

# How long after a speaker recognition match before we consider their "voice
# boost" decayed — also acts as the speech-veto window for music mode.
SPEAKER_BOOST_HOLD_S = 3.0

# VAD aggressiveness: 0=loose, 3=strict. 2 is a good default for room mics.
_VAD_AGGRESSIVENESS = 2
_VAD_FRAME_MS = 30


class AudioThread(QThread):
    """Drives capture + analysis, emits state for the rest of the app to consume.

    Signals:
        audio_state_changed: emitted whenever the music_mode flag flips.
            (music_mode: bool, music_score: float, speech_score: float)
        speaker_detected:    emitted whenever the speaker recognizer matches an
                              enrolled profile.
            (profile_id: str, name: str, score: float)
        voice_quiet:         emitted when VAD has been negative for >= 1 s.
        levels:              emitted every tick with the current input level.
            (rms: float)
    """

    audio_state_changed = pyqtSignal(bool, float, float)
    speaker_detected = pyqtSignal(str, str, float)
    voice_quiet = pyqtSignal()
    levels = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self, profile_store: ProfileStore, parent=None):
        super().__init__(parent)
        self._store = profile_store
        self._running = False

        self._capture = AudioCapture()
        self._speaker_rec = None        # lazy
        self._classifier = None          # lazy
        self._vad = None                 # lazy webrtcvad.Vad

        self._device_index: int | None = None
        self._enabled = False

        # Music-mode state
        self._music_mode = False
        self._music_candidate_since: float | None = None
        self._speech_candidate_since: float | None = None
        self._last_recognized_at: float = 0.0

        # Music classification controls (set true if YAMNet failed to load)
        self._classifier_failed = False
        # Speaker recognition controls (set true if SpeechBrain failed to load)
        self._speaker_rec_failed = False

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    @property
    def music_mode(self) -> bool:
        return self._music_mode

    @property
    def device_index(self) -> int | None:
        return self._device_index

    @property
    def is_capturing(self) -> bool:
        return self._capture.is_running

    def set_device(self, device_index: int | None):
        """Switch capture to ``device_index`` (None = system default).

        Call this from the UI thread. The change is applied on the next tick
        of the analysis loop.
        """
        self._device_index = device_index
        if self._enabled:
            self._capture.start(device_index)

    def set_enabled(self, enabled: bool):
        """Enable or disable audio analysis entirely.

        When disabled, capture is stopped and music_mode is forced off.
        """
        self._enabled = enabled
        if enabled:
            self._capture.start(self._device_index)
        else:
            self._capture.stop()
            if self._music_mode:
                self._music_mode = False
                self.audio_state_changed.emit(False, 0.0, 0.0)

    def reindex_voice_profiles(self):
        if self._speaker_rec is not None:
            self._speaker_rec.mark_index_dirty()

    # ------------------------------------------------------------------
    # Thread loop
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        while self._running:
            try:
                if self._enabled:
                    self._tick()
            except Exception as e:
                self.error.emit(str(e))
            self.msleep(_TICK_INTERVAL_MS)

    def stop(self):
        self._running = False
        self._capture.stop()
        self.wait()

    # ------------------------------------------------------------------
    # Per-tick analysis
    # ------------------------------------------------------------------

    def _tick(self):
        if not self._capture.is_running:
            return

        self.levels.emit(self._capture.last_rms)

        # Pull a window of the most-recent audio for analysis.
        window = self._capture.get_recent(_CLASSIFY_WINDOW_S)
        if window.size == 0 or np.all(np.abs(window) < 1e-5):
            self._update_music_mode(
                music_score=0.0, speech_score=0.0,
                music_dominant=False, speech_dominant=False,
                recognized_speaker=False,
            )
            return

        # ── 1. VAD on the latest 30ms-aligned chunk ───────────────────
        speech_active = self._vad_active(window)

        # ── 2. Speaker recognition (only when voice activity detected) ─
        recognized_speaker = False
        if speech_active and not self._speaker_rec_failed:
            recognized_speaker = self._run_speaker_recognition(window)

        # ── 3. Music vs speech classification (YAMNet) ────────────────
        if self._classifier_failed:
            cls = {
                "music_score": 0.0, "speech_score": 0.0,
                "music_dominant": False, "speech_dominant": False, "silent": False,
            }
        else:
            try:
                if self._classifier is None:
                    from audio_classifier import AudioClassifier
                    self._classifier = AudioClassifier()
                cls = self._classifier.classify(window)
            except Exception as e:
                print(f"AudioClassifier failed: {e}")
                self._classifier_failed = True
                cls = {
                    "music_score": 0.0, "speech_score": 0.0,
                    "music_dominant": False, "speech_dominant": False, "silent": False,
                }

        self._update_music_mode(
            music_score=cls["music_score"],
            speech_score=cls["speech_score"],
            music_dominant=cls["music_dominant"],
            speech_dominant=cls["speech_dominant"],
            recognized_speaker=recognized_speaker,
        )

        if not speech_active:
            self.voice_quiet.emit()

    # ------------------------------------------------------------------
    # VAD
    # ------------------------------------------------------------------

    def _vad_active(self, window: np.ndarray) -> bool:
        """Return True if WebRTC VAD flags any 30 ms frame as speech."""
        if self._vad is None:
            try:
                import webrtcvad
                self._vad = webrtcvad.Vad(_VAD_AGGRESSIVENESS)
            except Exception:
                self._vad = False    # signal: VAD unavailable
        if self._vad is False:
            # Without VAD, fall back to a simple energy threshold.
            return float(np.sqrt(np.mean(window ** 2))) > 0.01

        # Convert float32 [-1, 1] → int16 PCM that webrtcvad expects.
        pcm = np.clip(window * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        frame_bytes = int(SAMPLE_RATE * _VAD_FRAME_MS / 1000) * 2   # 2 bytes/sample
        n_active = 0
        for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
            try:
                if self._vad.is_speech(pcm[i:i + frame_bytes], SAMPLE_RATE):
                    n_active += 1
                    if n_active >= 2:    # require >=2 frames to suppress single-frame noise
                        return True
            except Exception:
                return False
        return False

    # ------------------------------------------------------------------
    # Speaker recognition
    # ------------------------------------------------------------------

    def _run_speaker_recognition(self, window: np.ndarray) -> bool:
        # No enrolled voice samples → no point loading SpeechBrain.
        if not any(p.voice_embeddings is not None and len(p.voice_embeddings) > 0
                   for p in self._store.list()):
            return False

        if self._speaker_rec is None:
            try:
                from speaker_recognizer import SpeakerRecognizer
                self._speaker_rec = SpeakerRecognizer(self._store)
            except Exception as e:
                print(f"SpeakerRecognizer init failed: {e}")
                self._speaker_rec_failed = True
                return False

        # Use the freshest portion of the window for matching.
        clip = window[-int(_SPEAKER_WINDOW_S * SAMPLE_RATE):]
        try:
            match = self._speaker_rec.identify(clip, sample_rate=SAMPLE_RATE)
        except Exception as e:
            print(f"Speaker identify failed: {e}")
            return False
        if match is None:
            return False
        self._last_recognized_at = time.monotonic()
        self.speaker_detected.emit(match["profile_id"], match["name"], match["score"])
        return True

    # ------------------------------------------------------------------
    # Music-mode state machine
    # ------------------------------------------------------------------

    def _update_music_mode(self, *, music_score: float, speech_score: float,
                           music_dominant: bool, speech_dominant: bool,
                           recognized_speaker: bool):
        now = time.monotonic()
        # The speech veto: if any enrolled speaker has been recognized in the
        # last SPEAKER_BOOST_HOLD_S seconds, treat the audio as speech-dominant
        # regardless of classifier output. Background music behind a sermon
        # therefore never trips music mode.
        speech_veto_active = (now - self._last_recognized_at) < SPEAKER_BOOST_HOLD_S
        effective_speech = speech_dominant or speech_veto_active
        effective_music = music_dominant and not speech_veto_active

        if not self._music_mode:
            # Currently in speech mode. Watch for sustained music with no veto.
            if effective_music and not effective_speech:
                if self._music_candidate_since is None:
                    self._music_candidate_since = now
                elif now - self._music_candidate_since >= _MUSIC_ENTER_HOLD:
                    self._music_mode = True
                    self._music_candidate_since = None
                    self._speech_candidate_since = None
                    self.audio_state_changed.emit(True, music_score, speech_score)
            else:
                self._music_candidate_since = None
        else:
            # Currently in music mode. Speech veto exits immediately; otherwise
            # require sustained speech/silence for _MUSIC_EXIT_HOLD seconds.
            if speech_veto_active:
                self._music_mode = False
                self._speech_candidate_since = None
                self._music_candidate_since = None
                self.audio_state_changed.emit(False, music_score, speech_score)
            elif effective_speech or (not effective_music and not effective_speech):
                if self._speech_candidate_since is None:
                    self._speech_candidate_since = now
                elif now - self._speech_candidate_since >= _MUSIC_EXIT_HOLD:
                    self._music_mode = False
                    self._speech_candidate_since = None
                    self.audio_state_changed.emit(False, music_score, speech_score)
            else:
                self._speech_candidate_since = None
