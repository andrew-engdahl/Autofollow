"""YAMNet-based audio classifier — distinguishes music, speech, applause, etc.

YAMNet (Google) operates on 16 kHz mono audio. It emits per-frame scores
across 521 AudioSet classes. We aggregate the top music-like and speech-like
classes into compact ``music_score`` and ``speech_score`` floats in [0, 1].

The classifier is lazy: the TF Hub module is fetched and loaded on first use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

YAMNET_SAMPLE_RATE = 16000

# AudioSet class IDs we care about. Indices reference YAMNet's class_map.csv.
# These are stable across YAMNet versions:
#   0   Speech
#   3   Child speech, kid speaking
#   4   Conversation
#   5   Narration, monologue
#   132 Music
#   137 Musical instrument
#   16  Singing
#   46  Choir
#   47  Yodeling
#   42  Chant
_SPEECH_CLASS_IDS = (0, 3, 4, 5)
_MUSIC_CLASS_IDS = (132, 137, 16, 46, 47, 42)


class AudioClassifier:
    """Speech-vs-music score extractor backed by YAMNet."""

    def __init__(self):
        self._model = None    # lazy TF hub model

    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return
        # Quiet TF logs at import time.
        import os as _os
        _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow_hub as hub
        # Cache to a stable location so we don't re-download every run.
        cache_dir = Path.home() / ".autofollow" / "models" / "yamnet"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _os.environ.setdefault("TFHUB_CACHE_DIR", str(cache_dir))
        self._model = hub.load("https://tfhub.dev/google/yamnet/1")

    # ------------------------------------------------------------------

    def classify(self, waveform: np.ndarray) -> dict:
        """Return aggregated scores for the given mono 16 kHz waveform.

        ``waveform`` should be a 1-D float32 array of at least ~1 s.

        Returns::

            {
                'music_score':         float in [0, 1],
                'speech_score':        float in [0, 1],
                'music_dominant':      bool — music outweighs speech,
                'speech_dominant':     bool — speech outweighs music,
                'silent':              bool — RMS too low to classify,
            }
        """
        if waveform is None or waveform.size == 0:
            return self._zero_result(silent=True)

        # Silence short-circuit — avoid loading YAMNet just to score noise.
        rms = float(np.sqrt(np.mean(waveform.astype(np.float32) ** 2) + 1e-12))
        if rms < 5e-4:
            return self._zero_result(silent=True)

        self._ensure_model()
        import tensorflow as tf

        wav = waveform.astype(np.float32)
        # YAMNet expects [-1, 1] floats; clip just in case.
        wav = np.clip(wav, -1.0, 1.0)
        scores_tf, _embeddings, _spec = self._model(tf.constant(wav))
        scores = scores_tf.numpy()    # (num_frames, 521)
        if scores.size == 0:
            return self._zero_result(silent=True)

        # Average across time, then take max across class group.
        mean_scores = scores.mean(axis=0)
        music = float(max(mean_scores[i] for i in _MUSIC_CLASS_IDS))
        speech = float(max(mean_scores[i] for i in _SPEECH_CLASS_IDS))
        # Margin: how clearly one class wins over the other. The minimum gap
        # avoids flickering when scores are very close.
        gap = 0.08
        music_dominant = music > speech + gap
        speech_dominant = speech > music + gap
        return {
            "music_score": music,
            "speech_score": speech,
            "music_dominant": music_dominant,
            "speech_dominant": speech_dominant,
            "silent": False,
        }

    @staticmethod
    def _zero_result(silent: bool) -> dict:
        return {
            "music_score": 0.0,
            "speech_score": 0.0,
            "music_dominant": False,
            "speech_dominant": False,
            "silent": silent,
        }
