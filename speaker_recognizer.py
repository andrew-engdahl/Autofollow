"""SpeechBrain ECAPA-TDNN speaker recognition.

Wraps the speechbrain/spkrec-ecapa-voxceleb model. Produces 192-dimensional
speaker embeddings; matches by cosine similarity (L2-normalised → dot product).

Cosine-similarity guidance for ECAPA-TDNN on clean speech:
    > 0.55  same speaker (high confidence)
    > 0.40  likely same speaker (default threshold)
    < 0.30  almost certainly different

The recognizer is lazy: the model is downloaded and loaded on first use.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from profiles import ProfileStore, Profile


DEFAULT_MATCH_THRESHOLD = 0.40
TARGET_SAMPLE_RATE = 16000   # ECAPA-VoxCeleb is trained at 16 kHz


class SpeakerRecognizer:
    """Embeds audio waveforms and matches them to enrolled profiles."""

    def __init__(self, store: ProfileStore,
                 match_threshold: float = DEFAULT_MATCH_THRESHOLD):
        self._store = store
        self.match_threshold = match_threshold
        self._model = None              # lazy SpeechBrain SpeakerRecognition
        self._index_dirty = True
        self._matrix: np.ndarray | None = None    # (M, 192) per-profile means
        self._matrix_ids: list[str] = []

    # ------------------------------------------------------------------
    # Lazy model init
    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return
        # speechbrain.inference uses Hugging Face Hub on first call to fetch
        # the model into a savedir; subsequent calls load from disk.
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        savedir = Path.home() / ".autofollow" / "models" / "ecapa-voxceleb"
        savedir.mkdir(parents=True, exist_ok=True)

        # Prefer Apple Silicon Metal (MPS) when available; fall back to CPU.
        device = "cpu"
        try:
            if torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            pass

        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir),
            run_opts={"device": device},
        )

    # ------------------------------------------------------------------
    # Profile index
    # ------------------------------------------------------------------

    def mark_index_dirty(self):
        self._index_dirty = True

    def _rebuild_index(self):
        rows: list[np.ndarray] = []
        ids: list[str] = []
        for profile in self._store.list():
            mean = profile.mean_voice_embedding
            if mean is None:
                continue
            rows.append(mean.astype(np.float32))
            ids.append(profile.id)
        if rows:
            self._matrix = np.stack(rows)
        else:
            self._matrix = None
        self._matrix_ids = ids
        self._index_dirty = False

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_waveform(self, waveform: np.ndarray,
                       sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray | None:
        """Compute a single L2-normalised 192-d embedding from a mono waveform.

        ``waveform`` is a 1-D float32 array. Returns None if the waveform is
        too short (< 0.5 s) or all-silent.
        """
        if waveform is None or waveform.size == 0:
            return None
        if sample_rate != TARGET_SAMPLE_RATE:
            # Lazy import; resampling is rare on a hot path because we capture
            # at 16 kHz natively.
            try:
                import librosa
                waveform = librosa.resample(
                    waveform, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
                )
            except ImportError:
                # No librosa → simple decimation by integer ratio
                ratio = max(1, int(sample_rate / TARGET_SAMPLE_RATE))
                waveform = waveform[::ratio]
        if waveform.size < TARGET_SAMPLE_RATE // 2:   # < 0.5 s
            return None
        rms = float(np.sqrt(np.mean(waveform.astype(np.float32) ** 2)))
        if rms < 1e-4:
            return None    # essentially silent

        self._ensure_model()
        import torch
        wav_t = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            emb = self._model.encode_batch(wav_t).squeeze().cpu().numpy()
        # SpeechBrain returns un-normalised embeddings — normalise so cosine
        # similarity reduces to a dot product.
        norm = np.linalg.norm(emb)
        if norm < 1e-9:
            return None
        return (emb / norm).astype(np.float32)

    def embed_file(self, audio_path: Path) -> np.ndarray | None:
        """Load an audio file (any sample rate) and embed it."""
        try:
            import soundfile as sf
        except ImportError:
            return None
        try:
            data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        except Exception:
            return None
        if data.ndim == 2:
            # Stereo → mono mix
            data = data.mean(axis=1)
        return self.embed_waveform(data, sample_rate=sr)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def identify(self, waveform: np.ndarray,
                 sample_rate: int = TARGET_SAMPLE_RATE) -> dict | None:
        """Embed and match the given waveform against enrolled profiles.

        Returns ``{'profile_id', 'name', 'score', 'priority'}`` on match,
        or None if no enrolled voice exceeds the threshold (or the waveform
        couldn't be embedded).
        """
        if self._index_dirty:
            self._rebuild_index()
        if self._matrix is None or not self._matrix_ids:
            return None
        emb = self.embed_waveform(waveform, sample_rate=sample_rate)
        if emb is None:
            return None
        sims = self._matrix @ emb            # (M,)
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < self.match_threshold:
            return None
        pid = self._matrix_ids[best]
        profile = self._store.get(pid)
        if profile is None:
            return None
        return {
            "profile_id": pid,
            "name": profile.name,
            "score": score,
            "priority": profile.priority,
        }

    # ------------------------------------------------------------------

    def rebuild_profile_voice_embeddings(self, profile: Profile) -> int:
        """Re-embed every voice sample of a profile from disk. Returns count."""
        embeddings: dict[str, np.ndarray] = {}
        for fn in list(profile.voice_filenames):
            path = self._store.voice_path(profile.id, fn)
            if path is None:
                continue
            emb = self.embed_file(path)
            if emb is not None:
                embeddings[fn] = emb
        self._store.save_voice_embeddings(profile, embeddings)
        self.mark_index_dirty()
        return len(embeddings)
