"""InsightFace-based face recognition for matching faces to People profiles.

Wraps insightface.app.FaceAnalysis (buffalo_l model) and matches each detected
face against the profile store via cosine similarity of L2-normalised
embeddings. InsightFace's ``normed_embedding`` is unit-length so the cosine
similarity is simply the dot product.

Cosine similarity guidance (buffalo_l, same person):
    > 0.45  almost certainly the same person
    > 0.35  likely the same person (default threshold)
    < 0.25  almost certainly different
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from pathlib import Path

from profiles import ProfileStore, Profile


# Default cosine similarity threshold for declaring a face match.
DEFAULT_MATCH_THRESHOLD = 0.35


class FaceRecognizer:
    """Detects faces in a frame and matches them to profiles.

    Detection and matching are both done by this class so callers don't need
    to know about InsightFace's API. The recognizer is lazily initialised — the
    first call to a method that needs the model will download and load it.
    """

    def __init__(self, store: ProfileStore,
                 det_size: tuple[int, int] = (640, 640),
                 match_threshold: float = DEFAULT_MATCH_THRESHOLD):
        self._store = store
        self._det_size = det_size
        self.match_threshold = match_threshold
        self._app = None              # insightface.app.FaceAnalysis, lazy
        self._index_dirty = True      # True when profile embeddings have changed
        self._matrix: np.ndarray | None = None   # (M, 512) — one row per profile mean embedding
        self._matrix_ids: list[str] = []         # profile IDs in row order

    # ------------------------------------------------------------------
    # Lazy model init
    # ------------------------------------------------------------------

    def _ensure_app(self):
        if self._app is not None:
            return
        # Defer the import so the rest of the app can run if insightface is missing.
        from insightface.app import FaceAnalysis

        # Use CPU only by default; CoreML can be unstable for some buffalo_l layers.
        # Users with explicit GPU/Coreml needs can set AUTOFOLLOW_FACE_PROVIDERS env.
        providers_env = os.environ.get("AUTOFOLLOW_FACE_PROVIDERS")
        if providers_env:
            providers = [p.strip() for p in providers_env.split(",") if p.strip()]
        else:
            providers = ["CPUExecutionProvider"]

        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        # ctx_id=0 picks the first provider in the list above (CPU here)
        self._app.prepare(ctx_id=0, det_size=self._det_size)

    # ------------------------------------------------------------------
    # Profile index
    # ------------------------------------------------------------------

    def mark_index_dirty(self):
        """Call after any profile add/edit/delete so the next identify() rebuilds."""
        self._index_dirty = True

    def _rebuild_index(self):
        rows: list[np.ndarray] = []
        ids: list[str] = []
        for profile in self._store.list():
            mean = profile.mean_embedding
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
    # Embedding & matching
    # ------------------------------------------------------------------

    def embed_image_path(self, image_path: Path) -> np.ndarray | None:
        """Compute a single embedding from a reference image file.

        Returns the L2-normalised 512-d embedding of the most prominent face,
        or None if no face was found.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        return self.embed_image(img)

    def embed_image(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """Embed the largest face in the given BGR image. Returns None on failure."""
        self._ensure_app()
        faces = self._app.get(img_bgr)
        if not faces:
            return None
        # Pick the largest face by bbox area
        def _area(f):
            x1, y1, x2, y2 = f.bbox
            return max(0, x2 - x1) * max(0, y2 - y1)
        f = max(faces, key=_area)
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            return None
        return np.asarray(emb, dtype=np.float32)

    def identify(self, frame_bgr: np.ndarray) -> list[dict]:
        """Detect and identify all faces in a frame.

        Returns a list of dicts, one per detected face::

            {
                'bbox':       (x1, y1, x2, y2),     # pixel coords in input frame
                'profile_id': str | None,            # best-matching profile, or None
                'name':       str | None,            # profile name (None if no match)
                'score':      float,                 # cosine similarity to that profile
                'priority':   int,                   # profile priority (0 if no match)
            }
        """
        self._ensure_app()
        if self._index_dirty:
            self._rebuild_index()

        faces = self._app.get(frame_bgr)
        results: list[dict] = []
        if not faces:
            return results

        # Stack query embeddings for vectorised cosine similarity
        embs = []
        bboxes = []
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            embs.append(np.asarray(emb, dtype=np.float32))
            x1, y1, x2, y2 = f.bbox
            bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        if not embs:
            return results

        Q = np.stack(embs)                            # (N, 512)
        if self._matrix is not None and self._matrix_ids:
            sims = Q @ self._matrix.T                 # (N, M)
            best_idx = sims.argmax(axis=1)
            best_sim = sims[np.arange(len(sims)), best_idx]
        else:
            best_idx = np.zeros(len(embs), dtype=np.int64)
            best_sim = np.zeros(len(embs), dtype=np.float32)

        for i, bbox in enumerate(bboxes):
            sim = float(best_sim[i])
            if self._matrix is not None and sim >= self.match_threshold:
                pid = self._matrix_ids[int(best_idx[i])]
                profile = self._store.get(pid)
                name = profile.name if profile else None
                priority = profile.priority if profile else 0
            else:
                pid = None
                name = None
                priority = 0
            results.append({
                "bbox": bbox,
                "profile_id": pid,
                "name": name,
                "score": sim,
                "priority": priority,
            })
        return results

    # ------------------------------------------------------------------
    # Convenience: embed every reference image of a profile from disk
    # ------------------------------------------------------------------

    def rebuild_profile_embeddings(self, profile: Profile) -> int:
        """Re-embed every reference image of a profile from disk. Returns count."""
        embeddings: dict[str, np.ndarray] = {}
        for fn in list(profile.image_filenames):
            path = self._store.image_path(profile.id, fn)
            if path is None:
                continue
            emb = self.embed_image_path(path)
            if emb is not None:
                embeddings[fn] = emb
        self._store.save_embeddings(profile, embeddings)
        self.mark_index_dirty()
        return len(embeddings)
