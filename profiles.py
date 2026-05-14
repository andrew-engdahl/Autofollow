"""Person profile store — filesystem-backed registry of recognized people.

Layout::

    ~/Library/Application Support/Autofollow/profiles/
        <profile_id>/
            profile.json          # {id, name, priority, created_at, updated_at}
            images/               # reference jpegs/pngs added by the user
                <hash>.jpg
                ...
            embeddings.npy        # (N, 512) face embeddings, one row per image
            embeddings_index.json # {image_filename: row_index}

`priority` is a 0–10 integer. 0 = no boost (normal candidate). 10 = always
preferred over unmatched subjects. Used by the switcher to bias selection
and dwell toward important people (e.g. the pastor at a church).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


_APP_SUPPORT = Path.home() / "Library" / "Application Support" / "Autofollow"
PROFILES_DIR = _APP_SUPPORT / "profiles"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_EMBED_DIM = 512  # InsightFace buffalo_l normed_embedding dim


def _slugify(name: str) -> str:
    s = _SLUG_RE.sub("-", name.lower()).strip("-")
    return s or "person"


@dataclass
class Profile:
    id: str
    name: str
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Runtime-loaded; not persisted in profile.json
    embeddings: np.ndarray | None = None        # (N, 512) float32, L2-normalised
    image_filenames: list[str] = field(default_factory=list)
    voice_embeddings: np.ndarray | None = None  # (M, 192) float32, L2-normalised
    voice_filenames: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @property
    def n_images(self) -> int:
        return len(self.image_filenames)

    @property
    def n_voice_samples(self) -> int:
        return len(self.voice_filenames)

    @property
    def mean_embedding(self) -> np.ndarray | None:
        if self.embeddings is None or len(self.embeddings) == 0:
            return None
        m = self.embeddings.mean(axis=0)
        n = np.linalg.norm(m)
        return m / n if n > 0 else None

    @property
    def mean_voice_embedding(self) -> np.ndarray | None:
        if self.voice_embeddings is None or len(self.voice_embeddings) == 0:
            return None
        m = self.voice_embeddings.mean(axis=0)
        n = np.linalg.norm(m)
        return m / n if n > 0 else None


class ProfileStore:
    """Filesystem-backed CRUD for person profiles."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else PROFILES_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, Profile] = {}
        self.reload()

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------

    def reload(self):
        """Re-scan the profiles directory from disk."""
        self._profiles.clear()
        if not self.root.exists():
            return
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir():
                continue
            meta_path = entry / "profile.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            profile = Profile(
                id=meta.get("id", entry.name),
                name=meta.get("name", entry.name),
                priority=int(meta.get("priority", 0)),
                created_at=float(meta.get("created_at", time.time())),
                updated_at=float(meta.get("updated_at", time.time())),
            )
            images_dir = entry / "images"
            if images_dir.exists():
                profile.image_filenames = sorted(
                    p.name for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
                )

            embed_path = entry / "embeddings.npy"
            index_path = entry / "embeddings_index.json"
            if embed_path.exists() and index_path.exists():
                try:
                    arr = np.load(embed_path)
                    with open(index_path) as f:
                        idx = json.load(f)
                    rows = []
                    for fn in profile.image_filenames:
                        if fn in idx:
                            rows.append(arr[idx[fn]])
                    if rows:
                        profile.embeddings = np.stack(rows).astype(np.float32)
                except (OSError, ValueError, KeyError):
                    profile.embeddings = None

            # Voice samples + cached embeddings
            voice_dir = entry / "voice"
            if voice_dir.exists():
                profile.voice_filenames = sorted(
                    p.name for p in voice_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in (".wav", ".flac", ".ogg")
                )
            v_embed_path = entry / "voice_embeddings.npy"
            v_index_path = entry / "voice_embeddings_index.json"
            if v_embed_path.exists() and v_index_path.exists():
                try:
                    arr = np.load(v_embed_path)
                    with open(v_index_path) as f:
                        idx = json.load(f)
                    rows = []
                    for fn in profile.voice_filenames:
                        if fn in idx:
                            rows.append(arr[idx[fn]])
                    if rows:
                        profile.voice_embeddings = np.stack(rows).astype(np.float32)
                except (OSError, ValueError, KeyError):
                    profile.voice_embeddings = None

            self._profiles[profile.id] = profile

    def save_meta(self, profile: Profile):
        """Persist profile.json for a profile (does not touch embeddings)."""
        profile_dir = self.root / profile.id
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "images").mkdir(exist_ok=True)
        profile.updated_at = time.time()
        with open(profile_dir / "profile.json", "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def save_embeddings(self, profile: Profile, embeddings_by_filename: dict[str, np.ndarray]):
        """Persist embeddings.npy + index for the given profile.

        embeddings_by_filename: {image_filename: 512-d float32 ndarray (L2-normalised)}.
        Filenames present in the profile but absent from the dict are dropped.
        """
        profile_dir = self.root / profile.id
        ordered = [fn for fn in profile.image_filenames if fn in embeddings_by_filename]
        if not ordered:
            embed_path = profile_dir / "embeddings.npy"
            index_path = profile_dir / "embeddings_index.json"
            if embed_path.exists():
                embed_path.unlink()
            if index_path.exists():
                index_path.unlink()
            profile.embeddings = None
            return
        arr = np.stack([embeddings_by_filename[fn] for fn in ordered]).astype(np.float32)
        np.save(profile_dir / "embeddings.npy", arr)
        index = {fn: i for i, fn in enumerate(ordered)}
        with open(profile_dir / "embeddings_index.json", "w") as f:
            json.dump(index, f)
        profile.embeddings = arr

    def save_voice_embeddings(self, profile: Profile,
                              embeddings_by_filename: dict[str, np.ndarray]):
        """Persist voice_embeddings.npy + index for the given profile.

        embeddings_by_filename: {voice_filename: 192-d float32 ndarray (L2-normalised)}.
        Filenames present in the profile but absent from the dict are dropped.
        """
        profile_dir = self.root / profile.id
        ordered = [fn for fn in profile.voice_filenames if fn in embeddings_by_filename]
        if not ordered:
            embed_path = profile_dir / "voice_embeddings.npy"
            index_path = profile_dir / "voice_embeddings_index.json"
            if embed_path.exists():
                embed_path.unlink()
            if index_path.exists():
                index_path.unlink()
            profile.voice_embeddings = None
            return
        arr = np.stack([embeddings_by_filename[fn] for fn in ordered]).astype(np.float32)
        np.save(profile_dir / "voice_embeddings.npy", arr)
        index = {fn: i for i, fn in enumerate(ordered)}
        with open(profile_dir / "voice_embeddings_index.json", "w") as f:
            json.dump(index, f)
        profile.voice_embeddings = arr

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list(self) -> list[Profile]:
        return sorted(self._profiles.values(), key=lambda p: p.name.lower())

    def get(self, profile_id: str) -> Profile | None:
        return self._profiles.get(profile_id)

    def get_by_name(self, name: str) -> Profile | None:
        for p in self._profiles.values():
            if p.name.lower() == name.lower():
                return p
        return None

    def create(self, name: str, priority: int = 0) -> Profile:
        """Create a new profile, choosing a unique slug for the directory."""
        base = _slugify(name)
        slug = base
        i = 2
        while (self.root / slug).exists() or slug in self._profiles:
            slug = f"{base}-{i}"
            i += 1
        profile = Profile(id=slug, name=name, priority=int(priority))
        self.save_meta(profile)
        self._profiles[profile.id] = profile
        return profile

    def update(self, profile_id: str, *, name: str | None = None,
               priority: int | None = None) -> Profile | None:
        profile = self._profiles.get(profile_id)
        if profile is None:
            return None
        if name is not None and name.strip():
            profile.name = name.strip()
        if priority is not None:
            profile.priority = max(0, min(10, int(priority)))
        self.save_meta(profile)
        return profile

    def delete(self, profile_id: str) -> bool:
        profile = self._profiles.pop(profile_id, None)
        if profile is None:
            return False
        profile_dir = self.root / profile_id
        if profile_dir.exists():
            for p in sorted(profile_dir.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            profile_dir.rmdir()
        return True

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def add_image_bytes(self, profile_id: str, image_bytes: bytes,
                        suffix: str = ".jpg") -> str | None:
        """Write raw image bytes into the profile's images/ dir. Returns filename."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            return None
        images_dir = self.root / profile.id / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        suffix = suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png"):
            suffix = ".jpg"
        filename = f"{uuid.uuid4().hex}{suffix}"
        with open(images_dir / filename, "wb") as f:
            f.write(image_bytes)
        profile.image_filenames.append(filename)
        return filename

    def remove_image(self, profile_id: str, filename: str) -> bool:
        profile = self._profiles.get(profile_id)
        if profile is None:
            return False
        path = self.root / profile.id / "images" / filename
        if path.exists():
            path.unlink()
        if filename in profile.image_filenames:
            profile.image_filenames.remove(filename)
        return True

    def image_path(self, profile_id: str, filename: str) -> Path | None:
        profile = self._profiles.get(profile_id)
        if profile is None:
            return None
        path = self.root / profile.id / "images" / filename
        return path if path.exists() else None

    # ------------------------------------------------------------------
    # Voice samples
    # ------------------------------------------------------------------

    def add_voice_sample_bytes(self, profile_id: str, audio_bytes: bytes,
                               suffix: str = ".wav") -> str | None:
        """Write raw audio bytes into the profile's voice/ dir. Returns filename."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            return None
        voice_dir = self.root / profile.id / "voice"
        voice_dir.mkdir(parents=True, exist_ok=True)
        suffix = suffix.lower()
        if suffix not in (".wav", ".flac", ".ogg"):
            suffix = ".wav"
        filename = f"{uuid.uuid4().hex}{suffix}"
        with open(voice_dir / filename, "wb") as f:
            f.write(audio_bytes)
        profile.voice_filenames.append(filename)
        return filename

    def remove_voice_sample(self, profile_id: str, filename: str) -> bool:
        profile = self._profiles.get(profile_id)
        if profile is None:
            return False
        path = self.root / profile.id / "voice" / filename
        if path.exists():
            path.unlink()
        if filename in profile.voice_filenames:
            profile.voice_filenames.remove(filename)
        return True

    def voice_path(self, profile_id: str, filename: str) -> Path | None:
        profile = self._profiles.get(profile_id)
        if profile is None:
            return None
        path = self.root / profile.id / "voice" / filename
        return path if path.exists() else None
