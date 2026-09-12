"""Camera discovery and opening — shared by the GUI, headless mode and --list-cameras."""

import cv2
import config


def scan_cameras(max_cams: int = 8, skip: set[int] | None = None) -> list[int]:
    """Return the indices of cameras that can be opened.

    Indices in `skip` are assumed available without probing — used so a rescan
    never tries to re-open the camera the video thread is currently streaming.
    Scanning stops after two consecutive misses: device indices are contiguous
    on macOS, and each failed probe is slow and spams stderr.
    """
    skip = skip or set()
    available = []
    misses = 0
    for i in range(max_cams):
        if i in skip:
            available.append(i)
            misses = 0
            continue
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        cap.release()
        if ok:
            available.append(i)
            misses = 0
        else:
            misses += 1
            if misses >= 2:
                break
    return sorted(available)


def open_capture(index: int) -> cv2.VideoCapture | None:
    """Open a camera, apply the configured capture size, and return it (or None)."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    if config.CAPTURE_WIDTH > 0 and config.CAPTURE_HEIGHT > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
    return cap


def describe_capture(cap: cv2.VideoCapture) -> tuple[int, int, float]:
    """Return (width, height, fps) for an open capture."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return w, h, fps
