"""Input-resolution geometry shared by the framing engine, smoother and tracker.

The framing pipeline is tuned in *reference pixels*: the pixel-unit constants
in config.py (pan/tilt speeds, activity noise floor, …) describe motion in a
1280×720 wide shot.  Every camera is different, so the actual input size is
detected from the frames the camera delivers and these helpers convert between
the two spaces.  At a 16:9 720p input every scale factor is exactly 1.0, so the
tuned behaviour is unchanged there; a 4K camera simply gets pan speeds three
times larger in source pixels, which is the same motion on screen.
"""

import numpy as np
import config


def wide_zoom(input_width: int, input_height: int) -> float:
    """Zoom at which the OUTPUT_WIDTH × OUTPUT_HEIGHT crop covers the whole frame.

    Zoom is expressed relative to the output size (zoom 1.0 = a crop of exactly
    OUTPUT_WIDTH × OUTPUT_HEIGHT source pixels), so this is the smallest zoom
    the virtual camera can reach on a given input.
    """
    return max(config.OUTPUT_WIDTH / max(1, int(input_width)),
               config.OUTPUT_HEIGHT / max(1, int(input_height)))


def px_scale(input_width: int, input_height: int) -> float:
    """Source pixels per reference pixel for this input size.

    This is the width of the wide-shot crop divided by OUTPUT_WIDTH: 1.0 for a
    720p input, 1.5 for 1080p, 3.0 for 4K, 0.5 for 640×480.  Multiply a
    pixel-unit tuning constant by it to get the equivalent in source pixels;
    multiply a zoom-unit constant by ``1 / px_scale`` (i.e. ``wide_zoom``)
    to get the equivalent zoom step, since zoom values shrink in proportion.
    """
    return 1.0 / wide_zoom(input_width, input_height)


def frame_size(frame: np.ndarray) -> tuple[int, int]:
    """(width, height) of a BGR frame."""
    h, w = frame.shape[:2]
    return int(w), int(h)
