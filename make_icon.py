#!/usr/bin/env python3
"""Generate Autofollow.app/Contents/Resources/AppIcon.icns from scratch using Pillow."""

import math
import os
import struct
import zlib
from pathlib import Path


def draw_icon(size: int) -> bytes:
    """Return raw RGBA bytes for the icon at the given square size."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    s = size
    # Background: rounded square, dark blue-grey
    pad = int(s * 0.08)
    corner = int(s * 0.22)
    d.rounded_rectangle([pad, pad, s - pad, s - pad], radius=corner,
                         fill=(24, 28, 38, 255))

    cx, cy = s / 2, s / 2

    # Lens rings (outer → inner)
    def circle(draw, cx, cy, r, fill=None, outline=None, width=1):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill,
                     outline=outline, width=width)

    r_outer  = s * 0.36
    r_glass  = s * 0.28
    r_iris   = s * 0.20
    r_pupil  = s * 0.12

    # Outer ring — blue accent
    circle(d, cx, cy, r_outer, outline=(70, 130, 220, 255),
           width=max(2, int(s * 0.025)))
    # Glass fill
    circle(d, cx, cy, r_glass, fill=(34, 44, 68, 255))
    # Iris ring
    circle(d, cx, cy, r_iris, outline=(55, 105, 185, 255),
           width=max(1, int(s * 0.015)))
    # Pupil
    circle(d, cx, cy, r_pupil, fill=(14, 16, 24, 255))

    # Specular highlight — small off-centre oval
    hx = cx - r_glass * 0.38
    hy = cy - r_glass * 0.38
    hr = r_glass * 0.22
    # Soft highlight via layered alpha ellipses
    for alpha in [40, 70, 100]:
        hr_i = hr * (1.4 - alpha / 100)
        d.ellipse([hx - hr_i, hy - hr_i * 0.65,
                   hx + hr_i, hy + hr_i * 0.65],
                  fill=(210, 228, 255, alpha))

    # "AF" text mark at bottom of lens for small sizes; wordmark for large
    if size >= 128:
        from PIL import ImageFont
        try:
            font_size = max(8, int(s * 0.13))
            # Try system fonts
            for path in [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/SFNSDisplay.ttf",
                "/System/Library/Fonts/Arial.ttf",
            ]:
                if os.path.exists(path):
                    font = ImageFont.truetype(path, font_size)
                    break
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        text = "AF"
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = cx - tw / 2
        ty = cy - th / 2
        d.text((tx, ty), text, fill=(200, 215, 245, 230), font=font)

    return img.tobytes("raw", "RGBA")


def make_png(size: int) -> bytes:
    """Return PNG bytes for the icon at the given size."""
    from PIL import Image
    import io
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    from PIL import ImageDraw
    # Re-draw onto image by compositing
    raw = draw_icon(size)
    img = Image.frombytes("RGBA", (size, size), raw)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_icns(out_path: Path):
    """Build a minimal .icns file containing several icon sizes."""
    # icns OSType → pixel size mapping (only uncompressed PNG icons used here)
    sizes = [
        (b'ic07',  128),
        (b'ic08',  256),
        (b'ic09',  512),
        (b'ic10', 1024),
        (b'ic11',   32),   # 16@2x
        (b'ic12',   64),   # 32@2x
        (b'ic13',  128),   # 64@2x (duplicate key ok — different OSType)
        (b'icp4',   16),
        (b'icp5',   32),
    ]

    chunks = b''
    for ostype, sz in sizes:
        png = make_png(sz)
        # Each chunk: 4-byte OSType + 4-byte length (including 8-byte header) + data
        chunk_len = 8 + len(png)
        chunks += ostype + struct.pack('>I', chunk_len) + png

    # File header: 'icns' + total file length
    total = 8 + len(chunks)
    data = b'icns' + struct.pack('>I', total) + chunks
    out_path.write_bytes(data)
    print(f"Written {out_path} ({total} bytes)")


if __name__ == '__main__':
    resources = Path(__file__).parent / 'Autofollow.app' / 'Contents' / 'Resources'
    resources.mkdir(parents=True, exist_ok=True)
    build_icns(resources / 'AppIcon.icns')
