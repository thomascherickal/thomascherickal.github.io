#!/usr/bin/env python3
"""
generate_favicons.py — Generate all website favicon files from assets/site-icon.webp.

Outputs:
  - assets/favicon-16x16.png (16x16)
  - assets/favicon-32x32.png (32x32)
  - assets/favicon-96x96.png (96x96)
  - assets/apple-touch-icon.png (180x180)
  - assets/android-chrome-192x192.png (192x192)
  - assets/android-chrome-512x512.png (512x512)
  - assets/favicon.ico (multi-res 16, 32, 48)
  - assets/favicon.svg (crisp SVG wrapper with embedded raster URI)
"""

import base64
import io
import os
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(REPO_DIR, "assets")
SOURCE_IMAGE_JPEG = os.path.join(ASSETS_DIR, "site-icon.jpeg")
SOURCE_IMAGE_WEBP = os.path.join(ASSETS_DIR, "site-icon.webp")
SOURCE_IMAGE = SOURCE_IMAGE_JPEG if os.path.exists(SOURCE_IMAGE_JPEG) else SOURCE_IMAGE_WEBP

def generate_favicons():
    if not os.path.exists(SOURCE_IMAGE):
        raise FileNotFoundError(f"Source image not found: neither {SOURCE_IMAGE_JPEG} nor {SOURCE_IMAGE_WEBP} exists")

    src = Image.open(SOURCE_IMAGE)
    print(f"Loaded master image: {SOURCE_IMAGE} ({src.size[0]}x{src.size[1]}, {src.mode})")

    # If source is JPEG, ensure a high-quality site-icon.webp is also saved for schema/logo compatibility
    if SOURCE_IMAGE == SOURCE_IMAGE_JPEG:
        webp_path = SOURCE_IMAGE_WEBP
        src.save(webp_path, format="WEBP", quality=95)
        print(f"Generated site-icon.webp (1024x1024): {os.path.getsize(webp_path):,} bytes")

    # 1. Standard PNG sizes
    png_targets = {
        "favicon-16x16.png": (16, 16),
        "favicon-32x32.png": (32, 32),
        "favicon-96x96.png": (96, 96),
        "apple-touch-icon.png": (180, 180),
        "android-chrome-192x192.png": (192, 192),
        "android-chrome-512x512.png": (512, 512),
    }

    for filename, size in png_targets.items():
        out_path = os.path.join(ASSETS_DIR, filename)
        resized = src.resize(size, Image.Resampling.LANCZOS)
        resized.save(out_path, format="PNG", optimize=True)
        print(f"Generated {filename} ({size[0]}x{size[1]}): {os.path.getsize(out_path):,} bytes")

    # 2. Multi-resolution ICO
    ico_path = os.path.join(ASSETS_DIR, "favicon.ico")
    ico_sizes = [(16, 16), (32, 32), (48, 48)]
    src.save(ico_path, format="ICO", sizes=ico_sizes)
    print(f"Generated favicon.ico (16, 32, 48): {os.path.getsize(ico_path):,} bytes")

    # 3. SVG Favicon (crisp 128x128 embedded PNG URI for universal vector fallback)
    svg_img = src.resize((128, 128), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    svg_img.save(buf, format="PNG", optimize=True)
    b64_png = base64.b64encode(buf.getvalue()).decode("ascii")

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="100%" height="100%">
  <image width="128" height="128" href="data:image/png;base64,{b64_png}"/>
</svg>
'''
    svg_path = os.path.join(ASSETS_DIR, "favicon.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Generated favicon.svg: {os.path.getsize(svg_path):,} bytes")

if __name__ == "__main__":
    generate_favicons()
