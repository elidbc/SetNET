import os
import math
from pkgutil import resolve_name
import random
from dataclasses import dataclass
from itertools import product

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 508


class CardSpec:
    shape: str
    color: str
    count: int
    shading: str

MASK_PATHS = {
    "squiggle": "data/masks/squiggle_mask.png",
    "pill": "data/masks/pill_mask.png",
    "diamond": "data/masks/diamond_mask.png",
}

COLORS = {
    "red": (210, 60, 60),
    "green": (40, 150, 90),
    "purple": (120, 70, 170),
    "BG": (245, 245, 242)
}

CARD_SPAN_FRAC = {1: 0.2, 2: 0.48, 3: 0.75}


def _load_master_mask(shape: str) -> Image.Image:
    # return 512x256 white on black mask of 1 shape
    return Image.open(MASK_PATHS[shape]).convert("L")

def _compute_bboxes(count: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    """
    Returns list of bboxes, one per symbol
    Symbol size constant
    """
    W, H = IMAGE_WIDTH, IMAGE_HEIGHT
    cx, cy = W // 2, H // 2

    # symbol bbox
    base_w = int(round(W * (2/3)))
    scale = rng.uniform(0.97, 1.03)
    bbox_w = max(10, int(round(base_w * scale)))
    bbox_h = max(10, int(round(bbox_w // 2)))

    # target
    target_span = int(round(H * CARD_SPAN_FRAC[count] * rng.uniform(0.97, 1.03)))

    if count == 1:
        span = bbox_h
        gap = 0
    else:
        min_gap = int(round(H * 0.05)) # ~5% of card height
        min_span = count * bbox_h + (count - 1) * min_gap
        span = max(target_span, min_span)
        gap = (span - count * bbox_h) / (count - 1)
    
    jitter_x = int(round(rng.gauss(0, W * 0.01)))
    jitter_y = int(round(rng.gauss(0, H * 0.01)))

    x0 = int(round(cx - bbox_w / 2 + jitter_x))
    x1 = x0 + bbox_w

    top = int(round(cy - span / 2 + jitter_y))

    bboxes = []
    for i in range(count):
        per_y = rng.randint(-2, 2)
        yi0 = int(round(top + i * (bbox_h + gap) + per_y))
        yi1 = yi0 + bbox_h
        bboxes.append((x0, yi0, x1, yi1))
    
    # didn't clamp
    return bboxes

def _paste_shape_masks(full_mask: Image.Image, master: Image.Image, bboxes: list[tuple[int, int, int, int]]):
    for (x0, y0, x1, y1) in bboxes:
        w, h = (x1 - x0), (y1 - y0)
        m = master.resize((w, h), resample=Image.LANCZOS)
        full_mask.paste(m, (x0, y0))

def _make_outline_mask(shape_mask_l: Image.Image, thickness_px: int) -> Image.Image:
    thickness_px = max(1, int(thickness_px))
    hard = shape_mask_l.point(lambda p: 255 if p > 127 else 0)

    k = 2 * thickness_px + 1
    dil = hard.filter(ImageFilter.MaxFilter(k))
    ero = hard.filter(ImageFilter.MinFilter(k))
    return ImageChops.subtract(dil, ero)

def _center_crop(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = img.size
    tw, th = size
    left = (w - tw) // 2
    top = (h - th) // 2
    return img.crop((left, top, left + tw, top + th))

def _draw_stripes_on_bg(bg: Image.Image, bboxes: list[tuple[int, int, int, int]], color_rgb: tuple[int, int, int], rng: random.Random) -> Image.Image:
    stripes_layer = bg.copy()
    n_lines = rng.randint(30, 35)
    #img = bg.copy()
    #d = ImageDraw.Draw(img)

    for (x0, y0, x1, y1) in bboxes:
        w, h = (x1 - x0), (y1 - y0)
        tile = Image.new("RGB", (w, h), COLORS["BG"])
        d = ImageDraw.Draw(tile)
        thickness = max(1, int(round(w / 220)))
        jitter = 0.35
        xs = []
        for i in range(n_lines):
            t = i / (n_lines - 1)  # 0..1
            x = int(round(t * (w - 1) + rng.uniform(-jitter, jitter)))
            x = max(0, min(w - 1, x))
            xs.append(x)
        for x in xs:
            d.line([(x, 0), (x, h - 1)], fill=color_rgb, width=thickness)
        
        angle = rng.uniform(-2.0, 2.0)
        if abs(angle) > 1e-6:
            rot = tile.rotate(
                angle,
                resample=Image.BICUBIC,
                expand=True,
                fillcolor=COLORS["BG"],
            )
            tile = _center_crop(rot, (w, h))

        stripes_layer.paste(tile, (x0, y0))

    return stripes_layer

def generate_mask(count: int, shape: str, seed:int | None = None):
    rng = random.Random(seed)
    master = _load_master_mask(shape)

    full_mask = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
    bboxes = _compute_bboxes(count, rng)
    _paste_shape_masks(full_mask, master, bboxes)
    return full_mask, bboxes


def generate_card(shape: str, shading: str, color: str, count: int, seed: int | None = None) -> Image.Image:
    # label: shape, color, count, shading
    rng = random.Random(seed)
    color_rgb = COLORS[color]

    bg = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), COLORS["BG"])

    shape_mask, bboxes = generate_mask(count, shape, seed=seed)

    outline_thick = max(2, int(round(min(IMAGE_WIDTH, IMAGE_HEIGHT) * 0.006)))
    outline_mask = _make_outline_mask(shape_mask, outline_thick)

    if shading == "open":
        outline_color = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color_rgb)
        out = Image.composite(outline_color, bg, outline_mask)
        return out

    if shading == "solid":
        fill = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color_rgb)
        out = Image.composite(fill, bg, shape_mask)
        outline_color = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color_rgb)
        out = Image.composite(outline_color, out, outline_mask)
        return out

    if shading == "striped":
        stripes_rgb = _draw_stripes_on_bg(bg, bboxes, color_rgb, rng)
        out = Image.composite(stripes_rgb, bg, shape_mask)

        outline_color = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color_rgb)
        out = Image.composite(outline_color, out, outline_mask)
        return out
    raise ValueError(f"Unknown Shading: {shading}")



if __name__ == "__main__":
    print("Generating green card with 2 squiggles, shaded")
    card = generate_card(shape="squiggle", shading="striped", color="green", count=3)
    card.save("generated_card.png")
    print("Generated card, saved to generated_card.png") 
