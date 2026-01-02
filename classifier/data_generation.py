import os
import math
from pkgutil import resolve_name
import random
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFilter, ImageChops
from visual_augmentations import augment_card

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 508


MASK_PATHS = {
    "squiggle": "data/masks/squiggle_mask.png",
    "pill": "data/masks/pill_mask.png",
    "diamond": "data/masks/diamond_mask.png",
}

RGBS = {
    "red": (210, 60, 60),
    "green": (40, 150, 90),
    "purple": (120, 70, 170),
    "BG": (245, 245, 242)
}

CARD_SPAN_FRAC = {1: 0.2, 2: 0.48, 3: 0.75}

SHAPES = ["pill", "diamond", "squiggle"]
COLORS = ["green", "red", "purple"]
SHADINGS = ["open", "solid", "striped"]
COUNTS = [1, 2, 3]

shape2i = {s:i for i,s in enumerate(SHAPES)}
color2i = {c:i for i,c in enumerate(COLORS)}
shad2i = {s:i for i,s in enumerate(SHADINGS)}
count2i = {c:i for i,c in enumerate(COUNTS)}

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
   
    for (x0, y0, x1, y1) in bboxes:
        w, h = (x1 - x0), (y1 - y0)
        tile = Image.new("RGB", (w, h), RGBS["BG"])
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
                fillcolor=RGBS["BG"],
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
    color_rgb = RGBS[color]

    bg = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), RGBS["BG"])

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

def _seed32(*xs) -> int:
    # deterministic 32-bit seed from tuple
    h = 2166136261
    for x in xs:
        for b in str(x).encode("utf-8"):
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
    return h

class SetCardDataset(Dataset):
    def __init__(self, n_samples: int, base_seed: int, augment: bool):
        self.n = n_samples
        self.base_seed = base_seed
        self.augment = augment

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        # Create 3 separate seeds from base_seed and idx for deterministic but independent randomness
        seed_labels = _seed32(self.base_seed, idx, "labels")
        seed_generation = _seed32(self.base_seed, idx, "generation")
        seed_augmentation = _seed32(self.base_seed, idx, "augmentation")

        # Use seed_labels for choosing card attributes
        rng_labels = random.Random(seed_labels)
        shape = rng_labels.choice(SHAPES)
        color = rng_labels.choice(COLORS)
        shading = rng_labels.choice(SHADINGS)
        count = rng_labels.choice(COUNTS)
        #print(f"Card randomly created — shape: {shape}, color: {color}, shading: {shading}, count: {count}")

        # Use seed_generation for card generation
        img = generate_card(shape=shape, shading=shading, color=color, count=count, seed=seed_generation)

        # Use seed_augmentation for visual augmentations
        if self.augment:
            img = augment_card(img, seed_augmentation)
        #print(f"saved card to sanity_check.png")
        img.save("sanity_check.png")
        
        x = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0

        y_shape = shape2i[shape]
        y_color = color2i[color]
        y_shad = shad2i[shading]
        y_count = count2i[count]

        return x, (y_shape, y_color, y_shad, y_count)

if __name__ == "__main__":
    base_seed = 45
    sample_ds = SetCardDataset(1, base_seed, True)
    #print(f"Data sample # one from sample_ds: {sample_ds[0][1:]}")
