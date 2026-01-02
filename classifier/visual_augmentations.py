# augmentations.py
import random
import numpy as np
from PIL import Image, ImageEnhance

BG = (245, 245, 242)

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def color_jitter_pil(img: Image.Image, rng: random.Random,
                     brightness: float = 0.25,
                     contrast: float = 0.25,
                     hue: float = 0.06) -> Image.Image:
    # brightness/contrast: multiply by factor in [1-b, 1+b]
    if brightness > 0:
        f = 1.0 + rng.uniform(-brightness, brightness)
        img = ImageEnhance.Brightness(img).enhance(f)

    if contrast > 0:
        f = 1.0 + rng.uniform(-contrast, contrast)
        img = ImageEnhance.Contrast(img).enhance(f)

    # hue: shift hue channel in HSV
    if hue > 0:
        dh = rng.uniform(-hue, hue)  # fraction of full circle
        hsv = img.convert("HSV")
        h, s, v = hsv.split()
        h_np = np.array(h, dtype=np.uint8)
        shift = int(round(dh * 255))  # HSV hue is 0..255 in PIL
        h_np = (h_np.astype(np.int16) + shift) % 256
        h2 = Image.fromarray(h_np.astype(np.uint8), mode="L")
        img = Image.merge("HSV", (h2, s, v)).convert("RGB")

    return img

def random_rotate_pil(img: Image.Image, rng: random.Random,
                      degrees: float = 6.0,
                      fill=BG) -> Image.Image:
    angle = rng.uniform(-degrees, degrees)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill)

def _find_perspective_coeffs(src_pts, dst_pts):
    # Solve for coefficients mapping src -> dst
    # Returns 8 coeffs for PIL Image.transform(PERSPECTIVE)
    A = []
    B = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        B.append(u)
        B.append(v)
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    coeffs = np.linalg.lstsq(A, B, rcond=None)[0]
    return coeffs.tolist()

def random_perspective_pil(img: Image.Image, rng: random.Random,
                           distortion: float = 0.08,
                           p: float = 0.7,
                           fill=BG) -> Image.Image:
    if rng.random() > p or distortion <= 0:
        return img

    w, h = img.size
    d = distortion * min(w, h)

    # Source corners
    src = [(0, 0), (w, 0), (w, h), (0, h)]

    # Jitter destination corners
    def jx(): return rng.uniform(-d, d)
    def jy(): return rng.uniform(-d, d)

    dst = [(0 + jx(), 0 + jy()),
           (w + jx(), 0 + jy()),
           (w + jx(), h + jy()),
           (0 + jx(), h + jy())]

    coeffs = _find_perspective_coeffs(src, dst)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         resample=Image.BICUBIC, fillcolor=fill)

def random_crop_resize_pil(img: Image.Image, rng: random.Random,
                           scale_min: float = 0.90,
                           scale_max: float = 1.00) -> Image.Image:
    w, h = img.size
    s = rng.uniform(scale_min, scale_max)
    cw = max(1, int(round(w * s)))
    ch = max(1, int(round(h * s)))

    if cw == w and ch == h:
        return img

    x0 = rng.randint(0, w - cw)
    y0 = rng.randint(0, h - ch)
    cropped = img.crop((x0, y0, x0 + cw, y0 + ch))
    return cropped.resize((w, h), resample=Image.BICUBIC)

def augment_card(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = color_jitter_pil(img, rng, brightness=0.25, contrast=0.25, hue=0.06)
    img = random_rotate_pil(img, rng, degrees=6.0, fill=BG)
    img = random_perspective_pil(img, rng, distortion=0.08, p=0.7, fill=BG)
    img = random_crop_resize_pil(img, rng, scale_min=0.90, scale_max=1.00)
    return img