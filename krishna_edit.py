#!/usr/bin/env python3

import argparse
import math
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
try:
	import cv2  # type: ignore
	HAS_CV2 = True
except Exception:
	cv2 = None  # type: ignore
	HAS_CV2 = False


BoundingBox = Tuple[int, int, int, int]


def detect_face_bbox(bgr_image: np.ndarray) -> Optional[BoundingBox]:
	if not HAS_CV2:
		return None
	gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
	# Use OpenCV's built-in haar cascade
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
	if len(faces) == 0:
		return None
	# Choose the largest face (in case of multiple)
	x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
	return int(x), int(y), int(w), int(h)


def draw_crown_and_feather(overlay: Image.Image, face_bbox: BoundingBox) -> None:
    draw = ImageDraw.Draw(overlay, 'RGBA')
    x, y, w, h = face_bbox

    crown_width = int(1.25 * w)
    crown_height = int(0.55 * h)
    crown_x = int(x + (w - crown_width) / 2)
    crown_y = int(y - 0.5 * crown_height)

    base_height = int(0.30 * crown_height)
    base_rect = [crown_x, crown_y + crown_height - base_height, crown_x + crown_width, crown_y + crown_height]

    gold = (218, 165, 32, 235)
    dark_gold = (184, 134, 11, 235)

    draw.rounded_rectangle(base_rect, radius=base_height // 3, fill=gold, outline=dark_gold, width=3)

    spikes = 5
    spike_width = crown_width // spikes
    spike_height = crown_height - base_height
    for i in range(spikes):
        sx0 = crown_x + i * spike_width
        sx1 = sx0 + spike_width
        apex_x = (sx0 + sx1) // 2
        apex_y = crown_y
        poly = [(sx0 + 6, crown_y + crown_height - base_height), (sx1 - 6, crown_y + crown_height - base_height), (apex_x, apex_y)]
        draw.polygon(poly, fill=gold, outline=dark_gold)
        jewel_r = max(3, spike_width // 10)
        draw.ellipse([apex_x - jewel_r, apex_y - jewel_r, apex_x + jewel_r, apex_y + jewel_r], fill=(138, 43, 226, 220), outline=(255, 255, 255, 230))

    # Peacock feather anchored near the right-most spike top
    feather_anchor = (crown_x + int(0.88 * crown_width), crown_y + int(0.12 * crown_height))
    draw_peacock_feather(draw, feather_anchor, length=int(0.9 * h))


def draw_peacock_feather(draw: ImageDraw.ImageDraw, anchor: Tuple[int, int], length: int = 200) -> None:
    ax, ay = anchor
    # Draw a stylized curved shaft
    points = []
    for t in np.linspace(0, 1, 30):
        dx = int(0.05 * length * math.sin(t * math.pi))
        px = ax - int(0.1 * length) + dx + int(t * 0.6 * length)
        py = ay - int(t * length)
        points.append((px, py))
    # shaft
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=(139, 69, 19, 220), width=3)
    # barbs
    for i, (px, py) in enumerate(points[5:]):
        frac = i / max(1, len(points) - 6)
        spread = int(12 + 30 * (1 - abs(0.5 - frac) * 2))
        color = (0, int(120 + 80 * frac), int(60 + 140 * frac), 180)
        draw.line([(px, py), (px - spread, py + 4)], fill=color, width=2)
        draw.line([(px, py), (px + spread, py + 4)], fill=color, width=2)
    # eye
    eye_center = points[-1]
    r_outer = max(10, length // 12)
    r_mid = int(r_outer * 0.65)
    r_inner = int(r_outer * 0.35)
    draw.ellipse([eye_center[0] - r_outer, eye_center[1] - r_outer, eye_center[0] + r_outer, eye_center[1] + r_outer], fill=(0, 128, 0, 180))
    draw.ellipse([eye_center[0] - r_mid, eye_center[1] - r_mid, eye_center[0] + r_mid, eye_center[1] + r_mid], fill=(0, 191, 255, 180))
    draw.ellipse([eye_center[0] - r_inner, eye_center[1] - r_inner, eye_center[0] + r_inner, eye_center[1] + r_inner], fill=(75, 0, 130, 220))


def draw_tilak(draw: ImageDraw.ImageDraw, face_bbox: BoundingBox) -> None:
    x, y, w, h = face_bbox
    center_x = x + w // 2
    forehead_y = y + int(0.18 * h)
    height = max(8, int(0.28 * h))
    width = max(6, int(0.16 * w))

    # U-shaped tilak
    thickness = max(2, height // 12)
    # Left arc
    bbox_left = [center_x - width, forehead_y, center_x, forehead_y + height]
    draw.arc(bbox_left, start=90, end=270, fill=(255, 255, 255, 230), width=thickness)
    # Right arc
    bbox_right = [center_x, forehead_y, center_x + width, forehead_y + height]
    draw.arc(bbox_right, start=-90, end=90, fill=(255, 255, 255, 230), width=thickness)
    # Central line
    draw.line([(center_x, forehead_y + height // 3), (center_x, forehead_y + height)], fill=(255, 255, 255, 230), width=thickness)


def draw_garland(overlay: Image.Image, face_bbox: BoundingBox) -> None:
    draw = ImageDraw.Draw(overlay, 'RGBA')
    x, y, w, h = face_bbox
    center_x = x + w // 2
    base_y = y + int(0.95 * h)

    radius_x = int(0.75 * w)
    radius_y = int(0.9 * h)

    beads = 28
    colors = [
        (255, 69, 0, 220),
        (255, 215, 0, 220),
        (60, 179, 113, 220),
        (30, 144, 255, 220),
        (186, 85, 211, 220),
    ]

    for i in range(beads):
        t = i / (beads - 1)
        angle = math.pi * (0.15 + 0.70 * t)
        px = int(center_x + radius_x * math.cos(angle))
        py = int(base_y + radius_y * math.sin(angle))
        r = max(4, int(w * 0.035))
        color = colors[i % len(colors)]
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=(255, 255, 255, 200))


def draw_flute(overlay: Image.Image, face_bbox: BoundingBox, image_size: Tuple[int, int]) -> None:
    W, H = image_size
    x, y, w, h = face_bbox

    length = int(1.6 * w)
    thickness = max(6, w // 18)

    # Position across the chest diagonally
    start_x = int(x - 0.2 * w)
    start_y = y + int(1.2 * h)
    end_x = start_x + int(length * math.cos(math.radians(-18)))
    end_y = start_y + int(length * math.sin(math.radians(-18)))

    flute_overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(flute_overlay, 'RGBA')

    d.line([(start_x, start_y), (end_x, end_y)], fill=(210, 180, 140, 235), width=thickness)

    holes = 6
    for i in range(holes):
        t = (i + 1) / (holes + 1)
        hx = int(start_x + t * (end_x - start_x))
        hy = int(start_y + t * (end_y - start_y))
        r = max(2, thickness // 4)
        d.ellipse([hx - r, hy - r, hx + r, hy + r], fill=(90, 60, 20, 230))

    overlay.alpha_composite(flute_overlay)


def draw_matki(overlay: Image.Image, anchor: Tuple[int, int], scale: int = 120) -> None:
    draw = ImageDraw.Draw(overlay, 'RGBA')
    ax, ay = anchor
    pot_w = int(1.2 * scale)
    pot_h = int(1.0 * scale)
    body_rect = [ax, ay, ax + pot_w, ay + pot_h]

    clay = (205, 133, 63, 235)
    dark = (139, 69, 19, 235)
    cream = (255, 250, 240, 235)

    draw.ellipse(body_rect, fill=clay, outline=dark, width=3)

    neck_h = int(0.33 * pot_h)
    neck_rect = [ax + int(0.18 * pot_w), ay - neck_h, ax + int(0.82 * pot_w), ay + int(0.05 * pot_h)]
    draw.rounded_rectangle(neck_rect, radius=neck_h // 2, fill=clay, outline=dark, width=3)

    # Butter spilling
    drip_w = int(0.64 * pot_w)
    drip_h = int(0.20 * pot_h)
    drip_rect = [ax + (pot_w - drip_w) // 2, ay - int(0.13 * pot_h), ax + (pot_w + drip_w) // 2, ay + int(0.07 * pot_h)]
    draw.ellipse(drip_rect, fill=cream, outline=None)


def compose_krishna(image: Image.Image, face_bbox: Optional[BoundingBox]) -> Image.Image:
    W, H = image.size
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))

    if face_bbox is None:
        # Fallback: center-ish box
        cx, cy = W // 2, H // 3
        estimate_w = W // 4
        estimate_h = H // 4
        face_bbox = (cx - estimate_w // 2, cy - estimate_h // 2, estimate_w, estimate_h)

    draw_crown_and_feather(overlay, face_bbox)
    draw_tilak(ImageDraw.Draw(overlay, 'RGBA'), face_bbox)
    draw_garland(overlay, face_bbox)
    draw_flute(overlay, face_bbox, (W, H))

    # Place pot near lower right corner
    margin = max(16, W // 40)
    draw_matki(overlay, (W - margin - int(W * 0.20), H - margin - int(W * 0.20)), scale=max(80, W // 8))

    composed = image.convert('RGBA')
    composed.alpha_composite(overlay)
    return composed


def main() -> None:
    parser = argparse.ArgumentParser(description='Create a Krishna-themed composite while keeping the face unchanged.')
    parser.add_argument('input', help='Path to input photo')
    parser.add_argument('-o', '--output', default='output_krishna.png', help='Path to output image (PNG recommended)')
    parser.add_argument('--no-face-detect', action='store_true', help='Disable face detection and use image center heuristic')
    args = parser.parse_args()

    try:
        pil_img = Image.open(args.input).convert('RGB')
    except Exception:
        raise SystemExit(f'Could not read input image: {args.input}')

    face_bbox = None
    if not args.no_face_detect and HAS_CV2:
        try:
            rgb = np.array(pil_img)
            bgr = rgb[:, :, ::-1].copy()
            face_bbox = detect_face_bbox(bgr)
        except Exception:
            face_bbox = None
    elif not args.no_face_detect and not HAS_CV2:
        print('OpenCV not available; skipping face detection.')

    result = compose_krishna(pil_img, face_bbox)
    result.save(args.output)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()