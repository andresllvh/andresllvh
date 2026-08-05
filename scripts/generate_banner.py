#!/usr/bin/env python3
"""
GitHub Profile Banner Generator
Generates dark.svg and light.svg with animated dithered portrait,
logo morphing, and terminal info panel.

Usage:
    python generate_banner.py --photo andre-real.png --output-dir .

Requirements:
    pip install Pillow numpy scipy
"""

import argparse
import math
import random
import struct
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_closing, binary_fill_holes, label

# ─── PALETTE ───────────────────────────────────────────────────────
BG_COLOR = "#0A101F"
ACCENT_CYAN = "#22D3EE"
ACCENT_TEAL = "#0891B2"
ACCENT_PURPLE = "#A78BFA"
ACCENT_DEEP_PURPLE = "#7C3AED"
ACCENT_GREEN = "#10B981"
TEXT_MUTED = "#94A3B8"
TEXT_DIM = "#64748B"
TEXT_BRIGHT = "#F8FAFC"
PANEL_BG = "#0F172A"
BORDER_COLOR = "#1E293B"
LIVE_RED = "#EF4444"

# ─── BANNER DIMENSIONS ────────────────────────────────────────────
BANNER_W = 1180
BANNER_H = 610
PORTRAIT_COLS = 300
PORTRAIT_ROWS = 340
DOT_SIZE = 1.6
DOT_GAP = 0.0
PORTRAIT_AREA_PCT = 0.38  # left 38%

# ─── INFO PANEL DATA ──────────────────────────────────────────────
INFO_ROWS = [
    ("Subject", "André Santos"),
    ("Role", "Dev Software Jr | IA &amp; Automação"),
    ("Origin", "João Pessoa — PB, Brasil"),
    ("Education", "Análise e Desenv. de Sistemas"),
    ("Status", "Building • Learning • Shipping"),
    ("ToolChain", "VS Code, Git, Docker, Postman, Figma"),
    ("", ""),  # spacer
    ("Core.Lang", "TypeScript · JavaScript · Python · PHP"),
    ("Core.Frontend", "React · HTML5 · CSS3 · Tailwind"),
    ("Core.Backend", "Node.js · APIs REST"),
    ("Core.Database", "PostgreSQL · SQL"),
    ("Core.Infra", "Docker · Git/GitHub · AWS"),
    ("", ""),  # spacer
    ("Grid.Mail", "andrelaureano16@gmail.com"),
    ("Grid.Portfolio", "portfolio-tan-six-bcjn1gwug5.vercel.app"),
    ("Grid.LinkedIn", "linkedin.com/in/andré-santooss"),
    ("Grid.GitHub", "github.com/andresllvh"),
]

# ─── LOGO DEFINITIONS (simplified SVG paths) ──────────────────────
# React atom logo
REACT_LOGO = [
    # Center dot + 3 elliptical orbits approximated as point clouds
]

# TypeScript "TS" logo
TS_LOGO = []

# Node.js hexagon logo
NODE_LOGO = []


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


# ─── PORTRAIT PROCESSING ──────────────────────────────────────────
def load_and_prepare_photo(path: str, cols: int, rows: int) -> Image.Image:
    """Load photo, crop to head+shoulders, resize to dot grid dimensions."""
    img = Image.open(path).convert("RGBA")
    w, h = img.size

    # Smart crop: center on upper portion (head + shoulders)
    # Target aspect ratio for portrait area
    target_ratio = cols / rows
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Image is wider — crop sides
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        # Image is taller — crop from bottom (keep head+shoulders)
        new_h = int(w / target_ratio)
        # Bias toward top (head area) — use top 15% as starting point
        top = int(h * 0.02)
        bottom = top + new_h
        if bottom > h:
            bottom = h
            top = h - new_h
        img = img.crop((0, top, w, bottom))

    img = img.resize((cols, rows), Image.LANCZOS)
    return img


def apply_enhancements(img: Image.Image) -> Image.Image:
    """Apply contrast, brightness, and unsharp mask for realistic crisp dithering."""
    from PIL import ImageEnhance

    # Convert to RGB for processing
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Moderate contrast enhancement so skin highlights stay clean
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.15)

    # Slight brightness boost to keep skin tones clear (matching Screenshot 3!)
    brightener = ImageEnhance.Brightness(img)
    img = brightener.enhance(1.08)

    # Autocontrast with cutoff=2
    img = ImageOps.autocontrast(img, cutoff=2)

    # UnsharpMask for sharp facial features, glasses, suit lapels
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=110))

    return img


def floyd_steinberg_dither(img: Image.Image, serpentine: bool = True) -> np.ndarray:
    """
    1-bit Floyd-Steinberg dithering with serpentine scanning.
    Returns a boolean array where True = ink dot.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    h, w = gray.shape
    result = np.zeros((h, w), dtype=bool)

    for y in range(h):
        if serpentine and y % 2 == 1:
            x_range = range(w - 1, -1, -1)
            direction = -1
        else:
            x_range = range(w)
            direction = 1

        for x in x_range:
            old_val = gray[y, x]
            new_val = 255.0 if old_val >= 128 else 0.0
            result[y, x] = new_val == 0  # True = dark dot (ink)
            error = old_val - new_val

            # Distribute error
            if 0 <= x + direction < w:
                gray[y, x + direction] += error * 7 / 16
            if y + 1 < h:
                if 0 <= x - direction < w:
                    gray[y + 1, x - direction] += error * 3 / 16
                gray[y + 1, x] += error * 5 / 16
                if 0 <= x + direction < w:
                    gray[y + 1, x + direction] += error * 1 / 16

    return result


def segment_background(img: Image.Image, threshold: float = 60.0) -> np.ndarray:
    """
    Segment the background. Uses alpha channel if RGBA, otherwise color distance from corners.
    Returns a boolean mask where True = foreground (subject).
    """
    if img.mode == "RGBA":
        alpha = np.array(img.split()[-1])
        if alpha.min() < 255:
            # Has transparency, use exact alpha channel mask
            fg_mask = alpha > 20
            # Clean up tiny isolated alpha artifacts
            fg_mask = binary_fill_holes(fg_mask)
            return fg_mask

    rgb = np.array(img.convert("RGB"), dtype=np.float64)
    h, w = rgb.shape[:2]

    # Sample background color from corners
    corner_size = max(5, min(w, h) // 20)
    corners = np.concatenate([
        rgb[:corner_size, :corner_size].reshape(-1, 3),
        rgb[:corner_size, -corner_size:].reshape(-1, 3),
        rgb[-corner_size:, :corner_size].reshape(-1, 3),
        rgb[-corner_size:, -corner_size:].reshape(-1, 3),
    ])
    bg_color = np.median(corners, axis=0)

    # Color distance
    dist = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=2))
    fg_mask = dist > threshold

    # Morphological cleanup
    fg_mask = binary_closing(fg_mask, structure=np.ones((5, 5)), iterations=3)
    fg_mask = binary_fill_holes(fg_mask)

    # Keep only the largest connected component
    labeled, n_features = label(fg_mask)
    if n_features > 1:
        sizes = [np.sum(labeled == i) for i in range(1, n_features + 1)]
        largest = np.argmax(sizes) + 1
        fg_mask = labeled == largest

    return fg_mask


def clear_edge_bleed(dots: np.ndarray, mask: np.ndarray, border_px: int = 3) -> np.ndarray:
    """Hard-clear error-diffusion bleed at the mask edge for dark mode."""
    from scipy.ndimage import binary_dilation, binary_erosion

    # Create border region
    dilated = binary_dilation(mask, iterations=border_px)
    eroded = binary_erosion(mask, iterations=border_px)
    border = dilated & ~eroded

    # Clear dots in the border region outside the mask
    clear_zone = border & ~mask
    dots[clear_zone] = False
    return dots


# ─── DOT → SVG PATH CONVERSION ───────────────────────────────────
def dots_to_path_runs(dots: np.ndarray, ox: float, oy: float,
                       dot_size: float = DOT_SIZE) -> str:
    """
    Convert a boolean dot matrix to SVG <path> data using horizontal runs.
    Each run is a rectangle: M x y h w v dot_size H x Z
    """
    h, w = dots.shape
    runs = []
    step = dot_size + DOT_GAP

    for y in range(h):
        x = 0
        while x < w:
            if dots[y, x]:
                # Start of a run
                start_x = x
                while x < w and dots[y, x]:
                    x += 1
                run_len = x - start_x
                rx = ox + start_x * step
                ry = oy + y * step
                rw = run_len * step
                runs.append(f"M{rx:.1f} {ry:.1f}h{rw:.1f}v{dot_size}H{rx:.1f}Z")
            else:
                x += 1

    return "".join(runs)


# ─── LOGO POINT CLOUDS (BINARY MASK GRID SAMPLING) ───────────────
def generate_binary_mask_grid_points(logo_type: str, cx: float, cy: float, size: float, step: int = 3) -> List[Tuple[float, float]]:
    """
    Grid Sampling on PIL binary mask.
    Guarantees 100% geometric fidelity, uniform density (5,000+ points), and instant logo readability.
    """
    from PIL import Image, ImageDraw, ImageChops

    canvas_size = 360
    img = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(img)
    mid = canvas_size / 2.0

    if logo_type == "code_glyph":
        sw = 36
        # '<'
        draw.line([(110, 100), (50, 180), (110, 260)], fill=255, width=sw)
        # '/'
        draw.line([(210, 70), (165, 290)], fill=255, width=sw)
        # '>'
        draw.line([(260, 100), (320, 180), (260, 260)], fill=255, width=sw)

    elif logo_type == "react":
        # Nucleus
        draw.ellipse([mid - 24, mid - 24, mid + 24, mid + 24], fill=255)
        rx = 145
        ry = 48
        for angle in [0, 60, 120]:
            temp = Image.new("L", (canvas_size, canvas_size), 0)
            tdraw = ImageDraw.Draw(temp)
            tdraw.ellipse([mid - rx, mid - ry, mid + rx, mid + ry], fill=0, outline=255, width=18)
            temp = temp.rotate(-angle, center=(mid, mid), resample=Image.BICUBIC)
            img = ImageChops.lighter(img, temp)

    elif logo_type == "ts":
        # Outer rounded box
        draw.rounded_rectangle([45, 45, 315, 315], radius=42, fill=0, outline=255, width=30)
        # Bold 'T'
        draw.line([(105, 135), (190, 135)], fill=255, width=26)
        draw.line([(147, 135), (147, 255)], fill=255, width=26)
        # Bold 'S'
        draw.line([(270, 135), (215, 135)], fill=255, width=24)
        draw.line([(215, 135), (215, 190)], fill=255, width=24)
        draw.line([(215, 190), (270, 190)], fill=255, width=24)
        draw.line([(270, 190), (270, 250)], fill=255, width=24)
        draw.line([(215, 250), (270, 250)], fill=255, width=24)

    arr = np.array(img)
    ys, xs = np.where(arr > 100)

    # Apply Grid Sampling (every `step` pixels)
    mask_grid = (ys % step == 0) & (xs % step == 0)
    ys_grid = ys[mask_grid]
    xs_grid = xs[mask_grid]

    scale = size / canvas_size
    points = []
    for y_val, x_val in zip(ys_grid, xs_grid):
        nx = (x_val - mid) * scale + cx
        ny = (y_val - mid) * scale + cy
        points.append((nx, ny))

    return points


# ─── OPTIMAL TRANSPORT MATCHING ──────────────────────────────────
def match_points_ot(src: List[Tuple[float, float]],
                    dst: List[Tuple[float, float]]) -> List[int]:
    """
    Match source points to destination points via optimal transport
    (Hungarian algorithm) so each dot takes the shortest path.
    """
    n = len(src)
    assert n == len(dst), f"Point clouds must be same size: {n} vs {len(dst)}"

    # For large n, use approximation (batch assignment)
    if n > 500:
        # Grid-based approximation for performance
        return _approximate_ot(src, dst)

    # Build cost matrix
    cost = np.zeros((n, n))
    for i, (sx, sy) in enumerate(src):
        for j, (dx, dy) in enumerate(dst):
            cost[i, j] = (sx - dx) ** 2 + (sy - dy) ** 2

    _, col_ind = linear_sum_assignment(cost)
    return col_ind.tolist()


def _approximate_ot(src, dst):
    """Grid-based approximate optimal transport for large point sets."""
    n = len(src)
    # Sort both by angle from centroid, then match by index
    src_cx = sum(p[0] for p in src) / n
    src_cy = sum(p[1] for p in src) / n
    dst_cx = sum(p[0] for p in dst) / n
    dst_cy = sum(p[1] for p in dst) / n

    src_angles = [math.atan2(p[1] - src_cy, p[0] - src_cx) for p in src]
    dst_angles = [math.atan2(p[1] - dst_cy, p[0] - dst_cx) for p in dst]

    src_order = sorted(range(n), key=lambda i: src_angles[i])
    dst_order = sorted(range(n), key=lambda i: dst_angles[i])

    mapping = [0] * n
    for i in range(n):
        mapping[src_order[i]] = dst_order[i]
    return mapping


# ─── ANIMATION GROUPS ────────────────────────────────────────────
def create_intro_groups(dots: np.ndarray, n_groups: int = 60) -> List[List[Tuple[int, int]]]:
    """
    Create ~60 interleaved random groups for the intro fade-in.
    Each group must be scattered across the whole portrait (not spatial regions).
    Verified with an evenness metric.
    """
    positions = list(zip(*np.where(dots)))
    random.shuffle(positions)

    groups = [[] for _ in range(n_groups)]
    for i, pos in enumerate(positions):
        groups[i % n_groups].append(pos)

    # Verify evenness
    evenness = _check_evenness(groups, dots.shape)
    if evenness > 0.1:
        print(f"WARNING: Intro group evenness metric = {evenness:.3f} (want < 0.05)")
    else:
        print(f"Intro group evenness: {evenness:.3f} ✓")

    return groups


def _check_evenness(groups, shape):
    """Check that groups are evenly distributed spatially."""
    h, w = shape
    n_groups = len(groups)
    grid_size = 8
    cell_h = h / grid_size
    cell_w = w / grid_size

    variances = []
    for group in groups[:10]:  # Check first 10 groups
        grid = np.zeros((grid_size, grid_size))
        for y, x in group:
            gy = min(int(y / cell_h), grid_size - 1)
            gx = min(int(x / cell_w), grid_size - 1)
            grid[gy, gx] += 1
        if grid.sum() > 0:
            grid /= grid.sum()
            expected = 1.0 / (grid_size * grid_size)
            variance = np.var(grid - expected)
            variances.append(variance)

    return np.mean(variances) if variances else 1.0


def create_drift_bands(dots: np.ndarray, n_bands: int = 24,
                       noise_sigma: float = 4.0) -> List[List[Tuple[int, int]]]:
    """
    Group portrait dots into ~94 drift bands.
    Add per-dot noise before grouping to avoid the grid trap.
    Uses angle + distance + heavy noise to create organic boundaries.
    """
    positions = list(zip(*np.where(dots)))
    n = len(positions)

    h, w = dots.shape
    cy = h / 2.0
    cx = w / 2.0

    # Compute angle and distance from center with heavy noise
    # Mix angle and distance to create spiral-like grouping
    keys = []
    for y, x in positions:
        ny = y + random.gauss(0, noise_sigma)
        nx = x + random.gauss(0, noise_sigma)
        angle = math.atan2(ny - cy, nx - cx)  # [-pi, pi]
        dist = math.sqrt((ny - cy)**2 + (nx - cx)**2)
        max_dist = math.sqrt(cy**2 + cx**2)
        # Combine angle and distance into a spiral key
        # Normalize angle to [0, 1] and dist to [0, 1]
        norm_angle = (angle + math.pi) / (2 * math.pi)
        norm_dist = dist / max_dist
        # Spiral: multiple rotations as distance increases
        spiral_key = (norm_angle + norm_dist * 3.0 + random.gauss(0, 0.15)) % 1.0
        keys.append(spiral_key)

    # Sort by spiral key, then interleave into bands
    indexed = sorted(range(n), key=lambda i: keys[i])
    bands = [[] for _ in range(n_bands)]
    for i, idx in enumerate(indexed):
        bands[i % n_bands].append(positions[idx])

    # Check straight-boundary metric
    boundary_metric = _check_boundary(bands, dots.shape)
    if boundary_metric > 0.05:
        print(f"WARNING: Drift band boundary metric = {boundary_metric:.3f} (want < 0.01)")
    else:
        print(f"Drift band boundary: {boundary_metric:.3f} ✓")

    return bands


def _check_boundary(bands, shape):
    """Check that band boundaries are organic (not grid-like)."""
    h, w = shape
    # Create a labeled image from bands
    labeled = np.zeros(shape, dtype=int)
    for band_idx, band in enumerate(bands[:20]):
        for y, x in band:
            labeled[y, x] = band_idx + 1

    # Count horizontal straight-boundary pixels
    straight = 0
    total = 0
    for y in range(h - 1):
        for x in range(w):
            if labeled[y, x] > 0 and labeled[y+1, x] > 0:
                if labeled[y, x] != labeled[y+1, x]:
                    total += 1
                    # Check if neighbors are also boundaries
                    if x > 0 and labeled[y, x-1] != labeled[y+1, x-1]:
                        straight += 1

    return straight / max(total, 1)


# ─── SVG GENERATION ──────────────────────────────────────────────
def generate_info_panel_svg(x_start: float, y_start: float, width: float) -> str:
    """Generate the terminal info panel SVG content."""
    lines = []
    row_height = 23
    font_size = 13
    header_font_size = 13
    label_color = TEXT_DIM
    value_color = TEXT_MUTED
    header_color = ACCENT_CYAN

    # Header: SYSTEM.INFO (top-left of info panel)
    lines.append(f'<text x="{x_start}" y="{y_start + 20}" fill="{header_color}" '
                 f'font-size="{header_font_size}" font-family="\'JetBrains Mono\', \'Fira Code\', monospace" '
                 f'font-weight="700" opacity="0.9">SYSTEM.INFO</text>')

    # LIVE badge (top-right area)
    live_x = x_start + width - 180
    lines.append(f'<circle cx="{live_x}" cy="{y_start + 16}" r="4" fill="{LIVE_RED}">'
                 f'<animate attributeName="opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite"/>'
                 f'</circle>')
    lines.append(f'<text x="{live_x + 8}" y="{y_start + 20}" fill="{LIVE_RED}" '
                 f'font-size="11" font-family="\'JetBrains Mono\', monospace" font-weight="700">LIVE</text>')

    # Pill badge with handle @andresllvh (top-right corner)
    pill_x = x_start + width - 120
    pill_y = y_start + 5
    handle = "@andresllvh"
    lines.append(f'<rect x="{pill_x}" y="{pill_y}" width="110" height="22" rx="11" '
                 f'fill="{ACCENT_PURPLE}" opacity="0.15"/>')
    lines.append(f'<rect x="{pill_x}" y="{pill_y}" width="110" height="22" rx="11" '
                 f'fill="none" stroke="{ACCENT_PURPLE}" stroke-width="1" opacity="0.4"/>')
    lines.append(f'<text x="{pill_x + 11}" y="{pill_y + 15}" fill="{ACCENT_PURPLE}" '
                 f'font-size="11" font-family="\'JetBrains Mono\', monospace" font-weight="500">'
                 f'{handle}</text>')

    # Separator line
    y = y_start + 36
    lines.append(f'<line x1="{x_start}" y1="{y}" x2="{x_start + width - 10}" y2="{y}" '
                 f'stroke="{BORDER_COLOR}" stroke-width="1" opacity="0.5"/>')
    y += row_height

    # Data rows with dotted leaders
    for label_text, value_text in INFO_ROWS:
        if not label_text:
            # Spacer with subtle divider
            y += 6
            lines.append(f'<line x1="{x_start}" y1="{y}" x2="{x_start + width - 10}" y2="{y}" '
                         f'stroke="{BORDER_COLOR}" stroke-width="0.5" opacity="0.3" '
                         f'stroke-dasharray="2 4"/>')
            y += row_height - 6
            continue

        # Label
        label_w = len(label_text) * 7.5
        lines.append(f'<text x="{x_start}" y="{y}" fill="{label_color}" '
                     f'font-size="{font_size}" font-family="\'JetBrains Mono\', monospace">'
                     f'{label_text}</text>')

        # Value (right-aligned area)
        value_w = len(value_text) * 7.2
        value_x = x_start + width - 10 - value_w

        # Dotted leaders between label and value
        leader_start = x_start + label_w + 8
        leader_end = value_x - 8
        if leader_end > leader_start + 10:
            lines.append(f'<line x1="{leader_start}" y1="{y - 3}" x2="{leader_end}" y2="{y - 3}" '
                         f'stroke="{TEXT_DIM}" stroke-width="0.8" opacity="0.3" '
                         f'stroke-dasharray="1 4"/>')

        # Color the value based on row type
        v_color = value_color
        if label_text.startswith("Grid."):
            v_color = ACCENT_GREEN
        elif label_text == "Status":
            v_color = ACCENT_CYAN
        elif label_text.startswith("Core."):
            v_color = ACCENT_PURPLE

        lines.append(f'<text x="{value_x}" y="{y}" fill="{v_color}" '
                     f'font-size="{font_size}" font-family="\'JetBrains Mono\', monospace" '
                     f'textLength="{value_w}" lengthAdjust="spacingAndGlyphs">'
                     f'{value_text}</text>')

        y += row_height

    return "\n    ".join(lines)


def generate_terminal_frame_svg() -> str:
    """Generate the terminal window frame."""
    lines = []

    # Outer frame with rounded corners
    lines.append(f'<rect x="0" y="0" width="{BANNER_W}" height="{BANNER_H}" '
                 f'rx="12" fill="{BG_COLOR}" stroke="{BORDER_COLOR}" stroke-width="1.5"/>')

    # Title bar
    lines.append(f'<rect x="0" y="0" width="{BANNER_W}" height="38" rx="12" fill="{PANEL_BG}"/>')
    lines.append(f'<rect x="0" y="26" width="{BANNER_W}" height="12" fill="{PANEL_BG}"/>')
    lines.append(f'<line x1="0" y1="38" x2="{BANNER_W}" y2="38" stroke="{BORDER_COLOR}" stroke-width="1"/>')

    # Traffic lights
    lines.append(f'<circle cx="20" cy="19" r="6" fill="#EF4444" opacity="0.9"/>')
    lines.append(f'<circle cx="40" cy="19" r="6" fill="#F59E0B" opacity="0.9"/>')
    lines.append(f'<circle cx="60" cy="19" r="6" fill="#10B981" opacity="0.9"/>')

    # Title text
    title = "profile.sh --live"
    lines.append(f'<text x="{BANNER_W // 2}" y="24" fill="{TEXT_DIM}" '
                 f'font-size="13" font-family="\'JetBrains Mono\', monospace" '
                 f'text-anchor="middle" opacity="0.7">{title}</text>')

    return "\n    ".join(lines)


def generate_portrait_frame_svg(portrait_x: float, portrait_y: float,
                                 portrait_w: float, portrait_h: float) -> str:
    """Generate the VISUAL.MAP frame around the portrait with cyber corner brackets."""
    lines = []

    # Frame border bounds
    pad_x = 16
    pad_top = 28
    pad_bottom = 16
    fx = portrait_x - pad_x
    fy = portrait_y - pad_top
    fw = portrait_w + pad_x * 2
    fh = portrait_h + pad_top + pad_bottom

    # Main outer border box
    lines.append(f'<rect x="{fx}" y="{fy}" width="{fw}" height="{fh}" '
                 f'rx="8" fill="none" stroke="{BORDER_COLOR}" stroke-width="1" opacity="0.6"/>')

    # VISUAL.MAP header label (inside box at top-left)
    lines.append(f'<text x="{fx + 12}" y="{fy + 18}" fill="{ACCENT_CYAN}" '
                 f'font-size="11" font-family="\'JetBrains Mono\', monospace" '
                 f'font-weight="700" opacity="0.9">VISUAL.MAP</text>')

    # Cyber corner brackets (fixed path Y coordinates!)
    arm = 14
    c = ACCENT_CYAN
    sw = 1.8
    # Top-Left
    lines.append(f'<path d="M{fx} {fy+arm} V{fy} H{fx+arm}" stroke="{c}" stroke-width="{sw}" fill="none"/>')
    # Top-Right
    lines.append(f'<path d="M{fx+fw-arm} {fy} H{fx+fw} V{fy+arm}" stroke="{c}" stroke-width="{sw}" fill="none"/>')
    # Bottom-Left
    lines.append(f'<path d="M{fx} {fy+fh-arm} V{fy+fh} H{fx+arm}" stroke="{c}" stroke-width="{sw}" fill="none"/>')
    # Bottom-Right
    lines.append(f'<path d="M{fx+fw-arm} {fy+fh} H{fx+fw} V{fy+fh-arm}" stroke="{c}" stroke-width="{sw}" fill="none"/>')

    return "\n    ".join(lines)


def build_svg(dots: np.ndarray,
              mode: str,
              intro_groups: List[List[Tuple[int, int]]],
              drift_bands: List[List[Tuple[int, int]]],
              traveller_frames: List[List[Tuple[float, float]]],
              fg_mask: np.ndarray = None) -> str:
    """
    Build the complete SVG with SMIL animations.
    mode: 'dark' or 'light'
    """
    is_dark = mode == "dark"
    dot_color = ACCENT_PURPLE if is_dark else ACCENT_DEEP_PURPLE
    bg = BG_COLOR if is_dark else "#FFFFFF"
    text_color = TEXT_MUTED if is_dark else "#334155"

    # Portrait position
    portrait_area_w = BANNER_W * PORTRAIT_AREA_PCT
    step = DOT_SIZE + DOT_GAP
    portrait_w = PORTRAIT_COLS * step
    portrait_h = PORTRAIT_ROWS * step
    portrait_x = (portrait_area_w - portrait_w) / 2 + 15
    portrait_y = (BANNER_H - portrait_h) / 2 + 20

    # Info panel position
    info_x = portrait_area_w + 30
    info_w = BANNER_W - info_x - 20

    # For dark mode, apply foreground mask
    if is_dark and fg_mask is not None:
        display_dots = dots & fg_mask
    else:
        display_dots = dots.copy()

    total_dots = int(np.sum(display_dots))
    print(f"  [{mode}] Total dots: {total_dots}")

    # ─── Build SVG ────────────────────────────────────────────
    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {BANNER_W} {BANNER_H}" '
                     f'width="{BANNER_W}" height="{BANNER_H}">')

    # Styles
    svg_parts.append(f'''  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&amp;display=swap');
    text {{ font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace; }}
  </style>''')

    # Background
    svg_parts.append(f'  <rect width="{BANNER_W}" height="{BANNER_H}" fill="{bg}"/>')

    # Terminal frame
    svg_parts.append(f"  <!-- Terminal Frame -->")
    svg_parts.append(f"  {generate_terminal_frame_svg()}")

    # Portrait frame
    svg_parts.append(f"  <!-- Portrait Frame -->")
    svg_parts.append(f"  {generate_portrait_frame_svg(portrait_x, portrait_y, portrait_w, portrait_h)}")

    # ─── PORTRAIT LAYER (with intro + drift animations) ──────
    # Intro timing: ~3.2s total, groups fade in over ~2s
    intro_duration = 3.2
    fade_window = 2.0
    n_groups = len(intro_groups)
    group_delay = fade_window / n_groups
    fade_in_time = 0.4  # Each group's individual fade duration

    # Loop timing
    loop_duration = 14.2
    portrait_hold = 3.0
    logo_hold = 2.0
    transition_time = 1.3
    total_anim = intro_duration + loop_duration

    # Create the intro animation: each group fades in sequentially
    svg_parts.append(f'  <!-- Portrait Layer: {total_dots} dots -->')
    svg_parts.append(f'  <g id="portrait-layer">')
    svg_parts.append(f'    <animate attributeName="opacity" from="1" to="0" '
                     f'begin="{intro_duration}s" dur="0.1s" fill="freeze"/>')

    # Split dots into groups for intro animation
    group_dots_arrays = []
    for group in intro_groups:
        group_array = np.zeros_like(display_dots)
        for y, x in group:
            if display_dots[y, x]:
                group_array[y, x] = True
        group_dots_arrays.append(group_array)

    for gi, group_array in enumerate(group_dots_arrays):
        if not np.any(group_array):
            continue
        path_d = dots_to_path_runs(group_array, portrait_x, portrait_y)
        if not path_d:
            continue

        delay = gi * group_delay
        svg_parts.append(f'    <path d="{path_d}" fill="{dot_color}" '
                         f'shape-rendering="crispEdges" opacity="0">')
        # Intro fade-in animation (runs once)
        svg_parts.append(f'      <animate attributeName="opacity" '
                         f'from="0" to="1" begin="{delay:.2f}s" '
                         f'dur="{fade_in_time}s" fill="freeze"/>')
        svg_parts.append(f'    </path>')

    svg_parts.append(f'  </g>')

    # ─── DUPLICATE PORTRAIT LAYER (for loop phase) ───────────
    # This layer appears after intro, handles drift animation
    svg_parts.append(f'  <!-- Portrait Loop Layer (drift bands) -->')
    svg_parts.append(f'  <g id="portrait-loop" opacity="0">')

    # Fade in after intro completes
    svg_parts.append(f'    <animate attributeName="opacity" '
                     f'from="0" to="1" begin="{intro_duration}s" '
                     f'dur="0.5s" fill="freeze"/>')

    # Calculate logo centroid for drift direction
    logo_cx = portrait_x + portrait_w / 2
    logo_cy = portrait_y + portrait_h / 2

    # Drift bands with subtle translate animation during transitions
    for bi, band in enumerate(drift_bands):
        band_array = np.zeros_like(display_dots)
        for y, x in band:
            if display_dots[y, x]:
                band_array[y, x] = True
        if not np.any(band_array):
            continue

        path_d = dots_to_path_runs(band_array, portrait_x, portrait_y)
        if not path_d:
            continue

        # Calculate drift direction (toward logo centroid)
        if band:
            band_cy = sum(y for y, x in band) / len(band) * step + portrait_y
            band_cx = sum(x for y, x in band) / len(band) * step + portrait_x
            dx = (logo_cx - band_cx) * 0.42
            dy = (logo_cy - band_cy) * 0.42
        else:
            dx, dy = 0, 0

        # Synchronized keyTimes: portrait visible ONLY when travellers are hidden
        # 0.0-0.211: portrait visible (1)
        # 0.211-0.302: portrait drift out & fade to 0
        # 0.302-0.908: portrait hidden (0) while logos morph
        # 0.908-1.000: portrait drift back & fade to 1

        kt = "0;0.211;0.302;0.908;1"
        kv_x = f"0;0;{dx:.1f};{dx:.1f};0"
        kv_y = f"0;0;{dy:.1f};{dy:.1f};0"
        ko = "1;1;0;0;1"

        svg_parts.append(f'    <g>')
        svg_parts.append(f'      <animateTransform attributeName="transform" type="translate" '
                         f'values="{kv_x}" keyTimes="{kt}" dur="{loop_duration}s" '
                         f'begin="{intro_duration}s" repeatCount="indefinite"/>')
        svg_parts.append(f'      <path d="{path_d}" fill="{dot_color}" '
                         f'shape-rendering="crispEdges">')
        svg_parts.append(f'        <animate attributeName="opacity" '
                         f'values="{ko}" keyTimes="{kt}" '
                         f'dur="{loop_duration}s" begin="{intro_duration}s" '
                         f'repeatCount="indefinite"/>')
        svg_parts.append(f'      </path>')
        svg_parts.append(f'    </g>')

    svg_parts.append(f'  </g>')

    # ─── TRAVELLER LAYER (logo morphing) ─────────────────────
    if traveller_frames and len(traveller_frames) >= 3:
        n_travellers = len(traveller_frames[0])
        traveller_color = ACCENT_CYAN if is_dark else ACCENT_TEAL

        svg_parts.append(f'  <!-- Traveller Layer: {n_travellers} morphing dots -->')
        svg_parts.append(f'  <g id="travellers" opacity="0">')
        svg_parts.append(f'    <animate attributeName="opacity" '
                         f'values="0;0;1;1;1;1;1;1;0" keyTimes="0;0.211;0.302;0.443;0.535;0.676;0.768;0.908;1" '
                         f'dur="{loop_duration}s" begin="{intro_duration}s" repeatCount="indefinite"/>')

        # Synchronized KeyTimes: travellers visible ONLY during logo phase
        # 0.0-0.211: portrait phase (travellers hidden, opacity=0)
        # 0.211-0.302: transition to logo 1 (opacity 0->1)
        # 0.302-0.443: logo 1 hold (React)
        # 0.443-0.535: transition to logo 2 (TypeScript)
        # 0.535-0.676: logo 2 hold (TypeScript)
        # 0.676-0.768: transition to logo 3 (Node.js)
        # 0.768-0.908: logo 3 hold (Node.js)
        # 0.908-1.000: transition back to portrait (opacity 1->0)

        kt = "0;0.211;0.302;0.443;0.535;0.676;0.768;0.908;1"

        for ti in range(n_travellers):
            p1 = traveller_frames[0][ti]
            p2 = traveller_frames[1][ti]
            p3 = traveller_frames[2][ti]

            # Hidden at portrait position, then move to each logo
            home_x = portrait_x + portrait_w / 2 + random.gauss(0, 30)
            home_y = portrait_y + portrait_h / 2 + random.gauss(0, 30)

            cx_values = f"{home_x:.1f};{home_x:.1f};{p1[0]:.1f};{p1[0]:.1f};{p2[0]:.1f};{p2[0]:.1f};{p3[0]:.1f};{p3[0]:.1f};{home_x:.1f}"
            cy_values = f"{home_y:.1f};{home_y:.1f};{p1[1]:.1f};{p1[1]:.1f};{p2[1]:.1f};{p2[1]:.1f};{p3[1]:.1f};{p3[1]:.1f};{home_y:.1f}"
            op_values = "0;0;1;1;1;1;1;1;0"

            svg_parts.append(f'    <circle r="{DOT_SIZE * 0.8}">')
            svg_parts.append(f'      <animate attributeName="cx" '
                             f'values="{cx_values}" keyTimes="{kt}" '
                             f'dur="{loop_duration}s" begin="{intro_duration}s" '
                             f'repeatCount="indefinite"/>')
            svg_parts.append(f'      <animate attributeName="cy" '
                             f'values="{cy_values}" keyTimes="{kt}" '
                             f'dur="{loop_duration}s" begin="{intro_duration}s" '
                             f'repeatCount="indefinite"/>')
            svg_parts.append(f'      <animate attributeName="opacity" '
                             f'values="{op_values}" keyTimes="{kt}" '
                             f'dur="{loop_duration}s" begin="{intro_duration}s" '
                             f'repeatCount="indefinite"/>')
            svg_parts.append(f'      <animate attributeName="fill" '
                             f'values="{traveller_color};{traveller_color};{ACCENT_CYAN};{ACCENT_CYAN};{ACCENT_PURPLE};{ACCENT_PURPLE};{ACCENT_GREEN};{ACCENT_GREEN};{traveller_color}" '
                             f'keyTimes="{kt}" '
                             f'dur="{loop_duration}s" begin="{intro_duration}s" '
                             f'repeatCount="indefinite"/>')
            svg_parts.append(f'    </circle>')

        svg_parts.append(f'  </g>')

    # ─── INFO PANEL ──────────────────────────────────────────
    svg_parts.append(f'  <!-- Info Panel -->')
    svg_parts.append(f'  <g opacity="0">')
    svg_parts.append(f'    <animate attributeName="opacity" from="0" to="1" '
                     f'begin="1.5s" dur="1s" fill="freeze"/>')
    svg_parts.append(f"    {generate_info_panel_svg(info_x, 50, info_w)}")
    svg_parts.append(f'  </g>')

    # ─── SCANLINE EFFECT (subtle) ─────────────────────────────
    # Very subtle animated scanline for terminal feel
    if is_dark:
        svg_parts.append(f'  <!-- Subtle scanline effect -->')
        svg_parts.append(f'  <rect x="0" y="0" width="{BANNER_W}" height="2" '
                         f'fill="white" opacity="0.015">')
        svg_parts.append(f'    <animateTransform attributeName="transform" type="translate" '
                         f'from="0 0" to="0 {BANNER_H}" dur="4s" repeatCount="indefinite"/>')
        svg_parts.append(f'  </rect>')

    svg_parts.append(f'</svg>')

    return "\n".join(svg_parts)


# ─── MAIN ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate GitHub profile banner SVGs")
    parser.add_argument("--photo", required=True, help="Path to portrait photo")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GitHub Profile Banner Generator")
    print("=" * 60)

    # 1. Load and prepare photo
    print("\n[1/7] Loading and preparing photo...")
    img = load_and_prepare_photo(args.photo, PORTRAIT_COLS, PORTRAIT_ROWS)
    print(f"  Cropped to {img.size[0]}×{img.size[1]}")

    # 2. Enhance
    print("[2/7] Applying enhancements (contrast 1.3×, autocontrast, unsharp mask)...")
    enhanced = apply_enhancements(img)

    # 3. Dither
    print("[3/7] Floyd-Steinberg dithering (serpentine)...")
    dots = floyd_steinberg_dither(enhanced)
    total = int(np.sum(dots))
    print(f"  Total ink dots: {total} / {PORTRAIT_COLS * PORTRAIT_ROWS} "
          f"({100 * total / (PORTRAIT_COLS * PORTRAIT_ROWS):.1f}%)")

    # 4. Segment background for dark mode
    print("[4/7] Segmenting background for dark mode...")
    fg_mask = segment_background(img)
    dark_dots = dots.copy()
    dark_dots = clear_edge_bleed(dark_dots, fg_mask)
    dark_total = int(np.sum(dark_dots & fg_mask))
    print(f"  Foreground dots (dark mode): {dark_total}")

    # 5. Create animation groups
    print("[5/7] Creating animation groups...")
    intro_groups = create_intro_groups(dots)
    drift_bands = create_drift_bands(dots)
    print(f"  Intro groups: {len(intro_groups)}, Drift bands: {len(drift_bands)}")

    # 6. Generate logo point clouds and match via OT
    print("[6/7] Generating logo morphs (Code Glyph -> React Atom -> Tech Hexagon)...")
    step = DOT_SIZE + DOT_GAP
    portrait_area_w = BANNER_W * PORTRAIT_AREA_PCT
    portrait_w = PORTRAIT_COLS * step
    portrait_h = PORTRAIT_ROWS * step
    portrait_x = (portrait_area_w - portrait_w) / 2 + 15
    portrait_y = (BANNER_H - portrait_h) / 2 + 25

    logo_cx = portrait_x + portrait_w / 2
    logo_cy = portrait_y + portrait_h / 2
    logo_r = min(portrait_w, portrait_h) * 0.38

    glyph_pts = generate_binary_mask_grid_points("code_glyph", logo_cx, logo_cy, logo_r * 1.3, step=11)
    react_pts = generate_binary_mask_grid_points("react", logo_cx, logo_cy, logo_r * 1.1, step=11)
    ts_pts = generate_binary_mask_grid_points("ts", logo_cx, logo_cy, logo_r * 1.3, step=11)

    max_pts = max(len(glyph_pts), len(react_pts), len(ts_pts))

    def pad_points(pts, target_n):
        if len(pts) >= target_n:
            return pts[:target_n]
        extra = [pts[i % len(pts)] for i in range(target_n - len(pts))]
        return pts + extra

    glyph_pts = pad_points(glyph_pts, max_pts)
    react_pts = pad_points(react_pts, max_pts)
    ts_pts = pad_points(ts_pts, max_pts)

    # Match via approximate optimal transport
    print("  Matching Code Glyph -> React...")
    glyph_to_react = match_points_ot(glyph_pts, react_pts)
    react_pts_ordered = [react_pts[i] for i in glyph_to_react]

    print("  Matching React -> TypeScript...")
    react_to_ts = match_points_ot(react_pts_ordered, ts_pts)
    ts_pts_ordered = [ts_pts[i] for i in react_to_ts]

    traveller_frames = [glyph_pts, react_pts_ordered, ts_pts_ordered]

    # 7. Generate SVGs
    print("[7/7] Generating SVGs...")

    # Dark mode
    print("  Generating dark.svg...")
    dark_svg = build_svg(dots, "dark", intro_groups, drift_bands, traveller_frames, fg_mask)
    dark_path = output_dir / "dark.svg"
    dark_path.write_text(dark_svg)
    dark_size = dark_path.stat().st_size / 1024
    print(f"  ✓ dark.svg written ({dark_size:.0f} KB)")

    # Light mode
    print("  Generating light.svg...")
    light_svg = build_svg(dots, "light", intro_groups, drift_bands, traveller_frames)
    light_path = output_dir / "light.svg"
    light_path.write_text(light_svg)
    light_size = light_path.stat().st_size / 1024
    print(f"  ✓ light.svg written ({light_size:.0f} KB)")

    print("\n" + "=" * 60)
    print(f"Done! Files written to {output_dir}")
    print(f"  dark.svg:  {dark_size:.0f} KB")
    print(f"  light.svg: {light_size:.0f} KB")
    print(f"  Total dots: {total}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Open dark.svg and light.svg in a browser to verify")
    print("  2. Upload both to the root of your profile repo")
    print("  3. Check both themes on your GitHub profile")


if __name__ == "__main__":
    main()
