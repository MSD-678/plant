"""
utils/visualizer.py
===================
Advanced Bio-Optical Diagnostics & Lesion Visualization Engine.

Generates multi-spectral diagnostic visual overlays:
  1. Lesion Heatmap Overlay: Identifies necrotic/chlorotic pathogen clusters and
     draws glowing amber-crimson contour boundaries around infected plant tissue.
  2. Chlorophyll Bio-Vegetation Map: Computes the Excess Green (ExG) index to
     visualize chlorophyll photosynthetic vitality across healthy leaf tissue.
  3. Quantitative Infection Severity: Measures exact pixel-level diseased surface area.
"""

from typing import Tuple, Dict, Any
import cv2
import numpy as np


def generate_diagnostic_overlays(
    img_bgr: np.ndarray,
    target_width: int = 700
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """
    Generate lesion inspection overlay, chlorophyll bio-map, and compute infection severity.

    Args:
        img_bgr: Original uint8 BGR image.
        target_width: Dimension to scale for high-definition rendering.

    Returns:
        lesion_overlay (np.ndarray): BGR image with highlighted diseased spots and boundaries.
        exg_bio_map (np.ndarray): BGR image rendering chlorophyll vegetation gradient.
        severity_pct (float): Percentage of leaf blade affected by disease (0.0 to 100.0).
        analysis_meta (dict): Detailed diagnostic metadata.
    """
    h, w = img_bgr.shape[:2]
    disp_w = target_width
    disp_h = max(100, int(h * (disp_w / float(w))))
    img_disp = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)
    b = img_rgb[:, :, 2].astype(np.float64)
    hsv = cv2.cvtColor(img_disp, cv2.COLOR_BGR2HSV)

    # Excess Green Index (ExG): 2G - R - B
    exg = 2.0 * g - r - b

    # ── 1. Botanical Tissue Masks ───────────────────────────────────────────
    # Healthy green foliage
    green_mask = (hsv[:, :, 0] >= 26) & (hsv[:, :, 0] <= 92) & \
                 (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 25) & (exg > -5.0)

    # Necrotic blight / chlorotic halo / fungal lesions
    blight_mask = (hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 28) & \
                  (hsv[:, :, 1] >= 30) & (hsv[:, :, 2] >= 25) & \
                  (b < 155.0) & (r > b + 15.0)

    plant_combined = (green_mask | blight_mask).astype(np.uint8) * 255

    # ── 2. Isolate Coherent Leaf Blade (reject background) ──────────────────
    kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(plant_combined, cv2.MORPH_CLOSE, kernel_lg)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaf_mask = np.zeros((disp_h, disp_w), dtype=np.uint8)
    if cnts:
        # Take the primary leaf contours (greater than 5% of frame)
        frame_area = disp_h * disp_w
        for c in cnts:
            if cv2.contourArea(c) > 0.04 * frame_area:
                cv2.drawContours(leaf_mask, [c], -1, 255, -1)
        if cv2.countNonZero(leaf_mask) == 0:
            main_cnt = max(cnts, key=cv2.contourArea)
            cv2.drawContours(leaf_mask, [main_cnt], -1, 255, -1)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_lg)
    else:
        leaf_mask = plant_combined

    # ── 3. Quantitative Severity ────────────────────────────────────────────
    total_leaf_pixels = max(1, cv2.countNonZero(leaf_mask))
    lesions = blight_mask & (leaf_mask > 0)
    diseased_pixels = int(np.count_nonzero(lesions))
    severity_pct = round((diseased_pixels / float(total_leaf_pixels)) * 100.0, 1)

    # ── 4. Map 1: Lesion Inspection Overlay ─────────────────────────────────
    lesion_overlay = img_disp.copy()
    tint_layer = img_disp.copy()
    # Tint diseased areas with an amber-red warmth [B, G, R] = [20, 90, 245]
    tint_layer[lesions] = [20, 90, 245]
    cv2.addWeighted(tint_layer, 0.45, lesion_overlay, 0.55, 0, lesion_overlay)

    # Outline lesion clusters with glowing golden-yellow contours
    lesion_uint8 = (lesions.astype(np.uint8)) * 255
    cnts_les, _ = cv2.findContours(lesion_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_count = 0
    for c in cnts_les:
        if cv2.contourArea(c) > 12:
            cv2.drawContours(lesion_overlay, [c], -1, (0, 215, 255), 2)
            lesion_count += 1

    # ── 5. Map 2: Chlorophyll Bio-Vegetation Map ─────────────────────────────
    # Normalize ExG into 0-255 range and apply Summer/Viridis-like vegetation gradient
    exg_clipped = np.clip((exg + 25.0) / 140.0 * 255.0, 0, 255).astype(np.uint8)
    exg_color = cv2.applyColorMap(exg_clipped, cv2.COLORMAP_SUMMER)

    exg_bio_map = np.zeros_like(img_disp)
    exg_bio_map[leaf_mask > 0] = exg_color[leaf_mask > 0]

    # Dim background context outside leaf
    bg_dimmed = (img_disp.astype(np.float32) * 0.15).astype(np.uint8)
    exg_bio_map[leaf_mask == 0] = bg_dimmed[leaf_mask == 0]

    meta = {
        "severity_pct": severity_pct,
        "lesion_cluster_count": lesion_count,
        "leaf_surface_pixels": total_leaf_pixels,
        "compromised_pixels": diseased_pixels,
        "chlorophyll_active_pct": round(100.0 - severity_pct, 1),
    }

    return lesion_overlay, exg_bio_map, severity_pct, meta
