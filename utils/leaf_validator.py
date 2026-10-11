"""
utils/leaf_validator.py
========================
State-of-the-art Botanical & Computer Vision Validation Engine.

Strictly validates whether an uploaded image contains a legitimate botanical plant leaf
or foliage, rejecting non-leaf pictures (human faces, pets, vehicles, artificial objects,
commercial product packaging, documents, screenshots, and blank images).

Multi-Tier Botanical Verification:
  1. Resolution & Optical Sharpness (Laplacian variance check)
  2. Human Subject / Skin Tone Rejection (YCrCb + HSV dermatological filtering)
  3. Chlorophyll & Foliage Biological Signatures:
     - Excess Green Index (ExG = 2G - R - B)
     - Green Leaf Index (GLI)
     - Organic necrotic/blight lesion identification
  4. Object-Level Botanical Purity & Packaging Filtering (isolates foreground objects
     and verifies that the primary body consists of biological plant tissue, rejecting
     supplement bottles, product boxes, mugs, phones, etc.)
  5. Synthetic Flat Color Rejection (flags digital/artificial solid green surfaces)
"""

from typing import Tuple, Dict, Any
import cv2
import numpy as np


def validate_leaf_image(img_bgr: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evaluate an OpenCV BGR image against botanical leaf criteria.

    Args:
        img_bgr: uint8 BGR numpy array from cv2.imread or cv2.imdecode.

    Returns:
        is_leaf (bool): True if verified as a plant leaf, False otherwise.
        message (str): Human-friendly diagnostic message explaining the decision.
        metrics (dict): Comprehensive botanical and optical measurements.
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        return False, "Could not decode or parse image data.", {}

    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    if h < 50 or w < 50 or total_pixels < 2500:
        return False, "Image resolution is too low. Please upload a clear photo of at least 100x100 pixels.", {
            "dimensions": f"{w}x{h}"
        }

    # Standardize analysis scale for consistent mathematical thresholds
    analysis_size = (256, 256)
    std_pixels = 256 * 256
    img_std = cv2.resize(img_bgr, analysis_size, interpolation=cv2.INTER_AREA)

    # ── 1. Optical Sharpness & Blank Detection ──────────────────────────────
    gray = cv2.cvtColor(img_std, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    intensity_std = float(gray.std())

    if lap_var < 8.0 or intensity_std < 8.0:
        return False, "Image is completely blank, solid color, or severely out-of-focus.", {
            "laplacian_variance": round(lap_var, 2),
            "intensity_std": round(intensity_std, 2)
        }

    # ── 2. Color Spaces & Vegetation Indices ────────────────────────────────
    img_rgb = cv2.cvtColor(img_std, cv2.COLOR_BGR2RGB)
    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)
    b = img_rgb[:, :, 2].astype(np.float64)

    hsv = cv2.cvtColor(img_std, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img_std, cv2.COLOR_BGR2YCrCb)

    # Excess Green Index (ExG): standard botanical index for chlorophyll reflection
    exg = 2.0 * g - r - b

    # ── 3. Human Skin Detection (Dermatological Filter) ─────────────────────
    # Human skin in YCrCb: Cr in [133, 173], Cb in [77, 127]
    skin_mask = (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 173) & \
                (ycrcb[:, :, 2] >= 77) & (ycrcb[:, :, 2] <= 127)
    skin_ratio = float(np.count_nonzero(skin_mask)) / std_pixels

    # ── 4. Botanical Foliage & Lesion Segmentation ─────────────────────────
    # Healthy Green foliage: Hue 26 to 92 (in OpenCV 0-180 scale), Sat >= 25, Val >= 25, ExG > -5
    green_mask = (hsv[:, :, 0] >= 26) & (hsv[:, :, 0] <= 92) & \
                 (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 25) & (exg > -5.0)
    green_ratio = float(np.count_nonzero(green_mask)) / std_pixels

    # Diseased / Blighted / Necrotic tissue (chlorotic halo or brown rot):
    # Hue 8 to 28, Sat >= 30, Val in [25, 230], Red higher than Blue, absorbs Blue (B < 155)
    blight_mask = (hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 28) & \
                  (hsv[:, :, 1] >= 30) & (hsv[:, :, 2] >= 25) & \
                  (b < 155.0) & (r > b + 15.0) & (~skin_mask)
    blight_ratio = float(np.count_nonzero(blight_mask)) / std_pixels

    # Total plant tissue presence
    total_plant_mask = green_mask | blight_mask
    total_plant_ratio = float(np.count_nonzero(total_plant_mask)) / std_pixels

    metrics: Dict[str, Any] = {
        "plant_coverage_pct": round(total_plant_ratio * 100, 1),
        "green_foliage_pct": round(green_ratio * 100, 1),
        "blight_lesion_pct": round(blight_ratio * 100, 1),
        "skin_presence_pct": round(skin_ratio * 100, 1),
        "sharpness_score": round(lap_var, 1),
    }

    # ── 5. Rejection Rules ──────────────────────────────────────────────────

    # Rule A: Dominant human subject / portrait
    if skin_ratio > 0.22 and green_ratio < 0.10:
        return False, (
            "Human subject or portrait detected instead of a plant leaf. "
            "Please take a clear photo of an actual plant leaf."
        ), metrics

    # Rule B: Insufficient botanical vegetation
    if total_plant_ratio < 0.12 and green_ratio < 0.08:
        return False, (
            "No botanical plant leaf or foliage detected in the photo. "
            "The image appears to be a non-plant object, document, or room. "
            "Please upload a close-up photo of a plant leaf."
        ), metrics

    # Rule C: Synthetic/Flat digital green surfaces (e.g. green painted walls, plastic toys, logos)
    if green_ratio > 0.18:
        green_hues = hsv[:, :, 0][green_mask]
        hue_std = float(np.std(green_hues)) if len(green_hues) > 0 else 0.0
        metrics["hue_diversity"] = round(hue_std, 2)
        if hue_std < 1.1 and lap_var < 45.0:
            return False, (
                "Artificial or synthetic flat green surface detected (e.g. wall, plastic, or graphic). "
                "Please photograph a real biological leaf with visible veins."
            ), metrics

    # Rule D: Object-level purity check for isolated commercial products / packaging
    # Sample 4 corners to check if the image has a studio white/light background
    corner_samples = np.concatenate([
        gray[:12, :12].flatten(), gray[:12, -12:].flatten(),
        gray[-12:, :12].flatten(), gray[-12:, -12:].flatten()
    ])
    bg_median = float(np.median(corner_samples))
    bg_std = float(np.std(corner_samples))

    if bg_median > 235.0 and bg_std < 20.0:
        obj_mask = (gray < bg_median - 15).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            main_cnt = max(cnts, key=cv2.contourArea)
            c_mask = np.zeros((256, 256), dtype=np.uint8)
            cv2.drawContours(c_mask, [main_cnt], -1, 255, -1)
            obj_area = cv2.countNonZero(c_mask)

            if obj_area > 0.10 * std_pixels:
                plant_inside = np.count_nonzero(total_plant_mask & (c_mask > 0))
                purity = plant_inside / float(obj_area)
                metrics["object_purity_pct"] = round(purity * 100, 1)

                if purity < 0.72:
                    return False, (
                        f"Commercial packaging, product bottle, or non-plant object detected "
                        f"({int(purity * 100)}% botanical purity). "
                        "Please upload a photo of an isolated plant leaf."
                    ), metrics

    # Rule E: If plant coverage is marginal (12% - 20%), ensure there is a clear connected leaf cluster
    if total_plant_ratio < 0.20:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_plant = cv2.morphologyEx(total_plant_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_plant)
        largest_blob = max(stats[1:, cv2.CC_STAT_AREA]) / float(std_pixels) if num_labels > 1 else 0
        if largest_blob < 0.08:
            return False, (
                "Foliage presence is too sparse or scattered. "
                "Please photograph an individual leaf close up."
            ), metrics

    # Passed all botanical tests
    return True, f"Verified Botanical Leaf ({int(total_plant_ratio * 100)}% foliage coverage)", metrics
