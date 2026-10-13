"""
utils/leaf_validator.py
========================
Botanical Gating Filter:
Ensures the image contains an actual leaf/plant blade and strictly rejects
animals, humans, indoor objects, and general outdoor landscapes.
"""

import cv2
import numpy as np


def is_valid_leaf(img_bgr: np.ndarray) -> tuple[bool, str]:
    """
    Validates whether the provided BGR image contains a genuine botanical leaf.
    Returns: (is_valid: bool, rejection_reason: str)
    """
    if img_bgr is None or img_bgr.size == 0:
        return False, "Empty or invalid image payload."

    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    # 1. Convert to HSV for accurate plant color segmentation
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # 2. Plant chlorophyll color mask (Yellowish-green to dark green & chlorotic yellow)
    # Hue: ~25 to 88 covers dying yellow, diseased foliage, and deep healthy green
    lower_plant = np.array([25, 40, 35])
    upper_plant = np.array([88, 255, 255])
    plant_mask = cv2.inRange(hsv, lower_plant, upper_plant)

    plant_pixel_count = cv2.countNonZero(plant_mask)
    plant_coverage_ratio = plant_pixel_count / total_pixels

    # Rule A: Minimum plant tissue coverage
    if plant_coverage_ratio < 0.20:
        return False, (
            "No significant plant/leaf chlorophyll detected. "
            f"(Coverage: {plant_coverage_ratio*100:.1f}%, required >= 20%)"
        )

    # 3. Foreground Subject Check (The "Cat/Human on Grass" filter)
    # In a real leaf photo, the central region MUST be plant tissue, NOT an animal/object.
    center_y1, center_y2 = int(h * 0.25), int(h * 0.75)
    center_x1, center_x2 = int(w * 0.25), int(w * 0.75)
    center_plant_crop = plant_mask[center_y1:center_y2, center_x1:center_x2]
    center_total = (center_y2 - center_y1) * (center_x2 - center_x1)
    center_plant_ratio = cv2.countNonZero(center_plant_crop) / center_total

    # If the center of the photo is non-plant (e.g. a cat body or a face), reject it
    if center_plant_ratio < 0.35:
        return False, (
            "Non-botanical subject detected in central focus. "
            "Please center the camera directly on the leaf blade."
        )

    # 4. Contour Structure Analysis (Leaf Blade vs. Scattered Grass Turf)
    # A leaf should form cohesive, substantial connected regions, not fragmented grass streaks.
    contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "Failed to isolate cohesive plant contours."

    max_contour = max(contours, key=cv2.contourArea)
    max_contour_area = cv2.contourArea(max_contour)
    max_contour_ratio = max_contour_area / total_pixels

    # The primary leaf blade must occupy at least 12% of the entire frame
    if max_contour_ratio < 0.12:
        return False, (
            "Scattered foliage or background grass detected without a primary leaf blade. "
            "Bring the camera closer to a single leaf."
        )

    # 5. Passed all botanical safety checks
    return True, "Valid botanical leaf verified."