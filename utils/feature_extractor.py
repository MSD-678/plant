"""
utils/feature_extractor.py
===========================
Extracts a rich feature vector from a leaf image using:
  - Color features   : RGB channel statistics and histograms
  - Texture features : GLCM (Gray-Level Co-occurrence Matrix) properties
  - Shape features   : Contour-based geometric descriptors

All images are resized to 64x64 before processing.
Total features per image: ~132
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


# ─── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE = (64, 64)          # Target resize dimensions
HIST_BINS  = 32                # Bins per channel for color histograms
GLCM_DISTANCES  = [1]          # Pixel distance(s) for GLCM
GLCM_ANGLES     = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 4 directions


# ─── Public API ───────────────────────────────────────────────────────────────

def load_and_preprocess(image_path: str) -> np.ndarray | None:
    """
    Load an image from disk and resize it to IMAGE_SIZE.

    Returns the BGR image (uint8, HxWx3) or None if loading fails.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return img


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract a flat feature vector from a preprocessed BGR image.

    Args:
        img_bgr: uint8 BGR image of shape (64, 64, 3)

    Returns:
        1-D numpy array of float64 with ~132 features.
    """
    color_feats   = _extract_color_features(img_bgr)
    texture_feats = _extract_texture_features(img_bgr)
    shape_feats   = _extract_shape_features(img_bgr)

    return np.concatenate([color_feats, texture_feats, shape_feats])


def extract_features_from_path(image_path: str) -> np.ndarray | None:
    """
    Convenience wrapper: load, preprocess, and extract features in one call.

    Returns:
        Feature vector or None if the image cannot be loaded.
    """
    img = load_and_preprocess(image_path)
    if img is None:
        return None
    return extract_features(img)


# ─── Color Features ───────────────────────────────────────────────────────────

def _extract_color_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract 102 color features:
      - Per-channel mean and std dev for B, G, R  → 6 values
      - Per-channel 32-bin normalised histogram    → 3 × 32 = 96 values
    """
    features = []

    # Convert BGR → RGB for more intuitive channel ordering
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Channel statistics
    for c in range(3):
        channel = img_rgb[:, :, c].astype(np.float64)
        features.append(channel.mean())
        features.append(channel.std())

    # Normalised histograms
    for c in range(3):
        hist = cv2.calcHist([img_rgb], [c], None, [HIST_BINS], [0, 256])
        hist = hist.flatten() / (img_rgb.shape[0] * img_rgb.shape[1])  # normalise
        features.extend(hist.tolist())

    return np.array(features, dtype=np.float64)


# ─── Texture Features ─────────────────────────────────────────────────────────

def _extract_texture_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract 24 GLCM texture features:
      6 properties × 1 distance × 4 angles → 24 values (mean across angles)

    Properties: contrast, dissimilarity, homogeneity, energy, correlation, ASM
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # GLCM expects uint8; scale to reduce quantisation levels for speed
    gray_scaled = (gray // 32).astype(np.uint8)   # 0–7, 8 levels

    glcm = graycomatrix(
        gray_scaled,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=8,
        symmetric=True,
        normed=True
    )

    properties = ['contrast', 'dissimilarity', 'homogeneity',
                  'energy', 'correlation', 'ASM']

    features = []
    for prop in properties:
        values = graycoprops(glcm, prop)   # shape: (len(distances), len(angles))
        features.append(float(values.mean()))   # average across distances & angles

    return np.array(features, dtype=np.float64)


# ─── Shape Features ───────────────────────────────────────────────────────────

def _extract_shape_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract 6 contour-based shape features:
      area, perimeter, aspect_ratio, extent, solidity, equiv_diameter
    """
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu threshold on blurred grayscale for robust segmentation
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Default safe values if no contour found
    area           = 0.0
    perimeter      = 0.0
    aspect_ratio   = 1.0
    extent_val     = 0.0
    solidity       = 0.0
    equiv_diameter = 0.0

    if contours:
        # Use the largest contour (assumed to be the leaf)
        cnt = max(contours, key=cv2.contourArea)

        area      = float(cv2.contourArea(cnt))
        perimeter = float(cv2.arcLength(cnt, True))

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0

        rect_area   = float(w * h)
        extent_val  = area / rect_area if rect_area > 0 else 0.0

        hull         = cv2.convexHull(cnt)
        hull_area    = float(cv2.contourArea(hull))
        solidity     = area / hull_area if hull_area > 0 else 0.0

        equiv_diameter = float(np.sqrt(4 * area / np.pi)) if area > 0 else 0.0

    # Normalise area and perimeter by image pixel count for scale invariance
    total_pixels = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    area           /= total_pixels
    perimeter      /= (4 * max(IMAGE_SIZE))   # rough normalisation

    return np.array([area, perimeter, aspect_ratio,
                     extent_val, solidity, equiv_diameter],
                    dtype=np.float64)
