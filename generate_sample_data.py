"""
generate_sample_data.py
========================
Generates realistic botanical leaf dataset samples in dataset/<ClassName>/
for training classical ML classifiers (Random Forest, SVM, KNN).

Creates realistic leaf morphology:
  - Natural leaf blades (ovate, elliptical, lanceolate)
  - Midrib and branching lateral venation
  - Realistic cellular texture and chlorophyll color distributions
  - Class-specific pathological lesions:
      * Healthy: pristine green with healthy venation
      * Early Blight: concentric circular brown rings with chlorotic yellow halos
      * Late Blight: irregular dark water-soaked necrotic lesions on margins
  - Multiple background styles: scientific white, neutral light gray, dark soil
"""

import os
import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

CLASSES = ["Healthy", "Early_Blight", "Late_Blight"]
NUM_SAMPLES_PER_CLASS = 80
IMAGE_SIZE = (64, 64)


def generate_realistic_leaf_sample(class_name: str, seed: int) -> np.ndarray:
    """
    Generate a high-fidelity synthetic botanical leaf sample image.
    """
    rng = np.random.default_rng(seed)
    h, w = 128, 128  # Generate at 128x128 then resize to 64x64 for smooth antialiasing

    # Background palette
    bg_style = rng.choice(["white", "light_gray", "soil"])
    if bg_style == "white":
        img = np.full((h, w, 3), 255.0, dtype=np.float32)
    elif bg_style == "light_gray":
        val = float(rng.integers(225, 246))
        img = np.full((h, w, 3), val, dtype=np.float32)
    else:
        # Dark soil/garden background
        b = float(rng.integers(20, 35))
        g = float(rng.integers(28, 45))
        r = float(rng.integers(25, 38))
        img = np.full((h, w, 3), (b, g, r), dtype=np.float32)

    center_x = int(64 + rng.integers(-4, 5))
    center_y = int(64 + rng.integers(-4, 5))
    length = int(rng.integers(40, 52))
    width = int(rng.integers(22, 32))
    angle = int(rng.integers(-25, 26))

    base_g = int(rng.integers(120, 180))
    base_r = int(rng.integers(30, 80))
    base_b = int(rng.integers(25, 60))

    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(leaf_mask, (center_x, center_y), (width, length), angle, 0, 360, 255, -1)

    leaf_tex = np.zeros((h, w, 3), dtype=np.float32)
    leaf_tex[:, :] = (float(base_b), float(base_g), float(base_r))
    noise = rng.normal(0, 8, (h, w, 3)).astype(np.float32)
    leaf_tex = np.clip(leaf_tex + noise, 0, 255)

    # Midrib and lateral veins
    vein_color = (
        float(max(0, base_b - 12)),
        float(min(255, base_g + 24)),
        float(min(255, base_r + 20))
    )
    cv2.line(leaf_tex, (center_x, center_y - length + 6), (center_x, center_y + length - 6), vein_color, 2)
    for vy in range(center_y - length + 14, center_y + length - 14, 10):
        cv2.line(leaf_tex, (center_x, vy), (center_x - width + 6, vy - 8), vein_color, 1)
        cv2.line(leaf_tex, (center_x, vy), (center_x + width - 6, vy - 8), vein_color, 1)

    # Class-specific disease lesions
    if class_name == "Early_Blight":
        # Concentric dark brown circles with yellow chlorotic halos (Alternaria solani)
        num_spots = int(rng.integers(2, 6))
        for _ in range(num_spots):
            sx = int(center_x + rng.integers(-width + 10, width - 10))
            sy = int(center_y + rng.integers(-length + 14, length - 14))
            s_radius = int(rng.integers(5, 13))
            # Yellow chlorotic halo
            cv2.circle(leaf_tex, (sx, sy), s_radius + 4, (30.0, 205.0, 215.0), -1)
            # Brown necrotic concentric rings
            cv2.circle(leaf_tex, (sx, sy), s_radius, (25.0, 60.0, 120.0), -1)
            cv2.circle(leaf_tex, (sx, sy), max(2, s_radius - 3), (20.0, 45.0, 95.0), 1)
            cv2.circle(leaf_tex, (sx, sy), max(1, s_radius - 6), (15.0, 32.0, 72.0), -1)

    elif class_name == "Late_Blight":
        # Irregular water-soaked brown/black necrotic lesions on margins (Phytophthora infestans)
        num_patches = int(rng.integers(2, 5))
        for _ in range(num_patches):
            px = int(center_x + rng.choice([-1, 1]) * rng.integers(width // 4, width - 4))
            py = int(center_y + rng.integers(-length + 12, length - 12))
            pr_x = int(rng.integers(10, 22))
            pr_y = int(rng.integers(12, 26))
            # Pale chlorotic margin
            cv2.ellipse(leaf_tex, (px, py), (pr_x + 3, pr_y + 3), int(rng.integers(0, 180)), 0, 360, (40.0, 130.0, 110.0), -1)
            # Dark necrotic center
            cv2.ellipse(leaf_tex, (px, py), (pr_x, pr_y), int(rng.integers(0, 180)), 0, 360, (20.0, 42.0, 80.0), -1)

    # Smooth antialiased boundary blending
    mask_float = cv2.GaussianBlur(leaf_mask.astype(np.float32) / 255.0, (3, 3), 0)[:, :, np.newaxis]
    blended = (leaf_tex * mask_float + img * (1.0 - mask_float)).astype(np.uint8)

    # Resize to target dataset dimensions
    out = cv2.resize(blended, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return out


def main():
    print("=" * 60)
    print("  Generating Botanical Leaf Dataset Samples")
    print("=" * 60)
    total = 0

    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(NUM_SAMPLES_PER_CLASS):
            img = generate_realistic_leaf_sample(class_name, seed=i * 107 + hash(class_name) % 997)
            path = os.path.join(class_dir, f"sample_{i:03d}.jpg")
            cv2.imwrite(path, img)
            total += 1

        print(f"  [{class_name}] {NUM_SAMPLES_PER_CLASS} images saved -> {class_dir}")

    # Also augment real sample images from static/samples into dataset
    sample_mappings = {
        "Healthy": "static/samples/healthy_leaf.jpg",
        "Early_Blight": "static/samples/early_blight.jpg",
        "Late_Blight": "static/samples/late_blight.jpg"
    }

    print("\n  Adding augmented real leaf crops to dataset ...")
    for class_name, src_path in sample_mappings.items():
        if os.path.exists(src_path):
            src_img = cv2.imread(src_path)
            class_dir = os.path.join(DATASET_DIR, class_name)
            for j in range(15):
                # Apply random crop and flip
                h, w = src_img.shape[:2]
                ch = int(h * np.random.uniform(0.7, 0.95))
                cw = int(w * np.random.uniform(0.7, 0.95))
                y0 = np.random.randint(0, h - ch + 1)
                x0 = np.random.randint(0, w - cw + 1)
                crop = src_img[y0:y0+ch, x0:x0+cw]
                if np.random.rand() > 0.5:
                    crop = cv2.flip(crop, 1)
                resized = cv2.resize(crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                crop_path = os.path.join(class_dir, f"real_aug_{j:03d}.jpg")
                cv2.imwrite(crop_path, resized)
                total += 1

    print(f"\n[OK] Successfully generated {total} botanical leaf samples across 3 classes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
