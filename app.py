"""
app.py
======
PlantGuard AI - Cutting-Edge Plant Disease Detection & Clinical Agronomy System.
Built with Classical Machine Learning (Random Forest / SVM / KNN), OpenCV, and Flask.

Key Capabilities:
  - Multi-tier Botanical Leaf & Pathogen Validation (Strictly rejects non-leaf images,
    human portraits, artificial packaging, cars, and documents)
  - Bio-Optical Pathological Inspection:
      * Interactive Lesion Heatmap Overlay (highlights active disease lesions)
      * Chlorophyll Bio-Vegetation Map (Excess Green Index ExG visualization)
      * Quantitative Infection Severity Meter (% damaged surface area)
  - Audio Voice Diagnosis Readout (Web Speech API)
  - Printable Medical Plant Prescription / Clinical Certificate
  - Live Webcam & Mobile Camera Scanner Mode
  - 1-Click Interactive Sample Test Bench
  - AI Agronomist Virtual Consultation Assistant
"""

import os
import io
import time
import uuid
import base64
import pickle
from datetime import datetime

import cv2
import numpy as np
from flask import (Flask, request, render_template,
                   jsonify, redirect, url_for, flash)
from werkzeug.utils import secure_filename

from utils.feature_extractor import extract_features
from utils.leaf_validator import validate_leaf_image
from utils.visualizer import generate_diagnostic_overlays


# ─── Application Configuration ────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "plantguard-ai-production-key-2024")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "model")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
CHARTS_DIR    = os.path.join(BASE_DIR, "static", "charts")
SAMPLES_DIR   = os.path.join(BASE_DIR, "static", "samples")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHARTS_DIR,    exist_ok=True)
os.makedirs(SAMPLES_DIR,   exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB limit

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}


# ─── Clinical Treatment & Fertilizer Knowledge Base ───────────────────────────
FERTILIZER_MAP = {
    "Healthy": {
        "title":       "Optimal Botanical Vitality",
        "scientific":  "Healthy Vegetative Tissue",
        "name":        "Balanced Plant Nutrition (NPK 10-10-10)",
        "summary":     "Your plant leaf is in prime health with robust cellular structure and rich chlorophyll density.",
        "application": "Maintain consistent bi-weekly watering. Apply balanced liquid NPK formula once every 21 days.",
        "organic":     "Cold-pressed seaweed extract, vermicompost tea, or fish emulsion foliar spray.",
        "action":      "No chemical fungicide needed. Regularly inspect leaf undersides for early pest colonization.",
        "icon":        "🌿",
        "color":       "#10b981",
        "badge_class": "badge-healthy",
        "urgency":     "Normal Maintenance",
    },
    "Early_Blight": {
        "title":       "Early Blight",
        "scientific":  "Alternaria solani",
        "name":        "High Nitrogen & Potassium Immune Booster (NPK 30-10-10)",
        "summary":     "Fungal pathogen causing concentric target-board lesions with chlorotic yellow halos.",
        "application": "Apply copper hydroxide or chlorothalonil fungicide every 7–10 days. Supplement with foliar nitrogen & potassium.",
        "organic":     "Bacillus subtilis biological fungicide, Bordeaux mixture, or 0.5% potassium bicarbonate spray.",
        "action":      "Prune affected lower leaves immediately. Disinfect pruning shears in 70% isopropyl alcohol. Avoid overhead irrigation.",
        "icon":        "⚠️",
        "color":       "#f59e0b",
        "badge_class": "badge-warning",
        "urgency":     "Moderate Action Required",
    },
    "Late_Blight": {
        "title":       "Late Blight",
        "scientific":  "Phytophthora infestans",
        "name":        "Potassium-Phosphorus Resilience Compound (NPK 5-10-20)",
        "summary":     "Aggressive oomycete water mold causing rapidly expanding water-soaked necrotic lesions and tissue collapse.",
        "application": "Apply systemic preventative fungicide (Mancozeb or Dimethomorph). Boost potassium levels to harden plant cell membranes.",
        "organic":     "Fixed copper soap fungicide applied promptly at 5-day intervals; strictly avoid foliage dampness.",
        "action":      "Quarantine plant area. Severely infected leaves must be incinerated or bagged (do NOT compost). Ventilate canopy.",
        "icon":        "🚨",
        "color":       "#ef4444",
        "badge_class": "badge-danger",
        "urgency":     "Critical Immediate Action",
    },
}


# ─── Load Machine Learning Artifacts ──────────────────────────────────────────
def load_ml_pipeline():
    """Load trained classifier, scaler, and label encoder."""
    paths = {
        "model":   os.path.join(MODEL_DIR, "best_model.pkl"),
        "scaler":  os.path.join(MODEL_DIR, "scaler.pkl"),
        "encoder": os.path.join(MODEL_DIR, "label_encoder.pkl"),
    }
    for p in paths.values():
        if not os.path.exists(p):
            return None, None, None

    try:
        with open(paths["model"],   "rb") as f: m  = pickle.load(f)
        with open(paths["scaler"],  "rb") as f: s  = pickle.load(f)
        with open(paths["encoder"], "rb") as f: le = pickle.load(f)
        return m, s, le
    except Exception as exc:
        print(f"[ERROR] Failed to load ML artifacts: {exc}")
        return None, None, None


model, scaler, label_encoder = load_ml_pipeline()


# ─── Image Processing Helpers ─────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_image_bytes(file_bytes: bytes) -> np.ndarray | None:
    """Decode raw bytes into an OpenCV BGR image."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def decode_base64_data_url(data_url: str) -> bytes | None:
    """Extract raw image bytes from a browser base64 data URL."""
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        return base64.b64decode(data_url)
    except Exception:
        return None


# ─── Core Prediction Pipeline ─────────────────────────────────────────────────
def run_full_diagnosis(file_bytes: bytes, original_filename: str = "upload.jpg"):
    """
    Runs the full end-to-end diagnosis:
      1. Image decoding & validation
      2. Strict botanical leaf verification
      3. Bio-optical lesion heatmap and ExG bio-map generation
      4. Classical ML feature extraction & scaled inference
      5. Clinical prescription mapping
    """
    img = decode_image_bytes(file_bytes)
    if img is None:
        return {
            "success": False,
            "is_leaf": False,
            "error_type": "decode_error",
            "message": "Could not decode file as a valid image. Please upload a standard JPG, PNG, or WEBP image."
        }

    # 1. Botanical Leaf Verification Engine
    is_leaf, leaf_msg, leaf_metrics = validate_leaf_image(img)
    if not is_leaf:
        return {
            "success": False,
            "is_leaf": False,
            "error_type": "not_a_leaf",
            "message": leaf_msg,
            "metrics": leaf_metrics,
            "original_filename": original_filename
        }

    if model is None:
        return {
            "success": False,
            "is_leaf": True,
            "error_type": "model_missing",
            "message": "ML Model artifacts not found. Please run 'python train_model.py' to initialize classifiers."
        }

    # 2. Save Original Image
    uid = uuid.uuid4().hex[:10]
    ext = "jpg"
    orig_name = f"leaf_{uid}_orig.{ext}"
    orig_path = os.path.join(UPLOAD_FOLDER, orig_name)
    cv2.imwrite(orig_path, img)

    # 3. Generate Lesion Overlay and Chlorophyll Bio-Map
    lesion_map, exg_map, severity_pct, visual_meta = generate_diagnostic_overlays(img)

    lesion_name = f"leaf_{uid}_lesions.jpg"
    exg_name    = f"leaf_{uid}_exg.jpg"
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, lesion_name), lesion_map)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, exg_name),    exg_map)

    # 4. Feature Extraction & Classical ML Inference
    img_64 = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    feats = extract_features(img_64).reshape(1, -1)
    feats_scaled = scaler.transform(feats)

    pred_raw = model.predict(feats_scaled)[0]
    proba = model.predict_proba(feats_scaled)[0]

    # Resolve predicted class name
    if isinstance(pred_raw, (int, np.integer)):
        pred_class = label_encoder.inverse_transform([pred_raw])[0]
    else:
        pred_class = str(pred_raw)

    confidence = float(max(proba))

    # Multi-class probability breakdown
    classes = list(model.classes_)
    if isinstance(classes[0], (int, np.integer)):
        classes = list(label_encoder.inverse_transform(classes))

    probabilities = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(classes, proba)
    }

    # Severity Grading
    if pred_class == "Healthy":
        severity_grade = "Negligible (<5%)"
        severity_color = "#10b981"
    elif severity_pct < 20:
        severity_grade = f"Mild ({severity_pct}%)"
        severity_color = "#f59e0b"
    elif severity_pct < 50:
        severity_grade = f"Moderate ({severity_pct}%)"
        severity_color = "#ea580c"
    else:
        severity_grade = f"Severe ({severity_pct}%)"
        severity_color = "#ef4444"

    # Base Treatment Template
    fert_template = FERTILIZER_MAP.get(pred_class, FERTILIZER_MAP["Healthy"])
    fertilizer = dict(fert_template)

    # Voice narration script for Web Speech API
    speech_text = (
        f"PlantGuard Diagnosis Complete. Condition identified: {fertilizer['title']} "
        f"with {round(confidence * 100)}% model confidence. "
        f"Infection severity is {severity_grade}. "
        f"Recommended treatment: {fertilizer['name']}. {fertilizer['action']}"
    )

    # Timestamp & Prescription ID
    rx_id = f"RX-{datetime.now().strftime('%Y%m%d')}-{uid[:5].upper()}"

    return {
        "success":            True,
        "is_leaf":            True,
        "prediction":         pred_class,
        "confidence":         round(confidence * 100, 1),
        "probabilities":      probabilities,
        "severity_pct":       severity_pct,
        "severity_grade":     severity_grade,
        "severity_color":     severity_color,
        "fertilizer":         fertilizer,
        "leaf_metrics":       leaf_metrics,
        "visual_meta":        visual_meta,
        "speech_text":        speech_text,
        "rx_id":              rx_id,
        "date_str":           datetime.now().strftime("%B %d, %Y - %I:%M %p"),
        "image_url":          url_for("static", filename=f"uploads/{orig_name}"),
        "lesion_url":         url_for("static", filename=f"uploads/{lesion_name}"),
        "exg_url":            url_for("static", filename=f"uploads/{exg_name}"),
        "original_name":      original_filename,
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main application portal."""
    model_ready = model is not None
    # Available sample test library
    samples = [
        {
            "id": "healthy_leaf",
            "name": "Healthy Foliage",
            "desc": "Pristine leaf with balanced chlorophyll",
            "tag": "100% Healthy",
            "img": url_for("static", filename="samples/healthy_leaf.jpg")
        },
        {
            "id": "early_blight",
            "name": "Early Blight",
            "desc": "Concentric rings & chlorotic halo",
            "tag": "Alternaria solani",
            "img": url_for("static", filename="samples/early_blight.jpg")
        },
        {
            "id": "late_blight",
            "name": "Late Blight",
            "desc": "Water-soaked dark lesions",
            "tag": "Phytophthora",
            "img": url_for("static", filename="samples/late_blight.jpg")
        },
        {
            "id": "non_leaf_sample",
            "name": "Non-Leaf Object",
            "desc": "Commercial packaging test",
            "tag": "Rejection Test",
            "img": url_for("static", filename="samples/non_leaf_sample.jpg")
        },
    ]
    return render_template("index.html", model_ready=model_ready, samples=samples)


@app.route("/predict", methods=["POST"])
def predict():
    """Form upload & Live Camera submission handler."""
    # Check for webcam base64 data
    camera_data = request.form.get("camera_data")
    if camera_data:
        file_bytes = decode_base64_data_url(camera_data)
        if not file_bytes:
            flash("Could not process camera capture. Please try snapping again.", "error")
            return redirect(url_for("index"))
        filename = f"webcam_snap_{int(time.time())}.jpg"
    else:
        if "file" not in request.files:
            flash("No image file was selected.", "error")
            return redirect(url_for("index"))

        file = request.files["file"]
        if file.filename == "":
            flash("No file was chosen. Please select an image or use the camera.", "error")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash(f"Unsupported file format. Please upload JPG, PNG, or WEBP.", "error")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        file_bytes = file.read()

    # Execute diagnosis pipeline
    diag = run_full_diagnosis(file_bytes, original_filename=filename)

    if not diag["success"]:
        # Render rejection experience with clear explanation
        return render_template("rejected.html", error=diag)

    return render_template("result.html", **diag)


@app.route("/predict_sample/<sample_id>")
def predict_sample(sample_id):
    """Instant 1-click test benchmark handler."""
    sample_file = os.path.join(SAMPLES_DIR, f"{sample_id}.jpg")
    if not os.path.exists(sample_file):
        flash("Sample test image not found.", "error")
        return redirect(url_for("index"))

    with open(sample_file, "rb") as f:
        file_bytes = f.read()

    diag = run_full_diagnosis(file_bytes, original_filename=f"{sample_id}.jpg")

    if not diag["success"]:
        return render_template("rejected.html", error=diag)

    return render_template("result.html", **diag)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API for mobile apps and external services."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Missing 'file' multipart field."}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": f"Allowed extensions: {list(ALLOWED_EXTENSIONS)}"}), 400

    file_bytes = file.read()
    diag = run_full_diagnosis(file_bytes, original_filename=secure_filename(file.filename))

    if not diag["success"]:
        return jsonify(diag), 422

    return jsonify(diag)


@app.route("/api/chat_agronomist", methods=["POST"])
def chat_agronomist():
    """Virtual Agronomist Assistant Q&A endpoint."""
    data = request.get_json(silent=True) or {}
    query = data.get("message", "").strip().lower()

    if not query:
        return jsonify({"reply": "Please ask a question regarding your plants, fertilizer dosing, or disease symptoms."})

    # Intelligent botanical knowledge responses
    if "dosage" in query or "how much" in query or "gram" in query or "dose" in query:
        reply = (
            "🌿 **Fertilizer Dosage Guide**:\n"
            "- For Healthy/Maintenance: 5g of balanced NPK 10-10-10 per 5 liters of water once every 3 weeks.\n"
            "- For Early Blight: 8g of NPK 30-10-10 per 5 liters of water applied directly at soil level.\n"
            "- For Late Blight: 10g of NPK 5-10-20 per 5 liters combined with recommended copper fungicide.\n"
            "Always moisten soil before applying fertilizer to prevent root burn."
        )
    elif "water" in query or "irrigation" in query or "sprinkler" in query:
        reply = (
            "💧 **Optimal Watering Protocol**:\n"
            "Never water from overhead or wet the leaf canopy! Fungal spores (Alternaria & Phytophthora) "
            "require damp leaf surfaces to germinate. Use drip irrigation or soak soil gently at ground level in the early morning."
        )
    elif "organic" in query or "natural" in query or "neem" in query:
        reply = (
            "🌱 **Organic Management Plan**:\n"
            "1. Cold-pressed Neem Oil (0.5% emulsion + mild organic castile soap) applied at sunset.\n"
            "2. Baking soda / Potassium bicarbonate spray (1 tsp per quart of water) modifies leaf pH to inhibit fungal growth.\n"
            "3. Compost tea rich in beneficial mycorrhizae to boost root resistance."
        )
    elif "prun" in query or "cut" in query or "remove" in query:
        reply = (
            "✂️ **Safe Pruning Procedure**:\n"
            "1. Always prune during dry, sunny weather.\n"
            "2. Dip shears into 70% rubbing alcohol between each cut.\n"
            "3. Cut 2 inches below visible necrotic spots into healthy tissue.\n"
            "4. Bag and dispose of diseased clippings; NEVER add blighted foliage to compost."
        )
    else:
        reply = (
            "🌾 **Plant Doctor Advice**:\n"
            "Early diagnosis is key to protecting your yield. Ensure 18–24 inches spacing between crops "
            "for airflow, mulch soil with straw to prevent fungal splash, and apply tailored NPK nutrition "
            "based on your diagnostic scan."
        )

    return jsonify({"reply": reply})


@app.route("/charts")
def charts():
    """Model analytics & comparison charts."""
    chart_files = [f for f in os.listdir(CHARTS_DIR) if f.endswith(".png")]
    chart_urls  = [url_for("static", filename=f"charts/{f}") for f in sorted(chart_files)]
    return render_template("charts.html", chart_urls=chart_urls)


@app.errorhandler(413)
def file_too_large(e):
    flash("The image exceeds the 15 MB file size limit. Please upload a smaller image.", "error")
    return redirect(url_for("index"))


# ─── Main Server Launch ───────────────────────────────────────────────────────
if __name__ == "__main__":
    status = "Active & Ready" if model is not None else "Model not loaded"
    print(f"\n=======================================================")
    print(f"  PlantGuard AI Server Started [Model: {status}]")
    print(f"  Access local app at: http://localhost:5000/")
    print(f"=======================================================\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
