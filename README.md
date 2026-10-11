# 🌿 PlantGuard AI — Plant Disease Detection & Fertilizer Suggestion

A production-ready plant disease detection system using **classical Machine Learning** (no deep learning).

## Features
- **Detects**: Healthy, Early Blight, Late Blight
- **ML Models**: Random Forest, SVM, KNN (GridSearchCV tuning)
- **Features**: Color (RGB stats + histograms), Texture (GLCM), Shape (contour geometry)
- **Flask Web App** with drag-and-drop UI + REST API
- **Charts**: Model comparison + confusion matrices

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset
Place your leaf images in:
```
dataset/
  Healthy/        ← healthy leaf images
  Early_Blight/   ← early blight images
  Late_Blight/    ← late blight images
```

> **OR** generate synthetic test images first:
> ```bash
> python generate_sample_data.py
> ```

### 3. Train the models
```bash
python train_model.py
```
Outputs: `model/best_model.pkl`, `model/scaler.pkl`, `model/label_encoder.pkl`, `static/charts/`

### 4. Run the web app
```bash
python app.py
```
Open **http://localhost:5000**

---

## REST API

```bash
curl -X POST -F "file=@your_leaf.jpg" http://localhost:5000/api/predict
```

Response:
```json
{
  "success": true,
  "class": "Early_Blight",
  "confidence": 0.91,
  "fertilizer": {
    "name": "High Nitrogen Fertilizer (NPK 30-10-10)",
    "description": "...",
    "icon": "⚠️",
    "color": "#f39c12"
  }
}
```

---

## Project Structure
```
plant/
├── dataset/               ← Leaf images by class
├── model/                 ← Trained model artifacts (after training)
├── static/
│   ├── css/style.css      ← Dark-themed premium UI
│   ├── js/main.js         ← Drag-and-drop + preview
│   ├── charts/            ← Comparison charts (after training)
│   └── uploads/           ← Temporary uploaded images
├── templates/
│   ├── index.html         ← Upload form
│   ├── result.html        ← Prediction result
│   └── charts.html        ← Model charts
├── utils/
│   └── feature_extractor.py  ← Core feature engineering
├── app.py                 ← Flask web app + REST API
├── train_model.py         ← Training pipeline
├── generate_sample_data.py   ← Synthetic dataset generator
└── requirements.txt
```

## Fertilizer Recommendations

| Disease | Fertilizer |
|---------|-----------|
| Healthy | Balanced Fertilizer (NPK 10-10-10) |
| Early Blight | High Nitrogen Fertilizer (NPK 30-10-10) |
| Late Blight | Potassium-rich Fertilizer (NPK 5-10-20) |

## Feature Engineering (~132 features per image)

| Type | Count | Description |
|------|-------|-------------|
| Color | 102 | RGB mean/std (6) + 32-bin histograms (96) |
| Texture | 6 | GLCM: contrast, dissimilarity, homogeneity, energy, correlation, ASM |
| Shape | 6 | Area, perimeter, aspect ratio, extent, solidity, equiv. diameter |
