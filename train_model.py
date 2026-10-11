"""
train_model.py
==============
Train and compare Random Forest, SVM, and KNN classifiers on leaf images.

Pipeline:
  1. Load images from dataset/<ClassName>/
  2. Extract color + texture + shape features
  3. Scale features with StandardScaler
  4. Split 70% train / 15% val / 15% test
  5. Hyperparameter tuning via GridSearchCV on each model
  6. Evaluate all models on validation set
  7. Evaluate the best model on hold-out test set
  8. Generate comparison charts + confusion matrices
  9. Save best model, scaler, and label encoder with pickle

Usage:
  python train_model.py
"""

import os
import time
import pickle
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

from utils.feature_extractor import extract_features_from_path

warnings.filterwarnings("ignore")


# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
MODEL_DIR    = os.path.join(BASE_DIR, "model")
CHARTS_DIR   = os.path.join(BASE_DIR, "static", "charts")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


# ─── Step 1: Load Dataset ─────────────────────────────────────────────────────

def load_dataset(dataset_dir: str):
    """
    Walk every sub-directory of dataset_dir. Each sub-directory name is
    treated as the class label. Images are loaded and features extracted.

    Returns:
        X  : numpy array of shape (n_samples, n_features)
        y  : list of string class labels, length n_samples
        classes : sorted list of unique class names
    """
    X, y = [], []
    classes = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )

    if not classes:
        raise FileNotFoundError(
            f"No class sub-directories found in '{dataset_dir}'.\n"
            "Run: python generate_sample_data.py"
        )

    print(f"\nFound classes: {classes}\n")
    print("Extracting features ...")

    for cls in classes:
        cls_dir    = os.path.join(dataset_dir, cls)
        img_files  = [f for f in os.listdir(cls_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        print(f"  [{cls}]  {len(img_files)} images", end="", flush=True)

        for fname in img_files:
            fpath = os.path.join(cls_dir, fname)
            feats = extract_features_from_path(fpath)
            if feats is not None:
                X.append(feats)
                y.append(cls)

        print(f"  -> {sum(1 for label in y if label == cls)} features extracted")

    return np.array(X, dtype=np.float64), y, classes


# ─── Step 2: Hyperparameter Grids ─────────────────────────────────────────────

PARAM_GRIDS = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth":    [None, 10, 20],
            "min_samples_split": [2, 5],
        },
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            "C":      [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma":  ["scale", "auto"],
        },
    },
    "KNN": {
        "model": KNeighborsClassifier(n_jobs=-1),
        "params": {
            "n_neighbors": [3, 5, 7, 11],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan"],
        },
    },
}


# ─── Step 3: Train with GridSearch ────────────────────────────────────────────

def train_models(X_train, y_train, cv_folds=5):
    """
    Run GridSearchCV for each classifier.

    Returns:
        dict: {model_name: best_estimator}
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, cfg in PARAM_GRIDS.items():
        print(f"\n  Tuning {name} (GridSearchCV, cv={cv_folds}) ...")
        t0 = time.time()

        gs = GridSearchCV(
            estimator  = cfg["model"],
            param_grid = cfg["params"],
            cv         = cv,
            scoring    = "f1_weighted",
            n_jobs     = -1,
            verbose    = 0,
        )
        gs.fit(X_train, y_train)

        elapsed = time.time() - t0
        print(f"    Best params : {gs.best_params_}")
        print(f"    CV F1       : {gs.best_score_:.4f}  ({elapsed:.1f}s)")

        results[name] = gs.best_estimator_

    return results


# ─── Step 4: Evaluate on a Split ──────────────────────────────────────────────

def evaluate_model(model, X, y, split_name="Validation"):
    """
    Compute classification metrics for `model` on data (X, y).

    Returns:
        dict with keys: accuracy, precision, recall, f1
    """
    y_pred = model.predict(X)
    metrics = {
        "accuracy":  accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y, y_pred, average="weighted", zero_division=0),
    }
    print(f"\n  [{split_name}] Accuracy={metrics['accuracy']:.4f}  "
          f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
          f"F1={metrics['f1']:.4f}")
    return metrics, y_pred


# ─── Step 5: Plot Comparison Chart ────────────────────────────────────────────

def plot_comparison(eval_results: dict, classes: list, save_dir: str):
    """
    Generate and save:
      - Bar chart comparing all models on the 4 metrics
      - Confusion matrix heatmap for each model
    """
    # ── Bar chart ────────────────────────────────────────────────────────────
    model_names = list(eval_results.keys())
    metrics     = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    x   = np.arange(len(metrics))
    w   = 0.22
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    colours = ["#e94560", "#0f3460", "#533483"]
    for i, (name, data) in enumerate(eval_results.items()):
        vals = [data["metrics"][m] for m in metrics]
        bars = ax.bar(x + i * w, vals, w, label=name, color=colours[i], alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    color="white", fontweight="bold")

    ax.set_xticks(x + w)
    ax.set_xticklabels(metric_labels, color="white", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color="white", fontsize=12)
    ax.set_title("Model Comparison - Validation Set", color="white", fontsize=14, pad=15)
    ax.legend(facecolor="#0f3460", labelcolor="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    comp_path = os.path.join(save_dir, "comparison.png")
    plt.savefig(comp_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Comparison chart saved -> {comp_path}")

    # ── Confusion matrices ────────────────────────────────────────────────────
    for name, data in eval_results.items():
        cm = confusion_matrix(data["y_true"], data["y_pred"], labels=classes)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, linewidths=0.5, linecolor="#333",
                    annot_kws={"size": 12, "weight": "bold"})

        ax.set_title(f"Confusion Matrix - {name}", color="white", fontsize=13, pad=12)
        ax.set_xlabel("Predicted", color="white", fontsize=11)
        ax.set_ylabel("Actual", color="white", fontsize=11)
        ax.tick_params(colors="white")
        plt.tight_layout()

        safe_name = name.replace(" ", "_").lower()
        cm_path   = os.path.join(save_dir, f"cm_{safe_name}.png")
        plt.savefig(cm_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Confusion matrix saved -> {cm_path}")


# ─── Step 6: Save Artefacts ───────────────────────────────────────────────────

def save_artifacts(model, scaler, label_encoder, model_name: str):
    """Persist model, scaler, and label encoder to disk using pickle."""
    objects = {
        "best_model.pkl":    model,
        "scaler.pkl":        scaler,
        "label_encoder.pkl": label_encoder,
    }
    for fname, obj in objects.items():
        path = os.path.join(MODEL_DIR, fname)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved: {path}")

    # also save the winning model name for reference
    info_path = os.path.join(MODEL_DIR, "model_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Best model: {model_name}\n")
    print(f"  Model info: {info_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("   Plant Disease Detection - Model Training")
    print("=" * 60)

    # 1. Load dataset
    X, y_labels, classes = load_dataset(DATASET_DIR)
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features, "
          f"{len(classes)} classes")

    # 2. Encode string labels to integers
    le = LabelEncoder()
    y  = le.fit_transform(y_labels)

    # 3. Scale features
    print("\nScaling features with StandardScaler ...")
    scaler = StandardScaler()

    # First split off the test set (15 %), then split the remainder 70/15
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.15 / 0.85,   # gives ~15 % of total
        random_state=42, stratify=y_trainval
    )

    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 4. Train all models
    print("\nTraining models with GridSearchCV ...")
    trained_models = train_models(X_train_s, le.inverse_transform(y_train))

    # 5. Evaluate on validation set
    print("\n" + "=" * 60)
    print("  Validation Set Results")
    print("=" * 60)

    eval_results = {}
    for name, model in trained_models.items():
        print(f"\n--- {name} ---")
        metrics, y_pred = evaluate_model(
            model, X_val_s, le.inverse_transform(y_val), split_name="Val"
        )
        eval_results[name] = {
            "metrics": metrics,
            "y_true":  le.inverse_transform(y_val),
            "y_pred":  y_pred,
        }
        # Full report
        print(classification_report(
            le.inverse_transform(y_val), y_pred,
            target_names=classes, zero_division=0
        ))

    # 6. Select best model by F1
    best_name  = max(eval_results, key=lambda n: eval_results[n]["metrics"]["f1"])
    best_model = trained_models[best_name]
    print(f"\n{'='*60}")
    print(f"  Best model: {best_name} "
          f"(Val F1 = {eval_results[best_name]['metrics']['f1']:.4f})")

    # 7. Final evaluation on hold-out test set
    print(f"\n--- Test Set Evaluation ({best_name}) ---")
    test_metrics, y_test_pred = evaluate_model(
        best_model, X_test_s, le.inverse_transform(y_test), split_name="Test"
    )
    print(classification_report(
        le.inverse_transform(y_test), y_test_pred,
        target_names=classes, zero_division=0
    ))

    # 8. Generate charts
    print("\nGenerating charts ...")
    plot_comparison(eval_results, classes, CHARTS_DIR)

    # 9. Save artefacts
    print("\nSaving model artefacts ...")
    save_artifacts(best_model, scaler, le, best_name)

    print("\n[OK] Training complete! Run 'python app.py' to start the web application.")
    print("=" * 60)


if __name__ == "__main__":
    main()
