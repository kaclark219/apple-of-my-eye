import os, sys, joblib
from pathlib import Path
from sklearn.metrics import classification_report
from src.data_utils import load_split
from src.train_models import make_bagged_knn, make_random_forest, make_gradient_boosting

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

for f in ["train.txt", "val.txt", "test.txt"]:
    p = SPLITS / f
    if not p.exists():
        sys.exit(f"Missing split file: {p}. Run your prepare script first.")

print("Loading features…")
X_train, y_train = load_split(SPLITS / "train.txt")
X_val,   y_val   = load_split(SPLITS / "val.txt")
X_test,  y_test  = load_split(SPLITS / "test.txt")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

models = {
    "bagged_knn": make_bagged_knn(),
    "random_forest": make_random_forest(),
    "gradient_boosting": make_gradient_boosting(),
}

best_name, best_model, best_score = None, None, -1.0

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"{name} val accuracy: {score:.3f}")
    out = MODELS / f"{name}.joblib"
    joblib.dump(model, out)
    print(f"Saved: {out}")
    if score > best_score:
        best_name, best_model, best_score = name, model, score

print(f"\nBest on val: {best_name} ({best_score:.3f}) — evaluating on TEST…")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

best_path = MODELS / "best.joblib"
joblib.dump(best_model, best_path)
print(f"Wrote best model to: {best_path}")