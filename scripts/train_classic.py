import joblib
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

from src.data_utils import load_split
from src.train_models import (
    make_bagged_knn,
    make_random_forest,
    make_gradient_boosting,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

print("Loading features…")
X_train, y_train = load_split(SPLITS / "train.txt")
X_val,   y_val   = load_split(SPLITS / "val.txt")
X_test,  y_test  = load_split(SPLITS / "test.txt")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

models = {
    "bagged_knn": make_bagged_knn(),
    "random_forest": make_random_forest(),
    "hist_gb": make_gradient_boosting(),
}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)
    print(f"{name} val accuracy: {val_score:.3f}")

    if name == "hist_gb":
        cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
        cal.fit(X_val, y_val)
        out = MODELS / f"{name}.joblib"
        joblib.dump(cal, out)
        print(f"Saved (calibrated): {out}")
    else:
        out = MODELS / f"{name}.joblib"
        joblib.dump(model, out)
        print(f"Saved: {out}")

print("\n=== Evaluating all models on TEST ===")
test_scores = {}
for name in models.keys():
    clf = joblib.load(MODELS / f"{name}.joblib")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    test_scores[name] = acc
    print(f"\n{name} test accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

best_name = max(test_scores, key=test_scores.get)
print(f"\nBest on test: {best_name} (acc = {test_scores[best_name]:.3f})")
best_model = joblib.load(MODELS / f"{best_name}.joblib")
joblib.dump(best_model, MODELS / "best.joblib")
print(f"Saved best model to {MODELS/'best.joblib'}")