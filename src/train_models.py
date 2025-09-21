# src/train_models.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

def make_bagged_knn():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("bag", BaggingClassifier(
            estimator=KNeighborsClassifier(n_neighbors=5),
            n_estimators=30,
            max_samples=0.8,
            bootstrap=True,
            n_jobs=-1,
            random_state=42))
    ])

def make_random_forest():
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )

def make_gradient_boosting():
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=250,
        validation_fraction=0.15,
        early_stopping=True,
        n_iter_no_change=20,
        l2_regularization=1e-3,
        random_state=42
    )
