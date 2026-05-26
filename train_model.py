"""Train a compact rainfall model (same features as the app)."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["state_code", "year", "month", "day", "normal"]
MODEL_PATH = Path(__file__).resolve().parent / "rf_model.pkl"
LEGACY_PATH = Path(__file__).resolve().parent / "rf_model_legacy.pkl"
N_SAMPLES = 12_000


def load_teacher():
    path = LEGACY_PATH if LEGACY_PATH.exists() else MODEL_PATH
    with open(path, "rb") as f:
        return pickle.load(f)


def build_dataset(teacher, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    for _ in range(N_SAMPLES):
        rows.append(
            [
                int(rng.integers(1, 37)),
                int(rng.integers(2010, 2027)),
                int(rng.integers(1, 13)),
                int(rng.integers(1, 29)),
                float(rng.uniform(0.5, 120.0)),
            ]
        )
    x = pd.DataFrame(rows, columns=FEATURES)
    y = teacher.predict(x)
    return x, y


def main():
    teacher = load_teacher()
    rng = np.random.default_rng(42)
    x, y = build_dataset(teacher, rng)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print(f"R² vs teacher: {r2_score(y_test, pred):.3f}")
    print(f"MAE vs teacher: {mean_absolute_error(y_test, pred):.2f} mm")

    import joblib

    joblib.dump(model, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"Saved {MODEL_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
