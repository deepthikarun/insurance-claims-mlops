# train.py — train offline (CI/CD) and save model.pkl for serving
import os, joblib, mlflow, mlflow.sklearn, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Shared preprocessing
from preprocess import preprocess_raw_df, _build_make_map, make_map

CSV_PATH = os.getenv("CSV_PATH", "fraud_oracle.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
TARGET_COL = os.getenv("TARGET_COL", "FraudFound_P")

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    df_raw = pd.read_csv(CSV_PATH)

    # Build make_map from training data
    if "Make" in df_raw.columns:
        make_map.clear()
        make_map.update(_build_make_map(df_raw["Make"]))

    df = preprocess_raw_df(df_raw)
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"Target column '{TARGET_COL}' not found after preprocessing")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # MLflow logging
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name="ci-train-rf"):
        mlflow.log_params({
            "n_estimators": clf.n_estimators,
            "class_weight": "balanced",
            "random_state": 42
        })
        mlflow.log_metric("train_score", clf.score(X_train, y_train))
        mlflow.log_metric("test_score", clf.score(X_test, y_test))
        mlflow.sklearn.log_model(clf, artifact_path="model")

    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Saved trained model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
