import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

DATA_PATH = Path(r"D:\Hackathon-Black Pearl\PTCG\train_features.csv")
MODEL_OUTPUT = Path(r"D:\Hackathon-Black Pearl\PTCG\xgb_model.json")
MAPPING_OUTPUT = Path(r"D:\Hackathon-Black Pearl\PTCG\action_label_mapping.json")


def load_and_preprocess_data(sample_limit: int = 2_000_000):
    print("Loading feature dataset...")
    df = pd.read_csv(DATA_PATH, nrows=sample_limit) if sample_limit else pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")

    action_counts = df["label_action_idx"].value_counts()
    valid_actions = action_counts[action_counts >= 50].index
    df = df[df["label_action_idx"].isin(valid_actions)].copy()

    X = df.drop(columns=["label_action_idx"])
    y = df["label_action_idx"]

    unique_labels = sorted(y.unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_mapped = y.map(label_to_idx)

    MAPPING_OUTPUT.write_text(
        json.dumps({"class_to_option": [int(v) for v in unique_labels]}),
        encoding="utf-8",
    )
    print(f"Class mapping saved to: {MAPPING_OUTPUT}")

    print(f"Effective Classes: {len(unique_labels)}, Feature Dimensions: {X.shape[1]}")
    return X, y_mapped, unique_labels


def main():
    X, y, class_mapping = load_and_preprocess_data(sample_limit=None)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    num_classes = len(class_mapping)
    print(f"Training set: {len(X_train):,} rows | Validation set: {len(X_val):,} rows")

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=num_classes,
        n_estimators=250,
        learning_rate=0.08,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    print("\nStarting LogLossBC Model Training...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=25,
    )

    print("\nEvaluating model performance...")
    val_probs = model.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)

    top1_acc = accuracy_score(y_val, val_preds)
    top3_acc = top_k_accuracy_score(y_val, val_probs, k=min(3, num_classes))

    print(f"Validation Top-1 Accuracy: {top1_acc * 100:.2f}%")
    print(f"Validation Top-3 Accuracy: {top3_acc * 100:.2f}%")

    model.save_model(MODEL_OUTPUT)
    print(f"\nModel successfully exported to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
