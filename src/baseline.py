"""Baseline: TF-IDF + Logistic Regression.

Idea: each text becomes a vector of TF-IDF weights of its word and character pieces (n-grams).
We then train a linear classifier on these vectors.
Character n-grams help with agglutinative languages like Turkish: "güzel", "güzeldi" and
"güzelmiş" are different words but share the piece "güzel" ("beautiful").

Usage:  python -m src.baseline
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import MODELS_DIR, REPORTS_DIR, SEED
from src.data import load_splits


def build_pipeline(C: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    # class_weight="balanced": mistakes on rare classes cost more.
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=SEED)
    return Pipeline([("tfidf", features), ("clf", clf)])


def main():
    train, val, test = load_splits()

    # Small hyperparameter search: C is the inverse of regularization strength (larger C = less penalty).
    # We pick the best C on the validation set; the test set is not touched at this stage.
    best_C, best_f1 = None, -1.0
    for C in (0.5, 1.0, 2.0, 4.0, 8.0):
        pipe = build_pipeline(C).fit(train["text"], train["label"])
        f1 = f1_score(val["label"], pipe.predict(val["text"]), average="macro")
        print(f"C={C:<4} val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_C, best_f1 = C, f1

    t0 = time.time()
    pipe = build_pipeline(best_C).fit(train["text"], train["label"])
    train_time = time.time() - t0

    proba = pipe.predict_proba(test["text"])
    classes = list(pipe.classes_)
    pred = np.array(classes)[proba.argmax(1)]

    out = test[["text", "label", "dataset"]].copy()
    out["pred"] = pred
    out["confidence"] = proba.max(1)
    for i, c in enumerate(classes):
        out[f"p_{c}"] = proba[:, i]
    out.to_csv(REPORTS_DIR / "preds_baseline.csv", index=False)

    metrics = {
        "model": "TF-IDF + LogReg",
        "best_C": best_C,
        "accuracy": accuracy_score(test["label"], pred),
        "f1_macro": f1_score(test["label"], pred, average="macro"),
        "train_time_sec": round(train_time, 1),
        "n_train": len(train),
    }
    (REPORTS_DIR / "metrics_baseline.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(pipe, MODELS_DIR / "baseline.joblib")

    # Interpretability: the highest-weighted features for each class.
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_
    top = {c: [names[j].split("__", 1)[1] for j in np.argsort(coefs[i])[-15:][::-1]]
           for i, c in enumerate(classes)}
    pd.DataFrame(top).to_csv(REPORTS_DIR / "baseline_top_features.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
