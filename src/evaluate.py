"""Compares the two models: metrics table, confusion matrices and examples for error analysis.

Usage:  python -m src.evaluate
"""
import json

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from src.config import LABELS, REPORTS_DIR

matplotlib.use("Agg")
MODELS = {"TF-IDF + LogReg": "preds_baseline.csv", "BERTurk": "preds_bert.csv"}
# Class names shown in plots (the dataset spells "Neutral" as "Notr")
DISPLAY_NAMES = {"Negative": "Negative", "Notr": "Neutral", "Positive": "Positive"}


def load_preds() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(REPORTS_DIR / f) for name, f in MODELS.items()}


def results_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    times = {"TF-IDF + LogReg": "metrics_baseline.json", "BERTurk": "metrics_bert.json"}
    rows = []
    for name, df in preds.items():
        m = json.loads((REPORTS_DIR / times[name]).read_text())
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(df["label"], df["pred"]),
            "Macro F1": f1_score(df["label"], df["pred"], average="macro"),
            # Separate F1 for the two real sentiment classes (excluding Neutral)
            "F1 (Neg+Pos)": f1_score(df["label"], df["pred"], labels=["Negative", "Positive"],
                                     average="macro"),
            "Train time (s)": m["train_time_sec"],
        })
    return pd.DataFrame(rows)


def plot_confusion(preds: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(1, len(preds), figsize=(6 * len(preds), 5))
    for ax, (name, df) in zip(axes, preds.items()):
        cm = confusion_matrix(df["label"], df["pred"], labels=LABELS)
        display_labels = [DISPLAY_NAMES[label] for label in LABELS]
        ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrices.png", dpi=120)
    plt.close(fig)


def error_examples(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """The n misclassified examples the model was most confident about.
    A confident mistake usually points to something interesting in the data."""
    wrong = df[df["label"] != df["pred"]].sort_values("confidence", ascending=False)
    return wrong.head(n)[["text", "label", "pred", "confidence", "dataset"]].reset_index(drop=True)


def main():
    preds = load_preds()
    table = results_table(preds)
    table.round(4).to_csv(REPORTS_DIR / "results.csv", index=False)
    print(table.round(4).to_markdown(index=False))

    for name, df in preds.items():
        print(f"\n=== {name} ===")
        print(classification_report(df["label"], df["pred"], labels=LABELS, digits=4))
        acc_by_src = df.assign(ok=df["label"] == df["pred"]).groupby("dataset")["ok"].mean()
        print("Accuracy by source:\n",acc_by_src.round(4).to_string())

    plot_confusion(preds)
    errors = error_examples(preds["BERTurk"])
    errors.to_csv(REPORTS_DIR / "errors_bert.csv", index=False)
    print(f"\nBERTurk misclassified examples: {(preds['BERTurk']['label'] != preds['BERTurk']['pred']).sum()}")


if __name__ == "__main__":
    main()
