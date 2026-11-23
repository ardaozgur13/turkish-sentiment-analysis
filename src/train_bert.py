"""BERTurk (dbmdz/bert-base-turkish-cased) fine-tuning.

Idea: BERTurk is a language model pre-trained on a large Turkish text corpus.
We add a small 3-output classification head on top and train the whole model on our data
for a few epochs (fine-tuning). Since the model already knows the language, it does well
even with little data.

Usage:  python -m src.train_bert [--epochs 2] [--batch-size 16]
"""
import argparse
import json
import time

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from src.config import (BERT_NAME, ID2LABEL, LABEL2ID, LABELS, MAX_LENGTH, MODELS_DIR,
                        REPORTS_DIR, SEED)
from src.data import load_splits

OUT_DIR = MODELS_DIR / "berturk"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro")}


def main(epochs: int, batch_size: int, lr: float):
    set_seed(SEED)
    use_gpu = torch.cuda.is_available()
    print(f"Device: {'GPU - ' + torch.cuda.get_device_name(0) if use_gpu else 'CPU'}")

    train, val, test = load_splits()
    tok = AutoTokenizer.from_pretrained(BERT_NAME)

    def to_ds(df):
        ds = Dataset.from_dict({"text": df["text"].tolist(),
                                "label": df["label"].map(LABEL2ID).tolist()})
        # Tokenization: splits text into sub-word pieces (WordPiece), then maps them to IDs.
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LENGTH),
                      batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = to_ds(train), to_ds(val), to_ds(test)

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    args = TrainingArguments(
        output_dir=str(MODELS_DIR / "berturk_ckpt"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,              # small LR for fine-tuning, so pre-trained knowledge isn't destroyed
        weight_decay=0.01,
        warmup_steps=0.1,              # (float = ratio) ramp LR up over the first 10% of steps -> more stable start
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,   # keep the epoch with the best validation macro-F1
        metric_for_best_model="f1_macro",
        train_sampling_strategy="group_by_length",  # batch texts of similar length -> less padding, faster
        fp16=use_gpu,
        logging_steps=25,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    trainer.save_model(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))
    write_test_predictions(trainer.model, tok, test, train_time, len(train), epochs, use_gpu)


@torch.no_grad()
def predict_proba(model, tok, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Predicts texts IN THE GIVEN ORDER.
    Note: we don't use trainer.predict(), because the group_by_length sampler can reorder
    examples, which matched predictions to the wrong rows (this happened in the first run)."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=MAX_LENGTH,
                  padding=True, return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits, dim=-1).cpu())
    return torch.cat(out).numpy()


def write_test_predictions(model, tok, test, train_time, n_train, epochs, use_gpu):
    proba = predict_proba(model, tok, test["text"].tolist())
    pred_ids = proba.argmax(1)

    out = test[["text", "label", "dataset"]].copy()
    out["pred"] = [ID2LABEL[i] for i in pred_ids]
    out["confidence"] = proba.max(1)
    for i, c in enumerate(LABELS):
        out[f"p_{c}"] = proba[:, i]
    out.to_csv(REPORTS_DIR / "preds_bert.csv", index=False)

    y_true = test["label"].map(LABEL2ID).to_numpy()
    metrics = {
        "model": "BERTurk (fine-tuned)",
        "accuracy": accuracy_score(y_true, pred_ids),
        "f1_macro": f1_score(y_true, pred_ids, average="macro"),
        "train_time_sec": round(train_time, 1),
        "n_train": n_train,
        "epochs": epochs,
        "device": "GPU" if use_gpu else "CPU",
    }
    (REPORTS_DIR / "metrics_bert.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def predict_only():
    """Loads the trained model from disk and only regenerates test predictions (no training)."""
    train, _, test = load_splits()
    old = json.loads((REPORTS_DIR / "metrics_bert.json").read_text())
    tok = AutoTokenizer.from_pretrained(OUT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR)
    write_test_predictions(model, tok, test, old["train_time_sec"], len(train), old["epochs"],
                           torch.cuda.is_available())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--predict-only", action="store_true",
                   help="Skip training; load models/berturk and write test predictions")
    a = p.parse_args()
    if a.predict_only:
        predict_only()
    else:
        main(a.epochs, a.batch_size, a.lr)
