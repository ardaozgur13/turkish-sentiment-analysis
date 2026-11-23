"""Gradio demo: enter a sentence and see both models' predictions side by side.
The interface text is in Turkish, since the models work on Turkish input.

Usage:  python app.py   ->  open http://127.0.0.1:7860 in a browser
"""
import gradio as gr
import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import MAX_LENGTH, MODELS_DIR

baseline = joblib.load(MODELS_DIR / "baseline.joblib")
tok = AutoTokenizer.from_pretrained(MODELS_DIR / "berturk")
bert = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR / "berturk").eval()

# Turkish display names for the labels shown in the UI
TR = {"Negative": "Olumsuz", "Notr": "Nötr", "Positive": "Olumlu"}


def predict(text: str):
    if not text.strip():
        return {}, {}
    p_base = baseline.predict_proba([text])[0]
    base_out = {TR[c]: float(p) for c, p in zip(baseline.classes_, p_base)}

    with torch.no_grad():
        enc = tok(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        p_bert = torch.softmax(bert(**enc).logits, dim=-1)[0]
    bert_out = {TR[bert.config.id2label[i]]: float(p) for i, p in enumerate(p_bert)}
    return base_out, bert_out


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, label="Türkçe metin", placeholder="Bir yorum veya tweet yaz..."),
    outputs=[gr.Label(label="TF-IDF + Logistic Regression"), gr.Label(label="BERTurk (fine-tuned)")],
    title="Türkçe Duygu Analizi",
    description="Aynı metin için klasik baseline ile BERTurk modelinin tahminlerini karşılaştır.",
    examples=[
        ["Ürün çok kaliteli, kargo da hızlıydı. Herkese tavsiye ederim."],
        ["Hiç fena değil, fiyatına göre gayet iyi."],
        ["İki gün sonra bozuldu, param çöpe gitti."],
        ["Kargo berbattı ama ürün beklediğimden güzel çıktı."],
        ["Harika, yine kargom kayboldu. Tebrikler!"],
        ["İstanbul, Türkiye'nin en kalabalık şehridir."],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
