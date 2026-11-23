"""Generates 02_results.ipynb (after running src.baseline, src.train_bert and src.evaluate)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 02 · Model Comparison and Error Analysis\n\n"
       "Both models are compared on the **same 1,000-example test set**:\n"
       "1. **TF-IDF + Logistic Regression** (baseline)\n"
       "2. **BERTurk** (`dbmdz/bert-base-turkish-cased`, fine-tuned)\n\n"
       "Predictions are read from `reports/preds_*.csv`; the models are not re-run here. "
       "The label `Notr` is the dataset's spelling of **Neutral**."),
    code("import sys; sys.path.insert(0, '..')\n"
         "import pandas as pd\n"
         "from IPython.display import Image, display\n"
         "from sklearn.metrics import classification_report\n"
         "from src.config import LABELS, REPORTS_DIR\n"
         "from src.evaluate import load_preds, results_table\n"
         "pd.set_option('display.max_colwidth', 200)\n"
         "preds = load_preds()\n"
         "results_table(preds).round(4)"),
    md("**Metrics:**\n"
       "- **Accuracy:** share of correct predictions. On imbalanced data it can be inflated by the majority class.\n"
       "- **Macro F1:** unweighted mean of per-class F1 scores; the minority class (Negative) counts equally.\n"
       "- **F1 (Neg+Pos):** only the real sentiment classes. As the EDA showed, the Neutral class consists solely of "
       "Wikipedia sentences and is therefore easy; this metric shows how well the model actually separates sentiment."),
    code("for name, df in preds.items():\n"
         "    print(f'=== {name} ===')\n"
         "    print(classification_report(df['label'], df['pred'], labels=LABELS, digits=3))"),
    md("## Confusion matrices\nRows are the true class, columns the prediction. Off-diagonal cells are errors."),
    code("display(Image(filename=str(REPORTS_DIR / 'confusion_matrices.png')))"),
    md("## Accuracy by source\nWhich kinds of text are hard for the models?"),
    code("pd.DataFrame({name: df.assign(ok=df['label'] == df['pred']).groupby('dataset')['ok'].mean()\n"
         "              for name, df in preds.items()}).assign(n=preds['BERTurk']['dataset'].value_counts()).round(3)"),
    md("## Where do the two models disagree?"),
    code("b, t = preds['TF-IDF + LogReg'], preds['BERTurk']\n"
         "b_ok, t_ok = b['label'] == b['pred'], t['label'] == t['pred']\n"
         "print('Both correct         :', (b_ok & t_ok).sum())\n"
         "print('Only BERTurk correct :', (~b_ok & t_ok).sum())\n"
         "print('Only baseline correct:', (b_ok & ~t_ok).sum())\n"
         "print('Both wrong           :', (~b_ok & ~t_ok).sum())\n"
         "t.assign(baseline_pred=b['pred'])[~b_ok & t_ok][['text', 'label', 'baseline_pred', 'pred']].head(8)"),
    md("## Error analysis: 20 BERTurk mistakes\n"
       "We take the misclassified examples sorted by the model's **confidence, highest first**. Confident mistakes are "
       "usually the most informative: either the label is wrong, or the model is missing something systematic. "
       "The `note` column explains the likely cause for each example (the texts themselves are in Turkish)."),
    code("err = pd.read_csv(REPORTS_DIR / 'error_analysis.csv')\n"
         "err[['text', 'label', 'pred', 'confidence', 'dataset', 'category', 'note']]"),
    code("err['category'].value_counts()"),
    md("## Conclusions\n"
       "- **Almost half of the confident errors are mixed-sentiment reviews** (e.g. \"shipping was great but the product is bad\"). "
       "The labels likely come from star ratings, so the correct answer is debatable; aspect-based sentiment would fit better.\n"
       "- **About a third have questionable labels**, so the model's true error rate is somewhat lower than reported.\n"
       "- The model **over-weights positive phrases at the start** of a review and **cannot handle irony**.\n"
       "- The Neutral class is solved almost perfectly only because of the **Wikipedia shortcut** described in the EDA."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, Path(__file__).with_name("02_results.ipynb"))
print("02_results.ipynb written")
