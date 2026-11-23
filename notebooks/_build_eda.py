"""Generates 01_eda.ipynb. (Edit this script and re-run it instead of editing the notebook by hand.)"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 01 · Exploratory Data Analysis (EDA)\n\n"
       "Dataset: [`winvoker/turkish-sentiment-analysis-dataset`](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset)\n\n"
       "Goal: understand the data before building a model. Are the classes balanced? How long are the texts? "
       "Is there anything in the data that could mislead a model?\n\n"
       "Note: the label `Notr` is the dataset's spelling of **Neutral**."),
    code("import sys; sys.path.insert(0, '..')\n"
         "import pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n"
         "from src.data import download_raw, clean\n"
         "sns.set_theme(style='whitegrid')\n"
         "raw_train, raw_test = download_raw()\n"
         "print('Raw size   -> train:', raw_train.shape, ' test:', raw_test.shape)\n"
         "train, test = clean(raw_train), clean(raw_test)\n"
         "print('Clean size -> train:', train.shape, ' test:', test.shape)\n"
         "train.head()"),
    md("## 1. Class distribution\n"
       "With imbalanced classes **accuracy is misleading**: if 70% of the data is one class, a model that always predicts "
       "that class gets 70% accuracy. That's why we also report **macro F1** (the unweighted mean of per-class F1 scores)."),
    code("fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
         "for a, (name, df) in zip(ax, [('train', train), ('test', test)]):\n"
         "    vc = df['label'].value_counts()\n"
         "    sns.barplot(x=vc.index, y=vc.values, ax=a, hue=vc.index, legend=False)\n"
         "    a.set_title(f'{name} class distribution'); a.set_ylabel('number of examples')\n"
         "    for i, v in enumerate(vc.values): a.text(i, v, f'{100*v/len(df):.1f}%', ha='center', va='bottom')\n"
         "plt.tight_layout(); plt.show()"),
    md("## 2. Where do the texts come from? (`dataset` column)\n"
       "The dataset combines several sources: product reviews (`urun_yorumlari`), store reviews (`magaza_yorumlari`), "
       "movie reviews (`HUMIR`), tweets (`tweet-pn`), Wikipedia sentences (`wiki`) and filler text (`random`). "
       "Let's look at how source and label relate."),
    code("pd.crosstab(train['dataset'], train['label'], margins=True)"),
    md("**Key observation:** the `Notr` (Neutral) class comes entirely from `wiki` (encyclopedia sentences) and `random`. "
       "No review or tweet is labeled Neutral. So instead of learning \"neutral sentiment\", a model can learn "
       "\"is this an encyclopedia sentence?\". This is called **shortcut learning**. The test set has the same structure, "
       "so scores look high, but in real use (e.g. a neutral product review) the model may fail. We keep this in mind "
       "when interpreting the results and also report a separate F1 for the Negative and Positive classes."),
    code("for src in train['dataset'].unique():\n"
         "    print(f'--- {src} ---')\n"
         "    for t in train[train['dataset'] == src].sample(min(3, (train['dataset'] == src).sum()), random_state=0)['text']: print('  •', t[:150])"),
    md("## 3. Text lengths\n"
       "Does BERT's `max_length=128` token limit cover most texts?"),
    code("train['n_chars'] = train['text'].str.len()\n"
         "train['n_words'] = train['text'].str.split().str.len()\n"
         "print(train.groupby('label')[['n_chars', 'n_words']].describe().round(1).T)\n"
         "fig, ax = plt.subplots(1, 2, figsize=(13, 4))\n"
         "sns.histplot(data=train, x='n_words', hue='label', bins=60, binrange=(0, 120), element='step', stat='density', common_norm=False, ax=ax[0])\n"
         "ax[0].set_title('Word count distribution (by class)'); ax[0].set_xlabel('words per text')\n"
         "sns.boxplot(data=train, x='dataset', y='n_words', showfliers=False, ax=ax[1]); ax[1].set_title('Word count by source')\n"
         "ax[1].set_ylabel('words per text'); ax[1].tick_params(axis='x', rotation=20)\n"
         "plt.tight_layout(); plt.show()"),
    code("from transformers import AutoTokenizer\n"
         "from src.config import BERT_NAME, MAX_LENGTH\n"
         "tok = AutoTokenizer.from_pretrained(BERT_NAME)\n"
         "s = train.sample(3000, random_state=0)\n"
         "n_tok = s['text'].map(lambda t: len(tok(t)['input_ids']))\n"
         "print(n_tok.describe().round(1))\n"
         "print(f'Share of texts longer than {MAX_LENGTH} tokens: {100*(n_tok > MAX_LENGTH).mean():.2f}%')\n"
         "# 'Shipping was very late, I wasn't satisfied at all.'\n"
         "print('Example tokenization:', tok.tokenize('Kargo çok geç geldi, hiç memnun kalmadım.'))"),
    md("BERTurk uses a sub-word (WordPiece) tokenizer: words it doesn't know as a whole are split into pieces starting with `##`. "
       "This lets the model learn Turkish suffixes as meaningful units (e.g. `kalmadım` → `kalmadı` + `##m`, where `##m` is the first-person suffix: \"didn't remain\" + \"I\")."),
    md("## 4. Duplicates and leakage check"),
    code("print('Duplicate texts in train:', raw_train['text'].duplicated().sum())\n"
         "overlap = set(train['text']) & set(test['text'])\n"
         "print('Texts appearing in both train and test:', len(overlap))\n"
         "print('-> src/data.py removes these texts from the test set.')"),
    md("## 5. Working subset\nThe small stratified subset created by `src/data.py`:"),
    code("from src.data import load_splits\n"
         "tr, va, te = load_splits()\n"
         "pd.DataFrame({n: d['label'].value_counts() for n, d in [('train', tr), ('val', va), ('test', te)]})"),
    md("## Summary\n"
       "- The classes are imbalanced (Negative ≈ 11.5%) → we use **macro F1** and `class_weight='balanced'` in the baseline.\n"
       "- Labels are strongly tied to the source (`dataset`) → watch out for shortcut learning.\n"
       "- The vast majority of texts are under 128 tokens → `max_length=128` is enough.\n"
       "- Train–test overlaps were removed → leakage-free evaluation."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, Path(__file__).with_name("01_eda.ipynb"))
print("01_eda.ipynb written")
