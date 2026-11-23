"""Hand-written notes on BERTurk's 20 most confident errors.

The list must follow the same order as reports/errors_bert.csv. Retraining the model changes
the errors, so these notes would then need to be rewritten.

Usage:  python -m src.error_comments   ->  reports/error_analysis.csv
"""
import pandas as pd

from src.config import REPORTS_DIR

MIXED = "Mixed sentiment"
LABEL = "Questionable label"
INDIRECT = "Indirect statement / comparison"
NEGATION = "Negation / very short text"
SHORTCUT = "Source shortcut (Neutral = encyclopedic style)"

COMMENTS = [
    (MIXED, "Criticizes quality but praises price and shipping. The label probably comes from the star rating; the positive words in the text ('worth choosing', 'no problems', 'thanks') dominate."),
    (LABEL, "'Turned into torture' is clearly negative, yet the label is Positive -> label error. The model said Neutral: a short headline-like sentence resembles the Wikipedia style."),
    (MIXED, "'small freezes ... but a good product': complaint and praise together; the model weighted the 'good product' part."),
    (MIXED, "'Not bad' (mildly positive) plus a suggestion to use an alternative. The negativity is indirect; the label is debatable."),
    (INDIRECT, "The negativity comes through a comparison with another product ('its sound quality is better', 'don't be fooled'). The positive words describe a different product; the model could not resolve that reference."),
    (MIXED, "Starts positive ('a decent towel, it dries your hands :)'); the criticism is phrased as a recommendation ('I'd recommend a better towel'). 'I recommend' is also one of the baseline's strongest Positive features."),
    (SHORTCUT, "A short, emotion-free factual sentence ending with a period; in the training data this style only appears in the Neutral (Wikipedia) class. The label is also debatable (Positive, but the sentence is neutral)."),
    (NEGATION, "Only two words: 'low quality'. 'Quality' appears very often in positive product reviews; the modifier 'low' was not learned from so little data. A short text gives no context."),
    (MIXED, "Praises the shipping first ('they packed it great'), then describes a product defect. The model was swayed by the positive opening."),
    (LABEL, "The text explains what the screen protector protects against, with no clear negativity -> the label probably comes from a low rating; the text alone does not support Negative."),
    (MIXED, "Lists problems but ends with 'the other aspects are great, it never slows down the device'. Which side outweighs the other is subjective."),
    (MIXED, "'The movie was good, recommended' + 'the first movie was better'. The overall tone is positive; the Negative label may come from the rating. It also has typos."),
    (LABEL, "'I bought the bed and set it up as instructed.' No sentiment; the review looks cut off. Negative cannot be inferred from the text."),
    (LABEL, "Rating-text mismatch: says 'one of the best animations' but 'I gave it 2 points'. Since the label comes from the rating, the model is arguably right about the text."),
    (MIXED, "Long story: starts with 'they are very fast' (positive), the real complaint comes at the end. The model weighted the positive start and did not follow the narrative."),
    (LABEL, "'Not nice at all' is clearly negative, yet labeled Positive -> most likely a tweet labeling error."),
    (INDIRECT, "Positive statement of trust in the website, but built from negative words such as 'fake', 'impossible', 'I won't even'. Broken Unicode (dotted i) also makes tokenization harder."),
    (MIXED, "'Below standard but the sound is quite loud': the part after 'but' is positive; in Turkish the clause after 'ama' (but) usually carries the emphasis, and the model read it that way. Debatable label."),
    (LABEL, "A tweet with political criticism and insults is labeled Positive -> label error, or an unclear definition of 'sentiment' (is the author's own opinion counted as positive?)."),
    (INDIRECT, "'Masterpiece' is positive, but the praise is indirect, using negative words ('anyone who hasn't watched it made a big mistake'). The model latched onto 'big mistake'."),
]


def main():
    err = pd.read_csv(REPORTS_DIR / "errors_bert.csv")
    assert len(err) == len(COMMENTS), "Number of notes does not match number of errors"
    err["category"] = [c[0] for c in COMMENTS]
    err["note"] = [c[1] for c in COMMENTS]
    err.to_csv(REPORTS_DIR / "error_analysis.csv", index=False)
    print(err["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
