# Sentiment Analysis on Movie Reviews

A Python NLP project built for the i2i Academy Natural Language Processing
homework. It takes a dataset of English text reviews, cleans the text, scores
each review with a pre-trained sentiment analysis model, and labels it as
Positive, Negative, or Neutral.

## Overview

Human language is messy: unlike a column of numbers, a sentence carries tone,
context, and ambiguity. This project shows a small end-to-end pipeline for
turning raw text into a structured signal a business could actually use,
without anyone having to read every single review by hand.

Given a CSV of reviews, the pipeline:

1. Loads the data with pandas
2. Cleans each review (lowercasing, punctuation/number removal, stop word filtering)
3. Scores sentiment polarity with TextBlob
4. Labels each review as Positive, Negative, or Neutral
5. Prints summary statistics and saves a distribution chart
6. Optionally checks its own accuracy against ground-truth labels

## Dataset

The dataset is the `movie_reviews` corpus bundled with NLTK: 2000 English
movie reviews, each already labeled `positive` or `negative` by the corpus
authors. It's included in this repo as `data/reviews.csv` with two columns:

| column       | description                                  |
|--------------|-----------------------------------------------|
| `review`     | the raw English review text                   |
| `true_label` | the ground-truth label (`positive`/`negative`) |

The CSV is committed to the repo on purpose, so the project runs immediately
after cloning with no extra download step. `src/data_loader.py` can also
rebuild this file from scratch from the NLTK corpus if it's ever missing.

## Project structure

```
i2i-Academy-NLP-11/
├── data/
│   └── reviews.csv                  # the review dataset
├── src/
│   ├── data_loader.py               # builds/loads the review dataset
│   ├── text_cleaner.py              # text cleaning + stop word removal
│   ├── sentiment.py                 # TextBlob polarity scoring and labeling
│   └── visualize.py                 # sentiment distribution chart
├── tests/
│   └── test_text_cleaner.py         # unit tests for the cleaning function
├── outputs/                          # generated CSV + chart (created on run)
├── main.py                           # pipeline entry point
├── requirements.txt
└── pytest.ini
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

No `nltk.download(...)` or `textblob.download_corpora` step is required. The
dataset CSV is already included in the repository, and the stop word list is
embedded directly in `src/text_cleaner.py`, so the pipeline works right after
`pip install` with no extra runtime downloads and no internet access needed.

Optional: if `data/reviews.csv` is ever deleted and you want to rebuild it from
the NLTK `movie_reviews` corpus instead of restoring it from git, run:

```bash
python -c "import nltk; nltk.download('movie_reviews')"
```

## Usage

Run the full pipeline:

```bash
python main.py
```

Run it and also check TextBlob's accuracy against the dataset's real labels:

```bash
python main.py --evaluate
```

Custom input/output paths:

```bash
python main.py --input data/reviews.csv --output outputs/reviews_with_sentiment.csv
```

## Sample output

```
Loaded 2000 reviews.

--- Sentiment stats ---
sentiment
Positive    1434
Neutral      478
Negative      88
Positive    71.7%
Neutral     23.9%
Negative     4.4%

Saved plot to outputs/sentiment_distribution.png

TextBlob accuracy against true labels: 53.70%
```

`outputs/reviews_with_sentiment.csv` contains the original review plus
`clean_review`, `polarity`, and `sentiment` columns for every row.
`outputs/sentiment_distribution.png` is a bar chart of the three sentiment
classes.

The ~54% accuracy against the ground-truth labels is expected rather than a
bug: TextBlob is a simple lexicon/rule-based analyzer, and movie reviews are
long, mixed-tone documents rather than short, single-opinion comments, so this
is a realistic baseline for this kind of tool on this kind of text.

## Design notes

- **Sentiment is scored on the original review text, not the cleaned one.**
  TextBlob relies on sentence structure, including negation words like "not",
  to score correctly. Feeding it the stop-word-stripped version (which would
  also strip "not") would silently break negation handling.
- **Stop words are a fixed, embedded list**, not fetched via
  `nltk.download('stopwords')` at runtime, so the pipeline has no network
  dependency once the packages are installed. Negation words (`not`, `no`,
  `never`, and similar) are deliberately excluded from the stop word list for
  the same reason as above.
- **Thresholds of ±0.05** on TextBlob's polarity score (which ranges from -1
  to 1) separate Positive, Negative, and Neutral.

## Tests

```bash
pytest tests/ -v
```

Two unit tests cover the text cleaning function: punctuation/lowercasing, and
stop word removal.

## Requirements

- Python 3.10+
- pandas, nltk, textblob, matplotlib, tqdm, pytest (see `requirements.txt`)
