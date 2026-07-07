from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import movie_reviews


def _ensure_movie_reviews_downloaded() -> None:
    try:
        movie_reviews.fileids()
    except LookupError:
        nltk.download("movie_reviews")


def build_reviews_csv(output_path: Path) -> None:
    """Write the movie_reviews corpus (2000 reviews) to disk as a CSV file."""
    _ensure_movie_reviews_downloaded()
    rows = []
    for file_id in movie_reviews.fileids():
        text = movie_reviews.raw(file_id)
        true_label = "positive" if file_id.startswith("pos") else "negative"
        rows.append({"review": text, "true_label": true_label})

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_reviews(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)