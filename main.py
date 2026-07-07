import argparse
from pathlib import Path

from tqdm import tqdm

from src.data_loader import build_reviews_csv, load_reviews
from src.text_cleaner import clean_text
from src.sentiment import get_polarity, label_sentiment
from src.visualize import plot_sentiment_distribution

DATA_PATH = Path("data/reviews.csv")
OUTPUT_PATH = Path("outputs/reviews_with_sentiment.csv")
PLOT_PATH = Path("outputs/sentiment_distribution.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review Sentiment Analysis")
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--evaluate", action="store_true",
                         help="Compare predictions against the true labels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"'{args.input}' not found, building it from the movie_reviews corpus...")
        build_reviews_csv(args.input)

    df = load_reviews(args.input)
    print(f"Loaded {len(df)} reviews.")

    tqdm.pandas(desc="Cleaning text")
    df["clean_review"] = df["review"].progress_apply(clean_text)

    # Score sentiment on the original review, not the stopword-stripped version:
    # TextBlob relies on sentence structure (e.g. negations like "not good")
    # that stopword removal would destroy.
    tqdm.pandas(desc="Scoring sentiment")
    df["polarity"] = df["review"].progress_apply(get_polarity)
    df["sentiment"] = df["polarity"].apply(label_sentiment)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print("\n--- Sentiment stats ---")
    stats = df["sentiment"].value_counts(normalize=True).mul(100).round(2)
    print(df["sentiment"].value_counts())
    print(stats.astype(str) + "%")

    plot_sentiment_distribution(df, PLOT_PATH)
    print(f"\nSaved plot to {PLOT_PATH}")

    if args.evaluate and "true_label" in df.columns:
        df["predicted_binary"] = df["sentiment"].map(
            {"Positive": "positive", "Negative": "negative", "Neutral": "positive"}
        )
        accuracy = (df["predicted_binary"] == df["true_label"]).mean()
        print(f"\nTextBlob accuracy against true labels: {accuracy:.2%}")


if __name__ == "__main__":
    main()