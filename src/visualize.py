from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_sentiment_distribution(df: pd.DataFrame, output_path: Path) -> None:
    counts = df["sentiment"].value_counts()
    counts.plot(kind="bar", color=["#4CAF50", "#F44336", "#9E9E9E"])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Review Count")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()