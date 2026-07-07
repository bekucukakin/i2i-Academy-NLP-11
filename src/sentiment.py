from textblob import TextBlob

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def get_polarity(text: str) -> float:
    return TextBlob(text).sentiment.polarity


def label_sentiment(polarity: float) -> str:
    if polarity > POSITIVE_THRESHOLD:
        return "Positive"
    if polarity < NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"