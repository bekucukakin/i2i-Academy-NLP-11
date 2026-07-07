import re
import string

import nltk
from nltk.corpus import stopwords


def _load_stop_words() -> set[str]:
    try:
        words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        words = set(stopwords.words("english"))
    return words


STOP_WORDS = _load_stop_words()


def clean_text(text: str) -> str:
    """Lowercase the text, strip punctuation and digits, drop stop words."""
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)