from src.text_cleaner import clean_text


def test_clean_text_lowercases_and_strips_punctuation():
    result = clean_text("This Movie is GREAT!!!")
    assert "!" not in result
    assert result == result.lower()


def test_clean_text_removes_stopwords():
    result = clean_text("this is a great movie")
    assert "is" not in result.split()
    assert "great" in result.split()