import re
from pathlib import Path
from nltk.stem.isri import ISRIStemmer #for stemmeng

def load_stopwords(path: Path) -> set:
    """
    Load Arabic stopwords from a text file (one word per line).
    """
    if not path.exists():
        raise FileNotFoundError(f"Stopwords file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return set(word.strip() for word in f if word.strip())


def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (tashkeel).
    """
    arabic_diacritics = re.compile(
        r"[\u0617-\u061A\u064B-\u0652]"
    )
    return re.sub(arabic_diacritics, "", text)


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic letters.
    """
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    return text


def remove_elongation(text: str) -> str:
    """
    Remove character elongation (e.g., هذييييي → هذي).
    """
    return re.sub(r"(.)\1+", r"\1", text)


def clean_text(text: str) -> str:
    """
    Remove numbers, punctuation, and extra spaces.
    """
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def stem_text(text: str) -> str:
    """
    Apply light Arabic stemming using ISRIStemmer.
    """
    stemmer = ISRIStemmer()
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)


def preprocess_text(text: str, stopwords: set) -> str:
    """
    Full Arabic preprocessing pipeline for one text.
    """
    text = text.lower()
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = remove_elongation(text)
    text = clean_text(text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    # Apply stemming
    tokens = stem_text(" ".join(tokens)).split()

    return " ".join(tokens)