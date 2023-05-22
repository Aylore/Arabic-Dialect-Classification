import re
import pandas as pd
from string import punctuation
import pyarabic.araby as araby
import nltk

nltk.download("stopwords")
STOP_WORDS = set(nltk.corpus.stopwords.words("arabic"))


def replace_punctuation(text: str) -> str:
    added_punctuation = punctuation + "؟،"
    return re.sub(rf"[{added_punctuation}]", " ", text)


def remove_arabic_diatrics(text: str) -> str:
    text = araby.strip_tashkeel(text)
    text = araby.normalize_ligature(text)
    return text


def keep_arabic(text: str) -> str:
    return re.sub(r"[^\u0600-\u06FF ]+", " ", text)


def remove_stop_words(text: str) -> str:
    return " ".join(word for word in text.split() if word not in STOP_WORDS)


def replace_repeated_chars(text: str) -> str:
    return re.sub(r"(\w)\1{2,}", r"\1\1", text)


def preprocess(text: str) -> str:
    text = replace_punctuation(text)
    text = remove_arabic_diatrics(text)
    text = keep_arabic(text)
    text = remove_stop_words(text)
    text = replace_repeated_chars(text)
    return text


def wrangle_ml(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].apply(preprocess)
    return df
