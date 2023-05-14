import re
import pandas as pd


def remove_user(text: str) -> str:
    return re.sub(r"@\w+", " ", text)


def replace_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text))


def preprocess(text: str) -> str:
    text = remove_user(text)
    text = replace_spaces(text)
    return text


def wrangle_ml(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].apply(preprocess)
    return df
