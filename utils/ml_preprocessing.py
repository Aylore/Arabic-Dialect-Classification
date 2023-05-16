import re
import pandas as pd
import demoji


def remove_urls(text):
    return re.sub(r'http\S+', '', text)
    

def remove_emj(text):
  demoji.download_codes()
  return demoji.replace(text, '')

def remove_user(text: str) -> str:
    return re.sub(r"@\w+", " ", text)


def replace_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text))


def preprocess(text: str) -> str:
    text = remove_urls(text)
    text = remove_emj(text)
    text = remove_user(text)
    text = replace_spaces(text)
    return text


def wrangle_ml(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].apply(preprocess)
    return df
