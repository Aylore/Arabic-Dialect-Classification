import csv
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split


def get_data() -> pd.DataFrame:
    conn = sqlite3.connect("data/dialects_database.db")
    df_label = pd.read_sql_query("SELECT * FROM id_text", conn)
    df_target = pd.read_sql_query("SELECT * FROM id_dialect", conn)
    df = pd.merge(df_label, df_target, on="id")
    conn.close()
    return df


def split_data(df):
    X = df["text"]
    y = df["dialect"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test
