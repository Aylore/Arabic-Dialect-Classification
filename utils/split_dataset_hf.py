import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def split_tf(df):
    df_copy = df.copy()

    df_copy.drop("id", axis=1, inplace=True)

    df_copy.rename(columns={"text": "text", "dialect": "label"}, inplace=True)
    # df_copy.columns = [DATA_COLUMN, LABEL_COLUMN]

    df_train, df_test = train_test_split(
        df_copy,
        random_state=3407,
        #  stratify = y ,
        test_size=0.15,
    )

    return df_train.df_test
