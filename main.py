from utils.ml_preprocessing import wrangle_ml
from utils.dl_preprocessing import wrangle_dl
from utils.fetch_data import get_data, split_data
from src import train, eval
from utils.const import (
    X_train,
    y_train,
    X_test,
    y_test,
    ML_MODEL_PATH,
    DL_MODEL_PATH,
    GRU_MODEL_PATH,
    SRNN_MODEL_PATH,
)


def ml_model(df):
    df_ml = wrangle_ml(df)
    model = train.fit_ml(df_ml)
    eval.eval_ml()


def dl_model(df):
    X_train, X_test, y_train, y_test = wrangle_dl(df)
    model = train.fit_dl(X_train, X_test, y_train, y_test)
    eval.eval_dl(X_test, y_test, DL_MODEL_PATH)


def main():
    df = get_data()
    ml_model(df)
    dl_model(df)


main()
