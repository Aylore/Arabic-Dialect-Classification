from utils.ml_preprocessing import wrangle_ml
from utils.dl_preprocessing import wrangle_dl
from utils.fetch_data import get_data, split_data
from src import train, evaluate
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


def ml_model(df, path=ML_MODEL_PATH):
    df_ml = wrangle_ml(df)
    model = train.fit_ml(df_ml)
    train.save_ml_model(model, path)
    evaluate.predict_ml("جامد", path)
    return evaluate.eval_ml(path)



def dl_model(df, path=DL_MODEL_PATH):
    X_train, X_test, y_train, y_test = wrangle_dl(df)
    model = train.fit_dl(X_train, X_test, y_train, y_test)
    train.save_dl_model(model, path)
    evaluate.predict_dl("جامد", path)
    return evaluate.eval_dl(X_test, y_test)

def main():
    df = get_data()
    ml_model(df, ML_MODEL_PATH)
    dl_model(df, DL_MODEL_PATH)

main()
