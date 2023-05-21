from sklearn.metrics import f1_score, accuracy_score
import joblib
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from utils.const import (
    X_test,
    X_train,
    y_train,
    y_test,
    ML_MODEL_PATH,
    DL_MODEL_PATH,
    MAX_SEQUENCE_LEN,
    MAX_WORDS,
)
from .train import load_ml_model, load_dl_model


def eval_ml(path=ML_MODEL_PATH):
    model = load_ml_model(path)
    y_pred = model.predict(X_test)
    print(f"Testing ML:\nAccuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Macro F1 score: {f1_score(y_test, y_pred, average='macro')}")


def eval_dl(X_test, y_test, path=DL_MODEL_PATH):
    model = load_dl_model(path)
    print(
        "Testing DL:\nAccuray: {}\nMacro F1 Score: {}".format(
            *model.evaluate(X_test, y_test)[1:]
        )
    )


def predict_ml_raw(sentence: str, path=ML_MODEL_PATH):
    """For Raw Data [TextTransformer() Pipeline]"""
    model = load_ml_model(path)
    text = pd.DataFrame([sentence], columns=["text"])
    predict_label = model.predict(text)[0]
    predict_probabiltiy = {
        country: prob
        for country, prob in zip(model.classes_, model.predict_proba(text)[0])
    }
    print(f"Dialect Prediction: {predict_label}\n Probabily: {predict_probabiltiy}")
    return predict_label, predict_probabiltiy


def predict_ml(sentence: str, path=ML_MODEL_PATH):
    """For wrangled data"""
    model = load_ml_model(path)
    predict_label = model.predict([sentence])
    predict_probabiltiy = {
        country: prob
        for country, prob in zip(model.classes_, model.predict_proba([sentence])[0])
    }
    # print(f"Dialect Prediction: {predict_label}\n Probabily: {predict_probabiltiy}")
    return predict_label, predict_probabiltiy


def predict_dl(sentence: str, path=DL_MODEL_PATH):
    model = load_dl_model(path)
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts([sentence])
    input_seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(input_seq, maxlen=MAX_SEQUENCE_LEN)
    labels = joblib.load('models/dl_labels.pickle')

    predict_probabiltiy = dict(zip(labels, model.predict(padded_seq)[0]))
    predict_label = max(predict_probabiltiy, key=predict_probabiltiy.get)
    print(f"Dialect Prediction: {predict_label}\n Probabily: {predict_probabiltiy}")

    return predict_label, predict_probabiltiy
