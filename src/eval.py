from sklearn.metrics import f1_score, accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from utils.const import X_test, X_train, y_train, y_test, ML_MODEL_PATH, DL_MODEL_PATH
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
