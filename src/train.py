import joblib
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
import tensorflow_addons as tfa
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils import fetch_data
from utils.const import (
    MAX_WORDS,
    MAX_SEQUENCE_LEN,
    NUM_CLASSES,
    EPOCHS,
    BATCH_SIZE,
    INPUT_LENGTH,
    ML_MODEL_PATH,
    SRNN_MODEL_PATH,
    DL_MODEL_PATH,
    GRU_MODEL_PATH,
)


def fit_ml(df):
    X_train, X_test, y_train, y_test = fetch_data.split_data(df)
    final_model = ComplementNB(alpha=0.3)
    pipe = Pipeline([("Vectorizer", TfidfVectorizer()), ("classifier", final_model)])
    pipe.fit(X_train, y_train)
    save_ml_model(pipe, ML_MODEL_PATH)
    return pipe


def save_ml_model(model, path=ML_MODEL_PATH):
    joblib.dump(model, path)


def load_ml_model(path=ML_MODEL_PATH):
    model = joblib.load(path)
    return model


def fit_dl(X_train_padded, X_test_padded, y_train, y_test):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Embedding(MAX_WORDS, 64, input_length=INPUT_LENGTH),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.summary()

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[
            "accuracy",
            tfa.metrics.F1Score(average="macro", num_classes=NUM_CLASSES),
        ],
    )

    history = model.fit(
        X_train_padded,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    save_dl_model(model, GRU_MODEL_PATH)
    return model


def save_dl_model(model, path=DL_MODEL_PATH):
    model.save(path)


def load_dl_model(path=DL_MODEL_PATH):
    model = tf.keras.models.load_model(path)
    return model
