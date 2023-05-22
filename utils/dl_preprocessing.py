import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from .fetch_data import split_data
from .const import MAX_WORDS, DL_LABELS_PATH, MAX_SEQUENCE_LEN
from .ml_preprocessing import wrangle_ml



def wrangle_dl(df):
    df_clean = wrangle_ml(df)

    X_train, X_test, y_train, y_test = split_data(df_clean)
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    tok = Tokenizer(num_words=MAX_WORDS)
    tok.fit_on_texts(X_train)

    sequences = tok.texts_to_sequences(X_train)
    X_train_padded = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LEN)
    y_train_ = to_categorical(y_train)

    test_sequences = tok.texts_to_sequences(X_test)
    X_test_padded = sequence.pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LEN)
    y_test_ = to_categorical(y_test)

    return X_train_padded, X_test_padded, y_train_, y_test_
