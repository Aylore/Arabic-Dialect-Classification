from .fetch_data import split_data, get_data

EPOCHS = 5
BATCH_SIZE = 32
MAX_WORDS = 10_000
X_train, X_test, y_train, y_test = split_data(get_data())
NUM_CLASSES = y_train.nunique()
MAX_SEQUENCE_LEN = max(len(sentence) for sentence in X_train)
INPUT_LENGTH = MAX_SEQUENCE_LEN
ML_MODEL_PATH = "models/ml_model.pkl"
DL_MODEL_PATH = "models/LSTM"
SRNN_MODEL_PATH = "models/SimpleRNN"
GRU_MODEL_PATH = "models/GRU"
