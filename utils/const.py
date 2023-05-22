from .fetch_data import split_data, get_data
import nltk
from .ml_preprocessing import wrangle_ml


EPOCHS = 5
BATCH_SIZE = 32
MAX_WORDS = 10_000
X_train, X_test, y_train, y_test = split_data(get_data())
df_clean = wrangle_ml(get_data())
X_train_dl, X_test_dl, y_train, y_test = split_data(df_clean)
MAX_SEQUENCE_LEN = INPUT_LENGTH = max(len(sentence) for sentence in X_train_dl)
NUM_CLASSES = y_train.nunique()
ML_MODEL_PATH = "models/ml_model.pkl"
DL_MODEL_PATH = "models/LSTM"
SRNN_MODEL_PATH = "models/SimpleRNN"
GRU_MODEL_PATH = "models/GRU"
DL_LABELS_PATH = 'models/dl_labels.pkl'
