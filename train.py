# train.py
import os
import pickle
import numpy as np
from preprocess import load_and_preprocess

os.makedirs("model", exist_ok=True)   
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore


# ----------------------------
# CONFIG
# ----------------------------
MAX_WORDS = 20000
MAX_LEN = 200
EMBEDDING_DIM = 128
EPOCHS = 5
BATCH_SIZE = 64

# ----------------------------
# LOAD & PREPROCESS DATA
# ----------------------------
print("Loading and preprocessing data...")
X, y = load_and_preprocess("dataset/drug_reviews.csv")

# ----------------------------
# TOKENIZATION
# ----------------------------
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

# ----------------------------
# MODEL
# ----------------------------
print("Building model...")
model = Sequential([
    Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# TRAIN
# ----------------------------
print("Training model...")
model.fit(
    padded_sequences,
    y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)

# ----------------------------
# SAVE MODEL & TOKENIZER
# ----------------------------
model.save("model/sentiment_model.h5")

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved successfully!")
