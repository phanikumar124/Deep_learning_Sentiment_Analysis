# predict.py

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from preprocess import clean_text

load_model = tf.keras.models.load_model
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

# ----------------------------
# CONFIG
# ----------------------------
MAX_LEN = 200

# ----------------------------
# LOAD MODEL & TOKENIZER
# ----------------------------
model = load_model("model/sentiment_model.h5")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ----------------------------
# SINGLE REVIEW PREDICTION
# ----------------------------
def predict_review(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

    prob = model.predict(padded)[0][0]
    sentiment = "Positive" if prob >= 0.5 else "Negative"

    return sentiment, float(prob)

# ----------------------------
# CSV PREDICTION + SUMMARY
# ----------------------------
def predict_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df[['review']]
    df.dropna(inplace=True)

    sentiments = []
    for review in df['review']:
        sentiment, _ = predict_review(review)
        sentiments.append(sentiment)

    df['Predicted_Sentiment'] = sentiments

    positive = sentiments.count("Positive")
    negative = sentiments.count("Negative")

    summary = {
        "Total Reviews": len(sentiments),
        "Positive Reviews": positive,
        "Negative Reviews": negative,
        "Overall Sentiment": "Positive"
        if positive >= negative else "Negative"
    }

    return df, summary
