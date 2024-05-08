from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the LSTM model
lstm_model = load_model("lstm_model.h5")
# Load the classification model
classification_model = load_model("classification_model.h5")

# Load and fit tokenizer during runtime
def load_tokenizer():
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")
    df_fake['class'] = 0
    df_true['class'] = 1
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df['text'] = df['text'].fillna('').astype(str)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['text'])
    return tokenizer

tokenizer = load_tokenizer()

# Maximum sequence length
max_len = 1000

def predict_fake_news(text):
    # Tokenize the text
    sequence = tokenizer.texts_to_sequences([text])
    # Pad sequences
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    # Predict using LSTM model
    lstm_prediction = lstm_model.predict(padded_sequence)
    # Predict using classification model
    classification_prediction = classification_model.predict(np.concatenate((padded_sequence, lstm_prediction.reshape(-1, 1)), axis=1))
    return classification_prediction[0][0]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    prediction = predict_fake_news(text)
    return render_template("index.html", prediction=prediction, text=text)

if __name__ == "__main__":
    app.run(debug=True)
