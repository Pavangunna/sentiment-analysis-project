from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

app = Flask(__name__)
maxlen = 100

def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen)

    pred = model.predict(padded)
    label = np.argmax(pred)

    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive",
        3: "Irrelevant"
    }

    return label_map[label]

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        text = request.form["text"]
        result = predict(text)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))